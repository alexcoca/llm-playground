import logging
import math
import time
from functools import cached_property
from pathlib import Path

import tiktoken
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import Dataset
from torchtyping import TensorType

from playground.dataloader import get_train_dataloader, get_validation_test_dataloader
from playground.losses import language_modelling_loss
from playground.metadata import (
    FINAL_NORM,
    LAYER_NORM_PRE_ATT,
    LAYER_NORM_PRE_FFN,
    POS_EMBEDDING,
    TOK_EMBEDDING,
)
from playground.trainer_utils import (
    CHECKPOINT_TEMPLATE,
    TRAINER_CONTROLLER_CONFIG_NAME,
    TRAINER_STATE_NAME,
    ActionResult,
    FinishedTraining,
    TrainerAction,
    TrainerControl,
    TrainerControlConfig,
    TrainerState,
    TrainingResult,
    count_tokens,
    ensure_determinism,
    move_to_device,
    prepare_model,
    set_seed,
)

logger = logging.getLogger(__name__)

NO_DECAY_PARAMS = [
    "bias",
    LAYER_NORM_PRE_FFN,
    LAYER_NORM_PRE_ATT,
    FINAL_NORM,
    TOK_EMBEDDING,
    POS_EMBEDDING,
]

MetricsDict = dict[str, float]
TRAINING_METRIC_KEY = "train"
EVAL_METRIC_KEY = "eval"


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        tokenizer: tiktoken.Encoding,
        optimiser_config: DictConfig,
        train_dataset: Dataset,
        data_loader_config: DictConfig,
        trainer_config: DictConfig,
        validation_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ):
        seed = trainer_config.seed
        if trainer_config.ensure_determinism:
            ensure_determinism(seed)
        else:
            set_seed(seed)
        self.device = torch.device(trainer_config.device)
        self._model = prepare_model(model, self.device)
        self._checkpoint_dir = Path(trainer_config.checkpoint_dir)
        self._allow_ckpt_override = trainer_config.allow_ckpt_override
        self._tokenizer = tokenizer
        self._train_dataset = train_dataset
        self._val_dataset = validation_dataset
        self.validation_data_loader = None
        self._test_dataset = test_dataset
        self._dataloader_config = data_loader_config
        self.num_epochs = trainer_config.num_epochs
        self.state = TrainerState(seed=seed)
        control_config = TrainerControlConfig(
            save_steps=trainer_config.save_steps,
            eval_steps=trainer_config.eval_steps,
            log_steps=trainer_config.log_steps,
            num_train_steps=self.num_train_steps,
            sample_steps=trainer_config.sampling.sample_steps,
            epoch_steps=self.num_train_steps // self.num_epochs,
        )
        self.control = TrainerControl(state=self.state, config=control_config)
        self.optimiser = self.init_optimiser(self._model, optimiser_config.optimiser)
        self.lr_scheduler = self.init_scheduler(self.optimiser, optimiser_config)
        self.ignore_token_idx = trainer_config.ignore_token_index
        self._sampling_config = trainer_config.sampling
        self.display_samples = trainer_config.sampling.display_samples
        self.samples_dir = trainer_config.sampling.samples_dir
        if self.samples_dir is not None:
            self.samples_dir = Path(self.samples_dir)
        self.loggers = instantiate(trainer_config.loggers) or []

    @staticmethod
    def init_optimiser(model: nn.Module, config: DictConfig) -> torch.optim.Optimizer:
        logger.info(f"Weight decay will not be applied to {NO_DECAY_PARAMS}")
        no_weight_decay, rest = [], []
        for name, param in model.named_parameters():
            if any(nd in name for nd in NO_DECAY_PARAMS):
                no_weight_decay.append(param)
            else:
                rest.append(param)

        param_groups = [
            {"params": rest, "weight_decay": config.weight_decay},
            {"params": no_weight_decay, "weight_decay": 0.0},
        ]
        return instantiate(config, params=param_groups)

    def init_scheduler(self, optimiser: torch.optim.Optimizer, config: DictConfig):
        return instantiate(
            config.scheduler,
            num_warmup_steps=self.num_train_steps * config.warmup_perc,
            num_training_steps=self.num_train_steps,
            optimizer=optimiser,
        )

    @cached_property
    def num_train_steps(self) -> int:

        batch_size = self._dataloader_config.train.batch_size
        drop_last = self._dataloader_config.train.drop_last

        if drop_last:
            steps_per_epoch = len(self._train_dataset) // batch_size
        else:
            steps_per_epoch = (len(self._train_dataset) + batch_size - 1) // batch_size

        return self.num_epochs * steps_per_epoch

    @cached_property
    def sample_inputs(self) -> torch.Tensor:
        """Tokenize and tensorize prompts for sampling."""

        prompts = self._sampling_config.prompts
        token_ids = [self._tokenizer.encode(ctx) for ctx in prompts]
        max_len = max(len(ids) for ids in token_ids)
        padded = [
            ids + [self._tokenizer.eot_token] * (max_len - len(ids))
            for ids in token_ids
        ]
        return torch.tensor(padded)

    def train(self, resume_from_checkpoint: str | Path | None = None) -> TrainingResult:
        if resume_from_checkpoint is not None:
            raise NotImplementedError("Checkpointing not implemented")
        loader_config = OmegaConf.to_container(
            self._dataloader_config.train, resolve=True
        )
        model = self._model
        dataloader = get_train_dataloader(self._train_dataset, **loader_config)
        start_time = time.time()
        try:
            for epoch in range(self.state.num_epochs_trained, self.num_epochs):
                for step, (inputs, targets) in enumerate(dataloader):
                    inputs, targets = move_to_device(
                        inputs, targets, device=self.device
                    )
                    loss = self.train_step(model, inputs, targets)
                    self.state.increment_seen_tokens(
                        count_tokens(inputs, self.ignore_token_idx)
                    )
                    self.state.increment_step()
                    actions = self.control.determine_actions()
                    results = self.execute_actions(
                        actions, metrics={f"{TRAINING_METRIC_KEY}/loss": loss}
                    )
                    self.process_action_results(results)
        except FinishedTraining:
            logger.info("Training finished")
        return TrainingResult(
            total_time=time.time() - start_time,
            total_tokens=self.state.tokens_seen,
            final_step=self.state.global_step,
            epochs_completed=self.state.num_epochs_trained,
        )

    def execute_actions(
        self, actions: list[TrainerAction], **kwargs
    ) -> list[ActionResult]:
        results = []
        metrics = kwargs.pop("metrics", {})
        for action in actions:
            if action == TrainerAction.SAVE:
                self.save(**kwargs)
            elif action == TrainerAction.EVALUATE:
                eval_metrics = self.evaluate(**kwargs)
                metrics |= eval_metrics
            elif action == TrainerAction.LOG:
                self.log(metrics=metrics, **kwargs)
            elif action == TrainerAction.SAMPLE:
                results.append(
                    ActionResult(
                        action=TrainerAction.SAMPLE, result=self.sample(**kwargs)
                    )
                )
            elif action == TrainerAction.STOP:
                results.append(
                    ActionResult(
                        action=TrainerAction.STOP,
                        result=True,
                    )
                )
            else:
                raise ValueError(f"Unknown Trainer action : {action}")

        return results

    def process_action_results(self, results: list[ActionResult | None]):
        for result in (r for r in results if r is not None):
            if result.action == TrainerAction.SAMPLE:
                if self.display_samples:
                    for logger_ in self.loggers:
                        logger_.display_samples(
                            result.result, step=self.state.global_step
                        )
                if (sample_dir := self.samples_dir) is not None:
                    fpath = sample_dir / f"samples_step_{self.state.global_step}.txt"
                    fpath.write_text("\n\n".join(result.result))

            elif result.action == TrainerAction.STOP:
                raise FinishedTraining("Training completed")

    def train_step(
        self, model: nn.Module, inputs: TensorType["B", "L", "D"], targets: ["B", "L"]
    ) -> torch.Tensor:
        self.optimiser.zero_grad()
        logits = model(inputs)
        loss = self.compute_loss(
            logits, targets, model=model, ignore_idx=self.ignore_token_idx
        )
        loss.backward()
        self.optimiser.step()
        self.lr_scheduler.step()
        return loss.item()

    def compute_loss(
        self,
        outputs: TensorType["B", "L", "V"],
        targets: TensorType["B", "L"],
        **kwargs,
    ):
        ignore_idx = kwargs.get("ignore_idx", -100)
        reduction = kwargs.get("reduction", "mean")
        return language_modelling_loss(
            next_token_logits=outputs,
            next_token_ids=targets,
            ignore_idx=ignore_idx,
            reduction=reduction,
        )

    def evaluate(self, **kwargs) -> MetricsDict:
        if self._val_dataset is None:
            return {}
        if self.validation_data_loader is None:
            loader_config = OmegaConf.to_container(
                self._dataloader_config.validation, resolve=True
            )
            self.validation_data_loader = get_validation_test_dataloader(
                self._val_dataset, **loader_config
            )
        model = self._model
        model.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(self.validation_data_loader):
                inputs, targets = move_to_device(inputs, targets, device=self.device)
                logits = self._model(inputs)
                loss = self.compute_loss(
                    logits, targets, ignore_idx=self.ignore_token_idx, reduction="sum"
                )
                total_loss += loss.item()
                total_tokens += count_tokens(targets, self.ignore_token_idx)
        model.train()
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return {
            f"{EVAL_METRIC_KEY}/avg_loss": avg_loss,
            f"{EVAL_METRIC_KEY}/perplexity": perplexity,
        }

    def predict(self):
        pass

    def log(self, **kwargs):
        metrics = kwargs.get("metrics", {})
        step = self.state.global_step
        for log in self.loggers:
            log.log(metrics, step)

    def save(self, **kwargs):
        this_checkpoint_dir = self._checkpoint_dir / CHECKPOINT_TEMPLATE.format(
            step=self.state.global_step
        )
        this_checkpoint_dir.mkdir(parents=True, exist_ok=self._allow_ckpt_override)
        checkpoint = {
            "model": self._model.state_dict(),
            "optimizer": self.optimiser.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        torch.save(checkpoint, this_checkpoint_dir / "model.pt")
        self.state.save_to_json(
            this_checkpoint_dir, TRAINER_STATE_NAME, what="Trainer state"
        )
        self.control.config.save_to_json(
            this_checkpoint_dir,
            TRAINER_CONTROLLER_CONFIG_NAME,
            what="Trainer controller config",
        )

    def sample(self, **kwargs) -> list[str]:
        logger.info(f"Sampling from the model at step {self.state.global_step}")
        model = self._model
        model.eval()

        with torch.no_grad():
            inputs = self.sample_inputs.to(self.device)
            outputs = model.generate(
                inputs,
                max_new_tokens=self._sampling_config.max_new_tokens,
                eos_token_id=self._tokenizer.eot_token,
                pad_token_id=self._tokenizer.eot_token,
            )

        samples = [self._tokenizer.decode(out.tolist()) for out in outputs]
        model.train()
        return samples
