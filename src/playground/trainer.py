import logging
from functools import cached_property
from pathlib import Path

import tiktoken
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import Dataset
from torchtyping import TensorType

from playground.dataloader import get_train_dataloader
from playground.losses import language_modelling_loss
from playground.metadata import (
    FINAL_NORM,
    LAYER_NORM_PRE_ATT,
    LAYER_NORM_PRE_FFN,
    POS_EMBEDDING,
    TOK_EMBEDDING,
)
from playground.trainer_utils import (
    TrainerControl,
    TrainerControlConfig,
    TrainerState,
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
        self._tokenizer = tokenizer
        self._train_dataset = train_dataset
        self._val_dataset = validation_dataset
        self._test_dataset = test_dataset
        self._checkpoint_dir = trainer_config.checkpoint_dir
        self._dataloader_config = data_loader_config
        self.num_epochs = trainer_config.num_epochs
        self.state = TrainerState(seed=seed)
        control_config = TrainerControlConfig(
            save_steps=trainer_config.save_steps,
            eval_steps=trainer_config.eval_steps,
            log_steps=trainer_config.log_steps,
            num_train_steps=self.num_train_steps,
            sample_steps=trainer_config.sample_steps,
            epoch_steps=self.num_train_steps // self.num_epochs,
        )
        self.control = TrainerControl(state=self.state, config=control_config)
        self.optimiser = self._init_optimiser(self._model, optimiser_config.optimiser)
        self.lr_scheduler = self._init_scheduler(self.optimiser, optimiser_config)
        self.ignore_token_idx = trainer_config.ignore_token_index

    @staticmethod
    def _init_optimiser(model: nn.Module, config: DictConfig) -> torch.optim.Optimizer:
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

    def _init_scheduler(self, optimiser: torch.optim.Optimizer, config: DictConfig):
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

    def train(self, resume_from_checkpoint: str | Path | None = None):
        if resume_from_checkpoint is not None:
            raise NotImplementedError("Checkpointing not implemented")
        loader_config = OmegaConf.to_container(
            self._dataloader_config.train, resolve=True
        )
        model = self._model
        dataloader = get_train_dataloader(self._train_dataset, **loader_config)
        for epoch in range(self.state.num_epochs_trained, self.num_epochs):
            for step, (inputs, targets) in enumerate(dataloader):
                inputs, targets = move_to_device(inputs, targets, device=self.device)
                self.train_step(model, inputs, targets)
                self.state.increment_step()

    def train_step(
        self, model: nn.Module, inputs: TensorType["B", "L", "D"], targets: ["B", "L"]
    ) -> torch.Tensor:
        self.optimiser.zero_grad()
        logits = model(inputs)
        loss = self.compute_loss(logits, targets, model=model)
        loss.backward()
        self.optimiser.step()
        self.lr_scheduler.step()
        return loss.item()

    def evaluate(self):
        pass

    def predict(self):
        pass

    def compute_loss(
        self,
        outputs: TensorType["B", "L", "V"],
        targets: TensorType["B", "L"],
        **kwargs,
    ):
        ignore_idx = kwargs.get("ignore_idx", -100)
        return language_modelling_loss(
            next_token_logits=outputs, next_token_ids=targets, ignore_idx=ignore_idx
        )
