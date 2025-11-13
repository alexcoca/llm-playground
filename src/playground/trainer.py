import json
import logging
from functools import cached_property
from pathlib import Path
from typing import Literal

import tiktoken
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
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
from playground.trainer_utils import ensure_determinism, set_seed

logger = logging.getLogger(__name__)

Split = Literal["train", "validation"]
TRAINER_STATE_NAME = "trainer_state.json"
NO_DECAY_PARAMS = [
    "bias",
    LAYER_NORM_PRE_FFN,
    LAYER_NORM_PRE_ATT,
    FINAL_NORM,
    TOK_EMBEDDING,
    POS_EMBEDDING,
]


class TrainerState(BaseModel):

    global_step: int = 0
    num_epochs_trained: int = 0
    steps_current_epoch: int = 0
    best_val_loss: float = 0.0
    best_val_loss_steps: float = 0
    best_train_loss: float = 0.0
    best_train_loss_steps: float = 0
    seed: int | None = None

    def new_epoch(self):
        self.num_epochs_trained += 1
        self.steps_current_epoch = 0

    def increment_step(self):
        self.global_step += 1
        self.steps_current_epoch += 1

    def maybe_update_best_loss(self, loss: float, split: Split):
        raise NotImplementedError

    @classmethod
    def load_from_json(cls, input_dir: Path | str) -> "TrainerState":
        """Loads trainer state from a JSON file, or returns a new state."""
        if not isinstance(input_dir, Path):
            input_dir = Path(input_dir)

        filepath = input_dir / TRAINER_STATE_NAME

        if not filepath.exists():
            logger.warning(f"No state file found at {filepath}. Starting new state.")
            return cls()

        try:
            json_data = filepath.read_text()
            logger.info(f"Loading state from {filepath}")
            return cls.model_validate_json(json_data)

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading trainer state: {e}.")
            raise e

    def save_to_json(self, output_dir: Path | str):
        """Saves the trainer state to a JSON file."""
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / TRAINER_STATE_NAME

        json_data = self.model_dump_json(indent=2)
        filepath.write_text(json_data)

        logger.info(f"Trainer state saved to {filepath}")


def prepare_model(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    model.train()
    model.to(device)
    return model


def prepare_tensor(
    *tensors: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(t.to(device) for t in tensors)


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
        self.state = TrainerState(seed=seed)
        self._save_steps = trainer_config.save_steps
        self._eval_steps = trainer_config.eval_steps
        self.num_epochs = trainer_config.num_epochs
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
                inputs, targets = prepare_tensor(inputs, targets, device=self.device)
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
