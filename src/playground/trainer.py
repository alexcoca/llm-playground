import json
import logging
from pathlib import Path
from typing import Literal

import tiktoken
import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from torch import nn
from torch.utils.data import Dataset

from playground.dataloader import get_train_dataloader
from playground.trainer_utils import ensure_determinism, set_seed

logger = logging.getLogger(__name__)

Split = Literal["train", "validation"]
TRAINER_STATE_NAME = "trainer_state.json"


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


def prepare_model(model: torch.nn.Module, device: torch.device):
    model.train()
    model.to(device)


def prepare_tensor(*tensors: torch.Tensor, device: torch.device):
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

    def train(self, resume_from_checkpoint: str | Path | None = None):
        if resume_from_checkpoint is not None:
            raise NotImplementedError("Checkpointing not implemented")
        loader_config = OmegaConf.to_container(
            self._dataloader_config.train, resolve=True
        )
        dataloader = get_train_dataloader(self._train_dataset, **loader_config)
        model = self._model
        for epoch in range(self.state.num_epochs_trained, self.num_epochs):
            for step, (inputs, targets) in enumerate(dataloader):
                inputs, targets = prepare_tensor(inputs, targets, device=self.device)
                self._train_step(model, inputs, targets)
                self.state.increment_step()

    def _train_step(
        self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor
    ):
        print(inputs.shape, targets.shape)

    def evaluate(self):
        pass

    def predict(self):
        pass

    def compute_loss(self, model: nn.Module, batch: torch.Tensor, **kwargs): ...
