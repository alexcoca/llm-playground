import json
import logging
import os
import random
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)


def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # ok if no cuda


def ensure_determinism(seed: int):

    logger.info("Setting seed")
    set_seed(seed)

    logger.info(
        "Setting debug environment for CUDA to ensure determinism and avoiding"
        " non-deterministic ops"
    )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    logger.info(f"CUDA_WORK_SPACE_CONFIG: {os.environ.get('CUDA_WORK_SPACE_CONFIG')}")
    torch.use_deterministic_algorithms(True)
    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrainerAction(str, Enum):
    """Actions that can be executed during training."""

    # Lifecycle events (always happen)
    TRAIN_BEGIN = "train_begin"
    TRAIN_END = "train_end"
    EPOCH_BEGIN = "epoch_begin"
    EPOCH_END = "epoch_end"
    STEP_BEGIN = "step_begin"
    STEP_END = "step_end"

    # Conditional actions (triggered by TrainerControl)
    SAVE = "save"
    EVALUATE = "evaluate"
    LOG = "log"
    SAMPLE = "sample"
    STOP = "stop"


Split = Literal["train", "validation"]
TRAINER_STATE_NAME = "trainer_state.json"


class SerializableConfig(BaseModel):
    def save_to_json(self, output_dir: Path | str, fname: str, what: str = "File"):
        """Saves config to a JSON file."""
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / fname

        json_data = self.model_dump_json(indent=2)
        filepath.write_text(json_data)

        logger.info(f"{what} saved to {filepath}")

    @classmethod
    def load_from_json(
        cls, input_dir: Path | str, fname: str, what: str = "file"
    ) -> "TrainerState":
        """Loads trainer state from a JSON file, or returns a new state."""
        if not isinstance(input_dir, Path):
            input_dir = Path(input_dir)

        filepath = input_dir / fname

        if not filepath.exists():
            logger.warning(f"No {what} found at {filepath}. Returning empty config.")
            return cls()

        try:
            json_data = filepath.read_text()
            logger.info(f"Loading {what} from {filepath}")
            return cls.model_validate_json(json_data)

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading trainer state: {e}.")
            raise e


class TrainerState(SerializableConfig):

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


class TrainerControlConfig(SerializableConfig):
    save_steps: int
    eval_steps: int
    log_steps: int
    epoch_steps: int
    num_train_steps: int
    sample_steps: int | Literal["epoch"] | None

    @model_validator(mode="after")
    def resolve_sample_steps(self):
        """Convert 'epoch' literal to actual epoch_steps value."""
        if self.sample_steps == "epoch":
            self.sample_steps = self.epoch_steps
        return self


class TrainerControl:

    def __init__(self, state: TrainerState, config: TrainerControlConfig):

        self.state = state
        self.save_steps = config.save_steps
        self.eval_steps = config.eval_steps
        self.log_steps = config.log_steps
        self.num_train_steps = config.num_train_steps
        self.sample_steps = config.sample_steps
        self._control_steps = (
            ("should_log", TrainerAction.LOG),
            ("should_save", TrainerAction.SAVE),
            ("should_evaluate", TrainerAction.EVALUATE),
            ("should_sample", TrainerAction.SAMPLE),
            ("should_stop", TrainerAction.STOP),
        )

    def should_save(self) -> bool:
        return self.state.global_step % self.save_steps == 0

    def should_evaluate(self) -> bool:
        return self.state.global_step % self.eval_steps == 0

    def should_stop(self) -> bool:
        # TODO: THINK - EARLY STOPPING INTEGRATION - PROBABLY NOT A
        #  CONCERN HERE AS THAT DEPENDS ON METRICS/USER SPECIFIED CRITERIA ETC
        return self.state.global_step == self.num_train_steps

    def should_log(self) -> bool:
        return self.state.global_step % self.log_steps == 0

    def should_sample(self) -> bool:
        return (
            self.sample_steps is not None
            and self.state.global_step % self.sample_steps == 0
        )

    def determine_actions(self) -> list[TrainerAction]:
        return [action for step, action in self._control_steps if getattr(self, step)()]


def prepare_model(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    model.train()
    model.to(device)
    return model


def move_to_device(
    *tensors: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(t.to(device) for t in tensors)
