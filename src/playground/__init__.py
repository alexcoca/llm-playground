from pathlib import Path

from omegaconf import OmegaConf


def _resolve_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


OmegaConf.register_new_resolver("root", lambda path: f"{_resolve_root() / path}")
