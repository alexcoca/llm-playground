from pathlib import Path

from omegaconf import OmegaConf


def _resolve_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def create_dir(pth: Path | str, suffix: str | None = None):
    if suffix is not None:
        pth = f"{str(pth)}_{suffix}"
    if isinstance(pth, str):
        pth = Path(pth)

    if not pth.exists():
        pth.mkdir(parents=True)

    return str(pth)


OmegaConf.register_new_resolver("root", lambda path: f"{_resolve_root() / path}")
OmegaConf.register_new_resolver(
    "create_dir", lambda pth, suffix=None: create_dir(pth, suffix)
)
