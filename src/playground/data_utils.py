import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

from tiktoken import Encoding

logger = logging.getLogger(__name__)


def open_url(url: str, fname: str | Path) -> tuple:
    return urllib.request.urlretrieve(url, fname)


def make_absolute(pth: str | Path) -> Path:
    if isinstance(pth, str):
        return Path(pth).absolute()
    return pth.absolute()


def load_text(pth: str | Path) -> str:
    pth = make_absolute(pth)

    with open(pth, encoding="utf-8") as f:
        text = f.read()
    logger.info(f"Read file: {pth}")
    return text


def dump_text(text: str, pth: str | Path) -> str:
    pth = make_absolute(pth)

    with open(pth, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"Wrote file {pth}")


def train_val_split(
    text: str,
    tokenizer: Encoding,
    train_ratio: float = 0.9,
) -> tuple[str, str]:

    encoded = tokenizer.encode(text)
    split_idx = int(len(encoded) * train_ratio)
    train_tok, val_tok = encoded[:split_idx], encoded[split_idx:]
    train_text = tokenizer.decode(train_tok)
    val_text = tokenizer.decode(val_tok)
    return train_text, val_text


def load_json(path: str | Path, data_info: str = "data"):
    logger.info(f"Loading {data_info} from data!")
    with open(path) as f:
        data = json.load(f)
    return data


def save_json(data: Any, path: str | Path, indent: int = 4, data_info: str = "data"):
    logger.info(f"Saving {data_info} at {path}.")
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)
