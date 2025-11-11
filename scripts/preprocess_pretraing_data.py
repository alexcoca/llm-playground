import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from playground.data_utils import dump_text, load_text, train_val_split

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="pretrain_preprocess")
def preprocess_data(cfg: DictConfig):

    cfg = instantiate(cfg)
    tokenizer = cfg.tokenizer
    input_dir = Path(cfg.input_dir)
    out_dir = Path(cfg.out_dir)
    logger.info(f"Loading data from {input_dir}")
    for data_file in input_dir.glob("*.txt"):
        shard_name = data_file.name
        train_text, val_text = train_val_split(
            load_text(data_file), tokenizer, train_ratio=cfg.train_ratio
        )
        dump_text(train_text, out_dir / "train" / shard_name)
        dump_text(val_text, out_dir / "validation" / shard_name)


if __name__ == "__main__":
    preprocess_data()
