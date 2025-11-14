import logging
from pathlib import Path

import torch
from tiktoken import Encoding, get_encoding
from torch.utils.data import DataLoader, Dataset

from playground.data_utils import load_text

logger = logging.getLogger(__name__)


def chunk_text(
    token_ids: list[int], max_len: int, stride: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

    context, targets = [], []
    for i in range(0, len(token_ids) - max_len, stride):
        input_chunk = token_ids[i : i + max_len]
        context.append(torch.tensor(input_chunk))
        target_chunk = token_ids[i + 1 : i + max_len + 1]
        targets.append(torch.tensor(target_chunk))

    return context, targets


class NextTokenPredictionDataset(Dataset):

    def __init__(
        self, path: str | Path, tokenizer: Encoding, max_length: int, stride: int
    ):
        text = load_text(path)
        token_ids = tokenizer.encode(text)
        self.input_ids, self.target_ids = chunk_text(token_ids, max_length, stride)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[item], self.target_ids[item]


def get_validation_test_dataloader(
    dataset: Dataset, batch_size: int = 1, num_workers: int = 0
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def get_train_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    logger.info("Creating train dataloader")
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        shuffle=True,
    )


def get_dataset(
    path: str | Path,
    max_length: int = 256,
    stride: int = 128,
):
    tokenizer = get_encoding("gpt2")
    return NextTokenPredictionDataset(path, tokenizer, max_length, stride)
