from abc import ABC, abstractmethod
from typing import Iterable

import torch
from torchtyping import TensorType

Logits = TensorType["B", "L", "V"]


class LogitsProcessor(ABC):
    @abstractmethod
    def __call__(self, logits: TensorType) -> TensorType:
        """Process logits before sampling."""
        pass


class TemperatureLogitsProcessor(LogitsProcessor):
    def __init__(self, temperature: float = 1.0):
        assert temperature > 0.0, "temperature must be > 0.0"
        self.temperature = temperature

    def __call__(self, logits: Logits) -> Logits:
        if self.temperature == 1.0:
            return logits
        return logits / self.temperature


class TopKLogitsProcessor(LogitsProcessor):
    def __init__(self, top_k: int):
        self.top_k = top_k

    def __call__(self, logits: Logits) -> Logits:
        top_k_values, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        logits_filtered = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits_filtered.scatter_(-1, top_k_indices, top_k_values)
        return logits_filtered


def apply_logits_processors(
    logits: Logits,
    processors: Iterable[LogitsProcessor] | None = None,
) -> Logits:
    for processor in processors or []:
        logits = processor(logits)
    return logits
