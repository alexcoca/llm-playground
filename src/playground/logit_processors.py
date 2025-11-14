from abc import ABC, abstractmethod

import torch
from torchtyping import TensorType


class LogitsProcessor(ABC):
    @abstractmethod
    def __call__(self, logits: TensorType) -> TensorType:
        """Process logits before sampling."""
        pass


class TemperatureLogitsProcessor(LogitsProcessor):
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def __call__(self, logits: TensorType["B", "L", "V"]) -> TensorType["B", "L", "V"]:
        return logits / self.temperature


class TopKLogitsProcessor(LogitsProcessor):
    def __init__(self, top_k: int):
        self.top_k = top_k

    def __call__(self, logits: TensorType["B", "L", "V"]) -> TensorType["B", "L", "V"]:
        top_k_values, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        logits_filtered = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits_filtered.scatter_(-1, top_k_indices, top_k_values)
        return logits_filtered
