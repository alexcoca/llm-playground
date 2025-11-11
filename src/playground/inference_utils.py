from typing import NamedTuple, TypeVar

import torch
from torchtyping import TensorType

B = TypeVar("B")  # batch size
Dh = TypeVar("Dh")  # attention head dimension
H = TypeVar("H")  # number of attention heads
L = TypeVar("L")  # sequence length
Lin = TypeVar("Lin")  # input sequence length
T = TypeVar("T")
Tmax = TypeVar("Tmax")  # max cache size
V = TypeVar("V")  # vocabulary size


Logits = TensorType[B, L, V]
NextTokenId = TensorType[B, 1]


class DecodingError(Exception):
    pass


class KVCache(NamedTuple):

    keys: TensorType[B, H, T, Dh]
    values: TensorType[B, H, T, Dh]


def get_next_token_ids(
    logits: Logits, positions: TensorType[B] | None = None
) -> NextTokenId:
    if positions is not None:
        idx = torch.arange(logits.shape[0], device=logits.device)
        logits = logits[idx, positions, :]
    else:
        logits = logits[:, -1, :]
    probas = torch.softmax(logits, dim=-1)
    return torch.argmax(probas, dim=-1, keepdim=True)


def should_stop_generation(
    next_token_ids: TensorType[B], eos_token_id: int | None = None
) -> bool:
    if eos_token_id is None:
        return False
    return (next_token_ids == eos_token_id).all()


def should_truncate(pos: TensorType[B], context_len: int) -> bool:
    return (pos >= context_len - 1).all().item()


def extend_with_next_token(
    seq: TensorType[B, L],
    pos: TensorType[B],
    next_token: TensorType[B, 1],
    *,
    in_place: bool = False,
) -> TensorType[B, L]:
    assert ((pos >= 0) & (pos < seq.size(1))).all().item(), "pos out of bounds"
    out = seq if in_place else seq.clone()
    idx = torch.arange(out.shape[0], device=out.device)
    out[idx, pos] = next_token.squeeze(1)
    return out


def increment_pos(
    pos: TensorType[B],
    next_token_ids: TensorType[B, 1],
    finished_mask: TensorType[B],
    eos_token_id: int | None = None,
):
    """In-place increment the index of the last decoded token."""
    if eos_token_id is None:
        active_mask = torch.ones_like(pos, dtype=torch.bool)
    else:
        not_eos = next_token_ids.squeeze(1) != eos_token_id
        active_mask = ~finished_mask & not_eos
        finished_mask |= ~not_eos
    pos[active_mask] += 1


def create_cache_key_mask(
    cache_pos: TensorType[B], Tmax: int
) -> TensorType[B, 1, 1, Tmax]:
    """Masks unfilled positions in cache to avoid attending
    over them during cached inference.

    Parameters
    ----------
    cache_pos
        The cache position currently being filled - should be *masked*
        until we write the new keys/values later during the forward pass.
    """
    B = cache_pos.shape[0]
    return (
        torch.arange(Tmax, device=cache_pos.device).expand(B, -1)
        < cache_pos.unsqueeze(1)
    ).view(B, 1, 1, Tmax)


def init_output(
    inputs: TensorType[B, Lin],
    max_step: int,
) -> TensorType[B, Lin]:

    outputs = torch.empty(
        inputs.size(0), max_step, dtype=inputs.dtype, device=inputs.device
    )
    outputs[:, : inputs.size(1)] = inputs
    return outputs


def generate_text_simple(
    model, idx: TensorType[B, L], max_new_tokens: int, context_size: int
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (B,1)
        idx = torch.cat((idx, idx_next), dim=-1)

    return idx
