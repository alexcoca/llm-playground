from typing import Iterable, Literal, NamedTuple

import torch
from torchtyping import TensorType

from playground.logit_processors import LogitsProcessor, apply_logits_processors

Logits = TensorType["B", "L", "V"]
NextTokenLogits = TensorType["B", "V"]
NextTokenId = TensorType["B", 1]


class DecodingError(Exception):
    pass


class KVCache(NamedTuple):

    keys: TensorType["B", "H", "T", "Dh"]
    values: TensorType["B", "H", "T", "Dh"]


def prepare_logits(
    logits: Logits,
    positions: TensorType["B"] | None,
    processors: Iterable[LogitsProcessor] | None = None,
) -> NextTokenLogits:
    """Slices the sequence logits tensor, selects the logits,
    for the next token and pre-processes the logits with
    user-defined transformations."""
    logits = apply_logits_processors(logits, processors)

    if positions is not None:
        positions = positions.to(device=logits.device, dtype=torch.long)
        idx = torch.arange(logits.shape[0], device=logits.device)
        logits = logits[idx, positions, :]
    else:
        logits = logits[:, -1, :]

    return logits


def get_next_token_ids(
    logits: Logits,
    positions: TensorType["B"] | None = None,
    processors: Iterable[LogitsProcessor] | None = None,
    decoding_type: Literal["greedy", "sample"] = "greedy",
) -> NextTokenId:

    this_step_logits = prepare_logits(logits, positions, processors)

    if decoding_type == "greedy":
        return torch.argmax(this_step_logits, dim=-1, keepdim=True)

    if decoding_type == "sample":
        probas = torch.softmax(this_step_logits, dim=-1)
        return torch.multinomial(probas, num_samples=1)

    raise ValueError(f"Unknown decoding type: {decoding_type}")


def should_stop_generation(
    next_token_ids: TensorType["B"], eos_token_id: int | None = None
) -> bool:
    if eos_token_id is None:
        return False
    return (next_token_ids == eos_token_id).all()


def should_truncate(pos: TensorType["B"], context_len: int) -> bool:
    return (pos >= context_len - 1).all().item()


def extend_with_next_token(
    seq: TensorType["B", "L"],
    pos: TensorType["B"],
    next_token: TensorType["B", 1],
    *,
    in_place: bool = False,
) -> TensorType["B", "L"]:
    assert ((pos >= 0) & (pos < seq.size(1))).all().item(), "pos out of bounds"
    out = seq if in_place else seq.clone()
    idx = torch.arange(out.shape[0], device=out.device)
    out[idx, pos] = next_token.squeeze(1)
    return out


def increment_pos(
    pos: TensorType["B"],
    next_token_ids: TensorType["B", 1],
    finished_mask: TensorType["B"],
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
    cache_pos: TensorType["B"], Tmax: int
) -> TensorType["B", 1, 1, "Tmax"]:
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


def create_query_mask(
    prompt_len: TensorType["B"],
    max_seq_len: int,
) -> TensorType["B", 1, "T", 1]:
    """Mask padded rows during batched decoding."""
    B = prompt_len.shape[0]
    j = torch.arange(max_seq_len, device=prompt_len.device).expand(B, -1)

    return (j < prompt_len[:, None]).view(B, 1, max_seq_len, 1)


def init_output(
    inputs: TensorType["B", "Lin"],
    max_step: int,
) -> TensorType["B", "Lin"]:

    outputs = torch.empty(
        inputs.size(0), max_step, dtype=inputs.dtype, device=inputs.device
    )
    outputs[:, : inputs.size(1)] = inputs
    return outputs


def generate_text_simple(
    model, idx: TensorType["B", "L"], max_new_tokens: int, context_size: int
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
