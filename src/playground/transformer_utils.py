import logging
from typing import TypeVar

import torch
from torchtyping import TensorType


logger = logging.getLogger(__name__)

B = TypeVar("B")  # batch size
Dh = TypeVar("Dh")  # attention head dimension
H = TypeVar("H")  # number of attention heads
Lk = TypeVar("Lk")  # key sequence length
Lq = TypeVar("Lq")  # query sequence length
L = TypeVar("L")  # input sequence length


def create_causal_padding_mask(
    keys: TensorType[B, H, Lk, Dh],
    queries: TensorType[B, H, Lq, Dh],
    padding_mask: TensorType[B, L],
) -> TensorType[B, H, Lq, Lk]:

    B, Lk, Lq = keys.shape[0], keys.shape[-2], queries.shape[-2]
    # SDPA mask semantics is True for positions (query, key pairs)
    # that should be attended to
    padding_mask = padding_mask.to(device=queries.device, dtype=torch.bool)
    key_padding = padding_mask.reshape(B, 1, 1, Lk)
    query_padding = padding_mask.reshape(B, 1, Lq, 1)
    causal = torch.tril(
        torch.ones(
            Lq,
            Lk,
            dtype=torch.bool,
            device=queries.device,
        ),
        diagonal=0,
    ).view(1, 1, Lq, Lk)
    attn_mask = causal & key_padding & query_padding
    return attn_mask


def create_pad_mask(
    inputs: TensorType[B, L], pad_token_id: int | None
) -> TensorType[B, L]:

    if pad_token_id is None:
        if inputs.shape[0] > 1:
            logger.warning(
                "ID of padding token has not been specified, "
                "batch elements will not be padded"
            )
        return torch.ones_like(inputs)

    return inputs != pad_token_id
