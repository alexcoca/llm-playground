from typing import TypeVar

import torch
from omegaconf import DictConfig
from torch import nn as nn
from torch.nn import functional as F, LayerNorm, GELU
from torchtyping import TensorType

from playground.inference_utils import KVCache
from playground.transformer_utils import create_causal_padding_mask


B = TypeVar("B")  # batch size
D = TypeVar("D")  # model embedding dimension
D_out = TypeVar("D_out")  # attention output dimension
Dh = TypeVar("Dh")  # attention head dimension
H = TypeVar("H")  # number of attention heads
L = TypeVar("L")  # sequence length
Lq = TypeVar("Lq")  # query sequence length
Tmax = TypeVar("Tmax")  # max cache size

QKVTuple = tuple[
    TensorType[B, H, L, Dh],
    TensorType[B, H, L, Dh],
    TensorType[B, H, L, Dh],
]
PadMask = TensorType[B, L]


class FeedForwardNetwork(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ff_hidden_size_ratio * cfg.embed_dim),
            GELU(),
            nn.Linear(cfg.ff_hidden_size_ratio * cfg.embed_dim, cfg.embed_dim),
        )

    def forward(self, input: TensorType[B, L, H]) -> TensorType[B, L, H]:
        return self.layers(input)


class MultiHeadAttentionOptimised(nn.Module):

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int = 1,
        dropout: float = 0.2,
        qkv_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.head_dim = d_out // num_heads
        self.num_heads = num_heads
        self.W_qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.W_out = nn.Linear(d_out, d_out)
        self.attn_dropout = nn.Dropout(dropout)
        assert self.W_out.in_features == self.num_heads * self.head_dim

    def forward(
        self,
        x: TensorType[B, L, D],
        attention_mask: TensorType[B, L] | None = None,
    ) -> TensorType[B, L, D_out]:

        q, k, v = self.prepare_qkv(x)
        if attention_mask is not None:
            attn_mask = create_causal_padding_mask(k, q, attention_mask)
        else:
            attn_mask = None
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True if attn_mask is None else False,
            attn_mask=attn_mask,
        )
        # (B, H, L, d_head) -> ... -> (B, L, d_out)
        context = attn.transpose(1, 2).flatten(2)
        print(f"forward {context.sum(dim=-1)=}")
        return self.W_out(context)

    def forward_cached(
        self,
        x_step: TensorType[B, Lq, D],
        kv_cache: KVCache,
        cache_pos: TensorType[B],
        cache_keys_allowed: TensorType[B, 1, 1, Tmax] | None = None,
    ) -> TensorType[B, Lq, D_out]:

        q, k, v = self.prepare_qkv(x_step)
        B, n_head, Lq, head_dim = q.shape
        Lk = kv_cache.keys.shape[-2]
        causal_mask = None
        attn_mask = cache_keys_allowed
        if Lq > 1:
            assert (
                (cache_pos + Lq) <= kv_cache.keys.shape[-2]
            ).all(), "Cache overflow: attempted to write beyond allocated cache length."

            time_offset = torch.arange(Lq, device=cache_pos.device)
            # rows are time slots to be filled in the cache for each
            # batch element
            cache_pos = cache_pos[:, None] + time_offset  # [B, Lq]
            key_pos = torch.arange(Lk, device=cache_pos.device)
            # [B, 1, Lq, Lk]
            causal_mask = key_pos[None, None, None, :] <= cache_pos[:, None, :, None]

        if causal_mask is not None:
            attn_mask = causal_mask & cache_keys_allowed
        idx = cache_pos.reshape(B, 1, Lq, 1).expand(
            -1, self.num_heads, -1, self.head_dim
        )
        # TODO: QUERY PADDING, FIX ATTENTION OVER CACHE
        kv_cache.keys.scatter_(dim=-2, index=idx, src=k)
        kv_cache.values.scatter_(dim=-2, index=idx, src=v)
        attn = F.scaled_dot_product_attention(
            q,
            kv_cache.keys,
            kv_cache.values,
            attn_mask=attn_mask,
        )
        context = attn.transpose(1, 2).flatten(2)
        print(f"forward_cached{context.sum(dim=-1)=}")

        return self.W_out(context)

    def prepare_qkv(self, x: TensorType[B, L, D]) -> QKVTuple:
        B, L, _ = x.shape
        q, k, v = self.W_qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        return q, k, v


class TransformerBlock(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.layer_norm_pre_att = LayerNorm(cfg.embed_dim)
        self.attention = MultiHeadAttentionOptimised(
            d_in=cfg.embed_dim,
            d_out=cfg.embed_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.att_drop_rate,
            qkv_bias=cfg.qkv_bias,
        )
        self.post_attn_dropout = nn.Dropout(cfg.post_attn_drop_rate)
        self.layer_norm_pre_ffn = LayerNorm(cfg.embed_dim)
        self.ffn = FeedForwardNetwork(cfg)
        self.post_ffn_dropout = nn.Dropout(cfg.post_ffn_drop_rate)

    def forward(
        self,
        res_stream: TensorType[B, L, D],
        attention_mask: PadMask | None = None,
    ) -> TensorType[B, L, D]:

        attention_input = self.layer_norm_pre_att(res_stream)
        attention_output = self.post_attn_dropout(
            self.attention(attention_input, attention_mask=attention_mask)
        )
        res_stream = res_stream + attention_output

        ffn_input = self.layer_norm_pre_ffn(res_stream)
        ffn_output = self.post_ffn_dropout(self.ffn(ffn_input))
        return res_stream + ffn_output

    def forward_cached(
        self,
        res_stream: TensorType[B, Lq, D],
        kv_cache: KVCache | None,
        cache_pos: TensorType[B],
        cache_keys_allowed: TensorType[B, 1, 1, Tmax] | None = None,
    ) -> tuple[TensorType[B, Lq, D], KVCache]:

        cache_pos = cache_pos.to(device=res_stream.device, dtype=torch.long)

        if kv_cache is None:
            assert (
                cache_keys_allowed is not None
            ), "Require keys mask to initialise cache size"
            kv_cache = self.init_cache(
                res_stream, cache_size=cache_keys_allowed.shape[-1]
            )
        else:
            cache_keys_allowed = cache_keys_allowed.to(
                device=res_stream.device, dtype=torch.bool
            )

        attention_input = self.layer_norm_pre_att(res_stream)
        attention_output = self.post_attn_dropout(
            self.attention.forward_cached(
                attention_input,
                kv_cache=kv_cache,
                cache_pos=cache_pos,
                cache_keys_allowed=cache_keys_allowed,
            )
        )
        res_stream = res_stream + attention_output

        ffn_input = self.layer_norm_pre_ffn(res_stream)
        ffn_output = self.post_ffn_dropout(self.ffn(ffn_input))

        return res_stream + ffn_output, kv_cache

    def init_cache(
        self, res_stream: TensorType[B, Lq, D], *, cache_size: int
    ) -> KVCache:
        B = res_stream.shape[0]
        H = self.attention.num_heads
        Dh = self.attention.head_dim
        cache_shape = (B, H, cache_size, Dh)
        return KVCache(
            keys=torch.zeros(
                *cache_shape,
                device=res_stream.device,
                dtype=res_stream.dtype,
                requires_grad=False,
            ),
            values=torch.zeros(
                *cache_shape,
                device=res_stream.device,
                dtype=res_stream.dtype,
                requires_grad=False,
            ),
        )
