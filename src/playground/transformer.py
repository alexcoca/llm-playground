import logging
from typing import TypeVar

from omegaconf import DictConfig
import torch
from torch import nn
from torchtyping import TensorType

from playground.inference_utils import (
    create_cache_key_mask,
    get_next_token_ids,
    KVCache,
    Logits,
    should_truncate,
    DecodingError,
    extend_with_next_token,
    increment_pos,
    init_output,
)
from playground.layers_optimised import TransformerBlock
from playground.transformer_mixin import TransformerMixin
from playground.transformer_utils import create_pad_mask

logger = logging.getLogger(__name__)

B = TypeVar("B")  # batch size
D = TypeVar("D")  # model embedding dimension
H = TypeVar("H")  # number of attention heads
L = TypeVar("L")  # sequence length
Lin = TypeVar("Lin")  # input sequence length
Lout = TypeVar("Lout")  # output sequence length
Tmax = TypeVar("Tmax")  # max cache size
V = TypeVar("V")  # vocabulary size


class Transformer(TransformerMixin, nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.tok_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embedding = nn.Embedding(cfg.context_length, cfg.embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.drop_embed = nn.Dropout(cfg.embed_drop_rate)
        self.final_norm = nn.LayerNorm(cfg.embed_dim)
        # don't register lm_head as a module if we tie weights to
        # avoid complex state dict management and custom load/save
        # logic -  use nn.functional.linear instead to get logits
        self.tie_weights = cfg.tie_weights
        if self.tie_weights:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        # cache positions
        self.register_buffer(
            "pos_ids",
            torch.arange(cfg.context_length, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )
        self.context_length = cfg.context_length

    def forward(
        self,
        inputs: TensorType[B, L],
        attention_mask: TensorType[B, L] | None = None,
    ) -> TensorType[B, L, V]:

        res_stream = self.encode_inputs(inputs)
        # print(self.blocks[0])
        # print(self.blocks[0].attention.__class__.__name__)
        for i, block in enumerate(self.blocks):
            res_stream = block(res_stream, attention_mask=attention_mask)
            # print(f"block {i=}, sum along dim -1 for res stream {res_stream.sum(dim=-1)=}")
        print(
            f"res stream in transformer forward, before final norm: {res_stream.sum(-1)}"
        )
        res_stream = self.final_norm(res_stream)
        print(f"res stream in transformer forward {res_stream.sum(-1)}")
        return self.get_logits(res_stream)

    def get_logits(self, res_stream: TensorType[B, L, D]) -> TensorType[B, L, V]:
        if self.tie_weights:
            logits = nn.functional.linear(res_stream, self.tok_embedding.weight)
        else:
            logits = self.lm_head(res_stream)
        return logits

    def encode_inputs(
        self, inputs: TensorType[B, L], pos_offset: TensorType[B] | None = None
    ) -> TensorType[B, L, H]:
        _, seq_len = inputs.shape
        # (B, L, H)
        tok_embeds = self.tok_embedding(inputs)
        pos_ids = self.pos_ids[:, :seq_len]
        if pos_offset is not None:
            pos_ids = pos_ids + pos_offset[:, None]
        assert (
            pos_ids.max().item() < self.context_length
        ), "position exceeds context length"
        pos_embeds = self.pos_embedding(pos_ids)
        res_stream = tok_embeds + pos_embeds
        return self.drop_embed(res_stream)

    def initialise_cache(
        self,
        res_stream: TensorType[B, L, H],
        cache_pos: TensorType[B],
        prompt_len: TensorType[B],
        max_new_tokens: int = 20,
    ) -> tuple[Logits | None, list[KVCache] | None]:
        """Run a forward pass through the model and save
        the keys and values for all prompt tokens across
        all layers."""
        max_step = max_new_tokens + prompt_len.max().item()
        keys_allowed = create_cache_key_mask(prompt_len, Tmax=max_step)
        layer_caches = []
        for i, block in enumerate(self.blocks):
            print(f"Running block {i}")
            res_stream, cache = block.forward_cached(
                res_stream, None, cache_pos, keys_allowed
            )
            layer_caches.append(cache)
        res_stream = self.final_norm(res_stream)
        logits = self.get_logits(res_stream)
        return logits, layer_caches

    @torch.no_grad()
    def generate(
        self,
        inputs: TensorType[B, L],
        use_cache: bool = True,
        max_new_tokens: int = 20,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> TensorType[B, Tmax]:
        """Batched generation for decoder-only transformers."""
        self.eval()
        if use_cache:
            return self._decode_with_cached_key_values(
                inputs,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                max_new_tokens=max_new_tokens,
            )
        else:
            return self._decode(
                inputs,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                max_new_tokens=max_new_tokens,
            )

    def _decode(
        self,
        inputs: TensorType[B, Lin],
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        max_new_tokens: int = 20,
    ) -> TensorType[B, Lout]:

        B = inputs.shape[0]
        pad_mask = create_pad_mask(inputs, pad_token_id)
        prompt_lengths = pad_mask.sum(dim=-1)
        max_prompt_len = prompt_lengths.max().item()
        max_step = max_prompt_len + max_new_tokens
        finished_mask = torch.zeros(B, dtype=torch.bool, device=inputs.device)
        prev_output = init_output(inputs, max_step)
        next_write_loc = prompt_lengths
        cur_len = max_prompt_len
        for step in range(max_new_tokens):
            if should_truncate(next_write_loc, self.context_length):
                raise DecodingError(
                    "Some outputs exceeded context length, truncation not supported."
                )
            j = torch.arange(cur_len, device=next_write_loc.device)
            attention_mask = j[None, :] < next_write_loc[:, None]
            logits = self.forward(
                prev_output[:, :cur_len], attention_mask=attention_mask
            )
            next_token_ids = get_next_token_ids(logits, positions=next_write_loc - 1)
            prev_output = extend_with_next_token(
                prev_output, next_write_loc, next_token_ids
            )
            increment_pos(
                next_write_loc, next_token_ids, finished_mask, eos_token_id=eos_token_id
            )
            cur_len += 1
            if finished_mask.all().item():
                break
        return prev_output[:, :cur_len]

    def _decode_with_cached_key_values(
        self,
        inputs: TensorType[B, Lin],
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        max_new_tokens: int = 20,
    ) -> TensorType[B, Lout]:
        B = inputs.shape[0]
        pad_mask = create_pad_mask(inputs, pad_token_id)
        prompt_lengths = pad_mask.sum(dim=-1)
        res_stream = self.encode_inputs(inputs)
        # Transformer blocks move the cache positions and
        # key masks to the correct device as necessary -
        # see TransformerBlock.forward_cached
        max_step = prompt_lengths.max().item() + max_new_tokens
        outputs = init_output(inputs, max_step)
        finished_mask = torch.zeros(B, dtype=torch.bool, device=inputs.device)
        next_write_pos = torch.zeros(B, dtype=torch.long, device=inputs.device)
        logits, caches = self.initialise_cache(
            res_stream,
            cache_pos=next_write_pos,
            prompt_len=prompt_lengths,
            max_new_tokens=max_new_tokens,
        )
        next_write_pos = prompt_lengths
        for step in range(max_new_tokens):
            if logits.shape[1] > 1:
                # input sequences may have different lengths, so we gather the logits
                next_token_ids = get_next_token_ids(
                    logits, positions=prompt_lengths - 1
                )
            else:
                # usually we generate a single token, no position info required
                next_token_ids = get_next_token_ids(logits)

            if should_truncate(next_write_pos, self.context_length):
                raise DecodingError(
                    "Some outputs exceeded model context length, "
                    "truncation not supported for cached decoding"
                )
            outputs = extend_with_next_token(outputs, next_write_pos, next_token_ids)
            j = torch.arange(max_step, device=next_write_pos.device)
            # mask out future time steps in the cache - note that the
            # cache is filled *before* we call the SDPA, that is
            # **after** next_write_pos slot in the cache has been filled,
            # hence the <= in mask construction below
            mask = (j[None, :] <= next_write_pos[:, None]).view(B, 1, 1, max_step)
            next_tokens = self.encode_inputs(next_token_ids, pos_offset=next_write_pos)
            res_stream = next_tokens
            for layer, block in enumerate(self.blocks):
                res_stream, _ = block.forward_cached(
                    res_stream, caches[layer], next_write_pos, cache_keys_allowed=mask
                )
            res_stream = self.final_norm(res_stream)
            logits = self.get_logits(res_stream)
            increment_pos(
                next_write_pos, next_token_ids, finished_mask, eos_token_id=eos_token_id
            )
            if finished_mask.all().item():
                break
        return outputs[:, : next_write_pos.max().item()]
