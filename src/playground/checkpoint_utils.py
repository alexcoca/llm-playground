"""Weight mapping utilities for loading HuggingFace GPT-2 weights into our Transformer."""

import logging
import re
from typing import Dict

import torch

from playground.metadata import (
    ATTENTION,
    BLOCKS,
    FFN,
    FFN_FC,
    FFN_PROJ,
    FINAL_NORM,
    LAYER_NORM_PRE_ATT,
    LAYER_NORM_PRE_FFN,
    POS_EMBEDDING,
    TOK_EMBEDDING,
    W_OUT,
    W_QKV,
)

# mapping from local configs to huggingface repo names
CONFIG_TO_HF_REPO = {
    "gpt2_small": "gpt2",
    # 'gpt2_medium': 'gpt2-medium',
    # 'gpt2_large': 'gpt2-large',
    # 'gpt2_xl': 'gpt2-xl',
}

logger = logging.getLogger(__name__)


def convert_hf_key_to_ours(hf_key: str) -> str:
    """
    Convert a HuggingFace GPT-2 state dict key to our model's naming convention.

    Args:
        hf_key: Key from HF's state dict (e.g., 'transformer.h.0.attn.c_attn.weight')

    Returns:
        Corresponding key in our model (e.g., 'blocks.0.attention.W_qkv.weight')

    Raises:
        ValueError: If the key pattern is unrecognized
    """
    # Embeddings
    if hf_key == "transformer.wte.weight":
        return f"{TOK_EMBEDDING}.weight"
    if hf_key == "transformer.wpe.weight":
        return f"{POS_EMBEDDING}.weight"

    # Final layer norm
    if hf_key.startswith("transformer.ln_f"):
        return hf_key.replace("transformer.ln_f", FINAL_NORM)

    # LM head (same name)
    if hf_key.startswith("lm_head"):
        return hf_key

    # Transformer blocks - use regex to extract layer number
    block_match = re.match(r"transformer\.h\.(\d+)\.(.*)", hf_key)
    if not block_match:
        raise ValueError(f"Unrecognized key pattern: {hf_key}")

    layer_num, rest = block_match.groups()

    # Map block components
    replacements = {
        "ln_1": f"{LAYER_NORM_PRE_ATT}",
        "attn.c_attn": f"{ATTENTION}.{W_QKV}",
        "attn.c_proj": f"{ATTENTION}.{W_OUT}",
        "ln_2": f"{LAYER_NORM_PRE_FFN}",
        "mlp.c_fc": f"{FFN}.{FFN_FC}",
        "mlp.c_proj": f"{FFN}.{FFN_PROJ}",
    }

    for hf_pattern, our_pattern in replacements.items():
        if rest.startswith(hf_pattern):
            param_name = rest[len(hf_pattern) :]  # e.g., '.weight' or '.bias'
            return f"{BLOCKS}.{layer_num}.{our_pattern}{param_name}"

    raise ValueError(f"Unrecognized block component: {rest}")


def load_hf_weights(
    model: torch.nn.Module, hf_state_dict: Dict[str, torch.Tensor], strict: bool = True
) -> Dict[str, str]:
    """
    Load HuggingFace GPT-2 weights into our Transformer model.

    Args:
        model: Our Transformer instance
        hf_state_dict: State dict from HuggingFace GPT2LMHeadModel
        strict: If True, raise error on missing/unexpected keys

    Returns:
        Dictionary mapping HF keys to our keys (for debugging)

    Raises:
        RuntimeError: If shapes don't match or required keys are missing
    """
    our_state_dict = model.state_dict()
    key_mapping = {}

    converted_weights = {}
    for hf_key, hf_weight in hf_state_dict.items():
        try:
            our_key = convert_hf_key_to_ours(hf_key)
        except ValueError as e:
            if strict:
                raise
            print(f"Warning: Skipping {hf_key}: {e}")
            continue

        # Handle transpose for Linear layers
        # HF stores weights as (in_features, out_features)
        # PyTorch expects (out_features, in_features)
        if our_key in our_state_dict and our_key not in {
            "tok_embedding.weight",
            "pos_embedding.weight",
        }:
            if len(hf_weight.shape) == 2:
                hf_weight = hf_weight.T

        converted_weights[our_key] = hf_weight
        key_mapping[hf_key] = our_key

    # Check for missing keys
    missing_keys = set(our_state_dict.keys()) - set(converted_weights.keys())
    unexpected_keys = set(converted_weights.keys()) - set(our_state_dict.keys())

    if missing_keys and strict:
        raise RuntimeError(f"Missing keys in HF weights: {missing_keys}")
    if unexpected_keys and strict:
        if unexpected_keys == {"lm_head.weight"} and model.tie_weights:
            pass
        else:
            raise RuntimeError(f"Unexpected keys from HF: {unexpected_keys}")

    model.load_state_dict(converted_weights, strict=False)

    logger.info(f"Successfully loaded {len(converted_weights)} weight tensors")
    if missing_keys:
        logger.warning(
            f"Warning: {len(missing_keys)} keys not found in HF weights: {missing_keys}"
        )

    return key_mapping


def resolve_config_to_hf_repo_name(config_name: str) -> str:
    try:
        return CONFIG_TO_HF_REPO[config_name]
    except KeyError:
        raise ValueError(f"No Huggingface repo specified for config {config_name}!")
