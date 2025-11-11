import torch
import torch.nn.functional as F
from torchtyping import TensorType


def language_modelling_loss(
    next_token_logits: TensorType["B", "L", "V"],
    next_token_ids: TensorType["B", "L"],
    ignore_idx: int = -100,
) -> torch.Tensor:
    """Compute language modelling loss at each
    position in a batch of sequences.

    Parameters
    ----------
    next_token_logits
        Next token logits.
    next_token_ids
        Next token IDs.
    ignore_idx
        Contribution of positions marked with this
        index in the target are ignored in the loss
        calculation (and therefore the input gradients).

    Returns
    -------
    Per token language modelling loss.
    """

    # predicting next token at every position is
    # equivalent to performing B * L next token
    # classifications in parallel
    logits_flat = next_token_logits.flatten(0, 1)
    targets_flat = next_token_ids.flatten()

    return F.cross_entropy(
        input=logits_flat, target=targets_flat, ignore_index=ignore_idx
    )
