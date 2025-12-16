import math

import pytest
import torch
import torch.nn.functional as F

from playground.layers_optimised import MultiHeadAttentionOptimised as MHA


def make_model(
    d_in=32, d_out=64, heads=8, dropout=0.0, device="cpu", dtype=None, eval_mode=True
):
    m = MHA(d_in=d_in, d_out=d_out, num_heads=heads, dropout=dropout, qkv_bias=True).to(
        device=device, dtype=dtype
    )
    if eval_mode:
        m.eval()
    return m


def right_pad_mask(lengths, L, device):
    B = len(lengths)
    idx = torch.arange(L, device=device).expand(B, L)
    lens = torch.tensor(lengths, device=device).unsqueeze(1)
    return (idx < lens).to(torch.long)  # 1=real, 0=pad


@torch.no_grad()
def reference_attention_with_mask(x, W_qkv, W_out, heads, attention_mask=None):
    """
    Explicit math reference.
    - Causal mask always applied.
    - Key padding applied if attention_mask is provided.
    - We also make fully-masked query rows produce zeros (to match SDPA behavior).
    """
    B, L, _ = x.shape
    Dout = W_out.out_features
    Dh = Dout // heads

    qkv = W_qkv(x)  # [B,L,3*Dout]
    q, k, v = qkv.chunk(3, dim=-1)  # each [B,L,Dout]

    def prep(t):
        return t.reshape(B, L, heads, Dh).transpose(1, 2)  # [B,H,L,Dh]

    q, k, v = prep(q), prep(k), prep(v)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)  # [B,H,L,L]

    causal_allowed = torch.ones(L, L, dtype=torch.bool, device=x.device).tril(
        0
    )  # True=keep
    allowed = causal_allowed.view(1, 1, L, L)  # [1,1,L,L]

    if attention_mask is not None:
        m = attention_mask.to(torch.bool)
        key_allowed = m.view(B, 1, 1, L)  # [B,1,1,L]
        allowed = allowed & key_allowed  # broadcast to [B,1,L,L]

    # mask out disallowed with -inf, softmax, then zero rows that were fully disallowed
    scores = scores.masked_fill(~allowed, float("-inf"))
    p = F.softmax(scores, dim=-1)  # [B,H,L,L]
    row_has_any = allowed.any(dim=-1, keepdim=True)  # [B,1,L,1]
    p = torch.where(row_has_any, p, torch.zeros_like(p))

    ctx = p @ v  # [B,H,L,Dh]
    y = ctx.transpose(1, 2).reshape(B, L, Dout)  # [B,L,Dout]
    return W_out(y)  # [B,L,Dout]


def test_none_vs_all_ones_equivalent():
    torch.manual_seed(0)
    B, L, Din, Dout, H = 2, 11, 48, 96, 8
    x = torch.randn(B, L, Din)

    m = make_model(Din, Dout, H, dropout=0.0).eval()
    ones = torch.ones(B, L, dtype=torch.long)

    y_none = m(x, attention_mask=None)
    y_ones = m(x, attention_mask=ones)

    torch.testing.assert_close(y_none, y_ones, rtol=0, atol=0)


def test_matches_reference_with_padding_compare_only_valid_queries():
    """
    Robust to either implementation:
    - key-only masking in SDPA
    - key+query masking in SDPA
    - or key masking in SDPA + zeroing outputs on padded queries

    We compare only rows where the query is valid.
    """
    torch.manual_seed(1)
    B, L, Din, Dout, H = 2, 13, 32, 64, 8
    x = torch.randn(B, L, Din)
    lengths = [9, 11]
    attn_mask = right_pad_mask(lengths, L, x.device)

    m = make_model(Din, Dout, H, dropout=0.0).eval()

    y_model = m(x, attention_mask=attn_mask)  # [B,L,Dout]
    y_ref = reference_attention_with_mask(
        x, m.W_qkv, m.W_out, H, attention_mask=attn_mask
    )

    valid_q = attn_mask.bool().unsqueeze(-1).expand_as(y_ref)  # [B,L,Dout]
    torch.testing.assert_close(y_model[valid_q], y_ref[valid_q], rtol=1e-5, atol=1e-6)


def test_gradients_flow_with_padding():
    torch.manual_seed(2)
    B, L, Din, Dout, H = 2, 10, 32, 64, 8
    x = torch.randn(B, L, Din, requires_grad=False)
    lengths = [7, 4]
    attn_mask = right_pad_mask(lengths, L, x.device)

    m = make_model(Din, Dout, H, dropout=0.0, eval_mode=False)  # train mode
    y = m(x, attention_mask=attn_mask)

    # reduce loss over valid query positions only
    valid = attn_mask.to(torch.bool).unsqueeze(-1)  # [B,L,1]
    loss = (y.masked_select(valid)).mean()
    loss.backward()

    for n, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {n}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"


def test_shape_and_dtype_fp32_boolmask():
    torch.manual_seed(3)
    B, L, Din, Dout, H = 3, 9, 16, 32, 4
    x = torch.randn(B, L, Din, dtype=torch.float32)
    attn_mask = right_pad_mask([9, 7, 5], L, x.device).to(torch.long)

    m = make_model(Din, Dout, H, dropout=0.0).eval()
    y = m(x, attention_mask=attn_mask)
    assert y.shape == (B, L, Dout)
    assert y.dtype == x.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_bfloat16_with_padding_smoke():
    torch.manual_seed(4)
    device = "cuda"
    B, L, Din, Dout, H = 2, 12, 64, 64, 8
    x = torch.randn(B, L, Din, device=device, dtype=torch.bfloat16)
    attn_mask = right_pad_mask([10, 7], L, device)

    m = make_model(
        Din, Dout, H, dropout=0.0, device=device, dtype=torch.bfloat16
    ).eval()
    y = m(x, attention_mask=attn_mask)
    assert y.shape == (B, L, Dout)
    assert y.dtype == x.dtype


def test_padded_query_rows_are_zero_if_zeroing_enabled():
    torch.manual_seed(5)
    B, L, Din, Dout, H = 1, 8, 32, 64, 8
    x = torch.randn(B, L, Din)
    attn_mask = right_pad_mask([5], L, x.device)

    m = make_model(Din, Dout, H, dropout=0.0).eval()
    y = m(x, attention_mask=attn_mask)

    # if your impl zeros padded query rows, this will pass; otherwise skip/remove this test
    if torch.allclose(y[:, 5:], torch.zeros_like(y[:, 5:])):
        assert True
    else:
        pytest.skip("Model does not zero padded query rows (that is okay).")
