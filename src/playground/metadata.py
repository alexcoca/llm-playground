PACKAGE_NAME = "playground"

# symbols used in our Transformers implementation
TOK_EMBEDDING = "tok_embedding"
POS_EMBEDDING = "pos_embedding"
BLOCKS = "blocks"
LAYER_NORM_PRE_ATT = "layer_norm_pre_att"
ATTENTION = "attention"
W_QKV = "W_qkv"
W_OUT = "W_out"
LAYER_NORM_PRE_FFN = "layer_norm_pre_ffn"
FFN = "ffn"
FFN_FC = "layers.0"  # First linear in FFN sequential
FFN_PROJ = "layers.2"  # Second linear in FFN sequential
FINAL_NORM = "final_norm"
LM_HEAD = "lm_head"
