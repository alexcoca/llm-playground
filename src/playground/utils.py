import torch.nn as nn


def print_num_params(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")


def print_attention_num_params(model: nn.Module):
    params = sum(p.numel() for p in model.blocks[0].attention.parameters())
    print(f"Total number of parameters of attention in layer 0: {params:,}")


def print_ffn_num_params(model: nn.Module):
    params = sum(p.numel() for p in model.blocks[0].ffn.parameters())
    print(f"Total number of parameters of FFN in layer 0: {params:,}")
