import math

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from xformers.ops import memory_efficient_attention


def eager_attention_forward(
    attention_mask,
    batch_size,
    head_dim,
    query_states,
    key_states,
    value_states,
):
    num_att_heads = 8
    num_key_value_heads = 1
    num_key_value_groups = num_att_heads // num_key_value_heads

    # query_states: batch_size, sequence_length, num_att_head, head_dim
    # key_states: batch_size, sequence_length, num_key_value_head, head_dim
    # value_states: batch_size, sequence_length, num_key_value_head, head_dim
    sequence_length = key_states.shape[1]

    key_states = key_states[:, :, :, None, :].expand(
        batch_size,
        sequence_length,
        num_key_value_heads,
        num_key_value_groups,
        head_dim,
    )
    key_states = key_states.reshape(
        batch_size,
        sequence_length,
        num_key_value_heads * num_key_value_groups,
        head_dim,
    )

    value_states = value_states[:, :, :, None, :].expand(
        batch_size,
        sequence_length,
        num_key_value_heads,
        num_key_value_groups,
        head_dim,
    )
    value_states = value_states.reshape(
        batch_size,
        sequence_length,
        num_key_value_heads * num_key_value_groups,
        head_dim,
    )

    # Attention here is upcasted to float32 to match the original eager implementation.

    query_states = query_states.to(dtype=torch.float32)
    key_states = key_states.to(dtype=torch.float32)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    att_weights *= head_dim**-0.5
    big_neg = -2.3819763e38  # See gemma/modules.py

    masked_att_weights = torch.where(
        attention_mask[:, None, :, :], att_weights, big_neg
    )

    probs = nn.functional.softmax(masked_att_weights, dim=-1)
    probs = probs.to(dtype=value_states.dtype)

    # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
    # value_states: batch_size, sequence_length, num_att_heads, head_dim

    att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

    att_output = att_output.permute(0, 2, 1, 3)
    # we use -1 because sequence length can change
    att_output = att_output.reshape(
        batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
    )

    return att_output


def new_eager_attention_forward(
    attention_mask,
    batch_size,
    head_dim,
    query_states,
    key_states,
    value_states,
) -> torch.Tensor:
    """
    Performs eager attention, optimized with torch.einsum.

    Args:
        query_states: Query tensor of shape [batch_size, seq_len, num_attention_heads, head_dim].
        key_states: Key tensor of shape [batch_size, seq_len, num_key_value_heads, head_dim].
        value_states: Value tensor of shape [batch_size, seq_len, num_key_value_heads, head_dim].
        attention_mask: Attention mask tensor, typically [batch_size, 1, seq_len, seq_len] or [batch_size, seq_len, seq_len].

    Returns:
        Output tensor of shape [batch_size, seq_len, num_attention_heads * head_dim].
    """
    bsize, seq_len, num_att_heads, head_dim = query_states.shape
    num_key_value_heads = key_states.shape[2]
    num_key_value_groups = num_att_heads // num_key_value_heads

    key_states = einops.repeat(
        key_states, "b l h d -> b l (h g) d", g=num_key_value_groups
    )
    value_states = einops.repeat(
        value_states, "b l h d -> b l (h g) d", g=num_key_value_groups
    )

    query_states_permuted = torch.einsum("blhd->bhld", query_states)
    key_states_permuted = torch.einsum("blhd->bhld", key_states)

    att_weights = torch.einsum(
        "bhqd,bhkd->bhqk", query_states_permuted, key_states_permuted
    )
    att_weights *= head_dim**-0.5

    big_neg = -2.3819763e38
    masked_att_weights = torch.where(
        attention_mask[:, None, :, :], att_weights, big_neg
    )

    probs = nn.functional.softmax(masked_att_weights, dim=-1)
    probs = probs.to(dtype=value_states.dtype)

    value_states_permuted = torch.einsum("blhd->bhld", value_states)  # [B, H, L_v, D]
    att_output = torch.einsum(
        "bhqk,bhkv->bhqv", probs, value_states_permuted
    )  # [B, H, L_q, D]
    att_output = torch.einsum("bhld->blhd", att_output)  # [B, L, H, D]
    att_output = att_output.reshape(bsize, seq_len, num_att_heads * head_dim)

    return att_output


def xformer_attention_forward(
    attention_mask,
    batch_size,
    head_dim,
    query_states,
    key_states,
    value_states,
):
    query_states = query_states.reshape([256, 800, 1, 8, 64])
    key_states = key_states.reshape([256, 800, 1, 1, 64]).expand(256, 800, 1, 8, 64)
    value_states = value_states.reshape([256, 800, 1, 1, 64]).expand(256, 800, 1, 8, 64)

    attention_mask = attention_mask.reshape([256, 1, 1, 800, 800]).expand(
        256, 1, 8, 800, 800
    )

    att_output = memory_efficient_attention(
        query=query_states.to(torch.float32),
        key=key_states.to(torch.float32),
        value=value_states.to(torch.float32),
        attn_bias=~attention_mask * (-1e23),
    )

    return att_output.reshape(256, 800, -1)

def new_xformer_attention_forward(
    attention_mask,
    batch_size,
    head_dim,
    query_states,
    key_states,
    value_states,
):
    bsize, seq_len, num_att_heads, head_dim = query_states.shape
    num_key_value_heads = key_states.shape[2]
    num_key_value_groups = num_att_heads // num_key_value_heads

    query_states = einops.rearrange(
        query_states, "b l (h g) d -> b l h g d", g=num_key_value_groups
    )
    key_states = einops.repeat(
        key_states, "b l h d -> b l h g d", g=num_key_value_groups
    )
    value_states = einops.repeat(
        value_states, "b l h d -> b l h g d", g=num_key_value_groups
    )
    attention_mask = einops.repeat(
        attention_mask,
        "b l s -> b h g l s",
        h=num_key_value_heads,
        g=num_key_value_groups,
    )

    big_neg = -2.3819763e38

    att_output = memory_efficient_attention(
        query=query_states.to(torch.float32),
        key=key_states.to(torch.float32),
        value=value_states.to(torch.float32),
        attn_bias=~attention_mask * big_neg,
    )
    att_output = att_output.reshape(bsize, seq_len, -1)

    return att_output


def sdpa_attention_forward(
    attention_mask,
    batch_size,
    head_dim,
    query_states,
    key_states,
    value_states,
):
    query_states = query_states.permute(0, 2, 1, 3)
    key_states = key_states.permute(0, 2, 1, 3)
    value_states = value_states.permute(0, 2, 1, 3)

    attention_mask = attention_mask[:, None]

    with sdpa_kernel(backends=[SDPBackend.MATH]):
        output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            enable_gqa=True,
            attn_mask=attention_mask,
        )
    return output


device = "cuda:1"
query_states = torch.randn((256, 800, 8, 64), device=device)
key_states = torch.randn((256, 800, 1, 64), device=device)
value_states = torch.randn((256, 800, 1, 64), device=device)
attention_mask = torch.ones((256, 800, 800), device=device, dtype=torch.bool)
attention_mask[:32, :, -100] = False


def test_func(n, fn, *args, **kwargs):
    for _ in range(n):
        fn(*args, **kwargs)


inputs = (
    attention_mask,
    256,
    64,
    query_states,
    key_states,
    value_states,
)

eager_out = eager_attention_forward(*inputs)
new_eager_out = new_eager_attention_forward(*inputs)
xformer_out = xformer_attention_forward(*inputs)
new_xformer_out = new_xformer_attention_forward(*inputs)

# output = test_func(
#     100,
#     eager_attention_forward,
#     attention_mask,
#     256,
#     64,
#     query_states,
#     key_states,
#     value_states,
# )
