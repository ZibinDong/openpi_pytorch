import math

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# from xformers.ops import memory_efficient_attention


def find_next_divisible_by_8_numpy(n: np.ndarray) -> np.ndarray:
    """
    Finds the smallest integers greater than each element in a NumPy array 'n'
    that are divisible by 8. Assumes non-negative integers.

    Args:
        n: A NumPy array of integers.

    Returns:
        A NumPy array containing the smallest integers greater than each input element
        that are divisible by 8.
    """
    remainder = n % 8
    # Calculate the amount to add: 0 if already divisible, otherwise 8 - remainder
    # np.where is efficient for conditional operations on arrays
    amount_to_add = np.where(remainder == 0, 8, 8 - remainder)
    return n + amount_to_add


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(
        0.0, 1.0, dimension // 2, dtype=torch.float32, device=device
    )
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.rand((bsize,), device=device).pow(1 / alpha)
    gamma2 = torch.rand((bsize,), device=device).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def eager_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
):
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


# def xformer_attention_forward(query_states, key_states, value_states, attention_mask):
#     bsize, seq_len, num_att_heads, head_dim = query_states.shape
#     num_key_value_heads = key_states.shape[2]
#     num_key_value_groups = num_att_heads // num_key_value_heads

#     query_states = einops.rearrange(
#         query_states, "b l (h g) d -> b l h g d", g=num_key_value_groups
#     )
#     key_states = einops.repeat(
#         key_states, "b l h d -> b l h g d", g=num_key_value_groups
#     )
#     value_states = einops.repeat(
#         value_states, "b l h d -> b l h g d", g=num_key_value_groups
#     )
#     aligned_attention_mask = torch.zeros(
#         (bsize, seq_len, find_next_divisible_by_8_numpy(seq_len).item()),
#         dtype=query_states.dtype,
#         device=attention_mask.device,
#     )
#     big_neg = -2.3819763e38
#     aligned_attention_mask[:, :, :seq_len] = ~attention_mask * big_neg
#     aligned_attention_mask = einops.repeat(
#         aligned_attention_mask,
#         "b l s -> b h g l s",
#         h=num_key_value_heads,
#         g=num_key_value_groups,
#     )[:, :, :, :, :seq_len]

#     att_output = memory_efficient_attention(
#         query=query_states,
#         key=key_states,
#         value=value_states,
#         attn_bias=aligned_attention_mask,
#     )
#     att_output = att_output.reshape(bsize, seq_len, -1)

#     return att_output


@torch.jit.script
def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    max_wavelength: float = 10_000.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    original_dtype = x.dtype
    d = x.shape[-1]
    d_half = d // 2
    device = x.device

    # Cast input to compute_dtype for all internal operations
    x_casted = x.to(dtype)
    positions_casted = positions.to(dtype)

    freq_exponents = (2.0 / d) * torch.arange(d_half, dtype=dtype, device=device)
    timescale = max_wavelength**freq_exponents
    radians = torch.einsum("bl,h->blh", positions_casted, 1.0 / timescale)

    radians = radians[..., None, :]  # [B, L, 1, D_half]

    sin = torch.sin(radians)
    cos = torch.cos(radians)

    x1, x2 = x_casted.split(d_half, dim=-1)

    res = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return res.to(original_dtype)
