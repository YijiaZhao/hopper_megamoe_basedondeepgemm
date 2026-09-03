"""Offline MXFP4 (E2M1 + per-32 E8M0 scale) quantization for the SM90 fused MegaMoE.

Kernel-side dequant strategy: the E8M0 scale is a pure power of two, so it can be
folded exactly into the E4M3 exponent field. The host normalises every weight row
(output channel) by a per-row reference exponent ``e_ref = e_max_row - 6`` and
stores, per 32-K group, the relative LUT index ``s = e - e_ref + 9`` (clamped to
``[0, 15]``). The kernel looks up ``E2M1 magnitude * 2^(s - 9)`` in a 16-row FP8
table (bit-exact, down to the E4M3 subnormal 2^-9), runs the FP8 WGMMA, and
multiplies the accumulator by the per-row factor ``2^e_ref`` in the epilogue.
Groups more than 2^14 below the row maximum flush to zero (index 0).
"""
import torch

from .quantization_nvfp4 import FP4_VALUES, FP4_MAX, fp32_to_fp4_nibble

MXFP4_GROUP_SIZE = 32
E8M0_BIAS = 127
# Must match deep_gemm/include/deep_gemm/quantization/nvfp4_dequant.cuh.
MXFP4_LUT_EXP_BIAS = 9
MXFP4_REF_EXP_MARGIN = 6
MXFP4_LUT_MAX_INDEX = 15
MXFP4_FUSED_ROW_BYTES = 80


def e8m0_to_fp32(scale_e8m0: torch.Tensor) -> torch.Tensor:
    return torch.exp2((scale_e8m0.to(torch.int32) - E8M0_BIAS).to(torch.float32))


def quantize_to_mxfp4(weight: torch.Tensor, group_size: int = MXFP4_GROUP_SIZE):
    """Quantize real-valued weights to packed E2M1 FP4 plus per-32 E8M0 scale.

    Returns ``(packed_uint8[..., K/2], scale_e8m0_uint8[..., K/32])``. The packed
    layout is the same Marlin-style nibble order as the NVFP4 path. The scale is
    the smallest power of two such that ``amax / scale <= 6`` (no saturation).
    """
    assert weight.is_floating_point()
    *outer_shape, K = weight.shape
    assert K % group_size == 0
    G = K // group_size
    w = weight.to(torch.float32).view(*outer_shape, G, group_size)
    max_abs = w.abs().amax(dim=-1, keepdim=True)
    # ceil(log2(amax / 6)); all-zero groups get the minimum exponent.
    exp = torch.ceil(torch.log2(max_abs.clamp(min=1e-38) / FP4_MAX))
    exp = torch.where(max_abs > 0, exp, torch.full_like(exp, -E8M0_BIAS))
    exp = exp.clamp(-E8M0_BIAS, 127)
    scale = torch.exp2(exp)
    nibbles = fp32_to_fp4_nibble((w / scale).clamp(-FP4_MAX, FP4_MAX)).view(*outer_shape, K)
    assert K % 8 == 0
    chunks = nibbles.view(*outer_shape, K // 8, 8)
    packed = (chunks[..., 4:8] | (chunks[..., 0:4] << 4)).to(torch.uint8).view(*outer_shape, K // 2).contiguous()
    scale_e8m0 = (exp.squeeze(-1).to(torch.int32) + E8M0_BIAS).to(torch.uint8).contiguous()
    return packed, scale_e8m0


def mxfp4_row_reference_exponent(scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Per-row reference exponent (unbiased int32) ``e_max_row - margin``."""
    e = scale_e8m0.to(torch.int32) - E8M0_BIAS
    return e.amax(dim=-1) - MXFP4_REF_EXP_MARGIN


def mxfp4_scale_to_relative_index(scale_e8m0: torch.Tensor, e_ref: torch.Tensor) -> torch.Tensor:
    """E8M0 scale ``(..., N, G)`` -> kernel LUT index ``(..., N, G)`` in ``[0, 15]``."""
    e = scale_e8m0.to(torch.int32) - E8M0_BIAS
    idx = e - e_ref.unsqueeze(-1) + MXFP4_LUT_EXP_BIAS
    assert int(idx.max()) <= MXFP4_LUT_MAX_INDEX
    return idx.clamp(0, MXFP4_LUT_MAX_INDEX).to(torch.uint8)


def mxfp4_effective_scale_fp32(scale_e8m0: torch.Tensor, e_ref: torch.Tensor) -> torch.Tensor:
    """What the kernel actually applies: ``2^e`` for in-range groups, 0 when flushed."""
    e = scale_e8m0.to(torch.int32) - E8M0_BIAS
    rel = e - e_ref.unsqueeze(-1) + MXFP4_LUT_EXP_BIAS
    return torch.where(rel >= 1, e8m0_to_fp32(scale_e8m0), torch.zeros((), dtype=torch.float32, device=scale_e8m0.device))


def mxfp4_scale_to_tile_major(
    scale_idx: torch.Tensor,
    block_n: int = 256,
    block_k: int = 128,
    group_size: int = MXFP4_GROUP_SIZE,
) -> torch.Tensor:
    """Row-major ``(E, N, K/32)`` -> ``(E, N/block_n, K/block_k, block_n, block_k/32)``."""
    assert scale_idx.dtype == torch.uint8 and scale_idx.dim() == 3
    groups_per_k_block = block_k // group_size
    E, N, G = scale_idx.shape
    assert N % block_n == 0 and G % groups_per_k_block == 0
    return (
        scale_idx.view(E, N // block_n, block_n, G // groups_per_k_block, groups_per_k_block)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )


def mxfp4_fuse_packed_with_scale_tile_major(
    packed: torch.Tensor,
    scale_tile_major: torch.Tensor,
    block_k: int = 128,
) -> torch.Tensor:
    """Pack each BK128 row as ``64B FP4 + 4B relative E8M0 index + 12B zero``.

    Same 80-byte row stride as the NVFP4 fused layout so the TMA descriptors and
    shared-memory staging are unchanged; only the scale bytes differ.
    """
    assert packed.dtype == torch.uint8 and scale_tile_major.dtype == torch.uint8
    assert packed.dim() == 3 and scale_tile_major.dim() == 5
    E, N, K_half = packed.shape
    E_s, n_blocks, k_blocks, block_n, groups_per_k_block = scale_tile_major.shape
    assert E == E_s and N == n_blocks * block_n and K_half == k_blocks * (block_k // 2)
    assert groups_per_k_block == block_k // MXFP4_GROUP_SIZE
    scale_offset = block_k // 2
    packed_tile = (
        packed.view(E, n_blocks, block_n, k_blocks, block_k // 2)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    fused = torch.zeros(
        (E, n_blocks, k_blocks, block_n, MXFP4_FUSED_ROW_BYTES),
        dtype=torch.uint8, device=packed.device,
    )
    fused[..., :scale_offset] = packed_tile
    fused[..., scale_offset:scale_offset + groups_per_k_block] = scale_tile_major
    return (
        fused.permute(0, 1, 3, 2, 4)
        .reshape(E, N, k_blocks * MXFP4_FUSED_ROW_BYTES)
        .contiguous()
    )


def unpack_fp4_values(packed: torch.Tensor) -> torch.Tensor:
    """Marlin-packed ``(..., K/2)`` uint8 -> signed E2M1 values ``(..., K)`` fp32."""
    *outer_shape, K_half = packed.shape
    K = K_half * 2
    pck = packed.view(*outer_shape, K // 8, 4)
    low = pck & 0x0F
    high = (pck >> 4) & 0x0F
    nibbles = torch.cat([high, low], dim=-1).view(*outer_shape, K)
    mag = FP4_VALUES.to(packed.device)[(nibbles & 0x7).long()]
    return torch.where(((nibbles >> 3) & 0x1).bool(), -mag, mag)


def dequantize_mxfp4_to_fp32(packed: torch.Tensor, scale_e8m0: torch.Tensor,
                             group_size: int = MXFP4_GROUP_SIZE) -> torch.Tensor:
    """Exact MXFP4 dequant of row-major ``(packed, scale)``."""
    values = unpack_fp4_values(packed)
    *outer_shape, K = values.shape
    scale = e8m0_to_fp32(scale_e8m0)
    return values * scale.unsqueeze(-1).expand(*outer_shape, K // group_size, group_size).reshape(*outer_shape, K)


def dequantize_mxfp4_kernel_exact_fp32(packed: torch.Tensor, scale_e8m0: torch.Tensor,
                                       e_ref: torch.Tensor,
                                       group_size: int = MXFP4_GROUP_SIZE) -> torch.Tensor:
    """Dequant reproducing the kernel's flush-to-zero of out-of-range groups."""
    values = unpack_fp4_values(packed)
    *outer_shape, K = values.shape
    scale = mxfp4_effective_scale_fp32(scale_e8m0, e_ref)
    return values * scale.unsqueeze(-1).expand(*outer_shape, K // group_size, group_size).reshape(*outer_shape, K)
