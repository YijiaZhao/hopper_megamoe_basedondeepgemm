"""Offline OCP MXFP4 quantization for SM90 fused MegaMoE.

MXFP4 = packed E2M1 values with one non-negative E8M0 (power-of-two) scale
per 32 K-elements. Unlike the NVFP4 path, the kernel decodes values UNSCALED
(E2M1 -> E4M3 is exact) and multiplies the E8M0 coefficient in the WGMMA
promotion, so the packed layout carries the raw E8M0 bytes.
"""
import os
import torch


FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
FP4_MAX = 6.0

MXFP4_GROUP_SIZE = 32


def fp32_to_fp4_nibble(x: torch.Tensor) -> torch.Tensor:
    sign = (x < 0).to(torch.uint8) << 3
    mag = x.abs().clamp_max(FP4_MAX)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=x.device,
        dtype=torch.float32,
    )
    nibble_idx = torch.bucketize(mag.to(torch.float32), boundaries).to(torch.uint8)
    return sign | nibble_idx


def fp32_to_e8m0_ceil(x: torch.Tensor) -> torch.Tensor:
    """Encode positive scales as the smallest power of two >= x.

    Returns the raw E8M0 byte (biased exponent). Code 0 (2^-127) is never
    emitted: the CUDA decode clamps it to 2^-126, so the quantizer floors at
    code 1 to keep the reference and the kernel bit-identical. Code 255 (NaN)
    is likewise never emitted.
    """
    x = x.to(torch.float32).clamp(min=2.0 ** -126, max=2.0 ** 127)
    exp = torch.ceil(torch.log2(x))
    return (exp + 127).to(torch.int32).clamp(1, 254).to(torch.uint8)


def e8m0_to_fp32(code: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 bytes to float, matching the kernel's max(code, 1) clamp."""
    e = code.to(torch.int32).clamp(min=1)
    return torch.exp2((e - 127).to(torch.float32))


def quantize_to_mxfp4(weight: torch.Tensor, group_size: int = MXFP4_GROUP_SIZE):
    """Quantize real-valued weights to packed E2M1 FP4 plus per-32 E8M0 scale."""
    assert weight.is_floating_point() or weight.dtype == torch.float8_e4m3fn
    *outer_shape, K = weight.shape
    assert K % group_size == 0
    G = K // group_size
    w = weight.to(torch.float32).view(*outer_shape, G, group_size)
    max_abs = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    desired_scale = max_abs / FP4_MAX
    scale_e8m0 = fp32_to_e8m0_ceil(desired_scale.squeeze(-1))
    scale = e8m0_to_fp32(scale_e8m0).unsqueeze(-1)
    w_normalized = w / scale
    nibbles = fp32_to_fp4_nibble(w_normalized.clamp(-FP4_MAX, FP4_MAX))
    nibbles = nibbles.view(*outer_shape, K)
    # Marlin permutation: chunk of 8 K nibbles -> 4 bytes with
    #   byte b: low = K[b+4], high = K[b].
    assert K % 8 == 0
    chunks = nibbles.view(*outer_shape, K // 8, 8)
    packed = (chunks[..., 4:8] | (chunks[..., 0:4] << 4)).to(torch.uint8).view(*outer_shape, K // 2).contiguous()
    return packed, scale_e8m0.contiguous()


def mxfp4_scale_to_tile_major(
    scale_e8m0: torch.Tensor,
    block_n: int = 128,
    block_k: int = 128,
    group_size: int = MXFP4_GROUP_SIZE,
) -> torch.Tensor:
    """Repack row-major ``(E, N, K/32)`` E8M0 scales for SM90 tile-local loads.

    Output layout: ``(E, N/block_n, K/block_k, block_n, block_k/32)``.
    """
    assert scale_e8m0.dtype == torch.uint8
    assert scale_e8m0.dim() == 3
    assert block_k % group_size == 0
    groups_per_k_block = block_k // group_size
    E, N, G = scale_e8m0.shape
    assert N % block_n == 0
    assert G % groups_per_k_block == 0
    return (
        scale_e8m0.view(E, N // block_n, block_n, G // groups_per_k_block, groups_per_k_block)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )


def _mxfp4_marlin_to_prmt_groups(packed: torch.Tensor) -> torch.Tensor:
    """Group each eight Marlin nibbles into two direct PRMT selectors."""
    assert packed.shape[-1] % 4 == 0
    chunks = packed.view(*packed.shape[:-1], -1, 4)
    high = chunks >> 4
    low = chunks & 0x0F
    return torch.stack(
        (
            high[..., 0] | (high[..., 1] << 4),
            high[..., 2] | (high[..., 3] << 4),
            low[..., 0] | (low[..., 1] << 4),
            low[..., 2] | (low[..., 3] << 4),
        ),
        dim=-1,
    ).reshape_as(packed).contiguous()


def _mxfp4_prmt_groups_to_marlin(packed: torch.Tensor) -> torch.Tensor:
    """Invert ``_mxfp4_marlin_to_prmt_groups`` exactly."""
    assert packed.shape[-1] % 4 == 0
    chunks = packed.view(*packed.shape[:-1], -1, 4)
    k0, k1 = chunks[..., 0] & 0x0F, chunks[..., 0] >> 4
    k2, k3 = chunks[..., 1] & 0x0F, chunks[..., 1] >> 4
    k4, k5 = chunks[..., 2] & 0x0F, chunks[..., 2] >> 4
    k6, k7 = chunks[..., 3] & 0x0F, chunks[..., 3] >> 4
    return torch.stack(
        (k4 | (k0 << 4), k5 | (k1 << 4),
         k6 | (k2 << 4), k7 | (k3 << 4)),
        dim=-1,
    ).reshape_as(packed).contiguous()




def _mxfp4_rf_fragment_reorder_nibbles(order_fwd: bool):
    """Element order inside each 32-K group for the RF-decode fragment layout.

    Thread c (c = lane % 4) reads one u32 that must decode to elements
    {4c..4c+3, 4c+16..4c+19}. Returns the permutation on 32 nibble indices.
    """
    import itertools
    perm = []
    for c in range(4):
        perm += [4 * c + i for i in range(4)]
        perm += [4 * c + 16 + i for i in range(4)]
    if order_fwd:
        return perm
    inv = [0] * 32
    for dst, src_i in enumerate(perm):
        inv[src_i] = dst
    return inv


def _mxfp4_apply_rf_fragment_order(packed_tile: torch.Tensor, forward: bool) -> torch.Tensor:
    """Permute Marlin-packed rows (last dim = 64B per 128 K) for RF fragments.

    Operates at nibble level within every 16-byte (32-element) K32 chunk;
    lossless and self-inverse via ``forward=False``.
    """
    *outer, row_bytes = packed_tile.shape
    assert row_bytes % 16 == 0
    chunks = packed_tile.view(*outer, row_bytes // 16, 16)
    # Marlin inverse: byte b of an 8-elem group holds elements (b, b+4).
    grp = chunks.view(*outer, row_bytes // 16, 4, 4)  # 4 marlin groups of 4B
    high = (grp >> 4) & 0x0F
    low = grp & 0x0F
    nibbles = torch.cat([high, low], dim=-1).view(*outer, row_bytes // 16, 32)
    perm = torch.tensor(_mxfp4_rf_fragment_reorder_nibbles(forward),
                        dtype=torch.long, device=packed_tile.device)
    nibbles = nibbles[..., perm]
    regrp = nibbles.view(*outer, row_bytes // 16, 4, 8)
    repacked = (regrp[..., 4:8] | (regrp[..., 0:4] << 4)).to(torch.uint8)
    return repacked.view(*outer, row_bytes).contiguous()




def _mxfp4_word_transpose(packed_tile: torch.Tensor) -> torch.Tensor:
    """Transpose 4-byte words inside each 64B row: [k32 b][thread c] -> [c][b].

    Lets each RF-decode thread fetch all four of its K32 words with a single
    16-byte vector load. Self-inverse.
    """
    *outer, row_bytes = packed_tile.shape
    assert row_bytes % 64 == 0
    w = packed_tile.view(*outer, row_bytes // 64, 4, 4, 4)  # [.., row, b, c, byte]
    return w.transpose(-3, -2).reshape(*outer, row_bytes).contiguous()


def mxfp4_fuse_packed_with_scale_tile_major(
    packed: torch.Tensor,
    scale_tile_major: torch.Tensor,
    block_k: int = 128,
    use_prmt_groups: bool = False,
    use_rf_fragments: bool = False,
) -> torch.Tensor:
    """Pack each BK128 MXFP4 row as ``64B FP4 + 4B E8M0 scales + 12B padding``.

    The 80-byte row stride matches the NVFP4 layout so the TMA descriptor and
    SMEM staging paths are shared. The returned tensor keeps the public 3D
    weight shape ``(E, N, K/128*80)``.
    """
    assert packed.dtype == torch.uint8
    assert scale_tile_major.dtype == torch.uint8
    assert packed.dim() == 3
    assert scale_tile_major.dim() == 5
    E, N, K_half = packed.shape
    E_s, n_blocks, k_blocks, block_n, groups_per_k_block = scale_tile_major.shape
    fused_row_bytes = 64  # values only; scales live in the tile-major sf tensor
    scale_offset = block_k // 2
    assert E == E_s
    assert N == n_blocks * block_n
    assert K_half == k_blocks * (block_k // 2)
    assert groups_per_k_block == block_k // MXFP4_GROUP_SIZE
    packed_tile = (
        packed.view(E, n_blocks, block_n, k_blocks, block_k // 2)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    if use_rf_fragments:
        packed_tile = _mxfp4_apply_rf_fragment_order(packed_tile, forward=True)
    if use_prmt_groups:
        packed_tile = _mxfp4_marlin_to_prmt_groups(packed_tile)
    if use_rf_fragments:
        packed_tile = _mxfp4_word_transpose(packed_tile)
    fused = packed_tile.view(E, n_blocks, k_blocks, block_n, fused_row_bytes).clone()
    return (
        fused.permute(0, 1, 3, 2, 4)
        .reshape(E, N, k_blocks * fused_row_bytes)
        .contiguous()
    )


def dequantize_mxfp4_to_fp32(packed: torch.Tensor, scale_e8m0: torch.Tensor,
                             group_size: int = MXFP4_GROUP_SIZE,
                             fused_input: bool = True) -> torch.Tensor:
    """Exact reference dequant; accepts raw or fused/tile-major inputs.

    Values-only fused tensors share the raw (E, N, K/2) shape, so with a 5D
    tile-major scale the caller must say whether `packed` carries the fused
    (word-transposed grouped-PRMT RF) layout (default) or raw Marlin bytes.
    """
    if scale_e8m0.dim() == 5 and fused_input:
        E, n_blocks, k_blocks, block_n, groups_per_k_block = scale_e8m0.shape
        fused_row_bytes = 64
        fused_k = k_blocks * fused_row_bytes
        if packed.dim() == 3 and packed.shape == (E, n_blocks * block_n, fused_k):
            packed = (
                packed.view(E, n_blocks, block_n, k_blocks, fused_row_bytes)
                .permute(0, 1, 3, 2, 4)
                .permute(0, 1, 3, 2, 4)
                .reshape(E, n_blocks * block_n, k_blocks * 64)
                .contiguous()
            )
            # Fused MXFP4 rows carry word-transposed grouped-PRMT selectors
            # over the RF-fragment element order; invert all three.
            packed = _mxfp4_word_transpose(packed)
            if not (
                os.environ.get("DG_W4A8_INT", "0") != "0" and
                os.environ.get("DG_W4A8_INT_DIRECT_NIBBLE", "0") != "0"
            ):
                packed = _mxfp4_prmt_groups_to_marlin(packed)
            packed = _mxfp4_apply_rf_fragment_order(packed, forward=False)
        scale_e8m0 = (
            scale_e8m0.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(E, n_blocks * block_n, k_blocks * groups_per_k_block)
        )
    elif scale_e8m0.dim() == 5:
        E, n_blocks, k_blocks, block_n, groups_per_k_block = scale_e8m0.shape
        scale_e8m0 = (
            scale_e8m0.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(E, n_blocks * block_n, k_blocks * groups_per_k_block)
        )
    *outer_shape, K_half = packed.shape
    K = K_half * 2
    G = K // group_size
    pck = packed.view(*outer_shape, K // 8, 4)
    low = pck & 0x0F
    high = (pck >> 4) & 0x0F
    nibbles = torch.cat([high, low], dim=-1).view(*outer_shape, K)
    sign_bit = (nibbles >> 3) & 0x1
    mag_idx = (nibbles & 0x7).to(torch.long)
    fp4_values = FP4_VALUES.to(packed.device)
    mag = fp4_values[mag_idx]
    values = torch.where(sign_bit.bool(), -mag, mag)
    scale = e8m0_to_fp32(scale_e8m0)
    scale_expanded = scale.unsqueeze(-1).expand(*outer_shape, G, group_size).reshape(*outer_shape, K)
    return values * scale_expanded


if __name__ == "__main__":
    torch.manual_seed(0)
    # N/K chosen so the shape policy in dequantize_mxfp4_to_fp32 selects the
    # plain Marlin layout (H <= 2I side).
    E, N, K = 4, 1024, 512
    w_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16) * 0.3
    w_fp32_ref = w_bf16.to(torch.float32)

    packed, scale_e8m0 = quantize_to_mxfp4(w_bf16)
    print(f"packed shape: {packed.shape}, scale shape: {scale_e8m0.shape}")

    w_recovered = dequantize_mxfp4_to_fp32(packed, scale_e8m0)
    err = (w_recovered - w_fp32_ref).abs()
    print(f"Element error: max_abs={err.max():.4f}  mean_abs={err.mean():.4f}")

    # Round-trip through tile-major + fused (grouped-PRMT) layouts must be
    # lossless; the production path always uses grouped-PRMT selectors.
    for block_n in (128, 256):
        sc_tm = mxfp4_scale_to_tile_major(scale_e8m0, block_n=block_n)
        fused = mxfp4_fuse_packed_with_scale_tile_major(
            packed, sc_tm, use_prmt_groups=True, use_rf_fragments=True)
        w_rt = dequantize_mxfp4_to_fp32(fused, sc_tm)
        assert torch.equal(w_rt, w_recovered), block_n

    # RF fragment order is a lossless self-inverting nibble permutation.
    probe = torch.arange(4 * 64, dtype=torch.uint8).view(1, 4, 64) % 251
    rf = _mxfp4_apply_rf_fragment_order(probe, forward=True)
    back = _mxfp4_apply_rf_fragment_order(rf, forward=False)
    assert torch.equal(back, probe)
    assert not torch.equal(rf, probe)
    print("OK")
