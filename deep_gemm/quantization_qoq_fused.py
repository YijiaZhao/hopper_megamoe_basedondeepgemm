"""Canonical QoQ (QServe-style) W4A8 quantization for the SM90 fused MegaMoE.

Weights: uint4 code per element, per (row, K128 group) zero point ``z`` (0..15)
and integer scale ``s2`` (1..15), per row BF16 ``s1``:
    w = (code - z) * s2 * s1
Activations: symmetric INT8 with one FP32 scale per token (whole row).

Kernel side: SHIFTXOR decode emits ``(code - z)`` int8, IGMMA accumulates in
int32, the per-K128 promote multiplies by ``s2[row] * s_act[token]`` and the
epilogue applies ``s1[row]`` through the same [E, N] row-scale path as MXFP4.
"""
import torch

QOQ_GROUP_SIZE = 128
QOQ_FUSED_ROW_BYTES = 80


def quantize_to_qoq(weight: torch.Tensor, group_size: int = QOQ_GROUP_SIZE):
    """Return ``(packed_uint8[E,N,K/2], s2_uint8[E,N,K/128], z_uint8[E,N,K/128], s1_fp32[E,N])``."""
    assert weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
    E, N, K = weight.shape
    assert K % group_size == 0
    w = weight.float()
    s1 = (w.abs().amax(dim=-1, keepdim=True) / 120.0).clamp_min(1e-30)
    s1 = s1.to(torch.bfloat16).to(torch.float32)          # bf16-rounded, exact in the kernel
    w8 = (w / s1).round().clamp(-112, 112)
    G = K // group_size
    w8g = w8.view(E, N, G, group_size)
    mn = w8g.amin(dim=-1, keepdim=True)
    mx = w8g.amax(dim=-1, keepdim=True)
    s2 = ((mx - mn) / 15.0).ceil().clamp(1, 15)
    z = (-mn / s2).round().clamp(0, 15)
    code = (w8g / s2 + z).round().clamp(0, 15).view(E, N, K)
    nib = code.to(torch.uint8).view(E, N, K // 8, 8)
    # Marlin nibble order (same as the FP4 paths): byte b = K[b] << 4 | K[b+4].
    packed = (nib[..., 4:8] | (nib[..., 0:4] << 4)).view(E, N, K // 2).contiguous()
    return (packed, s2.squeeze(-1).to(torch.uint8).contiguous(),
            z.squeeze(-1).to(torch.uint8).contiguous(), s1.squeeze(-1).contiguous())


def qoq_meta_to_tile_major(s2: torch.Tensor, z: torch.Tensor, block_n: int = 256) -> torch.Tensor:
    """``(E, N, K/128)`` x2 -> ``(E, N/block_n, K/128, block_n, 2)`` bytes ``[s2, z]``."""
    E, N, G = s2.shape
    meta = torch.stack((s2, z), dim=-1)                    # (E, N, G, 2)
    return meta.view(E, N // block_n, block_n, G, 2).permute(0, 1, 3, 2, 4).contiguous()


def qoq_fuse_packed_with_meta_tile_major(packed: torch.Tensor, meta_tm: torch.Tensor,
                                         block_k: int = QOQ_GROUP_SIZE) -> torch.Tensor:
    """Pack each BK128 row as ``64B codes + [s2, z] + 14B zero`` (80-byte rows)."""
    E, N, K_half = packed.shape
    E_s, n_blocks, k_blocks, block_n, two = meta_tm.shape
    assert E == E_s and N == n_blocks * block_n and K_half == k_blocks * (block_k // 2) and two == 2
    packed_tile = packed.view(E, n_blocks, block_n, k_blocks, block_k // 2).permute(0, 1, 3, 2, 4).contiguous()
    fused = torch.zeros((E, n_blocks, k_blocks, block_n, QOQ_FUSED_ROW_BYTES), dtype=torch.uint8, device=packed.device)
    fused[..., :block_k // 2] = packed_tile
    fused[..., block_k // 2:block_k // 2 + 2] = meta_tm
    return fused.permute(0, 1, 3, 2, 4).reshape(E, N, k_blocks * QOQ_FUSED_ROW_BYTES).contiguous()


def dequantize_qoq_to_fp32(packed: torch.Tensor, s2: torch.Tensor, z: torch.Tensor, s1: torch.Tensor,
                           group_size: int = QOQ_GROUP_SIZE) -> torch.Tensor:
    E, N, K_half = packed.shape
    K = K_half * 2
    pck = packed.view(E, N, K // 8, 4)
    codes = torch.cat([(pck >> 4) & 0x0F, pck & 0x0F], dim=-1).view(E, N, K // group_size, group_size).float()
    w = (codes - z.float().unsqueeze(-1)) * s2.float().unsqueeze(-1)
    return (w.view(E, N, K) * s1.unsqueeze(-1)).contiguous()


def per_token_cast_to_int8(x: torch.Tensor, gran_k: int = 128):
    """Symmetric per-token INT8. Returns ``(int8 bytes viewed as float8_e4m3fn, sf[M, K/gran_k])``.

    The fused kernel streams activation bytes through its FP8 buffers and reads
    one scale per K128 group; QoQ uses a single per-token scale, so it is
    repeated across the K128 slots.
    """
    M, K = x.shape
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    sf = amax / 127.0
    q = (xf / sf).round().clamp(-127, 127).to(torch.int8)
    x_bytes = q.view(torch.float8_e4m3fn)
    return x_bytes.contiguous(), sf.expand(M, K // gran_k).contiguous()


def int8_bytes_to_float(x_bytes: torch.Tensor) -> torch.Tensor:
    return x_bytes.view(torch.int8).float()
