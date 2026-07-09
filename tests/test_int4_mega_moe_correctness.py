"""Single-rank correctness gate for the SM90 MXFP4 MegaMoE kernel."""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.quantization_mxfp4 import (
    FP4_VALUES,
    dequantize_mxfp4_to_fp32,
    mxfp4_scale_to_tile_major,
    quantize_to_mxfp4,
    e8m0_to_fp32,
)
from deep_gemm.testing import get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist



# ===== W4A8-int helpers =====
from deep_gemm.quantization_mxfp4 import mxfp4_fuse_packed_with_scale_tile_major
from deep_gemm.mega import _interleave_l1_weights as _il1

INT4_GROUP = 128

def quantize_to_int4(weight, group_size=INT4_GROUP):
    *outer, K = weight.shape
    G = K // group_size
    w = weight.float().view(*outer, G, group_size)
    scale = (w.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-30)
    q = (w / scale).round().clamp(-7, 7).to(torch.int8).view(*outer, K)
    scale = scale.squeeze(-1).to(torch.float32)
    nib = (q & 0x0F).to(torch.uint8).view(*outer, K // 8, 8)
    packed = (nib[..., 4:8] | (nib[..., 0:4] << 4)).view(*outer, K // 2).contiguous()
    return packed, scale

def dequant_int4(packed, scale, N, K, group_size=INT4_GROUP):
    E = packed.shape[0]
    b = packed.view(E, N, K // 2)
    lo = (b & 0x0F); hi = (b >> 4)
    def s4(x):
        x = x.to(torch.int16); return torch.where(x >= 8, x - 16, x)
    q = torch.empty(E, N, K, dtype=torch.int16, device=packed.device)
    qv = q.view(E, N, K // 8, 8)
    pv = b.view(E, N, K // 8, 4)
    qv[..., 0:4] = s4(pv >> 4); qv[..., 4:8] = s4(pv & 0x0F)
    return (q.float().view(E, N, K // group_size, group_size)
            * scale.view(E, N, K // group_size, 1)).view(E, N, K)

def _int4_scale_tm(scale_fp32, block_n=128):
    E, N, Kb = scale_fp32.shape
    b = scale_fp32.contiguous().view(torch.uint8).view(E, N, Kb, 4)
    return (b.view(E, N // block_n, block_n, Kb, 4).permute(0, 1, 3, 2, 4).contiguous())

def transform_int4(packed, scale, is_l1, block_n=128):
    if is_l1:
        packed, scale = _il1((packed, scale))
    sc_tm = _int4_scale_tm(scale, block_n=block_n)
    pk = mxfp4_fuse_packed_with_scale_tile_major(
        packed.contiguous(), sc_tm, block_k=128,
        use_prmt_groups=True, use_rf_fragments=True)
    return pk, sc_tm

def per_token_int8(x_bf):
    x = x_bf.float()
    s = (x.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
    q = (x / s).round().clamp(-127, 127).to(torch.int8)
    return q, s.squeeze(-1)
# ===== end helpers =====

def _interleave_l1_n(tensor: torch.Tensor, gran: int = 8) -> torch.Tensor:
    groups, n_cols, *rest = tensor.shape
    half = n_cols // 2
    gate = tensor[:, :half].reshape(groups, half // gran, gran, *rest)
    up = tensor[:, half:].reshape(groups, half // gran, gran, *rest)
    return torch.empty_like(tensor).copy_(torch.stack([gate, up], dim=2).reshape(groups, n_cols, *rest))


def _pack_mxfp4_marlin(nibbles: torch.Tensor) -> torch.Tensor:
    *outer_shape, k = nibbles.shape
    assert k % 8 == 0
    chunks = nibbles.view(*outer_shape, k // 8, 8).to(torch.int16)
    return (chunks[..., 4:8] | (chunks[..., 0:4] << 4)).to(torch.uint8).view(*outer_shape, k // 2).contiguous()


_CUDA_DEQUANT_EXT = None


def _load_cuda_dequant_ext():
    global _CUDA_DEQUANT_EXT
    if _CUDA_DEQUANT_EXT is not None:
        return _CUDA_DEQUANT_EXT

    cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <deep_gemm/quantization/mxfp4_dequant.cuh>

// MXFP4 decode is scale-free (constant LUT). Exhaustively decode every
// nibble with both selector layouts, plus every E8M0 byte to float.
__global__ void mxfp4_decode_bytes_kernel(uint8_t* out, float* coeff_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 16 * 2) {
        const uint32_t nibble = static_cast<uint32_t>(idx / 2);
        const bool use_prmt = (idx & 1) != 0;
        const uint32_t q = nibble * 0x11111111u;
        const uint2 deq = use_prmt ?
            deep_gemm::mxfp4::decode_mxfp4_prmt_groups_to_fp8_pair(q) :
            deep_gemm::mxfp4::decode_mxfp4_to_fp8_pair(q);
        const uint8_t* hi = reinterpret_cast<const uint8_t*>(&deq.x);
        const uint8_t* lo = reinterpret_cast<const uint8_t*>(&deq.y);
        out[idx * 2] = hi[0];
        out[idx * 2 + 1] = lo[0];
    }
    if (idx < 256)
        coeff_out[idx] = deep_gemm::mxfp4::e8m0_to_float(static_cast<uint32_t>(idx));
}

std::vector<torch::Tensor> mxfp4_decode_bytes_cuda() {
    auto out = torch::empty({16, 2, 2}, torch::device(torch::kCUDA).dtype(torch::kUInt8));
    auto coeff = torch::empty({256}, torch::device(torch::kCUDA).dtype(torch::kFloat));
    mxfp4_decode_bytes_kernel<<<1, 256>>>(out.data_ptr<uint8_t>(), coeff.data_ptr<float>());
    return {out, coeff};
}
"""
    cpp_src = "std::vector<torch::Tensor> mxfp4_decode_bytes_cuda();"
    _CUDA_DEQUANT_EXT = load_inline(
        name="deepgemm_mxfp4_dequant_lut_test",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["mxfp4_decode_bytes_cuda"],
        extra_include_paths=[os.path.join(REPO_ROOT, "deep_gemm", "include")],
        extra_cuda_cflags=["--expt-relaxed-constexpr"],
        verbose=False,
    )
    return _CUDA_DEQUANT_EXT


def _run_cuda_dequant_lut_unit_test() -> None:
    ext = _load_cuda_dequant_ext()
    got_bytes, got_coeff = ext.mxfp4_decode_bytes_cuda()
    got_bytes = got_bytes.cpu()
    got_coeff = got_coeff.cpu()

    nibbles = torch.arange(16, dtype=torch.uint8, device="cpu")
    mag = FP4_VALUES[(nibbles & 0x7).long()]
    signed = torch.where(((nibbles >> 3) & 0x1).bool(), -mag, mag)
    expected = signed.to(torch.float8_e4m3fn).view(torch.uint8)
    # Preserve the negative-zero sign byte (torch folds -0.0 to +0.0).
    expected = torch.where(
        ((expected & 0x7F) == 0) & (((nibbles >> 3) & 0x1).bool()),
        expected | 0x80,
        expected,
    )
    expected = expected.view(16, 1, 1).expand(16, 2, 2).contiguous()
    torch.testing.assert_close(got_bytes, expected, rtol=0, atol=0)

    codes = torch.arange(256, dtype=torch.uint8)
    expected_coeff = e8m0_to_fp32(codes)
    # Code 255 is NaN per OCP; the kernel maps it to 2^128-ish garbage only if
    # fed, and prepack rejects it — compare finite range only.
    torch.testing.assert_close(got_coeff[:255], expected_coeff[:255], rtol=0, atol=0)
    print("MXFP4 CUDA decode unit test: PASS", flush=True)


def _run_dequant_unit_test() -> None:
    assert deep_gemm.get_nvfp4_mega_moe_sm90_block_n(2048) == 128
    assert deep_gemm.get_nvfp4_mega_moe_sm90_block_n(3072) == 128
    assert deep_gemm.get_nvfp4_mega_moe_sm90_weight_layout(4096, 2048) == "marlin"
    assert deep_gemm.get_nvfp4_mega_moe_sm90_weight_layout(7168, 3072) == "prmt_groups"

    # E8M0 sample codes: clamp floor (0 -> 2^-126), tiny, unit (127), large.
    scales = torch.tensor([0x00, 0x01, 0x40, 0x7F, 0x80, 0xC0, 0xFE], dtype=torch.uint8)
    nibbles = torch.arange(16, dtype=torch.uint8).repeat(2).view(1, 1, 32).expand(scales.numel(), 1, 32).clone()
    packed = _pack_mxfp4_marlin(nibbles)
    got = dequantize_mxfp4_to_fp32(packed, scales.view(-1, 1, 1), group_size=32)

    mag = FP4_VALUES.to(nibbles.device)[(nibbles & 0x7).long()]
    signed = torch.where(((nibbles >> 3) & 0x1).bool(), -mag, mag)
    expected = signed * e8m0_to_fp32(scales).view(-1, 1, 1)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)

    tile_weight = torch.randn(2, 256, 256, dtype=torch.bfloat16) * 0.1
    tile_packed, tile_scale = quantize_to_mxfp4(tile_weight, group_size=32)
    tile_scale_tm = mxfp4_scale_to_tile_major(tile_scale)
    torch.testing.assert_close(
        dequantize_mxfp4_to_fp32(tile_packed, tile_scale, group_size=32),
        dequantize_mxfp4_to_fp32(tile_packed, tile_scale_tm, group_size=32,
                                 fused_input=False),
        rtol=0,
        atol=0,
    )

    l1_weight = torch.randn(2, 512, 256, dtype=torch.bfloat16) * 0.1
    l2_weight = torch.randn(2, 256, 256, dtype=torch.bfloat16) * 0.1
    l1_packed, l1_scale = quantize_to_mxfp4(l1_weight, group_size=32)
    l2_packed, l2_scale = quantize_to_mxfp4(l2_weight, group_size=32)
    transformed_l1, transformed_l2 = deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90(
        (l1_packed, l1_scale), (l2_packed, l2_scale),
    )
    torch.testing.assert_close(
        dequantize_mxfp4_to_fp32(transformed_l1[0], transformed_l1[1], group_size=32),
        _interleave_l1_n(dequantize_mxfp4_to_fp32(l1_packed, l1_scale, group_size=32)),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        dequantize_mxfp4_to_fp32(transformed_l2[0], transformed_l2[1], group_size=32),
        dequantize_mxfp4_to_fp32(l2_packed, l2_scale, group_size=32),
        rtol=0,
        atol=0,
    )

    transformed_l1_bn256, transformed_l2_bn256 = deep_gemm.transform_mxfp4_weights_for_mega_moe_sm90(
        (l1_packed, l1_scale), (l2_packed, l2_scale), block_n=256,
    )
    torch.testing.assert_close(
        dequantize_mxfp4_to_fp32(transformed_l1_bn256[0], transformed_l1_bn256[1], group_size=32),
        _interleave_l1_n(dequantize_mxfp4_to_fp32(l1_packed, l1_scale, group_size=32)),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        dequantize_mxfp4_to_fp32(transformed_l2_bn256[0], transformed_l2_bn256[1], group_size=32),
        dequantize_mxfp4_to_fp32(l2_packed, l2_scale, group_size=32),
        rtol=0,
        atol=0,
    )
    print('MXFP4 dequant unit test: PASS', flush=True)




def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _prepare_model(args: argparse.Namespace, weight_scale: float,
                   num_local_experts: int, rank_idx: int):
    hidden = args.hidden
    intermediate_hidden = args.intermediate_hidden
    torch.manual_seed(
        args.weight_seed + int(weight_scale * 1000000) + rank_idx * 1000003)
    l1_bf = torch.randn(
        (num_local_experts, 2 * intermediate_hidden, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    ) * weight_scale
    l1_packed, l1_scale = quantize_to_int4(l1_bf)
    l1_dequant = dequant_int4(l1_packed, l1_scale, 2 * intermediate_hidden, hidden)
    del l1_bf

    l2_bf = torch.randn(
        (num_local_experts, hidden, intermediate_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    ) * weight_scale
    l2_packed, l2_scale = quantize_to_int4(l2_bf)
    l2_dequant = dequant_int4(l2_packed, l2_scale, hidden, intermediate_hidden)
    del l2_bf

    transformed_l1 = transform_int4(l1_packed, l1_scale, is_l1=True, block_n=args.block_n or 128)
    transformed_l2 = transform_int4(l2_packed, l2_scale, is_l1=False, block_n=args.block_n or 128)
    del l1_packed, l1_scale, l2_packed, l2_scale
    torch.cuda.empty_cache()
    return l1_dequant, l2_dequant, transformed_l1, transformed_l2


def _run_case(args: argparse.Namespace, m_tokens: int, weight_scale: float,
              global_scale_mode: str, routing_seed: int,
              rank_idx: int, group: dist.ProcessGroup, buffer,
              l1_dequant: torch.Tensor, l2_dequant: torch.Tensor,
              transformed_l1, transformed_l2):
    hidden = args.hidden
    intermediate_hidden = args.intermediate_hidden
    num_experts = args.num_experts
    num_topk = args.num_topk
    num_ranks = dist.get_world_size(group)
    num_local_experts = num_experts // num_ranks
    num_max_tokens_per_rank = args.num_max_tokens_per_rank or max(32, max(args.batches))
    selected_block_n = (
        args.block_n
        if args.block_n is not None
        else deep_gemm.get_nvfp4_mega_moe_sm90_block_n(intermediate_hidden)
    )

    if rank_idx == 0:
        print(
            f"=== MXFP4 correctness M={m_tokens}, "
            f"NE={num_experts}, NL={num_local_experts}, NK={num_topk}, "
            f"NMT={num_max_tokens_per_rank}, weight_scale={weight_scale:g}, "
            f"global_scale_mode={global_scale_mode}, routing_seed={routing_seed}, "
            f"block_n={selected_block_n}, reference=exact_int4 ===",
            flush=True,
        )

    torch.manual_seed(routing_seed + m_tokens * 1009 + rank_idx * 1000003)
    x_bf = torch.randn((m_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = torch.randn((m_tokens, num_experts), dtype=torch.float, device="cuda")
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1)
    x_i8, x_ts = per_token_int8(x_bf)
    x_fp8 = x_i8.view(torch.float8_e4m3fn)
    x_sf = x_ts.unsqueeze(1).repeat(1, hidden // 128).contiguous()

    l1_global_scales = None
    l2_global_scales = None
    if global_scale_mode == "expert":
        l1_global_scales = torch.linspace(
            0.73, 1.37, num_local_experts, dtype=torch.float32, device="cuda"
        )
        l2_global_scales = torch.linspace(
            1.31, 0.67, num_local_experts, dtype=torch.float32, device="cuda"
        )
    elif global_scale_mode != "none":
        raise ValueError(f"unsupported global_scale_mode={global_scale_mode}")

    cumulative_stats = torch.zeros(num_local_experts, dtype=torch.int, device="cuda")
    buffer.x[:m_tokens].view(torch.uint8).copy_(x_i8.view(torch.uint8))
    buffer.x_sf[:m_tokens].copy_(x_sf)
    buffer.topk_idx[:m_tokens].copy_(topk_idx.to(torch.int32))
    buffer.topk_weights[:m_tokens].copy_(topk_weights.to(torch.float32))

    y_kernel = torch.zeros((m_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    deep_gemm.mxfp4_mega_moe(
        y_kernel,
        transformed_l1,
        transformed_l2,
        buffer,
        cumulative_local_expert_recv_stats=cumulative_stats,
        l1_global_scales=l1_global_scales,
        l2_global_scales=l2_global_scales,
        recipe=None if args.block_n is None else (128, args.block_n, 128),
        activation="swiglu",
        activation_clamp=args.activation_clamp,
        fast_math=bool(args.fast_math),
    )
    torch.cuda.synchronize()
    dist.barrier(group=group)

    x_ref_local = (x_i8.float() * x_ts.float().unsqueeze(1))

    num_global_tokens = m_tokens * num_ranks
    x_ref = torch.empty(
        (num_global_tokens, hidden), device="cuda", dtype=torch.float32)
    topk_idx_global = torch.empty(
        (num_global_tokens, num_topk), device="cuda", dtype=topk_idx.dtype)
    topk_weights_global = torch.empty(
        (num_global_tokens, num_topk), device="cuda", dtype=topk_weights.dtype)
    dist.all_gather_into_tensor(x_ref, x_ref_local.contiguous(), group=group)
    dist.all_gather_into_tensor(topk_idx_global, topk_idx.contiguous(), group=group)
    dist.all_gather_into_tensor(
        topk_weights_global, topk_weights.contiguous(), group=group)

    # Each rank computes only the routes owned by its local experts. Route
    # slots are BF16-rounded like the kernel's combine buffer, then reduced
    # across expert owners before the source-rank slice is selected.
    route_outputs = torch.zeros(
        (num_global_tokens, num_topk, hidden), device="cuda", dtype=torch.float32)
    for token_idx in range(num_global_tokens):
        for topk_i in range(num_topk):
            expert_idx = topk_idx_global[token_idx, topk_i].item()
            owner_rank = expert_idx // num_local_experts
            if owner_rank != rank_idx:
                continue
            local_expert_idx = expert_idx % num_local_experts
            route_weight = topk_weights_global[token_idx, topk_i].item()
            l1_global_scale = 1.0 if l1_global_scales is None else l1_global_scales[local_expert_idx].item()
            l2_global_scale = 1.0 if l2_global_scales is None else l2_global_scales[local_expert_idx].item()
            l1_out = (l1_dequant[local_expert_idx].float() @ x_ref[token_idx]) * l1_global_scale
            gate, up = l1_out[:intermediate_hidden], l1_out[intermediate_hidden:]
            gate = gate.clamp(max=args.activation_clamp)
            up = up.clamp(min=-args.activation_clamp, max=args.activation_clamp)
            intermediate = _silu(gate) * up * route_weight
            # The split kernel persists L1 output as per-row, per-64 FP8 before
            # L2. Reproduce that runtime activation quantization while keeping
            # both weight matrices as exact dequantizations of source MXFP4.
            intermediate_view = intermediate.view(intermediate_hidden // 64, 64)
            intermediate_amax = intermediate_view.abs().amax(dim=-1)
            intermediate_sf = (intermediate_amax / 127.0).clamp_min(
                torch.finfo(torch.float32).tiny)
            intermediate = (
                torch.where((intermediate_view / intermediate_sf.unsqueeze(-1)).round().clamp(-126, 126) == -1,
                            torch.zeros(1, device="cuda"),
                            (intermediate_view / intermediate_sf.unsqueeze(-1)).round().clamp(-126, 126))
                * intermediate_sf.unsqueeze(-1)
            ).view(intermediate_hidden)
            route_outputs[token_idx, topk_i] = (
                (l2_dequant[local_expert_idx].float() @ intermediate)
                * l2_global_scale
            ).to(torch.bfloat16).float()

    dist.all_reduce(route_outputs, op=dist.ReduceOp.SUM, group=group)
    y_ref_global = route_outputs.sum(dim=1).to(torch.bfloat16).float()
    local_token_start = rank_idx * m_tokens
    y_ref = y_ref_global[local_token_start:local_token_start + m_tokens]

    diff = y_kernel.float() - y_ref
    cosine = torch.nn.functional.cosine_similarity(y_kernel.float(), y_ref, dim=-1)

    finite_tensor = torch.tensor(
        int(torch.isfinite(y_kernel).all().item()), dtype=torch.int32, device="cuda")
    cosine_min_tensor = cosine.min()
    cosine_sum_tensor = cosine.sum()
    kernel_norm_sq = y_kernel.float().square().sum()
    ref_norm_sq = y_ref.square().sum()
    abs_max_diff_tensor = diff.abs().max()
    abs_sum_diff_tensor = diff.abs().sum()
    ref_abs_max_tensor = y_ref.abs().max()
    dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(cosine_min_tensor, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(cosine_sum_tensor, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(kernel_norm_sq, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(ref_norm_sq, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(abs_max_diff_tensor, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(abs_sum_diff_tensor, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(ref_abs_max_tensor, op=dist.ReduceOp.MAX, group=group)

    finite = bool(finite_tensor.item())
    cosine_count = m_tokens * num_ranks
    cosine_min = cosine_min_tensor.item()
    cosine_sum = cosine_sum_tensor.item()
    cosine_mean = cosine_sum / cosine_count
    norm_ratio = torch.sqrt(kernel_norm_sq / ref_norm_sq.clamp_min(1e-30)).item()
    abs_max_diff = abs_max_diff_tensor.item()
    abs_mean_diff = abs_sum_diff_tensor.item() / (y_ref.numel() * num_ranks)
    ref_abs_max = ref_abs_max_tensor.item()

    if rank_idx == 0:
        print(f"cum_stats: {cumulative_stats.cpu().tolist()}", flush=True)
        print(
            f"Global diff: max_abs={abs_max_diff:.4e} mean_abs={abs_mean_diff:.4e} "
            f"finite={finite}",
            flush=True,
        )
        print(
            f"Global per-token cosine: min={cosine_min:.4f} mean={cosine_mean:.4f}",
            flush=True,
        )
        print(f"Global norm ratio: kernel/ref={norm_ratio:.4f}", flush=True)

    if not finite:
        raise AssertionError(f"M={m_tokens}, weight_scale={weight_scale:g}: kernel produced non-finite values")
    small_signal_abs_pass = (
        ref_abs_max <= args.small_signal_ref_abs_max
        and abs_max_diff <= args.small_signal_abs_max_threshold
        and abs_mean_diff <= args.small_signal_abs_mean_threshold
    )
    if small_signal_abs_pass:
        if rank_idx == 0:
            print(
                f"Small-signal absolute check PASS: ref_abs_max={ref_abs_max:.4e} "
                f"max_abs_diff={abs_max_diff:.4e} mean_abs_diff={abs_mean_diff:.4e}",
                flush=True,
            )
    elif cosine_mean < args.cosine_mean_threshold:
        raise AssertionError(
            f"M={m_tokens}, weight_scale={weight_scale:g}: cosine_mean={cosine_mean:.4f} < {args.cosine_mean_threshold:.4f}"
        )
    if not small_signal_abs_pass and cosine_min < args.cosine_min_threshold:
        raise AssertionError(
            f"M={m_tokens}, weight_scale={weight_scale:g}: cosine_min={cosine_min:.4f} < {args.cosine_min_threshold:.4f}"
        )
    if not small_signal_abs_pass and not (args.norm_ratio_min <= norm_ratio <= args.norm_ratio_max):
        raise AssertionError(
            f"M={m_tokens}, weight_scale={weight_scale:g}: norm_ratio={norm_ratio:.4f} "
            f"outside [{args.norm_ratio_min:.4f}, {args.norm_ratio_max:.4f}]"
        )

    if rank_idx == 0:
        print(
            f"PASS M={m_tokens} weight_scale={weight_scale:g}: "
            f"global_scale_mode={global_scale_mode} routing_seed={routing_seed} "
            f"cosine_min={cosine_min:.4f} cosine_mean={cosine_mean:.4f}",
            flush=True,
        )

    return cosine_min, cosine_sum, cosine_count, norm_ratio, finite


def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, _, group = init_dist(local_rank, num_local_ranks)
    deep_gemm.set_pdl(os.environ.get("DG_PDL", "0") == "1")
    buffer = None
    try:
        if rank_idx == 0:
            _run_dequant_unit_test()
            # load_inline can deadlock on the torch-extensions build lock in
            # multi-process runs; opt in explicitly when needed.
            if os.environ.get("DG_MXFP4_CUDA_UNIT_TEST", "0") == "1":
                _run_cuda_dequant_lut_unit_test()
        dist.barrier(group=group)
        if get_arch_major() != 9:
            if rank_idx == 0:
                print(f"[SKIP] requires SM90, got SM{get_arch_major()}0", flush=True)
            return
        if args.num_experts % num_local_ranks != 0:
            raise ValueError("num_experts must be divisible by num_processes")
        num_max_tokens_per_rank = args.num_max_tokens_per_rank or max(32, max(args.batches))
        if num_max_tokens_per_rank < max(args.batches):
            raise ValueError("num_max_tokens_per_rank is smaller than the largest M")
        num_local_experts = args.num_experts // num_local_ranks
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            args.num_experts,
            num_max_tokens_per_rank,
            args.num_topk,
            args.hidden,
            args.intermediate_hidden,
            mma_type="fp8xmxfp4",
            activation="swiglu",
        )

        all_results = []
        for weight_scale in args.weight_scales:
            model = _prepare_model(
                args, weight_scale, num_local_experts, rank_idx)
            for routing_seed in args.routing_seeds:
                for global_scale_mode in args.global_scale_modes:
                    for m_tokens in args.batches:
                        all_results.append(_run_case(
                            args, m_tokens, weight_scale, global_scale_mode, routing_seed,
                            rank_idx, group, buffer, *model))
            del model
            torch.cuda.empty_cache()

        if rank_idx == 0:
            cosine_min = min(result[0] for result in all_results)
            cosine_sum = sum(result[1] for result in all_results)
            cosine_count = sum(result[2] for result in all_results)
            norm_ratios = [result[3] for result in all_results]
            finite = all(result[4] for result in all_results)
            print(
                f"AGGREGATE reference=exact_int4 "
                f"cases={len(all_results)} finite={finite} "
                f"cosine_min={cosine_min:.6f} cosine_mean={cosine_sum / cosine_count:.6f} "
                f"norm_ratio_min={min(norm_ratios):.6f} "
                f"norm_ratio_mean={sum(norm_ratios) / len(norm_ratios):.6f} "
                f"norm_ratio_max={max(norm_ratios):.6f}",
                flush=True,
            )
    finally:
        if buffer is not None:
            buffer.destroy()
        dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SM90 MXFP4 MegaMoE correctness gate")
    parser.add_argument("--batches", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate-hidden", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument(
        "--block-n",
        type=int,
        choices=[128, 256],
        default=None,
        help="Override the model-shape layout policy; default selects at prepack time.",
    )
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=0)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument("--weight-seed", type=int, default=20260703)
    parser.add_argument("--routing-seeds", nargs="+", type=int, default=[17, 2027])
    parser.add_argument("--weight-scales", nargs="+", type=float, default=[0.05])
    parser.add_argument(
        "--global-scale-modes",
        nargs="+",
        choices=["none", "expert"],
        default=["none", "expert"],
        help="Run with no global scales and/or per-expert non-unit L1/L2 global scales.",
    )
    parser.add_argument("--cosine-mean-threshold", type=float, default=0.9)
    parser.add_argument("--cosine-min-threshold", type=float, default=0.9)
    parser.add_argument("--norm-ratio-min", type=float, default=0.5)
    parser.add_argument("--norm-ratio-max", type=float, default=2.0)
    parser.add_argument("--small-signal-ref-abs-max", type=float, default=1e-4)
    parser.add_argument("--small-signal-abs-max-threshold", type=float, default=1e-4)
    parser.add_argument("--small-signal-abs-mean-threshold", type=float, default=2e-5)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    torch.multiprocessing.spawn(
        _worker,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
        join=True,
    )
