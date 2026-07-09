"""Prologue-int benchmark: per-call int4->int8 decode pass + PRE-mode kernel.

Large-M strategy for W4A8-int, mirroring bench_mxfp4_prologue.py: instead of
paying the in-loop decode tax (weights re-decoded once per 64-token tile), run
a bandwidth-bound int4->int8 decode kernel per layer, then invoke the MegaMoE
kernel in DG_W4A8_INT_PRE mode (B TMA loads the decoded int8 rows straight
into smem_b, swizzled like A; the per-row fp32 scale plane and the int
promote path are unchanged, so numerics stay exactly the int4 scheme).
Weights stay 4-bit at rest; the int8 copy is a transient per-layer scratch.

Run with DG_W4A8_INT=1 DG_W4A8_INT_L2=1 DG_W4A8_INT_PRE=1.
"""

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
from deep_gemm.mega import _interleave_l1_weights as _il1
from deep_gemm.testing import bench_kineto, get_arch_major
from deep_gemm.utils.dist import dist_print, init_dist

INT4_GROUP = 128

_EXT = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// int4 (marlin nibble pack: byte j = elem(j+4) | elem(j) << 4 per 8-elem
// chunk) -> sign-extended int8. One thread = one packed word (8 elems).
__global__ void int4_prologue_decode(
        const uint32_t* __restrict__ packed, uint2* __restrict__ out,
        const long total_words) {
    const long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (tid >= total_words) return;
    const uint32_t q = __ldg(packed + tid);
    const uint32_t hi_n = (q >> 4) & 0x0F0F0F0Fu;   // elems 0..3
    const uint32_t lo_n = q & 0x0F0F0F0Fu;          // elems 4..7
    // per-byte sign extend: (v ^ 8) - 8
    const uint32_t e03 = __vsub4(hi_n ^ 0x08080808u, 0x08080808u);
    const uint32_t e47 = __vsub4(lo_n ^ 0x08080808u, 0x08080808u);
    out[tid] = make_uint2(e03, e47);
}

void int4_prologue(torch::Tensor packed, torch::Tensor out) {
    const long total_words = packed.numel() / 4;
    int4_prologue_decode<<<(total_words + 255) / 256, 256>>>(
        reinterpret_cast<const uint32_t*>(packed.data_ptr<uint8_t>()),
        reinterpret_cast<uint2*>(out.data_ptr<uint8_t>()), total_words);
}

// QoQ variant: fold the per-group INTEGER scale s2 into the decoded int8
// (integer-domain fold-at-decode; product <= 126 by construction).
__global__ void int4_prologue_qoq_decode(
        const uint32_t* __restrict__ packed, const uint8_t* __restrict__ s2p,
        uint2* __restrict__ out, const long total_words,
        const int words_per_row, const int groups_per_row) {
    const long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (tid >= total_words) return;
    const long row = tid / words_per_row;
    const int wir = (int)(tid - row * words_per_row);
    const int g = (wir * 8) / 128;
    const int s2 = (int)__ldg(s2p + row * groups_per_row + g);
    const uint32_t q = __ldg(packed + tid);
    const uint32_t hi_n = (q >> 4) & 0x0F0F0F0Fu;
    const uint32_t lo_n = q & 0x0F0F0F0Fu;
    const uint32_t e03 = __vsub4(hi_n ^ 0x08080808u, 0x08080808u);
    const uint32_t e47 = __vsub4(lo_n ^ 0x08080808u, 0x08080808u);
    uint2 d = make_uint2(e03, e47);
    int8_t b[8];
    *reinterpret_cast<uint2*>(b) = d;
    #pragma unroll
    for (int i = 0; i < 8; ++ i)
        b[i] = (int8_t)((int)b[i] * s2);
    out[tid] = *reinterpret_cast<uint2*>(b);
}

void int4_prologue_qoq(torch::Tensor packed, torch::Tensor s2, torch::Tensor out) {
    const long total_words = packed.numel() / 4;
    const int words_per_row = packed.size(-1) / 4;
    const int groups_per_row = s2.size(-1);
    int4_prologue_qoq_decode<<<(total_words + 255) / 256, 256>>>(
        reinterpret_cast<const uint32_t*>(packed.data_ptr<uint8_t>()),
        s2.data_ptr<uint8_t>(),
        reinterpret_cast<uint2*>(out.data_ptr<uint8_t>()),
        total_words, words_per_row, groups_per_row);
}
"""
    cpp_src = ("void int4_prologue(torch::Tensor, torch::Tensor);\n"
               "void int4_prologue_qoq(torch::Tensor, torch::Tensor, torch::Tensor);")
    _EXT = load_inline(
        name="deepgemm_int4_prologue",
        cpp_sources=cpp_src, cuda_sources=cuda_src,
        functions=["int4_prologue", "int4_prologue_qoq"],
        extra_cuda_cflags=["--expt-relaxed-constexpr"], verbose=False)
    return _EXT


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


def _int4_scale_tm(scale_fp32, block_n=128):
    E, N, Kb = scale_fp32.shape
    b = scale_fp32.contiguous().view(torch.uint8).view(E, N, Kb, 4)
    return (b.view(E, N // block_n, block_n, Kb, 4).permute(0, 1, 3, 2, 4).contiguous())


def quantize_qoq(weight, group=128):
    w = weight.float()
    s1 = (w.abs().amax(dim=-1, keepdim=True) / 120.0).clamp_min(1e-30)
    s1 = s1.to(torch.bfloat16).to(torch.float32)
    w8 = (w / s1).round().clamp(-120, 120)
    E, N, K = w8.shape
    w8g = w8.view(E, N, K // group, group)
    amax8 = w8g.abs().amax(dim=-1, keepdim=True)
    s2 = (amax8 / 7.0).ceil().clamp(1, 127)
    w4 = (w8g / s2).round().clamp(-7, 7).view(E, N, K)
    nib = (w4.to(torch.int8) & 0x0F).to(torch.uint8).view(E, N, K // 8, 8)
    packed = (nib[..., 4:8] | (nib[..., 0:4] << 4)).view(E, N, K // 2).contiguous()
    s2_t = s2.squeeze(-1).to(torch.uint8).contiguous()      # (E,N,K/128)
    # Coeff word plane: [s2:u8 | 0 | s1:bf16-hi16], int32 viewed as bytes.
    s1w = (s1.view(torch.int32) & -65536).expand(E, N, K // group)
    plane = (s1w | s2.squeeze(-1).to(torch.int32)).contiguous().view(torch.float32)
    return packed, s2_t, plane


def _torch_decode_ref(packed, N, K):
    E = packed.shape[0]
    b = packed.view(E, N, K // 8, 4)
    def s4(x):
        x = x.to(torch.int16); return torch.where(x >= 8, x - 16, x)
    q = torch.empty(E, N, K // 8, 8, dtype=torch.int16, device=packed.device)
    q[..., 0:4] = s4(b >> 4)
    q[..., 4:8] = s4(b & 0x0F)
    return q.to(torch.int8).view(E, N, K)


def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    deep_gemm.set_pdl(os.environ.get("DG_PDL", "0") == "1")
    buffer = None
    try:
        if get_arch_major() != 9:
            return
        assert os.environ.get("DG_W4A8_INT_PRE", "0") == "1", \
            "run with DG_W4A8_INT=1 DG_W4A8_INT_L2=1 DG_W4A8_INT_PRE=1"
        num_max = max(args.batches)
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group, args.num_experts, num_max, args.num_topk,
            args.hidden, args.intermediate_hidden,
            mma_type="fp8xmxfp4", activation="swiglu")
        nle = args.num_experts // num_ranks
        torch.manual_seed(args.weight_seed + rank_idx * 1000003)
        l1_bf = torch.randn((nle, 2 * args.intermediate_hidden, args.hidden),
                            dtype=torch.bfloat16, device="cuda") * args.weight_scale
        if args.qoq:
            l1_p, l1_s2, l1_s = quantize_qoq(l1_bf); del l1_bf
        else:
            l1_p, l1_s = quantize_to_int4(l1_bf); del l1_bf
        l2_bf = torch.randn((nle, args.hidden, args.intermediate_hidden),
                            dtype=torch.bfloat16, device="cuda") * args.weight_scale
        if args.qoq:
            l2_p, l2_s2, l2_s = quantize_qoq(l2_bf); del l2_bf
        else:
            l2_p, l2_s = quantize_to_int4(l2_bf); del l2_bf
        # Pre-interleave L1 rows so decoded int8 is already gate/up interleaved.
        if args.qoq:
            l1_p, l1_s2 = _il1((l1_p, l1_s2))
            l1_s = _il1((l1_s, l1_s))[0]
        else:
            l1_p, l1_s = _il1((l1_p, l1_s))
        l1_sc_tm = _int4_scale_tm(l1_s, block_n=args.block_n)
        l2_sc_tm = _int4_scale_tm(l2_s, block_n=args.block_n)
        ext = _load_ext()

        # Transient per-layer int8 scratch (reused every call).
        l1_w = torch.empty((nle, 2 * args.intermediate_hidden, args.hidden),
                           dtype=torch.uint8, device="cuda")
        l2_w = torch.empty((nle, args.hidden, args.intermediate_hidden),
                           dtype=torch.uint8, device="cuda")

        # One-time numeric sanity: CUDA decode == torch reference decode.
        ext.int4_prologue(l1_p, l1_w)
        ref = _torch_decode_ref(l1_p, 2 * args.intermediate_hidden, args.hidden)
        ok = bool((l1_w.view(torch.int8) == ref).all().item())
        if rank_idx == 0:
            print(f"PROLOGUE-DECODE-EXACT {ok}", flush=True)

        dist.barrier(group=group)
        dist_print(f"int prologue bench: ranks={num_ranks} H={args.hidden} "
                   f"I={args.intermediate_hidden} E={args.num_experts}", once_in_node=True)
        for m in args.batches:
            torch.manual_seed(args.input_seed + m * 1009 + rank_idx * 1000003)
            x_bf = torch.randn((m, args.hidden), dtype=torch.bfloat16, device="cuda")
            scores = torch.randn((m, args.num_experts), dtype=torch.float32, device="cuda")
            tw, ti = torch.topk(scores, args.num_topk, dim=-1, largest=True, sorted=False)
            if os.environ.get('DG_BALANCED_ROUTING', '0') != '0':
                _p = ((torch.arange(m, device='cuda', dtype=torch.int64)
                       + rank_idx * m) * args.num_topk).unsqueeze(1) + torch.arange(
                    args.num_topk, device='cuda', dtype=torch.int64).unsqueeze(0)
                ti = (_p * 101) % args.num_experts
                tw = scores.gather(1, ti)
            x = x_bf.float()
            x_ts = (x.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
            x_i8 = (x / x_ts).round().clamp(-127, 127).to(torch.int8)
            x_sf = x_ts.repeat(1, args.hidden // 128).contiguous()
            y = torch.empty((m, args.hidden), dtype=torch.bfloat16, device="cuda")
            stats = torch.zeros(nle, dtype=torch.int32, device="cuda")

            def run():
                buffer.x[:m].view(torch.uint8).copy_(x_i8.view(torch.uint8))
                buffer.x_sf[:m].copy_(x_sf)
                buffer.topk_idx[:m].copy_(ti)
                buffer.topk_weights[:m].copy_(tw)
                if args.qoq:
                    ext.int4_prologue_qoq(l1_p, l1_s2, l1_w)
                    ext.int4_prologue_qoq(l2_p, l2_s2, l2_w)
                else:
                    ext.int4_prologue(l1_p, l1_w)
                    ext.int4_prologue(l2_p, l2_w)
                deep_gemm.mxfp4_mega_moe(
                    y, (l1_w, l1_sc_tm), (l2_w, l2_sc_tm), buffer,
                    cumulative_local_expert_recv_stats=stats,
                    recipe=(128, args.block_n, 128),
                    activation="swiglu", activation_clamp=args.activation_clamp,
                    fast_math=True)
                return y

            for _ in range(args.warmup):
                run()
            torch.cuda.synchronize()
            dist.barrier(group=group)
            elapsed = bench_kineto(
                run, ("int4_prologue", "sm90_mxfp4_mega_moe"),
                barrier=lambda: dist.barrier(group=group),
                num_tests=args.num_tests, suppress_kineto_output=True,
                with_multiple_kernels=True)
            t = torch.tensor(sum(elapsed) if isinstance(elapsed, (list, tuple)) else elapsed,
                             dtype=torch.float64, device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
            if rank_idx == 0:
                print(f"RESULT M={m} bn={args.block_n} qoq={int(args.qoq)} int_prologue mean_us={t.item() / num_ranks * 1e6:.3f}",
                      flush=True)
    finally:
        if buffer is not None:
            buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--block-n", type=int, choices=[128, 256], default=128)
    parser.add_argument("--qoq", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1280, 1536, 2048, 3072, 4096, 8192])
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate-hidden", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--weight-scale", type=float, default=0.05)
    parser.add_argument("--weight-seed", type=int, default=20260703)
    parser.add_argument("--input-seed", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--num-tests", type=int, default=20)
    args = parser.parse_args()
    _load_ext()  # precompile once; workers hit the build cache
    torch.multiprocessing.spawn(_worker, args=(args.num_processes, args),
                                nprocs=args.num_processes, join=True)
