import argparse
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Tuple

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto


def import_baseline():
    # Load legacy implements from third-party
    deep_ep, tilelang_ops, do_bench, is_legacy_loaded = None, None, None, False
    # noinspection PyBroadException
    try:
        import deep_ep
        import importlib.util
        from tilelang.profiler.bench import do_bench
        spec = importlib.util.spec_from_file_location(
            'tilelang_ops',
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'third-party', 'tilelang_ops', '__init__.py'))
        tilelang_ops = importlib.util.module_from_spec(spec)
        sys.modules['tilelang_ops'] = tilelang_ops
        spec.loader.exec_module(tilelang_ops)
        is_legacy_loaded = True
    except Exception as ex:
        dist_print(f'Failed to load legacy code: {ex}, skip baseline benchmarking', once_in_node=True)
        dist_print(once_in_node=True)
    return deep_ep, tilelang_ops, do_bench, is_legacy_loaded


# TODO: skip the test for SM90
# noinspection PyUnboundLocalVariable,PyShadowingNames
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    # Settings
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = max(0, args.num_max_tokens_per_rank - random.randint(0, args.num_max_removed_tokens)) \
        if args.num_tokens == 0 else args.num_tokens
    # Owner layout for global M < num_ranks (TP4 x DP2 E2E layout): only `--active-ranks` carry a token, others 0
    if args.active_ranks:
        active = [int(v) for v in args.active_ranks.split(',')]
        num_tokens = num_tokens if rank_idx in active else 0
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank

    # Allocate symmetric memory
    act_format = args.act_format  # 'fp8' (W-MXFP4 x A-FP8), 'mxfp4' (W4A4, UE8M0/32), 'nvfp4' (W4A4, UE4M3/16)
    sf_vec = {'fp8': 32, 'mxfp4': 32, 'nvfp4': 16}[act_format]
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        act_format=act_format
    )

    # ---- FP4 quantization helpers (test-side reference implementations)
    from deep_gemm.utils.math import _quantize_to_fp4_e2m1, ceil_to_ue8m0
    def fp4_sf_for(amax: torch.Tensor) -> torch.Tensor:
        # kernel-matching SF rule: MXFP4 -> UE8M0 = 2^ceil(log2(amax / 6)); NVFP4 -> UE4M3(amax / 6)
        if act_format == 'nvfp4' and not os.environ.get('DG_TEST_NV_UE8M0'):
            return (amax / 6.0).to(torch.float8_e4m3fn).float()
        return ceil_to_ue8m0(amax / 6.0)
    def sf_bytes(sf_f32: torch.Tensor) -> torch.Tensor:
        # SF value -> byte encoding (UE8M0 exponent byte or UE4M3 bits)
        if act_format == 'nvfp4' and not os.environ.get('DG_TEST_NV_UE8M0'):
            return sf_f32.to(torch.float8_e4m3fn).view(torch.uint8)
        return (sf_f32.view(torch.int32) >> 23).to(torch.uint8)
    def cast_to_fp4_rows(t: torch.Tensor):
        # [M, K] -> packed int8 [M, K/2], SF float [M, K/vec], SF int32 (4 bytes per int, K-major) [M, K/(4 vec)]
        m, k = t.shape
        g = t.float().view(m, k // sf_vec, sf_vec)
        amax = g.abs().amax(dim=-1).clamp_min(1e-30)
        sf = fp4_sf_for(amax)
        if os.environ.get('DG_TEST_UNIT_SF'):  # debug: force all input SFs to 1.0 (values clip to +-6)
            sf = torch.ones_like(sf)
        if os.environ.get('DG_TEST_POW2_SF'):  # debug: power-of-two SFs only (exactly representable in both UE8M0 and UE4M3)
            sf = ceil_to_ue8m0(amax / 6.0).clamp(2.0 ** -6, 2.0 ** 8)
        if os.environ.get('DG_TEST_CONST_SF'):  # debug: constant SF (e.g. 0.75 -> tests mantissa handling)
            sf = torch.full_like(sf, float(os.environ['DG_TEST_CONST_SF']))
        codes = _quantize_to_fp4_e2m1(g / sf.unsqueeze(-1)).view(m, k)
        packed = (codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)
        sf_i32 = sf_bytes(sf).contiguous().view(torch.int32)
        return packed.contiguous(), sf, sf_i32
    def dequant_fp4_rows(packed: torch.Tensor, sf_f32: torch.Tensor) -> torch.Tensor:
        u = packed.view(torch.uint8)
        codes = torch.stack([(u & 0xF), (u >> 4)], dim=-1).reshape(*u.shape[:-1], u.shape[-1] * 2).long()
        vals = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float, device='cuda')[codes]
        return vals * sf_f32.repeat_interleave(sf_vec, dim=-1)

    # Create inputs
    # noinspection PyGlobalUndefined
    def create_inputs():
        global x, x_sf_f32, topk_idx, topk_weights, l1_weights, l2_weights, transformed_l1_weights, transformed_l2_weights
        global cumulative_local_expert_recv_stats_fused
        global cumulative_local_expert_recv_stats_baseline
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        l1_weights = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
        l2_weights = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        if args.balanced:
            # forced-balanced routing (same rule as the H20 harness): slot s -> rank s, expert = s*E_local + (g + 7 s) % E_local
            g = (torch.arange(num_tokens, device='cuda', dtype=torch.int64) + rank_idx * 1000003)  # distinct global ids
            s_idx = torch.arange(num_topk, device='cuda', dtype=torch.int64)
            topk_idx = (s_idx[None, :] * num_experts_per_rank + (g[:, None] + 7 * s_idx[None, :]) % num_experts_per_rank).to(topk_idx.dtype)
            topk_weights = torch.full((num_tokens, num_topk), 1.0 / num_topk, dtype=torch.float, device='cuda')
        cumulative_local_expert_recv_stats_fused = torch.randint(
            0, 100, (num_experts_per_rank, ), dtype=torch.int, device='cuda')
        cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_fused.clone()
        if args.masked_ratio > 0:
            rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
            topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)

        # Check SF requirements
        assert hidden % 128 == 0
        assert intermediate_hidden % 128 == 0
        assert l1_weights.shape[2] % 128 == 0 and l2_weights.shape[2] % 128 == 0

        # Cast inputs: FP8 with per-32 UE8M0 SF, or packed FP4 with per-vec SF (4 SF bytes per int)
        x_sf_f32 = None
        if act_format == 'fp8':
            x = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        else:
            x_packed, x_sf_f32, x_sf_i32 = cast_to_fp4_rows(x)
            x = (x_packed, x_sf_i32)

        # Cast grouped BF16 weights to FP4 with MN-major SF
        # TODO: merge with `cast_fp8_fp4_with_major`
        raw_sf_list = []
        def cast_grouped_weights_to_fp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            num_groups, n, k = bf16_weights.shape
            w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
            if act_format == 'fp8':
                w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
                for i in range(num_groups):
                    w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
                raw_sf_list.append(w_sf.clone())  # untransformed [E, N, K/32] fp32 SF for the torch reference
                w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
                return w, w_sf
            # W4A4: per-vec SF with the kernel's SF rule; MN-major packed int layout [E, N, K/(4 vec)] with strides (K/(4 vec) * N, 1, N)
            w_sf = torch.empty((num_groups, n, k // sf_vec), device='cuda', dtype=torch.float)
            w_sf_mn = torch.empty((num_groups, k // (4 * sf_vec), n), device='cuda', dtype=torch.int32).permute(0, 2, 1)
            for i in range(num_groups):
                w[i], w_sf[i], sf_i32 = cast_to_fp4_rows(bf16_weights[i])
                w_sf_mn[i].copy_(sf_i32)
            raw_sf_list.append(w_sf.clone())
            return w, w_sf_mn

        l1_weights = cast_grouped_weights_to_fp4(l1_weights)
        l2_weights = cast_grouped_weights_to_fp4(l2_weights)
        global q_l1_weights, q_l2_weights
        q_l1_weights, q_l2_weights = (l1_weights[0], raw_sf_list[0]), (l2_weights[0], raw_sf_list[1])
        transformed_l1_weights, transformed_l2_weights = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

    # Run fused mega MoE
    # NOTES: copy x into buffer before each call because debug mode zeros the entire buffer
    def copy_inputs():
        buffer.x[:num_tokens].copy_(x[0])
        buffer.x_sf[:num_tokens].copy_(x[1])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)

    def run_fused(copy: bool = True):
        if copy:
            copy_inputs()

        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        # noinspection PyTypeChecker
        (deep_gemm.fp8_fp4_mega_moe if act_format == 'fp8' else deep_gemm.fp4_fp4_mega_moe)(
            y,
            transformed_l1_weights, transformed_l2_weights,
            buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_fused,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math)
        )
        return y, cumulative_local_expert_recv_stats_fused

    dist_print('Config:', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(f' > Buffer: {buffer.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(once_in_node=True)

    # Only do NCU profiling
    if args.ncu_profile_only:
        create_inputs()
        dist_print(f'Run fused kernel:', once_in_node=True)
        run_fused()
        dist_print(f' > Done, exiting', once_in_node=True)

        # Destroy and exit
        dist.barrier()
        buffer.destroy()
        dist.destroy_process_group()
        return

    # Non-overlapped baseline: EP dispatch + GEMM + EP combine
    deep_ep, tilelang_ops, tilelang_bench, is_legacy_loaded = import_baseline()
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    ep_buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank, hidden=hidden,
        num_topk=num_topk, use_fp8_dispatch=True,
        explicitly_destroy=True,
        allow_multiple_reduction=False,
        num_gpu_timeout_secs=10, num_cpu_timeout_secs=30
    ) if is_legacy_loaded else None

    def run_baseline():
        recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
            x, topk_idx=topk_idx, topk_weights=topk_weights,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_baseline,
            num_experts=num_experts, expert_alignment=alignment,
            do_cpu_sync=False, do_handle_copy=False,
            do_expand=True, use_tma_aligned_col_major_sf=True,
        )
        n = recv_x[0].size(0)
        l1_y = torch.empty((n, intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            recv_x, l1_weights, l1_y, handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True, recipe=(1, 1, 32))
        # noinspection PyCallingNonCallable
        l1_y = tilelang_ops.swiglu_apply_weight_to_fp8(
            x=l1_y,
            topk_weights=recv_topk_weights,
            avail_tokens=handle.psum_num_recv_tokens_per_expert[-1],
            num_per_channels=32,
            use_col_major_scales=True,
            round_scale=True,
            ue8m0_scale=True,
            output_bf16=False,
            clamp_value=args.activation_clamp,
            fast_math=bool(args.fast_math)
        )
        l2_y = torch.empty((n, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            l1_y, l2_weights, l2_y, handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True, recipe=(1, 1, 32))
        return ep_buffer.combine(l2_y, handle=handle)[0], cumulative_local_expert_recv_stats_baseline

    # Check correctness (must be bitwise identical)
    num_correctness_tests = 1 if args.num_correctness_tests is None else args.num_correctness_tests
    # noinspection PyBroadException
    if is_legacy_loaded and num_correctness_tests > 0:
        dist_print('Running correctness tests:', once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            for fused_result, baseline_result in zip(run_fused(), run_baseline()):
                assert torch.equal(fused_result, baseline_result)
            if (i + 1) % 100 == 0 or i == num_correctness_tests - 1:
                dist_print(f' > Correctness test #{i + 1}/{num_correctness_tests} passed', once_in_node=True)
        dist_print(once_in_node=True)
    else:
        create_inputs()

    # Count local received tokens
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) | \
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    # Phase stamps (single clean run after the warm-ups, GPU0 prints): slots per the kernel comment
    if args.phase_stamps:
        stamps = torch.zeros(32, dtype=torch.int64, device='cuda')
        os.environ['DG_SM100_PHASE_STAMPS_PTR'] = str(stamps.data_ptr())
        for _ in range(3):
            run_fused()
        torch.cuda.synchronize(); dist.barrier()
        stamps.zero_(); stamps[0] = stamps[3] = 0x7fffffffffffffff
        torch.cuda.synchronize(); dist.barrier()
        # align the ranks on the stream (as in the streamed benchmark) so the barrier waits reflect the kernel, not launch skew
        _flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device='cuda'); _flush.zero_()
        copy_inputs(); dist.all_reduce(torch.ones(1, device='cuda'), group=group)
        run_fused(copy=False); torch.cuda.synchronize()
        st = stamps.cpu().tolist(); t0 = st[0]
        us = lambda i: (st[i] - t0) / 1e3 if st[i] not in (0, 0x7fffffffffffffff) else float('nan')
        names = {1: 'dispatch barrier#1/DONE acquired', 2: 'pool ready (arrival counts)', 3: 'first math task', 4: 'last L1 end',
                 5: 'last L2 end', 6: 'combine barrier#2 done', 7: 'combine end', 8: 'routing count done', 9: 'routing writes/pushes issued',
                 10: 'grid sync/DONE signalled', 12: 'barrier#3 done (kernel tail)'}
        if rank_idx == 0:
            print('PHASES us (GPU0, from kernel entry): ' + ' | '.join(f'{names[i]}={us(i):.1f}' for i in [8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 12]), flush=True)
            avg = lambda a, b: (st[a] / max(st[b], 1)) / 1e3
            print(f'TASKS (GPU0, all SMs): L1 tasks={st[21]} avg={avg(20, 21):.2f}us | L2 tasks={st[23]} avg={avg(22, 23):.2f}us | '
                  f'A-loader L1 arrival spin sum={st[24] / 1e3:.1f}us L2 mask spin sum={st[25] / 1e3:.1f}us | TMEM-empty wait avg={avg(26, 27):.2f}us', flush=True)
        del os.environ['DG_SM100_PHASE_STAMPS_PTR']
        dist.barrier()

    # Benchmark
    if args.bench_stream:
        # Streamed replays (Hopper README method): L2 flush -> on-stream all-reduce (aligns ranks) -> kernel, timed with CUDA events,
        # no host sync / barrier between iterations. Report GPU0 median of the last 3 and of all replays.
        n_warm, n_iter = 3, args.bench_stream
        flush_buf = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device='cuda')
        align_buf = torch.ones(1, dtype=torch.float, device='cuda')
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        for i in range(n_warm + n_iter):
            flush_buf.zero_()
            copy_inputs()
            dist.all_reduce(align_buf, group=group)
            if i >= n_warm: starts[i - n_warm].record()
            run_fused(copy=False)
            if i >= n_warm: ends[i - n_warm].record()
        torch.cuda.synchronize()
        ts = [st.elapsed_time(en) * 1e3 for st, en in zip(starts, ends)]
        last3 = sorted(ts[-3:])[1]; med = sorted(ts)[len(ts) // 2]; mn = min(ts)
        t_all = uneven_all_gather(torch.tensor([last3, med, mn], device='cuda').view(1, 3), group=group)
        dist_print(f'STREAM us [{act_format}] per-rank (last3-median): ' + ' '.join(f'{v:.1f}' for v in t_all[:, 0].tolist()) +
                   f' | GPU0 last3-median={t_all[0, 0]:.1f} all-median={t_all[0, 1]:.1f} min={t_all[0, 2]:.1f}', once_in_node=True)
        # Kernel span (profiler-recorded mega_moe kernel durations) over the same kind of streamed replays: the analog of the
        # Hopper README method (nsys, GPU0, median of the last 3 replays), free of launch gaps
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for i in range(n_iter):
                flush_buf.zero_(); copy_inputs(); dist.all_reduce(align_buf, group=group)
                run_fused(copy=False)
            torch.cuda.synchronize()
        ks = [e.device_time for e in prof.events() if 'mega_moe' in e.name and e.device_type.name == 'CUDA']
        ks = ks[-n_iter:] if len(ks) >= n_iter else ks
        k_last3 = sorted(ks[-3:])[1] if len(ks) >= 3 else float('nan'); k_med = sorted(ks)[len(ks) // 2] if ks else float('nan')
        k_all = uneven_all_gather(torch.tensor([k_last3, k_med, min(ks) if ks else float('nan')], device='cuda').view(1, 3), group=group)
        dist_print(f'KSPAN us [{act_format}] per-rank (last3-median kernel span): ' + ' '.join(f'{v:.1f}' for v in k_all[:, 0].tolist()) +
                   f' | GPU0 last3-median={k_all[0, 0]:.1f} all-median={k_all[0, 1]:.1f} min={k_all[0, 2]:.1f} n={len(ks)}', once_in_node=True)
        t_fused = last3 / 1e6
    else:
      t_fused = bench_kineto(
        run_fused, 'mega_moe',
        barrier=lambda: ep_buffer.barrier(use_comm_stream=False) if ep_buffer else dist.barrier(),
        trace_path=None if not args.dump_profile_traces else f'{args.dump_profile_traces}/mega_moe_rank{rank_idx}.json')
    t_baseline = tilelang_bench(run_baseline, _n_warmup=5, _n_repeat=1, backend='cudagraph', return_mode='median') / 1e3 if is_legacy_loaded else 0

    # TFLOPS: 3 matmuls (L1 left, L1 right, L2), each 2 * M * N * K
    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_fused)

    # HBM bytes: weights (FP4 packed = 0.5 bytes) + activations (FP8 = 1 byte) + output (BF16 = 2 bytes)
    num_touched_experts = torch.unique(gathered_topk_idx.flatten()).numel() - 1 # NOTES minus 1 to exclude "-1"
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden // 2 +   # L1 weights (FP4)
        num_touched_experts * hidden * intermediate_hidden // 2 +       # L2 weights (FP4)
        num_recv_tokens * hidden +                                      # L1 acts read (FP8)
        num_recv_tokens * intermediate_hidden +                         # L1 output write (FP8)
        num_recv_tokens * intermediate_hidden +                         # L2 acts read (FP8)
        num_recv_tokens * hidden * 2                                    # L2 output write (BF16)
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_fused)

    # NVLink bytes: dispatch pull + combine write-back
    num_nvlink_bytes = num_recv_tokens * hidden * 3
    nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, t_fused)

    # Combine reduction (serial) time approximation
    t_reduction = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12

    # Summary
    approx_factor = t_fused / (t_fused - t_reduction)
    if args.torch_ref:
        # ---- pure-torch reference (no DeepEP): dequantized FP8 x, dequantized FP4 weights, kernel's SwiGLU/clamp/per-32 UE8M0 requant
        FP4_VALS = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float, device='cuda')
        def dequant_fp4(packed_i8, sf_f32):  # packed [E, N, K/2] int8, sf [E, N, K/vec] float -> [E, N, K] fp32
            u = packed_i8.view(torch.uint8)
            lo, hi = (u & 0xF).long(), (u >> 4).long()
            codes = torch.stack([lo, hi], dim=-1).reshape(*u.shape[:-1], u.shape[-1] * 2)
            vals = FP4_VALS[codes]
            return vals * sf_f32.repeat_interleave(sf_vec if act_format != 'fp8' else 32, dim=-1)
        def dequant_fp8_ue8m0(x_fp8, sf_packed_int):  # x [M, K] e4m3, sf packed int [M, K/128] (4 ue8m0 per int) -> fp32
            if x_fp8.shape[0] == 0: return x_fp8.float()
            sf_u8 = sf_packed_int.view(torch.uint8).view(x_fp8.shape[0], -1)  # [M, K/32] exponents
            sf = torch.pow(2.0, sf_u8.float() - 127.0)
            return x_fp8.float() * sf.repeat_interleave(32, dim=-1)
        def requant_per32_ue8m0(a):  # kernel's intermediate requant: per-32 amax -> 2^ceil(log2(amax/448)) -> e4m3
            g = a.view(a.shape[0], -1, 32)
            amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
            sf = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
            return ((g / sf).to(torch.float8_e4m3fn).float() * sf).view_as(a)
        def requant_fp4(a):  # W4A4 kernel's intermediate requant: per-vec amax -> SF rule -> e2m1 -> dequant
            if a.shape[0] == 0: return a
            g = a.view(a.shape[0], -1, sf_vec)
            amax = g.abs().amax(dim=-1).clamp_min(1e-30)
            sf = fp4_sf_for(amax)
            codes = _quantize_to_fp4_e2m1(g / sf.unsqueeze(-1))
            vals = FP4_VALS[codes.long()]
            return (vals * sf.unsqueeze(-1)).view_as(a)
        if act_format == 'fp8':
            x_dq = dequant_fp8_ue8m0(x[0], x[1])                               # [T_local, H]
        else:
            x_dq = dequant_fp4_rows(x[0], x_sf_f32) if num_tokens > 0 else torch.zeros((0, hidden), device='cuda')
        X_all = uneven_all_gather(x_dq.contiguous(), group=group)              # [T, H], rank-major order
        IDX_all = uneven_all_gather(topk_idx.contiguous(), group=group)        # [T, topk]
        W_all = uneven_all_gather(topk_weights.contiguous(), group=group)      # [T, topk]
        counts = [c.item() for c in uneven_all_gather(torch.tensor([num_tokens], device='cuda'), group=group)]
        my_start = sum(counts[:rank_idx]); T = X_all.shape[0]
        W1 = dequant_fp4(q_l1_weights[0], q_l1_weights[1])                      # [E_local, 2I, H]
        W2 = dequant_fp4(q_l2_weights[0], q_l2_weights[1])                      # [E_local, H, I]
        y_part = torch.zeros((T, hidden), dtype=torch.float, device='cuda')
        c = float(args.activation_clamp)
        for e_local in range(num_experts_per_rank):
            e = rank_idx * num_experts_per_rank + e_local
            t_ids, s_ids = torch.nonzero(IDX_all == e, as_tuple=True)
            if t_ids.numel() == 0: continue
            h = X_all[t_ids] @ W1[e_local].T                                  # [n, 2I], fp32 accumulate like the MMA
            gate, up = h[:, :intermediate_hidden].to(torch.bfloat16), h[:, intermediate_hidden:].to(torch.bfloat16)
            gate = torch.clamp_max(gate, c); up = torch.clamp(up, -c, c)
            gate, up = gate.float(), up.float()
            act = gate * torch.sigmoid(gate) * up * W_all[t_ids, s_ids][:, None]
            act = requant_per32_ue8m0(act) if act_format == 'fp8' else requant_fp4(act)
            if os.environ.get('DG_TEST_DUMP_L2') and act_format != 'fp8' and t_ids.numel() == 1:
                # 1 token per non-empty local expert: the kernel places expert j's token at pool row 16 * j (BLOCK_M = 16 for these M)
                if 'dumped' not in globals():
                    globals()['dumped'] = True
                    run_fused(); torch.cuda.synchronize()
                    globals()['nonempty'] = [ee for ee in range(num_experts_per_rank) if (IDX_all == rank_idx * num_experts_per_rank + ee).any()]
                row = 16 * globals()['nonempty'].index(e_local)
                if rank_idx == 0:
                    sfr = (row // 16) * 128
                    nwx = hidden // (4 * sf_vec)
                    xw = buffer.l1_acts_sf[sfr, :nwx].contiguous().view(torch.uint8)
                    xa = buffer.l1_acts[row].view(torch.uint8)
                    dist_print(f' > [L1 in] rank 0 e_local {e_local} row {row}: l1_acts nz={int((xa != 0).sum())}/{xa.numel()} l1_acts_sf row {sfr} bytes[:6]={xw[:6].tolist()} nz={int((xw != 0).sum())}/{xw.numel()}')
                l2_packed = buffer.l2_acts[row:row + 1]
                nw = intermediate_hidden // (4 * sf_vec)
                sf_row = (row // 16) * 128 + ((row % 16) & ~127) + ((row % 16) & 31) * 4 + (((row % 16) >> 5) & 3)  # transform_sf_token_idx for BLOCK_M=16 / SF_BLOCK_M=128
                l2_sf_bytes = buffer.l2_acts_sf[sf_row, :nw].contiguous().view(torch.uint8)
                l2_sf = (l2_sf_bytes.view(torch.float8_e4m3fn).float() if (act_format == 'nvfp4' and not os.environ.get('DG_TEST_NV_UE8M0'))
                         else torch.exp2(l2_sf_bytes.float() - 127.0))
                k_act = dequant_fp4_rows(l2_packed, l2_sf[None, :])[0]; r_act = act[0]
                e = (k_act - r_act); fit = ((k_act * r_act).sum() / (r_act * r_act).sum().clamp_min(1e-12)).item()
                dist_print(f' > [L1 dump] rank {rank_idx} e_local {e_local} row {row}: rel_rmse={(e.norm() / r_act.norm().clamp_min(1e-12)).item():.4f} fit={fit:.3f} '
                           f'sf[:4]={l2_sf[:4].tolist()} per-64-slice rel: {[round(((k_act[i*64:(i+1)*64]-r_act[i*64:(i+1)*64]).norm()/r_act[i*64:(i+1)*64].norm().clamp_min(1e-9)).item(),2) for i in range(0, 20, 3)]}')
            y_part.index_add_(0, t_ids, act @ W2[e_local].T)
        dist.all_reduce(y_part, group=group)
        y_ref = y_part[my_start:my_start + num_tokens]
        y_k = run_fused()[0].float(); torch.cuda.synchronize()
        err = (y_k - y_ref); rel_rmse = (err.norm() / y_ref.norm().clamp_min(1e-12)).item()
        if y_ref.numel() and os.environ.get('DG_TEST_SF_DEBUG'):
            fit = ((y_k * y_ref).sum() / (y_ref * y_ref).sum().clamp_min(1e-12)).item()
            rel_fit = ((y_k - fit * y_ref).norm() / y_ref.norm().clamp_min(1e-12)).item()
            cols = (y_k - y_ref).abs().view(-1, hidden).mean(0)
            worst = torch.topk(cols, 8).indices.tolist()
            dist_print(f' > rank {rank_idx}: best-fit scale={fit:.4f} rel_rmse_after_fit={rel_fit:.4f} worst cols={worst} '
                       f'first8 y_k={y_k.view(-1)[:8].tolist()} y_ref={y_ref.view(-1)[:8].tolist()}')
        max_abs = err.abs().max().item() if err.numel() else 0.0; ref_scale = y_ref.abs().max().item() if y_ref.numel() else 0.0
        rel_list = uneven_all_gather(torch.tensor([rel_rmse], device='cuda'), group=group).tolist()
        dist_print(f'Correctness [{act_format}] vs torch reference (bf16-rounded gate/up, kernel requant rule): rel_rmse per rank = ' + ' '.join(f'{v:.4f}' for v in rel_list), once_in_node=True)
        dist_print(f' > rank {rank_idx}: rel_rmse={rel_rmse:.5f} max_abs_err={max_abs:.4f} (ref max |y|={ref_scale:.2f}, tokens={num_tokens})')
    # ---- cross-rank timing summary (kernel span from kineto, L2 flushed between runs)
    t_all = uneven_all_gather(torch.tensor([t_fused * 1e6], device='cuda'), group=group).tolist()
    dist_print(f'TIMING us per rank (kineto, fused kernel): ' + ' '.join(f'{v:.1f}' for v in t_all) + f' | rank0={t_all[0]:.1f} max={max(t_all):.1f} median={sorted(t_all)[len(t_all)//2]:.1f}', once_in_node=True)
    dist_print('Performance:', once_in_node=True)
    dist_print(f' > EP: {rank_idx:2}/{num_ranks} | '
               f'{tflops:4.0f} TFLOPS | '
               f'overlap: '
               f'{tflops * approx_factor:4.0f} TFLOPS, '
               f'HBM {hbm_gbs * approx_factor:4.0f} GB/s, '
               f'NVL {nvlink_gbs * approx_factor:3.0f} GB/s | '
               f'{t_fused * 1e6:4.0f} us, '
               f'reduction: {t_reduction * 1e6:4.1f} us | '
               f'{safe_div(t_baseline, t_fused):.2f}x legacy')

    # Exit
    dist.barrier()
    buffer.destroy()
    ep_buffer.destroy() if is_legacy_loaded else None
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PyTorch symmetric memory')

    # Resource settings
    parser.add_argument('--ncu-profile-only', action='store_true', help='Only run profiling without correctness test')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')

    # Model settings
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192, help='Number of maximum tokens per rank')
    parser.add_argument('--num-tokens', type=int, default=0, help='Number of tokens per rank (follow max minus removed if 0)')
    parser.add_argument('--balanced', action='store_true', help='forced-balanced routing (slot s -> rank s), weights 1/topk')
    parser.add_argument('--bench-stream', type=int, default=0, help='streamed replay benchmark with N replays (0 = kineto)')
    parser.add_argument('--phase-stamps', action='store_true', help='print globaltimer phase breakdown of one run (GPU0)')
    parser.add_argument('--push', type=int, default=None, help='DG_SM100_PUSH_DISPATCH (1 = push dispatch + DONE flags, 0 = barrier/pull)')
    parser.add_argument('--no-clean-barrier', type=int, default=None, help='DG_SM100_NO_CLEAN_BARRIER (rotated pool, needs push)')
    parser.add_argument('--act-format', type=str, default='fp8', choices=['fp8', 'mxfp4', 'nvfp4'], help='activation format: fp8 (W-MXFP4 x A-FP8), mxfp4 / nvfp4 (W4A4)')
    parser.add_argument('--torch-ref', action='store_true', help='check y against a pure-torch reference (no DeepEP needed)')
    parser.add_argument('--active-ranks', type=str, default='', help='Comma list of ranks that carry tokens; other ranks get 0 tokens (e.g. 0,4 for global M=2 in a TP4xDP2 layout)')
    parser.add_argument('--num-max-removed-tokens', type=int, default=0, help='Maximum number of tokens to remove')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=3072, help='Intermediate hidden size')
    parser.add_argument('--activation-clamp', type=float, default=10, help='Clamp value for activation')
    parser.add_argument('--num-experts', type=int, default=384, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of expert selections')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--fast-math', type=int, default=1, help='Enable fast math (0 or 1, default: 1)')

    # Test settings
    parser.add_argument('--num-correctness-tests', type=int, default=None, help='Pressure test')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    parser.add_argument('--local-rank-idx', type=int, default=None, help='Run as single process with this local rank (e.g. for NCU prof)')
    args = parser.parse_args()

    # Create dump trace directories
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    if args.local_rank_idx is not None:
        # Single-process mode: each process is launched separately (e.g. by NCU)
        test(args.local_rank_idx, args.num_processes, args)
    else:
        # Launch tests
        num_processes = args.num_processes
        if args.push is not None: os.environ['DG_SM100_PUSH_DISPATCH'] = str(args.push)
        if args.no_clean_barrier is not None: os.environ['DG_SM100_NO_CLEAN_BARRIER'] = str(args.no_clean_barrier)
        torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)
