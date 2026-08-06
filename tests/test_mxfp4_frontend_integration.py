"""8-rank integration gate for fused BF16 Router+FP8 quant+TopK frontend."""
import os
import sys
import types
import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.testing import get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist
sys.path.insert(0, os.path.dirname(__file__))
from bench_mxfp4_mega_moe_sm90 import _prepare_weights


def worker(local_rank: int, world: int):
    rank, nranks, group = init_dist(local_rank, world)
    assert nranks == world and get_arch_major() == 9
    tp = int(os.environ.get("TP", "1"))
    assert nranks % tp == 0
    tp_group = None
    for first in range(0, nranks, tp):
        ranks = list(range(first, first + tp))
        candidate = dist.new_group(ranks)
        if rank in ranks:
            tp_group = candidate
    assert tp_group is not None
    args = types.SimpleNamespace(
        weight_seed=20260703, weight_scale=0.05, hidden=4096,
        intermediate_hidden=2048, block_n=128,
    )
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, 128, 64, 6, 4096, 2048,
        mma_type="fp8xmxfp4", activation="swiglu")
    try:
        l1, l2 = _prepare_weights(args, 128 // nranks, rank)
        torch.manual_seed(20260805)
        router_weight = (torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
        stats = torch.zeros(128 // nranks, device="cuda", dtype=torch.int32)
        for m in (4, 8, 16, 32, 64):
            torch.manual_seed(17 + m * 1009 + rank * 1000003)
            hidden = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
            logits = hidden @ router_weight.t()
            vals, ids = torch.topk(logits, 6, dim=-1, sorted=False)
            weights = torch.softmax(vals.float(), dim=-1)
            x_fp8, x_sf = per_token_cast_to_fp8(
                hidden, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
            buffer.x[:m].copy_(x_fp8)
            buffer.x_sf[:m].copy_(x_sf)
            buffer.topk_idx[:m].copy_(ids)
            buffer.topk_weights[:m].copy_(weights)
            y_ref = torch.empty(m, 4096, device="cuda", dtype=torch.bfloat16)
            stats.zero_()
            deep_gemm.mxfp4_mega_moe(
                y_ref, l1, l2, buffer, cumulative_local_expert_recv_stats=stats,
                recipe=(128, 128, 128), activation_clamp=10.0,
                attn_tp_size=tp, attn_tp_group=tp_group, broadcast_output=False)
            torch.cuda.synchronize(); dist.barrier(group=group)
            if tp > 1:
                y_ref_global = torch.empty(m * tp, 4096, device="cuda", dtype=torch.bfloat16)
                dist.all_gather_into_tensor(y_ref_global, y_ref, group=tp_group)
                y_new = torch.empty_like(y_ref_global)
                local_y = torch.empty_like(y_ref)
            else:
                y_ref_global = y_ref
                y_new = torch.empty_like(y_ref)
                local_y = None
            stats.zero_()
            deep_gemm.mxfp4_mega_moe_from_bf16(
                y_new, hidden, router_weight, l1, l2, buffer,
                cumulative_local_expert_recv_stats=stats,
                recipe=(128, 128, 128), activation_clamp=10.0,
                attn_tp_size=tp, attn_tp_group=tp_group, broadcast_output=False,
                tp_combine_output=local_y)
            torch.cuda.synchronize(); dist.barrier(group=group)
            diff = (y_new.float() - y_ref_global.float()).abs()
            max_abs = torch.tensor(diff.max().item(), device="cuda")
            mean_abs = torch.tensor(diff.mean().item(), device="cuda")
            exact = torch.tensor(float(torch.equal(y_new, y_ref_global)), device="cuda")
            ids_exact = torch.tensor(float(torch.equal(buffer.topk_idx[:m], ids)), device="cuda")
            weight_err = torch.tensor((buffer.topk_weights[:m] - weights).abs().max().item(), device="cuda")
            cosine = torch.nn.functional.cosine_similarity(
                y_new.float(), y_ref_global.float(), dim=-1).min()
            dist.all_reduce(max_abs, op=dist.ReduceOp.MAX, group=group)
            dist.all_reduce(mean_abs, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(exact, op=dist.ReduceOp.MIN, group=group)
            dist.all_reduce(ids_exact, op=dist.ReduceOp.MIN, group=group)
            dist.all_reduce(weight_err, op=dist.ReduceOp.MAX, group=group)
            dist.all_reduce(cosine, op=dist.ReduceOp.MIN, group=group)
            if rank == 0:
                print(f"INTEGRATION TP={tp} M={m} exact={int(exact.item())} ids={int(ids_exact.item())} "
                      f"weight_err={weight_err.item():.6g} max_abs={max_abs.item():.6g} "
                      f"mean_abs={mean_abs.item()/nranks:.6g} cos_min={cosine.item():.8f}", flush=True)
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    world = int(os.environ.get("WORLD", "8"))
    torch.multiprocessing.spawn(worker, args=(world,), nprocs=world, join=True)
