"""Correctness and latency gate for the H20 Fable dynamic-M frontend."""
import os
import sys

import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import deep_gemm


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD, 384, 64, 8, 3072, 1280,
        mma_type="fp8xmxfp4", activation="swiglu")
    try:
        torch.manual_seed(20260903 + rank)
        router_weight = (torch.randn(384, 3072, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
        for m in (1, 2, 4, 8, 16, 32, 64):
            x = torch.randn(m, 3072, device="cuda", dtype=torch.bfloat16)
            for quant in ("mxfp4", "qoq"):
                deep_gemm.fable_router_quant_topk_frontend(x, router_weight, buffer, quant)
                torch.cuda.synchronize()
                idx = buffer.topk_idx[:m].clone()
                weights = buffer.topk_weights[:m].clone()
                logits = (x.float() @ router_weight.float().t()).to(torch.bfloat16).float()
                ref_values, _ = torch.topk(logits, 8, dim=-1)
                selected = torch.gather(logits, 1, idx)
                torch.testing.assert_close(
                    torch.sort(selected, dim=-1, descending=True).values,
                    ref_values, rtol=2e-2, atol=2e-2)
                torch.testing.assert_close(weights, torch.softmax(selected, dim=-1), rtol=5e-2, atol=5e-3)
                torch.testing.assert_close(weights.sum(-1), torch.ones(m, device="cuda"), rtol=1e-5, atol=1e-5)
                if quant == "mxfp4":
                    x_ref = (buffer.x[:m].float().view(m, 24, 128) * buffer.x_sf[:m].unsqueeze(-1)).view(m, 3072)
                    rel = ((x_ref - x.float()).abs().max() / x.float().abs().max()).item()
                    assert rel < 0.07, rel
                else:
                    assert torch.equal(buffer.x_sf[:m, :1].expand(-1, 24), buffer.x_sf[:m])
                    x_ref = buffer.x[:m].view(torch.int8).float() * buffer.x_sf[:m, :1]
                    rel = ((x_ref - x.float()).abs().max() / x.float().abs().max()).item()
                    assert rel < 0.005, rel
                for _ in range(10):
                    deep_gemm.fable_router_quant_topk_frontend(x, router_weight, buffer, quant)
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                for _ in range(100):
                    deep_gemm.fable_router_quant_topk_frontend(x, router_weight, buffer, quant)
                end.record(); end.synchronize()
                us = torch.tensor(start.elapsed_time(end) * 10.0, device="cuda", dtype=torch.float64)
                dist.all_reduce(us, op=dist.ReduceOp.SUM)
                if rank == 0:
                    print(f"FABLE PASS quant={quant} M={m} rel={rel:.6f} mean_rank_us={us.item()/8:.3f}", flush=True)
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
