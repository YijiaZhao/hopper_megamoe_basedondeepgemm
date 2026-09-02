"""Single-GPU A/B of the fused frontend implementations (legacy KF/generic vs tensor-core).

Run twice: DG_FRONTEND_TC=0 python tests/bench_frontend_tc_ab.py ; DG_FRONTEND_TC=1 ...
For each (h, e, topk) x M: validates against a torch reference (top-k values within
bf16 1-ulp tolerance, weights, dequantized activations) and reports kineto kernel time.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deep_gemm
from deep_gemm import _C

SHAPES = [(4096, 128, 6), (3072, 384, 8), (6144, 384, 8)]
BATCHES = [1, 4, 8, 16, 32, 64]


def kernel_us(fn, iters=50):
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot = 0.0
    for ev in prof.events():
        if ev.device_type.name == 'CUDA' and 'memcpy' not in ev.name.lower():
            tot += ev.time_range.elapsed_us()
    return tot / iters


def check(hidden, router_w, x, x_sf, idx, w, topk, mode):
    m, h = hidden.shape
    logits = (hidden.float() @ router_w.float().t()).to(torch.bfloat16).float()
    ref_v, _ = torch.topk(logits, topk, dim=-1)
    sel = torch.gather(logits, 1, idx)
    torch.testing.assert_close(torch.sort(sel, -1, descending=True).values, ref_v, rtol=2e-2, atol=2e-2)
    assert all(len(set(r.tolist())) == topk for r in idx)
    torch.testing.assert_close(w, torch.softmax(sel, -1), rtol=5e-2, atol=5e-3)
    if mode == 0:
        xq = (x.float().view(m, h // 128, 128) * x_sf.unsqueeze(-1)).view(m, h)
        tol = 0.07
    else:
        xq = x.view(torch.int8).float() * x_sf[:, :1]
        tol = 0.005
    rel = ((xq - hidden.float()).abs().max() / hidden.float().abs().max()).item()
    assert rel < tol, rel


def main():
    impl = os.environ.get('DG_FRONTEND_TC', '1')
    torch.manual_seed(0)
    for (h, e, topk) in SHAPES:
        router_w = (torch.randn(e, h, device='cuda', dtype=torch.bfloat16) * 0.05).contiguous()
        for mode, name in ((0, 'fp8'), (1, 'int8')):
            row = []
            for m in BATCHES:
                hidden = torch.randn(m, h, device='cuda', dtype=torch.bfloat16)
                x = torch.empty(m, h, device='cuda', dtype=torch.float8_e4m3fn)
                x_sf = torch.empty(m, h // 128, device='cuda', dtype=torch.float32)
                idx = torch.empty(m, topk, device='cuda', dtype=torch.int64)
                w = torch.empty(m, topk, device='cuda', dtype=torch.float32)
                if mode == 0:
                    fn = lambda: _C.mxfp4_router_quant_topk(hidden, router_w, x, x_sf, idx, w)
                else:
                    fn = lambda: _C.qoq_fused_router_quant_topk(hidden, router_w, x, x_sf, idx, w)
                fn(); torch.cuda.synchronize()
                check(hidden, router_w, x, x_sf, idx, w, topk, mode)
                row.append(f"M{m}={kernel_us(fn):.1f}")
            print(f"[TC={impl}] h={h} e={e} topk={topk} {name}: " + "  ".join(row), flush=True)


if __name__ == '__main__':
    main()
