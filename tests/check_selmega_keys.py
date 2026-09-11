"""Single-GPU check of the FE side of DG_FE_SELECT_IN_MEGA: knob-1 compact keys vs a Python logit
reference (bf16-rounded fp32 x @ w^T) and vs the FE knob-0 top-8. Prints the raw keys of experts
whose key value disagrees with the reference by > 0.05."""
import os, sys, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import deep_gemm  # noqa: E402
from ncu_frontend_tinym import HIDDEN, EXPERTS, TOPK, make_buffer  # noqa: E402
rows = int(sys.argv[1]) if len(sys.argv) > 1 else 2
torch.manual_seed(20260805)
w = (torch.randn(EXPERTS, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
buf = make_buffer(64)


def key_value(k):   # u32 key -> bf16 logit
    o = k >> 16
    b = torch.where((o & 0x8000) != 0, o & 0x7FFF, (~o) & 0xFFFF)
    return b.to(torch.int16).view(torch.bfloat16).float()


for seed in range(4):
    torch.manual_seed(17000 + seed * 7919)
    x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ref = (x.float() @ w.float().T)
    deep_gemm.fable_router_quant_topk_frontend(x, w, buf, quant="mxfp4", select_in_mega=0)
    idx0 = buf.topk_idx[:rows].clone()
    deep_gemm.fable_router_quant_topk_frontend(x, w, buf, quant="mxfp4", select_in_mega=1)
    torch.cuda.synchronize()
    flat = deep_gemm.fable_frontend_keys(buf).clone().view(-1).to(torch.int64) & 0xFFFFFFFF   # [1024] u32, row t at t*384
    for t in range(rows):
        keys = flat[t * EXPERTS:(t + 1) * EXPERTS]
        kidx = 0xFFFF - (keys & 0xFFFF)
        val = key_value(keys)
        bad = ((val - ref[t]).abs() > 0.05) | (kidx != torch.arange(EXPERTS, device="cuda"))
        top = torch.topk(keys, TOPK).indices
        print(f"seed {seed} row {t}: zero keys {int((keys == 0).sum())}, bad-value/idx experts {int(bad.sum())} "
              f"first {torch.nonzero(bad).view(-1)[:6].tolist()}; keys top8 {kidx[top].tolist()} knob0 {idx0[t].tolist()}")
        if bad.any():
            e = torch.nonzero(bad).view(-1)[:4]
            print(f"    experts {e.tolist()}: key 0x{[hex(int(v)) for v in keys[e]]} keyval {val[e].tolist()} ref {ref[t][e].tolist()}")
