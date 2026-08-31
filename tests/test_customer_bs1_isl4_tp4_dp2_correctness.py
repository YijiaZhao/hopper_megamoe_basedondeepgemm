"""Correctness for customer E384/H3072/I1280/TopK8, TP4/DP2/EP8."""
import os, sys, types
import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))
import deep_gemm
from deep_gemm.testing import get_arch_major
from deep_gemm.utils.dist import init_dist
from bench_mxfp4_mega_moe_sm90 import _prepare_weights

WORLD=8; TP=4; M=4; H=3072; I=1280; E=384; TOPK=8

def make_tp_group(rank):
    mine=None
    for first in range(0,WORLD,TP):
        ranks=list(range(first,first+TP)); g=dist.new_group(ranks)
        if rank in ranks: mine=g
    return mine

def worker(local_rank):
    rank, world, ep_group = init_dist(local_rank, WORLD)
    assert world == WORLD and get_arch_major() == 9
    tp_group=make_tp_group(rank); tp_rank=dist.get_rank(tp_group); dp_rank=rank//TP
    model=types.SimpleNamespace(weight_seed=20260703,weight_scale=0.05,
        hidden=H,intermediate_hidden=I,block_n=128)
    buf=deep_gemm.get_symm_buffer_for_mega_moe(
        ep_group,E,1,TOPK,H,I,mma_type='fp8xmxfp4',activation='swiglu')
    try:
        l1,l2=_prepare_weights(model,E//WORLD,rank)
        torch.manual_seed(20260805)
        rw=(torch.randn(E,H,device='cuda',dtype=torch.bfloat16)*0.05).contiguous()
        torch.manual_seed(17000+dp_rank*1000003)
        full=torch.randn(M,H,device='cuda',dtype=torch.bfloat16)
        partial=(full.float()/TP).to(torch.bfloat16).contiguous()
        local=torch.empty(1,H,device='cuda',dtype=torch.bfloat16)
        local_y=torch.empty_like(local); out=torch.empty_like(full)
        logits=torch.empty(1,E,device='cuda',dtype=torch.bfloat16)
        stats=torch.zeros(E//WORLD,device='cuda',dtype=torch.int32)
        global_token=torch.tensor([dp_rank*M+tp_rank],device='cuda',dtype=torch.int64)
        slots=torch.arange(TOPK,device='cuda',dtype=torch.int64)
        target_rank=slots
        offset=(global_token[:,None]*TOPK+slots[None,:]*7)%(E//WORLD)
        balanced_idx=(target_rank[None,:]*(E//WORLD)+offset).contiguous()
        balanced_w=torch.full((1,TOPK),1/TOPK,device='cuda',dtype=torch.float32)

        def frontend_core(local_x, local_out, global_out):
            stats.zero_()
            if os.environ.get('DG_W4A8_INT','0')!='0':
                deep_gemm._C.qoq_router_quant_topk(
                    local_x,rw,logits,buf.x[:1],buf.x_sf[:1],buf.topk_idx[:1],buf.topk_weights[:1])
            else:
                deep_gemm._C.mxfp4_router_quant_topk_split(
                    local_x,rw,logits,buf.x[:1],buf.x_sf[:1],buf.topk_idx[:1],buf.topk_weights[:1])
            buf.topk_idx[:1].copy_(balanced_idx); buf.topk_weights[:1].copy_(balanced_w)
            deep_gemm.mxfp4_mega_moe(local_out,l1,l2,buf,
                cumulative_local_expert_recv_stats=stats,
                recipe=(128,128,128),activation_clamp=10.0,
                attn_tp_size=TP,attn_tp_group=tp_group,broadcast_output=False)
            dist.all_gather_into_tensor(global_out,local_out,group=tp_group)

        work=partial.clone(); dist.reduce_scatter_tensor(local,work,group=tp_group)
        frontend_core(local,local_y,out)
        torch.cuda.synchronize(); dist.barrier(group=ep_group)

        ar=partial.clone(); dist.all_reduce(ar,group=tp_group)
        ref_local=ar[tp_rank:tp_rank+1].contiguous()
        ref_local_y=torch.empty_like(ref_local); ref=torch.empty_like(full)
        frontend_core(ref_local,ref_local_y,ref)
        torch.cuda.synchronize(); dist.barrier(group=ep_group)

        diff=(out.float()-ref.float()).abs(); max_abs=diff.max()
        cos=torch.nn.functional.cosine_similarity(out.float(),ref.float(),dim=-1).min()
        exact=torch.tensor(float(torch.equal(out,ref)),device='cuda')
        dist.all_reduce(max_abs,op=dist.ReduceOp.MAX,group=ep_group)
        dist.all_reduce(cos,op=dist.ReduceOp.MIN,group=ep_group)
        dist.all_reduce(exact,op=dist.ReduceOp.MIN,group=ep_group)
        if rank==0:
            mode='QoQ' if os.environ.get('DG_W4A8_INT','0')!='0' else 'scaled-MXFP4'
            print(f'CORRECT mode={mode} exact={int(exact.item())} max_abs={max_abs.item():.8g} cos_min={cos.item():.10f}',flush=True)
        assert cos.item()>0.999
    finally:
        buf.destroy(); dist.destroy_process_group()

if __name__=='__main__':
    torch.multiprocessing.spawn(worker,nprocs=WORLD,join=True)
