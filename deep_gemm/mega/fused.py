import types
import torch
import torch.distributed as dist
from .. import _C
from ..utils import align

_SM90_FUSED_LAYOUT_ATTR = "_deep_gemm_fp4_h20_fused_layout"

def _interleave(t, gran=8):
    g,n,*rest=t.shape; half=n//2
    gate=t[:,:half].reshape(g,half//gran,gran,*rest); up=t[:,half:].reshape(g,half//gran,gran,*rest)
    return torch.stack([gate,up],dim=2).reshape(g,n,*rest).contiguous()

def _braid(fused):
    e,n,k=fused.shape; rows=fused.view(e,n,k//80,80).clone(); packed=rows[...,:64].view(e,n,k//80,16,4)
    codes=torch.cat(((packed>>4)&15,packed&15),dim=-1); mag=codes&7; sign=codes>>3
    bs=torch.stack((sign[...,4],sign[...,0],sign[...,5],sign[...,1],sign[...,6],sign[...,2],sign[...,7],sign[...,3]),dim=-1)
    nib=mag|(bs<<3); rows[...,:64]=(nib[...,0::2]|(nib[...,1::2]<<4)).reshape(e,n,k//80,64)
    return rows.view(e,n,k).contiguous()

def transform_mxfp4_weights_for_mega_moe_fused(l1,l2):
    from ..quantization_mxfp4_fused import mxfp4_row_reference_exponent,mxfp4_scale_to_relative_index,mxfp4_scale_to_tile_major,mxfp4_fuse_packed_with_scale_tile_major
    p1,s1=l1; p2,s2=l2; p1,s1=_interleave(p1),_interleave(s1)
    def prep(p,s):
        er=mxfp4_row_reference_exponent(s); idx=mxfp4_scale_to_relative_index(s,er); tm=mxfp4_scale_to_tile_major(idx,block_n=256,block_k=128); f=_braid(mxfp4_fuse_packed_with_scale_tile_major(p,tm,block_k=128)); return f,tm,torch.exp2(er.float()).contiguous()
    return prep(p1,s1),prep(p2,s2)

def transform_qoq_weights_for_mega_moe_fused(l1,l2):
    from ..quantization_qoq_fused import qoq_meta_to_tile_major,qoq_fuse_packed_with_meta_tile_major
    def prep(p,s2,z,s1):
        tm=qoq_meta_to_tile_major(s2,z,block_n=256); return qoq_fuse_packed_with_meta_tile_major(p.contiguous(),tm),tm,s1.float().contiguous()
    p,s2,z,s1=l1; return prep(_interleave(p),_interleave(s2),_interleave(z),_interleave(s1.unsqueeze(-1)).squeeze(-1)),prep(*l2)

class FusedSymmBuffer:
    def __init__(self, group, num_experts, max_tokens, topk, hidden, intermediate):
        self.group=group; self.num_experts=num_experts; self.num_max_tokens_per_rank=max_tokens; self.num_topk=topk; self.hidden=hidden; self.intermediate_hidden=intermediate
        n,slice_fn=_C.get_symm_buffer_size_for_fused_mega_moe(group.size(),num_experts,max_tokens,topk,hidden,intermediate)
        from torch.distributed._symmetric_memory import empty as symm_empty, rendezvous
        self.buffer=symm_empty(n,dtype=torch.int8,device='cuda'); self.handle=rendezvous(self.buffer,group=group); self.buffer.zero_(); group.barrier(); torch.cuda.synchronize()
        self.x,self.x_sf,self.topk_idx,self.topk_weights,self.l1_acts,self.l1_acts_sf,self.l2_acts,self.l2_acts_sf=slice_fn(self.buffer)
    def destroy(self): self.handle=None; self.buffer=None; self.group=None

def get_fused_symm_buffer_for_mega_moe(group,num_experts,max_tokens,topk,hidden,intermediate):
    return FusedSymmBuffer(group,num_experts,max_tokens,topk,hidden,intermediate)

def mxfp4_mega_moe_fused(y,l1,l2,b,cumulative_local_expert_recv_stats=None,activation_clamp=10.0,fast_math=True):
    w1,sf1,rs1=l1; w2,sf2,rs2=l2
    _C.mxfp4_mega_moe_fused(y,(w1,sf1),(w2,sf2),cumulative_local_expert_recv_stats,rs1,rs2,b.buffer,b.handle.buffer_ptrs,b.group.rank(),b.num_max_tokens_per_rank,b.num_experts,b.num_topk,activation_clamp,fast_math)

def qoq_mega_moe_fused(y,l1,l2,b,cumulative_local_expert_recv_stats=None,activation_clamp=10.0,fast_math=True):
    w1,sf1,rs1=l1; w2,sf2,rs2=l2
    _C.qoq_mega_moe_fused(y,(w1,sf1),(w2,sf2),cumulative_local_expert_recv_stats,rs1,rs2,b.buffer,b.handle.buffer_ptrs,b.group.rank(),b.num_max_tokens_per_rank,b.num_experts,b.num_topk,activation_clamp,fast_math)
