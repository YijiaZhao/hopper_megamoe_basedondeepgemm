import os,sys,torch,torch.distributed as dist
ROOT='/raid/kimi/DeepGEMM_four_api_h20';sys.path.insert(0,ROOT)
import deep_gemm
from deep_gemm.quantization_mxfp4_fused import quantize_to_mxfp4
from deep_gemm.quantization_qoq_fused import quantize_to_qoq
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.quantization_qoq_fused import per_token_cast_to_int8
r=int(os.environ['RANK']);lr=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(lr);dist.init_process_group('nccl',device_id=torch.device(f'cuda:{lr}'))
E,H,I,K,M=384,3072,1280,8,1;le=E//8
torch.manual_seed(100+r);x=torch.randn(M,H,device='cuda',dtype=torch.bfloat16);w1=torch.randn(le,2*I,H,device='cuda',dtype=torch.bfloat16)*.05;w2=torch.randn(le,H,I,device='cuda',dtype=torch.bfloat16)*.05
try:
 for q in ('mxfp4','qoq'):
  b=deep_gemm.get_fused_symm_buffer_for_mega_moe(dist.group.WORLD,E,M,K,H,I)
  if q=='mxfp4':
   l1,l2=deep_gemm.transform_mxfp4_weights_for_mega_moe_fused(quantize_to_mxfp4(w1),quantize_to_mxfp4(w2));xi,xs=per_token_cast_to_fp8(x,use_ue8m0=False,gran_k=128);fn=deep_gemm.mxfp4_mega_moe_fused
  else:
   l1,l2=deep_gemm.transform_qoq_weights_for_mega_moe_fused(quantize_to_qoq(w1),quantize_to_qoq(w2));xi,xs=per_token_cast_to_int8(x,gran_k=128);xi=xi.view(torch.float8_e4m3fn);fn=deep_gemm.qoq_mega_moe_fused
  b.x[:M].copy_(xi);b.x_sf[:M].copy_(xs);s=torch.arange(K,device='cuda');b.topk_idx[:M].copy_((s*le+((r+s*7)%le))[None,:].to(torch.int64));b.topk_weights[:M].fill_(1/K);y=torch.empty(M,H,device='cuda',dtype=torch.bfloat16);fn(y,l1,l2,b);torch.cuda.synchronize();assert torch.isfinite(y).all();
  if r==0:print(q,'fused finite',y.abs().mean().item())
  b.destroy()
finally:dist.destroy_process_group()
