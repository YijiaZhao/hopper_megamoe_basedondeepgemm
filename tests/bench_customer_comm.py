"""Measure TP-group ReduceScatter followed by 8-rank AllGather."""
import argparse, os, statistics, torch, torch.distributed as dist

TOKENS=[4]
H=3072

def make_tp_group(rank,world,tp):
    mine=None
    for first in range(0,world,tp):
        ranks=list(range(first,first+tp)); g=dist.new_group(ranks)
        if rank in ranks: mine=g
    return mine

def worker(local_rank,world,tp,warmup,iters):
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl',init_method='tcp://127.0.0.1:29917',rank=local_rank,world_size=world,device_id=torch.device(f'cuda:{local_rank}'))
    tp_group=make_tp_group(local_rank,world,tp)
    dp=world//tp
    for mt in TOKENS:
        if mt < world:
            # Equal-slot padding: one local slot/rank, only mt slots are real.
            local_slots=1; group_slots=tp
        else:
            assert mt%world==0
            local_slots=mt//world; group_slots=local_slots*tp
        rs_in=torch.randn(group_slots,H,device='cuda',dtype=torch.bfloat16)
        rs_out=torch.empty(local_slots,H,device='cuda',dtype=torch.bfloat16)
        ag_out=torch.empty(local_slots*world,H,device='cuda',dtype=torch.bfloat16)
        def run():
            dist.reduce_scatter_tensor(rs_out,rs_in,group=tp_group)
            dist.all_gather_into_tensor(ag_out,rs_out,group=dist.group.WORLD)
        for _ in range(warmup): run()
        torch.cuda.synchronize(); dist.barrier()
        times=[]
        for _ in range(iters):
            dist.barrier(); st=torch.cuda.Event(True); en=torch.cuda.Event(True)
            st.record(); run(); en.record(); en.synchronize(); times.append(st.elapsed_time(en)*1000)
        med=statistics.median(times)
        val=torch.tensor(med,device='cuda',dtype=torch.float64)
        vals=[torch.zeros_like(val) for _ in range(world)]; dist.all_gather(vals,val)
        if local_rank==0:
            xs=[v.item() for v in vals]
            print(f'COMM TP={tp} M_total={mt} local_slots={local_slots} RS_AG_mean_us={statistics.fmean(xs):.3f} RS_AG_max_us={max(xs):.3f} COMM_X2_max_us={2*max(xs):.3f}',flush=True)
    dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--tp',type=int,choices=[2,4],required=True); p.add_argument('--warmup',type=int,default=20); p.add_argument('--iters',type=int,default=200); a=p.parse_args()
    torch.multiprocessing.spawn(worker,args=(8,a.tp,a.warmup,a.iters),nprocs=8,join=True)
