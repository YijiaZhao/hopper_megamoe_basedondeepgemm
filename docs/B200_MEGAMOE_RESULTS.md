# B200 MegaMoE results (8x B200, SM100)

Back to the [README](../README.md); Hopper results: [H20_MEGAMOE_RESULTS.md](H20_MEGAMOE_RESULTS.md).

Three fused MegaMoE kernels on Blackwell, all built on AichenF/DeepGEMM branch `megamoe_nvfp4_dev` (commit `70ff91b`):

| Kernel | Activations | Weights | MMA |
|---|---|---|---|
| W-MXFP4 x A-FP8 | E4M3, per-32 UE8M0 (L1 input and L2 input) | E2M1, per-32 UE8M0 | `tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale` (upstream DeepGEMM `sm100_fp8_fp4_mega_moe`) |
| W4A4 MXFP4 | E2M1, per-32 UE8M0 | E2M1, per-32 UE8M0 | `kind::mxf4.block_scale.block32` (new `sm100_fp4_fp4_mega_moe<32>`) |
| W4A4 NVFP4 | E2M1, per-16 UE4M3 | E2M1, per-16 UE4M3 | `kind::mxf4nvf4.block_scale.block16` (new `sm100_fp4_fp4_mega_moe<16>`) |

No software dequantisation anywhere: both operands' block scales are applied by the tensor core. The FP8-activation
kernel's FP4 weights are expanded to 8-bit smem containers by TMA (`16U4_ALIGN16B`), a requirement of the `mxf8f6f4` kind.

## Result table

Scale: Experts 384 (48 local per rank), Hidden 3072, Intermediate 1280, top-8, EP8; global M = 2 / 4 / 8 / 16 (one token per
active rank; M2 / M4 = ranks 0,4 / 0,1,4,5; M16 = 2 tokens per rank). Forced-balanced routing (`expert = s*48 + (g + 7s) % 48`,
weight 1/8), the same rule as the H20 table.

Kernel span on GPU 0, µs, median of the last 3 of 30 streamed replays (nsc-svg-slurm-1 job 2033326, SM clock locked at 1965 MHz):

| Global M | W-MXFP4 x A-FP8 | W4A4 MXFP4 | W4A4 NVFP4 | H20 MXFP4 Mega-only (1830 MHz, [H20 doc](H20_MEGAMOE_RESULTS.md)) |
|---|---:|---:|---:|---:|
| 2  | 39.4 | 35.4 | 36.1 | 38.8 |
| 4  | 43.0 | 39.9 | 40.2 | 47.4 |
| 8  | 49.2 | 45.3 | 45.6 | 56.7 |
| 16 | 55.8 | 53.9 | 51.6 | 76.2 |

Correctness: every cell has a relative RMSE of 0.23–0.25 % on every rank against a pure-torch reference (dequantised inputs,
fp32 matmul, bf16-rounded gate/up with clamp 10, SwiGLU, the kernel's own per-group requantisation rule for the intermediate).

Unoptimised starting point (the same kernels on their original barrier/pull path, event-based timing that also contains ~10 µs
of launch gaps): 56 / 57 / 62 / 68 (FP8 act), 56 / 57 / 64 / 67 (MXFP4), 59 / 60 / 67 / 73 (NVFP4).

## Measurement method

Per replay: 256 MiB L2 flush (memset) -> input copy into the symmetric buffer -> NCCL all-reduce on the stream (aligns the
ranks, no host sync or barrier between replays) -> MegaMoE kernel. 3 warm-up + 30 timed replays. The kernel duration comes
from the torch profiler (kineto) CUDA activity of the `mega_moe` kernel on GPU 0 (the analog of the H20 nsys kernel span);
report the median of the last 3. `tests/test_mega_moe.py ... --balanced --bench-stream 30` prints it as `KSPAN`.
Per-rank event timing (`STREAM`) is also printed; it includes launch gaps and is ~5–10 µs higher. The earlier per-iteration
host-barrier kineto method is 10–40 µs noisier for this collective kernel and was dropped.

GPU 0 starts first, so its span contains the wait for the slowest rank (~5–8 µs at M2); the H20 number has the same property.

## Optimisations (all three kernels, env knobs, defaults on)

| # | Change | Knob | Effect |
|---|---|---|---|
| 1 | Push dispatch + DONE flags (ported from the SM90 counters branch): the source rank takes one remote `atom.sys` ticket per routed row and writes token + SF words + top-k weight + metadata straight into the destination's fixed-stride pool; removes NVLink barrier #1, the count broadcast and the TMA pull loop. DONE is signalled by the warp whose pushed-row count completes the rank total (a CTA-count ticket serialised 148 CTAs, ~4.5 µs). Loaders no longer wait per-block arrival counts. | `DG_SM100_PUSH_DISPATCH` | M8 64 -> 55 (event timing) |
| 2 | Parity-rotated pool: drops NVLink barrier #3 after the workspace cleanup | `DG_SM100_NO_CLEAN_BARRIER` | part of 1 |
| 3 | DONE2: the CTA finishing the rank's last L2 task signals every rank; replaces NVLink barrier #2 before the combine | (with 1) | M2 -3 |
| 4 | Combine hidden split: (token, chunk) work items over all combine warps, 512 B chunks | `DG_SM100_COMBINE_SPLITS` (auto) | combine 4 -> 2.5 |
| 5 | BLOCK_K 128 -> 256 (FP8 act: 39.8 -> 39.4 at M2, within noise elsewhere): W4A4 packed rows are one 128 B swizzle atom; FP8-act rows span two atoms per stage (per-atom UMMA descriptors, two SF word planes) | heuristics | L1 task 5.3 -> 3.5 (W4A4), 4.9 -> 4.0 (FP8) |
| 6 | First pushed row prefetched into smem before the ticket round trip | (with 1) | ~1 |
| 7 | Static slots + early weight streaming: for tiny M every local expert owns one fixed (expert, n-block) -> SM slot set, empty experts are skipped at run time, and the weight loader starts streaming an expert's weights as soon as its live ticket count is nonzero (before DONE) | `DG_SM100_STATIC_SLOTS` (auto when `tokens * ranks <= BLOCK_M`) | M2 -4, M8 -4 |

Phase breakdown (`--phase-stamps`, globaltimer, GPU 0, M2 MXFP4, µs from kernel entry): pushes issued 4.4 | DONE signalled ~6-8 |
DONE acquired ~9 | first math task ~9.5 | last L1 end ~16.5 | last L2 end ~20.5 | all ranks' L2 done ~26 (wait on the slowest rank)
| combine end ~28. What is left: the dispatch chain is three serial NVLink round trips (ticket, remote write completion, DONE;
floor ~7 µs); an L1 task is bound by the per-SM L2->SMEM feed rate (~56 GB/s per SM for 196 KB of weights) and only 40 of 148
SMs have work at M2 (split-K or BLOCK_N = 64 would spread it); the FP8-activation kernel keeps half the pipeline depth because
its FP4 weights occupy 8-bit smem containers.

Why W4A4 is not much faster than FP8 activations here: weights are FP4 in all three, activation bytes are negligible at M <= 16,
and the 2x `mxf4` MMA rate is not on the critical path; NVFP4 additionally moves twice the scale-factor words.

## Code and reproduce

`sm100_b200/` holds the modified files (drop-in over AichenF/DeepGEMM `megamoe_nvfp4_dev` @ `70ff91b`):

| File here | Path in DeepGEMM |
|---|---|
| `sm100_fp4_fp4_mega_moe.cuh` (new), `sm100_fp8_fp4_mega_moe.cuh` | `deep_gemm/include/deep_gemm/impls/` |
| `ptx_tcgen05.cuh` (adds 2-CTA `mxf4` / `mxf4nvf4` wrappers), `ptx_ld_st.cuh` (CUDA 13.1 includes) | `deep_gemm/include/deep_gemm/ptx/tcgen05.cuh`, `ld_st.cuh` |
| `scheduler_mega_moe.cuh` (push mode, static slots), `layout_mega_moe.cuh` (DONE2 words) | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`, `layout/mega_moe.cuh` |
| `heuristics_mega_moe.hpp`, `sm100_fp4_fp4_mega_moe.hpp` (new), `sm100_fp8_fp4_mega_moe.hpp` | `csrc/jit_kernels/heuristics/`, `csrc/jit_kernels/impls/` |
| `apis_mega.hpp` (`fp4_fp4_mega_moe`, FP4 activation buffer layout), `mega___init__.py` | `csrc/apis/mega.hpp`, `deep_gemm/mega/__init__.py` |
| `test_mega_moe.py` (`--act-format fp8|mxfp4|nvfp4`, `--balanced`, `--active-ranks`, `--torch-ref`, `--bench-stream`, `--phase-stamps`, `--push`) | `tests/` |

Container: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` (CUDA 13.1, torch 2.11). Build: `bash develop.sh` (after
`rm -rf build dist *.egg-info deep_gemm/_C*.so` when host `.hpp` files change; header changes recompile through the JIT).

```bash
export PYTHONPATH=$PWD DG_JIT_CACHE_DIR=$PWD/jit_cache CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=10.0
# on 8 GPUs with the SM clock locked (Slurm: --gpu-freq=1965)
python tests/test_mega_moe.py --num-processes 8 --num-experts 384 --hidden 3072 --intermediate-hidden 1280 --num-topk 8 \
  --num-max-tokens-per-rank 128 --num-correctness-tests 0 --balanced --torch-ref --bench-stream 30 \
  --act-format mxfp4 --num-tokens 1 --active-ranks 0,4          # M2 (M4: --active-ranks 0,1,4,5; M8: drop it; M16: --num-tokens 2)
```

## Single-GPU Nsight Compute (what the kernel is bound by)

NCU cannot replay the 8-rank kernel (the DONE counters wait on peers), so the profile is a 1-rank run with all 48 experts
local, 1 token x top-8 (the same per-GPU GEMM work as global M8): `ncu --set full --kernel-name regex:mega_moe` on the
W4A4 MXFP4 kernel (`ncu_mxfp4_1rank_m8.ncu-rep`, NCU locks the clock to 1.13 GHz, kernel 53.5 µs there).

| Metric | Value |
|---|---|
| DRAM throughput / bytes read | 14 % of peak / 50.6 MB (= 8 experts x 5.9 MB of FP4 weights, read once) |
| L2 throughput / SM throughput / issue slots busy | 14 % / 12 % / 12 % |
| Schedulers: cycles with no eligible warp | 86 % (one instruction every 6.9 cycles per scheduler) |
| Warp stall sampling | barrier 40 %, long scoreboard 27 %, wait 10 %, branch resolving 5 % |
| Occupancy | 1 CTA/SM (512 threads, 128 regs/thread, 226 KB smem), 4 warps per scheduler, 1 wave |

Neither memory bandwidth nor tensor-core throughput is close to saturated; the kernel is latency-bound: warp-specialised roles
park at named barriers while the few active warps wait on TMA / mbarrier / flag round trips. That matches the phase stamps
(a 196 KB-weight L1 task takes 3.5 µs at ~56 GB/s per SM regardless of how many SMs are idle). The levers are therefore more
bytes per handshake (BLOCK_K 512 = two swizzle atoms for FP4), more SMs per expert at tiny M (split-K / BLOCK_N 64), and a
shorter dispatch chain, not more MMA throughput.

## Tried and kept off

- **Deterministic slots** (`DG_SM100_DETERMINISTIC_SLOTS=1`): for <= 2 tokens per rank the row of a routed token inside its expert
  block is `src_rank * 2 + token`, so the remote ticket is replaced by a direct write plus a fire-and-forget `red.or` valid bit
  (hole rows skipped in the L2 epilogue). Correct, but neutral: the ticket round trip was already hidden behind the first-row
  prefetch, and the release-scoped `red.or` adds a fence ("pushes issued" 4.5 -> 5.6 µs). The dispatch chain is bound by the
  remote row write + completion wait + DONE, not by the ticket.
- **Per-K-block L2 dependency wait** (L2 K block k waits only for the L1 output blocks that feed it): neutral here, the L1
  tasks of an expert finish within ~1 µs of each other; it only moved the wait inside the L2 task.
- **Concurrent SF / metadata stores with the row's bulk store**: within noise.
- **L2 slots offset by half a grid** (static slots; L2 tasks start at SM 74 so that at tiny M they sit on SMs without L1 work
  and their weights stream during L1): L2 task time drops (MXFP4 M2 4.4 -> 3.1 µs) but the L2 phase does not move (it is bound
  by the L1 -> L2 dependency, the activation loads and the remote BF16 epilogue, not by the weights). Kept (harmless).

## Run-to-run variance and what "GPU 0" means

Repeating a cell gives a ±1.5–2 µs spread on GPU 0. GPU 0 is usually the first rank to enter the kernel, so its span includes
the wait for the slowest rank; in runs where another rank enters first, GPU 0 reads 5–8 µs lower (e.g. MXFP4 M8 35.6 with the
other ranks at ~60). The rank-median of the per-rank spans is the more stable kernel-time estimate (MXFP4 M2 ~34, M8 ~35–43).
