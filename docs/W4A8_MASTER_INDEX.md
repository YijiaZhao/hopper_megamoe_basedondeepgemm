# W4A8 MegaMoE 全套资料总索引（SM90 / H20）

> 一份把 W4A8-int + MXFP4 MegaMoE 的**代码、仓库、设计、benchmark、环境、机器**串起来的入口文档。
> 深度设计细节见 [`W4A8_INT.md`](W4A8_INT.md)；早期 handoff 见 [`W4A8_INT_HANDOFF.md`](W4A8_INT_HANDOFF.md)；
> 计算流程图见 [`w4a8_int_flow.html`](w4a8_int_flow.html)。最后更新 2026-07-24。

---

## 0. 一句话

在 DeepGEMM 底座上做的 **SM90 融合 MoE 内核**，支持两条量化路径——
**W4A8-int（int4 权重 + int8 激活，QoQ 两级 + zero-point + SHIFTXOR 解码）** 与
**MXFP4×FP8（e2m1 权重 + e8m0 scale）**——把 dispatch + GG1 + SwiGLU + requant + GG2 + combine
**融进 2 个 kernel**，跨卡通信藏进计算。核心价值：**8 卡 EP 下 int4-SX 比 FlashInfer grouped-GEMM+通信快 2~16%**。

---

## 1. Git 仓库

| 仓库 | 用途 | 远端 | 关键分支 |
|---|---|---|---|
| **hopper_megamoe_basedondeepgemm** | MegaMoE 内核主仓 | `github` = github.com/YijiaZhao/hopper_megamoe_basedondeepgemm<br>`origin` = github.com/YijiaZhao/deepgemm_w4a8_mega_moe_on_hopper | `w_int4_a_int8_qserve_and_w_mxfp4_a_fp8`（已推 github + main） |
| **tensorcore_ptx_example** | WGMMA/解码 PTX 教学 + 逐bit验证 | github.com/YijiaZhao/tensorcore_ptx_example | main（快速解码示例本地已 commit，未 push） |

**本地检出**：`~/Downloads/agent/cc/hopper_megamoe_basedondeepgemm`（主仓，工作树干净）。
**tensorcore repo**：`computelab:/home/scratch.kimiz_gpu_2/docker_v/agent_work_space/tensorcore_ptx_example`。

### 关键提交（主仓 W4A8 相关）
| hash | 内容 |
|---|---|
| `ec8904d` | **int4-SX**：SHIFTXOR 解码（shift+mask + z-subtract，s2 延到 promote），推荐 tier |
| `7b591e1` | **mxfp4**：rel-LUT，flash M1 默认启用（per-K32 promote 最优 M2-127，rel-LUT M128+） |
| `545764e` | 放开 M-major SF 的 16B TMA 对齐 assert → 支持 **客户 I=640 / N=1280**（IH%256≠0） |
| `4076810` | 同步 .10 运行树的 occ2 gate（新增 `DG_MXFP4_OCC2` dormant 路径） |

> **重要一致性事实**：跑出所有 benchmark 的运行树是 `.10/.8:/root/fac/megamoe/DG_main_trial`（**非 git 管理**）。
> 2026-07-21 核对：`deep_gemm/include` + `csrc` 全 94 个源文件**与本地 git 字节级一致**（那 1 处 occ2 差异已并入 `4076810`）。→ **github 上的代码 == 跑出这些数的代码**。

---

## 2. 代码布局（关键文件）

```
deep_gemm/include/deep_gemm/
  impls/sm90_mxfp4_mega_moe.cuh          ← 主内核 (L1/L2 融合, 3200+行, int4+mxfp4 两路)
  quantization/int4_dequant.cuh          ← int4→int8 解码 (SHIFTXOR: shift/mask/vsub4)
  quantization/mxfp4_dequant.cuh         ← e2m1→fp8 解码 (prmt 查寄存器 LUT + e8m0 折叠)
csrc/jit_kernels/impls/sm90_mxfp4_mega_moe.hpp   ← host 侧 JIT launcher (读 env → 模板参数)
deep_gemm/mega/__init__.py              ← Python API (block_n 选择, weight transform)
tests/
  bench_mxfp4_mega_moe_sm90.py           ← 主 bench (int4/mxfp4, EP8 多进程, M≤1024)
  bench_int_prologue.py                  ← 大 M (≥1280) PRE 档 bench
  test_int4_mega_moe_correctness.py      ← 正确性门 (逐位/cosine)
  debug_int_l2_instrument.py             ← L2 int 调试 (Python 读 buffer 重建对比)
docs/W4A8_INT.md                         ← 设计 + 18点表 + 第6章 7点对比 (方法学+表)
```

编译走 **NVRTC 运行时 JIT**（读 .cuh，`/root/.deep_gemm` 缓存）；改 .cuh 无需重编，改 csrc 跑 `develop.sh`。

---

## 3. 设计要点（详见 W4A8_INT.md）

- **量化格式**：
  - **int4-SX (QoQ 两级)**：s2 int per-K128 组 + z uint4 + s1 bf16 per-channel + zero point。解码 = shift+mask+vsub4 减 z（无 xor，名字是历史遗留）。s2 延到 promote（组=128 对齐 BLOCK_K=128，每 K128 promote 一次）。
  - **mxfp4**：e2m1 权重 + e8m0 scale per-K32。解码 = prmt 从寄存器常量表（幅值非线性 {0,.5,1,1.5,2,3,4,6}）+ e8m0 折进 fp8 指数。**每 K32 promote（int4 的 4×）→ 比 int4 慢 ~16%**。
- **融合内核（2 kernel，warp-specialized）**：dispatch warp（NVLink 拉 token）/ TMA producer warp / 专职 decode warp / math warpgroup（WGMMA + SwiGLU/requant/combine epilogue）。三级 pingpong + mbarrier full/empty/decoded 流水。**通信（dispatch/combine）与 GEMM 同时在飞 = 护城河**。
- **两个 GEMM 非等量**：GG1(gate+up, N=2I=1280) FLOP = **2×** GG2(down, K=I=640)。bench 权重形状实测确认（`2*intermediate_hidden`）。
- **根因（为何单 GEMM 比 FlashInfer 慢）**：融合内核 smem 被 dispatch ring/coeff/SFA/SFB/decode 吃掉 → weight prefetch 只 3-4 级（需 8）→ latency-bound。FlashInfer 去融合让 GEMM 独占 smem 深 prefetch 更快，但没有跨卡通信融合。

---

## 4. 环境开关（进 JIT cache key）

```bash
# int4-SX 全套 (推荐 tier)
DG_W4A8_INT=1 DG_W4A8_INT_L2=1 DG_W4A8_INT_QOQ=1 \
DG_W4A8_INT_QOQ_ZP=1 DG_W4A8_INT_QOQ_ZP_SHIFTXOR=1 DG_W4A8_INT_SMALLM_OCC2=1

# mxfp4: 去掉全部 DG_W4A8_INT*  (DG_W4A8_INT=0)
# 大 M (≥1280) int PRE 档: 换 DG_W4A8_INT_PRE=1 + bench_int_prologue.py
# 通用: DG_BALANCED_ROUTING=1 (均衡路由) / DG_WALL_BENCH=1 (wall-clock 计时) / DG_MXFP4_OCC2=1 (mxfp4 也开 occ2)
```

---

## 5. 构建与运行（H20 sm_90a）

```bash
# 锁频
nvidia-smi -lgc 1830,1830
# 主 bench (int4-SX, EP8 8卡, 客户形状)
env <上面 int4 全套 env> DG_BALANCED_ROUTING=1 \
  python3 tests/bench_mxfp4_mega_moe_sm90.py --num-processes 8 \
  --hidden 4096 --intermediate-hidden 640 --num-experts 128 --num-topk 6 \
  --block-n 128 --batches 4 8 16 32 44 48 64
# 单卡: --num-processes 1 ; mxfp4: 去掉 INT env
# 正确性: python3 tests/test_int4_mega_moe_correctness.py  (期望 cosine=1.0)
```

---

## 6. Benchmark 结论（客户形状 H4096/I640/E128/topk6，.10 同机统一口径）

**8卡 EP8（含通信）· wall · median-of-5 · us**

| m | int4-SX | mxfp4 | FI GG默认 | 2×AR | **FI GG+2AR** |
|---|---|---|---|---|---|
| 4 | **106** | 118 | 98 | 28 | 126 |
| 8 | **105** | 118 | 96 | 28 | 124 |
| 16 | **110** | 119 | 95 | 28 | 123 |
| 32 | **126** | 139 | 95 | 34 | 129 |
| 44 | **142** | 161 | 114 | 35 | 148 |
| 48 | **145** | 164 | 132 | 35 | 167 |
| 64 | **146** | 165 | 135 | 36 | 171 |

- **8卡：int4-SX 全程赢 FI GG+2×AR 2~16%**（通信融合护城河）；mxfp4 仅打平（per-K32 promote 拖慢）。
- **单卡：FlashInfer 更快**（去融合 GEMM 快，MegaMoE 无通信可省）。
- **必须用 int4-SX，不是 mxfp4**。
- 方法学（同机/均衡/wall/median-of-5/FI 取库默认非 autotune）见 `W4A8_INT.md` 第 6.1 节。

---

## 7. 正确性验证

- int4-SX：L1 cosine 1.0、L2 修复后 flash+pro 各 36 case（M=1..1024）全 PASS、大 M smoke PASS。**数值有效、可上生产**。
- mxfp4：数值有效。
- **mxfp4-K128 探针**（`DG_MXFP4_PROBE_K128_PROMO` 编译宏）：cosine~0.74 **数值无效**，只测速度上限，不可用。
- 定位手法：`tests/debug_int_l2_instrument.py`——Python 读 `buffer.l2_acts` 按 fp8/int8 两种解释重建对比锁真凶。

---

## 8. 机器 / 运行树

- **H20-3e 四机池**：`root@10.6.131.7/8/9/10`（各 8×H20-3e，NVLink，sm_90a）。锁 1830。
- **运行树**：`.10/.8:/root/fac/megamoe/DG_main_trial`（dg_dev 容器，torch2.11+cu130，NVRTC JIT）。
- **.10 = 当前最干净基准机**（清场后 benchmark 出自这里）。FlashInfer PR#3738（v0.6.15 editable，`/root/flashinfer_pr3738`）+ SGLang custom AR（`ali_sglang_qwe3_235b` 容器）也在 .10。
- 跑 FI 需 `FLASHINFER_DISABLE_VERSION_CHECK=1`；.9/.10 dg_dev 容器无外网。

---

## 9. 相关：tensorcore_ptx_example（解码 PTX 教学）

`computelab:/home/scratch.kimiz_gpu_2/docker_v/agent_work_space/tensorcore_ptx_example`，github `YijiaZhao/tensorcore_ptx_example`。
W4A8 相关的**快速解码示例**（H20 实测各 512/512 逐bit精确）：
- `hopper/prmt_decode_mxfp4_to_e4m3_sm90/`：e2m1→fp8，`prmt.b32` 查寄存器 LUT，2 条 permute 解 8 个（MegaMoE 热路径）。
- `hopper/mask_decode_int4_to_int8_sm90/`：int4→int8，shift+mask 解 8 个（QoQ 再 vsub4 减 z）。
- 关键认知：**解码在 per-MAC 热路径，逐个 cvt 会成瓶颈 → 用 prmt/位运算批 8 个**；epilogue 的 fp8 量化（O(M·N)，量少）才用 cvt。

---

## 10. 待办 / 边界

- 代码已在 github（main + 分支）；tensorcore 快速解码 commit 未 push。
- 客户同型 HW（H20-96G 4.0）复现未做（computelab 节点排队 + 需搭环境；相对结论已锁定，不影响谁快）。
- PRE/QoQ/ZP 尚未并入 8 卡主 gate 矩阵；prepack API 待产品化（现在 tests/ 量化器里）。
- swapAB 段要求 block_n=128 prepack；PRE 段建议 block_n=256。
