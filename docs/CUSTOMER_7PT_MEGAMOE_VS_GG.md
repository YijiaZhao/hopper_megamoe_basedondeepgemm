# 客户形状 7 点表：MegaMoE vs FlashInfer grouped-GEMM + 2×AR

**形状** H=4096, I=640 (N=1280), E=128, topk=6, m ∈ {4, 8, 16, 32, 44, 48, 64}
**测量** 同一台机 .10（8×H20-3e @1830，清场）· 均衡路由 · CUDA-event wall-clock · median-of-5
（完整方法学/复现命令见 `W4A8_INT.md` 第六章）

> 注：这不是 18 点表（那是 `W4A8_INT.md` 第三节的 Flash/Pro 形状 H4096-I2048 / H7168）。
> 本文件是**客户形状 I=640/E=128 的 7 点对比**。

## 表一：单卡 1-rank（无跨卡通信）· us

外部GG = 第三方自测纯 2-GEMM（H20-96G）；其余我们机 H20-3e。

| m | 外部GG(w4a8) | 外部GG(wfp4afp8) | FI默认 | FI autotune | MegaMoE int4-SX | MegaMoE mxfp4 |
|---|---|---|---|---|---|---|
| 4  | 120 | 90  | 112 | 145 | 145 | 153 |
| 8  | 184 | 167 | 181 | 134 | 201 | 232 |
| 16 | 310 | 292 | 307 | 201 | 331 | 387 |
| 32 | 396 | 410 | 400 | 256 | 417 | 488 |
| 44 | 398 | 400 | 392 | 256 | 418 | 487 |
| 48 | 398 | 620 | 397 | 254 | 422 | 490 |
| 64 | 400 | 570 | 401 | 262 | 435 | 500 |

## 表二：8卡 EP8（含跨卡通信）· wall · median-of-5 · us

| m | MegaMoE int4-SX | MegaMoE mxfp4 | FI GG默认 | FI GG autotune | 2×AR | FI默认+2AR | FI autotune+2AR |
|---|---|---|---|---|---|---|---|
| 4  | **106** | 118 | 98  | 143 | 28 | 126 | 171 |
| 8  | **105** | 118 | 96  | 135 | 28 | 124 | 163 |
| 16 | **110** | 119 | 95  | 136 | 28 | 123 | 164 |
| 32 | **126** | 139 | 95  | 135 | 34 | 129 | 169 |
| 44 | **142** | 161 | 114 | 135 | 35 | 148 | 170 |
| 48 | **145** | 164 | 132 | 136 | 35 | 167 | 171 |
| 64 | **146** | 165 | 135 | 135 | 36 | 171 | 171 |

**列口径**：MegaMoE int4-SX/mxfp4 = 真 8 卡含融合通信；FI GG = 单卡 experts=16 shard 纯 GEMM 无通信；
FI 默认 = 库默认 tactic（小 M 最快），FI autotune = 显式 AutoTuner（小 M 反而慢）；
2×AR = SGLang custom allreduce ×2（L1前+L2后）；+2AR 两列 = FI GG 加通信后与 MegaMoE 的公平对比。

## 结论

- **8卡 EP8：MegaMoE int4-SX 全程赢 FI GG+2×AR 2~16%**（融合通信护城河，把 28~36us 的 dispatch/combine 藏进计算）。
- **单卡：FlashInfer 更快**（去融合，GEMM 独占 smem 深 prefetch；MegaMoE 融合内核单卡吃 per-GEMM 惩罚且无通信可省）。
- **必须用 int4-SX，不是 mxfp4**：mxfp4 每 K32 promote（int4 的 4×）拖慢 GEMM，仅与 GG+2×AR 打平。
