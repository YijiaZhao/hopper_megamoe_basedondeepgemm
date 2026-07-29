# Kernel Factory 优化记录（2026-07-24）

本文记录 NVIDIA Kernel Factory 在 H20-3e 上搜索出的 MXFP4 与 int4-SX
微内核策略，以及移植进完整 8-rank MegaMoE 后的七点结果。

## 1. 测试口径

- 硬件：8× NVIDIA H20-3e，锁频 1830 MHz。
- 形状：`H=4096, I=640, E=128, topk=6, block_n=128`。
- 路由：`DG_BALANCED_ROUTING=1`。
- 计时：`DG_WALL_BENCH=1`，每点 warmup 5 次、测量 30 次。
- int4 表取 6 轮中位数；KF-MXFP4 表取 3 轮中位数。
- 主循环 `BLOCK_K=128`，由四条 K32 WGMMA/IGMMA atom 构成。

## 2. MXFP4 scaled-PRMT

Kernel Factory campaign：`2jbvesebwh3cs104gk2xn138f4`。

策略：

1. MXFP4 的量化 scale group 仍为 K32。
2. 使用寄存器 LUT + PRMT，一次解码 8 个 E2M1 nibble。
3. 将每个 K32 的 E8M0 scale 直接折进生成的 E4M3 byte。
4. 解码值统一放大 `×256`，避免 FP8 subnormal/下溢；promotion 时统一乘 `1/256`。
5. L1 的四条 K32 WGMMA 在一个 K128 内共用 accumulator，只 promotion 一次。
6. L2 因 activation scale 在 K64 边界变化，每个 K128 使用两条链、promotion 两次。

实验开关：

```bash
DG_MXFP4_REL_LUT=1 DG_MXFP4_ABS_SCALE256=1
```

当前 scaled LUT 的有效 E8M0 code 范围为 `121..125`；本次 benchmark 权重实测
为 `121..123`。超出范围的 checkpoint 在加入通用 fallback 前不得启用此路径。

正确性：客户形状 M=8/64 对 exact MXFP4 reference 通过，
`cosine_min=0.999942`，`cosine_mean=0.999995`。

## 3. int4-SX direct-nibble

Kernel Factory campaign：`e61eqhgb2n0rqe1smbwsd7yshg`。

微基准赢家 `int4_sx_int8_wgmma_signed_v10`，七个 workload 全部通过，
geomean `31.03 us`，相对 PyTorch reference speedup `5.678×`。

可移植策略不是替换完整 MegaMoE，而是改变 RF 权重预打包：

- 原 SHIFTXOR grouped-PRMT 布局需要两条 PRMT 将 nibble 展开到 byte lane；
- direct-nibble 布局保留 RF-fragment reordered Marlin word；
- kernel 直接执行：

```cpp
w0 = (packed >> 4) & 0x0F0F0F0F;
w1 = packed & 0x0F0F0F0F;
d0 = __vsub4(w0, zero_points);
d1 = __vsub4(w1, zero_points);
```

这样删除两条 PRMT 及其临时寄存器，zero point、s1/s2、K128 int32 accumulator
和 promotion 语义保持不变。

实验开关：

```bash
DG_W4A8_INT_DIRECT_NIBBLE=1
```

该开关会同时改变离线 prepack 与 JIT decode，不能只开其中一侧。当前默认关闭；
fragment decode 已与 grouped-PRMT 路径逐 word 对比一致，但进入默认部署前仍需使用
真实 QoQ+ZP checkpoint 完成完整生产 correctness gate。

## 3.5 int4-SX ZSUB_XOR（uint8 域 + XOR，2026-07-29）

来源：LiquidGEMM（arXiv 2509.01229，ByteDance Seed + 上交）的 LiquidQuant 思路——
**留在 uint8 域算，最后一条 XOR 翻 MSB 换回 int8 补码**，从而避开非原生的字节级减法。

原 SHIFTXOR 的 z-subtract 用 `__vsub4`（模拟指令）。改为：

```cpp
const uint32_t zz128 = 0x80808080u - zz;   // (128-z) splat；z<=15 故不借位
return make_uint2((w0 + zz128) ^ 0x80808080u,
                  (w1 + zz128) ^ 0x80808080u);
```

每字节 `w4 + (128-z) ∈ [113,143]`，**恒在 [0,255] 内 ⇒ 跨 lane 不进位**，
所以可用普通 32-bit 加法替掉 `__vsub4`。

- **SASS 实测（sm_90a，每 8 元素）**：算术指令 **13 → 8（-38%）**
  （7 LOP3 + 2 IMAD.IADD → 4 LOP3）。
- **等价性**：合法 `z ∈ [0,15]` × 4096 图案 × 128 线程穷举，与 `__vsub4` **逐 bit 零失配**。
- **成立条件**：`zz` 每字节 ≤ 128（合法 zero point 必然满足）。垃圾 coeff（如在
  MXFP4 编码权重上跑 perf bench）会超出该条件，此时两式结果不同——**只影响数值，不影响计时**。

实验开关（仅在 SHIFTXOR 下生效）：

```bash
DG_W4A8_INT_ZSUB_XOR=1
```

**正确性 gate 现状**：`test_int4_mega_moe_correctness.py` 的 **QoQ+ZP 档本身即失败
（baseline cosine≈0.0001，与本改动无关；纯 int4+L2 档 cosine=1.0 PASS）**，对应
`W4A8_INT.md` 已记的待办「PRE/QoQ/ZP 尚未并入 8 卡主 gate 矩阵」。
本路径进默认前，必须先用真实 QoQ+ZP checkpoint 补齐该 gate。

## 4. 完整 MegaMoE 七点表

单位：微秒，数值越小越好。

单位：微秒，数值越小越好。前三列为 2026-07-24 KF 批次；ZSUB_XOR 列为 2026-07-29 单独批次
（配对 baseline 见表下，跨批次绝对值不可直接比，只看各自 delta）。

| M | 原 int4-SX | direct int4-SX | scaled MXFP4 | int4-SX+ZSUB_XOR★ | 最快 |
|---:|---:|---:|---:|---:|---|
| 4  | 106.3 | 107.0 | **103.7** | 108.9 | scaled MXFP4 |
| 8  | 106.0 | **104.2** | 105.5 | 105.5 | direct int4-SX |
| 16 | 108.8 | **104.7** | 107.9 | 110.1 | direct int4-SX |
| 32 | 124.5 | **119.2** | 120.0 | 122.9 | direct int4-SX |
| 44 | 141.1 | 137.7 | **132.5** | 139.5 | scaled MXFP4 |
| 48 | 143.2 | 140.1 | **134.7** | 143.1 | scaled MXFP4 |
| 64 | 143.8 | 140.8 | **136.2** | 142.2 | scaled MXFP4 |

> 「最快」列写全名：赢 M=8/16/32 的是 **direct** int4-SX（不是原 SHIFTXOR），
> 赢 M=4/44/48/64 的是 **scaled** MXFP4。**原 int4-SX（当前默认路径）七点全输。**
>
> ★ ZSUB_XOR 同批次的 SHIFTXOR baseline = 110.2 / 104.3 / 112.2 / 126.1 / 141.7 / 144.8 / 145.7。
> 相对自身 baseline：**M≥32 快 1.2~2.5%**（ON/OFF 三跑区间不重叠，非噪声）；M≤16 在噪声内。
> 它改的是原 SHIFTXOR 的 z-subtract，与 direct-nibble 的「删两条 PRMT」**互不冲突，可叠加**（未测）。

direct int4-SX 在 M=8..64 相对原 SHIFTXOR 快约 `1.7%..4.3%`，M4 回退约
`0.7%`。与 scaled MXFP4 比较，direct int4-SX 赢 M=8/16/32，scaled MXFP4 赢 M=4/44/48/64。

## 5. 当前结论

- MXFP4 的 group 没有改成 128：仍为 K32，只是在 K128 主循环内合并结算。
- int4-SX 的 s2/z group 为 K128，天然适合四条 K32 IGMMA 链式累加。
- M=8..32 对整数 decode 指令数敏感，direct-nibble 最有效。
- M≥44 时 scaled MXFP4 的 scale fold 和更轻 promotion 仍占优。
- 两条新路径均为实验开关，默认路径不受影响。
