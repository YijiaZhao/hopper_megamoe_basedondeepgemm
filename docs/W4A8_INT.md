# W4A8-int MegaMoE（SM90/H20）—— 设计、优化策略与运行指南

int4 权重 × int8 激活的 MoE 专家层 kernel,基于 DeepGEMM（deepseek-ai,MIT 协议,
经 AichenF fork 的 MegaMoE 分支）开发。本分支直接坐在 fork main（`54e2261`）之上,
上游 commit 历史完整保留,我们的全部改动为增量 commit,可逐个审阅
（`git log --oneline 54e2261..HEAD`）。

支持两种权重格式,按 checkpoint 自动选择：
- **GPTQ 式**：per-group-128 fp32 scale;
- **QServe（QoQ）式**：两级 scale + 可选 zero point（客户格式,本文档主线）。

> **命名说明**：int4 路径与 mxfp4 共享同一套 kernel 基建,实现位于
> `sm90_mxfp4_mega_moe.cuh` 等 mxfp4 命名的文件内,通过 `DG_W4A8_INT*` 环境变量启用;
> Python 入口 `deep_gemm.int4_mega_moe(...)`。不开任何 DG_W4A8 开关 = 原 MXFP4 路径。

---

## 一、量化格式（QoQ 两级 + zero point）

权重张量 (E, N, K)（E=expert,N=输出通道,K=归约维）,反量化语义：

```
w_int8 = (w4 − z) · s2          ← 整数域,bit-exact
输出   = t · s1 · I2F(int32累加)  ← 浮点只出现在这一步
```

| 量 | 粒度 | shape | dtype | 存放 |
|---|---|---|---|---|
| s1（一级 scale） | 每输出通道 | (E, N) | bf16（clamp ±112 防重构溢出） | coeff 字高 16 位,掩码读零成本 |
| s2（二级 scale） | 每通道每 128 个 K | (E, N, K/128) | 整数 u8（实际 ≤18） | coeff 字 byte0 |
| z（zero point） | 同 s2 | (E, N, K/128) | uint4 | coeff 字 byte1 |
| t（激活 scale） | 每 token | (T, 1) | fp32 | 随激活走 |

每 (通道, K128 组) 一个 4 字节 coeff 字 `[s2 | z | s1高16位]`。中间激活（L1→L2）
per-(token, 64 通道) 动态 int8 量化（amax/127,clamp ±126、−1→0 防 fp8-NaN 字节）。

**为什么 s2/z 必须是小整数**：`(w4−z)·s2` 就是还原出的原始 int8 权重——
s2 是"int8 细尺与 int4 粗尺的刻度比"（≤18）,z 是"粗尺 16 格里的零点位"（≤15）,
位宽由换算关系本身规定。这保证整数域全程精确、乘积恰好一个字节——后文所有
查表戏法的地基。

---

## 二、优化策略全记录

### 设计总纲（两条分水岭）

1. **scale 的归宿**：沿 K 变化的量（s2、z）必须在进累加器之前乘掉 → 折进解码;
   沿 K 不变的量（s1、t）留到累加之后 → epilogue 一次外积。
2. **两种瓶颈两种药**：发射受限（issue-bound）区间省指令;延迟受限区间加并行度
   填气泡;带宽受限区间省字节。每个 M 段先诊断属于哪种,再选武器。

### 小 M（M1-32,decode 延迟敏感区）

**1. swapAB**——token 少时权重占 WGMMA 的 M=64、token 走 N（n8/16/32/64 原子网格,
按 token 数选窄指令）。消灭 M=64 tile 的 padding：不这么做,M=1 时 63/64 的
tensor core 槽位是空转。

**2. 寄存器解码直喂 RS WGMMA**——4bit 权重 TMA 进 packed scratch,math warp 解进
寄存器直接作为 WGMMA 的寄存器操作数,零 smem 往返;解码指令藏在上一批 WGMMA 的
异步阴影里（tensor core 与 CUDA core 是不同执行单元,天然并行）。

**3. 缩放 LUT 折叠**——解码不做逐权重乘法。同一 K128 组内 s2/z 是常数、4bit 码
只有 16 种取值 → 把 8 种幅值乘积预先算好,查表分发：
```
建表: lut = s2 × 0x03020100 (+ nz·0x01010101)   ← 一条整数乘同时产出 4 个表项
查表: byte_perm 按裸码选字节 → 直接得 (w4−z)·s2 的成品 int8
```
每组乘法从 128 次压到 2 条指令。字节道不进位由 7·s2≤126 保证（量化契约）;
zero point 的 nz=(−z·s2) mod 256 合并必须用 `__vadd4`（逐字节加,普通 32 位加法
会跨字节进位污染邻项——实测踩过的坑）。

**4. 预存解码表（PRELUT）**——比现造更进一步：(s2, z) 的取值空间有硬上界
（18×16=288 种组合）,把全部组合的 8 字节表**离线枚举成编译期常量**（2.3KB 进
cubin,kernel 启动搬一次 smem）,解码时按 (s2,z) 取行,建表指令归零。
小 M 是发射受限区,静态指令数直接折时间——此优化 flash M1 -23%、pro M8-64 -4~5%。
表放 smem 而非 constant 的原因:查表下标 warp 内发散,constant 广播口会串行化,
smem 32-bank 并行口无此问题（constant 直读版实测:发散最重的点 +27%）。

**4b. SHIFTXOR 解码（客户建议,2026-07-15 认证,现推荐档,取代 PRELUT）**——
观察:s2 的组宽（128）恰好等于 inline 档的 promote 段长（K128 链）→ s2 根本
不必进解码,和 s1 一样推迟到 promote。解码退化成纯位运算:
```
展开: byte_perm 铺字节 → (t | t<<4) & 0x0F0F0F0F   ← 每 32bit 字 2 PRMT + 2 SHF + 2 LOP3
减 z: __vsub4(w4, z·0x01010101)                    ← 输出 (w4−z) ∈ [−15,15]
```
每 8 权重 ~14 条指令+每行一次 LDS.64 → **~8 条,零查表、零建表、零 smem 表**
（PRELUT 的 4KB smem 与启动搬运一并移除）。promote 侧 s2 一次 I2F 进寄存器,
每累加元素多一条 FMUL——`float(acc)·float(s2)` 是**精确**乘法
（整数积 |s2·acc| ≤ 18×128×15×128 < 2^24),输出与查表路径逐位相同
（实测 M1/16/2048 三点 bit-identical）。非 swapAB 的 smem 解码档保持折进解码,
改用裸 z 现场重建表（每行多 1 条 IMAD)。10-rep 配对 A/B(vs v3):
inline 十四点全赢,pro 一致 -28~-34%,flash -10~-41%,无回退。
occ-2 档腾出的 4KB 不足以塞进第 5 段流水（预算差 1288B,实测确认回落 4 段）。
env: `DG_W4A8_INT_QOQ_ZP_SHIFTXOR=1`（与 PRELUT 互斥,host 强制二选一）。

**5. 4bit 权重带宽减半**——M1 权重流量占主导,免费解码换一半字节,是小 M 对
FP8 的基础优势来源。

**6. 小 M 2-CTA/SM 低延迟档（occ-2）**——小 M 流水线是**延迟受限**（证据:4bit
读量只有 FP8 一半却只打平 → 带宽未饱和）。把每 CTA 资源瘦身到一半以下
（流水线 8 段→4 段、寄存器 setmaxnreg 配给、修正 smem 记账虚胖）,让两个 CTA
共驻一个 SM:两条独立流水线互相填掉对方的 mbarrier 唤醒与尾部收敛气泡。
SM 级预取总量不变（2×4=8 段）,变的是组织方式——"一列纵队冲带宽"换成
"两路纵队防空转"。实际行为逐点矩阵（运行时自动）：
```
        M1    M2~M8    M16~1024   M≥1280(PRE)
flash    1      2         1           1
pro      2      2         1           1
```
（数字 = 每 SM 驻留 CTA 数:1=常规档,单 CTA 独占 SM、8 段深流水;2=低延迟档,双 CTA 共驻、各 4 段互填空转。）
flash M1 是唯一例外——协议开销主导、无气泡可填,开双 CTA 实测 +43%,故排除;
M≥16 吞吐受限需深流水,自动回单 CTA,路径与 v2 逐位一致。
工程要点：2 CTA 共驻前必须 pin `PREFERRED_SHARED_MEMORY_CARVEOUT=100`
（否则驱动可能少切 smem,静默退化 1 CTA → persistent grid 死锁）;
`setmaxnreg` 目标不得超过 `__launch_bounds__` 隐含预算（超了 = 无警告的坏代码）。

### 中 M（M64-1024,与 FP4 同档竞技区）

**7. 链式单累加器**——scale 组宽 128 与 BLOCK_K 特意对齐 → 4 条 k32 WGMMA 用
scale_d 续同一本账,每 K128 才 promote 一次（t·s1·I2F）。反例已实测勿再试:
小 N 双账本（M16 劣化 11%）、全 K 续链搬 swapAB（寄存器压力反噬）。

**8. promote 不买重叠**（三轮实验定论）——promote 很瘦,被同 SM 邻居的并发自然
吸收;为显式 overlap 付出的寄存器代价必亏。唯一例外:L2 的 per-half 结构天然
两本账,drain 期 promote 白拿。

**9. L2 per-half 双链**——中间激活 SF 是 per-(token,64) 的（64=BLOCK_N/2 的 CTA
几何）,K128 内换两次尺子 → 两条链 + wait<2> 先 promote 前半。

### 大 M（M≥1280,吞吐区）

**10. PRE 预解码档**——inline 模式下同一份权重被每个 token-tile 重复解码
（M/64 遍,∝M）;PRE 档用一个带宽型 prologue kernel 每层一次性 `(w4−z)·s2 → int8`
写回显存,主 kernel TMA 直载,重复税变 O(1)。分界 M=1024 是"固定税 vs 重复税"
两条线的实测交点。小 M 绝不能用:固定开销比整条 kernel 还贵,且主 kernel 读
8bit 流量翻倍。

**11. BN256 大 tile**——大 M 换 BLOCK_N=256,tile 数减半、数据复用翻倍,实测
-19~-26%。

**12. QoQ 全 K 续链**——PRE 档 s1 per-row 恒定、激活 per-token 恒定 → 整条 K
一链到底,主循环零 promote（性能与每 K128 promote 持平,价值在简洁与 ZP 兼容）。

### 通信与融合（MegaMoE 底座）

**13. dispatch/GEMM/combine 单 kernel 融合**——通信藏进计算、kernel 边界清零,
对传统"通信-计算-通信"三段式在小 M 有 1.7-2× 的底座收益（上游实测）。
小 M 通信地板的构成已量化:本地掉队 skew 10-15us + 跨 rank 数据依赖等待
（小 M 时活跃 expert 集中于单 rank,其余 rank 等其数学尾巴）+ 协议本身仅
~3.5-6.5us——协议层已无肉,进一步优化属于 expert 调度域。

### 正确性方法论

- 整数域 bit-exact:单卡 smoke 要求 cosine 精确 1.000000（任何解码/打包改动的
  逐位等价门）;
- 8 卡主 gate:随机路由 × global scale 全组合,flash+pro 两形状;
- 协议改动先过 compute-sanitizer racecheck 再谈性能;racecheck 对
  `cp.async.bulk`+mbarrier expect_tx 模式有已知误报,裁决以"与已验证基线的
  报告集 diff"为准;
- mbarrier 军规:每个 waiter 必须观察静态 slot 类的每一轮 parity,否则
  try_wait.parity 会在 fresh barrier/隔轮/滞后 ≥2 轮时假通过;
- 性能裁决:同日同机、多进程 rep 交替轮转、取中位数（小 M 分布右尾重,均值
  测运气,中位数测实力）、锁频逐轮验证、外来进程监视,污染 rep 整体作废。

---

## 三、性能（我们列 2026-07-15 SHIFTXOR 认证 SX18;FP8/FP4 列 2026-07-14 TEN_18PT）

测量口径:8×H20-3e,SM 锁频 1830（每 run 前后验证）;FP8/FP4 两列来自
2026-07-14 三栈同日轮转认证（TEN_18PT,从零部署环境）;我们列为 2026-07-15
SHIFTXOR 解码变体（客户建议,见二.4b）全 18 点重新认证（同机同锁频,
run 间 idle 验证),另有同日 10-rep 配对交错 A/B(vs v3,M1-64 十四点全赢
-9.6%..-41%,无回退;p4_results/SHIFTXOR_STATUS.md)背书;
balanced routing 三列同口径;指标 = 8 rank 平均 us 的跨 rep median
（M≤64 ×5,M128-1024 ×3,M≥1280 ×2）;
小 M（M≤4）run-to-run 噪声 ±15-30%,以 median 与同日相对座次为准。
对比列:FP8 = AichenF fork SM90 FP8 MegaMoE;FP4 = deepseek PR#323 SM90 FP4 MegaMoE。

### Flash (H=4096, I=2048, E=256, topk6)（us,加粗=行最快）

| M | FP8 | FP4 (PR#323) | 我们 int4a8 (ZP, SHIFTXOR) |
|---|---|---|---|
| 1 | 206 | **199** | 224 † |
| 2 | 251 | 255 | **168** |
| 4 | 402 | 330 | **251** |
| 8 | 460 | 400 | **302** |
| 16 | 473 | 456 | **369** |
| 32 | 484 | 432 | **379** |
| 64 | 476 | 437 | **435** |
| 128 | **492** | 552 | 525 |
| 256 | **509** | 536 | 743 |
| 512 | **906** | 938 | 1171 |
| 819 | **1334** | 1710 | 1830 |
| 1024 | **1300** | 1709 | 2050 |
| 1280 | **1749** | 1759 | 2467 |
| 1536 | **2096** | 2563 | 2831 |
| 2048 | **2500** | 2540 | 3407 |
| 3072 | **3732** | 4158 | 4473 |
| 4096 | **4904** | 4982 | 5652 |
| 8192 | **9671** | 9838 | 10488 |

† flash M1 本轮认证 5 rep 离散大（131-417,协议受限点）;同日 10-rep 配对
A/B 中 SHIFTXOR 126 vs v3 215（-41%),该点建议以配对座次为准。

### Pro (H=7168, I=3072, E=384, topk6)（us）

| M | FP8 | FP4 (PR#323) | 我们 int4a8 (ZP, SHIFTXOR) |
|---|---|---|---|
| 1 | 381 | 454 | **348** |
| 2 | 600 | 481 | **365** |
| 4 | 962 | 821 | **596** |
| 8 | 1553 | 1398 | **955** |
| 16 | 1572 | 1371 | **1090** |
| 32 | 1570 | 1366 | **1097** |
| 64 | 1574 | 1413 | **1113** |
| 128 | 1594 | 1437 | **1359** |
| 256 | **1619** | 1678 | 1734 |
| 512 | **1643** | 3118 | 2526 |
| 819 | **3135** | 3140 | 4913 |
| 1024 | **3161** | 3183 | 4907 |
| 1280 | **4700** | 6161 | 7605 |
| 1536 | **4646** | 6166 | 7610 |
| 2048 | **6134** | 6223 | 9127 |
| 3072 | **9164** | 9323 | 12115 |
| 4096 | **12164** | 12338 | 15111 |
| 8192 | **24081** | 24614 | 27105 |

### 读表要点

- **SHIFTXOR 变体（客户建议,2026-07-15 认证替代 v3 的 PRELUT）**:inline 档
  解码指令 ~14 条+LDS.64/行 → ~8 条,无表无 smem;s2 推迟到 promote
  （逐位等价证明见二.4b）。inline 全档（M≤1024）相对 v3 提速 7-38%,
  PRE 档代码不变（本轮复测在 ±4% 内）。
- **小-中 M 反超**:pro M1-128 与 flash M2-64 现为全行最快——发射受限区
  静态解码指令数直接折时间,原先相对 FP4 的"整数解码格式税"大头已消除;
  flash M128 起、pro M256 起 FP8 领跑。
- **大 M（PRE 档）**:对 FP8 收敛到 +8-13%（8bit 权重读放大是主要残差）,
  权重显存为 FP8 的一半。
- **zero point 代价（SHIFTXOR 后）**:z 以原始值随 coeff 走,解码侧只多
  一条 `__vsub4`;s2 在 promote 侧一次 I2F + 每累加元素一条 FMUL,
  被 WGMMA 阴影吸收。

---

## 四、构建与运行

### 环境要求

- SM90 GPU（H20/H100/H200）,8 卡 NVLink 全互联（8-rank EP 形态;单卡可跑 debug）;
- CUDA 13.x + PyTorch 2.11+（需 `torch.distributed._symmetric_memory`）;
- Python 3.12,nvcc 可用（kernel JIT 编译）。

### 构建（一次性,~3 分钟）

```bash
bash develop.sh          # 编 _C 扩展并软链
# 之后只改 .cuh:无需重编（JIT include hash 自动触发);改 csrc: 重跑 develop.sh
```

### 运行开关（环境变量,进 JIT cache key）

| 开关 | 含义 |
|---|---|
| `DG_W4A8_INT=1 DG_W4A8_INT_L2=1` | int 路径总开关（两层全 int）,必开 |
| `DG_W4A8_INT_QOQ=1` | QServe 两级 scale |
| `DG_W4A8_INT_QOQ_ZP=1` | 加 zero point（客户格式） |
| `DG_W4A8_INT_QOQ_ZP_SHIFTXOR=1` | shift+mask 解码 + s2 推迟 promote（**推荐常开**,2026-07-15 认证,取代 PRELUT;两者互斥） |
| `DG_W4A8_INT_QOQ_ZP_PRELUT=1` | 预存解码表（旧推荐档 v3,被 SHIFTXOR 全点超越,留档） |
| `DG_W4A8_INT_SMALLM_OCC2=1` | 小 M 2-CTA/SM 档（自动门控 pro M1-8/flash M2-8;`_MIN_M/_MAX_M` 可覆盖） |
| `DG_W4A8_INT_PRE=1` | 大 M 预解码档（M≥1280 使用,--block-n 256） |
| `DG_BALANCED_ROUTING=1` | bench 均衡路由（对标口径） |

最快配置 = `DG_W4A8_INT/_L2/_QOQ/_QOQ_ZP/_QOQ_ZP_SHIFTXOR/_SMALLM_OCC2` 全开
（即 SHIFTXOR 档;v3 = 把 SHIFTXOR 换成 PRELUT,留作对照）。

### 正确性

```bash
# 单卡冒烟（逐位等价门,期望 cosine=1.000000）
DG_W4A8_INT=1 DG_W4A8_INT_L2=1 DG_W4A8_INT_QOQ=1 DG_W4A8_INT_QOQ_ZP=1 \
DG_W4A8_INT_QOQ_ZP_SHIFTXOR=1 python3 tests/debug_int_largem.py 1 16 64 2048

# 8 卡主 gate（flash;pro 加 --hidden 7168 --intermediate-hidden 3072 --num-experts 384）
DG_W4A8_INT=1 DG_W4A8_INT_L2=1 DG_W4A8_INT_SMALLM_OCC2=1 \
python3 tests/test_int4_mega_moe_correctness.py --batches 8 64 256 1024
```

### 性能复现（18 点表）

```bash
# 锁频先行
nvidia-smi -lgc 1830,1830

# M ≤ 1024（inline 档）
DG_BALANCED_ROUTING=1 DG_W4A8_INT=1 DG_W4A8_INT_L2=1 DG_W4A8_INT_QOQ=1 \
DG_W4A8_INT_QOQ_ZP=1 DG_W4A8_INT_QOQ_ZP_SHIFTXOR=1 DG_W4A8_INT_SMALLM_OCC2=1 \
python3 tests/bench_mxfp4_mega_moe_sm90.py --block-n 128 \
  --batches 1 2 4 8 16 32 64 128 256 512 819 1024

# M ≥ 1280（PRE 档,decode+GEMM 全计入）
DG_BALANCED_ROUTING=1 DG_W4A8_INT=1 DG_W4A8_INT_L2=1 DG_W4A8_INT_QOQ=1 \
DG_W4A8_INT_QOQ_ZP=1 DG_W4A8_INT_PRE=1 \
python3 tests/bench_int_prologue.py --qoq --block-n 256 \
  --batches 1280 1536 2048 3072 4096 8192
# pro 形状同上加 --hidden 7168 --intermediate-hidden 3072 --num-experts 384
```

测量纪律:确认无其他负载,小 M 用多进程 rep 取 median（M≤64 建议 ×5）。

---

## 五、附件与边界

- `docs/w4a8_int_flow.html` —— 计算流程（上半 SVG 总览图 + 下半分段文字说明,一个文件）;
- 已知边界:swapAB 段要求 block_n=128 prepack,PRE 段建议 block_n=256（两份
  scale plane 可共存）;实验性开关（`_SHADOW/_QOQ_FULLK/_SMALLN_DUAL/_PRELUT_CONST`
  等）默认关,为实测无收益或场景特定的留档;
- 待办:PRE/QoQ/ZP 并入 8 卡主 gate 矩阵、prepack API 产品化（现在 tests/ 的
  量化器里）。

---

## 六、客户形状对比:MegaMoE vs FlashInfer grouped-GEMM + 2×AR（同机统一口径，2026-07-21）

回答"MegaMoE 相对纯 grouped-GEMM 到底有没有优势"。客户形状
**H=4096, I=640(N=1280), E=128, topk=6, m∈{4,8,16,32,44,48,64}**。

### 6.1 测量口径（务必全部对齐，缺一即不可比）

- **同一台机器**:8×H20-3e，锁频 1830，跑前 `nvidia-smi --query-compute-apps` 确认清场（旁边有租户容器会引入 jitter，median 也压不平）；
- **统一计时 = CUDA-event wall-clock**（整操作端到端，含 launch 间隙）：
  MegaMoE 用 `DG_WALL_BENCH=1`；FlashInfer bench 本身就是整调用 event。
  **不要**一边 kineto 逐 kernel 累加、一边 full-call event——两者不可直接比大小；
- **两边都均衡路由**（禁止随机）：MegaMoE `DG_BALANCED_ROUTING=1`；FI bench 打
  `DG_FI_BALANCED=1` 补丁（round-robin `sel=(arange(m*topk)%e)`）。每专家 token=3m/8，
  仅 8|m（m=8/16/32/48/64）严格完美均衡，m=4/44 天然 ±1 余数（两边同模式，仍公平）；
- **FI 取库默认 tactic**（`--mode fixed --g1 -1 --g2 -1`）——小 MoE 形状库默认走
  autotuner 搜索空间之外的手写小-M kernel，**比 autotune 快 ~50%**（隔离逐 m 真调 53s
  仍慢），所以给 FI 最好成绩就是用默认，别开 autotune；
- **median-of-5**（清机后极稳，如 int4-SX m4 五跑 107/106/109/105/106）。

### 6.2 复现命令

```bash
# --- MegaMoE int4-SX，8卡EP8，均衡，wall-clock ---
E="DG_BALANCED_ROUTING=1 DG_WALL_BENCH=1 DG_W4A8_INT=1 DG_W4A8_INT_L2=1 DG_W4A8_INT_QOQ=1 \
   DG_W4A8_INT_QOQ_ZP=1 DG_W4A8_INT_QOQ_ZP_SHIFTXOR=1 DG_W4A8_INT_SMALLM_OCC2=1"
for r in 1 2 3 4 5; do env $E python3 tests/bench_mxfp4_mega_moe_sm90.py \
  --num-processes 8 --hidden 4096 --intermediate-hidden 640 --num-experts 128 \
  --num-topk 6 --block-n 128 --batches 4 8 16 32 44 48 64 | grep WALL; done
# mxfp4 基线:同上去掉全部 DG_W4A8_INT* 开关（DG_W4A8_INT=0）
# 单卡:--num-processes 1（其余不变；单卡=纯计算无跨卡通信）

# --- FlashInfer grouped-GEMM（纯 GEMM，无跨卡通信），均衡，库默认 ---
FLASHINFER_DISABLE_VERSION_CHECK=1 DG_FI_BALANCED=1 python3 bench_fi_mxfp4fp8_scope.py \
  --hidden 4096 --inter 640 --experts 16 --topk 6 --tokens 4,8,16,32,44,48,64 \
  --mode fixed --scope full --out /tmp/fi.json     # experts=16=128/8 一个rank的分片
# 单卡对照:--experts 128

# --- 2×custom AR（SGLang，L1前+L2后各一次），message=m×hidden×bf16 ---
# 8进程 mp.spawn，CustomAllreduce(ps._WORLD.cpu_group, dev, max_size=8MB)，
# 每 m 跑 t=randn(m,4096,bf16)，custom_all_reduce×100 取均值，结果×2。
```

### 6.3 结果（外部纯 grouped-GEMM 数据作交叉验证，量级与 FI 默认列一致，不单列）

**单卡 1-rank（无跨卡通信）· us**

| m | FI grouped-GEMM(默认) | MegaMoE int4-SX | MegaMoE mxfp4 |
|---|---|---|---|
| 4  | 112 | 145 | 153 |
| 8  | 181 | 201 | 232 |
| 16 | 307 | 331 | 387 |
| 32 | 400 | 417 | 488 |
| 44 | 392 | 418 | 487 |
| 48 | 397 | 422 | 490 |
| 64 | 401 | 435 | 500 |

**8卡 EP8（含跨卡通信）· wall · median-of-5 · us**

| m | MegaMoE int4-SX<br>(真8卡含通信) | MegaMoE mxfp4<br>(真8卡含通信) | FI GG默认<br>(单卡shard) | 2×AR | **FI GG+2×AR**<br>(公平对比) |
|---|---|---|---|---|---|
| 4  | **106** | 118 | 98  | 28 | 126 |
| 8  | **105** | 118 | 96  | 28 | 124 |
| 16 | **110** | 119 | 95  | 28 | 123 |
| 32 | **126** | 139 | 95  | 34 | 129 |
| 44 | **142** | 161 | 114 | 35 | 148 |
| 48 | **145** | 164 | 132 | 35 | 167 |
| 64 | **146** | 165 | 135 | 36 | 171 |

> 外部纯 grouped-GEMM 数据（第三方自测 2-GEMM，H20-96G）与上表 FI GG默认列量级一致，
> 作交叉验证，不在表内单列。

### 6.4 结论（两个 regime 相反）

- **单卡:FlashInfer 更快**。FI 去融合，GEMM 独占 smem 做深 prefetch；MegaMoE 融合内核
  单卡吃 per-GEMM 惩罚且无通信可省。→ 单卡场景用 FlashInfer。
- **8卡 EP8:MegaMoE int4-SX 全程赢 FI GG+2×AR 2~16%**。加回真实 dispatch/combine
  通信（2×custom AR=28~36us）后，FI 那点 GEMM 优势被通信吃掉；MegaMoE 把通信**融进计算**
  的护城河兑现。→ 多卡 EP 用 MegaMoE(int4-SX)。
- **必须用 int4-SX，不是 mxfp4**:mxfp4 每 K32 promote（int4 的 4×）拖慢 GEMM，
  仅与 FI GG+2×AR 打平，吃掉了融合优势。
