# W4A8-int MegaMoE (SM90/H20) — 分享与运行指南

int4 权重 × int8 激活的 MoE 专家层 kernel，支持 GPTQ 式 fp32 group scale 与
QServe 式两级量化（含 zero point）两种格式。基于 DeepGEMM（deepseek-ai，MIT 协议，经 AichenF fork 的 MegaMoE 分支）开发：本分支直接坐在 fork main（`54e2261`，含上游 #364 重构）之上，上游 commit 历史完整保留，我们的全部改动为其上的增量 commit（MXFP4 基建 bring-up → int4/QoQ/ZP 移植 → 性能对齐与 bench 口径修复 → ZP prepack 优化），可逐 commit 审阅。

> **命名说明**：int4 路径与 mxfp4 共享同一套 kernel 基建，因此实现位于
> `sm90_mxfp4_mega_moe.cuh` 等 mxfp4 命名的文件内，通过 `DG_W4A8_INT*` 环境
> 变量启用；Python 入口为 `deep_gemm.int4_mega_moe(...)`（`mxfp4_mega_moe` 的
> int4 正名别名，行为由环境变量决定）。

## 一、代码怎么拿 / 怎么给别人

仓库自带完整 git 历史（上游 DeepGEMM 全部 commit + 我们的增量 commit），推荐直接
push 到 GitHub 私有仓库共享；也可整目录打包（含 third-party/cutlass 子目录）：

```bash
tar czf deepgemm_w4a8int.tar.gz \
    --exclude='build' --exclude='*.egg-info' --exclude='.deep_gemm' hopper_megamoe_basedondeepgemm/
```

看我们相对上游改了什么：`git log --oneline 54e2261..HEAD`，逐 commit `git show` 即可。

## 二、环境要求

- SM90 GPU（H20/H100/H200），8 卡 NVLink 全互联（MoE 是 8-rank EP 形态；单卡可跑 debug 脚本）
- CUDA 13.x + PyTorch 2.11+（需 `torch.distributed._symmetric_memory`）
- Python 3.12；nvcc 可用（kernel 为 JIT 编译）
- 参考容器：`sglang-megamoe-pd` 系（py3.12 / torch2.11 / cu13.0）

## 三、构建（一次性，~3 分钟）

```bash
cd hopper_megamoe_basedondeepgemm
bash develop.sh          # 编 _C 扩展并软链到 deep_gemm/
# 之后只改 .cuh：无需重编，删 JIT 缓存即可： rm -rf ~/.deep_gemm/cache
# 改 csrc/*.hpp/cpp：重跑 develop.sh
```

## 四、运行开关（环境变量，进 JIT cache key）

| 开关 | 含义 |
|---|---|
| `DG_W4A8_INT=1 DG_W4A8_INT_L2=1` | int 路径总开关（两层全 int），必开 |
| `DG_W4A8_INT_QOQ=1` | QServe 两级 scale（per-channel bf16 s1 + per-group128 整数 s2 折叠解码） |
| `DG_W4A8_INT_QOQ_ZP=1` | 在 QOQ 上加 zero point（uint4 非对称，客户格式） |
| `DG_W4A8_INT_PRE=1` | 大 M 预解码档（int4→int8 prologue + BN256），token>1024 时使用 |
| `DG_BALANCED_ROUTING=1` | bench 用均衡路由（对标口径） |
| `DG_MXFP4_SWAP_AB_MAX_M` | swapAB 上限，默认 1024（分段阈值） |

不开任何 DG_W4A8 开关 = 原 MXFP4 路径，互不影响。

## 五、跑正确性

```bash
# 快速单进程冒烟（单卡即可；覆盖 swapAB + 非swapAB + PRE + QoQ + ZP）
DG_W4A8_INT=1 DG_W4A8_INT_L2=1 DG_W4A8_INT_QOQ=1 DG_W4A8_INT_QOQ_ZP=1 \
  python3 tests/debug_int_largem.py 1 16 64 2048        # 期望全 PASS, cosine≈1
# PRE 档：
DG_W4A8_INT=1 DG_W4A8_INT_L2=1 DG_W4A8_INT_QOQ=1 DG_W4A8_INT_QOQ_ZP=1 \
  DG_W4A8_INT_PRE=1 DG_TEST_BLOCK_N=256 python3 tests/debug_int_largem.py 64 2048

# 完整 8 卡主 gate（随机路由 × global scale 全组合，flash 形状）
DG_W4A8_INT=1 DG_W4A8_INT_L2=1 python3 tests/test_int4_mega_moe_correctness.py \
  --batches 8 64 256 1024
# pro 形状加：--hidden 7168 --intermediate-hidden 3072 --num-experts 384
```

## 六、跑性能（18 点表的复现方式）

```bash
# M ≤ 1024（inline swapAB 段，flash）：
DG_BALANCED_ROUTING=1 DG_W4A8_INT=1 DG_W4A8_INT_L2=1 DG_W4A8_INT_QOQ=1 [DG_W4A8_INT_QOQ_ZP=1] \
  python3 tests/bench_mxfp4_mega_moe_sm90.py --block-n 128 \
  --batches 1 2 4 8 16 32 64 128 256 512 819 1024

# M ≥ 1280（PRE 段，decode+GEMM 全计入）：
DG_BALANCED_ROUTING=1 DG_W4A8_INT=1 DG_W4A8_INT_L2=1 DG_W4A8_INT_QOQ=1 DG_W4A8_INT_PRE=1 \
  python3 tests/bench_int_prologue.py --qoq --block-n 256 \
  --batches 1280 1536 2048 3072 4096 8192
# pro 形状同上加 --hidden 7168 --intermediate-hidden 3072 --num-experts 384
```

测量注意：锁频（`nvidia-smi -lgc 1830`）、确认无其他负载（`nvidia-smi` 看 util/进程）、
小 M run-to-run ±15%，对外数字用 median-of-5。

## 七、文档地图

- `docs/w4a8_int_flowchart.html` —— SVG 计算流程图（浏览器打开，含两段实现/量化/promote 全细节）
- `docs/w4a8_int_flow.html` —— 分段文字版流程说明
- `docs/W4A8_INT_PERFORMANCE.md` —— 18 点性能表（flash/pro，对比 FP8/FP4 MegaMoE）与测量口径

## 八、已知边界 / 待办

- swapAB 段要求 block_n=128 的 prepack；PRE 段建议 block_n=256（两份 scale plane 可共存）
- 实验机关 `DG_W4A8_INT_SHADOW/_QOQ_FULLK/_SMALLN_DUAL` 默认关（后两个实测无收益，待删）
- 待办：PRE/QoQ/ZP 并入 8 卡主 gate、prepack API 产品化（现散在 tests/ 的量化器里）、
  M4-512 段 ncu 定位、（可选）shadow 完全体
