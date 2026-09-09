// Tiny-M CUDA-core GEMV math path (kTinyMGemv) for the SM90 fused MegaMoE.
// Included from `sm90_fp4_mega_moe_h20_fused_body.inl` inside the math-warp role
// (256 threads, 208 regs) and runs INSTEAD of the tensor-core task loop; the TMA
// loader warps and the interleaved task scheduler are idle. Design:
// docs/tinym_gemv_design.md. Everything communication-side (routing, dispatch pull,
// L1/L2 dependency words, L1 output pool + per-token SF, L2 NVLink scatter,
// fine-combine mailbox, combine, cleanup) is the unchanged body code.
//
// Work unit = (pool block p, 256-row weight tile n, K128 block k) == one dense
// 20480 B packed tile. Units are numbered (p, n, k) with k innermost; CTA `sm_idx`
// owns the contiguous stream-K range of the phase (`get_streamk_range`, worker ==
// sm_idx). All 8 warps walk the same units; warp w owns rows [32w, 32w+32), lane l
// rows 32w + 8r + (l >> 2) (r = 0..3) and the 16 B chunk c = l & 3 of each 80 B row
// (chunk c == RF word c of K32 groups 0..3 after the host word transpose).
#ifndef DG_FP4_TINYM_PREFETCH
#define DG_FP4_TINYM_PREFETCH 2
#endif
{
    DG_STATIC_ASSERT(BLOCK_M == 8 && BLOCK_N == 256 && TASK_BLOCK_N == 256 && TASK_BLOCK_N_L2 == 256,
                     "TinyM GEMV expects BM8 tokens and 256-row weight tiles");
    DG_STATIC_ASSERT(kNumEpilogueThreads == 256 && kNumEpilogueWarps == 8, "TinyM GEMV expects 8 math warps");
    DG_STATIC_ASSERT(BLOCK_K == 128 && L1_OUT_BLOCK_N == 128, "TinyM GEMV expects K128 blocks and 128 L1 output columns");
    DG_STATIC_ASSERT(kMXFP4 || kQoQ, "TinyM GEMV is only implemented for the MXFP4 and QoQ formats");
    DG_STATIC_ASSERT(L1_SHAPE_N / 256u == fused_layout::kSM90SplitKL1NumL1BlockNs &&
                     L2_SHAPE_N / 256u == fused_layout::kSM90SplitKL2NumL2BlockNs,
                     "TinyM GEMV reuses the split-K scratch slots per (pool block, tile)");

    constexpr uint32_t kTMPrefetch = DG_FP4_TINYM_PREFETCH;      // units in flight ahead of compute
    DG_STATIC_ASSERT(kTMPrefetch >= 1 && kTMPrefetch <= 4, "TinyM prefetch depth 1..4");
    constexpr uint32_t kTMRowsPerLane = 4;                        // 32 rows per warp, 4 lanes per row
    constexpr uint32_t kTMMaxTokens = BLOCK_M;                    // 8
    // NOTE: lambda array-reference parameters below spell these bounds as literals
    // (float (&acc)[4][8] etc.): nvcc 13.0's cicc segfaults on constexpr-local bounds.
    DG_STATIC_ASSERT(kTMRowsPerLane == 4 && kTMMaxTokens == 8, "Literal array bounds in the lambdas assume 4 rows x 8 tokens");
    constexpr uint32_t kTMMaxKBlocks = L1_SHAPE_K / BLOCK_K;      // 24 (L2 uses the first 10)
    constexpr uint32_t kTMActBytesPerToken = L1_SHAPE_K * (kMXFP4 ? 2u : 1u);  // fp16 or int8
    constexpr uint32_t kTMActBytes = kTMMaxTokens * kTMActBytesPerToken;
    constexpr uint32_t kTMSfOff = kTMActBytes;                                   // [8][24] fp32
    constexpr uint32_t kTMSumAOff = kTMSfOff + kTMMaxTokens * kTMMaxKBlocks * 4u; // [8][24] int32 (QoQ)
    constexpr uint32_t kTMLutOff = kTMSumAOff + kTMMaxTokens * kTMMaxKBlocks * 4u; // 16 x uint2 (MXFP4)
    constexpr uint32_t kTMPbExpertOff = kTMLutOff + 128u;                        // [64] u32
    constexpr uint32_t kTMPbValidOff = kTMPbExpertOff + 64u * 4u;                // [64] u32
    constexpr uint32_t kTMFlagOff = kTMPbValidOff + 64u * 4u;                    // u32
    constexpr uint32_t kTMSmemBytes = kTMFlagOff + 16u;
    DG_STATIC_ASSERT(kTMSmemBytes <= kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_PACKED_B_SIZE_PER_STAGE),
                     "TinyM GEMV staging must fit the idle pipeline stages");
    DG_STATIC_ASSERT(fused_layout::kSM90SplitKL1PartialBytes == 256u * kTMMaxTokens * 4u,
                     "TinyM partial slot is [256 rows][8 tokens] fp32");

    uint8_t* tm_smem = reinterpret_cast<uint8_t*>(smem_a[0]);
    float* tm_sf = reinterpret_cast<float*>(tm_smem + kTMSfOff);
    int32_t* tm_sum_a = reinterpret_cast<int32_t*>(tm_smem + kTMSumAOff);
    uint2* tm_lut = reinterpret_cast<uint2*>(tm_smem + kTMLutOff);
    uint32_t* tm_pb_expert = reinterpret_cast<uint32_t*>(tm_smem + kTMPbExpertOff);
    uint32_t* tm_pb_valid = reinterpret_cast<uint32_t*>(tm_smem + kTMPbValidOff);
    uint32_t* tm_flag = reinterpret_cast<uint32_t*>(tm_smem + kTMFlagOff);

    const uint32_t tm_row8 = lane_idx >> 2;   // row within an 8-row group
    const uint32_t tm_c = lane_idx & 3u;      // 16 B chunk / K column of the row
    const uint32_t tm_tid = epilogue_thread_idx;
    const auto tm_row = [&](const uint32_t& r) -> uint32_t {
        return epilogue_warp_idx * 32u + r * 8u + tm_row8;
    };
    const auto tm_bar = [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); };

    // ---------------- PTX helpers ----------------
    const auto tm_ldg16 = [](const uint8_t* p) -> uint4 {
        uint4 v;
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
        return v;
    };
    const auto tm_ldg4 = [](const uint8_t* p) -> uint32_t {
        uint32_t v;
        asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
        return v;
    };
    const auto tm_hfma2 = [](const uint32_t& a, const uint32_t& b, const uint32_t& c) -> uint32_t {
        uint32_t d;
        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
        return d;
    };
    const auto tm_hadd2 = [](const uint32_t& a, const uint32_t& b) -> uint32_t {
        uint32_t d;
        asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
        return d;
    };
    const auto tm_hmul2 = [](const uint32_t& a, const uint32_t& b) -> uint32_t {
        uint32_t d;
        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
        return d;
    };
    // fp16x2 -> float(x) + float(y)
    const auto tm_h2_to_f32_sum = [](const uint32_t& h) -> float {
        float a, b;
        asm("{.reg .b16 l, h;\n mov.b32 {l, h}, %2;\n cvt.f32.f16 %0, l;\n cvt.f32.f16 %1, h;}"
            : "=f"(a), "=f"(b) : "r"(h));
        return a + b;
    };
    // 4 e4m3 bytes -> two fp16x2 words (bytes 0,1 -> lo; 2,3 -> hi)
    const auto tm_e4m3x4_to_f16x2 = [](const uint32_t& w, uint32_t& lo, uint32_t& hi) {
        asm("{.reg .b16 l, h;\n mov.b32 {l, h}, %2;\n cvt.rn.f16x2.e4m3x2 %0, l;\n cvt.rn.f16x2.e4m3x2 %1, h;}"
            : "=r"(lo), "=r"(hi) : "r"(w));
    };
    const auto tm_dp4a_u8s8 = [](const uint32_t& a, const uint32_t& b, const int32_t& c) -> int32_t {
        int32_t d;
        asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
        return d;
    };
    const auto tm_dp4a_s8s8 = [](const uint32_t& a, const uint32_t& b, const int32_t& c) -> int32_t {
        int32_t d;
        asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
        return d;
    };
    // Exact int -> float for |v| < 2^22 without I2F
    const auto tm_i2f_small = [](const int32_t& v) -> float {
        return __int_as_float(0x4B400000 + v) - 12582912.0f;
    };
    const auto tm_red_add_f32 = [](float* p, const float& v) {
        asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
    };
    const auto tm_silu = [](const float& x) -> float {
        const float e = kFastMath ? __expf(-x) : expf(-x);
        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
        return x * sig;
    };

    // ---------------- One-time init ----------------
    // Every math warp caches the per-expert recv counts (the B loader normally does
    // this for the scheduler); warp 0 then tabulates (expert, valid_m) per pool block.
    interleaved_scheduler.fetch_expert_recv_count();
    const uint32_t tm_num_pool_blocks = interleaved_scheduler.num_total_m_blocks;
    DG_DEVICE_ASSERT(tm_num_pool_blocks <= fused_layout::kSM90SplitKL1MaxPoolBlocks);
    if (epilogue_warp_idx == 0) {
        for (uint32_t p = 0; p < tm_num_pool_blocks; ++ p) {
            const auto t = interleaved_scheduler.create_task(
                fused_sched::BlockPhase::Linear1, p * kNumRoutedL1BlockNs, kNumRoutedL1BlockNs,
                L1_SHAPE_N, L1_SHAPE_K);
            if (lane_idx == 0) {
                tm_pb_expert[p] = t.local_expert_idx;
                tm_pb_valid[p] = t.valid_m;
            }
        }
    }
    if constexpr (kMXFP4) {
        // LUT row s (host relative E8M0 index, 0 == flush to zero): hi byte of
        // fp16(m * 2^(s - 15)) for the E2M1 magnitudes m = 0, .5, 1, 1.5, 2, 3, 4, 6.
        if (tm_tid < 16) {
            const float mval[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
            const float scale = exp2f(static_cast<float>(static_cast<int>(tm_tid)) - 15.0f);
            uint32_t bytes[8];
            #pragma unroll
            for (uint32_t m = 0; m < 8; ++ m) {
                const float v = tm_tid == 0 ? 0.0f : mval[m] * scale;
                uint16_t hb;
                asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hb) : "f"(v));
                DG_DEVICE_ASSERT((hb & 0xffu) == 0u);
                bytes[m] = hb >> 8;
            }
            tm_lut[tm_tid] = make_uint2(bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24),
                                        bytes[4] | (bytes[5] << 8) | (bytes[6] << 16) | (bytes[7] << 24));
        }
    }
    tm_bar();

    // ---------------- Activation staging (CTA-wide, per phase x pool block) ----------------
    // MXFP4: fp8 e4m3 -> fp16 x 2^-6 (exact), [token][K] halves. QoQ: int8 copy plus the
    // per-(token, K128) int32 activation sum (for the z fold). Both: per-(token, K128)
    // fp32 SF (MXFP4: x 4096 = 2^6 weight LUT shift x 2^6 activation prescale).
    const auto tm_stage_acts = [&](const auto& is_l2_tag, const uint32_t& pool_block_idx,
                                   const uint32_t& valid_m) {
        constexpr bool kL2 = std::remove_cv_t<std::remove_reference_t<decltype(is_l2_tag)>>::value;
        constexpr uint32_t kK = kL2 ? L2_SHAPE_K : L1_SHAPE_K;
        constexpr uint32_t kNKB = kK / BLOCK_K;
        const uint32_t m_idx = pool_block_idx * BLOCK_M;
        if (tm_tid == 0) {
            if constexpr (!kL2) {
                const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                while (ptx::ld_acq(ptr) != valid_m) {}
            } else {
                constexpr uint64_t need = (1ull << kNKB) - 1ull;
                const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                while ((ptx::ld_acq_gpu(ptr) & need) != need) {}
            }
        }
        tm_bar();
        const uint8_t* src = kL2 ? l2_token_buffer.get_base_ptr<uint8_t>() : l1_token_buffer.get_base_ptr<uint8_t>();
        src += static_cast<size_t>(m_idx) * kK;
        constexpr uint32_t kChunksPerToken = kK / 16u;
        for (uint32_t i = tm_tid; i < valid_m * kChunksPerToken; i += kNumEpilogueThreads) {
            const uint32_t t = i / kChunksPerToken, off = (i % kChunksPerToken) * 16u;
            // Written by other SMs (TMA stores): bypass this SM's L1
            const uint4 v = __ldcg(reinterpret_cast<const uint4*>(src + static_cast<size_t>(t) * kK + off));
            if constexpr (kMXFP4) {
                constexpr uint32_t kHalfScale = 0x24002400u;  // fp16x2 2^-6
                uint32_t h[8];
                tm_e4m3x4_to_f16x2(v.x, h[0], h[1]);
                tm_e4m3x4_to_f16x2(v.y, h[2], h[3]);
                tm_e4m3x4_to_f16x2(v.z, h[4], h[5]);
                tm_e4m3x4_to_f16x2(v.w, h[6], h[7]);
                #pragma unroll
                for (uint32_t j = 0; j < 8; ++ j)
                    h[j] = tm_hmul2(h[j], kHalfScale);
                uint4* dst = reinterpret_cast<uint4*>(tm_smem + (t * kK + off) * 2u);
                dst[0] = make_uint4(h[0], h[1], h[2], h[3]);
                dst[1] = make_uint4(h[4], h[5], h[6], h[7]);
            } else {
                *reinterpret_cast<uint4*>(tm_smem + t * kK + off) = v;
            }
        }
        const float* sf_src = kL2 ? l2_sf_buffer.get_base_ptr<float>() : l1_sf_buffer.get_base_ptr<float>();
        for (uint32_t i = tm_tid; i < valid_m * kNKB; i += kNumEpilogueThreads) {
            const uint32_t t = i / kNKB, kb = i % kNKB;
            const float sf = __ldcg(sf_src + static_cast<size_t>(kb) * kNumPaddedSFPoolTokens + m_idx + t);
            tm_sf[t * kTMMaxKBlocks + kb] = kMXFP4 ? sf * 4096.0f : sf;
        }
        tm_bar();
        if constexpr (kQoQ) {
            for (uint32_t i = tm_tid; i < valid_m * kNKB; i += kNumEpilogueThreads) {
                const uint32_t t = i / kNKB, kb = i % kNKB;
                const uint32_t* a = reinterpret_cast<const uint32_t*>(tm_smem + t * kK + kb * BLOCK_K);
                int32_t s = 0;
                #pragma unroll
                for (uint32_t j = 0; j < BLOCK_K / 4; ++ j)
                    s = tm_dp4a_s8s8(a[j], 0x01010101u, s);
                tm_sum_a[t * kTMMaxKBlocks + kb] = s;
            }
            tm_bar();
        }
    };

    // ---------------- Tile completion: cross-CTA reduce + epilogue ----------------
    // acc[r][t]: this lane's fp32 partial (its K chunk c) for rows tm_row(r), tokens t.
    const auto tm_flush_tile = [&](const auto& is_l2_tag, const uint32_t& pool_block_idx,
                                   const uint32_t& n_block_idx, const uint32_t& valid_m,
                                   const uint32_t& k_first, const uint32_t& k_last,
                                   const uint32_t& num_splits, float (&acc)[4][8]) {
        constexpr bool kL2 = std::remove_cv_t<std::remove_reference_t<decltype(is_l2_tag)>>::value;
        constexpr uint32_t kNKB = (kL2 ? L2_SHAPE_K : L1_SHAPE_K) / BLOCK_K;
        const uint32_t m_idx = pool_block_idx * BLOCK_M;
        const uint32_t local_expert_idx = tm_pb_expert[pool_block_idx];
        // 1) the 4 lanes of a row hold K-chunk partials: reduce so all 4 hold the row sum
        #pragma unroll
        for (uint32_t r = 0; r < kTMRowsPerLane; ++ r) {
            #pragma unroll
            for (uint32_t t = 0; t < kTMMaxTokens; ++ t) {
                acc[r][t] += __shfl_xor_sync(0xffffffffu, acc[r][t], 1);
                acc[r][t] += __shfl_xor_sync(0xffffffffu, acc[r][t], 2);
            }
        }
        // 2) cross-CTA fixup unless this CTA covered the whole K range
        const bool full = (k_first == 0u) && (k_last == kNKB - 1u);
        if (!full) {
            float* slot = kL2 ? workspace.get_splitk_l2_scratch_ptr(pool_block_idx, n_block_idx)
                              : workspace.get_splitk_l1_scratch_ptr(pool_block_idx, n_block_idx);
            uint32_t* ticket = kL2 ? workspace.get_splitk_l2_flag_ptr(pool_block_idx, n_block_idx)
                                   : workspace.get_splitk_l1_flag_ptr(pool_block_idx, n_block_idx);
            // Lane c owns tokens c and c + 4 of its rows
            #pragma unroll
            for (uint32_t r = 0; r < kTMRowsPerLane; ++ r) {
                const uint32_t row = tm_row(r);
                #pragma unroll
                for (uint32_t j = 0; j < 2; ++ j) {
                    const uint32_t t = tm_c + j * 4u;
                    if (t < valid_m)
                        tm_red_add_f32(slot + row * kTMMaxTokens + t, acc[r][t]);
                }
            }
            tm_bar();
            if (tm_tid == 0) {
                const uint32_t old = ptx::atomic_add_acq_rel(ticket, 1u);
                const bool last = old + 1u == num_splits;
                if (last)
                    *ticket = 0u;  // reader-reset for the next launch (all arrivals are in)
                *tm_flag = last ? 1u : 0u;
            }
            tm_bar();
            if (*tm_flag == 0u)
                return;  // not the last arriver: the finisher runs the epilogue
            // Last arriver: the slot holds every contributor's partial (this CTA's
            // included, added above), so it REPLACES the registers; re-zero the slot.
            #pragma unroll
            for (uint32_t r = 0; r < kTMRowsPerLane; ++ r) {
                const uint32_t row = tm_row(r);
                #pragma unroll
                for (uint32_t j = 0; j < 2; ++ j) {
                    const uint32_t t = tm_c + j * 4u;
                    if (t < valid_m) {
                        float* p = slot + row * kTMMaxTokens + t;
                        acc[r][t] = __ldcg(p);
                        __stcg(p, 0.0f);
                    }
                }
            }
        }
        // 3) epilogue (all 256 threads; lane c handles tokens c, c + 4 of its 4 rows)
        if constexpr (!kL2) {
            // Rows 16q + j = gate, 16q + 8 + j = up of output column 8q + j; this lane's
            // rows tm_row(0/1) are the gate/up of column 16w + row8, rows tm_row(2/3) of
            // column 16w + 8 + row8.
            const float* __restrict__ row_scales =
                l1_global_scales + local_expert_idx * L1_SHAPE_N + n_block_idx * 256u;
            float v[2][2] = {};        // [column pair q][token j]
            float amax[2] = {};        // [token j]
            #pragma unroll
            for (uint32_t q = 0; q < 2; ++ q) {
                const float gs = __ldg(row_scales + tm_row(2 * q));
                const float us = __ldg(row_scales + tm_row(2 * q + 1));
                #pragma unroll
                for (uint32_t j = 0; j < 2; ++ j) {
                    const uint32_t t = tm_c + j * 4u;
                    if (t < valid_m) {
                        float g = 0.0f, u = 0.0f;
                        #pragma unroll
                        for (uint32_t tt = 0; tt < kTMMaxTokens; ++ tt) {
                            if (tt == t) { g = acc[2 * q][tt]; u = acc[2 * q + 1][tt]; }
                        }
                        g *= gs;
                        u *= us;
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                            g = cute::min(g, kActivationClamp);
                            u = cute::min(cute::max(u, -kActivationClamp), kActivationClamp);
                        }
                        const float weight = *l1_topk_weights_buffer.get_data_buffer(m_idx + t)
                            .template get_base_ptr<float>();
                        v[q][j] = tm_silu(g) * u * weight;
                        amax[j] = cute::max(amax[j], cute::abs(v[q][j]));
                    }
                }
            }
            // Per-token amax over the warp's 16 columns (lanes sharing c), then over warps
            #pragma unroll
            for (uint32_t j = 0; j < 2; ++ j) {
                amax[j] = cute::max(amax[j], __shfl_xor_sync(0xffffffffu, amax[j], 4));
                amax[j] = cute::max(amax[j], __shfl_xor_sync(0xffffffffu, amax[j], 8));
                amax[j] = cute::max(amax[j], __shfl_xor_sync(0xffffffffu, amax[j], 16));
            }
            if (lane_idx < 4) {
                #pragma unroll
                for (uint32_t j = 0; j < 2; ++ j) {
                    const uint32_t t = tm_c + j * 4u;
                    if (t < valid_m)
                        smem_cd_l1_shared_sf[t * kNumEpilogueWarps + epilogue_warp_idx] = amax[j];
                }
            }
            tm_bar();
            if (tm_tid < valid_m) {
                const uint32_t t = tm_tid;
                float a = 0.0f;
                #pragma unroll
                for (uint32_t w = 0; w < kNumEpilogueWarps; ++ w)
                    a = cute::max(a, smem_cd_l1_shared_sf[t * kNumEpilogueWarps + w]);
                float2 sf_pair, sf_inv_pair;
                if constexpr (kQoQ) {
                    sf_pair.x = a * (1.0f / 127.0f);
                    sf_inv_pair.x = a > 0.0f ? 127.0f / a : 0.0f;
                } else {
                    const float2 amax_pair = {a, a};
                    math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                }
                l2_sf_buffer.get_base_ptr<float>()[n_block_idx * kNumPaddedSFPoolTokens + m_idx + t] = sf_pair.x;
                smem_cd_l1_shared_sf[t * kNumEpilogueWarps] = sf_inv_pair.x;
            }
            tm_bar();
            #pragma unroll
            for (uint32_t q = 0; q < 2; ++ q) {
                const uint32_t col = epilogue_warp_idx * 16u + q * 8u + tm_row8;
                #pragma unroll
                for (uint32_t j = 0; j < 2; ++ j) {
                    const uint32_t t = tm_c + j * 4u;
                    if (t < valid_m) {
                        const float x = v[q][j] * smem_cd_l1_shared_sf[t * kNumEpilogueWarps];
                        uint8_t byte;
                        if constexpr (kQoQ) {
                            const int qv = cute::min(cute::max(__float2int_rn(x), -127), 127);
                            byte = static_cast<uint8_t>(static_cast<int8_t>(qv));
                        } else {
                            const __nv_fp8_e4m3 fq(x);
                            byte = *reinterpret_cast<const uint8_t*>(&fq);
                        }
                        reinterpret_cast<uint8_t*>(smem_cd_l1)[t * L1_OUT_BLOCK_N + col] = byte;
                    }
                }
            }
            tm_bar();
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                cute::tma_store_fence();
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_l1_output, smem_cd_l1,
                                              n_block_idx * L1_OUT_BLOCK_N, m_idx);
                cute::tma_store_arrive();
            }
            __syncwarp();
            ptx::tma_store_wait<0>();
            // The L2 stager reads the async-proxy (TMA) stores with generic loads
            asm volatile("fence.proxy.async.global;" ::: "memory");
            tm_bar();
            notify_l1_ready(pool_block_idx, n_block_idx);
        } else {
            // L2: bf16(acc x per-hidden-row scale) -> smem [token][256] -> NVLink scatter
            const float* __restrict__ row_scales =
                l2_global_scales + local_expert_idx * L2_SHAPE_N + n_block_idx * 256u;
            #pragma unroll
            for (uint32_t r = 0; r < kTMRowsPerLane; ++ r) {
                const uint32_t row = tm_row(r);
                const float sc = __ldg(row_scales + row);
                #pragma unroll
                for (uint32_t j = 0; j < 2; ++ j) {
                    const uint32_t t = tm_c + j * 4u;
                    if (t < valid_m) {
                        float x = 0.0f;
                        #pragma unroll
                        for (uint32_t tt = 0; tt < kTMMaxTokens; ++ tt)
                            if (tt == t) x = acc[r][tt];
                        smem_cd_l2[t * BLOCK_N + row] = __float2bfloat16_rn(x * sc);
                    }
                }
            }
            tm_bar();
            // Warp w scatters token w's 256 bf16 (512 B, 16 B per lane)
            const uint32_t token = epilogue_warp_idx;
            if (token < valid_m) {
                const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + token);
                const auto dst_token = combine_token_buffer.get_rank_buffer(src_metadata.topk_idx)
                                       .get_data_buffer(src_metadata.token_idx);
                const uint4 packed = *reinterpret_cast<const uint4*>(smem_cd_l2 + token * BLOCK_N + lane_idx * 8u);
                auto dst_ptr = math::advance_ptr<uint4>(
                    dst_token.get_base_ptr(),
                    (n_block_idx * 256u) * sizeof(nv_bfloat16) + lane_idx * sizeof(uint4));
                *sym_buffer.map(dst_ptr, src_metadata.rank_idx) = packed;
            }
            tm_bar();
            if constexpr (kFineCombine) {
                if (tm_tid == 0 && valid_m > 0) {
                    auto* mailbox = workspace.get_combine_mailbox_ptr(sm_idx);
                    while (combine_mailbox_seq - ptx::ld_volatile(mailbox + 1) >= fused_layout::kSM90FineCombineRingSize) {}
                    mailbox[4 + (combine_mailbox_seq & (fused_layout::kSM90FineCombineRingSize - 1))] =
                        pool_block_idx | (valid_m << 24);
                    ptx::st_rel_gpu(mailbox, combine_mailbox_seq + 1);
                    ++ combine_mailbox_seq;
                }
            }
        }
    };

    // ---------------- Phase runner ----------------
    const auto tm_run_phase = [&](const auto& is_l2_tag) {
        constexpr bool kL2 = std::remove_cv_t<std::remove_reference_t<decltype(is_l2_tag)>>::value;
        constexpr uint32_t kK = kL2 ? L2_SHAPE_K : L1_SHAPE_K;
        constexpr uint32_t kNKB = kK / BLOCK_K;
        constexpr uint32_t kNB = (kL2 ? L2_SHAPE_N : L1_SHAPE_N) / 256u;
        constexpr uint32_t kUnitsPerPoolBlock = kNB * kNKB;
        const uint8_t* weights = reinterpret_cast<const uint8_t*>(kL2 ? l2_weights_ptr : l1_weights_ptr);
        const uint32_t num_units = tm_num_pool_blocks * kUnitsPerPoolBlock;
        uint32_t u_begin = 0, u_end = 0;
        interleaved_scheduler_t::get_streamk_range(num_units, sm_idx, u_begin, u_end);
        if (u_begin >= u_end)
            return;

        // Lane's 4 row chunks (16 B nibbles + 4 B meta) of the next kTMPrefetch units
        uint4 raw_q[kTMPrefetch][kTMRowsPerLane];
        uint32_t raw_meta[kTMPrefetch][kTMRowsPerLane];
        const auto issue_unit = [&](uint4 (&dst_q)[4], uint32_t (&dst_meta)[4],
                                    const uint32_t& u) {
            const uint32_t p = u / kUnitsPerPoolBlock;
            const uint32_t rem = u - p * kUnitsPerPoolBlock;
            const uint32_t n = rem / kNKB, k = rem - n * kNKB;
            const uint32_t tile_row = tm_pb_expert[p] * kNB + n;
            const uint8_t* tile = weights +
                (static_cast<size_t>(tile_row) * kNKB + k) * kPackedTileBytes;
            #pragma unroll
            for (uint32_t r = 0; r < kTMRowsPerLane; ++ r) {
                const uint8_t* row_ptr = tile + tm_row(r) * 80u;
                dst_q[r] = tm_ldg16(row_ptr + tm_c * 16u);
                dst_meta[r] = tm_ldg4(row_ptr + 64u);
            }
        };

        float acc[kTMRowsPerLane][kTMMaxTokens];
        #pragma unroll
        for (uint32_t r = 0; r < kTMRowsPerLane; ++ r) {
            #pragma unroll
            for (uint32_t t = 0; t < kTMMaxTokens; ++ t)
                acc[r][t] = 0.0f;
        }
        uint32_t cur_pool_block = 0xffffffffu, cur_tile = 0xffffffffu;
        uint32_t cur_valid_m = 0, k_first = 0, k_prev = 0;

        // Compute one unit from its raw chunks
        const auto consume_unit = [&](const uint4 (&rw_q)[4], const uint32_t (&rw_meta)[4],
                                      const uint32_t& k, const uint32_t& valid_m) {
            const uint32_t k_off = k * BLOCK_K;
            if constexpr (kMXFP4) {
                // Decode 4 rows x 4 K32 groups -> fp16x2 pairs [row][group][pair]
                uint32_t wq[kTMRowsPerLane][4][4];
                #pragma unroll
                for (uint32_t r = 0; r < kTMRowsPerLane; ++ r) {
                    const uint32_t words[4] = {rw_q[r].x, rw_q[r].y, rw_q[r].z, rw_q[r].w};
                    #pragma unroll
                    for (uint32_t g = 0; g < 4; ++ g) {
                        const uint2 lut = tm_lut[(rw_meta[r] >> (g * 8u)) & 0xfu];
                        const uint32_t w = words[g];
                        const uint32_t sel = w & 0x77777777u;
                        uint32_t hb_hi = __byte_perm(lut.x, lut.y, sel);
                        uint32_t hb_lo = __byte_perm(lut.x, lut.y, sel >> 16);
                        hb_hi |= w & 0x80808080u;
                        hb_lo |= (w << 4) & 0x80808080u;
                        wq[r][g][0] = __byte_perm(hb_hi, 0u, 0x1404);
                        wq[r][g][1] = __byte_perm(hb_hi, 0u, 0x3424);
                        wq[r][g][2] = __byte_perm(hb_lo, 0u, 0x1404);
                        wq[r][g][3] = __byte_perm(hb_lo, 0u, 0x3424);
                    }
                }
                #pragma unroll
                for (uint32_t t = 0; t < kTMMaxTokens; ++ t) {
                    if (t < valid_m) {
                        const uint8_t* a_base = tm_smem + (t * kK + k_off) * 2u;
                        uint2 a_lo[4], a_hi[4];
                        #pragma unroll
                        for (uint32_t g = 0; g < 4; ++ g) {
                            a_lo[g] = *reinterpret_cast<const uint2*>(a_base + (g * 32u + tm_c * 4u) * 2u);
                            a_hi[g] = *reinterpret_cast<const uint2*>(a_base + (g * 32u + 16u + tm_c * 4u) * 2u);
                        }
                        const float sf = tm_sf[t * kTMMaxKBlocks + k];
                        #pragma unroll
                        for (uint32_t r = 0; r < kTMRowsPerLane; ++ r) {
                            uint32_t h0 = 0u, h1 = 0u;
                            #pragma unroll
                            for (uint32_t g = 0; g < 4; ++ g) {
                                h0 = tm_hfma2(wq[r][g][0], a_lo[g].x, h0);
                                h1 = tm_hfma2(wq[r][g][1], a_lo[g].y, h1);
                                h0 = tm_hfma2(wq[r][g][2], a_hi[g].x, h0);
                                h1 = tm_hfma2(wq[r][g][3], a_hi[g].y, h1);
                            }
                            acc[r][t] += sf * tm_h2_to_f32_sum(tm_hadd2(h0, h1));
                        }
                    }
                }
            } else {
                // QoQ: unsigned codes; z folded after the dot product via the activation sum
                uint32_t hi[kTMRowsPerLane][4], lo[kTMRowsPerLane][4];
                float s2f[kTMRowsPerLane];
                int32_t zi[kTMRowsPerLane];
                #pragma unroll
                for (uint32_t r = 0; r < kTMRowsPerLane; ++ r) {
                    const uint32_t words[4] = {rw_q[r].x, rw_q[r].y, rw_q[r].z, rw_q[r].w};
                    #pragma unroll
                    for (uint32_t g = 0; g < 4; ++ g) {
                        hi[r][g] = (words[g] >> 4) & 0x0f0f0f0fu;
                        lo[r][g] = words[g] & 0x0f0f0f0fu;
                    }
                    zi[r] = static_cast<int32_t>((rw_meta[r] >> 8) & 0xffu);
                    s2f[r] = tm_i2f_small(static_cast<int32_t>(rw_meta[r] & 0xffu));
                }
                #pragma unroll
                for (uint32_t t = 0; t < kTMMaxTokens; ++ t) {
                    if (t < valid_m) {
                        const uint8_t* a_base = tm_smem + t * kK + k_off;
                        uint32_t a_lo[4], a_hi[4];
                        #pragma unroll
                        for (uint32_t g = 0; g < 4; ++ g) {
                            a_lo[g] = *reinterpret_cast<const uint32_t*>(a_base + g * 32u + tm_c * 4u);
                            a_hi[g] = *reinterpret_cast<const uint32_t*>(a_base + g * 32u + 16u + tm_c * 4u);
                        }
                        const float sf = tm_sf[t * kTMMaxKBlocks + k];
                        const int32_t sum_a = tm_sum_a[t * kTMMaxKBlocks + k];
                        #pragma unroll
                        for (uint32_t r = 0; r < kTMRowsPerLane; ++ r) {
                            int32_t d0 = 0, d1 = 0;
                            #pragma unroll
                            for (uint32_t g = 0; g < 4; ++ g) {
                                d0 = tm_dp4a_u8s8(hi[r][g], a_lo[g], d0);
                                d1 = tm_dp4a_u8s8(lo[r][g], a_hi[g], d1);
                            }
                            const int32_t v = d0 + d1 - zi[r] * sum_a;
                            acc[r][t] += (sf * s2f[r]) * tm_i2f_small(v);
                        }
                    }
                }
            }
        };

        // Number of CTAs contributing to tile (p, n): owners of its first and last unit
        const auto num_splits_of = [&](const uint32_t& tile) -> uint32_t {
            const uint32_t first = interleaved_scheduler_t::get_streamk_worker_of_unit(num_units, tile * kNKB);
            const uint32_t last = interleaved_scheduler_t::get_streamk_worker_of_unit(num_units, tile * kNKB + kNKB - 1u);
            return last - first + 1u;
        };
        const auto flush_current = [&]() {
            if (cur_tile != 0xffffffffu) {
                const uint32_t p = cur_tile / kNB, n = cur_tile - p * kNB;
                tm_flush_tile(is_l2_tag, p, n, cur_valid_m, k_first, k_prev, num_splits_of(cur_tile), acc);
                #pragma unroll
                for (uint32_t r = 0; r < kTMRowsPerLane; ++ r) {
                    #pragma unroll
                    for (uint32_t t = 0; t < kTMMaxTokens; ++ t)
                        acc[r][t] = 0.0f;
                }
            }
        };

        // Prime the prefetch ring, then walk the range kTMPrefetch units per iteration
        #pragma unroll
        for (uint32_t s = 0; s < kTMPrefetch; ++ s) {
            if (u_begin + s < u_end)
                issue_unit(raw_q[s], raw_meta[s], u_begin + s);
        }
        for (uint32_t u = u_begin; u < u_end; u += kTMPrefetch) {
            #pragma unroll
            for (uint32_t s = 0; s < kTMPrefetch; ++ s) {
                const uint32_t uu = u + s;
                if (uu < u_end) {
                    const uint32_t p = uu / kUnitsPerPoolBlock;
                    const uint32_t rem = uu - p * kUnitsPerPoolBlock;
                    const uint32_t n = rem / kNKB, k = rem - n * kNKB;
                    const uint32_t tile = p * kNB + n;
                    if (tile != cur_tile) {
                        flush_current();
                        cur_tile = tile;
                        k_first = k;
                        if (p != cur_pool_block) {
                            cur_pool_block = p;
                            cur_valid_m = tm_pb_valid[p];
                            tm_stage_acts(is_l2_tag, p, cur_valid_m);
                        }
                    }
                    consume_unit(raw_q[s], raw_meta[s], k, cur_valid_m);
                    k_prev = k;
                    if (uu + kTMPrefetch < u_end)
                        issue_unit(raw_q[s], raw_meta[s], uu + kTMPrefetch);
                }
            }
        }
        flush_current();
    };

    if (tm_tid == 0) stamp_min(3);
    tm_run_phase(std::integral_constant<bool, false>{});
    if (tm_tid == 0) stamp_max(4);
    tm_run_phase(std::integral_constant<bool, true>{});
    if (tm_tid == 0) stamp_max(5);
}
