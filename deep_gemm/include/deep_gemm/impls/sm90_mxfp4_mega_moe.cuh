#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm89.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/quantization/mxfp4_dequant.cuh>
#include <deep_gemm/quantization/int4_dequant.cuh>
#define __CLION_IDE__

// W4A8-integer variant: reuses the whole swapAB-RF pipeline (weight/act TMA is
// byte-identical to MXFP4/FP8) with int4->int8 decode, IGMMA int32 accumulate,
// and an I2F epilogue. Injected as a compile #define by the JIT frontend.
#ifndef DG_W4A8_INT
#define DG_W4A8_INT 0
#endif
// L2 int path (int8 intermediate) still WIP; gated separately so default
// DG_W4A8_INT stays the validated L1-int / L2-mxfp4 mode.
#ifndef DG_W4A8_INT_L2
#define DG_W4A8_INT_L2 0
#endif
// Prologue-int (large M): weights arrive as pre-decoded int8 rows (one
// bandwidth-bound decode pass per layer outside the kernel); B TMA loads them
// straight into smem_b with the A-style 128B swizzle and the in-kernel decode
// stage disappears. Requires DG_W4A8_INT; host forces non-swapAB.
#ifndef DG_W4A8_INT_PRE
#define DG_W4A8_INT_PRE 0
#endif
// FP4-style shadow decode (BN256 math-dequant path): while a K128 block's
// WGMMAs are in flight, the math warpgroups decode the NEXT stage's B tile,
// moving the decode off the critical path.
#ifndef DG_W4A8_INT_SHADOW
#define DG_W4A8_INT_SHADOW 0
#endif
// QoQ (QServe-style) two-level weight scale: per-group int8 s2 folded into
// the pre-decoded int8 weights (integer-domain fold-at-decode) + per-row
// fp32 s1 that is K-constant. L1 chains ONE int32 accumulator across the
// whole K loop with zero in-loop promotes; s1 x act-scale applied once at
// the end. Requires DG_W4A8_INT_PRE (the fold happens in the prologue).
#ifndef DG_W4A8_INT_QOQ
#define DG_W4A8_INT_QOQ 0
#endif
// QoQ full-K chain for the swapAB L1: both scales are K-constant, so the
// int32 accumulator can persist across ALL K128 stages with a single promote
// after the loop (removes K/128-1 promote+I2F passes). A/B switch.
#ifndef DG_W4A8_INT_QOQ_FULLK
#define DG_W4A8_INT_QOQ_FULLK 0
#endif
// Asymmetric QoQ: level-2 carries a zero point z (uint4) in the coeff word's
// second byte ([s2 | z | s1:bf16]); -z*s2 folds into the decode LUT entries.
#ifndef DG_W4A8_INT_QOQ_ZP
#define DG_W4A8_INT_QOQ_ZP 0
#endif

// iter17: relative pre-scaled LUT fold. Each K32 group inside a K128 block is
// decoded against the LUT row for d = e_max - e_group, so the four WGMMAs of a
// block share one scale (2^(e_max - 127)) and one accumulator: the per-K32
// promotion tax (4 promote passes per K128) collapses to one (L1) or two (L2,
// where the activation SF changes at K64). Injected as a compile #define by
// the JIT frontend when DG_MXFP4_REL_LUT=1.
#ifndef DG_MXFP4_REL_LUT
#define DG_MXFP4_REL_LUT 0
#endif

namespace deep_gemm {

namespace mxfp4 {

template <bool kUsePRMTGroups, bool kIntDecode = false>
__device__ __forceinline__ uint2 decode_mxfp4_split_pair(const uint32_t packed) {
    // W4A8-int shares the packed-value layout with MXFP4; the SMEM dequant
    // stage decodes int4 nibbles to sign-extended int8 bytes instead of FP8.
    // The 4 coeff bytes per row (copied verbatim) then hold one fp32 weight
    // scale per K128 instead of four E8M0s. The flag is per-phase: hybrid
    // mode (int L1 + mxfp4 L2) decodes each phase's weights differently.
    if constexpr (kIntDecode) {
        static_assert(kUsePRMTGroups, "W4A8-int SMEM decode expects grouped-PRMT layout");
        return int4q::decode_int4_prmt_groups_to_int8_pair(packed);
    } else if constexpr (kUsePRMTGroups) {
        return decode_mxfp4_prmt_groups_to_fp8_pair(packed);
    } else {
        return decode_mxfp4_to_fp8_pair(packed);
    }
}

// MXFP4 80-byte packed row: 64B = 128 E2M1 nibbles, 4B = four E8M0 scales
// (one per 32 K-elements), 12B padding. Decode expands values to UNSCALED
// FP8 E4M3; the E8M0 bytes are copied verbatim into the per-stage coeff
// array (layout: uint32 per row = coeff bytes [row * 4 + k32_group]) and
// applied as float multipliers in the WGMMA promotion.

template <bool kUsePRMTGroups, bool kIntDecode = false, bool kQoQFold = false>
__device__ __forceinline__ void dequant_smem_b_from_packed_unscaled(
        uint8_t* __restrict__ smem_b,
        const uint8_t* __restrict__ packed_b,
        const uint32_t row,
        uint8_t* __restrict__ coeff_smem) {
    const uint8_t* __restrict__ row_ptr = packed_b + row * 64;
    const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
    uint4 fp4_quads[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        fp4_quads[i] = fp4_src[i];
    // QoQ fold: coeff word's low byte = integer s2; bake into the decode LUT.
    [[maybe_unused]] uint32_t qoq_s2 = 0, qoq_lut_lo = 0, qoq_lut_hi = 0;
    if constexpr (kQoQFold) {
        const uint32_t cw = reinterpret_cast<const uint32_t*>(coeff_smem)[row];
        qoq_s2 = cw & 0xffu;
#if DG_W4A8_INT_QOQ_ZP
        // Byte 1 of the coeff word carries nz = (-z*s2) mod 256, precomputed
        // at PREPACK time (saves the per-row multiply+negate here).
        const uint32_t nz = (cw >> 8) & 0xffu;
        qoq_lut_lo = __vadd4(qoq_s2 * 0x03020100u, nz * 0x01010101u);
        qoq_lut_hi = __vadd4(qoq_s2 * 0x07060504u, nz * 0x01010101u);
#else
        qoq_lut_lo = qoq_s2 * 0x03020100u;
        qoq_lut_hi = qoq_s2 * 0x07060504u;
#endif
    } else {
        (void)coeff_smem;  // coefficients are TMA-delivered
    }

    uint8_t* __restrict__ fp8_dst = smem_b + row * 128;
    const uint32_t row_swizzle = (row & 7u) << 4;
    // Word-transposed RF layout: quad c holds thread-c's words for K32
    // batches 0..3; word b decodes to elements {4c..4c+3} of batch b (lo
    // chunk 2b) and {4c+16..4c+19} (hi chunk 2b+1).
    // Decode all 16 words first, then assemble one 16B vector store per
    // output chunk (gathering word c of each thread-quad for batch b).
    uint2 dq[4][4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const uint4 fp4_quad = fp4_quads[c];
        if constexpr (kQoQFold) {
#if DG_W4A8_INT_QOQ_ZP
            dq[c][0] = int4q::decode_uint4_prmt_groups_to_int8_pair_lut_zp(fp4_quad.x, qoq_lut_lo, qoq_lut_hi, qoq_s2);
            dq[c][1] = int4q::decode_uint4_prmt_groups_to_int8_pair_lut_zp(fp4_quad.y, qoq_lut_lo, qoq_lut_hi, qoq_s2);
            dq[c][2] = int4q::decode_uint4_prmt_groups_to_int8_pair_lut_zp(fp4_quad.z, qoq_lut_lo, qoq_lut_hi, qoq_s2);
            dq[c][3] = int4q::decode_uint4_prmt_groups_to_int8_pair_lut_zp(fp4_quad.w, qoq_lut_lo, qoq_lut_hi, qoq_s2);
#else
            dq[c][0] = int4q::decode_int4_prmt_groups_to_int8_pair_lut(fp4_quad.x, qoq_lut_lo, qoq_lut_hi, qoq_s2);
            dq[c][1] = int4q::decode_int4_prmt_groups_to_int8_pair_lut(fp4_quad.y, qoq_lut_lo, qoq_lut_hi, qoq_s2);
            dq[c][2] = int4q::decode_int4_prmt_groups_to_int8_pair_lut(fp4_quad.z, qoq_lut_lo, qoq_lut_hi, qoq_s2);
            dq[c][3] = int4q::decode_int4_prmt_groups_to_int8_pair_lut(fp4_quad.w, qoq_lut_lo, qoq_lut_hi, qoq_s2);
#endif
        } else {
            dq[c][0] = decode_mxfp4_split_pair<kUsePRMTGroups, kIntDecode>(fp4_quad.x);
            dq[c][1] = decode_mxfp4_split_pair<kUsePRMTGroups, kIntDecode>(fp4_quad.y);
            dq[c][2] = decode_mxfp4_split_pair<kUsePRMTGroups, kIntDecode>(fp4_quad.z);
            dq[c][3] = decode_mxfp4_split_pair<kUsePRMTGroups, kIntDecode>(fp4_quad.w);
        }
    }
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        *reinterpret_cast<uint4*>(fp8_dst + (((b * 2) * 16) ^ row_swizzle)) =
            make_uint4(dq[0][b].x, dq[1][b].x, dq[2][b].x, dq[3][b].x);
        *reinterpret_cast<uint4*>(fp8_dst + (((b * 2 + 1) * 16) ^ row_swizzle)) =
            make_uint4(dq[0][b].y, dq[1][b].y, dq[2][b].y, dq[3][b].y);
    }
}

// Streaming half-row decode from the packed scratch: no in-place hazard, so
// no internal barriers; the caller synchronizes once after all threads finish.
template <bool kUsePRMTGroups, bool kIntDecode = false>
__device__ __forceinline__ void dequant_smem_b_from_packed_half_row_unscaled(
        uint8_t* __restrict__ smem_b,
        const uint8_t* __restrict__ packed_b,
        const uint32_t tid,
        uint8_t* __restrict__ coeff_smem) {
    const uint32_t row = tid & 127u;
    const uint32_t half = tid >> 7u;
    const uint8_t* __restrict__ row_ptr = packed_b + row * 80;
    const uint4 fp4_quad0 = *reinterpret_cast<const uint4*>(row_ptr + half * 32);
    const uint4 fp4_quad1 = *reinterpret_cast<const uint4*>(row_ptr + half * 32 + 16);
    const uint16_t scale_half =
        *reinterpret_cast<const uint16_t*>(row_ptr + 64 + half * 2);
    *reinterpret_cast<uint16_t*>(coeff_smem + row * 4 + half * 2) = scale_half;

    uint8_t* __restrict__ fp8_dst = smem_b + row * 128;
    const uint32_t row_swizzle = (row & 7u) << 4;
    #pragma unroll
    for (int quad_i = 0; quad_i < 2; ++quad_i) {
        const uint4 fp4_quad = quad_i == 0 ? fp4_quad0 : fp4_quad1;
        const int chunk_i0 = static_cast<int>(half) * 4 + quad_i * 2;
        const uint2 q0 = decode_mxfp4_split_pair<kUsePRMTGroups, kIntDecode>(fp4_quad.x);
        const uint2 q1 = decode_mxfp4_split_pair<kUsePRMTGroups, kIntDecode>(fp4_quad.y);
        *reinterpret_cast<uint4*>(fp8_dst + ((chunk_i0 * 16) ^ row_swizzle)) =
            make_uint4(q0.x, q0.y, q1.x, q1.y);

        const int chunk_i1 = chunk_i0 + 1;
        const uint2 q2 = decode_mxfp4_split_pair<kUsePRMTGroups, kIntDecode>(fp4_quad.z);
        const uint2 q3 = decode_mxfp4_split_pair<kUsePRMTGroups, kIntDecode>(fp4_quad.w);
        *reinterpret_cast<uint4*>(fp8_dst + ((chunk_i1 * 16) ^ row_swizzle)) =
            make_uint4(q2.x, q2.y, q3.x, q3.y);
    }
}

template <bool kUsePRMTGroups, uint32_t kNumThreads, uint32_t kBarrierIdx, bool kIntDecode = false>
__device__ __forceinline__ void dequant_smem_b_inplace_two_rows_unscaled(
        uint8_t* __restrict__ smem_b,
        const uint32_t tid,
        uint8_t* __restrict__ coeff_smem) {
    const uint32_t row0 = tid;
    const uint32_t row1 = tid + kNumThreads;
    const uint8_t* __restrict__ row_ptr0 = smem_b + row0 * 80;
    const uint8_t* __restrict__ row_ptr1 = smem_b + row1 * 80;
    uint4 fp4_quads0[4];
    uint4 fp4_quads1[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        fp4_quads0[i] = reinterpret_cast<const uint4*>(row_ptr0)[i];
        fp4_quads1[i] = reinterpret_cast<const uint4*>(row_ptr1)[i];
    }
    const uint32_t scale_word0 = *reinterpret_cast<const uint32_t*>(row_ptr0 + 64);
    const uint32_t scale_word1 = *reinterpret_cast<const uint32_t*>(row_ptr1 + 64);
    reinterpret_cast<uint32_t*>(coeff_smem)[row0] = scale_word0;
    reinterpret_cast<uint32_t*>(coeff_smem)[row1] = scale_word1;

    asm volatile("bar.sync %0, %1;" : : "n"(kBarrierIdx), "n"(kNumThreads));

    uint8_t* __restrict__ dst0 = smem_b + row0 * 128;
    uint8_t* __restrict__ dst1 = smem_b + row1 * 128;
    const uint32_t row_swizzle = (tid & 7u) << 4;
    #pragma unroll
    for (int quad_i = 0; quad_i < 4; ++quad_i) {
        #pragma unroll
        for (int pair_i = 0; pair_i < 2; ++pair_i) {
            const int chunk_i = quad_i * 2 + pair_i;
            const uint4 q0 = fp4_quads0[quad_i];
            const uint4 q1 = fp4_quads1[quad_i];
            const uint32_t packed00 = pair_i == 0 ? q0.x : q0.z;
            const uint32_t packed01 = pair_i == 0 ? q0.y : q0.w;
            const uint32_t packed10 = pair_i == 0 ? q1.x : q1.z;
            const uint32_t packed11 = pair_i == 0 ? q1.y : q1.w;
            const uint2 d00 = decode_mxfp4_split_pair<kUsePRMTGroups, kIntDecode>(packed00);
            const uint2 d01 = decode_mxfp4_split_pair<kUsePRMTGroups, kIntDecode>(packed01);
            const uint2 d10 = decode_mxfp4_split_pair<kUsePRMTGroups, kIntDecode>(packed10);
            const uint2 d11 = decode_mxfp4_split_pair<kUsePRMTGroups, kIntDecode>(packed11);
            const uint32_t offset = (chunk_i * 16) ^ row_swizzle;
            *reinterpret_cast<uint4*>(dst0 + offset) = make_uint4(d00.x, d00.y, d01.x, d01.y);
            *reinterpret_cast<uint4*>(dst1 + offset) = make_uint4(d10.x, d10.y, d11.x, d11.y);
        }
    }

    asm volatile("bar.sync %0, %1;" : : "n"(kBarrierIdx), "n"(kNumThreads));
}

template <bool kUsePRMTGroups, uint32_t kNumThreads, uint32_t kBarrierIdx, bool kIntDecode = false>
__device__ __forceinline__ void dequant_smem_b_inplace_half_row_unscaled(
        uint8_t* __restrict__ smem_b,
        const uint32_t tid,
        uint8_t* __restrict__ coeff_smem) {
    static_assert(kNumThreads == 256, "half-row decode requires two math warpgroups");
    const uint32_t row = tid & 127u;
    const uint32_t half = tid >> 7u;
    const uint8_t* __restrict__ row_ptr = smem_b + row * 80;
    const uint4 fp4_quad0 = *reinterpret_cast<const uint4*>(row_ptr + half * 32);
    const uint4 fp4_quad1 = *reinterpret_cast<const uint4*>(row_ptr + half * 32 + 16);
    const uint16_t scale_half =
        *reinterpret_cast<const uint16_t*>(row_ptr + 64 + half * 2);

#ifndef DG_MXFP4_PROBE_NO_DECODE_BARRIER
    asm volatile("bar.sync %0, %1;" : : "n"(kBarrierIdx), "n"(kNumThreads) : "memory");
#endif

    *reinterpret_cast<uint16_t*>(coeff_smem + row * 4 + half * 2) = scale_half;

    uint8_t* __restrict__ fp8_dst = smem_b + row * 128;
    const uint32_t row_swizzle = (row & 7u) << 4;
    #pragma unroll
    for (int quad_i = 0; quad_i < 2; ++quad_i) {
        const uint4 fp4_quad = quad_i == 0 ? fp4_quad0 : fp4_quad1;
        const int chunk_i0 = static_cast<int>(half) * 4 + quad_i * 2;
        const uint2 q0 = decode_mxfp4_split_pair<kUsePRMTGroups, kIntDecode>(fp4_quad.x);
        const uint2 q1 = decode_mxfp4_split_pair<kUsePRMTGroups, kIntDecode>(fp4_quad.y);
        *reinterpret_cast<uint4*>(fp8_dst + ((chunk_i0 * 16) ^ row_swizzle)) =
            make_uint4(q0.x, q0.y, q1.x, q1.y);

        const int chunk_i1 = chunk_i0 + 1;
        const uint2 q2 = decode_mxfp4_split_pair<kUsePRMTGroups, kIntDecode>(fp4_quad.z);
        const uint2 q3 = decode_mxfp4_split_pair<kUsePRMTGroups, kIntDecode>(fp4_quad.w);
        *reinterpret_cast<uint4*>(fp8_dst + ((chunk_i1 * 16) ^ row_swizzle)) =
            make_uint4(q2.x, q2.y, q3.x, q3.y);
    }

#ifndef DG_MXFP4_PROBE_NO_DECODE_BARRIER
    asm volatile("bar.sync %0, %1;" : : "n"(kBarrierIdx), "n"(kNumThreads) : "memory");
#endif
    cutlass::arch::fence_view_async_shared();
}

} // namespace mxfp4

// ============================================================================
// SM90 (Hopper) MXFP4 MegaMoE derived from the optimized FP8 split core.
// ----------------------------------------------------------------------------
// Pipeline (cluster=1, no TMA multicast):
//   * Dispatch warps: pull tokens (FP8) and SF (per-128 channel float) from
//     remote ranks via NVLink into the local L1 pool.
//   * GEMM TMA-load warps (1 for A+SFA, 1 for B+SFB) feed the pipeline stages.
//   * Math warpgroups (1 or 2, totalling kNumEpilogueThreads) consume each
//     stage with WGMMA, accumulate into registers, then run the epilogue:
//       - L1 (Linear1): SwiGLU with gate/up granularity-8 interleaved layout,
//         per-row amax over the 64 post-SwiGLU columns of this block, FP8 e4m3
//         quantize, STSM into SMEM, TMA store to local L1 output buffer.
//         The per-row SF is written as a *float* into the L2-acts SF buffer at
//         per-64 K granularity (one SF per L1 N block), so each block is fully
//         self-contained and no cross-CTA amax synchronisation is needed.
//       - L2 (Linear2): BF16 cast of the GEMM output, STSM into SMEM, then
//         NVLink scatter to remote combine buffers.
//   * After all GEMM blocks, the math warps run the COMBINE step (top-k
//     reduction in BF16) — ported verbatim from the SM100 kernel.
// ============================================================================

enum class MegaMoEPhaseKind {
    Linear1,
    Linear2
};

template <MegaMoEPhaseKind kKind>
struct MegaMoEPhasePolicy {
    static constexpr bool is_linear1_only = kKind == MegaMoEPhaseKind::Linear1;
    static constexpr bool is_linear2_only = kKind == MegaMoEPhaseKind::Linear2;
    static constexpr bool runs_linear1 = is_linear1_only;
    static constexpr bool runs_linear2 = is_linear2_only;
    static constexpr bool needs_dispatch_pull = runs_linear1;
    static constexpr bool needs_combine = runs_linear2;

    template <typename Scheduler, typename Func>
    CUTLASS_DEVICE static void for_each_selected_block(Scheduler& scheduler, Func&& func) {
        if constexpr (is_linear1_only) {
            scheduler.template for_each_phase_block<sched::BlockPhase::Linear1>(
                [&](const uint32_t& local_expert_idx, const uint32_t& num_k_blocks,
                    const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                    func(sched::BlockPhase::Linear1, local_expert_idx,
                         num_k_blocks, m_block_idx, n_block_idx);
                });
        } else {
            scheduler.template for_each_phase_block<sched::BlockPhase::Linear2>(
                [&](const uint32_t& local_expert_idx, const uint32_t& num_k_blocks,
                    const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                    func(sched::BlockPhase::Linear2, local_expert_idx,
                         num_k_blocks, m_block_idx, n_block_idx);
                });
        }
    }
};

using MegaMoELinear1Phase = MegaMoEPhasePolicy<MegaMoEPhaseKind::Linear1>;
using MegaMoELinear2Phase = MegaMoEPhasePolicy<MegaMoEPhaseKind::Linear2>;

#define DG_SM90_MXFP4_MOE_TEMPLATE_PARAMS \
    uint32_t kNumMaxTokensPerRank, \
    uint32_t kHidden, uint32_t kIntermediateHidden, \
    uint32_t kNumExperts, uint32_t kNumTopk, \
    uint32_t kNumExpertsPerWave, \
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, \
    uint32_t kNumRingTokens, \
    uint32_t kNumSFRingTokens, \
    uint32_t kSFRingStrideTokens, \
    uint32_t kNumStages, \
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads, \
    uint32_t kNumEpilogueThreads, \
    uint32_t kClusterSize, \
    uint32_t kNumSMs, uint32_t kNumRanks, \
    float kActivationClamp, \
    bool kFastMath, \
    bool kDirectL2ScatterRequested = false, \
    bool kPhaseProfileRequested = false, \
    bool kL2NMajorScheduleRequested = false, \
    bool kOneWarpCleanupRequested = false, \
    bool kMXFP4SwapAB = false, \
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2, \
    uint32_t L1_SHAPE_K = kHidden, \
    uint32_t L2_SHAPE_N = kHidden, \
    uint32_t L2_SHAPE_K = kIntermediateHidden, \
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32, \
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32, \
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32, \
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4, \
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads, \
    uint32_t kNumTokensPerWarp = 32 / kNumTopk, \
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks

#define DG_SM90_MXFP4_MOE_KERNEL_ARGS_DECL \
    void* y, \
    int* cumulative_local_expert_recv_stats, \
    const uint32_t num_tokens, \
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf, \
    const float* __restrict__ l1_global_scales, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights, \
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf, \
    const float* __restrict__ l2_global_scales

#define DG_SM90_MXFP4_MOE_CORE_ARGS_DECL \
    void* y, \
    int* cumulative_local_expert_recv_stats, \
    const uint32_t num_tokens, \
    const layout::SymBuffer<kNumRanks>& sym_buffer, \
    const cute::TmaDescriptor& tensor_map_l1_acts, \
    const cute::TmaDescriptor& tensor_map_l1_acts_sf, \
    const cute::TmaDescriptor& tensor_map_l1_weights, \
    const cute::TmaDescriptor& tensor_map_l1_weights_sf, \
    const float* __restrict__ l1_global_scales, \
    const cute::TmaDescriptor& tensor_map_l1_output, \
    const cute::TmaDescriptor& tensor_map_l2_acts, \
    const cute::TmaDescriptor& tensor_map_l2_acts_sf, \
    const cute::TmaDescriptor& tensor_map_l2_weights, \
    const cute::TmaDescriptor& tensor_map_l2_weights_sf, \
    const float* __restrict__ l2_global_scales

#define DG_SM90_MXFP4_MOE_KERNEL_ARGS \
    y, cumulative_local_expert_recv_stats, num_tokens, sym_buffer, \
    tensor_map_l1_acts, tensor_map_l1_acts_sf, tensor_map_l1_weights, \
    tensor_map_l1_weights_sf, \
    l1_global_scales, tensor_map_l1_output, tensor_map_l2_acts, \
    tensor_map_l2_acts_sf, tensor_map_l2_weights, tensor_map_l2_weights_sf, \
    l2_global_scales

#define DG_SM90_MXFP4_MOE_CORE_TEMPLATE_ARGS(PhasePolicy) \
    PhasePolicy, \
    kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, kNumTopk, \
    kNumExpertsPerWave, BLOCK_M, BLOCK_N, BLOCK_K, kNumRingTokens, \
    kNumSFRingTokens, kSFRingStrideTokens, kNumStages, kNumDispatchThreads, \
    kNumNonEpilogueThreads, kNumEpilogueThreads, kClusterSize, kNumSMs, \
    kNumRanks, kActivationClamp, kFastMath, kDirectL2ScatterRequested, \
    kPhaseProfileRequested, kL2NMajorScheduleRequested, kOneWarpCleanupRequested, kMXFP4SwapAB, \
    L1_SHAPE_N, \
    L1_SHAPE_K, L2_SHAPE_N, L2_SHAPE_K, kNumDispatchWarps, \
    kNumMMANonEpilogueWarps, kNumEpilogueWarps, kNumEpilogueWarpgroups, \
    kNumThreads, kNumTokensPerWarp, kNumExpertsPerRank

template <typename MegaMoEPhase, DG_SM90_MXFP4_MOE_TEMPLATE_PARAMS>
CUTLASS_DEVICE void
sm90_mxfp4_mega_moe_core(DG_SM90_MXFP4_MOE_CORE_ARGS_DECL) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // =====================================================================
    // Template checks
    // =====================================================================
    DG_STATIC_ASSERT(kNumDispatchThreads == 64 or kNumDispatchThreads % 128 == 0,
                     "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 64 or kNumNonEpilogueThreads == 128,
                     "Invalid number of GEMM TMA warps (2 or 4 warps expected)");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of math/epilogue threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT(kClusterSize == 1 or kClusterSize == 2, "Invalid cluster size");
    DG_STATIC_ASSERT(kNumSMs % kClusterSize == 0, "SM count must be divisible by cluster size");
    DG_STATIC_ASSERT(BLOCK_M % 64 == 0,
                     "BLOCK_M must be a multiple of WGMMA::M (64)");
    DG_STATIC_ASSERT(BLOCK_M == 64,
                     "Initial MXFP4 split implementation uses the proven M64 core");
    DG_STATIC_ASSERT(BLOCK_N == 128 or BLOCK_N == 256,
                     "MXFP4 split kernels support deployment BN128/BN256 layouts");
    DG_STATIC_ASSERT(BLOCK_K == 128, "BLOCK_K is fixed to 128 (per-128 SF)");
    DG_STATIC_ASSERT(kClusterSize == 1, "MXFP4 dequant currently requires cluster size 1");

    // =====================================================================
    // Thread / warp identification
    // =====================================================================
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

    // Prefetch the TMA descriptors used by this split phase.
    if (warp_idx == 0 and cute::elect_one_sync()) {
        if constexpr (MegaMoEPhase::runs_linear1) {
            cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
            cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
            cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
            cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        }
        if constexpr (MegaMoEPhase::runs_linear2) {
            cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
            cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
            cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        }
    }

    // =====================================================================
    // Workspaces and symmetric buffer slicing (mirror SM100 layout, except SF
    // for L2 activations uses per-64 K granularity)
    // =====================================================================
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk,
        kNumRingTokens);

    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto bf16_token_layout             = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    // Per-128 K float SF: 4 bytes per per-128 group => `kHidden / 32` bytes/token (same as SM100 packing)
    constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32);
    // Per-64 K float SF (SM90 only): 4 bytes per per-64 group => `kIntermediateHidden / 16` bytes/token
    constexpr auto fp8_intermediate_sf_layout    = layout::Data(kIntermediateHidden / 16);
    constexpr auto input_topk_idx_layout         = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = layout::Data(sizeof(float), false);

    // Registered input area
    const auto input_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer     = layout::Buffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

    // L1 input area
    const auto l1_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumRingTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumSFRingTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(l1_topk_weights_layout, 1, kNumRingTokens, l1_sf_buffer.get_end_ptr());

    // L2 input area
    const auto l2_token_buffer = layout::Buffer(fp8_intermediate_token_layout, 1, kNumRingTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer    = layout::Buffer(fp8_intermediate_sf_layout, 1, kNumSFRingTokens, l2_token_buffer.get_end_ptr());
    DG_STATIC_ASSERT(kSFRingStrideTokens <= kNumSFRingTokens,
                     "Logical SF ring stride must fit in the allocated SF ring capacity");
    // Ring protocol (P2 de-risked path): the host asserts the ring covers the
    // full pool, so pool indices never wrap (lap == 0 forever) and the legacy
    // arrival-count protocol maps 1:1 onto the ring `l1_full` counters, which
    // are indexed by the raw BLOCK_M-granularity pool block index below. The
    // `l1_empty` / `l2_full` / `l2_empty` counters are never incremented and
    // stay zero. Ring tightening (lap > 0) requires the full 4-counter
    // protocol and ring-phase SF/data addressing.

    // Combine input area
    const auto combine_token_buffer = layout::Buffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    // =====================================================================
    // GEMM data types and shape constants
    // =====================================================================
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::float_e4m3_t;
    constexpr bool kSplitNWarpgroups =
        BLOCK_M == 64 and kNumEpilogueWarpgroups > 1 and
        BLOCK_N % kNumEpilogueWarpgroups == 0 and
        ((BLOCK_N / kNumEpilogueWarpgroups == 64) or
         (BLOCK_N / kNumEpilogueWarpgroups == 128));
    constexpr bool kSplitMNWarpgroups =
        BLOCK_M == 128 and BLOCK_N == 256 and kNumEpilogueWarpgroups == 4;
    constexpr bool kSerialNWarpgroups = false;
    constexpr bool kWideNWarpgroups =
        BLOCK_N == 256 && kNumEpilogueWarpgroups == 1;
    constexpr uint32_t kWarpgroupSplitM = kSplitNWarpgroups ? 1 :
        (kSplitMNWarpgroups ? 2 : kNumEpilogueWarpgroups);
    constexpr uint32_t kWarpgroupSplitN = kSplitNWarpgroups ? kNumEpilogueWarpgroups :
        (kSplitMNWarpgroups ? 2 : 1);
    constexpr uint32_t WG_BLOCK_M = BLOCK_M / kWarpgroupSplitM;
    constexpr uint32_t WG_BLOCK_N = BLOCK_N / kWarpgroupSplitN;
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;       // post-SwiGLU tile N
    constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2; // post-SwiGLU per-WG N
    constexpr bool kSwapABEligible =
        kMXFP4SwapAB and kSplitNWarpgroups and BLOCK_M == 64 and BLOCK_N == 128 and
        kNumEpilogueWarpgroups == 2;
    constexpr bool kSwapABActive = kSwapABEligible;
    constexpr bool kMathDequantSplitN =
        kSplitNWarpgroups and kNumEpilogueThreads == 256 and
        kNumNonEpilogueThreads == 64;
    // MXFP4 decode has no per-scale LUT dependency, so the cheaper grouped-PRMT
    // selector layout is always profitable (unlike NVFP4's shape-gated policy).
    constexpr bool kUsePRMTGroups = true;
    // W4A8-int SMEM weight decode is per-phase: L1 weights are int4 whenever
    // DG_W4A8_INT is on; L2 weights are int4 only in full-int mode (hybrid
    // mode keeps L2 weights MXFP4).
    constexpr bool kIntDecodeB =
        DG_W4A8_INT && (MegaMoEPhase::runs_linear1 || DG_W4A8_INT_L2);
    // Inline QoQ: SMEM decode folds the per-group int8 scale s2 (coeff word's
    // low byte) into the emitted int8; the word's high 16 bits carry s1 (bf16).
    constexpr bool kQoQFoldB = kIntDecodeB && DG_W4A8_INT_QOQ && !DG_W4A8_INT_PRE;
    // RF decode: swapAB WGMMAs take the A operand from registers, decoded
    // per-thread straight from the packed scratch (fragment-ordered prepack).
    constexpr bool kRFDecode = kSwapABActive;
    constexpr uint32_t kSwapABTokenChunks = BLOCK_M / 8;
    DG_STATIC_ASSERT(not kSwapABEligible or (BLOCK_M % 8 == 0),
                     "swapAB epilogue token chunks assume BLOCK_M is a multiple of 8");
    constexpr bool kAsyncL1TMAStore = false;
    constexpr bool kSplitSFATMA = false;
    constexpr bool kDirectL2Scatter = (!kSwapABActive) && kDirectL2ScatterRequested && MegaMoEPhase::runs_linear2 &&
        (!kSerialNWarpgroups) && WG_BLOCK_N == 128;
    constexpr bool kL2DualAccum = false;
    constexpr bool kL1DualKAccum = false;
    constexpr bool kReuseAccumAsFinal = kSplitMNWarpgroups;
    using L1WGMMA   = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    using L2WGMMA   = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");
    DG_STATIC_ASSERT((!kSplitNWarpgroups) or
                     (BLOCK_M == 64 and (WG_BLOCK_N == 64 or WG_BLOCK_N == 128)),
                     "Split-N path expects M64N64 or M64N128 WGMMA consumers");

    // A is always CTA-local.  When kClusterSize=2 the scheduler pairs adjacent
    // M blocks with identical expert/N/K coordinates so the B TMA can multicast.
    constexpr uint32_t LOAD_BLOCK_M    = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N    = BLOCK_N;
    // Always stage packed rows in a separate scratch: decode becomes a pure
    // stream (no in-place overwrite hazard, no pre-store barrier drain).
    constexpr bool kPackedBScratch     = true;
    constexpr bool kPreDecodedB        = DG_W4A8_INT_PRE;
    DG_STATIC_ASSERT(!kPreDecodedB or DG_W4A8_INT, "PRE mode requires DG_W4A8_INT");
    DG_STATIC_ASSERT(!(kPreDecodedB and kSwapABActive),
                     "PRE mode is non-swapAB only (host must disable swapAB)");
    constexpr uint32_t kSwizzleAMode   = BLOCK_K * sizeof(a_dtype_t);   // 128
    constexpr uint32_t kSwizzleBMode   = BLOCK_K * sizeof(b_dtype_t);   // 128
    constexpr uint32_t kSwizzleCDMode  = 128;
    constexpr uint32_t kGranK          = 128;          // L1 acts SF, weights SF
    constexpr uint32_t kL2ActsSFGranK  = 64;           // L2 acts SF (per-64 K, SM90 only)

    // PDL: when launched with programmatic stream serialization, hold off
    // consuming the previous grid's outputs until they are visible.
    #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;" ::: "memory");
    #endif

    // =====================================================================
    // Shared memory layout
    // =====================================================================
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Combine reuses the pre-barrier SMEM region, so split L2 keeps this
    // dispatch scratch capacity even though it does not run dispatch pull.
    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment);
    // Per-stage E8M0 dequant coefficients: one uint32 (4 scale bytes, one per
    // 32 K-elements) per weight row, copied out of the packed rows by decode
    // and multiplied into the accumulator promotion.
    constexpr uint32_t SMEM_B_COEFF_SIZE_PER_STAGE = LOAD_BLOCK_N * 4u;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    // RF decode feeds WGMMA A straight from registers; the decoded-FP8 B
    // buffer is not needed and its SMEM is reclaimed for more stages.
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE =
        kRFDecode ? 0u : LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // Packed rows carry 64 value bytes (scales ride their own TMA); PRE mode
    // loads full 128B pre-decoded int8 rows instead.
    constexpr uint32_t B_LOAD_BYTES_PER_ROW = kPreDecodedB ? 128u : 64u;
    constexpr uint32_t SMEM_B_LOAD_SIZE_PER_STAGE = LOAD_BLOCK_N * B_LOAD_BYTES_PER_ROW;
    constexpr uint32_t SMEM_PACKED_B_SIZE_PER_STAGE = (kPackedBScratch and !kPreDecodedB) ?
        SMEM_B_LOAD_SIZE_PER_STAGE : 0u;
    // SFA per-stage must be sized for the larger of L1 (BLOCK_M floats) and L2
    // (two per-64-K halves). Each TMA destination must be 128B aligned.
    constexpr uint32_t kL2SFAHalfStride =
        math::constexpr_align<uint32_t>(BLOCK_M * sizeof(float), 128u) / sizeof(float);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = 2 * kL2SFAHalfStride * sizeof(float);
    // Block (128, 128) weight SF: 1 float per (BLOCK_N, BLOCK_K) tile for L2,
    // 2 floats (gate/up) for L1. Loaded by math warpgroup directly from global,
    // so no SMEM is needed.
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = 0;

    // CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte * num_wg) and
    // L2 BF16 (BLOCK_M * BLOCK_N * 2 bytes * num_wg).
    constexpr uint32_t SMEM_CD_ACCUM_SIZE = 0u;
    constexpr uint32_t SMEM_CD_L1_SIZE = MegaMoEPhase::runs_linear1 ?
        kNumEpilogueWarpgroups * WG_BLOCK_M * WG_L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) : 0u;
    constexpr uint32_t SMEM_CD_L2_SIZE = (!MegaMoEPhase::runs_linear2 || kDirectL2Scatter) ? 0u :
        kNumEpilogueWarpgroups * WG_BLOCK_M * WG_BLOCK_N * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_CD_L1_ASYNC_ELEMS =
        kNumEpilogueWarpgroups * WG_BLOCK_M * L1_OUT_BLOCK_N;
    constexpr uint32_t SMEM_CD_L1_ASYNC_SIZE = kAsyncL1TMAStore ?
        2 * SMEM_CD_L1_ASYNC_ELEMS * sizeof(cutlass::float_e4m3_t) : 0u;
    constexpr uint32_t SMEM_CD_SWAP_L1_FP32_SIZE =
        kSwapABActive ? BLOCK_M * L1_OUT_BLOCK_N * sizeof(float) : 0u;
    constexpr uint32_t SMEM_CD_SWAP_L1_FP8_SIZE =
        kSwapABActive ? BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) : 0u;
    constexpr uint32_t SMEM_CD_SWAP_L1_SIZE =
        kSwapABActive ? (SMEM_CD_SWAP_L1_FP32_SIZE + SMEM_CD_SWAP_L1_FP8_SIZE) : 0u;
    constexpr uint32_t SMEM_CD_OUTPUT_BASE_SIZE =
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_OUTPUT_UNALIGNED_SIZE =
        SMEM_CD_OUTPUT_BASE_SIZE > SMEM_CD_L1_ASYNC_SIZE ? SMEM_CD_OUTPUT_BASE_SIZE : SMEM_CD_L1_ASYNC_SIZE;
    constexpr uint32_t SMEM_CD_OUTPUT_WITH_SWAP_SIZE =
        SMEM_CD_OUTPUT_UNALIGNED_SIZE > SMEM_CD_SWAP_L1_SIZE ?
            SMEM_CD_OUTPUT_UNALIGNED_SIZE : SMEM_CD_SWAP_L1_SIZE;
    constexpr uint32_t SMEM_CD_OUTPUT_SIZE = math::constexpr_align(
        SMEM_CD_OUTPUT_WITH_SWAP_SIZE, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_ACCUM_SIZE + SMEM_CD_OUTPUT_SIZE;

    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE +
                      SMEM_PACKED_B_SIZE_PER_STAGE + SMEM_B_COEFF_SIZE_PER_STAGE);

    constexpr uint32_t kCombineHiddenBytes = kHidden * sizeof(nv_bfloat16);
    constexpr uint32_t kCombineChunkSlots = 3;
    constexpr uint32_t kCombineMaxRegistersForBuffer = 128;
    constexpr bool kCombineOneChunkFits =
        kCombineChunkSlots * kNumEpilogueWarps * kCombineHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE and
        kHidden <= 32 * kCombineMaxRegistersForBuffer;
    constexpr bool kCombineTwoChunksFits =
        kHidden % 2 == 0 and
        kCombineChunkSlots * kNumEpilogueWarps * (kCombineHiddenBytes / 2) <= SMEM_BEFORE_BARRIER_SIZE and
        kHidden <= 2 * 32 * kCombineMaxRegistersForBuffer;
    constexpr uint32_t kCombineNumChunks = kCombineOneChunkFits ? 1 :
        (kCombineTwoChunksFits ? 2 : 4);
    constexpr uint32_t kCombineChunkBytes = kCombineHiddenBytes / kCombineNumChunks;
    constexpr uint32_t SMEM_COMBINE_ALIAS_SIZE = MegaMoEPhase::needs_combine
        ? kCombineChunkSlots * kNumEpilogueWarps * kCombineChunkBytes : 0u;
    DG_STATIC_ASSERT(kHidden % kCombineNumChunks == 0, "Hidden must be divisible by number of combine chunks");
    DG_STATIC_ASSERT(SMEM_COMBINE_ALIAS_SIZE <= SMEM_BEFORE_BARRIER_SIZE,
                     "Combine SMEM alias exceeds the pre-barrier scratch region");

    // SMEM pointers
    auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));
    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE);

    auto smem_cd_base = math::advance_ptr<uint8_t>(smem_gemm_base, SMEM_CD_ACCUM_SIZE);
    // CD output is shared by L1 (FP8) and L2 (BF16); reinterpret-cast as needed.
    auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_cd_base);
    auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_cd_base);
    auto smem_cd_swap_l1_fp32 = reinterpret_cast<float*>(smem_cd_base);
    auto smem_cd_swap_l1_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(
        math::advance_ptr(smem_cd_base, SMEM_CD_SWAP_L1_FP32_SIZE));

    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto smem_packed_b = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint8_t*>(smem_gemm_base) + SMEM_CD_SIZE +
            kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) +
            i * SMEM_PACKED_B_SIZE_PER_STAGE;
    });
    auto smem_b_coeff = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint8_t*>(smem_gemm_base) + SMEM_CD_SIZE +
            kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE +
                          SMEM_PACKED_B_SIZE_PER_STAGE) +
            i * SMEM_B_COEFF_SIZE_PER_STAGE;
    });
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE +
                                     SMEM_PACKED_B_SIZE_PER_STAGE + SMEM_B_COEFF_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });

    // Barriers live after SF (SFB is loaded directly from global, no SMEM)
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE);
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
    auto empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages + i; });
    auto dequant_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i; });
    auto combine_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 3 + i; });

    // =====================================================================
    // Initialization
    // =====================================================================
    if (warp_idx == 0) {
        // Clean expert-count shared memory
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumExperts; i += 32)
            ptx::st_shared(smem_expert_count + i, 0u);
    } else if (warp_idx == 1) {
        // Init dispatch m-barriers
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM full/empty barriers and combine barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Producer arrivals: A(+SFA) + B, or A + B + SFA when
                // split-SFA uses an otherwise idle TMA warp.
                full_barriers[i]->init(kSplitSFATMA ? 3 : 2);
                // With cluster multicast the leader CTA's TMA warp waits on peer
                // empty barriers too, so every math warp releases both CTAs.
                empty_barriers[i]->init(kClusterSize * kNumEpilogueWarps);
                dequant_barriers[i]->init(1);
            }
            if constexpr (MegaMoEPhase::needs_combine) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                    combine_barriers[i]->init(1);
            }
        }
        cutlass::arch::fence_barrier_init();
    }
    if constexpr (kClusterSize > 1) {
        cute::cluster_sync();
    } else {
        __syncthreads();
    }

    // =====================================================================
    // Scheduler (cluster=1)
    // =====================================================================
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumExpertsPerWave,
        kNumSMs, kNumRanks, kClusterSize, kL2NMajorScheduleRequested, false>(workspace);

    // Pipeline state shared by TMA loaders and math warpgroups
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    const auto dequant_loaded_b_stage = [&](const uint32_t& stage,
                                            const uint32_t& stage_phase,
                                            const uint32_t& non_epilogue_thread_idx) {
        if constexpr (kPreDecodedB) {
            // B lands in smem_b already decoded; nothing to do here.
        } else if constexpr (!kMathDequantSplitN) {
            full_barriers[stage]->wait(stage_phase);
            if constexpr (kPackedBScratch) {
                #pragma unroll
                for (uint32_t row = non_epilogue_thread_idx; row < BLOCK_N;
                     row += kNumNonEpilogueThreads) {
                    mxfp4::dequant_smem_b_from_packed_unscaled<kUsePRMTGroups, kIntDecodeB, kQoQFoldB>(
                        reinterpret_cast<uint8_t*>(smem_b[stage]),
                        smem_packed_b[stage], row, smem_b_coeff[stage]);
                }
                asm volatile("bar.sync 8, %0;" : : "n"(kNumNonEpilogueThreads));
                if (non_epilogue_thread_idx == 0)
                    dequant_barriers[stage]->arrive();
            } else if constexpr (kNumNonEpilogueThreads == 64) {
                mxfp4::dequant_smem_b_inplace_two_rows_unscaled<
                    kUsePRMTGroups, 64u, 8u, kIntDecodeB>(
                    reinterpret_cast<uint8_t*>(smem_b[stage]),
                    non_epilogue_thread_idx, smem_b_coeff[stage]);
                if (non_epilogue_thread_idx == 0)
                    dequant_barriers[stage]->arrive();
            } else if (non_epilogue_thread_idx >= 64u) {
                const uint32_t dequant_tid = non_epilogue_thread_idx - 64u;
                mxfp4::dequant_smem_b_inplace_two_rows_unscaled<
                    kUsePRMTGroups, 64u, 8u, kIntDecodeB>(
                    reinterpret_cast<uint8_t*>(smem_b[stage]),
                    dequant_tid, smem_b_coeff[stage]);
                if (dequant_tid == 0)
                    dequant_barriers[stage]->arrive();
            }
        }
    };

    // Intra-SM barrier indices (mirroring SM100)
    constexpr uint32_t kDispatchBarrierIdx              = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx  = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx          = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx       = 3;

    // Cross-rank NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag    = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag   = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag   = 3;

    // Register reconfiguration counts (chosen to fit in 64512 reg budget).
    // For the 256-epilogue-thread case (block_m=128, 2 math WGs):
    //   128*48 + 128*40 + 256*208 = 64512 exactly.
    // The 512-epilogue-thread split-MN path trims front-end roles so the
    // four WGMMA warpgroups fit under the 64K CTA register budget.
    constexpr uint32_t kNumDispatchRegisters    = kNumEpilogueThreads == 512 ? 32 : 48;
    constexpr bool kCompactFrontendWarpgroup = (kNumDispatchWarps == 2 and kNumMMANonEpilogueWarps == 2);
    constexpr uint32_t kNumNonEpilogueRegisters = kNumEpilogueThreads == 512 ? 24 :
        (kCompactFrontendWarpgroup ? kNumDispatchRegisters : 40);
    constexpr uint32_t kNumEpilogueRegisters    = kNumEpilogueThreads == 512 ? 112 :
        ((kSerialNWarpgroups or kWideNWarpgroups) ? 256 : 208);
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    constexpr uint32_t kProfileDispatchTotal = 0;
    constexpr uint32_t kProfileDispatchPull = 1;
    constexpr uint32_t kProfileMathLoop = 2;
    constexpr uint32_t kProfileCombineBarrier = 3;
    constexpr uint32_t kProfileCombineReduce = 4;
    constexpr uint32_t kProfileGemmCore = 5;
    constexpr uint32_t kProfileL1Epilogue = 6;
    constexpr uint32_t kProfileL2Epilogue = 7;
    constexpr uint32_t kNumProfileMetrics = 8;
    const auto phase_profile_clock = [&]() -> unsigned long long {
        if constexpr (kPhaseProfileRequested) {
            unsigned long long t;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            return t;
        } else {
            return 0ull;
        }
    };
    const auto phase_profile_record = [&](const uint32_t& metric, const unsigned long long& cycles) {
        if constexpr (kPhaseProfileRequested) {
            if (cumulative_local_expert_recv_stats != nullptr and cycles > 0) {
                auto profile = reinterpret_cast<unsigned long long*>(
                    cumulative_local_expert_recv_stats + kNumExpertsPerRank);
                atomicAdd(profile + metric, cycles);
                atomicMax(profile + kNumProfileMetrics + metric, cycles);
                atomicAdd(profile + 2 * kNumProfileMetrics + metric, 1ull);
            }
        }
    };

    const auto for_each_selected_block = [&](auto&& func) {
        MegaMoEPhase::for_each_selected_block(scheduler, func);
    };

    const auto cleanup_workspace = [&]() {
        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
                const auto cleanup_pool_block_offset = scheduler.get_pool_block_offset(i);

                if constexpr (!kOneWarpCleanupRequested)
                    ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if constexpr (kOneWarpCleanupRequested) {
                    if (warp_idx == 0) {
                        if (lane_idx == 0) {
                            *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                            if (cumulative_local_expert_recv_stats != nullptr)
                                ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                        }
                    }
                } else {
                    if (warp_idx == 0) {
                        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                    } else if (warp_idx == 1) {
                        if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                            ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                        __syncwarp();
                    }
                }

                if constexpr (kOneWarpCleanupRequested) {
                    if (warp_idx == 0) {
                        for (uint32_t j = lane_idx; j < num_recv_m_blocks; j += 32) {
                            *workspace.get_l1_full_count_ptr(cleanup_pool_block_offset + j) = 0;
                        }
                        __syncwarp();
                    }
                } else {
                    for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                        *workspace.get_l1_full_count_ptr(cleanup_pool_block_offset + j) = 0;
                    }
                    __syncwarp();
                }
            }
        }
    };

    // =====================================================================
    // ROLE 1: DISPATCH WARPS
    //   Mirrors SM100 dispatch with two changes:
    //     * SF is per-128 channel float (no UTCCP transpose). We store the
    //       remote per-token SF directly into the local L1 SF buffer in
    //       MN-major layout: `local_sf[k_chunk * num_padded_sf_pool_tokens + token_idx]`.
    //     * The "token_idx_in_expert" → SF token index is now the simple
    //       per-block linear mapping (no 4×32 transpose).
    // =====================================================================
    if (warp_idx < kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();
        const unsigned long long dispatch_total_start = phase_profile_clock();

        if constexpr (MegaMoEPhase::is_linear2_only) {
            scheduler.fetch_expert_recv_count();
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
            ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
            cleanup_workspace();
            comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                                 kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
                workspace, sym_buffer, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
                true, false);
            return;
        }

        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx = [&](const auto& process) {
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    const int expert_idx = static_cast<int>(
                        __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count tokens per expert
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Stake out per-expert SM offsets via global atomic
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source token-topk indices to remote ranks
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            false, true);

        // Sync with epilogue warps before pulling tokens
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        const unsigned long long dispatch_pull_start = phase_profile_clock();

        // Token / SF pull loop
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = dispatch_barriers[warp_idx];

        scheduler.fetch_expert_recv_count();

        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int      current_expert_idx = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank)
                    break;
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }
            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }

            // Round-robin rank selection (identical to SM100)
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane = 0xffffffff;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }
                slot_idx -= num_round_tokens;
                offset += length;
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

            // TMA pull token data into SMEM
            if (cute::elect_one_sync()) {
                ptx::tma_load_1d(
                    pull_buffer.get_base_ptr(),
                    sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                   current_rank_in_expert_idx),
                    pull_mbarrier, kHidden);
            }
            __syncwarp();

            // Copy SF: per-128 K floats, written linearly (no UTCCP transpose).
            constexpr uint32_t kNumSFFloats = kHidden / 128;
            DG_STATIC_ASSERT(kNumSFFloats > 0 and kHidden % 128 == 0, "Invalid SF");
            const auto remote_sf_ptr = sym_buffer.map(
                input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>(),
                current_rank_in_expert_idx);
            const auto local_sf_ptr  = l1_sf_buffer.get_base_ptr<float>();
            const uint32_t sf_pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFFloats)
                    local_sf_ptr[j * kSFRingStrideTokens + sf_pool_token_idx] = remote_sf_ptr[j];
            }
            __syncwarp();

            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            if (cute::elect_one_sync()) {
                const auto weight = *sym_buffer.map(
                    input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);
                *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;

                ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                ptx::tma_store_1d(
                    l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                    pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
                ptx::red_add_rel(
                    workspace.get_l1_full_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
            }
            __syncwarp();
        }



        // Cleanup workspace, overlapping with combine
        const unsigned long long dispatch_pull_end = phase_profile_clock();
        if (lane_idx == 0) {
            phase_profile_record(kProfileDispatchPull, dispatch_pull_end - dispatch_pull_start);
            phase_profile_record(kProfileDispatchTotal, dispatch_pull_end - dispatch_total_start);
        }
        if constexpr (MegaMoEPhase::is_linear1_only)
            return;

        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        cleanup_workspace();

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            true, false);

    // =====================================================================
    // ROLE 2: GEMM TMA LOAD warps (load A+SFA, B+SFB)
    //   Default: 4 non-epilogue warps, two active and two idle.
    //   Compact frontend mode: 2 dispatch warps + 2 TMA warps share the first
    //   warpgroup, reducing total CTA threads for the M128/2WG path.
    // =====================================================================
    } else if (warp_idx == kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        for_each_selected_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            (void)block_phase;
            constexpr bool is_linear1_phase = MegaMoEPhase::runs_linear1;
            const auto tensor_map_a_ptr = !is_linear1_phase
                ? &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = !is_linear1_phase
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const bool has_valid_m = valid_m > 0;

            // Wait for the pool to be ready. Cluster peers can be dummy CTAs for
            // the tail M unit when an expert has an odd number of M blocks.
            if (has_valid_m) {
                if (is_linear1_phase) {
                    const auto ptr = workspace.get_l1_full_count_ptr(pool_block_idx);
                    const auto expected = valid_m;
                    while (ptx::ld_acq(ptr) != expected);
                }
            }
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    if (has_valid_m) {
                    const uint32_t m_idx = pool_block_idx * BLOCK_M;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    // TMA load A
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                        k_idx, m_idx, 1);

                    if constexpr (kSplitSFATMA) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE);
                    } else {
                        // TMA load SFA
                        if (is_linear1_phase) {
                            // L1 SFA per-128: load (BLOCK_M, 1) at K=k_block_idx
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                                m_idx, k_block_idx, 1);
                            full_barriers[stage_idx]->arrive_and_expect_tx(
                                SMEM_A_SIZE_PER_STAGE + BLOCK_M * sizeof(float));
                        } else {
                            // L2 SFA per-64: descriptor box is (block_mn, 1) (see make_tma_sf_desc),
                            // so we must issue two single-group TMAs and place them at smem offsets
                            // 0 and BLOCK_M to match math's load offsets (`+ 0 * BLOCK_M` / `+ 1 * BLOCK_M`).
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                                m_idx, k_block_idx * 2, 1);
                            tma::copy<BLOCK_M, 1, 0, float>(
                                tensor_map_sfa_ptr, full_barriers[stage_idx],
                                smem_sfa[stage_idx] + kL2SFAHalfStride,
                                m_idx, k_block_idx * 2 + 1, 1);
                            full_barriers[stage_idx]->arrive_and_expect_tx(
                                SMEM_A_SIZE_PER_STAGE + 2 * BLOCK_M * sizeof(float));
                        }
                    }
                    } else {
                        full_barriers[stage_idx]->arrive();
                    }
                }
                __syncwarp();
                dequant_loaded_b_stage(stage_idx, phase, lane_idx);
            }
        });

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        for_each_selected_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            (void)block_phase;
            constexpr bool is_linear1_phase = MegaMoEPhase::runs_linear1;
            const auto tensor_map_b_ptr =
                !is_linear1_phase ? &tensor_map_l2_weights : &tensor_map_l1_weights;

            const uint32_t shape_n = !is_linear1_phase ? L2_SHAPE_N : L1_SHAPE_N;

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                    const uint32_t k_idx = k_block_idx * B_LOAD_BYTES_PER_ROW;

                    // Each BK128 row contains 64B packed E2M1, 8B UE4M3, and
                    // 8B padding. BN128 expands in place; BN256 uses scratch.
                    if constexpr (kPreDecodedB) {
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, uint8_t>(
                            tensor_map_b_ptr, full_barriers[stage_idx],
                            reinterpret_cast<uint8_t*>(smem_b[stage_idx]),
                            k_idx, n_idx, kClusterSize);
                    } else {
                    tma::copy<B_LOAD_BYTES_PER_ROW, LOAD_BLOCK_N, 0, uint8_t>(
                        tensor_map_b_ptr, full_barriers[stage_idx],
                        kPackedBScratch ? smem_packed_b[stage_idx] :
                            reinterpret_cast<uint8_t*>(smem_b[stage_idx]),
                        k_idx, n_idx, kClusterSize);
                    }

                    // E8M0 scale plane: one 512B tile-major row per stage,
                    // delivered straight into the coefficient SMEM.
                    const auto tensor_map_sf_ptr =
                        !is_linear1_phase ? &tensor_map_l2_weights_sf : &tensor_map_l1_weights_sf;
                    const uint32_t kb_total = (!is_linear1_phase ? L2_SHAPE_K : L1_SHAPE_K) / BLOCK_K;
                    const uint32_t sf_row =
                        (local_expert_idx * (shape_n / BLOCK_N) + n_block_idx) * kb_total + k_block_idx;
                    constexpr uint32_t kSFRowsPerTile = BLOCK_N * 4 / 256;
                    tma::copy<256, kSFRowsPerTile, 0, uint8_t>(
                        tensor_map_sf_ptr, full_barriers[stage_idx],
                        smem_b_coeff[stage_idx], 0, sf_row * kSFRowsPerTile, kClusterSize);

                    full_barriers[stage_idx]->arrive_and_expect_tx(
                        SMEM_B_LOAD_SIZE_PER_STAGE + BLOCK_N * 4);
                }
                __syncwarp();
                dequant_loaded_b_stage(stage_idx, phase, 32u + lane_idx);
            }
        });

    } else if (kSplitSFATMA && warp_idx == kNumDispatchWarps + 2) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        for_each_selected_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            (void)block_phase;
            constexpr bool is_linear1_phase = MegaMoEPhase::runs_linear1;
            (void)local_expert_idx;
            (void)n_block_idx;
            const auto tensor_map_sfa_ptr = !is_linear1_phase
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const bool has_valid_m = valid_m > 0;

            if (has_valid_m) {
                if (is_linear1_phase) {
                    const auto ptr = workspace.get_l1_full_count_ptr(pool_block_idx);
                    const auto expected = valid_m;
                    while (ptx::ld_acq(ptr) != expected);
                }
            }

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    if (has_valid_m) {
                    const uint32_t m_idx = pool_block_idx * BLOCK_M;

                    if (is_linear1_phase) {
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            m_idx, k_block_idx, 1);
                        full_barriers[stage_idx]->arrive_and_expect_tx(BLOCK_M * sizeof(float));
                    } else {
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                            m_idx, k_block_idx * 2, 1);
                        tma::copy<BLOCK_M, 1, 0, float>(
                            tensor_map_sfa_ptr, full_barriers[stage_idx],
                            smem_sfa[stage_idx] + kL2SFAHalfStride,
                            m_idx, k_block_idx * 2 + 1, 1);
                        full_barriers[stage_idx]->arrive_and_expect_tx(2 * BLOCK_M * sizeof(float));
                    }
                    } else {
                        full_barriers[stage_idx]->arrive();
                    }
                }
                __syncwarp();
            }
        });

    } else if (warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Idle non-epilogue warps (kNumDispatchWarps+2, +3). They must still
        // participate in the warpgroup-collective `setmaxnreg.dec.sync.aligned`
        // so that the math warpgroup's `warpgroup_reg_alloc` can succeed.
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        const uint32_t non_epilogue_warp_idx = warp_idx - kNumDispatchWarps;
        const uint32_t non_epilogue_thread_idx = non_epilogue_warp_idx * 32 + lane_idx;
        for_each_selected_block([&](const sched::BlockPhase&,
                                     const uint32_t&, const uint32_t& num_k_blocks,
                                     const uint32_t&, const uint32_t&) {
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;
                 advance_pipeline(k_block_idx)) {
                dequant_loaded_b_stage(stage_idx, phase, non_epilogue_thread_idx);
                __syncwarp();
            }
        });

    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
    // =====================================================================
    // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
    // =====================================================================
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        const uint32_t epilogue_warp_idx  = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const uint32_t epilogue_wg_idx    = epilogue_warp_idx / 4;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t warp_idx_in_wg     = epilogue_warp_idx % 4;

        const auto wait_and_dequant_b_stage = [&](const uint32_t& stage,
                                                   const uint32_t& stage_phase,
                                                   const bool skip_decode = false) {
            if constexpr (kRFDecode or kPreDecodedB) {
                // RF: weights stay packed in the scratch, math threads decode
                // their own WGMMA A fragments at issue time. PRE: B arrived
                // already decoded via TMA. Either way, only the full barrier.
                full_barriers[stage]->wait(stage_phase);
            } else if constexpr (kMathDequantSplitN) {
                full_barriers[stage]->wait(stage_phase);
                if (skip_decode)
                    return;  // already decoded in the previous block's WGMMA shadow
                if constexpr (kPackedBScratch and BLOCK_N == 128) {
                    mxfp4::dequant_smem_b_from_packed_half_row_unscaled<kUsePRMTGroups, kIntDecodeB>(
                        reinterpret_cast<uint8_t*>(smem_b[stage]),
                        smem_packed_b[stage], epilogue_thread_idx,
                        smem_b_coeff[stage]);
                    asm volatile("bar.sync 8, 256;" ::: "memory");
                    cutlass::arch::fence_view_async_shared();
                } else if constexpr (kPackedBScratch) {
                    mxfp4::dequant_smem_b_from_packed_unscaled<kUsePRMTGroups, kIntDecodeB, kQoQFoldB>(
                        reinterpret_cast<uint8_t*>(smem_b[stage]),
                        smem_packed_b[stage], epilogue_thread_idx,
                        smem_b_coeff[stage]);
                    asm volatile("bar.sync 8, 256;" ::: "memory");
                    cutlass::arch::fence_view_async_shared();
                } else {
#ifndef DG_MXFP4_PROBE_SKIP_DECODE
                    mxfp4::dequant_smem_b_inplace_half_row_unscaled<
                        kUsePRMTGroups, 256u, 8u, kIntDecodeB>(
                        reinterpret_cast<uint8_t*>(smem_b[stage]),
                        epilogue_thread_idx, smem_b_coeff[stage]);
#else
                    asm volatile("bar.sync 8, 256;" ::: "memory");
                    cutlass::arch::fence_view_async_shared();
#endif
                }
            } else {
                dequant_barriers[stage]->wait(stage_phase);
            }
        };

        uint32_t async_l1_store_stage = 0;
        bool async_l1_store_pending[2] = {false, false};

        const auto arrive_empty_barrier = [&](const uint32_t& s) {
            if constexpr (kClusterSize == 1) {
                if (lane_idx == 0)
                    empty_barriers[s]->arrive();
            } else {
                if (lane_idx < kClusterSize)
                    empty_barriers[s]->arrive(lane_idx);
            }
        };

        const auto drain_async_l1_store_stage = [&](const uint32_t& store_stage) {
            if constexpr (kAsyncL1TMAStore) {
                if (async_l1_store_pending[store_stage]) {
                    // Two SMEM L1 store buffers are used in FIFO order; waiting
                    // for <=1 outstanding store makes the older buffer reusable.
                    ptx::tma_store_wait<1>();
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    async_l1_store_pending[store_stage] = false;
                }
            }
        };

        const auto drain_all_async_l1_stores = [&]() {
            if constexpr (kAsyncL1TMAStore) {
                if (async_l1_store_pending[0] or async_l1_store_pending[1]) {
                    ptx::tma_store_wait<0>();
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    async_l1_store_pending[0] = false;
                    async_l1_store_pending[1] = false;
                }
            }
        };

        // WGMMA-output register layout helpers
        const uint32_t row_idx = lane_idx / 4;
        const uint32_t col_idx = lane_idx % 4;
        const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
        const uint32_t r_1 = r_0 + 8;
        constexpr uint32_t WG_SMEM_CD_L1_STRIDE_N = kSplitNWarpgroups ? L1_OUT_BLOCK_N : WG_L1_OUT_BLOCK_N;

        DG_STATIC_ASSERT(kWarpgroupSplitM * kWarpgroupSplitN == kNumEpilogueWarpgroups, "Invalid warpgroup split");
        if constexpr (kSplitNWarpgroups or kSplitMNWarpgroups) {
            DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M and WG_BLOCK_N == L1WGMMA::N,
                             "Split WGs must each run one WGMMA tile per K-block");
        } else if constexpr (kSerialNWarpgroups) {
            DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M and WG_BLOCK_N == L1WGMMA::N,
                             "Serial-N path runs two M64N128 WGMMAs per K-block");
        } else {
            DG_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M, "Each warpgroup must run exactly one WGMMA per K-block");
        }

        // Sync with dispatch
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        const unsigned long long math_loop_start = phase_profile_clock();

        for_each_selected_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            (void)block_phase;
            constexpr bool is_linear1_phase = MegaMoEPhase::runs_linear1;
            const uint32_t valid_m = scheduler.template get_valid_m<false>();
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t m_idx = pool_block_idx * BLOCK_M;
            const uint32_t epilogue_wg_m_idx = epilogue_wg_idx / kWarpgroupSplitN;
            const uint32_t epilogue_wg_n_idx = epilogue_wg_idx - epilogue_wg_m_idx * kWarpgroupSplitN;
            const uint32_t wg_n_idx = epilogue_wg_n_idx * WG_BLOCK_N;
            const uint32_t wg_l1_out_n_idx = epilogue_wg_n_idx * WG_L1_OUT_BLOCK_N;
            const uint32_t n_idx = n_block_idx * BLOCK_N + wg_n_idx;
            const uint32_t row_block_offset = epilogue_wg_m_idx * WG_BLOCK_M;
            const uint32_t smem_cd_l1_wg_offset = kSplitNWarpgroups ? 0 :
                epilogue_wg_idx * WG_BLOCK_M * WG_L1_OUT_BLOCK_N;
            const uint32_t row_offset_r0 = row_block_offset + r_0;
            const uint32_t row_offset_r1 = row_block_offset + r_1;
            const bool valid_r0 = row_offset_r0 < valid_m;
            const bool valid_r1 = row_offset_r1 < valid_m;
            const float expert_global_scale = is_linear1_phase ?
                (l1_global_scales == nullptr ?
                    1.0f : __ldg(l1_global_scales + local_expert_idx)) :
                (l2_global_scales == nullptr ?
                    1.0f : __ldg(l2_global_scales + local_expert_idx));


            if constexpr (kAsyncL1TMAStore) {
                if (!is_linear1_phase)
                    drain_all_async_l1_stores();
            }

            if constexpr (kSerialNWarpgroups) {
                using WGMMA = L1WGMMA;
                constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;
                constexpr uint32_t kNumSerialN = 2;
                float final_accum[kNumSerialN][kAccumPerThread] = {};
                float accum[kAccumPerThread];

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    wait_and_dequant_b_stage(stage_idx, phase);

                    float scale_a_0_lo, scale_a_1_lo;
                    float scale_a_0_hi, scale_a_1_hi;
                    if (is_linear1_phase) {
                        scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                        scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                    } else {
                        scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                        scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                        scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r0);
                        scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r1);
                    }

                    constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                    constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
                    constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
                    constexpr uint32_t kL1SFPerExpert = (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                    constexpr uint32_t kL2SFPerExpert = (kHidden / 128) * kL2SFKBlocks;

                    #pragma unroll
                    for (uint32_t serial_n_idx = 0; serial_n_idx < kNumSerialN; ++serial_n_idx) {
                        const uint32_t serial_wg_n_idx = serial_n_idx * WG_BLOCK_N;
                        float gate_sf = 0.0f, up_sf = 0.0f, l2_sf = 0.0f;
                        if (is_linear1_phase) {
                            const float global_scale = l1_global_scales == nullptr ?
                                1.0f : __ldg(l1_global_scales + local_expert_idx);
                            gate_sf = global_scale;
                            up_sf = global_scale;

                            // One WGMMA batch per 32-K group: the E8M0 weight
                            // coefficient changes at K32 granularity, so promote
                            // after every WGMMA::K(=32) slice.
                            #pragma unroll
                            for (uint32_t k32 = 0; k32 < BLOCK_K / WGMMA::K; ++ k32) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_arrive();
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K + k32 * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + serial_wg_n_idx * BLOCK_K + k32 * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, false);
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_wait<0>();

                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                    const float sb = (i & 1u) ? up_sf : gate_sf;
                                    const uint32_t n_0 = serial_wg_n_idx + i * 8 + col_idx * 2;
                                    const float cw_0 = mxfp4::e8m0_to_float(smem_b_coeff[stage_idx][n_0 * 4 + k32]);
                                    const float cw_1 = mxfp4::e8m0_to_float(smem_b_coeff[stage_idx][(n_0 + 1) * 4 + k32]);
                                    final_accum[serial_n_idx][i*4+0] += scale_a_0_lo * sb * cw_0 * accum[i*4+0];
                                    final_accum[serial_n_idx][i*4+1] += scale_a_0_lo * sb * cw_1 * accum[i*4+1];
                                    final_accum[serial_n_idx][i*4+2] += scale_a_1_lo * sb * cw_0 * accum[i*4+2];
                                    final_accum[serial_n_idx][i*4+3] += scale_a_1_lo * sb * cw_1 * accum[i*4+3];
                                }
                            }
                        } else {
                            l2_sf = l2_global_scales == nullptr ?
                                1.0f : __ldg(l2_global_scales + local_expert_idx);

                            // Per-32-K batches; the L2 activation SF is per-64-K,
                            // so the first two K32 groups use the lo half and the
                            // last two use the hi half.
                            #pragma unroll
                            for (uint32_t k32 = 0; k32 < BLOCK_K / WGMMA::K; ++ k32) {
                                const float scale_a_0 = k32 < 2 ? scale_a_0_lo : scale_a_0_hi;
                                const float scale_a_1 = k32 < 2 ? scale_a_1_lo : scale_a_1_hi;
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_arrive();
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K + k32 * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + serial_wg_n_idx * BLOCK_K + k32 * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, accum, false);
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                                ptx::warpgroup_wait<0>();

                                #pragma unroll
                                for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                    const uint32_t n_0 = serial_wg_n_idx + i * 8 + col_idx * 2;
                                    const float cw_0 = mxfp4::e8m0_to_float(smem_b_coeff[stage_idx][n_0 * 4 + k32]);
                                    const float cw_1 = mxfp4::e8m0_to_float(smem_b_coeff[stage_idx][(n_0 + 1) * 4 + k32]);
                                    final_accum[serial_n_idx][i*4+0] += scale_a_0 * l2_sf * cw_0 * accum[i*4+0];
                                    final_accum[serial_n_idx][i*4+1] += scale_a_0 * l2_sf * cw_1 * accum[i*4+1];
                                    final_accum[serial_n_idx][i*4+2] += scale_a_1 * l2_sf * cw_0 * accum[i*4+2];
                                    final_accum[serial_n_idx][i*4+3] += scale_a_1 * l2_sf * cw_1 * accum[i*4+3];
                                }
                            }
                        }
                    }

                    arrive_empty_barrier(stage_idx);
                    __syncwarp();
                }

                if (row_block_offset >= valid_m) {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    return;
                }

                if (is_linear1_phase) {
                constexpr uint32_t kNumPairs = kAccumPerThread / 8;
                    #pragma unroll
                    for (uint32_t serial_n_idx = 0; serial_n_idx < kNumSerialN; ++serial_n_idx) {
                        const uint32_t serial_l1_out_n_idx = serial_n_idx * WG_L1_OUT_BLOCK_N;
                        float swiglu_r0[kNumPairs][2];
                        float swiglu_r1[kNumPairs][2];
                        float amax_r0 = 0.0f, amax_r1 = 0.0f;

                        #pragma unroll
                        for (uint32_t p = 0; p < kNumPairs; ++ p) {
                            const uint32_t gate = 2 * p, up = 2 * p + 1;
                            auto clamp_gate = [](float& x) {
                                if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                                    x = cute::min(x, kActivationClamp);
                            };
                            auto clamp_up = [](float& x) {
                                if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                                    x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                            };
                            float g_r0_c0 = final_accum[serial_n_idx][gate*4 + 0]; clamp_gate(g_r0_c0);
                            float g_r0_c1 = final_accum[serial_n_idx][gate*4 + 1]; clamp_gate(g_r0_c1);
                            float g_r1_c0 = final_accum[serial_n_idx][gate*4 + 2]; clamp_gate(g_r1_c0);
                            float g_r1_c1 = final_accum[serial_n_idx][gate*4 + 3]; clamp_gate(g_r1_c1);
                            float u_r0_c0 = final_accum[serial_n_idx][up*4   + 0]; clamp_up(u_r0_c0);
                            float u_r0_c1 = final_accum[serial_n_idx][up*4   + 1]; clamp_up(u_r0_c1);
                            float u_r1_c0 = final_accum[serial_n_idx][up*4   + 2]; clamp_up(u_r1_c0);
                            float u_r1_c1 = final_accum[serial_n_idx][up*4   + 3]; clamp_up(u_r1_c1);
                            auto silu = [](float x) -> float {
                                const float e = kFastMath ? __expf(-x) : expf(-x);
                                const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                                return x * sig;
                            };
                            if (valid_r0) {
                                swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
                                swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
                                amax_r0 = cute::max(amax_r0, cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
                            } else {
                                swiglu_r0[p][0] = 0.0f;
                                swiglu_r0[p][1] = 0.0f;
                            }
                            if (valid_r1) {
                                swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
                                swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
                                amax_r1 = cute::max(amax_r1, cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
                            } else {
                                swiglu_r1[p][0] = 0.0f;
                                swiglu_r1[p][1] = 0.0f;
                            }
                        }

                        const float weight_r0 = [&]() {
                            if constexpr (kNumMaxTokensPerRank <= 1024) {
                                float weight = 0.0f;
                                if (col_idx == 0)
                                    weight = valid_r0 ? *l1_topk_weights_buffer
                                        .get_data_buffer(m_idx + row_offset_r0)
                                        .get_base_ptr<float>() : 0.0f;
                                return __shfl_sync(0xffffffff, weight, static_cast<int>(lane_idx - col_idx));
                            } else {
                                return valid_r0 ? *l1_topk_weights_buffer
                                    .get_data_buffer(m_idx + row_offset_r0)
                                    .get_base_ptr<float>() : 0.0f;
                            }
                        }();
                        const float weight_r1 = [&]() {
                            if constexpr (kNumMaxTokensPerRank <= 1024) {
                                float weight = 0.0f;
                                if (col_idx == 0)
                                    weight = valid_r1 ? *l1_topk_weights_buffer
                                        .get_data_buffer(m_idx + row_offset_r1)
                                        .get_base_ptr<float>() : 0.0f;
                                return __shfl_sync(0xffffffff, weight, static_cast<int>(lane_idx - col_idx));
                            } else {
                                return valid_r1 ? *l1_topk_weights_buffer
                                    .get_data_buffer(m_idx + row_offset_r1)
                                    .get_base_ptr<float>() : 0.0f;
                            }
                        }();
                        #pragma unroll
                        for (uint32_t p = 0; p < kNumPairs; ++ p) {
                            swiglu_r0[p][0] *= weight_r0;
                            swiglu_r0[p][1] *= weight_r0;
                            swiglu_r1[p][0] *= weight_r1;
                            swiglu_r1[p][1] *= weight_r1;
                        }
                        amax_r0 *= cute::abs(weight_r0);
                        amax_r1 *= cute::abs(weight_r1);
                        amax_r0 = math::warp_reduce<4, false>(amax_r0, math::ReduceMax<float>());
                        amax_r1 = math::warp_reduce<4, false>(amax_r1, math::ReduceMax<float>());

                        float sf_r0, sf_inv_r0, sf_r1, sf_inv_r1;
#if DG_W4A8_INT && DG_W4A8_INT_L2
                        // int8 intermediate: SF = amax/127, quantize to int8.
                        sf_r0 = amax_r0 * (1.0f / 127.0f);
                        sf_r1 = amax_r1 * (1.0f / 127.0f);
                        sf_inv_r0 = amax_r0 > 0.0f ? 127.0f / amax_r0 : 0.0f;
                        sf_inv_r1 = amax_r1 > 0.0f ? 127.0f / amax_r1 : 0.0f;
                        #pragma unroll
                        for (uint32_t p = 0; p < kNumPairs; ++ p) {
                            auto q8 = [](float v) -> int8_t {
                                // Intermediate rides an fp8-typed buffer; avoid the
                                // two fp8 NaN byte patterns 0x7F(127) and 0xFF(-1).
                                float r = __float2int_rn(v);
                                r = r < -126.0f ? -126.0f : (r > 126.0f ? 126.0f : r);
                                int8_t q = static_cast<int8_t>(r);
                                return q == -1 ? 0 : q;
                            };
                            const int8_t q00 = q8(swiglu_r0[p][0] * sf_inv_r0);
                            const int8_t q01 = q8(swiglu_r0[p][1] * sf_inv_r0);
                            const int8_t q10 = q8(swiglu_r1[p][0] * sf_inv_r1);
                            const int8_t q11 = q8(swiglu_r1[p][1] * sf_inv_r1);
                            const uint16_t r0_pair = (uint8_t)q00 | ((uint16_t)(uint8_t)q01 << 8);
                            const uint16_t r1_pair = (uint8_t)q10 | ((uint16_t)(uint8_t)q11 << 8);
                            const uint32_t col = p * 8 + col_idx * 2;
                            auto* p0 = reinterpret_cast<uint16_t*>(
                                smem_cd_l1 + r_0 * L1_OUT_BLOCK_N + serial_l1_out_n_idx + col);
                            auto* p1 = reinterpret_cast<uint16_t*>(
                                smem_cd_l1 + r_1 * L1_OUT_BLOCK_N + serial_l1_out_n_idx + col);
                            if (valid_r0) *p0 = r0_pair;
                            if (valid_r1) *p1 = r1_pair;
                        }
#else
                        {
                            float2 amax_pair = {amax_r0, amax_r1};
                            float2 sf_pair, sf_inv_pair;
                            math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                            sf_r0 = sf_pair.x; sf_inv_r0 = sf_inv_pair.x;
                            sf_r1 = sf_pair.y; sf_inv_r1 = sf_inv_pair.y;
                        }

                        #pragma unroll
                        for (uint32_t p = 0; p < kNumPairs; ++ p) {
                            const float v00 = swiglu_r0[p][0] * sf_inv_r0;
                            const float v01 = swiglu_r0[p][1] * sf_inv_r0;
                            const float v10 = swiglu_r1[p][0] * sf_inv_r1;
                            const float v11 = swiglu_r1[p][1] * sf_inv_r1;
                            const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                            const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));
                            const uint32_t col = p * 8 + col_idx * 2;
                            auto* p0 = reinterpret_cast<uint16_t*>(
                                smem_cd_l1 + r_0 * L1_OUT_BLOCK_N + serial_l1_out_n_idx + col);
                            auto* p1 = reinterpret_cast<uint16_t*>(
                                smem_cd_l1 + r_1 * L1_OUT_BLOCK_N + serial_l1_out_n_idx + col);
                            if (valid_r0)
                                *p0 = r0_pair.__x;
                            if (valid_r1)
                                *p1 = r1_pair.__x;
                        }
#endif

                        if (col_idx == 0) {
                            auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                            const uint32_t token_r0 = pool_block_idx * BLOCK_M + row_offset_r0;
                            const uint32_t token_r1 = pool_block_idx * BLOCK_M + row_offset_r1;
                            const uint32_t k_sf_idx = (n_block_idx * L1_OUT_BLOCK_N + serial_l1_out_n_idx) / 64u;
                            if (valid_r0)
                                sf_base_ptr[k_sf_idx * kSFRingStrideTokens + token_r0] = sf_r0;
                            if (valid_r1)
                                sf_base_ptr[k_sf_idx * kSFRingStrideTokens + token_r1] = sf_r1;
                        }
                    }

                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1,
                            out_n_idx,
                            m_idx + row_block_offset);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    ptx::tma_store_wait<0>();
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                } else {
                    constexpr uint32_t kNumRowsPerWarp = WG_BLOCK_M / 8;
                    #pragma unroll
                    for (uint32_t serial_n_idx = 0; serial_n_idx < kNumSerialN; ++serial_n_idx) {
                        const uint32_t serial_n_idx_base = n_block_idx * BLOCK_N + serial_n_idx * WG_BLOCK_N;

                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                            const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
                            auto write_pair = [&](uint32_t row, uint32_t col, uint32_t packed) {
                                auto smem_ptr = smem_cd_l2 + row * WG_BLOCK_N + col;
                                *reinterpret_cast<uint32_t*>(smem_ptr) = packed;
                            };
                            if (valid_r0) {
                                const uint32_t r0_lo = math::cast_into_bf16_and_pack(
                                    final_accum[serial_n_idx][chunk_lo*4 + 0], final_accum[serial_n_idx][chunk_lo*4 + 1]);
                                const uint32_t r0_hi = math::cast_into_bf16_and_pack(
                                    final_accum[serial_n_idx][chunk_hi*4 + 0], final_accum[serial_n_idx][chunk_hi*4 + 1]);
                                write_pair(r_0, chunk_lo * 8 + col_idx * 2, r0_lo);
                                write_pair(r_0, chunk_hi * 8 + col_idx * 2, r0_hi);
                            }
                            if (valid_r1) {
                                const uint32_t r1_lo = math::cast_into_bf16_and_pack(
                                    final_accum[serial_n_idx][chunk_lo*4 + 2], final_accum[serial_n_idx][chunk_lo*4 + 3]);
                                const uint32_t r1_hi = math::cast_into_bf16_and_pack(
                                    final_accum[serial_n_idx][chunk_hi*4 + 2], final_accum[serial_n_idx][chunk_hi*4 + 3]);
                                write_pair(r_1, chunk_lo * 8 + col_idx * 2, r1_lo);
                                write_pair(r_1, chunk_hi * 8 + col_idx * 2, r1_hi);
                            }
                        }
                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        const uint32_t row_in_warp_block = lane_idx / 16;
                        const uint32_t lane_in_row = lane_idx % 16;
                        constexpr uint32_t cols_per_lane = WG_BLOCK_N / 16;
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                            const uint32_t row_in_wg = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                            const uint32_t m_idx_in_block = row_block_offset + row_in_wg;
                            if (m_idx_in_block >= valid_m) break;

                            const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                            const uint32_t dst_rank_idx = src_metadata.rank_idx;
                            const uint32_t dst_token_idx = src_metadata.token_idx;
                            const uint32_t dst_topk_idx = src_metadata.topk_idx;
                            auto smem_ptr = smem_cd_l2 + row_in_wg * WG_BLOCK_N + lane_in_row * cols_per_lane;
                            const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            const auto packed = *reinterpret_cast<uint4*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                serial_n_idx_base * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    }
                }
                return;
            }

            // ---------------- GEMM ----------------
            using WGMMA = L1WGMMA;
            constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;  // 64 for M=64,N=128
            float final_accum[kAccumPerThread] = {};
            float accum[kAccumPerThread];

            const unsigned long long block_gemm_start = phase_profile_clock();
            const auto run_default_gemm_loop = [&]() {
[[maybe_unused]] bool b_decoded_ahead = false;
#if DG_W4A8_INT && DG_W4A8_INT_QOQ
                [[maybe_unused]] int32_t qoq_iacc[kAccumPerThread];
                [[maybe_unused]] uint32_t qoq_last_stage = 0;
                [[maybe_unused]] float qoq_scale_a_0 = 0.0f, qoq_scale_a_1 = 0.0f;
#endif
for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                wait_and_dequant_b_stage(stage_idx, phase, b_decoded_ahead);
                b_decoded_ahead = false;

                // Read SF (must precede warpgroup_arrive)
                float scale_a_0_lo, scale_a_1_lo;
                float scale_a_0_hi, scale_a_1_hi;  // Only used in L2 (per-64 K)
                if (is_linear1_phase) {
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                } else {
                    // L2: SFA layout is (K=2, M=BLOCK_M) MN-major; first half SF at offset 0, second at BLOCK_M
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
                    scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r0);
                    scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r1);
                }

                // Per-16 UE4M3 weight scales are already included in decoded
                // FP8 B. Apply the optional expert scale once after the full
                // FP32 K reduction, matching exact MXFP4 reference semantics.
                constexpr float gate_sf = 1.0f;
                constexpr float up_sf = 1.0f;
                constexpr float l2_sf_lo = 1.0f;
                constexpr float l2_sf_hi = 1.0f;

                if (is_linear1_phase) {
                    if constexpr (kSwapABActive) {
                        auto run_swap_ab_l1 = [&]<uint32_t N_SWAP>() {
                            using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                            float swap_accum[kSwapAccum];

                            // Per-32-K batches with dual-accumulator ping-pong:
                            // batch b+1's WGMMA is issued before waiting on batch
                            // b (warpgroup_wait<1>), hiding the promotion FMAs and
                            // coefficient loads under the next batch's WGMMA.
                            float swap_accum2[kSwapAccum];
                            // Preload the two per-row E8M0 coefficient words once
                            // per stage; extract one byte per K32 batch.
                            const uint8_t* rf_row0 = smem_packed_b[stage_idx] + (wg_n_idx + r_0) * 64;
                            const uint8_t* rf_row1 = smem_packed_b[stage_idx] + (wg_n_idx + r_1) * 64;
                            // Coefficients arrive via their own TMA into coeff SMEM.
                            const uint32_t cw_word_r0 =
                                *reinterpret_cast<const uint32_t*>(smem_b_coeff[stage_idx] + (wg_n_idx + r_0) * 4);
                            const uint32_t cw_word_r1 =
                                *reinterpret_cast<const uint32_t*>(smem_b_coeff[stage_idx] + (wg_n_idx + r_1) * 4);
                            uint4 rf_words0, rf_words1;
                            if constexpr (kRFDecode) {
                                rf_words0 = *reinterpret_cast<const uint4*>(rf_row0 + col_idx * 16);
                                rf_words1 = *reinterpret_cast<const uint4*>(rf_row1 + col_idx * 16);
                            }

#if DG_W4A8_INT
                            // W4A8-integer L1 path: int4 weights sign-extend into
                            // A fragments, IGMMA accumulates raw products in int32
                            // (single accumulator -- group_size >= BLOCK_K gives one
                            // weight scale per K128, no intra-tile promotion), then
                            // epilogue I2F * per-group w_scale * per-token x_scale.
                            // Weight scale rides the coeff slot as one fp32 (4 bytes,
                            // same layout as the four E8M0 bytes it replaces).
                            {
                                using IntMMA = typename mma::sm90::INT8MMARSSelector<N_SWAP>::type;
                                // FP4-style chained accumulation: ONE int32
                                // accumulator carries all four K32 batches -- with a
                                // single weight scale per K128 (group_size >=
                                // BLOCK_K) nothing needs promoting mid-block.
                                // Same-warpgroup WGMMAs execute in order, so the
                                // chain runs at full throughput while batch B+1's
                                // decode still hides under batch B's WGMMA; one
                                // wait + one promote per K128 and a quarter of the
                                // accumulator registers of the old 4-array form.
                                // Small N_SWAP: two interleaved 2-chains (a:{0,2},
                                // b:{1,3}) restore WGMMA pipelining that a single
                                // 4-chain serializes when each m64nNk32 is tiny;
                                // large N keeps the register-lean single chain.
                                // Dual-chain at small N measured WORSE (2026-07-08:
                                // M4 384->503); default stays single-chain. Flip the
                                // header define for a controlled re-test.
#ifndef DG_W4A8_INT_SMALLN_DUAL
#define DG_W4A8_INT_SMALLN_DUAL 0
#endif
                                constexpr bool kIntChainAccum = (N_SWAP >= 32) or !DG_W4A8_INT_SMALLN_DUAL;
                                int32_t iacc[kSwapAccum];
                                [[maybe_unused]] int32_t iacc2[kSwapAccum];
#if DG_W4A8_INT_QOQ
                                // Inline QoQ: coeff word = [s2:u8 | nz:u8 | s1:bf16-hi16]
                                // (nz = (-z*s2) mod 256 under ZP, pad otherwise).
                                // s2 bakes into the decode LUT (fold-at-decode in the
                                // integer domain); s1 rides the promote as before.
                                const uint32_t s2_r0 = cw_word_r0 & 0xffu;
                                const uint32_t s2_r1 = cw_word_r1 & 0xffu;
#if DG_W4A8_INT_QOQ_ZP
                                // Coeff byte 1 = prepack-precomputed nz = (-z*s2) mod 256.
                                const uint32_t nz_r0 = (cw_word_r0 >> 8) & 0xffu;
                                const uint32_t nz_r1 = (cw_word_r1 >> 8) & 0xffu;
                                const uint32_t lutlo_r0 = __vadd4(s2_r0 * 0x03020100u, nz_r0 * 0x01010101u);
                                const uint32_t luthi_r0 = __vadd4(s2_r0 * 0x07060504u, nz_r0 * 0x01010101u);
                                const uint32_t lutlo_r1 = __vadd4(s2_r1 * 0x03020100u, nz_r1 * 0x01010101u);
                                const uint32_t luthi_r1 = __vadd4(s2_r1 * 0x07060504u, nz_r1 * 0x01010101u);
#else
                                const uint32_t lutlo_r0 = s2_r0 * 0x03020100u, luthi_r0 = s2_r0 * 0x07060504u;
                                const uint32_t lutlo_r1 = s2_r1 * 0x03020100u, luthi_r1 = s2_r1 * 0x07060504u;
#endif
                                const float w_r0 = __uint_as_float(cw_word_r0 & 0xFFFF0000u);
                                const float w_r1 = __uint_as_float(cw_word_r1 & 0xFFFF0000u);
#else
                                const float w_r0 = __uint_as_float(cw_word_r0);
                                const float w_r1 = __uint_as_float(cw_word_r1);
#endif
                                uint32_t ifrag[4][4];
                                auto issue_int = [&](int32_t (&acc)[kSwapAccum], const uint32_t B) {
                                    const uint32_t w0 = B == 0 ? rf_words0.x : B == 1 ? rf_words0.y :
                                                        B == 2 ? rf_words0.z : rf_words0.w;
                                    const uint32_t w1 = B == 0 ? rf_words1.x : B == 1 ? rf_words1.y :
                                                        B == 2 ? rf_words1.z : rf_words1.w;
#if DG_W4A8_INT_QOQ
#if DG_W4A8_INT_QOQ_ZP
                                    const uint2 d0 = int4q::decode_uint4_prmt_groups_to_int8_pair_lut_zp(
                                        w0, lutlo_r0, luthi_r0, s2_r0);
                                    const uint2 d1 = int4q::decode_uint4_prmt_groups_to_int8_pair_lut_zp(
                                        w1, lutlo_r1, luthi_r1, s2_r1);
#else
                                    const uint2 d0 = int4q::decode_int4_prmt_groups_to_int8_pair_lut(
                                        w0, lutlo_r0, luthi_r0, s2_r0);
                                    const uint2 d1 = int4q::decode_int4_prmt_groups_to_int8_pair_lut(
                                        w1, lutlo_r1, luthi_r1, s2_r1);
#endif
#else
                                    const uint2 d0 = int4q::decode_int4_prmt_groups_to_int8_pair(w0);
                                    const uint2 d1 = int4q::decode_int4_prmt_groups_to_int8_pair(w1);
#endif
                                    ifrag[B][0] = d0.x; ifrag[B][1] = d1.x; ifrag[B][2] = d0.y; ifrag[B][3] = d1.y;
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                        ptx::warpgroup_fence_operand(reinterpret_cast<float&>(acc[i]));
                                    #pragma unroll
                                    for (uint32_t i = 0; i < 4; ++ i)
                                        ptx::warpgroup_fence_operand(reinterpret_cast<float&>(ifrag[B][i]));
                                    ptx::warpgroup_arrive();
                                    auto desc_b = mma::sm90::make_smem_desc(smem_a[stage_idx] + B * IntMMA::K, 1);
                                    IntMMA::wgmma(ifrag[B], desc_b, acc, kIntChainAccum ? (B > 0) : (B >= 2));
                                    ptx::warpgroup_commit_batch();
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                        ptx::warpgroup_fence_operand(reinterpret_cast<float&>(acc[i]));
                                };
                                float itok[kSwapAccum / 4][2];
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    itok[i][0] = token_0 < valid_m ? ptx::ld_shared(smem_sfa[stage_idx] + token_0) : 0.0f;
                                    itok[i][1] = token_0 + 1 < valid_m ? ptx::ld_shared(smem_sfa[stage_idx] + token_0 + 1) : 0.0f;
                                }
                                // All four chained into one commit-group sequence,
                                // drained once.
#if DG_W4A8_INT_QOQ && DG_W4A8_INT_QOQ_FULLK
                                // Full-K chain: accumulate into the loop-scope
                                // qoq_iacc across stages; no mid-loop promote.
                                {
                                    auto issue_fk = [&](const uint32_t B) {
                                        const uint32_t w0 = B == 0 ? rf_words0.x : B == 1 ? rf_words0.y :
                                                            B == 2 ? rf_words0.z : rf_words0.w;
                                        const uint32_t w1 = B == 0 ? rf_words1.x : B == 1 ? rf_words1.y :
                                                            B == 2 ? rf_words1.z : rf_words1.w;
#if DG_W4A8_INT_QOQ_ZP
                                        const uint2 d0 = int4q::decode_uint4_prmt_groups_to_int8_pair_lut_zp(
                                            w0, lutlo_r0, luthi_r0, s2_r0);
                                        const uint2 d1 = int4q::decode_uint4_prmt_groups_to_int8_pair_lut_zp(
                                            w1, lutlo_r1, luthi_r1, s2_r1);
#else
                                        const uint2 d0 = int4q::decode_int4_prmt_groups_to_int8_pair_lut(
                                            w0, lutlo_r0, luthi_r0, s2_r0);
                                        const uint2 d1 = int4q::decode_int4_prmt_groups_to_int8_pair_lut(
                                            w1, lutlo_r1, luthi_r1, s2_r1);
#endif
                                        ifrag[B][0] = d0.x; ifrag[B][1] = d1.x; ifrag[B][2] = d0.y; ifrag[B][3] = d1.y;
                                        #pragma unroll
                                        for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                            ptx::warpgroup_fence_operand(reinterpret_cast<float&>(qoq_iacc[i]));
                                        #pragma unroll
                                        for (uint32_t i = 0; i < 4; ++ i)
                                            ptx::warpgroup_fence_operand(reinterpret_cast<float&>(ifrag[B][i]));
                                        ptx::warpgroup_arrive();
                                        auto desc_b = mma::sm90::make_smem_desc(smem_a[stage_idx] + B * IntMMA::K, 1);
                                        IntMMA::wgmma(ifrag[B], desc_b, qoq_iacc,
                                                      !(k_block_idx == 0 and B == 0));
                                        ptx::warpgroup_commit_batch();
                                        #pragma unroll
                                        for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                            ptx::warpgroup_fence_operand(reinterpret_cast<float&>(qoq_iacc[i]));
                                    };
                                    issue_fk(0u); issue_fk(1u); issue_fk(2u); issue_fk(3u);
                                    ptx::warpgroup_wait<0>();
                                    if (k_block_idx + 1 < num_k_blocks) {
                                        arrive_empty_barrier(stage_idx);
                                    } else {
                                        // Final stage: promote once with itok/s1
                                        // from THIS stage's SMEM, then release.
                                        #pragma unroll
                                        for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                            const uint32_t token_0 = i * 8 + col_idx * 2;
                                            const float t0 = token_0 < valid_m ? ptx::ld_shared(smem_sfa[stage_idx] + token_0) : 0.0f;
                                            const float t1 = token_0 + 1 < valid_m ? ptx::ld_shared(smem_sfa[stage_idx] + token_0 + 1) : 0.0f;
                                            final_accum[i*4+0] = t0 * w_r0 * static_cast<float>(qoq_iacc[i*4+0]);
                                            final_accum[i*4+2] = t0 * w_r1 * static_cast<float>(qoq_iacc[i*4+2]);
                                            final_accum[i*4+1] = t1 * w_r0 * static_cast<float>(qoq_iacc[i*4+1]);
                                            final_accum[i*4+3] = t1 * w_r1 * static_cast<float>(qoq_iacc[i*4+3]);
                                        }
                                        arrive_empty_barrier(stage_idx);
                                    }
                                    return;
                                }
#endif
                                if constexpr (kIntChainAccum) {
                                    issue_int(iacc, 0u);
                                    issue_int(iacc, 1u);
                                    issue_int(iacc, 2u);
                                    issue_int(iacc, 3u);
                                } else {
                                    issue_int(iacc, 0u);
                                    issue_int(iacc2, 1u);
                                    issue_int(iacc, 2u);
                                    issue_int(iacc2, 3u);
                                }
                                ptx::warpgroup_wait<0>();
                                arrive_empty_barrier(stage_idx);
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const auto acc_at = [&](const uint32_t j) {
                                        return kIntChainAccum ? iacc[j] : iacc[j] + iacc2[j];
                                    };
                                    const float s0 = static_cast<float>(acc_at(i*4+0));
                                    const float s2 = static_cast<float>(acc_at(i*4+2));
                                    const float s1 = static_cast<float>(acc_at(i*4+1));
                                    const float s3 = static_cast<float>(acc_at(i*4+3));
                                    final_accum[i * 4 + 0] += itok[i][0] * gate_sf * w_r0 * s0;
                                    final_accum[i * 4 + 2] += itok[i][0] * up_sf   * w_r1 * s2;
                                    final_accum[i * 4 + 1] += itok[i][1] * gate_sf * w_r0 * s1;
                                    final_accum[i * 4 + 3] += itok[i][1] * up_sf   * w_r1 * s3;
                                }
                                return;
                            }
#endif

                            using SwapRS = typename mma::sm90::FP8MMARSSelector<N_SWAP>::type;
                            auto issue_batch = [&](float (&acc)[kSwapAccum], uint32_t (&frag)[4], const uint32_t B) {
                                if constexpr (kRFDecode) {
                                    const uint32_t w0 = B == 0 ? rf_words0.x : B == 1 ? rf_words0.y :
                                                        B == 2 ? rf_words0.z : rf_words0.w;
                                    const uint32_t w1 = B == 0 ? rf_words1.x : B == 1 ? rf_words1.y :
                                                        B == 2 ? rf_words1.z : rf_words1.w;
                                    const uint2 d0 = mxfp4::decode_mxfp4_prmt_groups_to_fp8_pair(w0);
                                    const uint2 d1 = mxfp4::decode_mxfp4_prmt_groups_to_fp8_pair(w1);
                                    frag[0] = d0.x; frag[1] = d1.x; frag[2] = d0.y; frag[3] = d1.y;
                                }
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(acc[i]);
                                #pragma unroll
                                for (uint32_t i = 0; i < 4; ++ i)
                                    ptx::warpgroup_fence_operand(reinterpret_cast<float&>(frag[i]));
                                ptx::warpgroup_arrive();
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + B * SwapWGMMA::K, 1);
                                if constexpr (kRFDecode) {
                                    SwapRS::wgmma(frag, desc_b, acc, false);
                                } else {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + wg_n_idx * BLOCK_K + B * SwapWGMMA::K, 1);
                                    SwapWGMMA::wgmma(desc_a, desc_b, acc, false);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(acc[i]);
                            };
                            uint32_t rf_frags[4][4];
                            float tok_scale[kSwapAccum / 4][2];
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                const uint32_t token_0 = i * 8 + col_idx * 2;
                                tok_scale[i][0] = token_0 < valid_m ?
                                    ptx::ld_shared(smem_sfa[stage_idx] + token_0) : 0.0f;
                                tok_scale[i][1] = token_0 + 1 < valid_m ?
                                    ptx::ld_shared(smem_sfa[stage_idx] + token_0 + 1) : 0.0f;
                            }
                            auto promote_batch = [&](float (&acc)[kSwapAccum], const uint32_t B) {
                                const float cw_r0 = mxfp4::e8m0_to_float((cw_word_r0 >> (B * 8)) & 0xffu);
                                const float cw_r1 = mxfp4::e8m0_to_float((cw_word_r1 >> (B * 8)) & 0xffu);
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    final_accum[i * 4 + 0] += tok_scale[i][0] * gate_sf * cw_r0 * acc[i * 4 + 0];
                                    final_accum[i * 4 + 2] += tok_scale[i][0] * up_sf * cw_r1 * acc[i * 4 + 2];
                                    final_accum[i * 4 + 1] += tok_scale[i][1] * gate_sf * cw_r0 * acc[i * 4 + 1];
                                    final_accum[i * 4 + 3] += tok_scale[i][1] * up_sf * cw_r1 * acc[i * 4 + 3];
                                }
                            };

#if DG_MXFP4_REL_LUT
                            if constexpr (kRFDecode) {
                                DG_STATIC_ASSERT(BLOCK_K / SwapWGMMA::K == 4, "Expects 4 K32 batches");
                                // Relative-LUT fold: decode each K32 group against
                                // the row for d = e_max - e_group, so all four
                                // WGMMAs accumulate into ONE accumulator and the
                                // promotion happens once per K128 with the row's
                                // max coefficient. Kills 3 of 4 promote passes and
                                // 3 of 4 accumulator buffers.
                                uint32_t t0 = __vmaxu4(cw_word_r0, cw_word_r0 >> 16);
                                t0 = __vmaxu4(t0, t0 >> 8);
                                const uint32_t em_r0 = t0 & 0xffu;
                                uint32_t t1 = __vmaxu4(cw_word_r1, cw_word_r1 >> 16);
                                t1 = __vmaxu4(t1, t1 >> 8);
                                const uint32_t em_r1 = t1 & 0xffu;
                                const uint32_t dw_r0 = __vminu4(
                                    __vsub4(em_r0 * 0x01010101u, cw_word_r0), 0x0D0D0D0Du);
                                const uint32_t dw_r1 = __vminu4(
                                    __vsub4(em_r1 * 0x01010101u, cw_word_r1), 0x0D0D0D0Du);
                                // Same interleaved issue shape as the default
                                // path (decode B+1 hides under batch B's WGMMA),
                                // but all four batches accumulate into ONE
                                // accumulator: same-warpgroup WGMMAs execute in
                                // order, so cross-commit-group accumulation is
                                // safe (stock FP8 mainloop shape).
                                auto issue_rel = [&](uint32_t (&frag)[4], const uint32_t B) {
                                    const uint32_t w0 = B == 0 ? rf_words0.x : B == 1 ? rf_words0.y :
                                                        B == 2 ? rf_words0.z : rf_words0.w;
                                    const uint32_t w1 = B == 0 ? rf_words1.x : B == 1 ? rf_words1.y :
                                                        B == 2 ? rf_words1.z : rf_words1.w;
                                    const uint32_t d0i = (dw_r0 >> (B * 8)) & 0xffu;
                                    const uint32_t d1i = (dw_r1 >> (B * 8)) & 0xffu;
                                    const uint2 d0 = mxfp4::decode_mxfp4_prmt_groups_to_fp8_pair_lut(
                                        w0, mxfp4::kE2M1RelLut[d0i][0], mxfp4::kE2M1RelLut[d0i][1]);
                                    const uint2 d1 = mxfp4::decode_mxfp4_prmt_groups_to_fp8_pair_lut(
                                        w1, mxfp4::kE2M1RelLut[d1i][0], mxfp4::kE2M1RelLut[d1i][1]);
                                    frag[0] = d0.x; frag[1] = d1.x;
                                    frag[2] = d0.y; frag[3] = d1.y;
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                        ptx::warpgroup_fence_operand(swap_accum[i]);
                                    #pragma unroll
                                    for (uint32_t i = 0; i < 4; ++ i)
                                        ptx::warpgroup_fence_operand(
                                            reinterpret_cast<float&>(frag[i]));
                                    ptx::warpgroup_arrive();
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + B * SwapWGMMA::K, 1);
                                    SwapRS::wgmma(frag, desc_b, swap_accum, B > 0);
                                    ptx::warpgroup_commit_batch();
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                        ptx::warpgroup_fence_operand(swap_accum[i]);
                                };
                                issue_rel(rf_frags[0], 0u);
                                issue_rel(rf_frags[1], 1u);
                                issue_rel(rf_frags[2], 2u);
                                issue_rel(rf_frags[3], 3u);
                                ptx::warpgroup_wait<0>();
                                // Token scales were preloaded above; all SMEM reads
                                // (scratch words, coeffs, WGMMA activations) done.
                                arrive_empty_barrier(stage_idx);
                                const float cw_r0 = mxfp4::e8m0_to_float(em_r0);
                                const float cw_r1 = mxfp4::e8m0_to_float(em_r1);
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    final_accum[i * 4 + 0] += tok_scale[i][0] * gate_sf * cw_r0 * swap_accum[i * 4 + 0];
                                    final_accum[i * 4 + 2] += tok_scale[i][0] * up_sf * cw_r1 * swap_accum[i * 4 + 2];
                                    final_accum[i * 4 + 1] += tok_scale[i][1] * gate_sf * cw_r0 * swap_accum[i * 4 + 1];
                                    final_accum[i * 4 + 3] += tok_scale[i][1] * up_sf * cw_r1 * swap_accum[i * 4 + 3];
                                }
                                return;
                            }
#endif

                            DG_STATIC_ASSERT(BLOCK_K / SwapWGMMA::K == 4, "Expects 4 K32 batches");
                            // Mirror the FP8 pipeline: all four WGMMAs issued
                            // back-to-back (one commit group each, own accum
                            // registers), then progressively drained so every
                            // promotion overlaps the remaining in-flight WGMMAs.
                            float swap_accum3[kSwapAccum], swap_accum4[kSwapAccum];
                            issue_batch(swap_accum, rf_frags[0], 0u);
                            issue_batch(swap_accum2, rf_frags[1], 1u);
                            issue_batch(swap_accum3, rf_frags[2], 2u);
                            issue_batch(swap_accum4, rf_frags[3], 3u);
                            ptx::warpgroup_wait<3>();
                            promote_batch(swap_accum, 0u);
                            ptx::warpgroup_wait<2>();
                            promote_batch(swap_accum2, 1u);
                            ptx::warpgroup_wait<1>();
                            promote_batch(swap_accum3, 2u);
                            ptx::warpgroup_wait<0>();
                            // All SMEM reads (scratch words, coeffs, token scales,
                            // WGMMA activations) completed: release the producer
                            // before the final register-only promotion.
                            arrive_empty_barrier(stage_idx);
                            promote_batch(swap_accum4, 3u);
                        };

                        const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                        if constexpr (kIntermediateHidden <= 2048) {
                            if (n_swap <= 8) {
                                run_swap_ab_l1.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l1.template operator()<16>();
                            } else if (n_swap <= 32) {
                                run_swap_ab_l1.template operator()<32>();
                            } else {
                                run_swap_ab_l1.template operator()<64>();
                            }
                        } else {
#if DG_W4A8_INT
                            // int8 RS GMMA atoms exist only at N in {8,16,32,64};
                            // round N_SWAP up (24->32, 40/48/56->64) so pro shapes
                            // stay on a supported atom.
                            if (n_swap <= 8) {
                                run_swap_ab_l1.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l1.template operator()<16>();
                            } else if (n_swap <= 32) {
                                run_swap_ab_l1.template operator()<32>();
                            } else {
                                run_swap_ab_l1.template operator()<64>();
                            }
#else
                            switch (n_swap) {
                                case 8:  run_swap_ab_l1.template operator()<8>();  break;
                                case 16: run_swap_ab_l1.template operator()<16>(); break;
                                case 24: run_swap_ab_l1.template operator()<24>(); break;
                                case 32: run_swap_ab_l1.template operator()<32>(); break;
                                case 40: run_swap_ab_l1.template operator()<40>(); break;
                                case 48: run_swap_ab_l1.template operator()<48>(); break;
                                case 56: run_swap_ab_l1.template operator()<56>(); break;
                                default: run_swap_ab_l1.template operator()<64>(); break;
                            }
#endif
                        }
                    } else {
#if DG_W4A8_INT && DG_W4A8_INT_QOQ
                        // QoQ L1: both scales are K-constant (s2 folded into
                        // the int8 weights, s1 per-row, act scale per-token),
                        // so ONE int32 accumulator chains across the whole K
                        // loop -- zero in-loop promotes; conversion happens
                        // once after the loop.
                        {
                            using IntWGMMA = typename mma::sm90::INT8MMASelector<WG_BLOCK_N>::type;
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(reinterpret_cast<float&>(qoq_iacc[i]));
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k32 = 0; k32 < BLOCK_K / WGMMA::K; ++ k32) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K + k32 * IntWGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + wg_n_idx * BLOCK_K + k32 * IntWGMMA::K, 1);
                                IntWGMMA::wgmma(desc_a, desc_b, qoq_iacc,
                                                !(k_block_idx == 0 and k32 == 0));
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(reinterpret_cast<float&>(qoq_iacc[i]));
                            ptx::warpgroup_wait<0>();
                            if (k_block_idx + 1 < num_k_blocks) {
                                arrive_empty_barrier(stage_idx);
                            } else {
                                // Keep the stage alive: the after-loop promote
                                // still reads s1 from its coeff SMEM.
                                qoq_last_stage = stage_idx;
                                qoq_scale_a_0 = scale_a_0_lo;
                                qoq_scale_a_1 = scale_a_1_lo;
                            }
                        }
#elif DG_W4A8_INT
                        // W4A8-int non-swapAB L1: SMEM holds int8-decoded
                        // weights; one int32 accumulator chained across the
                        // four K32 batches (single weight scale per K128,
                        // group_size >= BLOCK_K), then a single I2F promote
                        // with the fp32 scale riding the coeff slot.
                        {
                            using IntWGMMA = typename mma::sm90::INT8MMASelector<WG_BLOCK_N>::type;
                            int32_t iacc[kAccumPerThread];
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(reinterpret_cast<float&>(iacc[i]));
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k32 = 0; k32 < BLOCK_K / WGMMA::K; ++ k32) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K + k32 * IntWGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + wg_n_idx * BLOCK_K + k32 * IntWGMMA::K, 1);
                                IntWGMMA::wgmma(desc_a, desc_b, iacc, k32 > 0);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(reinterpret_cast<float&>(iacc[i]));
#if DG_W4A8_INT_SHADOW
                            // Shadow decode: this block's WGMMAs are committed
                            // and in flight; use the idle issue slots to decode
                            // the NEXT stage's B tile (math-dequant path only).
                            if constexpr (kMathDequantSplitN and !kPreDecodedB and kPackedBScratch) {
                                if (k_block_idx + 1 < num_k_blocks) {
                                    const uint32_t next_stage = stage_idx == kNumStages - 1 ? 0u : stage_idx + 1u;
                                    const uint32_t next_phase = next_stage == 0u ? phase ^ 1u : phase;
                                    full_barriers[next_stage]->wait(next_phase);
                                    if constexpr (BLOCK_N == 128) {
                                        mxfp4::dequant_smem_b_from_packed_half_row_unscaled<kUsePRMTGroups, kIntDecodeB>(
                                            reinterpret_cast<uint8_t*>(smem_b[next_stage]),
                                            smem_packed_b[next_stage], epilogue_thread_idx,
                                            smem_b_coeff[next_stage]);
                                    } else {
                                        mxfp4::dequant_smem_b_from_packed_unscaled<kUsePRMTGroups, kIntDecodeB, kQoQFoldB>(
                                            reinterpret_cast<uint8_t*>(smem_b[next_stage]),
                                            smem_packed_b[next_stage], epilogue_thread_idx,
                                            smem_b_coeff[next_stage]);
                                    }
                                    asm volatile("bar.sync 8, 256;" ::: "memory");
                                    cutlass::arch::fence_view_async_shared();
                                    b_decoded_ahead = true;
                                }
                            }
#endif
                            ptx::warpgroup_wait<0>();

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                const float sb = (i & 1u) ? up_sf : gate_sf;
                                const uint32_t n_0 = wg_n_idx + i * 8 + col_idx * 2;
                                const float w_0 = __uint_as_float(*reinterpret_cast<const uint32_t*>(
                                    &smem_b_coeff[stage_idx][n_0 * 4]));
                                const float w_1 = __uint_as_float(*reinterpret_cast<const uint32_t*>(
                                    &smem_b_coeff[stage_idx][(n_0 + 1) * 4]));
                                final_accum[i*4+0] += scale_a_0_lo * sb * w_0 * static_cast<float>(iacc[i*4+0]);
                                final_accum[i*4+1] += scale_a_0_lo * sb * w_1 * static_cast<float>(iacc[i*4+1]);
                                final_accum[i*4+2] += scale_a_1_lo * sb * w_0 * static_cast<float>(iacc[i*4+2]);
                                final_accum[i*4+3] += scale_a_1_lo * sb * w_1 * static_cast<float>(iacc[i*4+3]);
                            }

                            // Released only after promotion (coeff SMEM must live).
                            arrive_empty_barrier(stage_idx);
                        }
#else
                        // Per-32-K WGMMA batches; the E8M0 weight coefficient is a
                        // per-column multiplier (columns are weight rows here).
                        #pragma unroll
                        for (uint32_t k32 = 0; k32 < BLOCK_K / WGMMA::K; ++ k32) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + row_block_offset * BLOCK_K + k32 * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + wg_n_idx * BLOCK_K + k32 * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, false);
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            // L1: gate/up alternate at gran=8 along N; each `i` block of 8
                            // cols belongs entirely to one of {gate, up}, so .x and .y
                            // share the same scalar.
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                const float sb = (i & 1u) ? up_sf : gate_sf;
                                const uint32_t n_0 = wg_n_idx + i * 8 + col_idx * 2;
                                const float cw_0 = mxfp4::e8m0_to_float(smem_b_coeff[stage_idx][n_0 * 4 + k32]);
                                const float cw_1 = mxfp4::e8m0_to_float(smem_b_coeff[stage_idx][(n_0 + 1) * 4 + k32]);
                                final_accum[i*4+0] += scale_a_0_lo * sb * cw_0 * accum[i*4+0];
                                final_accum[i*4+1] += scale_a_0_lo * sb * cw_1 * accum[i*4+1];
                                final_accum[i*4+2] += scale_a_1_lo * sb * cw_0 * accum[i*4+2];
                                final_accum[i*4+3] += scale_a_1_lo * sb * cw_1 * accum[i*4+3];
                            }
                        }

                        // NOTE: the empty barrier is released only after the last
                        // promotion: coefficients live in SMEM and must not be
                        // overwritten by the next stage's decode until read.
                        arrive_empty_barrier(stage_idx);
#endif
                    }
                } else {
                    if constexpr (kSwapABActive) {
                        auto run_swap_ab_l2 = [&]<uint32_t N_SWAP>() {
                            using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
                            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
                            float swap_accum[kSwapAccum];

                            float swap_accum2[kSwapAccum];
                            const uint8_t* rf_row0 = smem_packed_b[stage_idx] + (wg_n_idx + r_0) * 64;
                            const uint8_t* rf_row1 = smem_packed_b[stage_idx] + (wg_n_idx + r_1) * 64;
                            // Coefficients arrive via their own TMA into coeff SMEM.
                            const uint32_t cw_word_r0 =
                                *reinterpret_cast<const uint32_t*>(smem_b_coeff[stage_idx] + (wg_n_idx + r_0) * 4);
                            const uint32_t cw_word_r1 =
                                *reinterpret_cast<const uint32_t*>(smem_b_coeff[stage_idx] + (wg_n_idx + r_1) * 4);
                            uint4 rf_words0, rf_words1;
                            if constexpr (kRFDecode) {
                                rf_words0 = *reinterpret_cast<const uint4*>(rf_row0 + col_idx * 16);
                                rf_words1 = *reinterpret_cast<const uint4*>(rf_row1 + col_idx * 16);
                            }

                            using SwapRS = typename mma::sm90::FP8MMARSSelector<N_SWAP>::type;
                            auto issue_batch = [&](float (&acc)[kSwapAccum], uint32_t (&frag)[4], const uint32_t B) {
                                if constexpr (kRFDecode) {
                                    const uint32_t w0 = B == 0 ? rf_words0.x : B == 1 ? rf_words0.y :
                                                        B == 2 ? rf_words0.z : rf_words0.w;
                                    const uint32_t w1 = B == 0 ? rf_words1.x : B == 1 ? rf_words1.y :
                                                        B == 2 ? rf_words1.z : rf_words1.w;
                                    const uint2 d0 = mxfp4::decode_mxfp4_prmt_groups_to_fp8_pair(w0);
                                    const uint2 d1 = mxfp4::decode_mxfp4_prmt_groups_to_fp8_pair(w1);
                                    frag[0] = d0.x; frag[1] = d1.x; frag[2] = d0.y; frag[3] = d1.y;
                                }
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(acc[i]);
                                #pragma unroll
                                for (uint32_t i = 0; i < 4; ++ i)
                                    ptx::warpgroup_fence_operand(reinterpret_cast<float&>(frag[i]));
                                ptx::warpgroup_arrive();
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + B * SwapWGMMA::K, 1);
                                if constexpr (kRFDecode) {
                                    SwapRS::wgmma(frag, desc_b, acc, false);
                                } else {
                                    auto desc_a = mma::sm90::make_smem_desc(
                                        smem_b[stage_idx] + wg_n_idx * BLOCK_K + B * SwapWGMMA::K, 1);
                                    SwapWGMMA::wgmma(desc_a, desc_b, acc, false);
                                }
                                ptx::warpgroup_commit_batch();
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                    ptx::warpgroup_fence_operand(acc[i]);
                            };
                            uint32_t rf_frags[4][4];
                            // L2 activation SF half (lo for K32 batches 0/1, hi for 2/3).
                            float tok_scale[2][kSwapAccum / 4][2];
                            #pragma unroll
                            for (uint32_t sf = 0; sf < 2; ++ sf) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    const uint32_t token_0 = i * 8 + col_idx * 2;
                                    tok_scale[sf][i][0] = token_0 < valid_m ?
                                        ptx::ld_shared(smem_sfa[stage_idx] + sf * kL2SFAHalfStride + token_0) : 0.0f;
                                    tok_scale[sf][i][1] = token_0 + 1 < valid_m ?
                                        ptx::ld_shared(smem_sfa[stage_idx] + sf * kL2SFAHalfStride + token_0 + 1) : 0.0f;
                                }
                            }
                            auto promote_batch = [&](float (&acc)[kSwapAccum], const uint32_t B) {
                                const uint32_t kSFGroup = B / 2;
                                const float cw_r0 = mxfp4::e8m0_to_float((cw_word_r0 >> (B * 8)) & 0xffu);
                                const float cw_r1 = mxfp4::e8m0_to_float((cw_word_r1 >> (B * 8)) & 0xffu);
                                #pragma unroll
                                for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                    final_accum[i * 4 + 0] += tok_scale[kSFGroup][i][0] * l2_sf_lo * cw_r0 * acc[i * 4 + 0];
                                    final_accum[i * 4 + 2] += tok_scale[kSFGroup][i][0] * l2_sf_lo * cw_r1 * acc[i * 4 + 2];
                                    final_accum[i * 4 + 1] += tok_scale[kSFGroup][i][1] * l2_sf_lo * cw_r0 * acc[i * 4 + 1];
                                    final_accum[i * 4 + 3] += tok_scale[kSFGroup][i][1] * l2_sf_lo * cw_r1 * acc[i * 4 + 3];
                                }
                            };

#if DG_W4A8_INT && DG_W4A8_INT_L2
                            // W4A8-integer L2: int4 weights + int8 intermediate.
                            // Four int32 accumulators; the activation SF changes
                            // at K64 so batches {0,1} and {2,3} carry different
                            // per-token scales -> sum each half separately, one
                            // weight scale (group>=BLOCK_K) rides both.
                            {
                                using IntMMA = typename mma::sm90::INT8MMARSSelector<N_SWAP>::type;
                                // Per-half chained accumulation (the activation SF
                                // changes at K64): batches {0,1} chain into one
                                // int32 accumulator, {2,3} into the other, mirroring
                                // the rel-LUT fold's two-half progressive drain --
                                // half 0's promotion overlaps half 1's in-flight
                                // WGMMAs, and the 4-way sums disappear.
                                int32_t iacc_lo[kSwapAccum], iacc_hi[kSwapAccum];
#if DG_W4A8_INT_QOQ
                                const uint32_t s2_r0 = cw_word_r0 & 0xffu;
                                const uint32_t s2_r1 = cw_word_r1 & 0xffu;
#if DG_W4A8_INT_QOQ_ZP
                                // Coeff byte 1 = prepack-precomputed nz = (-z*s2) mod 256.
                                const uint32_t nz_r0 = (cw_word_r0 >> 8) & 0xffu;
                                const uint32_t nz_r1 = (cw_word_r1 >> 8) & 0xffu;
                                const uint32_t lutlo_r0 = __vadd4(s2_r0 * 0x03020100u, nz_r0 * 0x01010101u);
                                const uint32_t luthi_r0 = __vadd4(s2_r0 * 0x07060504u, nz_r0 * 0x01010101u);
                                const uint32_t lutlo_r1 = __vadd4(s2_r1 * 0x03020100u, nz_r1 * 0x01010101u);
                                const uint32_t luthi_r1 = __vadd4(s2_r1 * 0x07060504u, nz_r1 * 0x01010101u);
#else
                                const uint32_t lutlo_r0 = s2_r0 * 0x03020100u, luthi_r0 = s2_r0 * 0x07060504u;
                                const uint32_t lutlo_r1 = s2_r1 * 0x03020100u, luthi_r1 = s2_r1 * 0x07060504u;
#endif
                                const float w_r0 = __uint_as_float(cw_word_r0 & 0xFFFF0000u);
                                const float w_r1 = __uint_as_float(cw_word_r1 & 0xFFFF0000u);
#else
                                const float w_r0 = __uint_as_float(cw_word_r0);
                                const float w_r1 = __uint_as_float(cw_word_r1);
#endif
                                uint32_t ifr[4][4];
                                auto issue_int = [&](int32_t (&acc)[kSwapAccum], const uint32_t B) {
                                    const uint32_t w0 = B == 0 ? rf_words0.x : B == 1 ? rf_words0.y :
                                                        B == 2 ? rf_words0.z : rf_words0.w;
                                    const uint32_t w1 = B == 0 ? rf_words1.x : B == 1 ? rf_words1.y :
                                                        B == 2 ? rf_words1.z : rf_words1.w;
#if DG_W4A8_INT_QOQ
#if DG_W4A8_INT_QOQ_ZP
                                    const uint2 d0 = int4q::decode_uint4_prmt_groups_to_int8_pair_lut_zp(
                                        w0, lutlo_r0, luthi_r0, s2_r0);
                                    const uint2 d1 = int4q::decode_uint4_prmt_groups_to_int8_pair_lut_zp(
                                        w1, lutlo_r1, luthi_r1, s2_r1);
#else
                                    const uint2 d0 = int4q::decode_int4_prmt_groups_to_int8_pair_lut(
                                        w0, lutlo_r0, luthi_r0, s2_r0);
                                    const uint2 d1 = int4q::decode_int4_prmt_groups_to_int8_pair_lut(
                                        w1, lutlo_r1, luthi_r1, s2_r1);
#endif
#else
                                    const uint2 d0 = int4q::decode_int4_prmt_groups_to_int8_pair(w0);
                                    const uint2 d1 = int4q::decode_int4_prmt_groups_to_int8_pair(w1);
#endif
                                    ifr[B][0] = d0.x; ifr[B][1] = d1.x; ifr[B][2] = d0.y; ifr[B][3] = d1.y;
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                        ptx::warpgroup_fence_operand(reinterpret_cast<float&>(acc[i]));
                                    #pragma unroll
                                    for (uint32_t i = 0; i < 4; ++ i)
                                        ptx::warpgroup_fence_operand(reinterpret_cast<float&>(ifr[B][i]));
                                    ptx::warpgroup_arrive();
                                    auto desc_b = mma::sm90::make_smem_desc(smem_a[stage_idx] + B * IntMMA::K, 1);
                                    IntMMA::wgmma(ifr[B], desc_b, acc, (B % 2) != 0);
                                    ptx::warpgroup_commit_batch();
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                        ptx::warpgroup_fence_operand(reinterpret_cast<float&>(acc[i]));
                                };
                                auto promote_int_half = [&](int32_t (&acc)[kSwapAccum], const uint32_t h) {
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                        final_accum[i*4+0] += w_r0 * tok_scale[h][i][0] * static_cast<float>(acc[i*4+0]);
                                        final_accum[i*4+2] += w_r1 * tok_scale[h][i][0] * static_cast<float>(acc[i*4+2]);
                                        final_accum[i*4+1] += w_r0 * tok_scale[h][i][1] * static_cast<float>(acc[i*4+1]);
                                        final_accum[i*4+3] += w_r1 * tok_scale[h][i][1] * static_cast<float>(acc[i*4+3]);
                                    }
                                };
                                issue_int(iacc_lo, 0u);
                                issue_int(iacc_lo, 1u);
                                issue_int(iacc_hi, 2u);
                                issue_int(iacc_hi, 3u);
#if DG_W4A8_INT_SHADOW
                            // Shadow decode: this block's WGMMAs are committed
                            // and in flight; use the idle issue slots to decode
                            // the NEXT stage's B tile (math-dequant path only).
                            if constexpr (kMathDequantSplitN and !kPreDecodedB and kPackedBScratch) {
                                if (k_block_idx + 1 < num_k_blocks) {
                                    const uint32_t next_stage = stage_idx == kNumStages - 1 ? 0u : stage_idx + 1u;
                                    const uint32_t next_phase = next_stage == 0u ? phase ^ 1u : phase;
                                    full_barriers[next_stage]->wait(next_phase);
                                    if constexpr (BLOCK_N == 128) {
                                        mxfp4::dequant_smem_b_from_packed_half_row_unscaled<kUsePRMTGroups, kIntDecodeB>(
                                            reinterpret_cast<uint8_t*>(smem_b[next_stage]),
                                            smem_packed_b[next_stage], epilogue_thread_idx,
                                            smem_b_coeff[next_stage]);
                                    } else {
                                        mxfp4::dequant_smem_b_from_packed_unscaled<kUsePRMTGroups, kIntDecodeB, kQoQFoldB>(
                                            reinterpret_cast<uint8_t*>(smem_b[next_stage]),
                                            smem_packed_b[next_stage], epilogue_thread_idx,
                                            smem_b_coeff[next_stage]);
                                    }
                                    asm volatile("bar.sync 8, 256;" ::: "memory");
                                    cutlass::arch::fence_view_async_shared();
                                    b_decoded_ahead = true;
                                }
                            }
#endif
                                ptx::warpgroup_wait<2>();
                                promote_int_half(iacc_lo, 0u);
                                ptx::warpgroup_wait<0>();
                                arrive_empty_barrier(stage_idx);
                                promote_int_half(iacc_hi, 1u);
                                return;
                            }
#endif

#if DG_MXFP4_REL_LUT
                            if constexpr (kRFDecode) {
                                DG_STATIC_ASSERT(BLOCK_K / SwapWGMMA::K == 4, "Expects 4 K32 batches");
                                // Relative-LUT fold, L2 variant: the activation SF
                                // changes at K64, so fold per HALF (batches {0,1}
                                // and {2,3}) -> two accumulators, two promotions
                                // per K128 instead of four. Half 0's promotion
                                // still overlaps half 1's in-flight WGMMAs.
                                uint32_t em_h[2][2], dw_h[2][2];
                                #pragma unroll
                                for (uint32_t h = 0; h < 2; ++ h) {
                                    #pragma unroll
                                    for (uint32_t r = 0; r < 2; ++ r) {
                                        const uint32_t cw = r == 0 ? cw_word_r0 : cw_word_r1;
                                        const uint32_t b0 = (cw >> (h * 16)) & 0xffu;
                                        const uint32_t b1 = (cw >> (h * 16 + 8)) & 0xffu;
                                        const uint32_t em = max(b0, b1);
                                        em_h[h][r] = em;
                                        dw_h[h][r] = min(em - b0, 13u) | (min(em - b1, 13u) << 8);
                                    }
                                }
                                // Interleaved issue (decode B+1 hides under batch
                                // B's WGMMA) with per-half accumulators: the two
                                // batches of a half accumulate across commit
                                // groups (same-warpgroup WGMMAs execute in order).
                                auto issue_rel = [&](float (&acc)[kSwapAccum], uint32_t (&frag)[4],
                                                     const uint32_t B) {
                                    const uint32_t w0 = B == 0 ? rf_words0.x : B == 1 ? rf_words0.y :
                                                        B == 2 ? rf_words0.z : rf_words0.w;
                                    const uint32_t w1 = B == 0 ? rf_words1.x : B == 1 ? rf_words1.y :
                                                        B == 2 ? rf_words1.z : rf_words1.w;
                                    const uint32_t d0i = (dw_h[B / 2][0] >> ((B % 2) * 8)) & 0xffu;
                                    const uint32_t d1i = (dw_h[B / 2][1] >> ((B % 2) * 8)) & 0xffu;
                                    const uint2 d0 = mxfp4::decode_mxfp4_prmt_groups_to_fp8_pair_lut(
                                        w0, mxfp4::kE2M1RelLut[d0i][0], mxfp4::kE2M1RelLut[d0i][1]);
                                    const uint2 d1 = mxfp4::decode_mxfp4_prmt_groups_to_fp8_pair_lut(
                                        w1, mxfp4::kE2M1RelLut[d1i][0], mxfp4::kE2M1RelLut[d1i][1]);
                                    frag[0] = d0.x; frag[1] = d1.x;
                                    frag[2] = d0.y; frag[3] = d1.y;
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                        ptx::warpgroup_fence_operand(acc[i]);
                                    #pragma unroll
                                    for (uint32_t i = 0; i < 4; ++ i)
                                        ptx::warpgroup_fence_operand(
                                            reinterpret_cast<float&>(frag[i]));
                                    ptx::warpgroup_arrive();
                                    auto desc_b = mma::sm90::make_smem_desc(
                                        smem_a[stage_idx] + B * SwapWGMMA::K, 1);
                                    SwapRS::wgmma(frag, desc_b, acc, (B % 2) != 0);
                                    ptx::warpgroup_commit_batch();
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                        ptx::warpgroup_fence_operand(acc[i]);
                                };
                                auto promote_half = [&](float (&acc)[kSwapAccum], const uint32_t h) {
                                    const float cw_r0 = mxfp4::e8m0_to_float(em_h[h][0]);
                                    const float cw_r1 = mxfp4::e8m0_to_float(em_h[h][1]);
                                    #pragma unroll
                                    for (uint32_t i = 0; i < kSwapAccum / 4; ++ i) {
                                        final_accum[i * 4 + 0] += tok_scale[h][i][0] * l2_sf_lo * cw_r0 * acc[i * 4 + 0];
                                        final_accum[i * 4 + 2] += tok_scale[h][i][0] * l2_sf_lo * cw_r1 * acc[i * 4 + 2];
                                        final_accum[i * 4 + 1] += tok_scale[h][i][1] * l2_sf_lo * cw_r0 * acc[i * 4 + 1];
                                        final_accum[i * 4 + 3] += tok_scale[h][i][1] * l2_sf_lo * cw_r1 * acc[i * 4 + 3];
                                    }
                                };
                                issue_rel(swap_accum, rf_frags[0], 0u);
                                issue_rel(swap_accum, rf_frags[1], 1u);
                                issue_rel(swap_accum2, rf_frags[2], 2u);
                                issue_rel(swap_accum2, rf_frags[3], 3u);
                                ptx::warpgroup_wait<2>();
                                promote_half(swap_accum, 0u);
                                ptx::warpgroup_wait<0>();
                                arrive_empty_barrier(stage_idx);
                                promote_half(swap_accum2, 1u);
                                return;
                            }
#endif

                            DG_STATIC_ASSERT(BLOCK_K / SwapWGMMA::K == 4, "Expects 4 K32 batches");
                            // Mirror the FP8 pipeline: all four WGMMAs issued
                            // back-to-back (one commit group each, own accum
                            // registers), then progressively drained so every
                            // promotion overlaps the remaining in-flight WGMMAs.
                            float swap_accum3[kSwapAccum], swap_accum4[kSwapAccum];
                            issue_batch(swap_accum, rf_frags[0], 0u);
                            issue_batch(swap_accum2, rf_frags[1], 1u);
                            issue_batch(swap_accum3, rf_frags[2], 2u);
                            issue_batch(swap_accum4, rf_frags[3], 3u);
                            ptx::warpgroup_wait<3>();
                            promote_batch(swap_accum, 0u);
                            ptx::warpgroup_wait<2>();
                            promote_batch(swap_accum2, 1u);
                            ptx::warpgroup_wait<1>();
                            promote_batch(swap_accum3, 2u);
                            ptx::warpgroup_wait<0>();
                            // All SMEM reads (scratch words, coeffs, token scales,
                            // WGMMA activations) completed: release the producer
                            // before the final register-only promotion.
                            arrive_empty_barrier(stage_idx);
                            promote_batch(swap_accum4, 3u);
                        };

                        const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
                        if constexpr (kIntermediateHidden <= 2048) {
                            if (n_swap <= 8) {
                                run_swap_ab_l2.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l2.template operator()<16>();
                            } else if (n_swap <= 32) {
                                run_swap_ab_l2.template operator()<32>();
                            } else {
                                run_swap_ab_l2.template operator()<64>();
                            }
                        } else {
#if DG_W4A8_INT && DG_W4A8_INT_L2
                            if (n_swap <= 8) {
                                run_swap_ab_l2.template operator()<8>();
                            } else if (n_swap <= 16) {
                                run_swap_ab_l2.template operator()<16>();
                            } else if (n_swap <= 32) {
                                run_swap_ab_l2.template operator()<32>();
                            } else {
                                run_swap_ab_l2.template operator()<64>();
                            }
#else
                            switch (n_swap) {
                                case 8:  run_swap_ab_l2.template operator()<8>();  break;
                                case 16: run_swap_ab_l2.template operator()<16>(); break;
                                case 24: run_swap_ab_l2.template operator()<24>(); break;
                                case 32: run_swap_ab_l2.template operator()<32>(); break;
                                case 40: run_swap_ab_l2.template operator()<40>(); break;
                                case 48: run_swap_ab_l2.template operator()<48>(); break;
                                case 56: run_swap_ab_l2.template operator()<56>(); break;
                                default: run_swap_ab_l2.template operator()<64>(); break;
                            }
#endif
                        }
                    } else if constexpr (kL2DualAccum) {
                        float accum_hi[kAccumPerThread];

                        const auto desc_a_lo0 = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K, 1);
                        const auto desc_b_lo0 = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K, 1);
                        const auto desc_a_lo1 = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + WGMMA::K, 1);
                        const auto desc_b_lo1 = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + WGMMA::K, 1);
                        const auto desc_a_hi0 = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + BLOCK_K / 2, 1);
                        const auto desc_b_hi0 = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + BLOCK_K / 2, 1);
                        const auto desc_a_hi1 = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + BLOCK_K / 2 + WGMMA::K, 1);
                        const auto desc_b_hi1 = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + BLOCK_K / 2 + WGMMA::K, 1);

                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) {
                            ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_fence_operand(accum_hi[i]);
                        }
                        ptx::warpgroup_arrive();
                        WGMMA::wgmma(desc_a_lo0, desc_b_lo0, accum, false);
                        WGMMA::wgmma(desc_a_lo1, desc_b_lo1, accum, true);
                        WGMMA::wgmma(desc_a_hi0, desc_b_hi0, accum_hi, false);
                        WGMMA::wgmma(desc_a_hi1, desc_b_hi1, accum_hi, true);
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) {
                            ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_fence_operand(accum_hi[i]);
                        }
                        ptx::warpgroup_wait<0>();

                        arrive_empty_barrier(stage_idx);

                        if constexpr (WG_BLOCK_N == 128) {
                            const float scale_0_lo = scale_a_0_lo * l2_sf_lo;
                            const float scale_1_lo = scale_a_1_lo * l2_sf_lo;
                            const float scale_0_hi = scale_a_0_hi * l2_sf_lo;
                            const float scale_1_hi = scale_a_1_hi * l2_sf_lo;
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += scale_0_lo * accum[i*4+0];
                                final_accum[i*4+1] += scale_0_lo * accum[i*4+1];
                                final_accum[i*4+2] += scale_1_lo * accum[i*4+2];
                                final_accum[i*4+3] += scale_1_lo * accum[i*4+3];
                                final_accum[i*4+0] += scale_0_hi * accum_hi[i*4+0];
                                final_accum[i*4+1] += scale_0_hi * accum_hi[i*4+1];
                                final_accum[i*4+2] += scale_1_hi * accum_hi[i*4+2];
                                final_accum[i*4+3] += scale_1_hi * accum_hi[i*4+3];
                            }
                        } else {
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                const float l2_sf = (i < 16u) ? l2_sf_lo : l2_sf_hi;
                                final_accum[i*4+0] += scale_a_0_lo * l2_sf * accum[i*4+0];
                                final_accum[i*4+1] += scale_a_0_lo * l2_sf * accum[i*4+1];
                                final_accum[i*4+2] += scale_a_1_lo * l2_sf * accum[i*4+2];
                                final_accum[i*4+3] += scale_a_1_lo * l2_sf * accum[i*4+3];
                                final_accum[i*4+0] += scale_a_0_hi * l2_sf * accum_hi[i*4+0];
                                final_accum[i*4+1] += scale_a_0_hi * l2_sf * accum_hi[i*4+1];
                                final_accum[i*4+2] += scale_a_1_hi * l2_sf * accum_hi[i*4+2];
                                final_accum[i*4+3] += scale_a_1_hi * l2_sf * accum_hi[i*4+3];
                            }
                        }
                    } else {
#ifdef DG_MXFP4_PROBE_K128_PROMO
                    // PROBE ONLY (numerics invalid): single k128 commit group +
                    // one promotion, emulating fold-2^e-into-decode cost.
                    {
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_wait<0>();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            const uint32_t n_0 = wg_n_idx + i * 8 + col_idx * 2;
                            const float cw_0 = mxfp4::e8m0_to_float(smem_b_coeff[stage_idx][n_0 * 4]);
                            const float cw_1 = mxfp4::e8m0_to_float(smem_b_coeff[stage_idx][(n_0 + 1) * 4]);
                            final_accum[i*4+0] += scale_a_0_lo * cw_0 * accum[i*4+0];
                            final_accum[i*4+1] += scale_a_0_lo * cw_1 * accum[i*4+1];
                            final_accum[i*4+2] += scale_a_1_lo * cw_0 * accum[i*4+2];
                            final_accum[i*4+3] += scale_a_1_lo * cw_1 * accum[i*4+3];
                        }
                        arrive_empty_barrier(stage_idx);
                    }
#elif DG_W4A8_INT && DG_W4A8_INT_L2 && DG_W4A8_INT_QOQ
                    // QoQ L2: s2 is folded into the int8 weights; the per-half
                    // promote keeps only the per-K64 intermediate activation
                    // SF, and the per-row s1 is applied once after the loop.
                    {
                        using IntMMA2 = typename mma::sm90::INT8MMASelector<WG_BLOCK_N>::type;
                        int32_t iacc_q[kAccumPerThread];
                        const auto run_half_q = [&](const uint32_t& h,
                                                    const float& sa_0, const float& sa_1) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(reinterpret_cast<float&>(iacc_q[i]));
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t b = 0; b < 2; ++ b) {
                                const uint32_t k32 = h * 2 + b;
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K + k32 * IntMMA2::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + wg_n_idx * BLOCK_K + k32 * IntMMA2::K, 1);
                                IntMMA2::wgmma(desc_a, desc_b, iacc_q, b > 0);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(reinterpret_cast<float&>(iacc_q[i]));
                            ptx::warpgroup_wait<0>();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                final_accum[i*4+0] += sa_0 * static_cast<float>(iacc_q[i*4+0]);
                                final_accum[i*4+1] += sa_0 * static_cast<float>(iacc_q[i*4+1]);
                                final_accum[i*4+2] += sa_1 * static_cast<float>(iacc_q[i*4+2]);
                                final_accum[i*4+3] += sa_1 * static_cast<float>(iacc_q[i*4+3]);
                            }
                        };
                        run_half_q(0u, scale_a_0_lo, scale_a_1_lo);
                        run_half_q(1u, scale_a_0_hi, scale_a_1_hi);
                        if (k_block_idx + 1 < num_k_blocks)
                            arrive_empty_barrier(stage_idx);
                        else
                            qoq_last_stage = stage_idx;
                    }
#elif DG_W4A8_INT && DG_W4A8_INT_L2
                    // W4A8-int non-swapAB L2: the int8 intermediate SF is
                    // per-64-K, so chain one int32 accumulator per K64 half
                    // (two K32 WGMMAs each) and promote each half once. The
                    // weight scale is one fp32 per K128 riding the coeff slot.
                    {
                        using IntWGMMA = typename mma::sm90::INT8MMASelector<WG_BLOCK_N>::type;
                        int32_t iacc[kAccumPerThread];
                        const auto run_half = [&](const uint32_t& h,
                                                  const float& scale_a_0, const float& scale_a_1) {
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(reinterpret_cast<float&>(iacc[i]));
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t b = 0; b < 2; ++ b) {
                                const uint32_t k32 = h * 2 + b;
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K + k32 * IntWGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + wg_n_idx * BLOCK_K + k32 * IntWGMMA::K, 1);
                                IntWGMMA::wgmma(desc_a, desc_b, iacc, b > 0);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(reinterpret_cast<float&>(iacc[i]));
                            ptx::warpgroup_wait<0>();

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                                const float l2_sf = (i < 16u) ? l2_sf_lo : l2_sf_hi;
                                const uint32_t n_0 = wg_n_idx + i * 8 + col_idx * 2;
                                const float w_0 = __uint_as_float(*reinterpret_cast<const uint32_t*>(
                                    &smem_b_coeff[stage_idx][n_0 * 4]));
                                const float w_1 = __uint_as_float(*reinterpret_cast<const uint32_t*>(
                                    &smem_b_coeff[stage_idx][(n_0 + 1) * 4]));
                                final_accum[i*4+0] += scale_a_0 * l2_sf * w_0 * static_cast<float>(iacc[i*4+0]);
                                final_accum[i*4+1] += scale_a_0 * l2_sf * w_1 * static_cast<float>(iacc[i*4+1]);
                                final_accum[i*4+2] += scale_a_1 * l2_sf * w_0 * static_cast<float>(iacc[i*4+2]);
                                final_accum[i*4+3] += scale_a_1 * l2_sf * w_1 * static_cast<float>(iacc[i*4+3]);
                            }
                        };
                        run_half(0u, scale_a_0_lo, scale_a_1_lo);
                        run_half(1u, scale_a_0_hi, scale_a_1_hi);

                        // Released only after the last promotion (SMEM coefficients).
                        arrive_empty_barrier(stage_idx);
                    }
#else
                    // L2: per-32-K WGMMA batches. The activation SF is per-64-K
                    // (lo for k32 0/1, hi for 2/3); the E8M0 weight coefficient
                    // is a per-column, per-K32 multiplier.
                    #pragma unroll
                    for (uint32_t k32 = 0; k32 < BLOCK_K / WGMMA::K; ++ k32) {
                        const float scale_a_0 = k32 < 2 ? scale_a_0_lo : scale_a_0_hi;
                        const float scale_a_1 = k32 < 2 ? scale_a_1_lo : scale_a_1_hi;
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_arrive();
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + row_block_offset * BLOCK_K + k32 * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + wg_n_idx * BLOCK_K + k32 * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, false);
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_wait<0>();

                        // The legacy per-(128,128) weight SF (l2_sf_lo/hi by N chunk)
                        // is unused for MXFP4 (kept at 1.0); the real dequant scale
                        // is the per-row E8M0 coefficient.
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                            const float l2_sf = (i < 16u) ? l2_sf_lo : l2_sf_hi;
                            const uint32_t n_0 = wg_n_idx + i * 8 + col_idx * 2;
                            const float cw_0 = mxfp4::e8m0_to_float(smem_b_coeff[stage_idx][n_0 * 4 + k32]);
                            const float cw_1 = mxfp4::e8m0_to_float(smem_b_coeff[stage_idx][(n_0 + 1) * 4 + k32]);
                            final_accum[i*4+0] += scale_a_0 * l2_sf * cw_0 * accum[i*4+0];
                            final_accum[i*4+1] += scale_a_0 * l2_sf * cw_1 * accum[i*4+1];
                            final_accum[i*4+2] += scale_a_1 * l2_sf * cw_0 * accum[i*4+2];
                            final_accum[i*4+3] += scale_a_1 * l2_sf * cw_1 * accum[i*4+3];
                        }
                    }

                    // Released only after the last promotion (SMEM coefficients).
                    arrive_empty_barrier(stage_idx);
#endif
                    }
                }
            }
#if DG_W4A8_INT && DG_W4A8_INT_QOQ
            // QoQ end-of-loop promote: s1 (per-row, K-constant) read from the
            // last stage's coeff SMEM, then release that stage. swapAB blocks
            // promote (w = s1) and release inside the loop -- skip entirely.
            if constexpr (!kSwapABActive) {
                if (is_linear1_phase) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const uint32_t n_0 = wg_n_idx + i * 8 + col_idx * 2;
                        const float s1_0 = __uint_as_float(*reinterpret_cast<const uint32_t*>(
                            &smem_b_coeff[qoq_last_stage][n_0 * 4]) & 0xFFFF0000u);
                        const float s1_1 = __uint_as_float(*reinterpret_cast<const uint32_t*>(
                            &smem_b_coeff[qoq_last_stage][(n_0 + 1) * 4]) & 0xFFFF0000u);
                        final_accum[i*4+0] = qoq_scale_a_0 * s1_0 * static_cast<float>(qoq_iacc[i*4+0]);
                        final_accum[i*4+1] = qoq_scale_a_0 * s1_1 * static_cast<float>(qoq_iacc[i*4+1]);
                        final_accum[i*4+2] = qoq_scale_a_1 * s1_0 * static_cast<float>(qoq_iacc[i*4+2]);
                        final_accum[i*4+3] = qoq_scale_a_1 * s1_1 * static_cast<float>(qoq_iacc[i*4+3]);
                    }
                } else {
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const uint32_t n_0 = wg_n_idx + i * 8 + col_idx * 2;
                        const float s1_0 = __uint_as_float(*reinterpret_cast<const uint32_t*>(
                            &smem_b_coeff[qoq_last_stage][n_0 * 4]) & 0xFFFF0000u);
                        const float s1_1 = __uint_as_float(*reinterpret_cast<const uint32_t*>(
                            &smem_b_coeff[qoq_last_stage][(n_0 + 1) * 4]) & 0xFFFF0000u);
                        final_accum[i*4+0] *= s1_0;
                        final_accum[i*4+1] *= s1_1;
                        final_accum[i*4+2] *= s1_0;
                        final_accum[i*4+3] *= s1_1;
                    }
                }
                arrive_empty_barrier(qoq_last_stage);
            }
#endif
            };

            const auto run_inplace_scaled_gemm_loop = [&]() {
                DG_STATIC_ASSERT(not kReuseAccumAsFinal or
                                 (WG_BLOCK_M == 64 and WG_BLOCK_N == 128),
                                 "In-place scaled accumulation expects M64N128 per warpgroup");

                const auto scale_final = [&](const float& scale_r0_gate,
                                             const float& scale_r1_gate,
                                             const float& scale_r0_up,
                                             const float& scale_r1_up) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const float scale_r0 = (i & 1u) ? scale_r0_up : scale_r0_gate;
                        const float scale_r1 = (i & 1u) ? scale_r1_up : scale_r1_gate;
                        final_accum[i*4+0] *= scale_r0;
                        final_accum[i*4+1] *= scale_r0;
                        final_accum[i*4+2] *= scale_r1;
                        final_accum[i*4+3] *= scale_r1;
                    }
                };

                const auto reciprocal = [&](const float& value) {
                    return kFastMath ? math::fast_rcp(value) : 1.0f / value;
                };

                constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
                constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
                constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
                constexpr uint32_t kL1SFPerExpert = (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
                constexpr uint32_t kL2SFPerExpert = (kHidden / 128) * kL2SFKBlocks;

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;
                     advance_pipeline(k_block_idx)) {
                    wait_and_dequant_b_stage(stage_idx, phase);

                    const float scale_a_0_lo =
                        ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
                    const float scale_a_1_lo =
                        ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);

                    if (is_linear1_phase) {
                        constexpr float gate_sf = 1.0f;
                        constexpr float up_sf = 1.0f;
                        const float scale_r0_gate = scale_a_0_lo * gate_sf;
                        const float scale_r1_gate = scale_a_1_lo * gate_sf;
                        const float scale_r0_up = scale_a_0_lo * up_sf;
                        const float scale_r1_up = scale_a_1_lo * up_sf;

                        scale_final(reciprocal(scale_r0_gate), reciprocal(scale_r1_gate),
                                    reciprocal(scale_r0_up), reciprocal(scale_r1_up));

                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(final_accum[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            auto desc_a = mma::sm90::make_smem_desc(
                                smem_a[stage_idx] + row_block_offset * BLOCK_K +
                                    k * WGMMA::K, 1);
                            auto desc_b = mma::sm90::make_smem_desc(
                                smem_b[stage_idx] + wg_n_idx * BLOCK_K +
                                    k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                            ptx::warpgroup_fence_operand(final_accum[i]);
                        ptx::warpgroup_wait<0>();
                        arrive_empty_barrier(stage_idx);

                        scale_final(scale_r0_gate, scale_r1_gate,
                                    scale_r0_up, scale_r1_up);
                    } else {
                        const float scale_a_0_hi = ptx::ld_shared(
                            smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r0);
                        const float scale_a_1_hi = ptx::ld_shared(
                            smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r1);
                        constexpr float l2_sf = 1.0f;

                        const auto run_l2_half = [&](const uint32_t& k_offset,
                                                     const float& scale_a_0,
                                                     const float& scale_a_1) {
                            const float scale_r0 = scale_a_0 * l2_sf;
                            const float scale_r1 = scale_a_1 * l2_sf;
                            scale_final(reciprocal(scale_r0), reciprocal(scale_r1),
                                        reciprocal(scale_r0), reciprocal(scale_r1));

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++ k) {
                                auto desc_a = mma::sm90::make_smem_desc(
                                    smem_a[stage_idx] + row_block_offset * BLOCK_K +
                                        k_offset + k * WGMMA::K, 1);
                                auto desc_b = mma::sm90::make_smem_desc(
                                    smem_b[stage_idx] + wg_n_idx * BLOCK_K +
                                        k_offset + k * WGMMA::K, 1);
                                WGMMA::wgmma(desc_a, desc_b, final_accum, true);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                                ptx::warpgroup_fence_operand(final_accum[i]);
                            ptx::warpgroup_wait<0>();
                            scale_final(scale_r0, scale_r1, scale_r0, scale_r1);
                        };

                        run_l2_half(0, scale_a_0_lo, scale_a_1_lo);
                        run_l2_half(BLOCK_K / 2, scale_a_0_hi, scale_a_1_hi);
                        arrive_empty_barrier(stage_idx);
                    }
                }
            };

            const auto run_l1_dual_k_gemm_loop = [&]() {
                DG_STATIC_ASSERT((kHidden / BLOCK_K) % 2 == 0, "L1 dual-K expects an even number of K blocks");
                constexpr float global_scale = 1.0f;
                float accum_b[kAccumPerThread];

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks;) {
                    const uint32_t stage0 = stage_idx;
                    const uint32_t phase0 = phase;
                    dequant_barriers[stage0]->wait(phase0);

                    const float scale_a0_r0 = ptx::ld_shared(smem_sfa[stage0] + row_offset_r0);
                    const float scale_a0_r1 = ptx::ld_shared(smem_sfa[stage0] + row_offset_r1);
                    const float gate_sf0 = global_scale;
                    const float up_sf0 = global_scale;

                    advance_pipeline(k_block_idx);
                    const uint32_t stage1 = stage_idx;
                    const uint32_t phase1 = phase;
                    dequant_barriers[stage1]->wait(phase1);

                    const float scale_a1_r0 = ptx::ld_shared(smem_sfa[stage1] + row_offset_r0);
                    const float scale_a1_r1 = ptx::ld_shared(smem_sfa[stage1] + row_offset_r1);
                    const float gate_sf1 = global_scale;
                    const float up_sf1 = global_scale;

                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) {
                        ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_fence_operand(accum_b[i]);
                    }
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage0] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage0] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage1] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage1] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum_b, k);
                    }
                    ptx::warpgroup_commit_batch();
                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++ i) {
                        ptx::warpgroup_fence_operand(accum[i]);
                        ptx::warpgroup_fence_operand(accum_b[i]);
                    }
                    ptx::warpgroup_wait<0>();

                    arrive_empty_barrier(stage0);
                    arrive_empty_barrier(stage1);

                    #pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++ i) {
                        const float sb0 = (i & 1u) ? up_sf0 : gate_sf0;
                        const float sb1 = (i & 1u) ? up_sf1 : gate_sf1;
                        final_accum[i*4+0] += scale_a0_r0 * sb0 * accum[i*4+0];
                        final_accum[i*4+1] += scale_a0_r0 * sb0 * accum[i*4+1];
                        final_accum[i*4+2] += scale_a0_r1 * sb0 * accum[i*4+2];
                        final_accum[i*4+3] += scale_a0_r1 * sb0 * accum[i*4+3];
                        final_accum[i*4+0] += scale_a1_r0 * sb1 * accum_b[i*4+0];
                        final_accum[i*4+1] += scale_a1_r0 * sb1 * accum_b[i*4+1];
                        final_accum[i*4+2] += scale_a1_r1 * sb1 * accum_b[i*4+2];
                        final_accum[i*4+3] += scale_a1_r1 * sb1 * accum_b[i*4+3];
                    }

                    advance_pipeline(k_block_idx);
                }
            };

            if constexpr (kReuseAccumAsFinal) {
                run_inplace_scaled_gemm_loop();
            } else if constexpr (kL1DualKAccum) {
                if (is_linear1_phase)
                    run_l1_dual_k_gemm_loop();
                else
                    run_default_gemm_loop();
            } else {
                run_default_gemm_loop();
            }

            #pragma unroll
            for (uint32_t i = 0; i < kAccumPerThread; ++ i)
                final_accum[i] *= expert_global_scale;

            const unsigned long long block_gemm_end = phase_profile_clock();
            if (epilogue_warp_idx == 0 and lane_idx == 0)
                phase_profile_record(kProfileGemmCore, block_gemm_end - block_gemm_start);

            // Skip epilogue when block is past valid M (still must release via empty).
            // A dummy cluster peer may still carry an async L1 store from the
            // previous valid block, so drain it before leaving the L1 wave.
            if (row_block_offset >= valid_m) {
                if constexpr (kAsyncL1TMAStore) {
                    if (is_linear1_phase)
                        drain_all_async_l1_stores();
                }
                if constexpr (MegaMoEPhase::runs_linear1 and kWarpgroupSplitM > 1 and not kSplitNWarpgroups) {
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                } else {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
                return;
            }

            const unsigned long long block_epilogue_start = phase_profile_clock();
            if (is_linear1_phase) {
                if constexpr (kSwapABActive) {
                    auto silu = [](float x) -> float {
                        const float e = kFastMath ? __expf(-x) : expf(-x);
                        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                        return x * sig;
                    };
                    auto clamp_gate = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                    };
                    auto clamp_up = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                    };

                    const uint32_t out_col_base = wg_l1_out_n_idx + warp_idx_in_wg * 8 + row_idx;
                    auto store_l1_swap_chunk = [&](const uint32_t& i) {
                        const uint32_t token_0 = i * 8 + col_idx * 2;
                        const uint32_t token_1 = token_0 + 1;
                        if (token_0 < valid_m) {
                            float g0 = final_accum[i * 4 + 0];
                            float u0 = final_accum[i * 4 + 2];
                            clamp_gate(g0);
                            clamp_up(u0);
                            const float weight_0 = *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + token_0)
                                .get_base_ptr<float>();
                            smem_cd_swap_l1_fp32[token_0 * L1_OUT_BLOCK_N + out_col_base] =
                                silu(g0) * u0 * weight_0;
                        }
                        if (token_1 < valid_m) {
                            float g1 = final_accum[i * 4 + 1];
                            float u1 = final_accum[i * 4 + 3];
                            clamp_gate(g1);
                            clamp_up(u1);
                            const float weight_1 = *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + token_1)
                                .get_base_ptr<float>();
                            smem_cd_swap_l1_fp32[token_1 * L1_OUT_BLOCK_N + out_col_base] =
                                silu(g1) * u1 * weight_1;
                        }
                    };

                    const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
                    store_l1_swap_chunk(0);
                    if (valid_m > 8) {
                        #pragma unroll
                        for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
                            if (i < num_swap_token_chunks)
                                store_l1_swap_chunk(i);
                        }
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    for (uint32_t token = epilogue_thread_idx; token < valid_m; token += kNumEpilogueThreads) {
                        float amax = 0.0f;
                        #pragma unroll
                        for (uint32_t col = 0; col < L1_OUT_BLOCK_N; ++ col) {
                            const float v = smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col];
                            amax = cute::max(amax, cute::abs(v));
                        }
#if DG_W4A8_INT && DG_W4A8_INT_L2
                        // int8 intermediate: SF = amax/127; the bytes ride the
                        // fp8-typed buffer so the two fp8-NaN patterns
                        // 0x7F(+127)/0xFF(-1) are avoided (clamp 126, -1 -> 0).
                        const float sf = amax * (1.0f / 127.0f);
                        const float sf_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
#else
                        float2 amax_pair = {amax, amax};
                        float2 sf_pair, sf_inv_pair;
                        math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                        const float sf = sf_pair.x;
                        const float sf_inv = sf_inv_pair.x;
#endif

                        auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                        const uint32_t token_idx = pool_block_idx * BLOCK_M + token;
                        sf_base_ptr[n_block_idx * kSFRingStrideTokens + token_idx] = sf;

#if DG_W4A8_INT && DG_W4A8_INT_L2
                        #pragma unroll
                        for (uint32_t col = 0; col < L1_OUT_BLOCK_N; col += 2) {
                            auto q8 = [](float v) -> uint8_t {
                                float r = __float2int_rn(v);
                                r = r < -126.0f ? -126.0f : (r > 126.0f ? 126.0f : r);
                                const int8_t q = static_cast<int8_t>(r);
                                return static_cast<uint8_t>(q == -1 ? 0 : q);
                            };
                            const uint8_t q0 = q8(smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col + 0] * sf_inv);
                            const uint8_t q1 = q8(smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col + 1] * sf_inv);
                            const uint16_t pair = q0 | (static_cast<uint16_t>(q1) << 8);
                            auto* ptr = reinterpret_cast<uint16_t*>(
                                smem_cd_swap_l1_fp8 + token * L1_OUT_BLOCK_N + col);
                            *ptr = pair;
                        }
#else
                        #pragma unroll
                        for (uint32_t col = 0; col < L1_OUT_BLOCK_N; col += 2) {
                            const float v0 = smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col + 0] * sf_inv;
                            const float v1 = smem_cd_swap_l1_fp32[token * L1_OUT_BLOCK_N + col + 1] * sf_inv;
                            const __nv_fp8x2_e4m3 pair(make_float2(v0, v1));
                            auto* ptr = reinterpret_cast<uint16_t*>(
                                smem_cd_swap_l1_fp8 + token * L1_OUT_BLOCK_N + col);
                            *ptr = pair.__x;
                        }
#endif
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                    if (epilogue_wg_idx == 0 and warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_swap_l1_fp8,
                            out_n_idx,
                            m_idx);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    ptx::tma_store_wait<0>();

                    const unsigned long long block_epilogue_end = phase_profile_clock();
                    if (epilogue_warp_idx == 0 and lane_idx == 0)
                        phase_profile_record(kProfileL1Epilogue, block_epilogue_end - block_epilogue_start);
                } else {
                // ---------------- L1 EPILOGUE: SwiGLU + FP8 quantize + TMA store ----------------
                // Layout in `final_accum`:
                //   16 chunks of 8 N-cols, each chunk = 4 floats per thread = (r0c0, r0c1, r1c0, r1c1).
                //   Gate chunks: even (0, 2, ..., 14). Up chunks: odd (1, 3, ..., 15).
                //   Pair `p` ∈ [0, 8): gate chunk = 2p, up chunk = 2p+1.
                //
                // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
                // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

                constexpr uint32_t kNumPairs = kAccumPerThread / 8;
                constexpr uint32_t kNumSFGroups = WG_L1_OUT_BLOCK_N / 64;
                DG_STATIC_ASSERT(WG_L1_OUT_BLOCK_N % 64 == 0, "L1 output SF is per 64 columns");
                float swiglu_r0[kNumPairs][2];
                float swiglu_r1[kNumPairs][2];

                // Per-row amax, one scale for each 64-col L1 output group.
                float amax_r0[kNumSFGroups] = {};
                float amax_r1[kNumSFGroups] = {};

                // Compute SwiGLU + per-group amax.
                #pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++ p) {
                    const uint32_t gate = 2 * p, up = 2 * p + 1;
                    const uint32_t sf_group = p / 8;

                    auto clamp_gate = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                    };
                    auto clamp_up = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                    };
                    float g_r0_c0 = final_accum[gate*4 + 0]; clamp_gate(g_r0_c0);
                    float g_r0_c1 = final_accum[gate*4 + 1]; clamp_gate(g_r0_c1);
                    float g_r1_c0 = final_accum[gate*4 + 2]; clamp_gate(g_r1_c0);
                    float g_r1_c1 = final_accum[gate*4 + 3]; clamp_gate(g_r1_c1);
                    float u_r0_c0 = final_accum[up*4   + 0]; clamp_up(u_r0_c0);
                    float u_r0_c1 = final_accum[up*4   + 1]; clamp_up(u_r0_c1);
                    float u_r1_c0 = final_accum[up*4   + 2]; clamp_up(u_r1_c0);
                    float u_r1_c1 = final_accum[up*4   + 3]; clamp_up(u_r1_c1);

                    auto silu = [](float x) {
                        const float e = kFastMath ? __expf(-x) : expf(-x);
                        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
                        return x * sig;
                    };

                    if (valid_r0) {
                        swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
                        swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
                        amax_r0[sf_group] = cute::max(
                            amax_r0[sf_group],
                            cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
                    } else {
                        swiglu_r0[p][0] = 0.0f;
                        swiglu_r0[p][1] = 0.0f;
                    }
                    if (valid_r1) {
                        swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
                        swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
                        amax_r1[sf_group] = cute::max(
                            amax_r1[sf_group],
                            cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
                    } else {
                        swiglu_r1[p][0] = 0.0f;
                        swiglu_r1[p][1] = 0.0f;
                    }
                }


                float weight_r0 = 0.0f, weight_r1 = 0.0f;
                if constexpr (kNumMaxTokensPerRank <= 1024) {
                    const int topk_weight_src_lane = static_cast<int>(lane_idx - col_idx);
                    if (col_idx == 0) {
                        weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r0)
                            .get_base_ptr<float>() : 0.0f;
                        weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                            .get_data_buffer(m_idx + row_offset_r1)
                            .get_base_ptr<float>() : 0.0f;
                    }
                    weight_r0 = __shfl_sync(0xffffffff, weight_r0, topk_weight_src_lane);
                    weight_r1 = __shfl_sync(0xffffffff, weight_r1, topk_weight_src_lane);
                } else {
                    weight_r0 = valid_r0 ? *l1_topk_weights_buffer
                        .get_data_buffer(m_idx + row_offset_r0)
                        .get_base_ptr<float>() : 0.0f;
                    weight_r1 = valid_r1 ? *l1_topk_weights_buffer
                        .get_data_buffer(m_idx + row_offset_r1)
                        .get_base_ptr<float>() : 0.0f;
                }
                #pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++ p) {
                    swiglu_r0[p][0] *= weight_r0;
                    swiglu_r0[p][1] *= weight_r0;
                    swiglu_r1[p][0] *= weight_r1;
                    swiglu_r1[p][1] *= weight_r1;
                }
                #pragma unroll
                for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                    amax_r0[g] *= cute::abs(weight_r0);
                    amax_r1[g] *= cute::abs(weight_r1);
                }
                #pragma unroll
                for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                    amax_r0[g] = math::warp_reduce<4, false>(amax_r0[g], math::ReduceMax<float>());
                    amax_r1[g] = math::warp_reduce<4, false>(amax_r1[g], math::ReduceMax<float>());
                }

                float sf_r0[kNumSFGroups], sf_inv_r0[kNumSFGroups];
                float sf_r1[kNumSFGroups], sf_inv_r1[kNumSFGroups];
#if DG_W4A8_INT && DG_W4A8_INT_L2
                #pragma unroll
                for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                    sf_r0[g] = amax_r0[g] * (1.0f / 127.0f);
                    sf_r1[g] = amax_r1[g] * (1.0f / 127.0f);
                    sf_inv_r0[g] = amax_r0[g] > 0.0f ? 127.0f / amax_r0[g] : 0.0f;
                    sf_inv_r1[g] = amax_r1[g] > 0.0f ? 127.0f / amax_r1[g] : 0.0f;
                }
#else
                #pragma unroll
                for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                    float2 amax_pair = {amax_r0[g], amax_r1[g]};
                    float2 sf_pair, sf_inv_pair;
                    math::get_e4m3_sf_and_sf_inv(amax_pair, sf_pair, sf_inv_pair);
                    sf_r0[g] = sf_pair.x; sf_inv_r0[g] = sf_inv_pair.x;
                    sf_r1[g] = sf_pair.y; sf_inv_r1[g] = sf_inv_pair.y;
                }
#endif

                // Quantize and write to smem_cd_l1 (row-major, no swizzle).
                const uint32_t l1_store_stage = kAsyncL1TMAStore ? async_l1_store_stage : 0u;
                if constexpr (kAsyncL1TMAStore)
                    drain_async_l1_store_stage(l1_store_stage);
                auto* smem_cd_l1_wg = smem_cd_l1
                    + l1_store_stage * SMEM_CD_L1_ASYNC_ELEMS
                    + smem_cd_l1_wg_offset;
                #pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++ p) {
                    const uint32_t sf_group = p / 8;
                    const float v00 = swiglu_r0[p][0] * sf_inv_r0[sf_group];
                    const float v01 = swiglu_r0[p][1] * sf_inv_r0[sf_group];
                    const float v10 = swiglu_r1[p][0] * sf_inv_r1[sf_group];
                    const float v11 = swiglu_r1[p][1] * sf_inv_r1[sf_group];

#if DG_W4A8_INT && DG_W4A8_INT_L2
                    auto q8 = [](float v) -> uint8_t {
                        float r = __float2int_rn(v);
                        r = r < -126.0f ? -126.0f : (r > 126.0f ? 126.0f : r);
                        const int8_t q = static_cast<int8_t>(r);
                        return static_cast<uint8_t>(q == -1 ? 0 : q);
                    };
                    struct { uint16_t __x; } r0_pair{}, r1_pair{};
                    r0_pair.__x = q8(v00) | (static_cast<uint16_t>(q8(v01)) << 8);
                    r1_pair.__x = q8(v10) | (static_cast<uint16_t>(q8(v11)) << 8);
#else
                    const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                    const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));
#endif

                    const uint32_t col = p * 8 + col_idx * 2;
                    auto* p0 = reinterpret_cast<uint16_t*>(
                        smem_cd_l1_wg + r_0 * WG_SMEM_CD_L1_STRIDE_N +
                        (kSplitNWarpgroups ? wg_l1_out_n_idx : 0) + col);
                    auto* p1 = reinterpret_cast<uint16_t*>(
                        smem_cd_l1_wg + r_1 * WG_SMEM_CD_L1_STRIDE_N +
                        (kSplitNWarpgroups ? wg_l1_out_n_idx : 0) + col);
                    if (valid_r0)
                        *p0 = r0_pair.__x;
                    if (valid_r1)
                        *p1 = r1_pair.__x;
                }

                // Write L2-activation SF as float, one value per 64 output columns.
                if (col_idx == 0) {
                    auto sf_base_ptr = l2_sf_buffer.get_base_ptr<float>();
                    const uint32_t token_r0 = pool_block_idx * BLOCK_M + row_offset_r0;
                    const uint32_t token_r1 = pool_block_idx * BLOCK_M + row_offset_r1;
                    const uint32_t base_k_sf_idx = (n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx) / 64u;
                    #pragma unroll
                    for (uint32_t g = 0; g < kNumSFGroups; ++ g) {
                        if (valid_r0)
                            sf_base_ptr[(base_k_sf_idx + g) * kSFRingStrideTokens + token_r0] = sf_r0[g];
                        if (valid_r1)
                            sf_base_ptr[(base_k_sf_idx + g) * kSFRingStrideTokens + token_r1] = sf_r1[g];
                    }
                }

                // Issue TMA store of the entire tile. Padding rows beyond
                // `valid_m` are written with stale/garbage FP8 to the L1-output
                // pool buffer, but they are never consumed downstream: the L2
                // GEMM tile loads them, but its NVLink-scatter epilogue is
                // gated by `m_idx_in_block >= valid_m`, and stale SF in the
                // padding rows can produce NaN accumulators that simply stay
                // in registers (only valid rows are converted to BF16 and
                // STSM'd into smem). Using TMA for partial tiles is a large
                // win for low-batch / decode where every tile is partial.
                if constexpr (kSplitNWarpgroups) {
                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1,
                            out_n_idx,
                            m_idx);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    if constexpr (kAsyncL1TMAStore) {
                        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                        async_l1_store_pending[l1_store_stage] = true;
                        async_l1_store_stage ^= 1u;
                    } else {
                        ptx::tma_store_wait<0>();
                    }
                } else {
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N + wg_l1_out_n_idx;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd_l1_wg,
                            out_n_idx,
                            m_idx + row_block_offset);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                    if constexpr (kAsyncL1TMAStore) {
                        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
                        async_l1_store_pending[l1_store_stage] = true;
                        async_l1_store_stage ^= 1u;
                    } else {
                        ptx::tma_store_wait<0>();
                    }
                }
                const unsigned long long block_epilogue_end = phase_profile_clock();
                if (epilogue_warp_idx == 0 and lane_idx == 0)
                    phase_profile_record(kProfileL1Epilogue, block_epilogue_end - block_epilogue_start);
                }
            } else {
                // ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------
                constexpr uint32_t kNumRowsPerWarp = WG_BLOCK_M / 8;

                if constexpr (kDirectL2Scatter) {
                    DG_STATIC_ASSERT(WG_BLOCK_N == 128, "Direct L2 scatter prototype only supports N128");

                    auto scatter_direct_row = [&](const uint32_t& row_offset, const bool& valid_row,
                                                  const uint32_t& row_accum_offset) {
                        if (valid_row) {
                            uint32_t dst_rank_idx = 0, dst_token_idx = 0, dst_topk_idx = 0;
                            const uint32_t row_group_base = lane_idx - col_idx;
                            if (col_idx == 0) {
                                const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + row_offset);
                                dst_rank_idx = src_metadata.rank_idx;
                                dst_token_idx = src_metadata.token_idx;
                                dst_topk_idx = src_metadata.topk_idx;
                            }
                            const uint32_t row_group_mask = 0xfu << row_group_base;
                            const int src_lane = static_cast<int>(row_group_base);
                            dst_rank_idx = __shfl_sync(row_group_mask, dst_rank_idx, src_lane);
                            dst_token_idx = __shfl_sync(row_group_mask, dst_token_idx, src_lane);
                            dst_topk_idx = __shfl_sync(row_group_mask, dst_topk_idx, src_lane);
                            const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                                   .get_data_buffer(dst_token_idx);
                            auto dst_base = math::advance_ptr<uint8_t>(
                                dst_token.get_base_ptr(), n_idx * sizeof(nv_bfloat16));
                            auto mapped_dst_base = sym_buffer.map(dst_base, dst_rank_idx);

                            #pragma unroll
                            for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                                const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
                                const uint32_t col_lo = chunk_lo * 8 + col_idx * 2;
                                const uint32_t col_hi = chunk_hi * 8 + col_idx * 2;
                                const uint32_t packed_lo = math::cast_into_bf16_and_pack(
                                    final_accum[chunk_lo * 4 + row_accum_offset + 0],
                                    final_accum[chunk_lo * 4 + row_accum_offset + 1]);
                                const uint32_t packed_hi = math::cast_into_bf16_and_pack(
                                    final_accum[chunk_hi * 4 + row_accum_offset + 0],
                                    final_accum[chunk_hi * 4 + row_accum_offset + 1]);
                                *reinterpret_cast<uint32_t*>(mapped_dst_base + col_lo * sizeof(nv_bfloat16)) = packed_lo;
                                *reinterpret_cast<uint32_t*>(mapped_dst_base + col_hi * sizeof(nv_bfloat16)) = packed_hi;
                            }
                        }
                    };

	                    scatter_direct_row(row_offset_r0, valid_r0, 0);
	                    scatter_direct_row(row_offset_r1, valid_r1, 2);
	                } else {
	                    if constexpr (kSwapABActive) {
	                        auto store_bf16 = [&](const uint32_t& token, const uint32_t& col, float value) {
	                            smem_cd_l2[epilogue_wg_idx * WG_BLOCK_M * WG_BLOCK_N + token * WG_BLOCK_N + col] =
	                                __float2bfloat16_rn(value);
	                        };

	                        auto store_l2_swap_chunk = [&](const uint32_t& i) {
	                            const uint32_t token_0 = i * 8 + col_idx * 2;
	                            const uint32_t token_1 = token_0 + 1;
	                            if (token_0 < valid_m) {
	                                store_bf16(token_0, r_0, final_accum[i * 4 + 0]);
	                                store_bf16(token_0, r_1, final_accum[i * 4 + 2]);
	                            }
	                            if (token_1 < valid_m) {
	                                store_bf16(token_1, r_0, final_accum[i * 4 + 1]);
	                                store_bf16(token_1, r_1, final_accum[i * 4 + 3]);
	                            }
	                        };

	                        const uint32_t num_swap_token_chunks = (valid_m + 7u) / 8u;
	                        store_l2_swap_chunk(0);
	                        if (valid_m > 8) {
	                            #pragma unroll
	                            for (uint32_t i = 1; i < kSwapABTokenChunks; ++ i) {
	                                if (i < num_swap_token_chunks)
	                                    store_l2_swap_chunk(i);
	                            }
	                        }
	                    } else {
	                    // STSM into smem_cd_l2 (BF16). Reuse SM100 column-swizzle layout.
	                    #pragma unroll
	                    for (uint32_t i = 0; i < kAccumPerThread / 8; ++ i) {
                        // Each i consumes 8 floats (one 16x256b chunk in SM100 terms).
                        // For SM90 WGMMA layout, 8 floats per i correspond to 2 chunks of 4 floats:
                        //   final_accum[i*8 + (0..3)] = chunk 2i: (r0c0, r0c1, r1c0, r1c1)
                        //   final_accum[i*8 + (4..7)] = chunk 2i+1: same shape
                        const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;

                        // Write to SMEM at appropriate position
                        // Row r_0 cols [chunk_lo*8 + col_idx*2, chunk_lo*8 + col_idx*2 + 1] = r0_lo
                        // Row r_0 cols [chunk_hi*8 + col_idx*2, chunk_hi*8 + col_idx*2 + 1] = r0_hi
                        // Row r_1 cols [chunk_lo*8 + col_idx*2, chunk_lo*8 + col_idx*2 + 1] = r1_lo
                        // Row r_1 cols [chunk_hi*8 + col_idx*2, chunk_hi*8 + col_idx*2 + 1] = r1_hi
                        auto write_pair = [&](uint32_t row, uint32_t col, uint32_t packed) {
                            auto smem_ptr = smem_cd_l2
                                + epilogue_wg_idx * WG_BLOCK_M * WG_BLOCK_N
                                + row * WG_BLOCK_N
                                + col;
                            // BF16 STS: 2 bf16 elements
                            *reinterpret_cast<uint32_t*>(smem_ptr) = packed;
                        };
                        if (valid_r0) {
                            const uint32_t r0_lo = math::cast_into_bf16_and_pack(
                                final_accum[chunk_lo*4 + 0], final_accum[chunk_lo*4 + 1]);
                            const uint32_t r0_hi = math::cast_into_bf16_and_pack(
                                final_accum[chunk_hi*4 + 0], final_accum[chunk_hi*4 + 1]);
                            write_pair(r_0, chunk_lo * 8 + col_idx * 2, r0_lo);
                            write_pair(r_0, chunk_hi * 8 + col_idx * 2, r0_hi);
                        }
                        if (valid_r1) {
                            const uint32_t r1_lo = math::cast_into_bf16_and_pack(
                                final_accum[chunk_lo*4 + 2], final_accum[chunk_lo*4 + 3]);
                            const uint32_t r1_hi = math::cast_into_bf16_and_pack(
                                final_accum[chunk_hi*4 + 2], final_accum[chunk_hi*4 + 3]);
                            write_pair(r_1, chunk_lo * 8 + col_idx * 2, r1_lo);
	                            write_pair(r_1, chunk_hi * 8 + col_idx * 2, r1_hi);
	                        }
	                    }
	                    }

	                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Scatter to remote ranks via NVLink (one row per warp-pair)
                    // Each warpgroup-warp covers 8 unique rows × 2 (r_0 + r_1 doubled by warps)
                    // Lane group of 16 within a warp → 1 row.
                    const uint32_t row_in_warp_block = lane_idx / 16;  // 0 or 1
                    const uint32_t lane_in_row = lane_idx % 16;
                    const uint32_t cols_per_lane = WG_BLOCK_N / 16;
                    static_assert(WG_BLOCK_N == 64 or WG_BLOCK_N == 128 or WG_BLOCK_N == 256,
                                  "L2 scatter supports per-WG N64/N128/N256");

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_wg = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                        const uint32_t m_idx_in_block = row_block_offset + row_in_wg;
                        if (m_idx_in_block >= valid_m) break;

                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx;
                        const uint32_t dst_token_idx = src_metadata.token_idx;
                        const uint32_t dst_topk_idx = src_metadata.topk_idx;

                        auto smem_ptr = smem_cd_l2
                            + epilogue_wg_idx * WG_BLOCK_M * WG_BLOCK_N
                            + row_in_wg * WG_BLOCK_N
                            + lane_in_row * cols_per_lane;
                        const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);

                        if constexpr (WG_BLOCK_N == 256) {
                            const auto packed0 = *reinterpret_cast<uint4*>(smem_ptr);
                            const auto packed1 = *(reinterpret_cast<uint4*>(smem_ptr) + 1);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                n_idx * sizeof(nv_bfloat16) + lane_in_row * 2u * sizeof(uint4));
                            auto mapped_dst_ptr = sym_buffer.map(dst_ptr, dst_rank_idx);
                            mapped_dst_ptr[0] = packed0;
                            mapped_dst_ptr[1] = packed1;
                        } else if constexpr (WG_BLOCK_N == 128) {
                            const auto packed = *reinterpret_cast<uint4*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint4>(
                                dst_token.get_base_ptr(),
                                n_idx * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        } else {
                            const auto packed = *reinterpret_cast<uint2*>(smem_ptr);
                            auto dst_ptr = math::advance_ptr<uint2>(
                                dst_token.get_base_ptr(),
                                n_idx * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint2));
                            *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                        }
                    }

                    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                }
                const unsigned long long block_epilogue_end = phase_profile_clock();
                if (epilogue_warp_idx == 0 and lane_idx == 0)
                    phase_profile_record(kProfileL2Epilogue, block_epilogue_end - block_epilogue_start);
            }
        });
        const unsigned long long math_loop_end = phase_profile_clock();
        if (epilogue_warp_idx == 0 and lane_idx == 0)
            phase_profile_record(kProfileMathLoop, math_loop_end - math_loop_start);

        if constexpr (!MegaMoEPhase::needs_combine) {
            if constexpr (kAsyncL1TMAStore)
                drain_all_async_l1_stores();
            return;
        }

        // ---------------- COMBINE ----------------
        // NVLink barrier first: signals remote ranks that this rank's GEMM
        // outputs (NVLink scatter targets) are fully written.
        const unsigned long long combine_barrier_start = phase_profile_clock();
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                             kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );
        const unsigned long long combine_barrier_end = phase_profile_clock();
        if (epilogue_warp_idx == 0 and lane_idx == 0)
            phase_profile_record(kProfileCombineBarrier, combine_barrier_end - combine_barrier_start);

        // Sync with dispatch (paired with dispatch's pre-cleanup sync) so that
        // dispatch may now safely clean workspace state.
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
        const unsigned long long combine_reduce_start = phase_profile_clock();

        constexpr uint32_t kNumHiddenBytes = kCombineHiddenBytes;
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        constexpr uint32_t kNumChunkSlots = kCombineChunkSlots;
        constexpr uint32_t kNumChunks = kCombineNumChunks;
        constexpr uint32_t kNumChunkBytes = kCombineChunkBytes;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer = math::advance_ptr<uint4>(
            smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumEpilogueWarps) {
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;
                        mask ^= 1 << slot_idx;
                        if (cute::elect_one_sync()) {
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                bool do_reduce = move_mask_and_load(load_stage_idx);

                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx;
                    load_stage_idx ^= 1;
                }

                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
        const unsigned long long combine_reduce_end = phase_profile_clock();
        if (epilogue_warp_idx == 0 and lane_idx == 0)
            phase_profile_record(kProfileCombineReduce, combine_reduce_end - combine_reduce_start);
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
}

template <DG_SM90_MXFP4_MOE_TEMPLATE_PARAMS>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_mxfp4_mega_moe_l1_impl(DG_SM90_MXFP4_MOE_KERNEL_ARGS_DECL) {
    sm90_mxfp4_mega_moe_core<DG_SM90_MXFP4_MOE_CORE_TEMPLATE_ARGS(MegaMoELinear1Phase)>(
        DG_SM90_MXFP4_MOE_KERNEL_ARGS);
}

template <DG_SM90_MXFP4_MOE_TEMPLATE_PARAMS>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_mxfp4_mega_moe_l2_impl(DG_SM90_MXFP4_MOE_KERNEL_ARGS_DECL) {
    sm90_mxfp4_mega_moe_core<DG_SM90_MXFP4_MOE_CORE_TEMPLATE_ARGS(MegaMoELinear2Phase)>(
        DG_SM90_MXFP4_MOE_KERNEL_ARGS);
}

#undef DG_SM90_MXFP4_MOE_TEMPLATE_PARAMS
#undef DG_SM90_MXFP4_MOE_KERNEL_ARGS_DECL
#undef DG_SM90_MXFP4_MOE_CORE_ARGS_DECL
#undef DG_SM90_MXFP4_MOE_KERNEL_ARGS
#undef DG_SM90_MXFP4_MOE_CORE_TEMPLATE_ARGS

} // namespace deep_gemm

#pragma clang diagnostic pop
