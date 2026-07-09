// SPDX-License-Identifier: MIT
//
// INT4 (signed, two's-complement nibble) -> INT8 decode helpers for the
// W4A8-integer SM90 MegaMoE. Decode is UNSCALED: int4 sign-extends exactly
// into int8, IGMMA accumulates the raw integer products in int32, and the
// per-group weight scale x per-token activation scale are applied as float
// multipliers in the epilogue (int32 -> float via I2F, then FMA).
//
// Mirrors int4_dequant's MXFP4 sibling structurally (same PRMT + swapAB-RF
// pipeline); the differences are (1) sign-extend instead of float magnitude
// LUT, (2) no power-of-two fold path -- integer scales are arbitrary FP, so
// the promotion tax is avoided by aligning group_size >= BLOCK_K, not by
// shifting.

#pragma once

#include <cstdint>

namespace deep_gemm {
namespace int4q {

#define DG_INT4_INLINE __device__ __forceinline__

// Magnitude table for the low 3 bits of a nibble: {0,1,2,3,4,5,6,7} as int8.
// .x holds 0..3, .y holds 4..7 -- selected by __byte_perm, same as the MXFP4
// magnitude LUT but with integer values.
constexpr uint32_t kInt4MagLo = 0x03020100u;
constexpr uint32_t kInt4MagHi = 0x07060504u;

// Decode 8 signed int4 nibbles (element i in bits [4i:4i+3]) into 8 int8
// bytes, sign-extended. Two's complement: value v -> (v & 7) - (v & 8), done
// per byte so the -8 never borrows across byte lanes (__vsub4).
DG_INT4_INLINE uint2 decode_int4_to_int8_pair(const uint32_t uq) {
    // Per-nibble low-3-bit selectors for the two output words (elements
    // 0,2,4,6 in one, 1,3,5,7 in the other -- matching the RF fragment order).
    const uint32_t sel_lo = (uq & 0x00000007u) |
                            ((uq >> 4) & 0x00000070u) |
                            ((uq >> 8) & 0x00000700u) |
                            ((uq >> 12) & 0x00007000u);
    const uint32_t sel_hi = ((uq >> 4) & 0x00000007u) |
                            ((uq >> 8) & 0x00000070u) |
                            ((uq >> 12) & 0x00000700u) |
                            ((uq >> 16) & 0x00007000u);

    uint32_t mag_lo = __byte_perm(kInt4MagLo, kInt4MagHi, sel_lo);
    uint32_t mag_hi = __byte_perm(kInt4MagLo, kInt4MagHi, sel_hi);

    // Sign byte = 0x08 where the nibble's bit 3 is set, else 0x00. Nibbles
    // 0,2,4,6 (low word) have their bit 3 at positions 3,11,19,27 == the
    // 0x08080808 mask directly; nibbles 1,3,5,7 (high word) sit one nibble
    // up (positions 7,15,23,31 == 0x80808080), shifted down into place.
    const uint32_t sgn_lo = uq & 0x08080808u;
    const uint32_t sgn_hi = (uq & 0x80808080u) >> 4;

    // (v & 7) - (sign ? 8 : 0), per byte, no cross-lane borrow.
    return make_uint2(__vsub4(mag_lo, sgn_lo), __vsub4(mag_hi, sgn_hi));
}

// Grouped-PRMT variant matching decode_mxfp4_prmt_groups_to_fp8_pair's byte
// order exactly (selectors0 = uq, selectors1 = uq >> 16), so the SAME RF /
// tile-major value prepack used for MXFP4 feeds this decode unchanged -- only
// the scale bytes differ (fp32 weight scale vs E8M0). Sign is two's-complement
// (subtract 8 where the nibble's bit 3 is set) instead of a sign bit; the 0x08
// subtrahend is derived from the MXFP4 sign trick (prmt sign-extend of 0x40).
DG_INT4_INLINE uint2 decode_int4_prmt_groups_to_int8_pair(const uint32_t uq) {
    const uint32_t selectors0 = uq;
    const uint32_t selectors1 = uq >> 16;

    uint32_t mag0 = __byte_perm(kInt4MagLo, kInt4MagHi, selectors0 & 0x7777u);
    uint32_t mag1 = __byte_perm(kInt4MagLo, kInt4MagHi, selectors1 & 0x7777u);
    uint32_t sign_seed0, sign_seed1;
    asm volatile("prmt.b32 %0, %1, %1, %2;"
                 : "=r"(sign_seed0) : "r"(0x40404040u), "r"(selectors0));
    asm volatile("prmt.b32 %0, %1, %1, %2;"
                 : "=r"(sign_seed1) : "r"(0x40404040u), "r"(selectors1));
    // (sign_seed ^ 0x40404040) is 0x40 per byte where bit 3 set; >> 3 -> 0x08.
    const uint32_t sub0 = (sign_seed0 ^ 0x40404040u) >> 3;
    const uint32_t sub1 = (sign_seed1 ^ 0x40404040u) >> 3;
    return make_uint2(__vsub4(mag0, sub0), __vsub4(mag1, sub1));
}

// QoQ scaled-LUT variant: the per-group INTEGER scale s2 is baked into the
// magnitude LUT (lut_lo = s2*{0..3}, lut_hi = s2*{4..7}; both single 32-bit
// multiplies, no cross-byte carry since 7*s2 <= 126), so decode emits the
// FOLDED int8 w4*s2 directly -- integer-domain fold-at-decode. The sign
// subtrahend scales to 8*s2 per signed lane (fits a byte; vsub4 wraps mod
// 256 which is exactly two's complement).
DG_INT4_INLINE uint2 decode_int4_prmt_groups_to_int8_pair_lut(
        const uint32_t uq, const uint32_t lut_lo, const uint32_t lut_hi,
        const uint32_t s2) {
    const uint32_t selectors0 = uq;
    const uint32_t selectors1 = uq >> 16;
    uint32_t mag0 = __byte_perm(lut_lo, lut_hi, selectors0 & 0x7777u);
    uint32_t mag1 = __byte_perm(lut_lo, lut_hi, selectors1 & 0x7777u);
    uint32_t sign_seed0, sign_seed1;
    asm volatile("prmt.b32 %0, %1, %1, %2;"
                 : "=r"(sign_seed0) : "r"(0x40404040u), "r"(selectors0));
    asm volatile("prmt.b32 %0, %1, %1, %2;"
                 : "=r"(sign_seed1) : "r"(0x40404040u), "r"(selectors1));
    const uint32_t sub0 = (((sign_seed0 ^ 0x40404040u) >> 3) * s2);
    const uint32_t sub1 = (((sign_seed1 ^ 0x40404040u) >> 3) * s2);
    return make_uint2(__vsub4(mag0, sub0), __vsub4(mag1, sub1));
}

// Asymmetric (zero-point) variant: weights are uint4 in [0,15] with
// w8 = (w4 - z) * s2. The caller folds -z*s2 into every LUT entry
// (lut = s2*{0..3|4..7} + ((-z*s2) mod 256)*0x01010101), so the low-3-bit
// lookup already carries the offset; lanes with bit3 set ADD 8*s2 (the
// +8 half of the uint4 range) instead of the symmetric variant's subtract.
DG_INT4_INLINE uint2 decode_uint4_prmt_groups_to_int8_pair_lut_zp(
        const uint32_t uq, const uint32_t lut_lo, const uint32_t lut_hi,
        const uint32_t s2) {
    const uint32_t selectors0 = uq;
    const uint32_t selectors1 = uq >> 16;
    uint32_t mag0 = __byte_perm(lut_lo, lut_hi, selectors0 & 0x7777u);
    uint32_t mag1 = __byte_perm(lut_lo, lut_hi, selectors1 & 0x7777u);
    uint32_t seed0, seed1;
    asm volatile("prmt.b32 %0, %1, %1, %2;"
                 : "=r"(seed0) : "r"(0x40404040u), "r"(selectors0));
    asm volatile("prmt.b32 %0, %1, %1, %2;"
                 : "=r"(seed1) : "r"(0x40404040u), "r"(selectors1));
    const uint32_t add0 = (((seed0 ^ 0x40404040u) >> 3) * s2);
    const uint32_t add1 = (((seed1 ^ 0x40404040u) >> 3) * s2);
    return make_uint2(__vadd4(mag0, add0), __vadd4(mag1, add1));
}

// int32 accumulator -> dequantized float. w_scale is the per-group weight
// scale (constant over the tile-K reduction when group_size >= BLOCK_K),
// x_scale the per-token activation scale. Both fp32.
DG_INT4_INLINE float dequant_i32(const int32_t acc, const float w_scale,
                                 const float x_scale) {
    return static_cast<float>(acc) * w_scale * x_scale;
}

#undef DG_INT4_INLINE

} // namespace int4q
} // namespace deep_gemm
