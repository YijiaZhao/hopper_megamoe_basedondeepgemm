// SPDX-License-Identifier: MIT
//
// MXFP4 (E2M1 + E8M0 scale per 32 K-elements) -> FP8 (E4M3) decode helpers
// for SM90 MegaMoE. Decode is UNSCALED by design: every E2M1 magnitude
// {0, 0.5, 1, 1.5, 2, 3, 4, 6} is exactly representable in E4M3, so a single
// constant 8-byte LUT maps nibbles to FP8 bytes. The E8M0 dequant coefficient
// is NOT folded into the values; it is applied as a float multiplier in the
// accumulator promotion (epilogue) at per-32-K granularity.

#pragma once

#include <cuda_fp8.h>
#include <cstdint>

namespace deep_gemm {
namespace mxfp4 {

#define DG_MXFP4_INLINE __device__ __forceinline__

// E2M1 magnitudes 0..7 -> E4M3 bytes {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
// Packed for __byte_perm: .x holds magnitudes 0..3, .y holds 4..7.
constexpr uint32_t kE2M1ToE4M3MagLo = 0x3C383000u;
constexpr uint32_t kE2M1ToE4M3MagHi = 0x4C484440u;

// Decode 8 E2M1 nibbles (Marlin-style packing: nibble i = element i,
// low nibble first) into 8 unscaled FP8 E4M3 bytes.
// Same selector arithmetic as the NVFP4 path, but the LUT is a compile-time
// constant pair of registers instead of a per-scale shared-memory row.
DG_MXFP4_INLINE uint2 decode_mxfp4_to_fp8_pair(const uint32_t uq) {
    const uint32_t sel_hi = ((uq >> 4) & 0x00000007u) |
                            ((uq >> 8) & 0x00000070u) |
                            ((uq >> 12) & 0x00000700u) |
                            ((uq >> 16) & 0x00007000u);
    const uint32_t sel_lo = (uq & 0x00000007u) |
                            ((uq >> 4) & 0x00000070u) |
                            ((uq >> 8) & 0x00000700u) |
                            ((uq >> 12) & 0x00007000u);

    uint32_t out_hi = __byte_perm(kE2M1ToE4M3MagLo, kE2M1ToE4M3MagHi, sel_hi);
    uint32_t out_lo = __byte_perm(kE2M1ToE4M3MagLo, kE2M1ToE4M3MagHi, sel_lo);
    out_hi |= uq & 0x80808080u;
    out_lo |= (uq << 4) & 0x80808080u;
    return make_uint2(out_hi, out_lo);
}

// Grouped-PRMT variant (iter-89 layout): the prepack regroups eight nibbles
// into two 16-bit PRMT selectors so decode is two byte_perms plus generic
// PRMT sign extraction, identical control flow to the NVFP4 grouped path.
DG_MXFP4_INLINE uint2 decode_mxfp4_prmt_groups_to_fp8_pair(const uint32_t uq) {
    constexpr uint32_t kSignSeed = 0x40404040u;
    const uint32_t selectors0 = uq;
    const uint32_t selectors1 = uq >> 16;

    uint32_t out0 = __byte_perm(kE2M1ToE4M3MagLo, kE2M1ToE4M3MagHi, selectors0 & 0x7777u);
    uint32_t out1 = __byte_perm(kE2M1ToE4M3MagLo, kE2M1ToE4M3MagHi, selectors1 & 0x7777u);
    uint32_t sign_seed0, sign_seed1;
    asm volatile("prmt.b32 %0, %1, %1, %2;"
                 : "=r"(sign_seed0) : "r"(kSignSeed), "r"(selectors0));
    asm volatile("prmt.b32 %0, %1, %1, %2;"
                 : "=r"(sign_seed1) : "r"(kSignSeed), "r"(selectors1));
    out0 |= (sign_seed0 ^ kSignSeed) << 1;
    out1 |= (sign_seed1 ^ kSignSeed) << 1;
    return make_uint2(out0, out1);
}

// E8M0 scale byte -> float dequant coefficient 2^(e - 127).
// e == 0 maps to 2^-127 which is a float denormal; clamp to the smallest
// normal instead (quantizers do not emit it for real weights).
// e == 0xff is NaN per OCP MX spec; callers must reject it at prepack time.
DG_MXFP4_INLINE float e8m0_to_float(const uint32_t e) {
    return __uint_as_float(max(e, 1u) << 23);
}

// Relative pre-scaled LUT: row d holds the E2M1 magnitudes multiplied by
// 2^-d and re-encoded as E4M3 with round-to-nearest-even. Row 0 equals the
// unscaled constants above, so d == 0 (uniform scales inside a K128 block)
// decodes bit-exactly. Rows 0..8 are pure exponent shifts (still exact);
// RNE rounding first appears at d == 9 and row 13 flushes to zero, which
// bounds the K128-local underflow of the single-promotion path. Values are
// magnitudes only (bit 7 clear), so the PRMT sign trick applies unchanged.
constexpr uint32_t kRelLutRows = 14;
__device__ constexpr uint32_t kE2M1RelLut[kRelLutRows][2] = {
    {0x3C383000u, 0x4C484440u},  // d = 0 (== kE2M1ToE4M3Mag{Lo,Hi})
    {0x34302800u, 0x44403C38u},  // d = 1
    {0x2C282000u, 0x3C383430u},  // d = 2
    {0x24201800u, 0x34302C28u},  // d = 3
    {0x1C181000u, 0x2C282420u},  // d = 4
    {0x14100800u, 0x24201C18u},  // d = 5
    {0x0C080400u, 0x1C181410u},  // d = 6
    {0x06040200u, 0x14100C08u},  // d = 7
    {0x03020100u, 0x0C080604u},  // d = 8
    {0x02010000u, 0x06040302u},  // d = 9 (first RNE rounding)
    {0x01000000u, 0x03020201u},  // d = 10
    {0x00000000u, 0x02010100u},  // d = 11
    {0x00000000u, 0x01000000u},  // d = 12
    {0x00000000u, 0x00000000u},  // d = 13 (full flush)
};

// Grouped-PRMT decode against a caller-provided LUT row (relative pre-scaled
// path). Identical instruction count to the constant-register variant: the
// two magnitude words simply come from registers loaded off the SMEM row.
DG_MXFP4_INLINE uint2 decode_mxfp4_prmt_groups_to_fp8_pair_lut(
        const uint32_t uq, const uint32_t lut_lo, const uint32_t lut_hi) {
    constexpr uint32_t kSignSeed = 0x40404040u;
    const uint32_t selectors0 = uq;
    const uint32_t selectors1 = uq >> 16;

    uint32_t out0 = __byte_perm(lut_lo, lut_hi, selectors0 & 0x7777u);
    uint32_t out1 = __byte_perm(lut_lo, lut_hi, selectors1 & 0x7777u);
    uint32_t sign_seed0, sign_seed1;
    asm volatile("prmt.b32 %0, %1, %1, %2;"
                 : "=r"(sign_seed0) : "r"(kSignSeed), "r"(selectors0));
    asm volatile("prmt.b32 %0, %1, %1, %2;"
                 : "=r"(sign_seed1) : "r"(kSignSeed), "r"(selectors1));
    out0 |= (sign_seed0 ^ kSignSeed) << 1;
    out1 |= (sign_seed1 ^ kSignSeed) << 1;
    return make_uint2(out0, out1);
}

// Absolute E8M0-scaled decode used by the Kernel Factory scaled-PRMT path.
// The emitted E4M3 values include an exact x256 factor, removed once during
// accumulator promotion. Valid for E8M0 codes 121..125.
DG_MXFP4_INLINE uint2 decode_mxfp4_prmt_groups_to_fp8_pair_scaled256(
        const uint32_t uq, const uint32_t scale) {
    const uint32_t exponent = scale - 113u;
    const uint32_t lut_lo = exponent * 0x08080800u + 0x0c080000u;
    const uint32_t lut_hi = exponent * 0x08080808u + 0x1c181410u;
    return decode_mxfp4_prmt_groups_to_fp8_pair_lut(uq, lut_lo, lut_hi);
}

#undef DG_MXFP4_INLINE

} // namespace mxfp4
} // namespace deep_gemm
