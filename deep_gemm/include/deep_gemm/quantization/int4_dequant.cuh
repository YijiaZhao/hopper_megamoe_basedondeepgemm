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
// w8 = (w4 - z) * s2. The caller folds nz = (-z*s2) mod 256 into every LUT
// entry (lut = s2*{0..3|4..7} + nz*0x01010101; nz is PREPACK-precomputed in
// the coeff word's second byte), so the low-3-bit lookup already carries the
// offset; lanes with bit3 set ADD 8*s2 (the +8 half of the uint4 range)
// instead of the symmetric variant's subtract.
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

// SHIFTXOR decode (DG_W4A8_INT_QOQ_ZP_SHIFTXOR=1, customer-suggested): pure
// shift+mask nibble spread + per-byte zero-point subtract. NO s2 fold, NO
// table, NO per-row LUT build -- s2's group width (128) equals the inline
// tier's K128 promote segment, so s2 is deferred to the fp32 promote like s1.
// Output byte i = (w4_i - z), int8 in [-15, 15]. zz is the zero point
// splatted across bytes (z * 0x01010101); __vsub4 keeps the subtract inside
// byte lanes (a plain sub would borrow across lanes). The grouped-PRMT byte
// order is preserved: out.x = nibbles 0..3, out.y = nibbles 4..7 as bytes.
#ifndef DG_W4A8_INT_ZSUB_XOR
#define DG_W4A8_INT_ZSUB_XOR 0
#endif

DG_INT4_INLINE uint2 decode_uint4_prmt_groups_to_int8_pair_zsub(
        const uint32_t uq, const uint32_t zz) {
    // One PRMT each spreads the source bytes with zero gaps:
    // t0 = [b0, 0, b1, 0], t1 = [b2, 0, b3, 0]; then (t | t<<4) & 0x0F0F0F0F
    // drops every nibble into its own byte lane (SHF + one LOP3).
    const uint32_t t0 = __byte_perm(uq, 0u, 0x4140u);
    const uint32_t t1 = __byte_perm(uq, 0u, 0x4342u);
    const uint32_t w0 = (t0 | (t0 << 4)) & 0x0F0F0F0Fu;
    const uint32_t w1 = (t1 | (t1 << 4)) & 0x0F0F0F0Fu;
    return make_uint2(__vsub4(w0, zz), __vsub4(w1, zz));
}

// Direct Marlin-word decode for the RF-fragment reordered layout. Keeping the
// four source bytes intact makes the high and low nibbles directly usable as
// the two IGMMA register words, eliminating both PRMT nibble-spread ops.
DG_INT4_INLINE uint2 decode_uint4_direct_to_int8_pair_zsub(
        const uint32_t uq, const uint32_t zz) {
    constexpr uint32_t kNibbleMask = 0x0F0F0F0Fu;
    const uint32_t w0 = (uq >> 4) & kNibbleMask;
    const uint32_t w1 = uq & kNibbleMask;
#if DG_W4A8_INT_ZSUB_XOR
    // LiquidQuant-style: stay in the uint8 domain so no byte ever borrows,
    // then one XOR flips the MSB back into int8 two's complement.
    const uint32_t zz128 = 0x80808080u - zz;
    return make_uint2((w0 + zz128) ^ 0x80808080u, (w1 + zz128) ^ 0x80808080u);
#else
    return make_uint2(__vsub4(w0, zz), __vsub4(w1, zz));
#endif
}

// Prestored ZP decode LUT (DG_W4A8_INT_QOQ_ZP_PRELUT=1): instead of building
// the per-(row, K128) LUT arithmetically (nz extract + 2 IMAD + 2 vadd4), the
// decode site issues one LDS.64 from a shared-memory table copied from this
// constant array at kernel start. Layout: 32 rows indexed by the RAW s2 byte
// masked to 5 bits (real QoQ-ZP data has s2 in 1..18; rows 0 and 19..31 are
// padding so garbage coeff bytes -- e.g. perf benches on MXFP4-encoded
// weights -- can never index outside the table) x 16 zero points z. Entry
// (s2, z) is a uint2 {lo: bytes i=0..3, hi: bytes i=4..7} with byte
// ((i - z) * s2) mod 256 -- bit-identical to the runtime build, since
// per byte (i*s2 + nz) mod 256 == ((i - z) * s2) mod 256 for
// nz = (-z*s2) mod 256. Generated by scripts/gen_zp_prelut_table.py.
#ifndef DG_W4A8_INT_QOQ_ZP_PRELUT
#define DG_W4A8_INT_QOQ_ZP_PRELUT 0
#endif

#if DG_W4A8_INT_QOQ_ZP_PRELUT
#define DG_ZP_PRELUT_WORDS { \
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, \
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, \
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, \
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, \
    0x03020100u, 0x07060504u, 0x020100ffu, 0x06050403u, 0x0100fffeu, 0x05040302u, 0x00fffefdu, 0x04030201u, \
    0xfffefdfcu, 0x03020100u, 0xfefdfcfbu, 0x020100ffu, 0xfdfcfbfau, 0x0100fffeu, 0xfcfbfaf9u, 0x00fffefdu, \
    0xfbfaf9f8u, 0xfffefdfcu, 0xfaf9f8f7u, 0xfefdfcfbu, 0xf9f8f7f6u, 0xfdfcfbfau, 0xf8f7f6f5u, 0xfcfbfaf9u, \
    0xf7f6f5f4u, 0xfbfaf9f8u, 0xf6f5f4f3u, 0xfaf9f8f7u, 0xf5f4f3f2u, 0xf9f8f7f6u, 0xf4f3f2f1u, 0xf8f7f6f5u, \
    0x06040200u, 0x0e0c0a08u, 0x040200feu, 0x0c0a0806u, 0x0200fefcu, 0x0a080604u, 0x00fefcfau, 0x08060402u, \
    0xfefcfaf8u, 0x06040200u, 0xfcfaf8f6u, 0x040200feu, 0xfaf8f6f4u, 0x0200fefcu, 0xf8f6f4f2u, 0x00fefcfau, \
    0xf6f4f2f0u, 0xfefcfaf8u, 0xf4f2f0eeu, 0xfcfaf8f6u, 0xf2f0eeecu, 0xfaf8f6f4u, 0xf0eeeceau, 0xf8f6f4f2u, \
    0xeeeceae8u, 0xf6f4f2f0u, 0xeceae8e6u, 0xf4f2f0eeu, 0xeae8e6e4u, 0xf2f0eeecu, 0xe8e6e4e2u, 0xf0eeeceau, \
    0x09060300u, 0x15120f0cu, 0x060300fdu, 0x120f0c09u, 0x0300fdfau, 0x0f0c0906u, 0x00fdfaf7u, 0x0c090603u, \
    0xfdfaf7f4u, 0x09060300u, 0xfaf7f4f1u, 0x060300fdu, 0xf7f4f1eeu, 0x0300fdfau, 0xf4f1eeebu, 0x00fdfaf7u, \
    0xf1eeebe8u, 0xfdfaf7f4u, 0xeeebe8e5u, 0xfaf7f4f1u, 0xebe8e5e2u, 0xf7f4f1eeu, 0xe8e5e2dfu, 0xf4f1eeebu, \
    0xe5e2dfdcu, 0xf1eeebe8u, 0xe2dfdcd9u, 0xeeebe8e5u, 0xdfdcd9d6u, 0xebe8e5e2u, 0xdcd9d6d3u, 0xe8e5e2dfu, \
    0x0c080400u, 0x1c181410u, 0x080400fcu, 0x1814100cu, 0x0400fcf8u, 0x14100c08u, 0x00fcf8f4u, 0x100c0804u, \
    0xfcf8f4f0u, 0x0c080400u, 0xf8f4f0ecu, 0x080400fcu, 0xf4f0ece8u, 0x0400fcf8u, 0xf0ece8e4u, 0x00fcf8f4u, \
    0xece8e4e0u, 0xfcf8f4f0u, 0xe8e4e0dcu, 0xf8f4f0ecu, 0xe4e0dcd8u, 0xf4f0ece8u, 0xe0dcd8d4u, 0xf0ece8e4u, \
    0xdcd8d4d0u, 0xece8e4e0u, 0xd8d4d0ccu, 0xe8e4e0dcu, 0xd4d0ccc8u, 0xe4e0dcd8u, 0xd0ccc8c4u, 0xe0dcd8d4u, \
    0x0f0a0500u, 0x231e1914u, 0x0a0500fbu, 0x1e19140fu, 0x0500fbf6u, 0x19140f0au, 0x00fbf6f1u, 0x140f0a05u, \
    0xfbf6f1ecu, 0x0f0a0500u, 0xf6f1ece7u, 0x0a0500fbu, 0xf1ece7e2u, 0x0500fbf6u, 0xece7e2ddu, 0x00fbf6f1u, \
    0xe7e2ddd8u, 0xfbf6f1ecu, 0xe2ddd8d3u, 0xf6f1ece7u, 0xddd8d3ceu, 0xf1ece7e2u, 0xd8d3cec9u, 0xece7e2ddu, \
    0xd3cec9c4u, 0xe7e2ddd8u, 0xcec9c4bfu, 0xe2ddd8d3u, 0xc9c4bfbau, 0xddd8d3ceu, 0xc4bfbab5u, 0xd8d3cec9u, \
    0x120c0600u, 0x2a241e18u, 0x0c0600fau, 0x241e1812u, 0x0600faf4u, 0x1e18120cu, 0x00faf4eeu, 0x18120c06u, \
    0xfaf4eee8u, 0x120c0600u, 0xf4eee8e2u, 0x0c0600fau, 0xeee8e2dcu, 0x0600faf4u, 0xe8e2dcd6u, 0x00faf4eeu, \
    0xe2dcd6d0u, 0xfaf4eee8u, 0xdcd6d0cau, 0xf4eee8e2u, 0xd6d0cac4u, 0xeee8e2dcu, 0xd0cac4beu, 0xe8e2dcd6u, \
    0xcac4beb8u, 0xe2dcd6d0u, 0xc4beb8b2u, 0xdcd6d0cau, 0xbeb8b2acu, 0xd6d0cac4u, 0xb8b2aca6u, 0xd0cac4beu, \
    0x150e0700u, 0x312a231cu, 0x0e0700f9u, 0x2a231c15u, 0x0700f9f2u, 0x231c150eu, 0x00f9f2ebu, 0x1c150e07u, \
    0xf9f2ebe4u, 0x150e0700u, 0xf2ebe4ddu, 0x0e0700f9u, 0xebe4ddd6u, 0x0700f9f2u, 0xe4ddd6cfu, 0x00f9f2ebu, \
    0xddd6cfc8u, 0xf9f2ebe4u, 0xd6cfc8c1u, 0xf2ebe4ddu, 0xcfc8c1bau, 0xebe4ddd6u, 0xc8c1bab3u, 0xe4ddd6cfu, \
    0xc1bab3acu, 0xddd6cfc8u, 0xbab3aca5u, 0xd6cfc8c1u, 0xb3aca59eu, 0xcfc8c1bau, 0xaca59e97u, 0xc8c1bab3u, \
    0x18100800u, 0x38302820u, 0x100800f8u, 0x30282018u, 0x0800f8f0u, 0x28201810u, 0x00f8f0e8u, 0x20181008u, \
    0xf8f0e8e0u, 0x18100800u, 0xf0e8e0d8u, 0x100800f8u, 0xe8e0d8d0u, 0x0800f8f0u, 0xe0d8d0c8u, 0x00f8f0e8u, \
    0xd8d0c8c0u, 0xf8f0e8e0u, 0xd0c8c0b8u, 0xf0e8e0d8u, 0xc8c0b8b0u, 0xe8e0d8d0u, 0xc0b8b0a8u, 0xe0d8d0c8u, \
    0xb8b0a8a0u, 0xd8d0c8c0u, 0xb0a8a098u, 0xd0c8c0b8u, 0xa8a09890u, 0xc8c0b8b0u, 0xa0989088u, 0xc0b8b0a8u, \
    0x1b120900u, 0x3f362d24u, 0x120900f7u, 0x362d241bu, 0x0900f7eeu, 0x2d241b12u, 0x00f7eee5u, 0x241b1209u, \
    0xf7eee5dcu, 0x1b120900u, 0xeee5dcd3u, 0x120900f7u, 0xe5dcd3cau, 0x0900f7eeu, 0xdcd3cac1u, 0x00f7eee5u, \
    0xd3cac1b8u, 0xf7eee5dcu, 0xcac1b8afu, 0xeee5dcd3u, 0xc1b8afa6u, 0xe5dcd3cau, 0xb8afa69du, 0xdcd3cac1u, \
    0xafa69d94u, 0xd3cac1b8u, 0xa69d948bu, 0xcac1b8afu, 0x9d948b82u, 0xc1b8afa6u, 0x948b8279u, 0xb8afa69du, \
    0x1e140a00u, 0x463c3228u, 0x140a00f6u, 0x3c32281eu, 0x0a00f6ecu, 0x32281e14u, 0x00f6ece2u, 0x281e140au, \
    0xf6ece2d8u, 0x1e140a00u, 0xece2d8ceu, 0x140a00f6u, 0xe2d8cec4u, 0x0a00f6ecu, 0xd8cec4bau, 0x00f6ece2u, \
    0xcec4bab0u, 0xf6ece2d8u, 0xc4bab0a6u, 0xece2d8ceu, 0xbab0a69cu, 0xe2d8cec4u, 0xb0a69c92u, 0xd8cec4bau, \
    0xa69c9288u, 0xcec4bab0u, 0x9c92887eu, 0xc4bab0a6u, 0x92887e74u, 0xbab0a69cu, 0x887e746au, 0xb0a69c92u, \
    0x21160b00u, 0x4d42372cu, 0x160b00f5u, 0x42372c21u, 0x0b00f5eau, 0x372c2116u, 0x00f5eadfu, 0x2c21160bu, \
    0xf5eadfd4u, 0x21160b00u, 0xeadfd4c9u, 0x160b00f5u, 0xdfd4c9beu, 0x0b00f5eau, 0xd4c9beb3u, 0x00f5eadfu, \
    0xc9beb3a8u, 0xf5eadfd4u, 0xbeb3a89du, 0xeadfd4c9u, 0xb3a89d92u, 0xdfd4c9beu, 0xa89d9287u, 0xd4c9beb3u, \
    0x9d92877cu, 0xc9beb3a8u, 0x92877c71u, 0xbeb3a89du, 0x877c7166u, 0xb3a89d92u, 0x7c71665bu, 0xa89d9287u, \
    0x24180c00u, 0x54483c30u, 0x180c00f4u, 0x483c3024u, 0x0c00f4e8u, 0x3c302418u, 0x00f4e8dcu, 0x3024180cu, \
    0xf4e8dcd0u, 0x24180c00u, 0xe8dcd0c4u, 0x180c00f4u, 0xdcd0c4b8u, 0x0c00f4e8u, 0xd0c4b8acu, 0x00f4e8dcu, \
    0xc4b8aca0u, 0xf4e8dcd0u, 0xb8aca094u, 0xe8dcd0c4u, 0xaca09488u, 0xdcd0c4b8u, 0xa094887cu, 0xd0c4b8acu, \
    0x94887c70u, 0xc4b8aca0u, 0x887c7064u, 0xb8aca094u, 0x7c706458u, 0xaca09488u, 0x7064584cu, 0xa094887cu, \
    0x271a0d00u, 0x5b4e4134u, 0x1a0d00f3u, 0x4e413427u, 0x0d00f3e6u, 0x4134271au, 0x00f3e6d9u, 0x34271a0du, \
    0xf3e6d9ccu, 0x271a0d00u, 0xe6d9ccbfu, 0x1a0d00f3u, 0xd9ccbfb2u, 0x0d00f3e6u, 0xccbfb2a5u, 0x00f3e6d9u, \
    0xbfb2a598u, 0xf3e6d9ccu, 0xb2a5988bu, 0xe6d9ccbfu, 0xa5988b7eu, 0xd9ccbfb2u, 0x988b7e71u, 0xccbfb2a5u, \
    0x8b7e7164u, 0xbfb2a598u, 0x7e716457u, 0xb2a5988bu, 0x7164574au, 0xa5988b7eu, 0x64574a3du, 0x988b7e71u, \
    0x2a1c0e00u, 0x62544638u, 0x1c0e00f2u, 0x5446382au, 0x0e00f2e4u, 0x46382a1cu, 0x00f2e4d6u, 0x382a1c0eu, \
    0xf2e4d6c8u, 0x2a1c0e00u, 0xe4d6c8bau, 0x1c0e00f2u, 0xd6c8baacu, 0x0e00f2e4u, 0xc8baac9eu, 0x00f2e4d6u, \
    0xbaac9e90u, 0xf2e4d6c8u, 0xac9e9082u, 0xe4d6c8bau, 0x9e908274u, 0xd6c8baacu, 0x90827466u, 0xc8baac9eu, \
    0x82746658u, 0xbaac9e90u, 0x7466584au, 0xac9e9082u, 0x66584a3cu, 0x9e908274u, 0x584a3c2eu, 0x90827466u, \
    0x2d1e0f00u, 0x695a4b3cu, 0x1e0f00f1u, 0x5a4b3c2du, 0x0f00f1e2u, 0x4b3c2d1eu, 0x00f1e2d3u, 0x3c2d1e0fu, \
    0xf1e2d3c4u, 0x2d1e0f00u, 0xe2d3c4b5u, 0x1e0f00f1u, 0xd3c4b5a6u, 0x0f00f1e2u, 0xc4b5a697u, 0x00f1e2d3u, \
    0xb5a69788u, 0xf1e2d3c4u, 0xa6978879u, 0xe2d3c4b5u, 0x9788796au, 0xd3c4b5a6u, 0x88796a5bu, 0xc4b5a697u, \
    0x796a5b4cu, 0xb5a69788u, 0x6a5b4c3du, 0xa6978879u, 0x5b4c3d2eu, 0x9788796au, 0x4c3d2e1fu, 0x88796a5bu, \
    0x30201000u, 0x70605040u, 0x201000f0u, 0x60504030u, 0x1000f0e0u, 0x50403020u, 0x00f0e0d0u, 0x40302010u, \
    0xf0e0d0c0u, 0x30201000u, 0xe0d0c0b0u, 0x201000f0u, 0xd0c0b0a0u, 0x1000f0e0u, 0xc0b0a090u, 0x00f0e0d0u, \
    0xb0a09080u, 0xf0e0d0c0u, 0xa0908070u, 0xe0d0c0b0u, 0x90807060u, 0xd0c0b0a0u, 0x80706050u, 0xc0b0a090u, \
    0x70605040u, 0xb0a09080u, 0x60504030u, 0xa0908070u, 0x50403020u, 0x90807060u, 0x40302010u, 0x80706050u, \
    0x33221100u, 0x77665544u, 0x221100efu, 0x66554433u, 0x1100efdeu, 0x55443322u, 0x00efdecdu, 0x44332211u, \
    0xefdecdbcu, 0x33221100u, 0xdecdbcabu, 0x221100efu, 0xcdbcab9au, 0x1100efdeu, 0xbcab9a89u, 0x00efdecdu, \
    0xab9a8978u, 0xefdecdbcu, 0x9a897867u, 0xdecdbcabu, 0x89786756u, 0xcdbcab9au, 0x78675645u, 0xbcab9a89u, \
    0x67564534u, 0xab9a8978u, 0x56453423u, 0x9a897867u, 0x45342312u, 0x89786756u, 0x34231201u, 0x78675645u, \
    0x36241200u, 0x7e6c5a48u, 0x241200eeu, 0x6c5a4836u, 0x1200eedcu, 0x5a483624u, 0x00eedccau, 0x48362412u, \
    0xeedccab8u, 0x36241200u, 0xdccab8a6u, 0x241200eeu, 0xcab8a694u, 0x1200eedcu, 0xb8a69482u, 0x00eedccau, \
    0xa6948270u, 0xeedccab8u, 0x9482705eu, 0xdccab8a6u, 0x82705e4cu, 0xcab8a694u, 0x705e4c3au, 0xb8a69482u, \
    0x5e4c3a28u, 0xa6948270u, 0x4c3a2816u, 0x9482705eu, 0x3a281604u, 0x82705e4cu, 0x281604f2u, 0x705e4c3au, \
    0x39261300u, 0x85725f4cu, 0x261300edu, 0x725f4c39u, 0x1300eddau, 0x5f4c3926u, 0x00eddac7u, 0x4c392613u, \
    0xeddac7b4u, 0x39261300u, 0xdac7b4a1u, 0x261300edu, 0xc7b4a18eu, 0x1300eddau, 0xb4a18e7bu, 0x00eddac7u, \
    0xa18e7b68u, 0xeddac7b4u, 0x8e7b6855u, 0xdac7b4a1u, 0x7b685542u, 0xc7b4a18eu, 0x6855422fu, 0xb4a18e7bu, \
    0x55422f1cu, 0xa18e7b68u, 0x422f1c09u, 0x8e7b6855u, 0x2f1c09f6u, 0x7b685542u, 0x1c09f6e3u, 0x6855422fu, \
    0x3c281400u, 0x8c786450u, 0x281400ecu, 0x7864503cu, 0x1400ecd8u, 0x64503c28u, 0x00ecd8c4u, 0x503c2814u, \
    0xecd8c4b0u, 0x3c281400u, 0xd8c4b09cu, 0x281400ecu, 0xc4b09c88u, 0x1400ecd8u, 0xb09c8874u, 0x00ecd8c4u, \
    0x9c887460u, 0xecd8c4b0u, 0x8874604cu, 0xd8c4b09cu, 0x74604c38u, 0xc4b09c88u, 0x604c3824u, 0xb09c8874u, \
    0x4c382410u, 0x9c887460u, 0x382410fcu, 0x8874604cu, 0x2410fce8u, 0x74604c38u, 0x10fce8d4u, 0x604c3824u, \
    0x3f2a1500u, 0x937e6954u, 0x2a1500ebu, 0x7e69543fu, 0x1500ebd6u, 0x69543f2au, 0x00ebd6c1u, 0x543f2a15u, \
    0xebd6c1acu, 0x3f2a1500u, 0xd6c1ac97u, 0x2a1500ebu, 0xc1ac9782u, 0x1500ebd6u, 0xac97826du, 0x00ebd6c1u, \
    0x97826d58u, 0xebd6c1acu, 0x826d5843u, 0xd6c1ac97u, 0x6d58432eu, 0xc1ac9782u, 0x58432e19u, 0xac97826du, \
    0x432e1904u, 0x97826d58u, 0x2e1904efu, 0x826d5843u, 0x1904efdau, 0x6d58432eu, 0x04efdac5u, 0x58432e19u, \
    0x422c1600u, 0x9a846e58u, 0x2c1600eau, 0x846e5842u, 0x1600ead4u, 0x6e58422cu, 0x00ead4beu, 0x58422c16u, \
    0xead4bea8u, 0x422c1600u, 0xd4bea892u, 0x2c1600eau, 0xbea8927cu, 0x1600ead4u, 0xa8927c66u, 0x00ead4beu, \
    0x927c6650u, 0xead4bea8u, 0x7c66503au, 0xd4bea892u, 0x66503a24u, 0xbea8927cu, 0x503a240eu, 0xa8927c66u, \
    0x3a240ef8u, 0x927c6650u, 0x240ef8e2u, 0x7c66503au, 0x0ef8e2ccu, 0x66503a24u, 0xf8e2ccb6u, 0x503a240eu, \
    0x452e1700u, 0xa18a735cu, 0x2e1700e9u, 0x8a735c45u, 0x1700e9d2u, 0x735c452eu, 0x00e9d2bbu, 0x5c452e17u, \
    0xe9d2bba4u, 0x452e1700u, 0xd2bba48du, 0x2e1700e9u, 0xbba48d76u, 0x1700e9d2u, 0xa48d765fu, 0x00e9d2bbu, \
    0x8d765f48u, 0xe9d2bba4u, 0x765f4831u, 0xd2bba48du, 0x5f48311au, 0xbba48d76u, 0x48311a03u, 0xa48d765fu, \
    0x311a03ecu, 0x8d765f48u, 0x1a03ecd5u, 0x765f4831u, 0x03ecd5beu, 0x5f48311au, 0xecd5bea7u, 0x48311a03u, \
    0x48301800u, 0xa8907860u, 0x301800e8u, 0x90786048u, 0x1800e8d0u, 0x78604830u, 0x00e8d0b8u, 0x60483018u, \
    0xe8d0b8a0u, 0x48301800u, 0xd0b8a088u, 0x301800e8u, 0xb8a08870u, 0x1800e8d0u, 0xa0887058u, 0x00e8d0b8u, \
    0x88705840u, 0xe8d0b8a0u, 0x70584028u, 0xd0b8a088u, 0x58402810u, 0xb8a08870u, 0x402810f8u, 0xa0887058u, \
    0x2810f8e0u, 0x88705840u, 0x10f8e0c8u, 0x70584028u, 0xf8e0c8b0u, 0x58402810u, 0xe0c8b098u, 0x402810f8u, \
    0x4b321900u, 0xaf967d64u, 0x321900e7u, 0x967d644bu, 0x1900e7ceu, 0x7d644b32u, 0x00e7ceb5u, 0x644b3219u, \
    0xe7ceb59cu, 0x4b321900u, 0xceb59c83u, 0x321900e7u, 0xb59c836au, 0x1900e7ceu, 0x9c836a51u, 0x00e7ceb5u, \
    0x836a5138u, 0xe7ceb59cu, 0x6a51381fu, 0xceb59c83u, 0x51381f06u, 0xb59c836au, 0x381f06edu, 0x9c836a51u, \
    0x1f06edd4u, 0x836a5138u, 0x06edd4bbu, 0x6a51381fu, 0xedd4bba2u, 0x51381f06u, 0xd4bba289u, 0x381f06edu, \
    0x4e341a00u, 0xb69c8268u, 0x341a00e6u, 0x9c82684eu, 0x1a00e6ccu, 0x82684e34u, 0x00e6ccb2u, 0x684e341au, \
    0xe6ccb298u, 0x4e341a00u, 0xccb2987eu, 0x341a00e6u, 0xb2987e64u, 0x1a00e6ccu, 0x987e644au, 0x00e6ccb2u, \
    0x7e644a30u, 0xe6ccb298u, 0x644a3016u, 0xccb2987eu, 0x4a3016fcu, 0xb2987e64u, 0x3016fce2u, 0x987e644au, \
    0x16fce2c8u, 0x7e644a30u, 0xfce2c8aeu, 0x644a3016u, 0xe2c8ae94u, 0x4a3016fcu, 0xc8ae947au, 0x3016fce2u, \
    0x51361b00u, 0xbda2876cu, 0x361b00e5u, 0xa2876c51u, 0x1b00e5cau, 0x876c5136u, 0x00e5caafu, 0x6c51361bu, \
    0xe5caaf94u, 0x51361b00u, 0xcaaf9479u, 0x361b00e5u, 0xaf94795eu, 0x1b00e5cau, 0x94795e43u, 0x00e5caafu, \
    0x795e4328u, 0xe5caaf94u, 0x5e43280du, 0xcaaf9479u, 0x43280df2u, 0xaf94795eu, 0x280df2d7u, 0x94795e43u, \
    0x0df2d7bcu, 0x795e4328u, 0xf2d7bca1u, 0x5e43280du, 0xd7bca186u, 0x43280df2u, 0xbca1866bu, 0x280df2d7u, \
    0x54381c00u, 0xc4a88c70u, 0x381c00e4u, 0xa88c7054u, 0x1c00e4c8u, 0x8c705438u, 0x00e4c8acu, 0x7054381cu, \
    0xe4c8ac90u, 0x54381c00u, 0xc8ac9074u, 0x381c00e4u, 0xac907458u, 0x1c00e4c8u, 0x9074583cu, 0x00e4c8acu, \
    0x74583c20u, 0xe4c8ac90u, 0x583c2004u, 0xc8ac9074u, 0x3c2004e8u, 0xac907458u, 0x2004e8ccu, 0x9074583cu, \
    0x04e8ccb0u, 0x74583c20u, 0xe8ccb094u, 0x583c2004u, 0xccb09478u, 0x3c2004e8u, 0xb094785cu, 0x2004e8ccu, \
    0x573a1d00u, 0xcbae9174u, 0x3a1d00e3u, 0xae917457u, 0x1d00e3c6u, 0x9174573au, 0x00e3c6a9u, 0x74573a1du, \
    0xe3c6a98cu, 0x573a1d00u, 0xc6a98c6fu, 0x3a1d00e3u, 0xa98c6f52u, 0x1d00e3c6u, 0x8c6f5235u, 0x00e3c6a9u, \
    0x6f523518u, 0xe3c6a98cu, 0x523518fbu, 0xc6a98c6fu, 0x3518fbdeu, 0xa98c6f52u, 0x18fbdec1u, 0x8c6f5235u, \
    0xfbdec1a4u, 0x6f523518u, 0xdec1a487u, 0x523518fbu, 0xc1a4876au, 0x3518fbdeu, 0xa4876a4du, 0x18fbdec1u, \
    0x5a3c1e00u, 0xd2b49678u, 0x3c1e00e2u, 0xb496785au, 0x1e00e2c4u, 0x96785a3cu, 0x00e2c4a6u, 0x785a3c1eu, \
    0xe2c4a688u, 0x5a3c1e00u, 0xc4a6886au, 0x3c1e00e2u, 0xa6886a4cu, 0x1e00e2c4u, 0x886a4c2eu, 0x00e2c4a6u, \
    0x6a4c2e10u, 0xe2c4a688u, 0x4c2e10f2u, 0xc4a6886au, 0x2e10f2d4u, 0xa6886a4cu, 0x10f2d4b6u, 0x886a4c2eu, \
    0xf2d4b698u, 0x6a4c2e10u, 0xd4b6987au, 0x4c2e10f2u, 0xb6987a5cu, 0x2e10f2d4u, 0x987a5c3eu, 0x10f2d4b6u, \
    0x5d3e1f00u, 0xd9ba9b7cu, 0x3e1f00e1u, 0xba9b7c5du, 0x1f00e1c2u, 0x9b7c5d3eu, 0x00e1c2a3u, 0x7c5d3e1fu, \
    0xe1c2a384u, 0x5d3e1f00u, 0xc2a38465u, 0x3e1f00e1u, 0xa3846546u, 0x1f00e1c2u, 0x84654627u, 0x00e1c2a3u, \
    0x65462708u, 0xe1c2a384u, 0x462708e9u, 0xc2a38465u, 0x2708e9cau, 0xa3846546u, 0x08e9caabu, 0x84654627u, \
    0xe9caab8cu, 0x65462708u, 0xcaab8c6du, 0x462708e9u, 0xab8c6d4eu, 0x2708e9cau, 0x8c6d4e2fu, 0x08e9caabu \
}

// constexpr shadow of the table for the compile-time equivalence check
// (reading __constant__ storage is not allowed in constant expressions).
constexpr uint32_t kZpPreLutWords[1024] = DG_ZP_PRELUT_WORDS;
// Device-side source for the kernel-start copy into shared memory (2 words
// per entry: [((s2 & 31) * 16 + z) * 2 + {0: lo, 1: hi}], 4096 bytes).
__device__ __constant__ uint32_t kZpPreLutConst[1024] = DG_ZP_PRELUT_WORDS;

// Compile-time check: every table entry equals the arithmetic build the
// runtime path produces (per-byte mod-256 ((i - z) * s2), which the python
// generator and the __vadd4(s2*pattern, nz*0x01010101) build both compute).
constexpr bool zp_prelut_table_matches_arithmetic_build() {
    for (int s2 = 0; s2 < 32; ++s2) {
        for (int z = 0; z < 16; ++z) {
            for (int h = 0; h < 2; ++h) {
                uint32_t w = 0;
                for (int b = 0; b < 4; ++b) {
                    const int i = h * 4 + b;
                    w |= (static_cast<uint32_t>((i - z) * s2) & 0xffu) << (8 * b);
                }
                if (kZpPreLutWords[(s2 * 16 + z) * 2 + h] != w)
                    return false;
            }
        }
    }
    return true;
}
static_assert(zp_prelut_table_matches_arithmetic_build(),
              "ZP prestored LUT deviates from the arithmetic decode build");
#endif // DG_W4A8_INT_QOQ_ZP_PRELUT

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
