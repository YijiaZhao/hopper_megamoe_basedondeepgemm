"""Generate the prestored ZP decode LUT for the W4A8-int QoQ zero-point path.

Emits the initializer list for kZpPreLutWords in
deep_gemm/include/deep_gemm/quantization/int4_dequant.cuh.

Layout: 32 rows (raw s2 value 0..31; real data uses 1..18, the rest is
padding so a garbage coeff byte -- e.g. perf benches running on MXFP4-encoded
weights -- can never index outside the table) x 16 zero points z in 0..15.
Entry (s2, z) is a uint2: .x bytes i=0..3, .y bytes i=4..7, each byte
((i - z) * s2) mod 256 -- bit-identical to the runtime __vadd4 build
    lut_lo = __vadd4(s2 * 0x03020100, nz * 0x01010101)  (nz = (-z*s2) mod 256)
since per byte (i*s2 + nz) mod 256 == ((i - z) * s2) mod 256.

Usage: python3 scripts/gen_zp_prelut_table.py > /tmp/zp_lut_words.txt
"""


def word(s2: int, z: int, i0: int) -> int:
    w = 0
    for b in range(4):
        w |= (((i0 + b - z) * s2) & 0xff) << (8 * b)
    return w


def main() -> None:
    words = []
    for s2 in range(32):
        for z in range(16):
            words.append(word(s2, z, 0))
            words.append(word(s2, z, 4))
    assert len(words) == 1024
    lines = []
    for r in range(0, 1024, 8):
        row = ", ".join(f"0x{w:08x}u" for w in words[r:r + 8])
        lines.append(f"    {row},")
    lines[-1] = lines[-1].rstrip(",")
    print(" \\\n".join(["#define DG_ZP_PRELUT_WORDS {"] + lines + ["}"]))


if __name__ == "__main__":
    main()
