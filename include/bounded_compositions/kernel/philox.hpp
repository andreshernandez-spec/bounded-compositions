// Philox4x64-10, the counter-based generator of Salmon et al. (SC 2011).
//
// A block of four 64-bit words is a pure function of a 4-word counter and a
// 2-word key, so point i of a sequence can draw its random bits from counter
// (i, ...) with no state carried between points. This is the same bijection
// as numpy's Philox bit generator (which increments its counter before each
// block).
#ifndef BCOMP_KERNEL_PHILOX_HPP
#define BCOMP_KERNEL_PHILOX_HPP

#include "bounded_compositions/kernel/config.hpp"
#include "bounded_compositions/kernel/limbs.hpp"

namespace bounded_compositions {
namespace kernel {

struct philox_block {
    limb v[4];
};

BCOMP_KERNEL philox_block philox4x64_10(const limb counter[4], const limb key[2]) {
    const limb m0 = 0xD2E7470EE14C6C93u;
    const limb m1 = 0xCA5A826395121157u;
    const limb w0 = 0x9E3779B97F4A7C15u;
    const limb w1 = 0xBB67AE8584CAA73Bu;
    limb c0 = counter[0], c1 = counter[1], c2 = counter[2], c3 = counter[3];
    limb k0 = key[0], k1 = key[1];
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            k0 += w0;
            k1 += w1;
        }
        const limb_pair p0 = mul_wide(m0, c0);
        const limb_pair p1 = mul_wide(m1, c2);
        const limb n0 = p1.hi ^ c1 ^ k0;
        const limb n1 = p1.lo;
        const limb n2 = p0.hi ^ c3 ^ k1;
        const limb n3 = p0.lo;
        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
    }
    philox_block out;
    out.v[0] = c0;
    out.v[1] = c1;
    out.v[2] = c2;
    out.v[3] = c3;
    return out;
}

// Writes a uniform value below bound (bound_limbs limbs, bound > 0) to
// out[0..bound_limbs), by rejection: draw bitlen(bound - 1) random bits from
// counter (index, attempt, word block, 0) until the value is below the bound.
// Each attempt succeeds with probability above 1/2.
BCOMP_KERNEL status uniform_below(const limb* bound, int bound_limbs, const limb key[2],
                                  std::uint64_t index, limb* out) {
    // Width of bound - 1, which every value below the bound fits in.
    const int bits =
        limbs_bit_length(bound, bound_limbs) - (limbs_is_power_of_two(bound, bound_limbs) ? 1 : 0);
    if (bits == 0) {  // bound == 1
        limbs_zero(out, bound_limbs);
        return status::ok;
    }
    const int words = (bits + 63) / 64;
    const int top_bits = bits - 64 * (words - 1);
    const limb top_mask = top_bits == 64 ? ~limb{0} : ((limb{1} << top_bits) - 1);
    for (limb attempt = 0; attempt < 256; ++attempt) {
        limbs_zero(out, bound_limbs);
        for (int w = 0; w < words; w += 4) {
            const limb counter[4] = {index, attempt, static_cast<limb>(w / 4), 0};
            const philox_block b = philox4x64_10(counter, key);
            for (int j = 0; j < 4 && w + j < words; ++j) out[w + j] = b.v[j];
        }
        out[words - 1] &= top_mask;
        if (limbs_cmp(out, bound_limbs, bound, bound_limbs) < 0) return status::ok;
    }
    return status::out_of_range;  // probability below 2^-256
}

}  // namespace kernel
}  // namespace bounded_compositions

#endif
