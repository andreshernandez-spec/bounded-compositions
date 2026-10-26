// Base-2 radical-inverse order of the ranks, and its restriction to a region.
//
// A space of size L has m = max(1, bitlen(L - 1)) rank bits and an m-bit mask
// M (zero unless scrambled). Its order sorts the ranks x by
//
//     key(x) = reverse_m(x ^ M).
//
// The first 2^j keys belong to the ranks congruent to M modulo 2^(m-j), an
// even grid, and every prefix stays spread out. A region [b, e) lists its
// ranks in the same order, so sampling a region continues any earlier
// sampling of the space, or of a region holding it, without repeats.
//
// A sequence holds fewer than 2^64 points. For a region of S ranks let
// f = max(0, bitlen(S - 1) - 64). The ranks congruent to M modulo 2^f have the
// smallest keys, and there are C <= 2^64 of them in the region; the sequence
// lists those, written x = x0 + 2^f z with z in [0, C). Their keys order as
// reverse((v0 + z) ^ M') with v0 = x0 >> f and M' = M >> f, so finding z only
// takes the low 64 bits of v0 and M'. For f = 0 the sequence is the whole
// region.
#ifndef BCOMP_KERNEL_RADICAL_HPP
#define BCOMP_KERNEL_RADICAL_HPP

#include <cstdint>

#include "bounded_compositions/kernel/config.hpp"
#include "bounded_compositions/kernel/limbs.hpp"

namespace bounded_compositions {
namespace kernel {

struct radical_view {
    const limb* origin;  // x0
    std::int32_t origin_limbs;
    std::int32_t shift;  // f
    limb origin_high;    // (x0 >> f) mod 2^64
    limb mask;           // (M >> f) mod 2^64
    std::uint64_t last;  // C - 1
};

// Reverses the low w bits of x (1 <= w <= 64).
BCOMP_KERNEL limb bit_reverse(limb x, int w) {
    x = ((x >> 1) & 0x5555555555555555u) | ((x & 0x5555555555555555u) << 1);
    x = ((x >> 2) & 0x3333333333333333u) | ((x & 0x3333333333333333u) << 2);
    x = ((x >> 4) & 0x0F0F0F0F0F0F0F0Fu) | ((x & 0x0F0F0F0F0F0F0F0Fu) << 4);
    x = ((x >> 8) & 0x00FF00FF00FF00FFu) | ((x & 0x00FF00FF00FF00FFu) << 8);
    x = ((x >> 16) & 0x0000FFFF0000FFFFu) | ((x & 0x0000FFFF0000FFFFu) << 16);
    x = (x >> 32) | (x << 32);
    return x >> (64 - w);
}

// Number of z in [0, last] with z = d modulo 2^bits, for d < 2^bits and
// 1 <= bits <= 64.
BCOMP_KERNEL std::uint64_t radical_class_size(std::uint64_t last, std::uint64_t d, int bits) {
    if (d > last) return 0;
    return bits == 64 ? 1 : ((last - d) >> bits) + 1;
}

// The z whose key is the i-th smallest, for i <= last. Fixes the bits of
// v = v0 + z from the lowest up (the key's most significant first), counting
// the z left in each class, until one is left.
BCOMP_KERNEL std::uint64_t radical_select(const radical_view& v, std::uint64_t i) {
    limb c = 0;                   // low bits of v fixed so far
    std::uint64_t d = 0;          // z modulo 2^l, from c
    std::uint64_t left = v.last;  // size of the current class, minus one
    for (int l = 0; left != 0; ++l) {
        const limb bit = limb{1} << l;
        const limb low = l == 63 ? ~limb{0} : (bit << 1) - 1;
        // Key bit 0 first: v's bit l equal to the mask's.
        const limb c0 = c | (v.mask & bit);
        const std::uint64_t d0 = (c0 - v.origin_high) & low;
        const std::uint64_t n0 = radical_class_size(v.last, d0, l + 1);
        if (i < n0) {
            c = c0;
            d = d0;
            left = n0 - 1;
        } else {
            i -= n0;
            c = c0 ^ bit;
            d = (c - v.origin_high) & low;
            left = radical_class_size(v.last, d, l + 1) - 1;
        }
    }
    return d;
}

// rank = x0 + z * 2^f into out[0..out_limbs).
BCOMP_KERNEL void radical_rank(const radical_view& v, std::uint64_t z, limb* out, int out_limbs) {
    limbs_zero(out, out_limbs);
    const int word = v.shift / 64;
    const int bit = v.shift % 64;
    if (word < out_limbs) out[word] = z << bit;
    if (bit != 0 && word + 1 < out_limbs) out[word + 1] = z >> (64 - bit);
    limbs_add(out, out_limbs, v.origin, v.origin_limbs, out, out_limbs);
}

}  // namespace kernel
}  // namespace bounded_compositions

#endif
