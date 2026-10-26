// Base-2 radical-inverse ordering of the ranks [0, L), with no repeats.
//
// Let m = max(1, bitlen(L - 1)) and w = min(m, 64). Raw index t (a 64-bit
// counter below 2^w) gives X(t) = reverse_w(t) ^ mask_hi, and the rank
//
//     r(t) = X(t) * 2^(m - w) + mask_lo,
//
// where mask_lo < 2^(m - w) is fixed. Raw indices whose rank is not below L are
// skipped; point i of the sequence is the i-th raw index that is kept. With no
// mask and w == m, the first 2^j points are the even grid {c * 2^(m-j)}, and
// every prefix stays spread out; the mask (an XOR on all m bits) gives an
// independent replicate with the same structure.
//
// r(t) < L exactly when X(t) < threshold, with threshold = (L >> (m - w)) + 1
// if mask_lo < L mod 2^(m - w), else L >> (m - w). That threshold can be 2^64,
// which accept_all stands for.
#ifndef BCOMP_KERNEL_RADICAL_HPP
#define BCOMP_KERNEL_RADICAL_HPP

#include <cstdint>

#include "bounded_compositions/kernel/config.hpp"
#include "bounded_compositions/kernel/limbs.hpp"

namespace bounded_compositions {
namespace kernel {

struct radical_view {
    limb threshold;       // keep t when X(t) < threshold (unless accept_all)
    limb mask_hi;         // XOR on the reversed index, below 2^w
    const limb* mask_lo;  // low m - w bits of the mask, added to every rank
    std::int32_t mask_lo_limbs;
    std::int32_t index_bits;  // w, 1..64
    std::int32_t shift;       // m - w
    std::int32_t accept_all;  // threshold == 2^64
    std::uint64_t capacity;   // number of kept raw indices below 2^w
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

BCOMP_KERNEL limb radical_x(const radical_view& v, std::uint64_t t) {
    return bit_reverse(t, v.index_bits) ^ v.mask_hi;
}

BCOMP_KERNEL bool radical_keeps(const radical_view& v, std::uint64_t t) {
    return v.accept_all != 0 || radical_x(v, t) < v.threshold;
}

// Kept raw indices in [0, j), for j < 2^w. Splits [0, j) into aligned blocks
// [a, a + 2^k): inside one, X runs over an arithmetic progression with step
// 2^(w-k) and a fixed offset below the step, so its kept count is one division.
BCOMP_KERNEL std::uint64_t radical_count_below(const radical_view& v, std::uint64_t j) {
    const int w = v.index_bits;
    std::uint64_t total = 0;
    std::uint64_t a = 0;
    for (int k = w - 1; k >= 0; --k) {
        if (((j >> k) & 1u) == 0) continue;
        const std::uint64_t block = std::uint64_t{1} << k;
        if (v.accept_all != 0) {
            total += block;
        } else {
            const int s = w - k;  // 1..64
            const limb low_mask = s == 64 ? ~limb{0} : ((limb{1} << s) - 1);
            const limb offset = (bit_reverse(a, w) ^ v.mask_hi) & low_mask;
            if (v.threshold > offset) {
                // ceil((threshold - offset) / 2^s), capped at the block length
                const limb span = v.threshold - offset;
                const limb kept = s == 64 ? 1 : ((span - 1) >> s) + 1;
                total += kept < block ? kept : block;
            }
        }
        a += block;
    }
    return total;
}

// Raw index of point i, for i < capacity: the smallest t with
// count_below(t) + keeps(t) == i + 1.
BCOMP_KERNEL std::uint64_t radical_raw_index(const radical_view& v, std::uint64_t i) {
    std::uint64_t lo = 0;
    std::uint64_t hi =
        v.index_bits == 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << v.index_bits) - 1);
    while (lo < hi) {
        const std::uint64_t mid = lo + (hi - lo) / 2;
        const std::uint64_t upto = radical_count_below(v, mid) + (radical_keeps(v, mid) ? 1u : 0u);
        if (upto >= i + 1) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

// rank = X * 2^shift + mask_lo into out[0..out_limbs). X's bits and mask_lo's
// bits do not overlap, so this is a placement, not an addition.
BCOMP_KERNEL void radical_rank(const radical_view& v, limb x, limb* out, int out_limbs) {
    limbs_copy(v.mask_lo, v.mask_lo_limbs, out, out_limbs);
    const int word = v.shift / 64;
    const int bit = v.shift % 64;
    if (word < out_limbs) out[word] |= x << bit;
    if (bit != 0 && word + 1 < out_limbs) out[word + 1] |= x >> (64 - bit);
}

}  // namespace kernel
}  // namespace bounded_compositions

#endif
