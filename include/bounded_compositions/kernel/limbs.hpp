// Unsigned multiprecision arithmetic on little-endian arrays of 64-bit limbs.
//
// Every operand carries its own length. A shorter operand reads as if it were
// zero-extended, so values stored with different widths can be mixed. Outputs
// may alias inputs at the same position.
#ifndef BCOMP_KERNEL_LIMBS_HPP
#define BCOMP_KERNEL_LIMBS_HPP

#include <cstdint>

#include "bounded_compositions/kernel/config.hpp"

namespace bounded_compositions {
namespace kernel {

struct limb_pair {
    limb lo;
    limb hi;
};

BCOMP_KERNEL limb limb_at(const limb* a, int na, int i) { return i < na ? a[i] : 0; }

BCOMP_KERNEL void limbs_zero(limb* a, int n) {
    for (int i = 0; i < n; ++i) a[i] = 0;
}

// out[0..n) = a, truncated or zero-extended.
BCOMP_KERNEL void limbs_copy(const limb* a, int na, limb* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = limb_at(a, na, i);
}

// Number of limbs up to and including the highest nonzero one.
BCOMP_KERNEL int limbs_length(const limb* a, int n) {
    while (n > 0 && a[n - 1] == 0) --n;
    return n;
}

BCOMP_KERNEL bool limbs_is_zero(const limb* a, int n) { return limbs_length(a, n) == 0; }

BCOMP_KERNEL int limbs_bit_length(const limb* a, int n) {
    n = limbs_length(a, n);
    if (n == 0) return 0;
    limb top = a[n - 1];
    int bits = 0;
    while (top != 0) {
        top >>= 1;
        ++bits;
    }
    return 64 * (n - 1) + bits;
}

// True if a is a power of two (a single set bit).
BCOMP_KERNEL bool limbs_is_power_of_two(const limb* a, int n) {
    n = limbs_length(a, n);
    if (n == 0) return false;
    for (int i = 0; i + 1 < n; ++i) {
        if (a[i] != 0) return false;
    }
    return (a[n - 1] & (a[n - 1] - 1)) == 0;
}

// -1, 0 or 1 as a < b, a == b, a > b.
BCOMP_KERNEL int limbs_cmp(const limb* a, int na, const limb* b, int nb) {
    for (int i = (na > nb ? na : nb) - 1; i >= 0; --i) {
        const limb x = limb_at(a, na, i);
        const limb y = limb_at(b, nb, i);
        if (x != y) return x < y ? -1 : 1;
    }
    return 0;
}

// out[0..n) = a + b. Returns the carry out of the top limb.
BCOMP_KERNEL limb limbs_add(const limb* a, int na, const limb* b, int nb, limb* out, int n) {
    limb carry = 0;
    for (int i = 0; i < n; ++i) {
        const limb x = limb_at(a, na, i);
        const limb s = x + limb_at(b, nb, i);
        const limb c1 = s < x ? 1 : 0;
        const limb t = s + carry;
        const limb c2 = t < s ? 1 : 0;
        out[i] = t;
        carry = c1 | c2;
    }
    return carry;
}

// out[0..n) = a - b. Returns 1 if b > a (the result then wrapped).
BCOMP_KERNEL limb limbs_sub(const limb* a, int na, const limb* b, int nb, limb* out, int n) {
    limb borrow = 0;
    for (int i = 0; i < n; ++i) {
        const limb x = limb_at(a, na, i);
        const limb y = limb_at(b, nb, i);
        const limb d = x - y;
        const limb b1 = x < y ? 1 : 0;
        const limb t = d - borrow;
        const limb b2 = d < borrow ? 1 : 0;
        out[i] = t;
        borrow = b1 | b2;
    }
    return borrow;
}

// Full 64 x 64 -> 128 bit product from 32-bit halves. MSVC has no __int128, and
// kernels may not include compiler intrinsics headers.
BCOMP_KERNEL limb_pair mul_wide(limb a, limb b) {
    const limb mask = 0xffffffffu;
    const limb a0 = a & mask, a1 = a >> 32;
    const limb b0 = b & mask, b1 = b >> 32;
    const limb p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const limb mid = (p00 >> 32) + (p01 & mask) + (p10 & mask);
    limb_pair r;
    r.lo = (mid << 32) | (p00 & mask);
    r.hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return r;
}

// a[0..n) = a * m + add. Returns the limb carried out of the top.
BCOMP_KERNEL limb limbs_mul_add_word(limb* a, int n, limb m, limb add) {
    limb carry = add;
    for (int i = 0; i < n; ++i) {
        const limb_pair p = mul_wide(a[i], m);
        const limb lo = p.lo + carry;
        carry = p.hi + (lo < p.lo ? 1 : 0);
        a[i] = lo;
    }
    return carry;
}

// a[0..n) = a / d with 0 < d < 2^32. Returns a % d.
BCOMP_KERNEL std::uint32_t limbs_divmod_u32(limb* a, int n, std::uint32_t d) {
    limb rem = 0;
    for (int i = n - 1; i >= 0; --i) {
        const limb hi = (rem << 32) | (a[i] >> 32);
        const limb qh = hi / d;
        rem = hi % d;
        const limb lo = (rem << 32) | (a[i] & 0xffffffffu);
        const limb ql = lo / d;
        rem = lo % d;
        a[i] = (qh << 32) | ql;
    }
    return static_cast<std::uint32_t>(rem);
}

}  // namespace kernel
}  // namespace bounded_compositions

#endif
