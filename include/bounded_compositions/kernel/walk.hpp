// Rank, unrank and lexicographic enumeration over a space_view.
//
// Order is lexicographic in (x[0], ..., x[n-1]), first part slowest. With the
// shifted values y[i] = x[i] - lo[i], the compositions whose part i takes
// value v (given k left to place) number N[i+1][k - v], so the ones with a
// smaller part i number
//
//   cum(v - 1) = sum_{u < v} N[i+1][k - u] = P[i+1][k] - P[i+1][k - v],
//
// with P[.][j] = 0 for j < 0. Every walk below is a loop over the parts.
#ifndef BCOMP_KERNEL_WALK_HPP
#define BCOMP_KERNEL_WALK_HPP

#include <cstdint>

#include "bounded_compositions/kernel/config.hpp"
#include "bounded_compositions/kernel/limbs.hpp"
#include "bounded_compositions/kernel/views.hpp"

namespace bounded_compositions {
namespace kernel {

// Writes the composition of the given rank to out[0..n).
BCOMP_KERNEL status unrank(const space_view& s, const limb* rank, int rank_limbs,
                           std::int32_t* out) {
    if (s.size_limbs == 0 || limbs_cmp(rank, rank_limbs, s.size, s.size_limbs) >= 0) {
        return status::out_of_range;
    }
    limb r[max_limbs];
    limb t[max_limbs];
    int w = s.row_limbs[0];
    limbs_copy(rank, rank_limbs, r, w);
    std::int32_t k = s.total;
    for (int i = 0; i + 1 < s.n; ++i) {
        const int wn = s.row_limbs[i + 1];
        // r < N[i][k]. With T = P[i+1][k] - r > 0, cum(v) > r exactly when
        // P[i+1][k - v - 1] < T, and that holds from some v on, so binary
        // search for the first such v needs comparisons only.
        limbs_sub(table_entry(s, i + 1, k), wn, r, w, t, w);
        std::int32_t a = 0;
        std::int32_t b = s.cap[i] < k ? s.cap[i] : k;
        while (a < b) {
            const std::int32_t mid = a + (b - a) / 2;
            const std::int32_t j = k - mid - 1;
            const bool below = j < 0 || limbs_cmp(table_entry(s, i + 1, j), wn, t, w) < 0;
            if (below) {
                b = mid;
            } else {
                a = mid + 1;
            }
        }
        // New rank inside the chosen part: r - cum(a - 1) = P[i+1][k - a] - T.
        limbs_sub(table_entry(s, i + 1, k - a), wn, t, w, r, w);
        w = wn;
        out[i] = a + s.lo[i];
        k -= a;
    }
    out[s.n - 1] = k + s.lo[s.n - 1];
    return status::ok;
}

// True if x[0..n) lies within the bounds and sums to the total.
BCOMP_KERNEL bool contains(const space_view& s, const std::int32_t* x) {
    if (s.size_limbs == 0) return false;
    std::int64_t sum = 0;
    for (int i = 0; i < s.n; ++i) {
        const std::int64_t y = static_cast<std::int64_t>(x[i]) - s.lo[i];
        if (y < 0 || y > s.cap[i]) return false;
        sum += y;
    }
    return sum == s.total;
}

// Writes the rank of x[0..n) to out[0..out_limbs), out_limbs >= row_limbs[0].
BCOMP_KERNEL status rank(const space_view& s, const std::int32_t* x, limb* out, int out_limbs) {
    if (!contains(s, x)) return status::invalid;
    if (out_limbs < s.row_limbs[0]) return status::too_many_limbs;
    limb t[max_limbs];
    limbs_zero(out, out_limbs);
    std::int32_t k = s.total;
    for (int i = 0; i + 1 < s.n; ++i) {
        const std::int32_t y = x[i] - s.lo[i];
        if (y > 0) {
            const int wn = s.row_limbs[i + 1];
            limbs_sub(table_entry(s, i + 1, k), wn, table_entry(s, i + 1, k - y), wn, t, wn);
            limbs_add(out, out_limbs, t, wn, out, out_limbs);
        }
        k -= y;
    }
    return status::ok;
}

// Fills out[i..n) with the smallest completion that places k into parts
// i..n-1. Requires k <= cap_suffix[i].
BCOMP_KERNEL void fill_smallest(const space_view& s, int i, std::int32_t k, std::int32_t* out) {
    for (int j = i; j < s.n; ++j) {
        const std::int32_t rest = s.cap_suffix[j + 1];
        const std::int32_t y = k > rest ? k - rest : 0;
        out[j] = y + s.lo[j];
        k -= y;
    }
}

// Writes the first composition (rank 0).
BCOMP_KERNEL status first(const space_view& s, std::int32_t* out) {
    if (s.size_limbs == 0) return status::out_of_range;
    fill_smallest(s, 0, s.total, out);
    return status::ok;
}

// Replaces x by its lexicographic successor. Returns out_of_range, leaving x
// unchanged, if x is the last composition. x must be in the space.
BCOMP_KERNEL status next(const space_view& s, std::int32_t* x) {
    if (s.size_limbs == 0) return status::out_of_range;
    std::int32_t suffix = x[s.n - 1] - s.lo[s.n - 1];
    for (int i = s.n - 2; i >= 0; --i) {
        const std::int32_t y = x[i] - s.lo[i];
        suffix += y;
        // Increase part i by one if the parts after it can take the rest.
        const std::int32_t rest = suffix - y - 1;
        if (y < s.cap[i] && rest >= 0 && rest <= s.cap_suffix[i + 1]) {
            x[i] = y + 1 + s.lo[i];
            fill_smallest(s, i + 1, rest, x);
            return status::ok;
        }
    }
    return status::out_of_range;
}

}  // namespace kernel
}  // namespace bounded_compositions

#endif
