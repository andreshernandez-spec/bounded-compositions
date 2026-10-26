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
#include "bounded_compositions/kernel/logsum.hpp"
#include "bounded_compositions/kernel/views.hpp"

namespace bounded_compositions {
namespace kernel {

// Writes the composition of the given rank to out[0..n).
// Value (shifted, 0-based) of part i when k is left to place and r < N[i][k]
// (w limbs) is the position among the completions: the v whose cumulative
// range [cum(v - 1), cum(v)) contains r. If rest is not null, writes
// r - cum(v - 1), the position inside the chosen value's completions, to
// rest[0..w).
BCOMP_KERNEL std::int32_t part_value(const space_view& s, int i, std::int32_t k, const limb* r,
                                     int w, limb* rest) {
    const int wn = s.row_limbs[i + 1];
    limb t[max_limbs];
    // With T = P[i+1][k] - r > 0, cum(v) > r exactly when P[i+1][k - v - 1] < T,
    // which holds from some v on, so the binary search needs comparisons only.
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
    // r - cum(a - 1) = P[i+1][k - a] - T.
    if (rest != nullptr) limbs_sub(table_entry(s, i + 1, k - a), wn, t, w, rest, w);
    return a;
}

// Writes the composition of the given rank to out[0..n).
BCOMP_KERNEL status unrank(const space_view& s, const limb* rank, int rank_limbs,
                           std::int32_t* out) {
    if (s.size_limbs == 0 || limbs_cmp(rank, rank_limbs, s.size, s.size_limbs) >= 0) {
        return status::out_of_range;
    }
    limb r[max_limbs];
    int w = s.row_limbs[0];
    limbs_copy(rank, rank_limbs, r, w);
    std::int32_t k = s.total;
    for (int i = 0; i + 1 < s.n; ++i) {
        const std::int32_t a = part_value(s, i, k, r, w, r);
        w = s.row_limbs[i + 1];  // the remainder is below N[i+1][k - a]
        out[i] = a + s.lo[i];
        k -= a;
    }
    out[s.n - 1] = k + s.lo[s.n - 1];
    return status::ok;
}

// The unit-cube decoder: part i (i < n - 1) takes the value whose cumulative
// range of N[i][k] contains floor(u[i] * N[i][k] / 2^64), where u[i] is a
// 64-bit fraction of the unit interval and k what is left to place; the last
// part takes the rest. With independent uniform u this is the sequential
// conditional draw, so compositions come out uniformly (up to 2^-64 rounding
// per part); a value whose conditional share is below 2^-64 cannot be reached.
BCOMP_KERNEL status decode_log(const space_view& s, const limb* u, std::int32_t* out);

BCOMP_KERNEL status decode(const space_view& s, const limb* u, std::int32_t* out) {
    if (s.log_mode != 0) return decode_log(s, u, out);
    if (s.size_limbs == 0) return status::out_of_range;
    limb count[max_limbs + 1];
    std::int32_t k = s.total;
    for (int i = 0; i + 1 < s.n; ++i) {
        const int w = s.row_limbs[i];
        // N[i][k] = P[i][k] - P[i][k - 1], times u[i], keeping the top w limbs.
        const limb* hi = table_entry(s, i, k);
        if (k > 0) {
            limbs_sub(hi, w, hi - w, w, count, w);
        } else {
            limbs_copy(hi, w, count, w);
        }
        count[w] = limbs_mul_add_word(count, w, u[i], 0);
        const std::int32_t a = part_value(s, i, k, count + 1, w, nullptr);
        out[i] = a + s.lo[i];
        k -= a;
    }
    out[s.n - 1] = k + s.lo[s.n - 1];
    return status::ok;
}

// True if x[0..n) lies within the bounds and sums to the total.
BCOMP_KERNEL bool contains(const space_view& s, const std::int32_t* x) {
    if (s.feasible == 0) return false;
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
    if (s.table == nullptr) return status::out_of_range;  // no exact counts
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
    if (s.feasible == 0) return status::out_of_range;
    fill_smallest(s, 0, s.total, out);
    return status::ok;
}

// Replaces x by its lexicographic successor. Returns out_of_range, leaving x
// unchanged, if x is the last composition. x must be in the space.
BCOMP_KERNEL status next(const space_view& s, std::int32_t* x) {
    if (s.feasible == 0) return status::out_of_range;
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

// Log mode: log of N[row][lo] + ... + N[row][hi] (lo <= hi), summed from the
// side with less mass outside the window, P[hi] - P[lo - 1] or S[lo] - S[hi + 1].
// If side is not null, writes 0 for the prefix side and 1 for the suffix side.
BCOMP_KERNEL double log_window(const space_view& s, int row, std::int32_t lo, std::int32_t hi,
                               int* side) {
    const double below = log_prefix_at(s, row, lo - 1);
    const double above = log_suffix_at(s, row, hi + 1);
    if (below <= above) {
        if (side != nullptr) *side = 0;
        return log_sub(log_prefix_at(s, row, hi), below);
    }
    if (side != nullptr) *side = 1;
    return log_sub(log_suffix_at(s, row, lo), above);
}

// u / 2^64 as a double strictly below 1.
BCOMP_KERNEL double unit_fraction(limb u) {
    const double x = static_cast<double>(u) * 0x1p-64;
    return x < 1.0 ? x : 0x1.fffffffffffffp-1;
}

// decode for the logarithmic counting mode. Part i's choice is a window
// [k - m, k] of row i + 1 (value v leaves k - v), searched on the side with
// less mass outside it. The result is clamped to the values that leave a
// feasible completion, so rounding can never produce an invalid composition.
BCOMP_KERNEL status decode_log(const space_view& s, const limb* u, std::int32_t* out) {
    if (s.feasible == 0) return status::out_of_range;
    std::int32_t k = s.total;
    for (int i = 0; i + 1 < s.n; ++i) {
        const int row = i + 1;
        const std::int32_t m = s.cap[i] < k ? s.cap[i] : k;
        const std::int32_t rest = s.cap_suffix[row];
        const std::int32_t vmin = k > rest ? k - rest : 0;
        const double f = unit_fraction(u[i]);
        const double below = log_prefix_at(s, row, k - m - 1);
        const double above = log_suffix_at(s, row, k + 1);
        std::int32_t a = 0;
        std::int32_t b = m;
        if (below <= above) {
            // cum(v) = P[k] - P[k - v - 1] > f W  <=>  P[k - v - 1] < below + (1 - f) W
            const double w = log_sub(log_prefix_at(s, row, k), below);
            const double t = log_add(below, BCOMP_LOG1P(-f) + w);
            while (a < b) {
                const std::int32_t mid = a + (b - a) / 2;
                if (log_prefix_at(s, row, k - mid - 1) < t) {
                    b = mid;
                } else {
                    a = mid + 1;
                }
            }
        } else {
            // cum(v) = S[k - v] - S[k + 1] > f W  <=>  S[k - v] > above + f W
            const double w = log_sub(log_suffix_at(s, row, k - m), above);
            const double t = f > 0.0 ? log_add(above, BCOMP_LOG(f) + w) : above;
            while (a < b) {
                const std::int32_t mid = a + (b - a) / 2;
                if (log_suffix_at(s, row, k - mid) > t) {
                    b = mid;
                } else {
                    a = mid + 1;
                }
            }
        }
        if (a < vmin) a = vmin;
        out[i] = a + s.lo[i];
        k -= a;
    }
    out[s.n - 1] = k + s.lo[s.n - 1];
    return status::ok;
}

}  // namespace kernel
}  // namespace bounded_compositions

#endif
