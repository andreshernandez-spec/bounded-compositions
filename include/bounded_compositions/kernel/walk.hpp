// Rank, unrank and lexicographic enumeration over a space_view.
//
// Order is lexicographic in (x[0], ..., x[n-1]), first part slowest. With the
// shifted values y[i] = x[i] - lo[i], the compositions whose part i takes
// value v (given k left to place and layer c) number N[i+1][k - v][c'], with
// c' = c + the step of v's segment, so the ones with a smaller part i inside
// a segment [a, b] number
//
//   sum_{a <= u < v} N[i+1][k - u][c'] = P[i+1][c'][k - a] - P[i+1][c'][k - v],
//
// with P[.][.][j] = 0 for j < 0, plus the whole of the segments before it.
// Every walk below is a loop over the parts.
#ifndef BCOMP_KERNEL_WALK_HPP
#define BCOMP_KERNEL_WALK_HPP

#include <cstdint>

#include "bounded_compositions/kernel/config.hpp"
#include "bounded_compositions/kernel/limbs.hpp"
#include "bounded_compositions/kernel/logsum.hpp"
#include "bounded_compositions/kernel/views.hpp"

namespace bounded_compositions {
namespace kernel {

// Segment g of part i: allowed shifted values first..last, moving the layer
// by step (views.hpp).
BCOMP_KERNEL std::int32_t seg_first(const space_view& s, int i, int g) {
    return s.seg[9 * i + 3 * g];
}
BCOMP_KERNEL std::int32_t seg_last(const space_view& s, int i, int g) {
    return s.seg[9 * i + 3 * g + 1];
}
BCOMP_KERNEL std::int32_t seg_step(const space_view& s, int i, int g) {
    return s.seg[9 * i + 3 * g + 2];
}

// The step of the segment holding shifted value y of part i, or -1 if no
// segment holds it.
BCOMP_KERNEL std::int32_t step_of(const space_view& s, int i, std::int32_t y) {
    for (int g = 0; g < s.segs[i]; ++g) {
        if (y >= seg_first(s, i, g) && y <= seg_last(s, i, g)) return seg_step(s, i, g);
    }
    return -1;
}

// N[row][layer][k] > 0.
BCOMP_KERNEL bool has_completion(const space_view& s, int row, std::int32_t layer, std::int32_t k) {
    if (k < 0 || layer < 0 || layer >= s.layers) return false;
    const int w = s.row_limbs[row];
    const limb* p = table_entry(s, row, layer, k);
    return k == 0 ? !limbs_is_zero(p, w) : limbs_cmp(p, w, p - w, w) != 0;
}

// N[row][layer][k] into out[0..row_limbs[row]), zero outside the table.
BCOMP_KERNEL void completions(const space_view& s, int row, std::int32_t layer, std::int32_t k,
                              limb* out) {
    const int w = s.row_limbs[row];
    if (k < 0 || layer < 0 || layer >= s.layers) {
        limbs_zero(out, w);
        return;
    }
    const limb* p = table_entry(s, row, layer, k);
    if (k == 0) {
        limbs_copy(p, w, out, w);
    } else {
        limbs_sub(p, w, p - w, w, out, w);
    }
}

// Inside segment [a, b] of part i, moving to layer cn: the smallest v with
// P[i+1][cn][k - v - 1] < t (true at v = b), where t = P[i+1][cn][k - a] minus
// the position inside the segment. Writes the position inside v's
// completions, P[i+1][cn][k - v] - t, to rest if it is not null.
BCOMP_KERNEL std::int32_t search_segment(const space_view& s, int i, std::int32_t k,
                                         std::int32_t cn, std::int32_t a, std::int32_t b,
                                         const limb* t, int wt, limb* rest) {
    const int wn = s.row_limbs[i + 1];
    while (a < b) {
        const std::int32_t mid = a + (b - a) / 2;
        const std::int32_t j = k - mid - 1;
        if (j < 0 || limbs_cmp(table_entry(s, i + 1, cn, j), wn, t, wt) < 0) {
            b = mid;
        } else {
            a = mid + 1;
        }
    }
    if (rest != nullptr) limbs_sub(table_entry(s, i + 1, cn, k - a), wn, t, wt, rest, wn);
    return a;
}

// The shifted value of part i, in layer c with k left to place, whose
// completions hold position r < N[i][k][c] (w limbs), values counted in
// increasing order. Writes the next layer to *layer and, if rest is not null,
// r minus the completions of the smaller values to rest[0..row_limbs[i + 1]).
// Inside a segment, value v's completions are N[i+1][k - v][cn], so the ones
// of the segment's values below v sum to P[i+1][cn][k - a] - P[i+1][cn][k - v]:
// with T = P[k - a] - r, v holds r exactly when P[k - v - 1] < T <= P[k - v],
// and the search needs comparisons only. Returns -1 if r is too large.
BCOMP_KERNEL std::int32_t part_value(const space_view& s, int i, std::int32_t k, std::int32_t c,
                                     const limb* r, int w, limb* rest, std::int32_t* layer) {
    const int wn = s.row_limbs[i + 1];
    const int wt = w > wn ? w : wn;
    limb t[max_limbs];
    if (s.segs[i] == 1) {
        // The only segment holds every r below N[i][k][c].
        const std::int32_t a = seg_first(s, i, 0);
        const std::int32_t b = seg_last(s, i, 0) < k ? seg_last(s, i, 0) : k;
        const std::int32_t cn = c + seg_step(s, i, 0);
        *layer = -1;
        if (a > b || cn >= s.layers) return -1;
        limbs_sub(table_entry(s, i + 1, cn, k - a), wn, r, w, t, wt);
        *layer = cn;
        return search_segment(s, i, k, cn, a, b, t, wt, rest);
    }
    limb left[max_limbs];  // r minus the completions of the segments passed
    limbs_copy(r, w, left, wt);
    for (int g = 0; g < s.segs[i]; ++g) {
        const std::int32_t a = seg_first(s, i, g);
        const std::int32_t b = seg_last(s, i, g) < k ? seg_last(s, i, g) : k;
        const std::int32_t cn = c + seg_step(s, i, g);
        if (a > b || cn >= s.layers) continue;
        const limb* top = table_entry(s, i + 1, cn, k - a);
        const limb* bottom = k - b - 1 >= 0 ? table_entry(s, i + 1, cn, k - b - 1) : nullptr;
        bool inside = false;
        if (limbs_cmp(top, wn, left, wt) > 0) {
            limbs_sub(top, wn, left, wt, t, wt);
            inside = bottom == nullptr || limbs_cmp(bottom, wn, t, wt) < 0;
        }
        if (!inside) {
            // left -= P[k - a] - P[k - b - 1]
            limbs_sub(left, wt, top, wn, left, wt);
            if (bottom != nullptr) limbs_add(left, wt, bottom, wn, left, wt);
            continue;
        }
        *layer = cn;
        return search_segment(s, i, k, cn, a, b, t, wt, rest);
    }
    *layer = -1;
    return -1;
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
    std::int32_t c = s.layer_base;
    for (int i = 0; i + 1 < s.n; ++i) {
        const std::int32_t a = part_value(s, i, k, c, r, w, r, &c);
        if (a < 0) return status::invalid;
        w = s.row_limbs[i + 1];  // the remainder is below N[i+1][k - a][c]
        out[i] = a + s.lo[i];
        k -= a;
    }
    out[s.n - 1] = k + s.lo[s.n - 1];
    return status::ok;
}

// The unit-cube decoder: part i (i < n - 1) takes the value whose cumulative
// range of N[i][k][c] contains floor(u[i] * N[i][k][c] / 2^64), where u[i] is
// a 64-bit fraction of the unit interval and k what is left to place; the last
// part takes the rest. With independent uniform u this is the sequential
// conditional draw, so compositions come out uniformly (up to 2^-64 rounding
// per part); a value whose conditional share is below 2^-64 cannot be reached.
BCOMP_KERNEL status decode_log(const space_view& s, const limb* u, std::int32_t* out);

BCOMP_KERNEL status decode(const space_view& s, const limb* u, std::int32_t* out) {
    if (s.log_mode != 0) return decode_log(s, u, out);
    if (s.size_limbs == 0) return status::out_of_range;
    limb count[max_limbs + 1];
    std::int32_t k = s.total;
    std::int32_t c = s.layer_base;
    for (int i = 0; i + 1 < s.n; ++i) {
        const int w = s.row_limbs[i];
        // N[i][k][c] times u[i], keeping the top w limbs.
        completions(s, i, c, k, count);
        count[w] = limbs_mul_add_word(count, w, u[i], 0);
        const std::int32_t a = part_value(s, i, k, c, count + 1, w, nullptr, &c);
        if (a < 0) return status::invalid;
        out[i] = a + s.lo[i];
        k -= a;
    }
    out[s.n - 1] = k + s.lo[s.n - 1];
    return status::ok;
}

// True if x[0..n) lies within the bounds and segments, sums to the total and
// has an allowed number of nonzero parts.
BCOMP_KERNEL bool contains(const space_view& s, const std::int32_t* x) {
    if (s.feasible == 0) return false;
    std::int64_t sum = 0;
    std::int32_t nonzero = s.layer_base;
    for (int i = 0; i < s.n; ++i) {
        const std::int64_t y = static_cast<std::int64_t>(x[i]) - s.lo[i];
        if (y < 0 || y > s.cap[i]) return false;
        if (step_of(s, i, static_cast<std::int32_t>(y)) < 0) return false;
        if (x[i] != 0) ++nonzero;
        sum += y;
    }
    return sum == s.total && nonzero >= s.count_min && nonzero <= s.count_max;
}

// Adds to out[0..out_limbs) the completions of part i's allowed values below y,
// in layer c with k left to place.
BCOMP_KERNEL void add_below(const space_view& s, int i, std::int32_t k, std::int32_t c,
                            std::int32_t y, limb* out, int out_limbs) {
    const int wn = s.row_limbs[i + 1];
    limb t[max_limbs];
    for (int g = 0; g < s.segs[i]; ++g) {
        const std::int32_t a = seg_first(s, i, g);
        if (a >= y) break;
        std::int32_t b = seg_last(s, i, g);
        if (b > y - 1) b = y - 1;
        if (b > k) b = k;
        const std::int32_t cn = c + seg_step(s, i, g);
        if (a > b || cn >= s.layers) continue;
        const limb* top = table_entry(s, i + 1, cn, k - a);
        if (k - b - 1 >= 0) {
            limbs_sub(top, wn, table_entry(s, i + 1, cn, k - b - 1), wn, t, wn);
        } else {
            limbs_copy(top, wn, t, wn);
        }
        limbs_add(out, out_limbs, t, wn, out, out_limbs);
    }
}

// Writes the rank of x[0..n) to out[0..out_limbs), out_limbs >= row_limbs[0].
BCOMP_KERNEL status rank(const space_view& s, const std::int32_t* x, limb* out, int out_limbs) {
    if (s.table == nullptr) return status::out_of_range;  // no exact counts
    if (!contains(s, x)) return status::invalid;
    if (out_limbs < s.row_limbs[0]) return status::too_many_limbs;
    limbs_zero(out, out_limbs);
    std::int32_t k = s.total;
    std::int32_t c = s.layer_base;
    for (int i = 0; i + 1 < s.n; ++i) {
        const std::int32_t y = x[i] - s.lo[i];
        add_below(s, i, k, c, y, out, out_limbs);
        c += step_of(s, i, y);
        k -= y;
    }
    return status::ok;
}

// Fills out[i..n) with the smallest completion that places k into parts
// i..n-1 from layer c. Requires one to exist.
BCOMP_KERNEL void fill_smallest(const space_view& s, int i, std::int32_t k, std::int32_t c,
                                std::int32_t* out) {
    if (s.plain != 0) {
        for (int j = i; j < s.n; ++j) {
            const std::int32_t rest = s.cap_suffix[j + 1];
            const std::int32_t y = k > rest ? k - rest : 0;
            out[j] = y + s.lo[j];
            k -= y;
        }
        return;
    }
    for (int j = i; j + 1 < s.n; ++j) {
        bool placed = false;
        for (int g = 0; g < s.segs[j] && !placed; ++g) {
            const std::int32_t cn = c + seg_step(s, j, g);
            for (std::int32_t v = seg_first(s, j, g); v <= seg_last(s, j, g) && v <= k; ++v) {
                if (has_completion(s, j + 1, cn, k - v)) {
                    out[j] = v + s.lo[j];
                    k -= v;
                    c = cn;
                    placed = true;
                    break;
                }
            }
        }
    }
    out[s.n - 1] = k + s.lo[s.n - 1];
}

// Writes the first composition (rank 0).
BCOMP_KERNEL status first(const space_view& s, std::int32_t* out) {
    if (s.feasible == 0) return status::out_of_range;
    fill_smallest(s, 0, s.total, s.layer_base, out);
    return status::ok;
}

// Replaces x by its lexicographic successor. Returns out_of_range, leaving x
// unchanged, if x is the last composition. x must be in the space.
BCOMP_KERNEL status next(const space_view& s, std::int32_t* x) {
    if (s.feasible == 0) return status::out_of_range;
    if (s.plain != 0) {
        std::int32_t suffix = x[s.n - 1] - s.lo[s.n - 1];
        for (int i = s.n - 2; i >= 0; --i) {
            const std::int32_t y = x[i] - s.lo[i];
            suffix += y;
            // Increase part i by one if the parts after it can take the rest.
            const std::int32_t rest = suffix - y - 1;
            if (y < s.cap[i] && rest >= 0 && rest <= s.cap_suffix[i + 1]) {
                x[i] = y + 1 + s.lo[i];
                fill_smallest(s, i + 1, rest, 0, x);
                return status::ok;
            }
        }
        return status::out_of_range;
    }
    // The layer after every part, then back from the right: before part i
    // the layer is c and parts i..n-1 hold `suffix`.
    std::int32_t c = s.layer_base;
    for (int i = 0; i < s.n; ++i) c += step_of(s, i, x[i] - s.lo[i]);
    std::int32_t suffix = 0;
    for (int i = s.n - 1; i >= 0; --i) {
        const std::int32_t y = x[i] - s.lo[i];
        suffix += y;
        c -= step_of(s, i, y);
        if (i == s.n - 1) continue;
        // The smallest allowed value above y that leaves a completion.
        for (int g = 0; g < s.segs[i]; ++g) {
            const std::int32_t cn = c + seg_step(s, i, g);
            const std::int32_t from = seg_first(s, i, g) > y ? seg_first(s, i, g) : y + 1;
            for (std::int32_t v = from; v <= seg_last(s, i, g) && v <= suffix; ++v) {
                if (has_completion(s, i + 1, cn, suffix - v)) {
                    x[i] = v + s.lo[i];
                    fill_smallest(s, i + 1, suffix - v, cn, x);
                    return status::ok;
                }
            }
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
