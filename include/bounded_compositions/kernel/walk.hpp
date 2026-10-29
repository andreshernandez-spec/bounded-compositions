// Rank, unrank and lexicographic enumeration over a space_view.
//
// Order is lexicographic in (x[0], ..., x[n-1]), first part slowest. With the
// shifted values y[i] = x[i] - lo[i], part i in state (k, c) moves to
// (k - v, c') for value v, and every value of a segment allowed by the limits
// leads to the same layer c' (views.hpp, move below). So the compositions
// with a smaller part i inside a segment [a, b] number
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

// Segment g of part i: allowed shifted values first..last, moving the support
// count by step (views.hpp).
BCOMP_KERNEL std::int32_t seg_first(const space_view& s, int i, int g) {
    return s.seg[9 * i + 3 * g];
}
BCOMP_KERNEL std::int32_t seg_last(const space_view& s, int i, int g) {
    return s.seg[9 * i + 3 * g + 1];
}
BCOMP_KERNEL std::int32_t seg_step(const space_view& s, int i, int g) {
    return s.seg[9 * i + 3 * g + 2];
}

// The values of segment g of part i that the limits allow from state
// (k, layer), as [*first, *last], and the layer of row i + 1 they all lead
// to. Returns false if there are none.
BCOMP_KERNEL bool move(const space_view& s, int i, int g, std::int32_t k, std::int32_t layer,
                       std::int32_t* first, std::int32_t* last, std::int32_t* next) {
    if (s.plain != 0) {  // one segment, 0..cap, one layer
        *first = 0;
        *last = seg_last(s, i, 0) < k ? seg_last(s, i, 0) : k;
        *next = 0;
        return true;
    }
    std::int32_t a = seg_first(s, i, g);
    std::int32_t b = seg_last(s, i, g) < k ? seg_last(s, i, g) : k;
    const std::int32_t count =
        (s.support_layers == 1 ? 0 : layer % s.support_layers) + seg_step(s, i, g);
    if (count >= s.support_layers) return false;
    std::int32_t to = count;
    for (int e = s.act_start[i]; e < s.act_start[i + 1]; ++e) {
        const std::int32_t* q = s.act + 4 * e;
        const std::int32_t grp = q[0];
        const std::int32_t flags = q[1];
        std::int32_t r = 0;  // the group's shifted sum before part i
        if ((flags & 2) == 0) {
            const std::int32_t* o = s.row_open + 4 * (s.row_open_start[i] + q[2]);
            const std::int32_t coord = (layer / o[3]) % o[2];
            r = o[1] != 0 ? coord - k : coord;
        }
        if (r < 0 || r > s.grp_hi[grp]) return false;  // a state no composition reaches
        if ((flags & 1) != 0) {
            if (b > s.grp_hi[grp] - r) b = s.grp_hi[grp] - r;
            if ((flags & 4) != 0 && a < s.grp_lo[grp] - r) a = s.grp_lo[grp] - r;
        }
        if ((flags & 4) == 0) {
            // After part i: r + v + (k - v) = r + k for a member, r otherwise.
            const std::int32_t* o = s.row_open + 4 * (s.row_open_start[i + 1] + q[3]);
            const std::int32_t coord = (flags & 1) != 0 ? r + k : r;
            if (coord >= o[2]) return false;
            to += coord * o[3];
        }
    }
    if (a > b) return false;
    *first = a;
    *last = b;
    *next = to;
    return true;
}

// The layer after part i takes shifted value y from state (k, layer), or -1
// if y is not allowed there.
BCOMP_KERNEL std::int32_t layer_after(const space_view& s, int i, std::int32_t k,
                                      std::int32_t layer, std::int32_t y) {
    for (int g = 0; g < s.segs[i]; ++g) {
        if (y < seg_first(s, i, g) || y > seg_last(s, i, g)) continue;
        std::int32_t a = 0, b = 0, next = 0;
        if (!move(s, i, g, k, layer, &a, &b, &next) || y < a || y > b) return -1;
        return next;
    }
    return -1;
}

// N[row][layer][k] > 0, in either counting mode.
BCOMP_KERNEL bool has_completion(const space_view& s, int row, std::int32_t layer, std::int32_t k) {
    if (k < 0 || layer < 0 || layer >= s.row_layers[row]) return false;
    if (s.log_mode != 0) return s.positive[(s.row_layer_start[row] + layer) * s.table_k + k] != 0;
    const int w = s.row_limbs[row];
    const limb* p = table_entry(s, row, layer, k);
    return k == 0 ? !limbs_is_zero(p, w) : limbs_cmp(p, w, p - w, w) != 0;
}

// N[row][layer][k] into out[0..row_limbs[row]), zero outside the table.
BCOMP_KERNEL void completions(const space_view& s, int row, std::int32_t layer, std::int32_t k,
                              limb* out) {
    const int w = s.row_limbs[row];
    if (k < 0 || layer < 0 || layer >= s.row_layers[row]) {
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

// Inside segment values [a, b] of part i, moving to layer cn: the smallest v
// with P[i+1][cn][k - v - 1] < t (true at v = b), where t = P[i+1][cn][k - a]
// minus the position inside the segment. Writes the position inside v's
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

// One halving of search_segment's search, without a branch on the comparison,
// so that several searches can run interleaved and their table reads overlap.
// The value lies in [*base, *base + *len), *len >= 1, inside a segment's
// values [a, b] with b <= k, and the predicate P[k - v - 1] < t is false below
// it and true from it on. The step probes v = *base + *len / 2 - 1 <= b - 1,
// so k - v - 1 >= 0. row points at P[i+1][cn][0], entries of wn limbs, and t
// is at most P[i+1][cn][k - *base], so wn limbs hold it. At *len == 1 the step
// reads the entry at k - *base and changes nothing.
BCOMP_KERNEL void search_step(const limb* row, int wn, std::int32_t k, const limb* t,
                              std::int32_t* base, std::int32_t* len) {
    const std::int32_t half = *len / 2;
    const limb* e = row + static_cast<std::int64_t>(k - (*base + half - 1) - 1) * wn;
    // The top limbs nearly always differ; compare the rest only when not.
    bool below = e[wn - 1] < t[wn - 1];
    if (e[wn - 1] == t[wn - 1]) below = limbs_cmp(e, wn - 1, t, wn - 1) < 0;
    *base += below ? 0 : half;
    *len -= half;
}

// The segment of part i, in state (k, c), whose values' completions hold
// position r < N[i][k][c] (w limbs), values counted in increasing order.
// Writes the segment's values [*a, *b], the layer they lead to, and
// t = P[i+1][layer][k - a] minus the position inside the segment to
// t[0..max(w, row_limbs[i + 1])). Inside a segment, value v's completions are
// N[i+1][k - v][layer], so the ones of the segment's values below v sum to
// P[i+1][layer][k - a] - P[i+1][layer][k - v]: v holds r exactly when
// P[k - v - 1] < t <= P[k - v], so the search for v needs comparisons only
// (search_segment, search_step). Returns false if r is too large.
BCOMP_KERNEL bool part_window(const space_view& s, int i, std::int32_t k, std::int32_t c,
                              const limb* r, int w, std::int32_t* a, std::int32_t* b,
                              std::int32_t* layer, limb* t) {
    const int wn = s.row_limbs[i + 1];
    const int wt = w > wn ? w : wn;
    if (s.segs[i] == 1) {
        // The only segment holds every r below N[i][k][c].
        if (!move(s, i, 0, k, c, a, b, layer)) return false;
        limbs_sub(table_entry(s, i + 1, *layer, k - *a), wn, r, w, t, wt);
        return true;
    }
    limb left[max_limbs];  // r minus the completions of the segments passed
    limbs_copy(r, w, left, wt);
    for (int g = 0; g < s.segs[i]; ++g) {
        if (!move(s, i, g, k, c, a, b, layer)) continue;
        const limb* top = table_entry(s, i + 1, *layer, k - *a);
        const limb* bottom = k - *b - 1 >= 0 ? table_entry(s, i + 1, *layer, k - *b - 1) : nullptr;
        bool inside = false;
        if (limbs_cmp(top, wn, left, wt) > 0) {
            limbs_sub(top, wn, left, wt, t, wt);
            inside = bottom == nullptr || limbs_cmp(bottom, wn, t, wt) < 0;
        }
        if (inside) return true;
        // left -= P[k - a] - P[k - b - 1]
        limbs_sub(left, wt, top, wn, left, wt);
        if (bottom != nullptr) limbs_add(left, wt, bottom, wn, left, wt);
    }
    return false;
}

// The shifted value of part i, in state (k, c), whose completions hold
// position r < N[i][k][c] (w limbs), values counted in increasing order.
// Writes the next layer to *layer and, if rest is not null, r minus the
// completions of the smaller values to rest[0..row_limbs[i + 1]). Returns -1
// if r is too large.
BCOMP_KERNEL std::int32_t part_value(const space_view& s, int i, std::int32_t k, std::int32_t c,
                                     const limb* r, int w, limb* rest, std::int32_t* layer) {
    const int wn = s.row_limbs[i + 1];
    const int wt = w > wn ? w : wn;
    limb t[max_limbs];
    std::int32_t a = 0, b = 0, cn = 0;
    *layer = -1;
    if (!part_window(s, i, k, c, r, w, &a, &b, &cn, t)) return -1;
    *layer = cn;
    return search_segment(s, i, k, cn, a, b, t, wt, rest);
}

// part_value for a plain space (one segment 0..cap, one layer). It is small
// enough to inline into the loops of unrank and decode; the call to
// part_value cost them about 10% on 100 parts.
BCOMP_KERNEL std::int32_t plain_value(const space_view& s, int i, std::int32_t k, const limb* r,
                                      int w, limb* rest) {
    const int wn = s.row_limbs[i + 1];
    const int wt = w > wn ? w : wn;
    limb t[max_limbs];
    limbs_sub(table_entry(s, i + 1, 0, k), wn, r, w, t, wt);
    const std::int32_t b = seg_last(s, i, 0) < k ? seg_last(s, i, 0) : k;
    return search_segment(s, i, k, 0, 0, b, t, wt, rest);
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
        const std::int32_t a =
            s.plain != 0 ? plain_value(s, i, k, r, w, r) : part_value(s, i, k, c, r, w, r, &c);
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
        if (s.plain != 0) {  // completions() without its checks
            const limb* hi = table_entry(s, i, 0, k);
            if (k > 0) {
                limbs_sub(hi, w, hi - w, w, count, w);
            } else {
                limbs_copy(hi, w, count, w);
            }
        } else {
            completions(s, i, c, k, count);
        }
        count[w] = limbs_mul_add_word(count, w, u[i], 0);
        const std::int32_t a = s.plain != 0 ? plain_value(s, i, k, count + 1, w, nullptr)
                                            : part_value(s, i, k, c, count + 1, w, nullptr, &c);
        if (a < 0) return status::invalid;
        out[i] = a + s.lo[i];
        k -= a;
    }
    out[s.n - 1] = k + s.lo[s.n - 1];
    return status::ok;
}

// True if x[0..n) lies within the bounds and segments, sums to the total and
// keeps the support and group limits.
BCOMP_KERNEL bool contains(const space_view& s, const std::int32_t* x) {
    if (s.feasible == 0) return false;
    if (s.plain != 0) {
        std::int64_t sum = 0;
        for (int i = 0; i < s.n; ++i) {
            const std::int64_t y = static_cast<std::int64_t>(x[i]) - s.lo[i];
            if (y < 0 || y > s.cap[i]) return false;
            sum += y;
        }
        return sum == s.total;
    }
    std::int32_t k = s.total;
    std::int32_t c = s.layer_base;
    for (int i = 0; i < s.n; ++i) {
        const std::int64_t y = static_cast<std::int64_t>(x[i]) - s.lo[i];
        if (y < 0 || y > s.cap[i] || y > k) return false;
        c = layer_after(s, i, k, c, static_cast<std::int32_t>(y));
        if (c < 0) return false;
        k -= static_cast<std::int32_t>(y);
    }
    // Row n holds the support count alone.
    return k == 0 && c >= s.count_min && c <= s.count_max;
}

// Adds to out[0..out_limbs) the completions of part i's allowed values below y,
// from state (k, c).
BCOMP_KERNEL void add_below(const space_view& s, int i, std::int32_t k, std::int32_t c,
                            std::int32_t y, limb* out, int out_limbs) {
    const int wn = s.row_limbs[i + 1];
    limb t[max_limbs];
    for (int g = 0; g < s.segs[i]; ++g) {
        if (seg_first(s, i, g) >= y) break;
        std::int32_t a = 0, b = 0, cn = 0;
        if (!move(s, i, g, k, c, &a, &b, &cn)) continue;
        if (b > y - 1) b = y - 1;
        if (a > b) continue;
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
    if (s.plain != 0) {
        // The values below y leave P[i+1][k] - P[i+1][k - y] completions.
        limb t[max_limbs];
        for (int i = 0; i + 1 < s.n; ++i) {
            const std::int32_t y = x[i] - s.lo[i];
            if (y > 0) {
                const int wn = s.row_limbs[i + 1];
                limbs_sub(table_entry(s, i + 1, 0, k), wn, table_entry(s, i + 1, 0, k - y), wn, t,
                          wn);
                limbs_add(out, out_limbs, t, wn, out, out_limbs);
            }
            k -= y;
        }
        return status::ok;
    }
    std::int32_t c = s.layer_base;
    for (int i = 0; i + 1 < s.n; ++i) {
        const std::int32_t y = x[i] - s.lo[i];
        add_below(s, i, k, c, y, out, out_limbs);
        c = layer_after(s, i, k, c, y);
        k -= y;
    }
    return status::ok;
}

// Fills out[i..n) with the smallest completion of parts i..n-1 from state
// (k, c). Requires one to exist.
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
            std::int32_t a = 0, b = 0, cn = 0;
            if (!move(s, j, g, k, c, &a, &b, &cn)) continue;
            for (std::int32_t v = a; v <= b; ++v) {
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
// unchanged, if x is the last composition. x must be in the space. scratch
// holds n values; it may be null for a plain space.
BCOMP_KERNEL status next(const space_view& s, std::int32_t* x, std::int32_t* scratch) {
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
    // The layer before every part, then back from the right: before part i
    // parts i..n-1 hold `suffix`.
    std::int32_t k = s.total;
    std::int32_t c = s.layer_base;
    for (int i = 0; i < s.n; ++i) {
        scratch[i] = c;
        const std::int32_t y = x[i] - s.lo[i];
        c = layer_after(s, i, k, c, y);
        k -= y;
    }
    std::int32_t suffix = 0;
    for (int i = s.n - 1; i >= 0; --i) {
        const std::int32_t y = x[i] - s.lo[i];
        suffix += y;
        if (i == s.n - 1) continue;
        // The smallest allowed value above y that leaves a completion.
        for (int g = 0; g < s.segs[i]; ++g) {
            std::int32_t a = 0, b = 0, cn = 0;
            if (!move(s, i, g, suffix, scratch[i], &a, &b, &cn)) continue;
            for (std::int32_t v = a > y ? a : y + 1; v <= b; ++v) {
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

// Log mode: log of N[row][lo][layer] + ... + N[row][hi][layer] (lo <= hi),
// summed from the side with less mass outside the window, P[hi] - P[lo - 1] or
// S[lo] - S[hi + 1]. Zero windows come out as log_zero exactly, since the sums
// do not change across zero entries. If side is not null, writes 0 for the
// prefix side and 1 for the suffix side.
BCOMP_KERNEL double log_window(const space_view& s, int row, std::int32_t layer, std::int32_t lo,
                               std::int32_t hi, int* side) {
    const double below = log_prefix_at(s, row, layer, lo - 1);
    const double above = log_suffix_at(s, row, layer, hi + 1);
    if (below <= above) {
        if (side != nullptr) *side = 0;
        return log_sub(log_prefix_at(s, row, layer, hi), below);
    }
    if (side != nullptr) *side = 1;
    return log_sub(log_suffix_at(s, row, layer, lo), above);
}

// u / 2^64 as a double strictly below 1.
BCOMP_KERNEL double unit_fraction(limb u) {
    const double x = static_cast<double>(u) * 0x1p-64;
    return x < 1.0 ? x : 0x1.fffffffffffffp-1;
}

// log of the completions of segment g of part i from state (k, c): the window
// [k - b, k - a] of row i + 1 in the layer the segment leads to.
BCOMP_KERNEL double log_segment(const space_view& s, int i, int g, std::int32_t k, std::int32_t c) {
    std::int32_t a = 0, b = 0, cn = 0;
    if (!move(s, i, g, k, c, &a, &b, &cn)) return log_zero;
    return log_window(s, i + 1, cn, k - b, k - a, nullptr);
}

// One part's search in decode_log. The value is the smallest v in [a, b] with
// entries[offset - v] < bound on the prefix side, where entries is the
// layer's log P row and offset k - 1, and entries[offset - v] > bound on the
// suffix side, where entries is its log S row and offset k; b when there is
// none. Either way the comparison turns true at one v and stays true above
// it, since log P grows and log S shrinks along a row, so any order of probes
// finds the same value.
struct log_search {
    std::int32_t a;
    std::int32_t b;
    std::int32_t layer;  // of row i + 1, where every value of [a, b] leads
    std::int32_t offset;
    std::int32_t prefix;  // 1 on the prefix side, 0 on the suffix side
    const double* entries;
    double bound;
};

// Part i's search in decode_log, from state (k, c) and the fraction u / 2^64.
// The part picks a segment by its share of N[i][k][c], then a value inside
// it: the segment's values [a, b] are the window [k - b, k - a] of row i + 1
// in its next layer (value v leaves k - v), searched on the side with less
// mass outside it. Returns false if no value of part i has a completion.
BCOMP_KERNEL bool log_part_window(const space_view& s, int i, std::int32_t k, std::int32_t c,
                                  limb u, log_search* q) {
    const int row = i + 1;
    // The segment whose cumulative share passes f, then f within it. A single
    // segment holds everything.
    double f = unit_fraction(u);
    int g = 0;
    if (s.segs[i] > 1) {
        double mass[3] = {log_zero, log_zero, log_zero};
        double whole = log_zero;
        for (int h = 0; h < s.segs[i]; ++h) {
            mass[h] = log_segment(s, i, h, k, c);
            whole = log_add(whole, mass[h]);
        }
        g = -1;
        int last = -1;
        for (int h = 0; h < s.segs[i]; ++h) {
            if (log_is_zero(mass[h])) continue;
            last = h;
            const double share = BCOMP_EXP(mass[h] - whole);
            if (f < share) {
                g = h;
                f /= share;
                break;
            }
            f -= share;
        }
        if (last < 0) return false;  // no completion at all
        if (g < 0) {                 // rounding past the last segment
            g = last;
            f = 0x1.fffffffffffffp-1;
        }
    }
    if (f >= 1.0) f = 0x1.fffffffffffffp-1;
    if (!(f >= 0.0)) f = 0.0;

    std::int32_t a = 0, b = 0, cn = 0;
    if (!move(s, i, g, k, c, &a, &b, &cn)) return false;
    q->a = a;
    q->b = b;
    q->layer = cn;
    const std::int64_t start = (s.row_layer_start[row] + cn) * s.log_stride;
    const double below = log_prefix_at(s, row, cn, k - b - 1);
    const double above = log_suffix_at(s, row, cn, k - a + 1);
    if (below <= above) {
        // cum(v) = P[k - a] - P[k - v - 1] > f W  <=>  P[k - v - 1] < below + (1 - f) W
        const double w = log_sub(log_prefix_at(s, row, cn, k - a), below);
        q->entries = s.log_prefix + start;
        q->offset = k - 1;
        q->prefix = 1;
        q->bound = log_add(below, BCOMP_LOG1P(-f) + w);
    } else {
        // cum(v) = S[k - v] - S[k - a + 1] > f W  <=>  S[k - v] > above + f W
        const double w = log_sub(log_suffix_at(s, row, cn, k - b), above);
        q->entries = s.log_suffix + start;
        q->offset = k;
        q->prefix = 0;
        q->bound = f > 0.0 ? log_add(above, BCOMP_LOG(f) + w) : above;
    }
    return true;
}

// The comparison of a log_search at value v, a <= v <= b - 1 (offset - v is
// then inside the row).
BCOMP_KERNEL bool log_search_below(const log_search& q, std::int32_t v) {
    // Both comparisons, then a select: lanes on different sides do not
    // mispredict a branch.
    const double x = q.entries[q.offset - v];
    const bool less = x < q.bound;
    const bool more = x > q.bound;
    return q.prefix != 0 ? less : more;
}

// One halving of a log_search without a branch on the comparison, as
// search_step: the value lies in [*base, *base + *len), *len >= 1, and the
// step probes v = *base + *len / 2 - 1 <= b - 1.
BCOMP_KERNEL void log_search_step(const log_search& q, std::int32_t* base, std::int32_t* len) {
    const std::int32_t half = *len / 2;
    *base += log_search_below(q, *base + half - 1) ? 0 : half;
    *len -= half;
}

// The value decode_log keeps for part i once its search found v. A value
// without completions is never kept: when rounding lands on one, the nearest
// value in the segment that has completions is taken, so the result is always
// valid.
BCOMP_KERNEL std::int32_t log_part_value(const space_view& s, int i, std::int32_t k,
                                         const log_search& q, std::int32_t v) {
    const int row = i + 1;
    if (s.plain != 0) {
        // Without limits the values left with a completion are those that
        // leave the later parts no more than they can hold.
        const std::int32_t rest = s.cap_suffix[row];
        return v < k - rest ? k - rest : v;
    }
    if (has_completion(s, row, q.layer, k - v)) return v;
    for (std::int32_t d = 1; d <= q.b - q.a; ++d) {
        if (v + d <= q.b && has_completion(s, row, q.layer, k - v - d)) return v + d;
        if (v - d >= q.a && has_completion(s, row, q.layer, k - v + d)) return v - d;
    }
    return v;
}

// decode for the logarithmic counting mode, part by part: log_part_window,
// the search, then log_part_value.
BCOMP_KERNEL status decode_log(const space_view& s, const limb* u, std::int32_t* out) {
    if (s.feasible == 0) return status::out_of_range;
    std::int32_t k = s.total;
    std::int32_t c = s.layer_base;
    for (int i = 0; i + 1 < s.n; ++i) {
        log_search q = {};
        if (!log_part_window(s, i, k, c, u[i], &q)) return status::invalid;
        std::int32_t lo = q.a;
        std::int32_t hi = q.b;
        while (lo < hi) {
            const std::int32_t mid = lo + (hi - lo) / 2;
            if (log_search_below(q, mid)) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo = log_part_value(s, i, k, q, lo);
        out[i] = lo + s.lo[i];
        k -= lo;
        c = q.layer;
    }
    out[s.n - 1] = k + s.lo[s.n - 1];
    return status::ok;
}

}  // namespace kernel
}  // namespace bounded_compositions

#endif
