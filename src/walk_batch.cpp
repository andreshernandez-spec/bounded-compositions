#include "walk_batch.hpp"

#include <algorithm>

#include "bounded_compositions/kernel/limbs.hpp"
#include "bounded_compositions/kernel/walk.hpp"

namespace bounded_compositions {
namespace detail {

namespace k = kernel;

namespace {

// Points walked together: enough reads in flight to hide memory latency,
// with each point's scratch (two limb arrays) on the stack.
constexpr std::size_t lanes = 16;

// Walks up to `lanes` points part by part, each part's searches in lockstep.
// For unrank, ranks is not null and each point's position carries over from
// part to part as kernel::unrank's remainder; for decode, the position at
// part i is floor(N[i][k][c] * u[i] / 2^64), as in kernel::decode.
k::status walk_lanes(const k::space_view& s, const k::limb* ranks, int rank_limbs, const k::limb* u,
                     std::size_t u_stride, std::size_t count, std::int32_t* out,
                     std::size_t out_stride) {
    // Positions take one limb more for decode's product.
    k::limb pos[lanes][k::max_limbs + 1];
    k::limb t[lanes][k::max_limbs];
    std::int32_t left[lanes], layer[lanes], base[lanes], len[lanes];
    const k::limb* row[lanes];
    const std::size_t rank_width = static_cast<std::size_t>(rank_limbs);
    int w = s.row_limbs[0];
    for (std::size_t p = 0; p < count; ++p) {
        if (ranks != nullptr) k::limbs_copy(ranks + p * rank_width, rank_limbs, pos[p], w);
        left[p] = s.total;
        layer[p] = s.layer_base;
    }
    for (int i = 0; i + 1 < s.n; ++i) {
        const std::size_t col = static_cast<std::size_t>(i);
        const int wn = s.row_limbs[i + 1];
        const int wt = std::max(w, wn);
        std::int32_t most = 1;
        for (std::size_t p = 0; p < count; ++p) {
            const k::limb* r = pos[p];
            if (ranks == nullptr) {
                k::completions(s, i, layer[p], left[p], pos[p]);
                pos[p][w] = k::limbs_mul_add_word(pos[p], w, u[p * u_stride + col], 0);
                r = pos[p] + 1;
            }
            std::int32_t a = 0, b = 0;
            if (!k::part_window(s, i, left[p], layer[p], r, w, &a, &b, &layer[p], t[p])) {
                return k::status::invalid;
            }
            base[p] = a;
            len[p] = b - a + 1;
            row[p] = k::table_entry(s, i + 1, layer[p], 0);
            most = std::max(most, len[p]);
        }
        // Every search halves its range (rounding up) per step, so the
        // longest decides when all are done.
        while (most > 1) {
            for (std::size_t p = 0; p < count; ++p) {
                k::search_step(row[p], wn, left[p], t[p], &base[p], &len[p]);
            }
            most -= most / 2;
        }
        for (std::size_t p = 0; p < count; ++p) {
            if (ranks != nullptr) {
                const k::limb* hit = row[p] + static_cast<std::int64_t>(left[p] - base[p]) * wn;
                k::limbs_sub(hit, wn, t[p], wt, pos[p], wn);
            }
            out[p * out_stride + col] = base[p] + s.lo[i];
            left[p] -= base[p];
        }
        w = wn;
    }
    for (std::size_t p = 0; p < count; ++p) {
        out[p * out_stride + static_cast<std::size_t>(s.n - 1)] = left[p] + s.lo[s.n - 1];
    }
    return k::status::ok;
}

// kernel::decode_log for up to `lanes` points, each part's searches in
// lockstep.
k::status walk_log_lanes(const k::space_view& s, const k::limb* u, std::size_t u_stride,
                         std::size_t count, std::int32_t* out, std::size_t out_stride) {
    k::log_search q[lanes];
    std::int32_t left[lanes], layer[lanes], base[lanes], len[lanes];
    for (std::size_t p = 0; p < count; ++p) {
        left[p] = s.total;
        layer[p] = s.layer_base;
    }
    for (int i = 0; i + 1 < s.n; ++i) {
        const std::size_t col = static_cast<std::size_t>(i);
        std::int32_t most = 1;
        for (std::size_t p = 0; p < count; ++p) {
            if (!k::log_part_window(s, i, left[p], layer[p], u[p * u_stride + col], &q[p])) {
                return k::status::invalid;
            }
            base[p] = q[p].a;
            len[p] = q[p].b - q[p].a + 1;
            most = std::max(most, len[p]);
        }
        while (most > 1) {
            for (std::size_t p = 0; p < count; ++p) k::log_search_step(q[p], &base[p], &len[p]);
            most -= most / 2;
        }
        for (std::size_t p = 0; p < count; ++p) {
            const std::int32_t v = k::log_part_value(s, i, left[p], q[p], base[p]);
            out[p * out_stride + col] = v + s.lo[i];
            left[p] -= v;
            layer[p] = q[p].layer;
        }
    }
    for (std::size_t p = 0; p < count; ++p) {
        out[p * out_stride + static_cast<std::size_t>(s.n - 1)] = left[p] + s.lo[s.n - 1];
    }
    return k::status::ok;
}

}  // namespace

k::status unrank_points(const k::space_view& s, const k::limb* ranks, int rank_limbs,
                        std::size_t count, std::int32_t* out, std::size_t out_stride) {
    const std::size_t rank_width = static_cast<std::size_t>(rank_limbs);
    for (std::size_t p = 0; p < count; ++p) {
        if (s.size_limbs == 0 ||
            k::limbs_cmp(ranks + p * rank_width, rank_limbs, s.size, s.size_limbs) >= 0) {
            return k::status::out_of_range;
        }
    }
    for (std::size_t p = 0; p < count; p += lanes) {
        const k::status st =
            walk_lanes(s, ranks + p * rank_width, rank_limbs, nullptr, 0,
                       std::min(lanes, count - p), out + p * out_stride, out_stride);
        if (st != k::status::ok) return st;
    }
    return k::status::ok;
}

k::status decode_points(const k::space_view& s, const k::limb* u, std::size_t u_stride,
                        std::size_t count, std::int32_t* out, std::size_t out_stride) {
    const bool log_mode = s.log_mode != 0;
    if (count == 0) return k::status::ok;  // as decoding no point one by one
    if (log_mode ? s.feasible == 0 : s.size_limbs == 0) return k::status::out_of_range;
    for (std::size_t p = 0; p < count; p += lanes) {
        const std::size_t m = std::min(lanes, count - p);
        const k::status st = log_mode ? walk_log_lanes(s, u + p * u_stride, u_stride, m,
                                                       out + p * out_stride, out_stride)
                                      : walk_lanes(s, nullptr, 0, u + p * u_stride, u_stride, m,
                                                   out + p * out_stride, out_stride);
        if (st != k::status::ok) return st;
    }
    return k::status::ok;
}

}  // namespace detail
}  // namespace bounded_compositions
