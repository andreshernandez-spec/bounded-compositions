#include "bounded_compositions/repair.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include "bounded_compositions/kernel/walk.hpp"

namespace bounded_compositions {

namespace k = kernel;

namespace {

constexpr std::int64_t max_states = std::int64_t{1} << 27;

// z as the costs see it. Subtracting the same t from every z[i] changes
// every composition's squared distance by the same amount, since the parts
// sum to the total, and so does subtracting a constant per part, since every
// composition has one value per part. With t the median of z and part i's
// constant (q[i] - z[i] + t)^2, for q[i] the integer within its bounds
// nearest z[i] - t, the costs below keep a large common offset, or one far
// part, from swamping the differences that decide. All are scaled by a power of two, below 1 only
// when z spreads over more than about 2^960, to keep sums of them finite.
struct target {
    std::vector<double> w;  // (z[i] - t) * scale
    std::vector<std::int32_t> q;
    double scale = 1.0;

    // Part i's next unit from x: (x + 1 - z)^2 - (x - z)^2, times scale.
    double next_unit(int i, std::int32_t x) const {
        return 2.0 * (static_cast<double>(x) * scale - w[static_cast<std::size_t>(i)]) + scale;
    }
    // (x - z)^2 - (q - z)^2 = (x - q) (x + q - 2 z), times scale^2.
    double cost(int i, std::int32_t x) const {
        const std::size_t u = static_cast<std::size_t>(i);
        const std::int64_t xq = std::int64_t{x} - q[u];
        const double sum = static_cast<double>(std::int64_t{x} + q[u]) * scale;
        return static_cast<double>(xq) * scale * (sum - 2.0 * w[u]);
    }
};

target shift(const k::space_view& v, const double* z) {
    const std::size_t n = static_cast<std::size_t>(v.n);
    std::vector<double> sorted(z, z + n);
    const auto middle = sorted.begin() + static_cast<std::ptrdiff_t>(n / 2);
    std::nth_element(sorted.begin(), middle, sorted.end());
    const double t = *middle;
    double half = 0.0;  // half the largest |z[i] - t|, without overflow
    for (std::size_t i = 0; i < n; ++i) half = std::max(half, std::abs(z[i] / 2 - t / 2));
    // With half + 2^31 < 2^e, |x - q| <= 2^32 and |x + q - 2 (z - t)| < 2^(e + 2)
    // (parts are int32), so a sum of n costs is below 2^(e + e_n + 34). Scale
    // by 2^-k to keep that at most 2^1000; z - t then fits too.
    int e = 0, e_n = 0;
    std::frexp(half + 2147483648.0, &e);
    std::frexp(static_cast<double>(n), &e_n);
    const int k = std::max(0, (e + e_n + 34 - 1000 + 1) / 2);
    target out;
    out.scale = std::ldexp(1.0, -k);
    out.w.resize(n);
    out.q.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.w[i] = z[i] * out.scale - t * out.scale;
        const double lo = v.lo[i], hi = static_cast<double>(std::int64_t{v.lo[i]} + v.cap[i]);
        out.q[i] =
            static_cast<std::int32_t>(std::nearbyint(std::clamp(out.w[i] / out.scale, lo, hi)));
    }
    return out;
}

// Bounds only: from the lower bounds, add the shifted total one unit at a
// time where it costs least. Part i's next unit costs
// (x + 1 - z)^2 - (x - z)^2 = 2 (x - z) + 1, which grows with x, so the
// greedy order is optimal. Among equal costs the last part goes first, which
// keeps the earlier parts small: the first optimum in rank order.
void repair_plain(const k::space_view& v, const target& z, std::int32_t* out) {
    struct unit {
        double cost;
        int part;
    };
    auto later = [](const unit& a, const unit& b) {
        return a.cost > b.cost || (a.cost == b.cost && a.part < b.part);
    };
    std::priority_queue<unit, std::vector<unit>, decltype(later)> heap(later);
    auto next_unit = [&](int i) { return unit{z.next_unit(i, out[i]), i}; };
    for (int i = 0; i < v.n; ++i) {
        out[i] = v.lo[i];
        if (v.cap[i] > 0) heap.push(next_unit(i));
    }
    for (std::int32_t left = v.total; left > 0; --left) {
        const unit u = heap.top();
        heap.pop();
        const int i = u.part;
        ++out[i];
        if (out[i] - v.lo[i] < v.cap[i]) heap.push(next_unit(i));
    }
}

// With limits: D[i][c][k] is the least cost of parts i..n-1 from state
// (k, c), over the moves of kernel::move; D[n] is 0 at k = 0 for an allowed
// count. Then walk forward taking, part by part, the first value in rank
// order with the least cost.
void repair_states(const k::space_view& v, const target& z, std::int32_t* out) {
    const std::size_t n = static_cast<std::size_t>(v.n);
    const std::size_t width = static_cast<std::size_t>(v.total) + 1;
    std::int64_t states = 0;
    for (std::size_t r = 0; r <= n; ++r) states += std::int64_t{v.row_layers[r]};
    if (states * static_cast<std::int64_t>(width) > max_states) {
        throw std::length_error("bounded_compositions::repair: more than 2^27 states");
    }
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> D(n + 1);
    for (std::size_t r = 0; r <= n; ++r) {
        D[r].assign(static_cast<std::size_t>(v.row_layers[r]) * width, inf);
    }
    for (std::int32_t c = 0; c < v.row_layers[n]; ++c) {
        if (c >= v.count_min && c <= v.count_max) D[n][static_cast<std::size_t>(c) * width] = 0.0;
    }
    auto at = [&](std::size_t r, std::int32_t c, std::int32_t kk) {
        return D[r][static_cast<std::size_t>(c) * width + static_cast<std::size_t>(kk)];
    };
    // Each part's cost for every shifted value y it can take; no more
    // entries than one layer of D per part.
    std::vector<std::vector<double>> part_cost(n);
    for (std::size_t r = 0; r < n; ++r) {
        part_cost[r].resize(static_cast<std::size_t>(std::min(v.cap[r], v.total)) + 1);
        for (std::size_t y = 0; y < part_cost[r].size(); ++y) {
            part_cost[r][y] = z.cost(static_cast<int>(r), static_cast<std::int32_t>(y) + v.lo[r]);
        }
    }
    for (std::size_t r = n; r-- > 0;) {
        const int part = static_cast<int>(r);
        for (std::int32_t c = 0; c < v.row_layers[r]; ++c) {
            for (std::int32_t kk = 0; kk <= v.total; ++kk) {
                double best = inf;
                for (int g = 0; g < v.segs[part]; ++g) {
                    std::int32_t a = 0, b = 0, cn = 0;
                    if (!k::move(v, part, g, kk, c, &a, &b, &cn)) continue;
                    for (std::int32_t y = a; y <= b; ++y) {
                        const double rest = at(r + 1, cn, kk - y);
                        if (rest == inf) continue;
                        const double cost = part_cost[r][static_cast<std::size_t>(y)] + rest;
                        if (cost < best) best = cost;
                    }
                }
                D[r][static_cast<std::size_t>(c) * width + static_cast<std::size_t>(kk)] = best;
            }
        }
    }
    std::int32_t kk = v.total;
    std::int32_t c = v.layer_base;
    for (std::size_t r = 0; r < n; ++r) {
        const int part = static_cast<int>(r);
        double best = inf;
        std::int32_t pick = -1, pick_layer = 0;
        for (int g = 0; g < v.segs[part]; ++g) {
            std::int32_t a = 0, b = 0, cn = 0;
            if (!k::move(v, part, g, kk, c, &a, &b, &cn)) continue;
            for (std::int32_t y = a; y <= b; ++y) {
                const double rest = at(r + 1, cn, kk - y);
                if (rest == inf) continue;
                const double cost = part_cost[r][static_cast<std::size_t>(y)] + rest;
                if (cost < best) {
                    best = cost;
                    pick = y;
                    pick_layer = cn;
                }
            }
        }
        if (pick < 0) throw std::logic_error("bounded_compositions::repair: no feasible value");
        out[r] = pick + v.lo[r];
        kk -= pick;
        c = pick_layer;
    }
}

}  // namespace

void repair(const space& s, const double* z, std::int32_t* out) {
    if (s.empty()) throw std::invalid_argument("bounded_compositions::repair: the space is empty");
    for (std::size_t i = 0; i < s.parts(); ++i) {
        if (!std::isfinite(z[i])) {
            throw std::invalid_argument("bounded_compositions::repair: z[" + std::to_string(i) +
                                        "] is not finite");
        }
    }
    const k::space_view v = s.view();
    const target shifted = shift(v, z);
    if (v.plain != 0) {
        repair_plain(v, shifted, out);
    } else {
        repair_states(v, shifted, out);
    }
}

std::vector<std::int32_t> repair(const space& s, const std::vector<double>& z) {
    if (z.size() != s.parts()) {
        throw std::invalid_argument("bounded_compositions::repair: expected " +
                                    std::to_string(s.parts()) + " values, got " +
                                    std::to_string(z.size()));
    }
    std::vector<std::int32_t> out(s.parts());
    repair(s, z.data(), out.data());
    return out;
}

}  // namespace bounded_compositions
