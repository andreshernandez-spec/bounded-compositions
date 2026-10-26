#include "bounded_compositions/repair.hpp"

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

double square(double d) { return d * d; }

// Bounds only: from the lower bounds, add the shifted total one unit at a
// time where it costs least. Part i's next unit costs
// (x + 1 - z)^2 - (x - z)^2 = 2 (x - z) + 1, which grows with x, so the
// greedy order is optimal. Among equal costs the last part goes first, which
// keeps the earlier parts small: the first optimum in rank order.
void repair_plain(const k::space_view& v, const double* z, std::int32_t* out) {
    struct unit {
        double cost;
        int part;
    };
    auto later = [](const unit& a, const unit& b) {
        return a.cost > b.cost || (a.cost == b.cost && a.part < b.part);
    };
    std::priority_queue<unit, std::vector<unit>, decltype(later)> heap(later);
    for (int i = 0; i < v.n; ++i) {
        out[i] = v.lo[i];
        if (v.cap[i] > 0) heap.push({2.0 * (out[i] - z[i]) + 1.0, i});
    }
    for (std::int32_t left = v.total; left > 0; --left) {
        const unit u = heap.top();
        heap.pop();
        const int i = u.part;
        ++out[i];
        if (out[i] - v.lo[i] < v.cap[i]) heap.push({2.0 * (out[i] - z[i]) + 1.0, i});
    }
}

// With limits: D[i][c][k] is the least cost of parts i..n-1 from state
// (k, c), over the moves of kernel::move; D[n] is 0 at k = 0 for an allowed
// count. Then walk forward taking, part by part, the smallest value that
// keeps the least cost.
void repair_states(const k::space_view& v, const double* z, std::int32_t* out) {
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
                        const double cost = square(y + v.lo[r] - z[r]) + rest;
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
        const double target = at(r, c, kk);
        bool placed = false;
        for (int g = 0; g < v.segs[part] && !placed; ++g) {
            std::int32_t a = 0, b = 0, cn = 0;
            if (!k::move(v, part, g, kk, c, &a, &b, &cn)) continue;
            for (std::int32_t y = a; y <= b; ++y) {
                const double rest = at(r + 1, cn, kk - y);
                if (rest == inf || square(y + v.lo[r] - z[r]) + rest != target) continue;
                out[r] = y + v.lo[r];
                kk -= y;
                c = cn;
                placed = true;
                break;
            }
        }
        if (!placed) throw std::logic_error("bounded_compositions::repair: lost the least cost");
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
    if (v.plain != 0) {
        repair_plain(v, z, out);
    } else {
        repair_states(v, z, out);
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
