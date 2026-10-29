#include "bounded_compositions/codec.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "bounded_compositions/kernel/walk.hpp"
#include "counts.hpp"
#include "walk_batch.hpp"

namespace bounded_compositions {

namespace k = kernel;

namespace {

int dims_of(const space& s) { return static_cast<int>(s.parts()) - 1; }

// floor(n * u / 2^64).
natural scaled(const natural& n, std::uint64_t u) {
    std::vector<k::limb> wide(n.limb_count() + 1, 0);
    for (std::size_t i = 0; i < n.limb_count(); ++i) wide[i] = n.limbs()[i];
    wide.back() = k::limbs_mul_add_word(wide.data(), static_cast<int>(n.limb_count()), u, 0);
    return natural::from_limbs(wide.data() + 1, n.limb_count());
}

// The smallest u with floor(n * u / 2^64) >= target, if there is one.
bool first_code(const natural& n, const natural& target, std::uint64_t& u) {
    std::uint64_t lo = 0, hi = std::numeric_limits<std::uint64_t>::max();
    if (scaled(n, hi) < target) return false;
    while (lo < hi) {
        const std::uint64_t mid = lo + (hi - lo) / 2;
        if (scaled(n, mid) >= target) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    u = lo;
    return true;
}

// Part i's cell in exact mode: decode reads u[i] as position
// floor(count * u[i] / 2^64) among the count completions left, and the
// positions [below, end) give x[i]. Codes that decode into a cell are a range,
// since that position grows with u.
struct cell {
    natural count;
    natural below;
    natural end;

    bool holds(std::uint64_t u) const {
        const natural at = scaled(count, u);
        return below <= at && at < end;
    }
};

std::vector<cell> cells_of(const space& s, const std::int32_t* x) {
    const k::space_view v = s.view();
    std::vector<cell> out;
    std::int32_t left = v.total;
    std::int32_t layer = v.layer_base;
    for (int i = 0; i < dims_of(s); ++i) {
        const std::int32_t y = x[i] - v.lo[i];
        const std::int32_t next = k::layer_after(v, i, left, layer, y);
        cell c{detail::completions_of(v, i, layer, left), detail::below_of(v, i, left, layer, y),
               natural()};
        c.end = c.below + detail::completions_of(v, i + 1, next, left - y);
        out.push_back(std::move(c));
        left -= y;
        layer = next;
    }
    return out;
}

// The centre of the cell, in half units so that a one-wide cell is not
// encoded at its edge: the smallest u with u * count >= (below + end) / 2.
// A cell narrower than 2^-64 of the count can miss its centre and still hold
// codes; it then gets the middle one.
std::uint64_t code_of(const cell& c, int i) {
    natural doubled = c.count;
    doubled.mul_small(2);
    std::uint64_t u = 0;
    if (first_code(doubled, c.below + c.end, u) && c.holds(u)) return u;
    std::uint64_t first = 0, after = 0;
    if (!first_code(c.count, c.below, first) || !c.holds(first)) {
        throw std::domain_error(
            "bounded_compositions::encode: no 64-bit code decodes to the "
            "value of part " +
            std::to_string(i));
    }
    const std::uint64_t last =
        first_code(c.count, c.end, after) ? after - 1 : std::numeric_limits<std::uint64_t>::max();
    return first + (last - first) / 2;
}

double to_unit(std::uint64_t u) {
    const double d = std::ldexp(static_cast<double>(u), -64);
    return d < 1.0 ? d : std::nextafter(1.0, 0.0);
}

std::uint64_t to_fraction(double u, int i) {
    if (!(u >= 0.0 && u < 1.0)) {  // also rejects NaN
        throw std::out_of_range("bounded_compositions::decode: u[" + std::to_string(i) +
                                "] is not in [0, 1)");
    }
    return static_cast<std::uint64_t>(std::ldexp(u, 64));
}

// Log mode: centre of each part's cell as a double, from the same two-sided
// window sums decode uses: the segments wholly below the value, the part of
// its own segment below it, and half its own completions.
std::vector<double> encode_log(const space& s, const std::int32_t* x) {
    const k::space_view v = s.view();
    std::vector<double> u(static_cast<std::size_t>(dims_of(s)));
    std::int32_t left = v.total;
    std::int32_t layer = v.layer_base;
    for (int i = 0; i < dims_of(s); ++i) {
        const int row = i + 1;
        const std::int32_t y = x[i] - v.lo[i];
        double whole = k::log_zero;
        double before = k::log_zero;
        std::int32_t next = layer;
        for (int g = 0; g < v.segs[i]; ++g) {
            std::int32_t a = 0, b = 0, cn = 0;
            if (!k::move(v, i, g, left, layer, &a, &b, &cn)) continue;
            const double mass = k::log_window(v, row, cn, left - b, left - a, nullptr);
            whole = k::log_add(whole, mass);
            if (b < y) {
                before = k::log_add(before, mass);
            } else if (a <= y) {
                next = cn;
                if (y > a) {
                    before = k::log_add(
                        before, k::log_window(v, row, next, left - y + 1, left - a, nullptr));
                }
            }
        }
        const double width = k::log_window(v, row, next, left - y, left - y, nullptr);
        const double centre = k::log_add(before, width - std::log(2.0));
        double f = std::exp(centre - whole);
        if (!(f >= 0.0)) f = 0.0;
        if (f >= 1.0) f = std::nextafter(1.0, 0.0);
        u[static_cast<std::size_t>(i)] = f;
        left -= y;
        layer = next;
    }
    return u;
}

void check_round_trip(const space& s, const std::int32_t* x, const std::vector<std::int32_t>& back,
                      const char* precision) {
    for (std::size_t i = 0; i < back.size(); ++i) {
        if (back[i] != x[i]) {
            throw std::domain_error(
                std::string("bounded_compositions::encode: the composition's cell "
                            "is narrower than ") +
                precision + " can represent");
        }
    }
    (void)s;
}

}  // namespace

void decode(const space& s, const std::uint64_t* u, std::int32_t* out) {
    if (k::decode(s.view(), u, out) != k::status::ok) {
        throw std::out_of_range("bounded_compositions::decode: the space is empty");
    }
}

void decode(const space& s, const double* u, std::int32_t* out) {
    std::vector<std::uint64_t> fixed(static_cast<std::size_t>(dims_of(s)));
    for (int i = 0; i < dims_of(s); ++i) fixed[static_cast<std::size_t>(i)] = to_fraction(u[i], i);
    decode(s, fixed.data(), out);
}

void decode(const space& s, const std::uint64_t* u, std::size_t points, std::int32_t* out) {
    const std::size_t dims = s.parts() - 1;
    if (detail::decode_points(s.view(), u, dims, points, out, s.parts()) != k::status::ok) {
        throw std::out_of_range("bounded_compositions::decode: the space is empty");
    }
}

void decode(const space& s, const double* u, std::size_t points, std::int32_t* out) {
    // Converted in chunks, so the copy stays small whatever the input.
    constexpr std::size_t chunk = 256;
    const std::size_t dims = s.parts() - 1;
    std::vector<std::uint64_t> fixed(std::min(chunk, points) * dims);
    for (std::size_t p = 0; p < points; p += chunk) {
        const std::size_t m = std::min(chunk, points - p);
        for (std::size_t e = 0; e < m * dims; ++e) {
            fixed[e] = to_fraction(u[p * dims + e], static_cast<int>(e % dims));
        }
        decode(s, fixed.data(), m, out + p * s.parts());
    }
}

void encode(const space& s, const std::int32_t* x, std::uint64_t* u) {
    if (!s.contains(x)) {
        throw std::invalid_argument(
            "bounded_compositions::encode: not a composition of this space");
    }
    if (s.counting_mode() == counting::logarithmic) {
        const std::vector<double> f = encode_log(s, x);
        for (std::size_t i = 0; i < f.size(); ++i) {
            u[i] = static_cast<std::uint64_t>(std::ldexp(f[i], 64));
        }
        std::vector<std::int32_t> back(s.parts());
        decode(s, u, back.data());
        check_round_trip(s, x, back, "a logarithmic space");
        return;
    }
    const std::vector<cell> cells = cells_of(s, x);
    for (int i = 0; i < dims_of(s); ++i) u[i] = code_of(cells[static_cast<std::size_t>(i)], i);
}

void encode(const space& s, const std::int32_t* x, double* u) {
    if (!s.contains(x)) {
        throw std::invalid_argument(
            "bounded_compositions::encode: not a composition of this space");
    }
    if (s.counting_mode() == counting::logarithmic) {
        const std::vector<double> f = encode_log(s, x);
        std::copy(f.begin(), f.end(), u);
        std::vector<std::int32_t> back(s.parts());
        decode(s, u, back.data());
        check_round_trip(s, x, back, "a logarithmic space");
        return;
    }
    // The 64-bit code rounded to a double, which a cell narrower than the
    // doubles near it can lose; then the smallest double at or above the
    // cell's first code, if that is still in the cell.
    const std::vector<cell> cells = cells_of(s, x);
    for (int i = 0; i < dims_of(s); ++i) {
        const cell& c = cells[static_cast<std::size_t>(i)];
        double d = to_unit(code_of(c, i));
        std::uint64_t first = 0;
        if (!c.holds(to_fraction(d, i)) && first_code(c.count, c.below, first)) {
            d = to_unit(first);
            if (to_fraction(d, i) < first) d = std::nextafter(d, 1.0);
        }
        if (d >= 1.0 || !c.holds(to_fraction(d, i))) {
            throw std::domain_error(
                "bounded_compositions::encode: no double decodes to the value "
                "of part " +
                std::to_string(i) + "; use the 64-bit encode");
        }
        u[i] = d;
    }
}

}  // namespace bounded_compositions
