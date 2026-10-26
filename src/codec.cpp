#include "bounded_compositions/codec.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "bounded_compositions/kernel/walk.hpp"
#include "counts.hpp"

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
            const double mass = k::log_segment(v, i, g, left, layer);
            whole = k::log_add(whole, mass);
            const std::int32_t a = k::seg_first(v, i, g);
            if (k::seg_last(v, i, g) < y) {
                before = k::log_add(before, mass);
            } else if (a <= y) {
                next = layer + k::seg_step(v, i, g);
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
    const k::space_view v = s.view();
    std::int32_t left = v.total;
    std::int32_t layer = v.layer_base;
    for (int i = 0; i < dims_of(s); ++i) {
        const std::int32_t y = x[i] - v.lo[i];
        const std::int32_t next = layer + k::step_of(v, i, y);
        const natural count = detail::completions_of(v, i, layer, left);
        const natural below = detail::below_of(v, i, left, layer, y);
        const natural width = detail::completions_of(v, i + 1, next, left - y);
        // Centre of the cell [below, below + width), in half units so that a
        // one-wide cell is not encoded at its edge: the smallest fraction u
        // with u * count >= below + width / 2.
        natural doubled = count;
        doubled.mul_small(2);
        natural target = below + below;
        target += width;
        std::uint64_t lo = 0, hi = std::numeric_limits<std::uint64_t>::max();
        while (lo < hi) {
            const std::uint64_t mid = lo + (hi - lo) / 2;
            if (scaled(doubled, mid) >= target) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        const natural at = scaled(count, lo);
        if (at < below || at >= below + width) {
            throw std::domain_error("bounded_compositions::encode: the cell of part " +
                                    std::to_string(i) +
                                    " is narrower than 2^-64 and cannot be represented");
        }
        u[i] = lo;
        left -= y;
        layer = next;
    }
}

void encode(const space& s, const std::int32_t* x, double* u) {
    if (s.counting_mode() == counting::logarithmic) {
        if (!s.contains(x)) {
            throw std::invalid_argument(
                "bounded_compositions::encode: not a composition of this space");
        }
        const std::vector<double> f = encode_log(s, x);
        std::copy(f.begin(), f.end(), u);
        std::vector<std::int32_t> back(s.parts());
        decode(s, u, back.data());
        check_round_trip(s, x, back, "a logarithmic space");
        return;
    }
    std::vector<std::uint64_t> fixed(static_cast<std::size_t>(dims_of(s)));
    encode(s, x, fixed.data());
    for (std::size_t i = 0; i < fixed.size(); ++i) {
        double d = std::ldexp(static_cast<double>(fixed[i]), -64);
        if (d >= 1.0) d = std::nextafter(1.0, 0.0);
        u[i] = d;
    }
    // Rounding to 53 bits can leave a narrow cell; check the round trip.
    std::vector<std::int32_t> back(s.parts());
    decode(s, u, back.data());
    for (std::size_t i = 0; i < back.size(); ++i) {
        if (back[i] != x[i]) {
            throw std::domain_error(
                "bounded_compositions::encode: the composition's cell is narrower than double "
                "precision can represent; use the 64-bit encode");
        }
    }
}

}  // namespace bounded_compositions
