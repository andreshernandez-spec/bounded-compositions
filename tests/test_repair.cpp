// repair against brute force: the nearest composition in squared distance,
// ties to the first in rank order.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "bounded_compositions/bounded_compositions.hpp"

namespace bc = bounded_compositions;
using bc::natural;
using composition = std::vector<std::int32_t>;

namespace {

struct case_ {
    std::int32_t total;
    composition lo, hi;
    bc::support_limits sup;
    std::vector<bc::group_limit> groups;
};

// Spaces with every kind of limit, drawn around a random vector of the box.
std::vector<case_> random_cases(int count, unsigned seed, bool with_limits) {
    std::mt19937 rng(seed);
    auto pick = [&](int a, int b) { return std::uniform_int_distribution<int>(a, b)(rng); };
    std::vector<case_> out;
    while (static_cast<int>(out.size()) < count) {
        case_ c;
        const std::size_t n = static_cast<std::size_t>(pick(1, 6));
        const int ni = static_cast<int>(n);
        c.lo.resize(n);
        c.hi.resize(n);
        composition x0(n);
        c.total = 0;
        int nonzero = 0;
        for (std::size_t i = 0; i < n; ++i) {
            c.lo[i] = pick(0, 3) == 0 ? pick(-3, 1) : 0;
            c.hi[i] = c.lo[i] + pick(0, 6);
            x0[i] = pick(c.lo[i], c.hi[i]);
            c.total += x0[i];
            if (x0[i] != 0) ++nonzero;
        }
        if (with_limits) {
            if (pick(0, 1) == 0) {
                c.sup.min_count = std::max(0, nonzero - pick(0, 2));
                c.sup.max_count = std::min(ni, nonzero + pick(0, 2));
            }
            if (pick(0, 2) == 0) {
                c.sup.min_size.resize(n);
                for (auto& m : c.sup.min_size) m = pick(1, 3);
            }
            const int groups = pick(0, 2);
            for (int g = 0; g < groups; ++g) {
                bc::group_limit gl;
                for (int p = 0; p < ni; ++p) {
                    if (pick(0, 1) == 0) gl.parts.push_back(p);
                }
                if (gl.parts.empty()) gl.parts.push_back(pick(0, ni - 1));
                std::int32_t sum0 = 0;
                for (std::int32_t p : gl.parts) sum0 += x0[static_cast<std::size_t>(p)];
                gl.min_sum = sum0 - pick(0, 3);
                gl.max_sum = sum0 + pick(0, 3);
                c.groups.push_back(gl);
            }
        }
        out.push_back(std::move(c));
    }
    return out;
}

double distance(const composition& x, const std::vector<double>& z) {
    double d = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) d += (x[i] - z[i]) * (x[i] - z[i]);
    return d;
}

// The nearest compositions by brute force, in rank order; equal within a
// rounding tolerance.
std::vector<composition> nearest(const bc::space& s, const std::vector<double>& z) {
    composition x(s.parts());
    std::vector<composition> best;
    double least = std::numeric_limits<double>::infinity();
    if (!s.first(x.data())) return best;
    do {
        const double d = distance(x, z);
        if (best.empty() || d < least - 1e-9 * std::max(1.0, least)) {
            least = d;
            best.assign(1, x);
        } else if (d <= least + 1e-9 * std::max(1.0, least)) {
            best.push_back(x);
        }
    } while (s.next(x.data()));
    return best;
}

std::vector<double> random_z(const case_& c, std::mt19937& rng, bool round_to_halves) {
    std::vector<double> z(c.lo.size());
    for (std::size_t i = 0; i < z.size(); ++i) {
        std::uniform_real_distribution<double> u(c.lo[i] - 3.0, c.hi[i] + 3.0);
        z[i] = u(rng);
        if (round_to_halves) z[i] = std::round(z[i] * 2.0) / 2.0;  // exact ties
    }
    return z;
}

void check(const bc::space& s, const std::vector<double>& z) {
    const std::vector<composition> best = nearest(s, z);
    const composition got = bc::repair(s, z);
    std::string what = "total " + std::to_string(s.total());
    for (std::size_t i = 0; i < z.size(); ++i) {
        what += "; part " + std::to_string(i) + ": " + std::to_string(s.lower()[i]) + ".." +
                std::to_string(s.upper()[i]) + ", z " + std::to_string(z[i]) + ", got " +
                std::to_string(got[i]) + ", best " + std::to_string(best.front()[i]);
    }
    INFO(what);
    REQUIRE(s.contains(got));
    REQUIRE(std::abs(distance(got, z) - distance(best.front(), z)) <=
            1e-9 * std::max(1.0, distance(best.front(), z)));
    REQUIRE(got == best.front());  // the first nearest in rank order
}

}  // namespace

TEST_CASE("repair finds the nearest composition, first in rank order, under bounds alone") {
    std::mt19937 rng(1);
    int checked = 0;
    for (const case_& c : random_cases(400, 2, false)) {
        const bc::space s(c.total, c.lo, c.hi);
        if (s.empty()) continue;
        for (int t = 0; t < 6; ++t) check(s, random_z(c, rng, t % 2 == 1));
        ++checked;
    }
    CHECK(checked > 300);
}

TEST_CASE("repair finds the nearest composition under support and group limits") {
    std::mt19937 rng(3);
    int checked = 0;
    for (const case_& c : random_cases(500, 4, true)) {
        const bc::space s(c.total, c.lo, c.hi, c.sup, c.groups);
        if (s.empty()) continue;
        for (int t = 0; t < 6; ++t) check(s, random_z(c, rng, t % 2 == 1));
        // In the logarithmic mode it is the same walk over the same states.
        const bc::space logs(c.total, c.lo, c.hi, c.sup, c.groups, bc::counting::logarithmic);
        const std::vector<double> z = random_z(c, rng, false);
        REQUIRE(bc::repair(logs, z) == bc::repair(s, z));
        // A subspace repairs within itself.
        if (c.lo.size() > 1) {
            const composition head{s.unrank(natural(0))[0]};
            const bc::space sub = s.subspace(head);
            check(sub, std::vector<double>(z.begin() + 1, z.end()));
        }
        ++checked;
    }
    CHECK(checked > 300);
}

TEST_CASE("repair leaves compositions of the space unchanged") {
    for (const case_& c : random_cases(150, 5, true)) {
        const bc::space s(c.total, c.lo, c.hi, c.sup, c.groups);
        composition x(c.lo.size());
        if (!s.first(x.data())) continue;
        do {
            const std::vector<double> z(x.begin(), x.end());
            REQUIRE(bc::repair(s, z) == x);
        } while (s.next(x.data()));
    }
}

TEST_CASE("the greedy repair agrees with the state search on a larger space") {
    // A group over every part with no binding limits makes the space take
    // the state search while holding the same compositions.
    const composition lo(30, 0), hi(30, 12);
    const bc::space plain(100, lo, hi);
    bc::group_limit all;
    for (int p = 0; p < 30; ++p) all.parts.push_back(p);
    const bc::space searched(100, lo, hi, bc::support_limits{}, {all});
    REQUIRE(searched.size() == plain.size());
    std::mt19937 rng(9);
    std::uniform_real_distribution<double> u(-2.0, 14.0);
    for (int t = 0; t < 40; ++t) {
        std::vector<double> z(30);
        for (double& v : z) v = t % 2 == 0 ? u(rng) : std::round(u(rng) * 2.0) / 2.0;
        CHECK(bc::repair(plain, z) == bc::repair(searched, z));
    }
}

TEST_CASE("repair ignores a common offset and a redundant limit") {
    // The review's case: [1, 1] for both, at 2 below [0, 2]; the group made
    // the state search answer [0, 2], and at 1e200 overflow in it.
    const bc::space two(2, composition{0, 0}, composition{2, 2});
    const bc::space two_grouped(2, composition{0, 0}, composition{2, 2}, bc::support_limits{},
                                {bc::group_limit{{0, 1}}});
    for (double offset : {1e12, 1e200}) {
        const std::vector<double> z{offset, offset};
        CHECK(bc::repair(two, z) == composition{1, 1});
        CHECK(bc::repair(two_grouped, z) == composition{1, 1});
    }
    // Random spaces and z = offset + w, with w on a grid of 1/8. z[i] - z[0]
    // is exact (the values share an exponent, or are small), so brute force
    // on those differences is the exact answer, ties included.
    std::mt19937 rng(11);
    int checked = 0;
    for (const case_& c : random_cases(300, 12, false)) {
        if (c.lo.size() < 2) continue;
        const bc::space s(c.total, c.lo, c.hi);
        if (s.empty()) continue;
        bc::group_limit all;
        for (std::size_t p = 0; p < c.lo.size(); ++p) all.parts.push_back(static_cast<int>(p));
        const bc::space grouped(c.total, c.lo, c.hi, bc::support_limits{}, {all});
        const std::vector<double> w = random_z(c, rng, false);
        for (double offset : {0.0, 1e6, 1e12, 1e15, 1e100, 1e200, -1e300}) {
            std::vector<double> z(w.size()), exact(w.size());
            for (std::size_t i = 0; i < w.size(); ++i) z[i] = offset + std::round(w[i] * 8) / 8;
            for (std::size_t i = 0; i < w.size(); ++i) exact[i] = z[i] - z[0];
            const composition want = nearest(s, exact).front();
            INFO("offset " << offset);
            REQUIRE(bc::repair(s, z) == want);
            REQUIRE(bc::repair(grouped, z) == want);
        }
        ++checked;
    }
    CHECK(checked > 200);
}

TEST_CASE("repair compares z spread past the range of squares") {
    // Part 0 wants its upper bound and part 3 its lower one, whatever the
    // others do; parts 1 and 2 split the rest by their own distances.
    const composition lo(4, 0), hi(4, 3);
    bc::group_limit all{{0, 1, 2, 3},
                        std::numeric_limits<std::int32_t>::min(),
                        std::numeric_limits<std::int32_t>::max()};
    const bc::space s(6, lo, hi);
    const bc::space grouped(6, lo, hi, bc::support_limits{}, {all});
    const double most = std::numeric_limits<double>::max();
    for (double far : {1e160, 1e300, most}) {
        const std::vector<double> z{far, 0.4, 1.6, -far};
        INFO("far " << far);
        CHECK(bc::repair(s, z) == composition{3, 1, 2, 0});
        CHECK(bc::repair(grouped, z) == composition{3, 1, 2, 0});
    }
}

TEST_CASE("repair rejects empty spaces and bad input") {
    const bc::space none(50, composition(3, 0), composition(3, 5));
    CHECK_THROWS_AS(bc::repair(none, std::vector<double>(3, 1.0)), std::invalid_argument);
    const bc::space s(10, composition(3, 0), composition(3, 5));
    CHECK_THROWS_AS(bc::repair(s, std::vector<double>(2, 1.0)), std::invalid_argument);
    CHECK_THROWS_AS(bc::repair(s, std::vector<double>{1.0, std::nan(""), 2.0}),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        bc::repair(s, std::vector<double>{1.0, std::numeric_limits<double>::infinity(), 2.0}),
        std::invalid_argument);
    CHECK(bc::repair(s, std::vector<double>{100.0, -40.0, 3.0}) == composition{5, 0, 5});
}
