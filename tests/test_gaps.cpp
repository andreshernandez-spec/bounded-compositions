// Gaps (values a part may not take), alone and with support and group limits,
// against brute-force enumeration.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "bounded_compositions/bounded_compositions.hpp"

namespace bc = bounded_compositions;
using bc::natural;
using composition = std::vector<std::int32_t>;

namespace {

bool allowed(const composition& x, const bc::support_limits& sup,
             const std::vector<bc::group_limit>& groups, const std::vector<bc::part_gap>& gaps) {
    std::int64_t nonzero = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] == 0) continue;
        ++nonzero;
        const std::int32_t gap = sup.min_size.empty() ? 1 : sup.min_size[i];
        if (std::abs(x[i]) < gap) return false;
    }
    if (nonzero < sup.min_count || nonzero > sup.max_count) return false;
    for (const bc::group_limit& g : groups) {
        std::int64_t sum = 0;
        for (std::int32_t p : g.parts) sum += x[static_cast<std::size_t>(p)];
        if (sum < g.min_sum || sum > g.max_sum) return false;
    }
    for (const bc::part_gap& z : gaps) {
        const std::int32_t v = x[static_cast<std::size_t>(z.part)];
        if (z.first <= v && v <= z.last) return false;
    }
    return true;
}

void box(std::size_t i, const composition& lo, const composition& hi, composition& cur,
         std::vector<composition>& out) {
    if (i == lo.size()) {
        out.push_back(cur);
        return;
    }
    for (std::int32_t v = lo[i]; v <= hi[i]; ++v) {
        cur[i] = v;
        box(i + 1, lo, hi, cur, out);
    }
}

struct case_ {
    std::int32_t total;
    composition lo, hi;
    bc::support_limits sup;
    std::vector<bc::group_limit> groups;
    std::vector<bc::part_gap> gaps;  // as given: unsorted, overlapping, past the bounds
    std::vector<composition> in;     // the space, in rank order
    std::vector<composition> out;    // box vectors with the right sum the limits reject
};

// Random small spaces with gaps, some with support or group limits too.
std::vector<case_> random_cases(int count, unsigned seed) {
    std::mt19937 rng(seed);
    auto pick = [&](int a, int b) { return std::uniform_int_distribution<int>(a, b)(rng); };
    std::vector<case_> cases;
    while (static_cast<int>(cases.size()) < count) {
        case_ c;
        const std::size_t n = static_cast<std::size_t>(pick(1, 5));
        const int ni = static_cast<int>(n);
        c.lo.resize(n);
        c.hi.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            c.lo[i] = pick(0, 3) == 0 ? pick(-4, 2) : 0;
            c.hi[i] = c.lo[i] + pick(0, 8);
        }
        c.total = pick(-3, 20);
        const int gaps = pick(0, 5);
        for (int g = 0; g < gaps; ++g) {
            const std::int32_t part = pick(0, ni - 1);
            const std::int32_t first = pick(-6, 10);
            c.gaps.push_back({part, first, first + pick(0, 3)});
        }
        if (pick(0, 2) == 0) {
            c.sup.min_count = pick(0, ni);
            c.sup.max_count = pick(c.sup.min_count, ni);
            if (pick(0, 1) == 0) {
                c.sup.min_size.resize(n);
                for (auto& m : c.sup.min_size) m = pick(1, 3);
            }
        }
        if (n > 1 && pick(0, 2) == 0) {
            bc::group_limit g;
            const int a = pick(0, ni - 2);
            g.parts = {a, a + 1};
            g.min_sum = pick(-2, 6);
            g.max_sum = g.min_sum + pick(0, 8);
            c.groups.push_back(g);
        }
        std::vector<composition> all;
        composition cur(n);
        box(0, c.lo, c.hi, cur, all);
        for (const composition& x : all) {
            std::int64_t sum = 0;
            for (std::int32_t v : x) sum += v;
            if (sum != c.total) continue;
            (allowed(x, c.sup, c.groups, c.gaps) ? c.in : c.out).push_back(x);
        }
        cases.push_back(std::move(c));
    }
    return cases;
}

bc::space make(const case_& c, bc::counting mode = bc::counting::exact) {
    return bc::space(c.total, c.lo, c.hi, c.sup, c.groups, c.gaps, mode);
}

double distance(const composition& x, const std::vector<double>& z) {
    double d = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) d += (x[i] - z[i]) * (x[i] - z[i]);
    return d;
}

}  // namespace

TEST_CASE("gaps: size order rank enumeration and codec match brute force") {
    int nonempty = 0;
    for (const case_& c : random_cases(800, 5)) {
        INFO("parts " << c.lo.size() << ", total " << c.total << ", gaps " << c.gaps.size()
                      << ", groups " << c.groups.size());
        const bc::space s = make(c);
        REQUIRE(s.size() == natural(c.in.size()));
        REQUIRE(s.empty() == c.in.empty());
        composition x(c.lo.size());
        REQUIRE(s.first(x.data()) == !c.in.empty());
        for (std::size_t r = 0; r < c.in.size(); ++r) {
            REQUIRE(x == c.in[r]);
            REQUIRE(s.contains(c.in[r]));
            REQUIRE(s.rank(c.in[r]) == natural(r));
            REQUIRE(s.unrank(natural(r)) == c.in[r]);
            std::vector<std::uint64_t> u(c.lo.size() > 1 ? c.lo.size() - 1 : 1);
            composition back(c.lo.size());
            bc::encode(s, c.in[r].data(), u.data());
            bc::decode(s, u.data(), back.data());
            REQUIRE(back == c.in[r]);
            REQUIRE(s.next(x.data()) == (r + 1 < c.in.size()));
        }
        for (const composition& y : c.out) {
            REQUIRE_FALSE(s.contains(y));
            CHECK_THROWS_AS(s.rank(y), std::invalid_argument);
        }
        if (!c.in.empty()) ++nonempty;
    }
    CHECK(nonempty > 200);
}

TEST_CASE("gaps: subtrees describe and subspaces match brute force") {
    for (const case_& c : random_cases(300, 9)) {
        if (c.in.empty()) continue;
        const bc::space s = make(c);
        const std::size_t n = c.lo.size();
        std::set<composition> prefixes;
        for (const composition& x : c.in) {
            for (std::size_t d = 1; d < n; ++d) {
                prefixes.insert(composition(x.begin(), x.begin() + static_cast<long>(d)));
            }
        }
        for (const composition& prefix : prefixes) {
            std::size_t a = c.in.size(), b = 0;
            for (std::size_t r = 0; r < c.in.size(); ++r) {
                if (!std::equal(prefix.begin(), prefix.end(), c.in[r].begin())) continue;
                a = std::min(a, r);
                b = std::max(b, r + 1);
            }
            const bc::region t = bc::subtree(s, prefix);
            REQUIRE(t.begin() == natural(a));
            REQUIRE(t.end() == natural(b));
            const std::vector<bc::part_range> ranges = bc::describe(s, t);
            for (std::size_t p = 0; p < n; ++p) {
                std::int32_t lo = c.in[a][p], hi = c.in[a][p];
                for (std::size_t r = a; r < b; ++r) {
                    lo = std::min(lo, c.in[r][p]);
                    hi = std::max(hi, c.in[r][p]);
                }
                REQUIRE(ranges[p].lo == lo);
                REQUIRE(ranges[p].hi == hi);
            }
            // The subspace's metadata, gaps included, describes its compositions.
            const bc::space sub = s.subspace(prefix);
            REQUIRE(sub.size() == natural(b - a));
            const bc::space same(sub.total(), sub.lower(), sub.upper(), sub.support(), sub.groups(),
                                 sub.gaps());
            REQUIRE(same.gaps() == sub.gaps());
            REQUIRE(same.size() == sub.size());
            for (std::size_t r = a; r < b; ++r) {
                const composition tail(c.in[r].begin() + static_cast<long>(prefix.size()),
                                       c.in[r].end());
                REQUIRE(sub.rank(tail) == natural(r - a));
                REQUIRE(same.rank(tail) == natural(r - a));
            }
        }
    }
}

TEST_CASE("gaps in the logarithmic mode match exact counting") {
    std::mt19937_64 rng(4);
    int compared = 0;
    for (const case_& c : random_cases(400, 11)) {
        const bc::space logs = make(c, bc::counting::logarithmic);
        REQUIRE(logs.empty() == c.in.empty());
        if (c.in.empty()) continue;
        const double want = std::log(static_cast<double>(c.in.size()));
        REQUIRE(std::abs(logs.log_size() - want) <= 1e-11 * std::max(1.0, want));
        const std::size_t n = c.lo.size();
        composition x(n), back(n);
        std::vector<double> u(n);
        REQUIRE(logs.first(x.data()));
        for (std::size_t r = 0; r < c.in.size(); ++r) {
            REQUIRE(x == c.in[r]);
            bc::encode(logs, x.data(), u.data());
            bc::decode(logs, u.data(), back.data());
            REQUIRE(back == x);
            REQUIRE(logs.next(x.data()) == (r + 1 < c.in.size()));
        }
        // Random codes decode to compositions of the space.
        std::vector<std::uint64_t> code(n > 1 ? n - 1 : 1);
        for (int t = 0; t < 20; ++t) {
            for (auto& v : code) v = rng();
            bc::decode(logs, code.data(), back.data());
            REQUIRE(logs.contains(back));
        }
        ++compared;
    }
    CHECK(compared > 100);
}

TEST_CASE("gaps: repair finds the nearest composition") {
    std::mt19937 rng(6);
    int checked = 0;
    for (const case_& c : random_cases(400, 13)) {
        if (c.in.empty()) continue;
        const bc::space s = make(c);
        for (int t = 0; t < 4; ++t) {
            std::vector<double> z(c.lo.size());
            for (std::size_t i = 0; i < z.size(); ++i) {
                z[i] = std::uniform_real_distribution<double>(c.lo[i] - 3.0, c.hi[i] + 3.0)(rng);
            }
            double best = std::numeric_limits<double>::infinity();
            composition first;
            for (const composition& x : c.in) {
                if (distance(x, z) < best) {
                    best = distance(x, z);
                    first = x;
                }
            }
            REQUIRE(bc::repair(s, z) == first);
        }
        ++checked;
    }
    CHECK(checked > 100);
}

TEST_CASE("gaps are normalized and validated") {
    const composition lo{-3, 0, 0}, hi{5, 6, 6};
    const bc::space s(4, lo, hi, bc::support_limits{}, {},
                      {{1, 4, 9}, {0, -9, -2}, {1, 2, 3}, {0, 0, 0}, {2, 7, 9}, {0, 1, 1}});
    // Clipped to the bounds, sorted, adjacent ones merged, the empty one dropped.
    const std::vector<bc::part_gap> want{{0, -3, -2}, {0, 0, 1}, {1, 2, 6}};
    CHECK(s.gaps() == want);
    CHECK_THROWS_AS(bc::space(4, lo, hi, bc::support_limits{}, {}, {{3, 0, 1}}),
                    std::invalid_argument);
    CHECK_THROWS_AS(bc::space(4, lo, hi, bc::support_limits{}, {}, {{-1, 0, 1}}),
                    std::invalid_argument);
    CHECK_THROWS_AS(bc::space(4, lo, hi, bc::support_limits{}, {}, {{0, 2, 1}}),
                    std::invalid_argument);
    // A gap over a whole part leaves no composition.
    CHECK(bc::space(4, lo, hi, bc::support_limits{}, {}, {{2, -1, 7}}).empty());
    // Regions of a space with other gaps belong to another space.
    const bc::space other(4, lo, hi, bc::support_limits{}, {}, {{1, 2, 5}});
    const bc::region t = bc::subtree(s, {2});
    CHECK_NOTHROW(bc::children(s, t));
    CHECK_THROWS_AS(bc::children(other, t), std::invalid_argument);
}

TEST_CASE("a part split into many segments decodes and draws within the space") {
    // Part 0 allows only multiples of 3 up to 60: 21 segments, more than the
    // log mode keeps per part while it picks one.
    std::vector<bc::part_gap> gaps;
    for (std::int32_t v = 1; v < 60; v += 3) gaps.push_back({0, v, v + 1});
    const composition lo{0, 0, 0}, hi{60, 30, 30};
    const bc::space exact(70, lo, hi, bc::support_limits{}, {}, gaps);
    const bc::space logs(70, lo, hi, bc::support_limits{}, {}, gaps, bc::counting::logarithmic);
    REQUIRE(exact.view().segs[0] == 21);
    REQUIRE(std::abs(logs.log_size() - std::log(exact.size().to_double())) <= 1e-11);
    // Uniform draws: part 0 takes each allowed value with its share of the size.
    const std::size_t draws = 60000;
    std::vector<std::int32_t> x(draws * 3);
    bc::iid_sequence(logs, 2).points(0, draws, x.data());
    std::vector<double> seen(61, 0.0);
    for (std::size_t p = 0; p < draws; ++p) {
        REQUIRE(exact.contains(x.data() + 3 * p));
        seen[static_cast<std::size_t>(x[3 * p])] += 1.0;
    }
    for (std::int32_t v = 0; v <= 60; v += 3) {
        // Compositions with x[0] = v: the other two parts sum to 70 - v.
        const std::int32_t rest = 70 - v;
        const double ways = std::max(0, std::min(30, rest) - std::max(0, rest - 30) + 1);
        const double p = ways / exact.size().to_double();
        const double sd = std::sqrt(draws * p * (1 - p));
        CHECK(std::abs(seen[static_cast<std::size_t>(v)] - draws * p) <= 5 * sd + 1);
    }
}
