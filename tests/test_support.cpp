// Support limits (number of nonzero parts, semicontinuous parts) against
// brute-force enumeration.
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

bool allowed(const composition& x, const bc::support_limits& sup) {
    std::int64_t nonzero = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] == 0) continue;
        ++nonzero;
        const std::int32_t gap = sup.min_size.empty() ? 1 : sup.min_size[i];
        if (std::abs(x[i]) < gap) return false;
    }
    return nonzero >= sup.min_count && nonzero <= sup.max_count;
}

// Every vector in the box, in lexicographic order, with its sum.
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
    std::vector<composition> in;   // the space, in rank order
    std::vector<composition> out;  // box vectors with the right sum that the limits reject
};

std::vector<case_> random_cases(int count, unsigned seed) {
    std::mt19937 rng(seed);
    auto pick = [&](int a, int b) { return std::uniform_int_distribution<int>(a, b)(rng); };
    std::vector<case_> cases;
    while (static_cast<int>(cases.size()) < count) {
        case_ c;
        const std::size_t n = static_cast<std::size_t>(pick(1, 6));
        c.lo.resize(n);
        c.hi.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            c.lo[i] = pick(0, 3) == 0 ? pick(-4, 2) : 0;
            c.hi[i] = c.lo[i] + pick(0, 7);
        }
        c.total = pick(-3, 18);
        const int ni = static_cast<int>(n);
        c.sup.min_count = pick(0, 2) == 0 ? 0 : pick(0, ni);
        c.sup.max_count = pick(0, 3) == 0 ? std::numeric_limits<std::int32_t>::max()
                                          : pick(c.sup.min_count, ni + 1);
        if (pick(0, 1) == 0) {
            c.sup.min_size.resize(n);
            for (auto& m : c.sup.min_size) m = pick(1, 4);
        }
        std::vector<composition> all;
        composition cur(n);
        box(0, c.lo, c.hi, cur, all);
        for (const composition& x : all) {
            std::int64_t sum = 0;
            for (std::int32_t v : x) sum += v;
            if (sum != c.total) continue;
            (allowed(x, c.sup) ? c.in : c.out).push_back(x);
        }
        cases.push_back(std::move(c));
    }
    return cases;
}

bool starts_with(const composition& x, const composition& prefix) {
    return std::equal(prefix.begin(), prefix.end(), x.begin());
}

}  // namespace

TEST_CASE("support limits: size, order, rank, enumeration and codec match brute force") {
    for (const case_& c : random_cases(600, 3)) {
        INFO("parts " << c.lo.size() << ", total " << c.total << ", count " << c.sup.min_count
                      << ".." << c.sup.max_count << ", sizes " << c.sup.min_size.size());
        const bc::space s(c.total, c.lo, c.hi, c.sup);
        REQUIRE(s.size() == natural(c.in.size()));
        REQUIRE(s.empty() == c.in.empty());
        composition x(c.lo.size());
        REQUIRE(s.first(x.data()) == !c.in.empty());
        for (std::size_t r = 0; r < c.in.size(); ++r) {
            REQUIRE(x == c.in[r]);  // first/next walk the list in order
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
    }
}

TEST_CASE("support limits: subtrees, children, describe and subspaces match brute force") {
    for (const case_& c : random_cases(300, 7)) {
        if (c.in.empty()) continue;
        const bc::space s(c.total, c.lo, c.hi, c.sup);
        const std::size_t n = c.lo.size();
        std::set<composition> prefixes;
        for (const composition& x : c.in) {
            for (std::size_t d = 0; d <= n; ++d)
                prefixes.insert(composition(x.begin(), x.begin() + static_cast<long>(d)));
        }
        for (const composition& prefix : prefixes) {
            std::size_t a = c.in.size(), b = 0;
            for (std::size_t r = 0; r < c.in.size(); ++r) {
                if (!starts_with(c.in[r], prefix)) continue;
                a = std::min(a, r);
                b = std::max(b, r + 1);
            }
            const bc::region t = bc::subtree(s, prefix);
            REQUIRE(t.begin() == natural(a));
            REQUIRE(t.end() == natural(b));
            // children partition the subtree in rank order
            natural at = t.begin();
            for (const bc::region& kid : bc::children(s, t)) {
                REQUIRE(kid.begin() == at);
                REQUIRE(!kid.empty());
                at = kid.end();
            }
            if (b - a > 1) REQUIRE(at == t.end());
            // describe is exact on subtrees
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
            if (prefix.size() >= n) continue;
            // the subspace is the subtree as a space, with the limits left over
            const bc::space sub = s.subspace(prefix);
            REQUIRE(sub.size() == natural(b - a));
            for (std::size_t r = a; r < b; ++r) {
                const composition tail(c.in[r].begin() + static_cast<long>(prefix.size()),
                                       c.in[r].end());
                REQUIRE(sub.unrank(natural(r - a)) == tail);
                REQUIRE(sub.rank(tail) == natural(r - a));
            }
            const bc::space same(sub.total(), sub.lower(), sub.upper(), sub.support());
            REQUIRE(same.size() == sub.size());
            if (prefix.size() + 1 < n) {
                // a subspace of a subspace
                const composition more(c.in[a].begin() + static_cast<long>(prefix.size()),
                                       c.in[a].begin() + static_cast<long>(prefix.size()) + 1);
                composition longer = prefix;
                longer.insert(longer.end(), more.begin(), more.end());
                CHECK(sub.subspace(more).size() == s.subspace(longer).size());
            }
        }
    }
}

TEST_CASE("support limits: rank sequences cover the space, iid draws are uniform") {
    const bc::support_limits sup{1, 3, {1, 2, 2, 3, 1, 2}};
    const bc::space s(14, composition{-2, 0, 0, 0, 0, -1}, composition{6, 9, 9, 9, 9, 6}, sup);
    const std::uint64_t size = s.size().to_u64();
    REQUIRE(size > 200);
    for (const bc::sequence& seq : {bc::rank_sequence(s), bc::rank_sequence(s, 5)}) {
        std::set<std::uint64_t> seen;
        std::vector<std::int32_t> pts(size * 6);
        seq.points(0, size, pts.data());
        for (std::uint64_t i = 0; i < size; ++i) {
            REQUIRE(s.contains(pts.data() + i * 6));
            seen.insert(s.rank(pts.data() + i * 6).to_u64());
        }
        CHECK(seen.size() == size);
    }
    const std::uint64_t draws = size * 200;
    const bc::sequence iid = bc::iid_sequence(s, 9);
    std::vector<double> hits(size, 0.0);
    for (std::uint64_t i = 0; i < draws; ++i) hits[iid.rank_at(i).to_u64()] += 1.0;
    double chi2 = 0.0;
    for (double h : hits) chi2 += (h - 200.0) * (h - 200.0) / 200.0;
    // Wilson-Hilferty upper quantile at p = 1e-6.
    const double df = static_cast<double>(size - 1), z = 4.753;
    const double limit = df * std::pow(1.0 - 2.0 / (9.0 * df) + z * std::sqrt(2.0 / (9.0 * df)), 3);
    INFO("chi2 = " << chi2 << ", limit = " << limit);
    CHECK(chi2 < limit);
    // Coordinate points decode to compositions of the space.
    std::vector<std::int32_t> cpts(500 * 6);
    bc::coordinate_sequence(s, 3).points(0, 500, cpts.data());
    for (std::size_t i = 0; i < 500; ++i) CHECK(s.contains(cpts.data() + i * 6));
}

namespace {

natural binomial(std::uint32_t n, std::uint32_t k) {
    natural out(1);
    for (std::uint32_t i = 0; i < k; ++i) {
        out.mul_small(n - i);
        out.divmod_small(i + 1);
    }
    return out;
}

}  // namespace

TEST_CASE("support limits on a space wider than 64 bits") {
    // 100 parts summing to 100, at most 10 nonzero, each nonzero at least 3:
    // choose the m nonzero parts, then a composition of 100 into m parts of at
    // least 3, C(100 - 2m - 1, m - 1) of them.
    bc::support_limits sup;
    sup.max_count = 10;
    sup.min_size.assign(100, 3);
    const bc::space s(100, composition(100, 0), composition(100, 100), sup);
    natural expected;
    for (std::uint32_t m = 1; m <= 10; ++m) {
        natural term = binomial(100, m);
        const natural ways = binomial(100 - 2 * m - 1, m - 1);
        natural product;
        for (std::size_t w = 0; w < ways.limb_count(); ++w) {
            // term * ways, one limb of ways at a time
            natural part = term;
            const std::uint64_t limb = ways.limbs()[w];
            natural hi = part, lo = part;
            hi.mul_small(static_cast<std::uint32_t>(limb >> 32));
            for (int b = 0; b < 32; ++b) hi.mul_small(2);
            lo.mul_small(static_cast<std::uint32_t>(limb & 0xFFFFFFFFu));
            part = hi + lo;
            for (std::size_t z = 0; z < w; ++z) {
                for (int b = 0; b < 64; ++b) part.mul_small(2);
            }
            product += part;
        }
        expected += product;
    }
    REQUIRE(s.size() == expected);
    std::mt19937_64 rng(1);
    const bc::sequence iid = bc::iid_sequence(s, 2);
    for (std::uint64_t i = 0; i < 300; ++i) {
        const natural r = iid.rank_at(i);
        const composition x = s.unrank(r);
        REQUIRE(s.contains(x));
        int nonzero = 0;
        for (std::int32_t v : x) {
            if (v != 0) {
                ++nonzero;
                REQUIRE(v >= 3);
            }
        }
        REQUIRE(nonzero <= 10);
        REQUIRE(s.rank(x) == r);
    }
    const bc::region t = bc::subtree(s, composition{0, 5});
    const bc::space sub = s.subspace(composition{0, 5});
    CHECK(sub.size() == t.size());
    CHECK(sub.support().max_count == 9);
}

TEST_CASE("support limits are validated, and the logarithmic mode refuses them") {
    const composition lo(4, 0), hi(4, 5);
    CHECK_THROWS_AS(bc::space(8, lo, hi, bc::support_limits{-1, 3, {}}), std::invalid_argument);
    CHECK_THROWS_AS(bc::space(8, lo, hi, bc::support_limits{3, 2, {}}), std::invalid_argument);
    CHECK_THROWS_AS(bc::space(8, lo, hi, bc::support_limits{0, 4, {1, 2}}), std::invalid_argument);
    CHECK_THROWS_AS(bc::space(8, lo, hi, bc::support_limits{0, 4, {1, 0, 1, 1}}),
                    std::invalid_argument);
    CHECK_THROWS_AS(bc::space(8, lo, hi, bc::support_limits{0, 2, {}}, bc::counting::logarithmic),
                    std::logic_error);
    // Limits that change nothing are fine in either mode.
    const bc::space logs(8, lo, hi, bc::support_limits{0, 4, {1, 1, 1, 1}},
                         bc::counting::logarithmic);
    CHECK(std::abs(logs.log_size() - std::log(bc::space(8, lo, hi).size().to_double())) < 1e-12);
    // No composition meets the limits: an empty space, not an error.
    const bc::space none(8, lo, hi, bc::support_limits{0, 1, {}});
    CHECK(none.empty());
    CHECK(none.size() == natural(0));
    // Sequences of spaces with different limits are not the same order.
    const bc::space a(8, lo, hi, bc::support_limits{0, 3, {}});
    const bc::space b(8, lo, hi);
    CHECK_THROWS_AS(bc::rank_sequence(a).overlap(bc::rank_sequence(b), 1), std::invalid_argument);
    CHECK(a.support().max_count == 3);
}
