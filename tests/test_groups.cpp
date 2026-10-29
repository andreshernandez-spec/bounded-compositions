// Group limits (alone and with support limits) against brute-force enumeration.
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
#include "bounded_compositions/kernel/philox.hpp"

namespace bc = bounded_compositions;
using bc::natural;
using composition = std::vector<std::int32_t>;

namespace {

bool allowed(const composition& x, const bc::support_limits& sup,
             const std::vector<bc::group_limit>& groups) {
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
    std::vector<composition> in;   // the space, in rank order
    std::vector<composition> out;  // right sum, rejected by the limits
};

// Groups of every shape: contiguous, scattered, nested and crossing. Limits
// are drawn around a random vector of the box, so most spaces are not empty.
std::vector<case_> random_cases(int count, unsigned seed) {
    std::mt19937 rng(seed);
    auto pick = [&](int a, int b) { return std::uniform_int_distribution<int>(a, b)(rng); };
    std::vector<case_> cases;
    while (static_cast<int>(cases.size()) < count) {
        case_ c;
        const std::size_t n = static_cast<std::size_t>(pick(1, 6));
        const int ni = static_cast<int>(n);
        c.lo.resize(n);
        c.hi.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            c.lo[i] = pick(0, 3) == 0 ? pick(-3, 1) : 0;
            c.hi[i] = c.lo[i] + pick(0, 6);
        }
        composition x0(n);
        std::int32_t nonzero0 = 0;
        c.total = 0;
        for (std::size_t i = 0; i < n; ++i) {
            x0[i] = pick(c.lo[i], c.hi[i]);
            c.total += x0[i];
            if (x0[i] != 0) ++nonzero0;
        }
        if (pick(0, 2) == 0) {
            c.sup.min_count = std::max(0, nonzero0 - pick(0, 2));
            c.sup.max_count = std::min(ni, nonzero0 + pick(0, 2));
        }
        if (pick(0, 3) == 0) {
            c.sup.min_size.resize(n);
            for (auto& m : c.sup.min_size) m = pick(1, 3);
        }
        const int groups = pick(1, 3);
        for (int g = 0; g < groups; ++g) {
            bc::group_limit gl;
            if (pick(0, 1) == 0) {  // contiguous
                const int a = pick(0, ni - 1);
                const int b = pick(a, ni - 1);
                for (int p = a; p <= b; ++p) gl.parts.push_back(p);
            } else {  // any subset, in any order
                for (int p = 0; p < ni; ++p) {
                    if (pick(0, 1) == 0) gl.parts.push_back(p);
                }
                if (gl.parts.empty()) gl.parts.push_back(pick(0, ni - 1));
                std::shuffle(gl.parts.begin(), gl.parts.end(), rng);
            }
            std::int32_t sum0 = 0;
            for (std::int32_t p : gl.parts) sum0 += x0[static_cast<std::size_t>(p)];
            if (pick(0, 2) != 0) gl.min_sum = sum0 - pick(0, 4);
            if (pick(0, 2) != 0) gl.max_sum = sum0 + pick(0, 4);
            c.groups.push_back(gl);
        }
        std::vector<composition> all;
        composition cur(n);
        box(0, c.lo, c.hi, cur, all);
        for (const composition& x : all) {
            std::int64_t sum = 0;
            for (std::int32_t v : x) sum += v;
            if (sum != c.total) continue;
            (allowed(x, c.sup, c.groups) ? c.in : c.out).push_back(x);
        }
        cases.push_back(std::move(c));
    }
    return cases;
}

bool starts_with(const composition& x, const composition& prefix) {
    return std::equal(prefix.begin(), prefix.end(), x.begin());
}

void same_points(const bc::space& exact, const bc::space& logs, unsigned seed, int draws) {
    const std::size_t n = exact.parts();
    const bc::kernel::limb key[2] = {seed, 3};
    std::vector<std::uint64_t> u(n + 3);
    composition a(n), b(n);
    for (int t = 0; t < draws; ++t) {
        for (std::size_t w = 0; w + 1 < n; w += 4) {
            const bc::kernel::limb counter[4] = {static_cast<std::uint64_t>(t), 0, w / 4, 0};
            const auto block = bc::kernel::philox4x64_10(counter, key);
            for (std::size_t j = 0; j < 4; ++j) u[w + j] = block.v[j];
        }
        bc::decode(exact, u.data(), a.data());
        bc::decode(logs, u.data(), b.data());
        REQUIRE(a == b);
    }
}

}  // namespace

TEST_CASE("group limits: size, order, rank, enumeration and codec match brute force") {
    int nonempty = 0;
    for (const case_& c : random_cases(700, 5)) {
        INFO("parts " << c.lo.size() << ", total " << c.total << ", groups " << c.groups.size()
                      << ", count " << c.sup.min_count << ".." << c.sup.max_count);
        const bc::space s(c.total, c.lo, c.hi, c.sup, c.groups);
        REQUIRE(s.size() == natural(c.in.size()));
        REQUIRE(s.empty() == c.in.empty());
        composition x(c.lo.size());
        REQUIRE(s.first(x.data()) == !c.in.empty());
        for (std::size_t r = 0; r < c.in.size(); ++r) {
            REQUIRE(x == c.in[r]);
            REQUIRE(s.contains(c.in[r]));
            REQUIRE(s.rank(c.in[r]) == natural(r));
            REQUIRE(s.unrank(natural(r)) == c.in[r]);
            std::vector<std::uint64_t> u(c.lo.size());
            composition back(c.lo.size());
            bc::encode(s, c.in[r].data(), u.data());
            bc::decode(s, u.data(), back.data());
            REQUIRE(back == c.in[r]);
            REQUIRE(s.next(x.data()) == (r + 1 < c.in.size()));
        }
        for (const composition& y : c.out) REQUIRE_FALSE(s.contains(y));
        if (!c.in.empty()) ++nonempty;
    }
    CHECK(nonempty > 450);
}

TEST_CASE("group limits: subtrees, children, describe and subspaces match brute force") {
    for (const case_& c : random_cases(300, 9)) {
        if (c.in.empty()) continue;
        const bc::space s(c.total, c.lo, c.hi, c.sup, c.groups);
        const std::size_t n = c.lo.size();
        std::set<composition> prefixes;
        for (const composition& x : c.in) {
            for (std::size_t d = 0; d <= n; ++d) {
                prefixes.insert(composition(x.begin(), x.begin() + static_cast<long>(d)));
            }
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
            natural at = t.begin();
            for (const bc::region& kid : bc::children(s, t)) {
                REQUIRE(kid.begin() == at);
                REQUIRE(!kid.empty());
                at = kid.end();
            }
            if (b - a > 1) REQUIRE(at == t.end());
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
            const bc::space sub = s.subspace(prefix);
            REQUIRE(sub.size() == natural(b - a));
            for (std::size_t r = a; r < b; ++r) {
                const composition tail(c.in[r].begin() + static_cast<long>(prefix.size()),
                                       c.in[r].end());
                REQUIRE(sub.unrank(natural(r - a)) == tail);
                REQUIRE(sub.rank(tail) == natural(r - a));
            }
            // The limits left over describe the same space.
            const bc::space same(sub.total(), sub.lower(), sub.upper(), sub.support(),
                                 sub.groups());
            REQUIRE(same.size() == sub.size());
            for (std::size_t r = a; r < b; ++r) {
                const composition tail(c.in[r].begin() + static_cast<long>(prefix.size()),
                                       c.in[r].end());
                REQUIRE(same.rank(tail) == natural(r - a));
            }
        }
    }
}

TEST_CASE("group limits in the logarithmic mode match exact counting") {
    int compared = 0;
    for (const case_& c : random_cases(400, 13)) {
        const bc::space exact(c.total, c.lo, c.hi, c.sup, c.groups);
        const bc::space logs(c.total, c.lo, c.hi, c.sup, c.groups, bc::counting::logarithmic);
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
        for (const composition& y : c.out) REQUIRE_FALSE(logs.contains(y));
        same_points(exact, logs, static_cast<unsigned>(compared), 150);
        if (n > 1) {
            const composition prefix(c.in.back().begin(), c.in.back().begin() + 1);
            const bc::space es = exact.subspace(prefix);
            const bc::space ls = logs.subspace(prefix);
            REQUIRE(ls.groups() == es.groups());
            REQUIRE(std::abs(ls.log_size() - es.log_size()) <=
                    1e-11 * std::max(1.0, es.log_size()));
            same_points(es, ls, static_cast<unsigned>(compared) + 7, 50);
        }
        ++compared;
    }
    CHECK(compared > 250);
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

TEST_CASE("contiguous groups on a larger space match a convolution") {
    // 30 parts in 5 groups of 6, total 100, each group summing to at most 40:
    // the product over groups of C(t + 5, 5), summed over group totals.
    std::vector<bc::group_limit> groups;
    for (int g = 0; g < 5; ++g) {
        bc::group_limit gl;
        for (int p = 0; p < 6; ++p) gl.parts.push_back(6 * g + p);
        gl.max_sum = 40;
        groups.push_back(gl);
    }
    const bc::space s(100, composition(30, 0), composition(30, 100), bc::support_limits{}, groups);
    std::vector<natural> ways(101);  // ways[T]: groups so far summing to T
    ways[0] = natural(1);
    for (int g = 0; g < 5; ++g) {
        std::vector<natural> next(101);
        for (std::uint32_t T = 0; T <= 100; ++T) {
            for (std::uint32_t t = 0; t <= 40 && t <= T; ++t) {
                natural term = ways[T - t];
                const natural c = binomial(t + 5, 5);
                REQUIRE(c.fits_u64());
                term.mul_small(static_cast<std::uint32_t>(c.to_u64() & 0xFFFFFFFFu));
                // binomial(45, 5) < 2^32, so one small multiply is exact
                REQUIRE(c.to_u64() < (std::uint64_t{1} << 32));
                next[T] += term;
            }
        }
        ways.swap(next);
    }
    CHECK(s.size() == ways[100]);
    // Every rank round-trips and keeps the limits.
    const bc::sequence iid = bc::iid_sequence(s, 3);
    for (std::uint64_t i = 0; i < 300; ++i) {
        const natural r = iid.rank_at(i);
        const composition x = s.unrank(r);
        for (int g = 0; g < 5; ++g) {
            std::int32_t sum = 0;
            for (int p = 0; p < 6; ++p) sum += x[static_cast<std::size_t>(6 * g + p)];
            REQUIRE(sum <= 40);
        }
        REQUIRE(s.rank(x) == r);
    }
}

TEST_CASE("rank sequences cover a grouped space and iid draws are uniform") {
    const std::vector<bc::group_limit> groups = {
        {{0, 2, 4}, 3, 9}, {{1, 2}, -1, 5}, {{3, 4, 5}, 2, 8}};
    const bc::space s(15, composition{0, -2, 0, 0, 0, -1}, composition{7, 6, 7, 7, 7, 6},
                      bc::support_limits{}, groups);
    const std::uint64_t size = s.size().to_u64();
    REQUIRE(size > 100);
    for (const bc::sequence& seq : {bc::rank_sequence(s), bc::rank_sequence(s, 4)}) {
        std::set<std::uint64_t> seen;
        std::vector<std::int32_t> pts(size * 6);
        seq.points(0, size, pts.data());
        for (std::uint64_t i = 0; i < size; ++i) seen.insert(s.rank(pts.data() + i * 6).to_u64());
        CHECK(seen.size() == size);
    }
    const std::uint64_t draws = size * 200;
    const bc::sequence iid = bc::iid_sequence(s, 8);
    std::vector<double> hits(size, 0.0);
    for (std::uint64_t i = 0; i < draws; ++i) hits[iid.rank_at(i).to_u64()] += 1.0;
    double chi2 = 0.0;
    for (double h : hits) chi2 += (h - 200.0) * (h - 200.0) / 200.0;
    const double df = static_cast<double>(size - 1), z = 4.753;
    const double limit = df * std::pow(1.0 - 2.0 / (9.0 * df) + z * std::sqrt(2.0 / (9.0 * df)), 3);
    INFO("chi2 = " << chi2 << ", limit = " << limit);
    CHECK(chi2 < limit);
}

TEST_CASE("subspaces near the int32 extremes describe their compositions or are rejected") {
    const std::int64_t least = std::numeric_limits<std::int32_t>::min();
    const std::int64_t most = std::numeric_limits<std::int32_t>::max();
    std::mt19937_64 rng(77);
    auto pick = [&](std::int64_t a, std::int64_t b) {
        return std::uniform_int_distribution<std::int64_t>(a, b)(rng);
    };
    auto i32 = [](std::int64_t v) { return static_cast<std::int32_t>(v); };
    int checked = 0, rejected = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const std::size_t n = static_cast<std::size_t>(pick(2, 4));
        composition lo(n), hi(n), x(n);
        std::int64_t total = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const std::int64_t where = pick(0, 2);
            const std::int64_t a = where == 0   ? pick(least, least + 5)
                                   : where == 1 ? pick(most - 8, most - 3)
                                                : pick(-5, 5);
            lo[i] = i32(a);
            hi[i] = i32(a + pick(0, 3));
            x[i] = i32(pick(lo[i], hi[i]));
            total += x[i];
        }
        if (total < least || total > most) continue;
        // One group around x's sum over its parts: limits absent, or a little
        // off that sum and clamped to int32.
        bc::group_limit g;
        std::int64_t at = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (pick(0, 1) == 1) continue;
            g.parts.push_back(static_cast<std::int32_t>(i));
            at += x[i];
        }
        if (g.parts.empty()) continue;
        if (pick(0, 1) == 1) g.min_sum = i32(std::clamp(at - pick(0, 4), least, most));
        if (pick(0, 1) == 1) g.max_sum = i32(std::clamp(at + pick(0, 4), least, most));
        if (g.min_sum > g.max_sum) continue;
        const bc::space s(i32(total), lo, hi, bc::support_limits{}, {g});
        std::vector<composition> all;
        composition y(n);
        for (bool more = s.first(y.data()); more; more = s.next(y.data())) all.push_back(y);
        for (const composition& z : all) {
            for (std::size_t d = 1; d < n; ++d) {
                const composition prefix(z.begin(), z.begin() + static_cast<long>(d));
                bool made = false;
                try {
                    const bc::space sub = s.subspace(prefix);
                    made = true;
                    const bc::space rebuilt(sub.total(), sub.lower(), sub.upper(), sub.support(),
                                            sub.groups());
                    REQUIRE(rebuilt.size() == sub.size());
                    for (std::uint64_t r = 0; r < sub.size().to_u64(); ++r) {
                        REQUIRE(rebuilt.unrank(natural(r)) == sub.unrank(natural(r)));
                    }
                    ++checked;
                } catch (const std::invalid_argument&) {
                    REQUIRE(!made);
                    ++rejected;
                }
            }
        }
    }
    CHECK(checked > 1000);
    CHECK(rejected > 10);
}

TEST_CASE("group limits are validated and their cost is bounded") {
    const composition lo(4, 0), hi(4, 5);
    const bc::support_limits none;
    CHECK_THROWS_AS(bc::space(8, lo, hi, none, {bc::group_limit{{}, 0, 3}}), std::invalid_argument);
    CHECK_THROWS_AS(bc::space(8, lo, hi, none, {bc::group_limit{{0, 4}, 0, 3}}),
                    std::invalid_argument);
    CHECK_THROWS_AS(bc::space(8, lo, hi, none, {bc::group_limit{{1, 1}, 0, 3}}),
                    std::invalid_argument);
    CHECK_THROWS_AS(bc::space(8, lo, hi, none, {bc::group_limit{{0, 1}, 4, 3}}),
                    std::invalid_argument);
    // Limits no composition meets: an empty space.
    CHECK(bc::space(8, lo, hi, none, {bc::group_limit{{0, 1, 2, 3}, 9, 12}}).empty());
    CHECK(bc::space(8, lo, hi, none, {bc::group_limit{{0, 1}, 11, 20}}).empty());
    // Many interleaved groups on a fine grid exceed the table limit.
    std::vector<bc::group_limit> crossing;
    for (int g = 0; g < 6; ++g) crossing.push_back(bc::group_limit{{g, g + 6}, 0, 900});
    CHECK_THROWS_AS(bc::space(1000, composition(12, 0), composition(12, 1000), none, crossing),
                    std::length_error);
    // The widest total is rejected before the group arithmetic could
    // overflow int32 (the sanitizer builds check the order).
    const std::int32_t most = std::numeric_limits<std::int32_t>::max();
    CHECK_THROWS_AS(
        bc::space(most, composition{0}, composition{most}, none, {bc::group_limit{{0}}}),
        std::length_error);
    // Spaces with different groups are different orders.
    const bc::space a(8, lo, hi, none, {bc::group_limit{{0, 1}, 0, 6}});
    CHECK_THROWS_AS(bc::rank_sequence(a).overlap(bc::rank_sequence(bc::space(8, lo, hi)), 1),
                    std::invalid_argument);
    CHECK(a.groups().size() == 1);
}
