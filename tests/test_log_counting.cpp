#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "bounded_compositions/bounded_compositions.hpp"
#include "bounded_compositions/kernel/philox.hpp"

namespace bc = bounded_compositions;
using composition = std::vector<std::int32_t>;

namespace {

struct spec {
    std::int32_t total;
    composition lo, hi;
};

// Random spaces, half of them with the total close to the sum of the upper
// bounds, where the counts of each row decrease and one-sided log sums cancel.
std::vector<spec> random_specs(int count, unsigned seed, int max_parts, int max_cap) {
    std::mt19937 rng(seed);
    auto pick = [&](int a, int b) { return std::uniform_int_distribution<int>(a, b)(rng); };
    std::vector<spec> out;
    while (static_cast<int>(out.size()) < count) {
        spec s;
        const std::size_t n = static_cast<std::size_t>(pick(2, max_parts));
        s.lo.resize(n);
        s.hi.resize(n);
        std::int64_t lo_sum = 0, hi_sum = 0;
        for (std::size_t i = 0; i < n; ++i) {
            s.lo[i] = pick(-3, 3);
            s.hi[i] = s.lo[i] + pick(1, max_cap);
            lo_sum += s.lo[i];
            hi_sum += s.hi[i];
        }
        const bool near_top = out.size() % 2 == 1;
        const std::int64_t total =
            near_top ? hi_sum - pick(0, 5) : lo_sum + (hi_sum - lo_sum) * pick(1, 9) / 10;
        s.total = static_cast<std::int32_t>(total);
        out.push_back(s);
    }
    return out;
}

void same_points(const bc::space& exact, const bc::space& logs, unsigned seed, int draws) {
    const std::size_t n = exact.parts();
    const bounded_compositions::kernel::limb key[2] = {seed, 1};
    std::vector<std::uint64_t> u(n + 3);
    composition a(n), b(n);
    for (int t = 0; t < draws; ++t) {
        for (std::size_t w = 0; w + 1 < n; w += 4) {
            const bounded_compositions::kernel::limb counter[4] = {static_cast<std::uint64_t>(t), 0,
                                                                   w / 4, 0};
            const auto block = bounded_compositions::kernel::philox4x64_10(counter, key);
            for (std::size_t j = 0; j < 4; ++j) u[w + j] = block.v[j];
        }
        bc::decode(exact, u.data(), a.data());
        bc::decode(logs, u.data(), b.data());
        REQUIRE(a == b);
    }
}

}  // namespace

TEST_CASE("logarithmic sizes match exact ones") {
    for (const spec& p : random_specs(200, 4, 40, 40)) {
        const bc::space exact(p.total, p.lo, p.hi);
        const bc::space logs(p.total, p.lo, p.hi, bc::counting::logarithmic);
        REQUIRE(logs.counting_mode() == bc::counting::logarithmic);
        REQUIRE(exact.empty() == logs.empty());
        if (exact.empty()) {
            CHECK(std::isinf(logs.log_size()));
            continue;
        }
        const double want = exact.log_size();
        INFO("log size " << want);
        CHECK(std::abs(logs.log_size() - want) <= 1e-11 * std::max(1.0, std::abs(want)));
    }
}

TEST_CASE("logarithmic decode picks the same composition as exact decode") {
    int compared = 0;
    for (const spec& p : random_specs(120, 5, 40, 30)) {
        const bc::space exact(p.total, p.lo, p.hi);
        if (exact.empty()) continue;
        const bc::space logs(p.total, p.lo, p.hi, bc::counting::logarithmic);
        same_points(exact, logs, static_cast<unsigned>(compared), 400);
        ++compared;
    }
    CHECK(compared > 100);
}

TEST_CASE("coordinate sequences agree between the counting modes") {
    const bc::space exact(590, composition(30, 0), composition(30, 20));  // near the cap sum
    const bc::space logs(590, composition(30, 0), composition(30, 20), bc::counting::logarithmic);
    const std::uint64_t count = std::uint64_t{1} << 14;
    std::vector<std::int32_t> pa(count * 30), pb(count * 30);
    // Scrambled points are never exactly on a cell boundary: identical.
    for (std::uint64_t seed : {17u, 18u}) {
        bc::coordinate_sequence(exact, seed).points(0, count, pa.data());
        bc::coordinate_sequence(logs, seed).points(0, count, pb.data());
        CHECK(pa == pb);
    }
    // Unscrambled points are dyadic fractions, some exactly on a boundary
    // that log counts cannot place; those few land in a neighbouring cell.
    bc::coordinate_sequence(exact).points(0, count, pa.data());
    bc::coordinate_sequence(logs).points(0, count, pb.data());
    std::uint64_t differ = 0;
    for (std::uint64_t i = 0; i < count; ++i) {
        REQUIRE(logs.contains(pb.data() + i * 30));
        if (!std::equal(pa.begin() + static_cast<long>(i * 30),
                        pa.begin() + static_cast<long>(i * 30 + 30),
                        pb.begin() + static_cast<long>(i * 30))) {
            ++differ;
        }
    }
    INFO(differ << " of " << count << " unscrambled points differ");
    CHECK(differ <= count / 1000);
}

TEST_CASE("iid sequences on a logarithmic space are uniform") {
    const bc::space exact(6, composition(4, 0), composition(4, 6));  // 84 compositions
    const bc::space logs(6, composition(4, 0), composition(4, 6), bc::counting::logarithmic);
    const bc::sequence seq = bc::iid_sequence(logs, 2026);
    const std::uint64_t draws = 84 * 500;
    std::vector<std::int32_t> pts(draws * 4);
    seq.points(0, draws, pts.data());
    std::vector<double> hits(84, 0.0);
    for (std::uint64_t i = 0; i < draws; ++i) hits[exact.rank(pts.data() + i * 4).to_u64()] += 1.0;
    double chi2 = 0.0;
    for (double h : hits) chi2 += (h - 500.0) * (h - 500.0) / 500.0;
    const double df = 83.0, z = 4.753;
    const double limit = df * std::pow(1.0 - 2.0 / (9.0 * df) + z * std::sqrt(2.0 / (9.0 * df)), 3);
    INFO("chi2 = " << chi2 << ", limit = " << limit);
    CHECK(chi2 < limit);
}

TEST_CASE("a space too large for exact counts works in logarithms") {
    // C(4199, 1999) needs about 4190 bits, more than the 4096-bit limit.
    const composition lo(2000, 0), hi(2000, 2200);
    CHECK_THROWS_AS(bc::space(2200, lo, hi), std::length_error);
    const bc::space logs(2200, lo, hi, bc::counting::logarithmic);
    const double want = std::lgamma(4200.0) - std::lgamma(2201.0) - std::lgamma(2000.0);
    CHECK(std::abs(logs.log_size() - want) <= 1e-11 * want);
    std::vector<std::int32_t> pts(64 * 2000);
    for (const bc::sequence& seq : {bc::coordinate_sequence(logs, 3), bc::iid_sequence(logs, 3)}) {
        seq.points(0, 64, pts.data());
        for (std::size_t i = 0; i < 64; ++i) REQUIRE(logs.contains(pts.data() + i * 2000));
    }
    std::vector<double> u(1999);
    composition back(2000);
    bc::encode(logs, pts.data(), u.data());
    bc::decode(logs, u.data(), back.data());
    CHECK(std::equal(back.begin(), back.end(), pts.begin()));
}

TEST_CASE("exact-only operations refuse a logarithmic space") {
    const bc::space logs(10, composition{0, 0, 0}, composition{5, 5, 5}, bc::counting::logarithmic);
    composition x{5, 5, 0};
    CHECK_THROWS_AS(logs.size(), std::logic_error);
    CHECK_THROWS_AS(logs.rank(x), std::logic_error);
    CHECK_THROWS_AS(logs.unrank(bc::natural(0)), std::logic_error);
    CHECK_THROWS_AS(bc::whole(logs), std::logic_error);
    CHECK_THROWS_AS(bc::subtree(logs, composition{5}), std::logic_error);
    CHECK_THROWS_AS(bc::rank_sequence(logs), std::logic_error);
    CHECK_THROWS_AS(bc::rank_sequence(logs, 1), std::logic_error);
    CHECK_THROWS_AS(bc::scaled_rank_sequence(logs), std::logic_error);
    // What does work: membership and enumeration, like the exact space.
    const bc::space exact(10, composition{0, 0, 0}, composition{5, 5, 5});
    composition a(3), b(3);
    REQUIRE(exact.first(a.data()));
    REQUIRE(logs.first(b.data()));
    do {
        REQUIRE(a == b);
        REQUIRE(logs.contains(b));
        const bool more_a = exact.next(a.data());
        REQUIRE(more_a == logs.next(b.data()));
        if (!more_a) break;
    } while (true);
    // An empty logarithmic space.
    const bc::space none(20, composition{0, 0, 0}, composition{5, 5, 5}, bc::counting::logarithmic);
    CHECK(none.empty());
    CHECK(std::isinf(none.log_size()));
    CHECK_FALSE(none.first(a.data()));
    CHECK(bc::coordinate_sequence(none).capacity() == 0);
    const std::uint64_t u[2] = {0, 0};
    CHECK_THROWS_AS(bc::decode(none, u, a.data()), std::out_of_range);
}

TEST_CASE("logarithmic subspaces match exact subspaces") {
    const bc::space exact(300, composition(25, 0), composition(25, 15));
    const bc::space logs(300, composition(25, 0), composition(25, 15), bc::counting::logarithmic);
    for (const composition& prefix :
         {composition{15}, composition{3, 15, 0}, composition{15, 15, 15, 15}}) {
        const bc::space es = exact.subspace(prefix);
        const bc::space ls = logs.subspace(prefix);
        CHECK(ls.counting_mode() == bc::counting::logarithmic);
        CHECK(std::abs(ls.log_size() - es.log_size()) <= 1e-11 * std::max(1.0, es.log_size()));
        same_points(es, ls, 9, 300);
    }
}

TEST_CASE("logarithmic encode inverts decode") {
    for (const spec& p : random_specs(60, 6, 6, 8)) {
        const bc::space logs(p.total, p.lo, p.hi, bc::counting::logarithmic);
        if (logs.empty()) continue;
        const std::size_t n = logs.parts();
        composition x(n), back(n);
        std::vector<double> u(n);
        std::vector<std::uint64_t> fixed(n);
        REQUIRE(logs.first(x.data()));
        do {
            bc::encode(logs, x.data(), u.data());
            bc::decode(logs, u.data(), back.data());
            REQUIRE(back == x);
            bc::encode(logs, x.data(), fixed.data());
            bc::decode(logs, fixed.data(), back.data());
            REQUIRE(back == x);
        } while (logs.next(x.data()));
    }
}
