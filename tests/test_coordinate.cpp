#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

#include "bounded_compositions/bounded_compositions.hpp"
#include "bounded_compositions/kernel/philox.hpp"
#include "sobol.hpp"

namespace bc = bounded_compositions;
namespace k = bounded_compositions::kernel;
using bc::natural;
using composition = std::vector<std::int32_t>;

namespace {

struct sobol_case {
    int dim;
    std::uint64_t index;
    std::uint32_t top30;
};

const sobol_case sobol_cases[] = {
#include "data/sobol_vectors.inc"
};

// Every power-of-two prefix (up to 2^max_k points) of every coordinate puts
// exactly one point in each interval [c / 2^k, (c + 1) / 2^k).
bool stratified(const bc::detail::sobol_tables& t, int max_k) {
    const k::sobol_view v = t.view();
    const std::size_t count = std::size_t{1} << max_k;
    std::vector<k::limb> pts(count * static_cast<std::size_t>(v.dims));
    for (std::size_t i = 0; i < count; ++i)
        k::sobol_point(v, i, &pts[i * static_cast<std::size_t>(v.dims)]);
    for (int d = 0; d < v.dims; ++d) {
        for (int kk = 0; kk <= max_k; ++kk) {
            const std::size_t n = std::size_t{1} << kk;
            std::vector<k::limb> cells(n);
            for (std::size_t i = 0; i < n; ++i) {
                const k::limb x =
                    pts[i * static_cast<std::size_t>(v.dims) + static_cast<std::size_t>(d)];
                cells[i] = kk == 0 ? 0 : x >> (64 - kk);
            }
            std::sort(cells.begin(), cells.end());
            for (std::size_t c = 0; c < n; ++c) {
                if (cells[c] != c) return false;
            }
        }
    }
    return true;
}

}  // namespace

TEST_CASE("unscrambled Sobol points match scipy, including the last dimensions") {
    const bc::detail::sobol_tables t = bc::detail::make_sobol(21201, false, 0);
    const k::sobol_view v = t.view();
    std::map<std::uint64_t, std::vector<k::limb>> points;
    for (const sobol_case& c : sobol_cases) {
        auto it = points.find(c.index);
        if (it == points.end()) {
            it = points.emplace(c.index, std::vector<k::limb>(21201)).first;
            k::sobol_point(v, c.index, it->second.data());
        }
        INFO("dimension " << c.dim << ", index " << c.index);
        CHECK((it->second[static_cast<std::size_t>(c.dim)] >> 34) == c.top30);
    }
    CHECK_THROWS_AS(bc::detail::make_sobol(21202, false, 0), std::invalid_argument);
}

TEST_CASE("scrambling keeps every coordinate stratified and changes the points") {
    CHECK(stratified(bc::detail::make_sobol(40, false, 0), 12));
    for (std::uint64_t seed : {1u, 2u, 12345u}) {
        CHECK(stratified(bc::detail::make_sobol(40, true, seed), 12));
    }
    const auto a = bc::detail::make_sobol(8, true, 1), b = bc::detail::make_sobol(8, true, 2);
    const auto plain = bc::detail::make_sobol(8, false, 0);
    CHECK(a.directions != b.directions);
    CHECK(a.directions != plain.directions);
    CHECK(a.shift != b.shift);
}

TEST_CASE("decode inverts encode on every composition of small spaces") {
    std::mt19937 rng(8);
    auto pick = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
    for (int trial = 0; trial < 300; ++trial) {
        const std::size_t n = static_cast<std::size_t>(pick(1, 6));
        composition lo(n), hi(n);
        for (std::size_t i = 0; i < n; ++i) {
            lo[i] = pick(-2, 2);
            hi[i] = lo[i] + pick(0, 7);
        }
        const bc::space s(pick(-3, 20), lo, hi);
        if (s.empty()) continue;
        composition x(n), back(n);
        std::vector<std::uint64_t> fixed(n);
        std::vector<double> u(n);
        REQUIRE(s.first(x.data()));
        do {
            bc::encode(s, x.data(), fixed.data());
            bc::decode(s, fixed.data(), back.data());
            REQUIRE(back == x);
            bc::encode(s, x.data(), u.data());
            for (std::size_t i = 0; i + 1 < n; ++i) REQUIRE((u[i] >= 0.0 && u[i] < 1.0));
            bc::decode(s, u.data(), back.data());
            REQUIRE(back == x);
        } while (s.next(x.data()));
    }
}

TEST_CASE("decoding uniform coordinates gives uniform compositions") {
    const bc::space s(6, composition(4, 0), composition(4, 6));  // 84 compositions
    std::vector<double> hits(84, 0.0);
    const k::limb key[2] = {99, 7};
    composition x(4);
    for (std::uint64_t i = 0; i < 84 * 500; ++i) {
        const k::limb counter[4] = {i, 0, 0, 0};
        const k::philox_block b = k::philox4x64_10(counter, key);
        const std::uint64_t u[3] = {b.v[0], b.v[1], b.v[2]};
        bc::decode(s, u, x.data());
        hits[s.rank(x).to_u64()] += 1.0;
    }
    double chi2 = 0.0;
    for (double h : hits) chi2 += (h - 500.0) * (h - 500.0) / 500.0;
    const double df = 83.0, z = 4.753;  // Wilson-Hilferty, p = 1e-6
    const double limit = df * std::pow(1.0 - 2.0 / (9.0 * df) + z * std::sqrt(2.0 / (9.0 * df)), 3);
    INFO("chi2 = " << chi2 << ", limit = " << limit);
    CHECK(chi2 < limit);
}

TEST_CASE("cells too narrow for the representation are reported") {
    // (30, 0, ..., 0) has one completion out of C(59, 29), about 2^55.7: below
    // double resolution, within 64 bits.
    const bc::space mid(30, composition(30, 0), composition(30, 30));
    composition x(30, 0);
    x[0] = 30;
    std::vector<std::uint64_t> fixed(29);
    std::vector<double> u(29);
    bc::encode(mid, x.data(), fixed.data());
    composition back(30);
    bc::decode(mid, fixed.data(), back.data());
    CHECK(back == x);
    CHECK_THROWS_AS(bc::encode(mid, x.data(), u.data()), std::domain_error);
    // (40, 0, ..., 0) out of C(79, 39), about 2^75: below 64-bit resolution.
    const bc::space wide(40, composition(40, 0), composition(40, 40));
    composition y(40, 0);
    y[0] = 40;
    std::vector<std::uint64_t> fixed40(39);
    CHECK_THROWS_AS(bc::encode(wide, y.data(), fixed40.data()), std::domain_error);
    // Invalid inputs.
    composition bad(40, 0);
    CHECK_THROWS_AS(bc::encode(wide, bad.data(), fixed40.data()), std::invalid_argument);
    std::vector<double> outside(39, 0.5);
    outside[3] = 1.0;
    CHECK_THROWS_AS(bc::decode(wide, outside.data(), y.data()), std::out_of_range);
    outside[3] = std::nan("");
    CHECK_THROWS_AS(bc::decode(wide, outside.data(), y.data()), std::out_of_range);
    const bc::space none(9, composition{0, 0}, composition{3, 3});
    const std::uint64_t one[1] = {0};
    composition two(2);
    CHECK_THROWS_AS(bc::decode(none, one, two.data()), std::out_of_range);
}

TEST_CASE("coordinate sequences: valid points, first part stratified, extensible") {
    const bc::space s(30, composition(6, 0), composition(6, 12));
    const std::uint64_t size = s.size().to_u64();
    // Exact share of each first-part value: compositions with that value.
    std::vector<std::uint64_t> exact(13, 0);
    composition x(6);
    REQUIRE(s.first(x.data()));
    do {
        exact[static_cast<std::size_t>(x[0])] += 1;
    } while (s.next(x.data()));
    for (const bc::sequence& seq : {bc::coordinate_sequence(s), bc::coordinate_sequence(s, 5)}) {
        CHECK(seq.kind() == bc::sequence_kind::coordinate);
        const std::uint64_t count = 4096;
        std::vector<std::int32_t> pts(count * 6);
        seq.points(0, count, pts.data());
        for (std::uint64_t i = 0; i < count; ++i) REQUIRE(s.contains(pts.data() + i * 6));
        // At N = 2^j points, the count of each first-part value is within two
        // of N * share (the first coordinate alone decides that part).
        for (std::uint64_t n = 1; n <= count; n *= 2) {
            std::vector<std::int64_t> got(13, 0);
            for (std::uint64_t i = 0; i < n; ++i) got[static_cast<std::size_t>(pts[i * 6])] += 1;
            for (std::size_t v = 0; v < 13; ++v) {
                const double expected = static_cast<double>(n) * static_cast<double>(exact[v]) /
                                        static_cast<double>(size);
                REQUIRE(std::abs(static_cast<double>(got[v]) - expected) < 2.0);
            }
        }
        // Nesting, thread independence and rank_at agree with points().
        bc::sequence copy = seq;
        std::vector<std::int32_t> first(1000 * 6), more(96 * 6);
        copy.take(1000, first.data());
        copy.take(96, more.data());
        first.insert(first.end(), more.begin(), more.end());
        CHECK(std::equal(first.begin(), first.end(), pts.begin()));
        bc::thread_pool pool(7);
        std::vector<std::int32_t> threaded(count * 6);
        seq.points(0, count, threaded.data(), &pool);
        CHECK(threaded == pts);
        CHECK(seq.rank_at(777) == s.rank(pts.data() + 777 * 6));
    }
}

TEST_CASE("a subtree's coordinate sequence is the subspace's with the prefix in front") {
    const bc::space s(30, composition(6, 0), composition(6, 12));
    const composition prefix{4, 7};
    const bc::region t = bc::subtree(s, prefix);
    const bc::space sub = s.subspace(t.prefix());
    const bc::sequence in_tree = bc::coordinate_sequence(s, t, 3);
    const bc::sequence in_sub = bc::coordinate_sequence(sub, 3);
    std::vector<std::int32_t> full(500 * 6), tail(500 * 4);
    in_tree.points(0, 500, full.data());
    in_sub.points(0, 500, tail.data());
    for (std::size_t i = 0; i < 500; ++i) {
        REQUIRE(full[i * 6] == 4);
        REQUIRE(full[i * 6 + 1] == 7);
        REQUIRE(std::equal(tail.begin() + static_cast<long>(i * 4),
                           tail.begin() + static_cast<long>(i * 4 + 4),
                           full.begin() + static_cast<long>(i * 6 + 2)));
    }
    CHECK_THROWS_AS(bc::coordinate_sequence(s, bc::interval(s, natural(1), natural(5))),
                    std::invalid_argument);
    // A subtree that is a single composition.
    composition x(6);
    REQUIRE(s.first(x.data()));
    const bc::sequence single = bc::coordinate_sequence(s, bc::subtree(s, x));
    composition got(6);
    single.points(12345, 1, got.data());
    CHECK(got == x);
}

TEST_CASE("coordinate sequences need at most 21202 parts") {
    const bc::space many(0, composition(21203, 0), composition(21203, 0));
    CHECK_THROWS_AS(bc::coordinate_sequence(many), std::invalid_argument);
    const bc::space most(0, composition(21202, 0), composition(21202, 0));
    composition x(21202, -1);
    bc::coordinate_sequence(most).points(0, 1, x.data());
    CHECK(std::all_of(x.begin(), x.end(), [](std::int32_t v) { return v == 0; }));
}
