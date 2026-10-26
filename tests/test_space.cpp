#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "bounded_compositions/bounded_compositions.hpp"

using bounded_compositions::natural;
using bounded_compositions::space;
using composition = std::vector<std::int32_t>;

namespace {

// Lexicographic enumeration by plain recursion: the oracle.
void enumerate(std::size_t i, std::int64_t left, const composition& lo, const composition& hi,
               composition& cur, std::vector<composition>& out) {
    const std::size_t n = lo.size();
    if (i + 1 == n) {
        if (left >= lo[i] && left <= hi[i]) {
            cur[i] = static_cast<std::int32_t>(left);
            out.push_back(cur);
        }
        return;
    }
    for (std::int32_t v = lo[i]; v <= hi[i]; ++v) {
        cur[i] = v;
        enumerate(i + 1, left - v, lo, hi, cur, out);
    }
}

std::vector<composition> brute_force(std::int32_t total, const composition& lo,
                                     const composition& hi) {
    std::vector<composition> out;
    composition cur(lo.size());
    enumerate(0, total, lo, hi, cur, out);
    return out;
}

std::string describe(std::int32_t total, const composition& lo, const composition& hi) {
    std::string s = "total " + std::to_string(total) + ", bounds";
    for (std::size_t i = 0; i < lo.size(); ++i) {
        s += " " + std::to_string(lo[i]) + ".." + std::to_string(hi[i]);
    }
    return s;
}

}  // namespace

TEST_CASE("space matches brute-force enumeration on random small spaces") {
    std::mt19937 rng(12345);
    auto pick = [&](int a, int b) { return std::uniform_int_distribution<int>(a, b)(rng); };
    int nonempty = 0;
    for (int trial = 0; trial < 600; ++trial) {
        const std::size_t n = static_cast<std::size_t>(pick(1, 7));
        composition lo(n), hi(n);
        for (std::size_t i = 0; i < n; ++i) {
            lo[i] = pick(0, 3) == 0 ? pick(-3, 3) : 0;
            hi[i] = lo[i] + (pick(0, 5) == 0 ? 0 : pick(0, 6));  // some 0..0 parts
        }
        const std::int32_t total = pick(-4, 16);
        INFO(describe(total, lo, hi));

        const std::vector<composition> ref = brute_force(total, lo, hi);
        const space s(total, lo, hi);
        REQUIRE(s.size() == natural(ref.size()));
        REQUIRE(s.empty() == ref.empty());
        composition x(n);
        REQUIRE(s.first(x.data()) == !ref.empty());
        if (ref.empty()) {
            CHECK_THROWS_AS(s.unrank(natural(0)), std::out_of_range);
            continue;
        }
        ++nonempty;
        CHECK(x == ref[0]);
        for (std::size_t r = 0; r < ref.size(); ++r) {
            REQUIRE(s.unrank(natural(r)) == ref[r]);
            REQUIRE(s.rank(ref[r]) == natural(r));
            REQUIRE(s.contains(ref[r]));
            composition y = ref[r];
            const bool more = s.next(y.data());
            REQUIRE(more == (r + 1 < ref.size()));
            REQUIRE(y == (more ? ref[r + 1] : ref[r]));  // unchanged after the last
        }
        CHECK_THROWS_AS(s.unrank(natural(ref.size())), std::out_of_range);
    }
    CHECK(nonempty >= 200);  // at least a third of the random spaces are non-empty
}

TEST_CASE("contains and rank reject points outside the space") {
    const space s(10, {0, 0, 0}, {5, 5, 5});
    CHECK_FALSE(s.contains(composition{5, 5, 1}));   // sum 11
    CHECK_FALSE(s.contains(composition{6, 4, 0}));   // above a bound
    CHECK_FALSE(s.contains(composition{-1, 6, 5}));  // below a bound
    CHECK_FALSE(s.contains(composition{5, 5}));      // wrong length
    CHECK_THROWS_AS(s.rank(composition{5, 5, 1}), std::invalid_argument);
    CHECK_THROWS_AS(s.rank(composition{5, 5}), std::invalid_argument);
    CHECK(s.contains(composition{5, 5, 0}));
}

TEST_CASE("invalid construction arguments throw") {
    CHECK_THROWS_AS(space(3, composition{}, composition{}), std::invalid_argument);
    CHECK_THROWS_AS(space(3, composition{0, 0}, composition{1}), std::invalid_argument);
    CHECK_THROWS_AS(space(3, composition{2, 0}, composition{1, 5}), std::invalid_argument);
}

namespace {

struct size_case {
    int parts;
    int total;
    int lower;
    int upper;
    const char* size;
};

const size_case size_cases[] = {
#include "data/space_sizes.inc"
};

}  // namespace

TEST_CASE("space sizes match inclusion-exclusion counts") {
    for (const size_case& c : size_cases) {
        INFO("parts " << c.parts << ", total " << c.total << ", bounds " << c.lower << ".."
                      << c.upper);
        const composition lo(static_cast<std::size_t>(c.parts), c.lower);
        const composition hi(static_cast<std::size_t>(c.parts), c.upper);
        const space s(c.total, lo, hi);
        CHECK(s.size().to_string() == c.size);
        if (s.empty()) continue;
        // The ends and a middle rank survive a round trip.
        const natural last = s.size() - natural(1);
        natural mid = s.size();
        mid.divmod_small(2);
        for (const natural& r : {natural(0), mid, last}) {
            const composition x = s.unrank(r);
            CHECK(s.contains(x));
            CHECK(s.rank(x) == r);
        }
    }
}

TEST_CASE("the first composition is rank 0 and the successor of rank r is rank r + 1") {
    // n = 100, total = 100, no binding upper bound: C(199, 99), a 195-bit size.
    const composition lo(100, 0), hi(100, 100);
    const space s(100, lo, hi);
    CHECK(s.size().bit_length() == 195);
    composition x(100);
    REQUIRE(s.first(x.data()));
    CHECK(s.rank(x).is_zero());
    std::mt19937_64 rng(7);
    for (int trial = 0; trial < 200; ++trial) {
        // A random rank below the size, from random limbs reduced by rejection.
        natural r;
        do {
            std::vector<bounded_compositions::kernel::limb> limbs(s.size().limb_count());
            for (auto& l : limbs) l = rng();
            limbs.back() &=
                (bounded_compositions::kernel::limb{1} << (s.size().bit_length() % 64)) - 1;
            r = natural::from_limbs(limbs.data(), limbs.size());
        } while (r >= s.size() - natural(1));
        composition y = s.unrank(r);
        REQUIRE(s.next(y.data()));
        CHECK(s.rank(y) == r + natural(1));
    }
}

TEST_CASE("concurrent unranking of one space gives the serial results") {
    const composition lo(30, -2), hi(30, 9);
    const space s(80, lo, hi);
    const std::size_t count = 2000;
    natural step = s.size();
    step.divmod_small(static_cast<std::uint32_t>(count));
    std::vector<composition> serial(count);
    for (std::size_t i = 0; i < count; ++i) {
        natural r = step;
        r.mul_small(static_cast<std::uint32_t>(i));
        serial[i] = s.unrank(r);
    }
    std::vector<composition> parallel(count);
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < 8; ++t) {
        threads.emplace_back([&, t] {
            const space copy = s;  // copies share the table
            for (std::size_t i = t; i < count; i += 8) {
                natural r = step;
                r.mul_small(static_cast<std::uint32_t>(i));
                parallel[i] = copy.unrank(r);
            }
        });
    }
    for (auto& th : threads) th.join();
    CHECK(parallel == serial);
}

TEST_CASE("the kernel view is plain data") {
    STATIC_REQUIRE(std::is_trivially_copyable_v<bounded_compositions::kernel::space_view>);
    const space s(5, {0, 0}, {5, 5});
    CHECK(s.view().n == 2);
    CHECK(s.view().size_limbs == 1);
    CHECK(s.table_bytes() > 0);
}
