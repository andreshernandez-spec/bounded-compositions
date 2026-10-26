// Built with BCOMP_MAX_LIMBS=2, so the limb limit is reachable with small tables.
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

#include "bounded_compositions/bounded_compositions.hpp"

static_assert(bounded_compositions::kernel::max_limbs == 2, "this test needs BCOMP_MAX_LIMBS=2");

TEST_CASE("spaces within the limb limit still work") {
    // C(30 + 19, 19) needs 45 bits: one limb.
    const std::vector<std::int32_t> lo(20, 0), hi(20, 30);
    const bounded_compositions::space s(30, lo, hi);
    CHECK(s.size().limb_count() == 1);
    const auto x = s.unrank(s.size() - bounded_compositions::natural(1));
    CHECK(s.rank(x) == s.size() - bounded_compositions::natural(1));
}

TEST_CASE("a space needing more limbs than the limit is rejected") {
    // C(100 + 99, 99) needs 195 bits: four limbs.
    const std::vector<std::int32_t> lo(100, 0), hi(100, 100);
    CHECK_THROWS_AS(bounded_compositions::space(100, lo, hi), std::length_error);
}
