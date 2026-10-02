#include <catch2/catch_test_macros.hpp>

#include <cstring>

#include "bounded_compositions/bounded_compositions.hpp"
#include "bounded_compositions/kernel/config.hpp"

TEST_CASE("version string is set") { REQUIRE(std::strlen(bounded_compositions::version()) > 0); }

TEST_CASE("kernel limb budget is at least 64 bits") {
    STATIC_REQUIRE(bounded_compositions::kernel::max_limbs >= 1);
    STATIC_REQUIRE(sizeof(bounded_compositions::kernel::limb) == 8);
}
