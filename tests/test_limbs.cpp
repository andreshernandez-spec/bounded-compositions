#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "bounded_compositions/kernel/limbs.hpp"

namespace k = bounded_compositions::kernel;

TEST_CASE("mul_wide matches known 128-bit products") {
    const k::limb max = ~k::limb{0};
    k::limb_pair p = k::mul_wide(max, max);  // (2^64 - 1)^2 = 2^128 - 2^65 + 1
    CHECK(p.hi == max - 1);
    CHECK(p.lo == 1);
    p = k::mul_wide(k::limb{1} << 63, 2);
    CHECK(p.hi == 1);
    CHECK(p.lo == 0);
    p = k::mul_wide(0xffffffffu, 0xffffffffu);
    CHECK(p.hi == 0);
    CHECK(p.lo == 0xfffffffe00000001u);
    p = k::mul_wide(0x123456789abcdef0u, 0);
    CHECK(p.hi == 0);
    CHECK(p.lo == 0);
}

TEST_CASE("add and sub propagate carries across limbs") {
    const k::limb max = ~k::limb{0};
    k::limb a[3] = {max, max, 0};
    const k::limb one[1] = {1};
    k::limb out[3];
    CHECK(k::limbs_add(a, 3, one, 1, out, 3) == 0);
    CHECK(out[0] == 0);
    CHECK(out[1] == 0);
    CHECK(out[2] == 1);
    CHECK(k::limbs_sub(out, 3, one, 1, out, 3) == 0);  // in place
    CHECK(out[0] == max);
    CHECK(out[1] == max);
    CHECK(out[2] == 0);
    CHECK(k::limbs_sub(one, 1, a, 2, out, 2) == 1);  // 1 - (2^128 - 1) wraps
}

TEST_CASE("compare treats a shorter operand as zero-extended") {
    const k::limb a[3] = {5, 0, 0};
    const k::limb b[1] = {5};
    const k::limb c[2] = {0, 1};
    CHECK(k::limbs_cmp(a, 3, b, 1) == 0);
    CHECK(k::limbs_cmp(b, 1, c, 2) < 0);
    CHECK(k::limbs_cmp(c, 2, a, 3) > 0);
    CHECK(k::limbs_length(a, 3) == 1);
    CHECK(k::limbs_bit_length(c, 2) == 65);
    CHECK(k::limbs_bit_length(a, 0) == 0);
}

TEST_CASE("divmod by a 32-bit word inverts mul_add_word") {
    k::limb a[3] = {0x0123456789abcdefu, 0xfedcba9876543210u, 0};
    const k::limb before[3] = {a[0], a[1], a[2]};
    const std::uint32_t m = 4000000007u;
    a[2] = k::limbs_mul_add_word(a, 2, m, 123);
    CHECK(k::limbs_divmod_u32(a, 3, m) == 123);
    CHECK(k::limbs_cmp(a, 3, before, 3) == 0);
}

TEST_CASE("power-of-two detection") {
    const k::limb zero[2] = {0, 0};
    const k::limb one[1] = {1};
    const k::limb big[3] = {0, 0, k::limb{1} << 40};
    const k::limb not_pow[3] = {1, 0, k::limb{1} << 40};
    const k::limb three[1] = {3};
    CHECK_FALSE(k::limbs_is_power_of_two(zero, 2));
    CHECK(k::limbs_is_power_of_two(one, 1));
    CHECK(k::limbs_is_power_of_two(big, 3));
    CHECK_FALSE(k::limbs_is_power_of_two(not_pow, 3));
    CHECK_FALSE(k::limbs_is_power_of_two(three, 1));
}
