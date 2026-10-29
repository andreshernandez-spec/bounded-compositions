#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "bounded_compositions/natural.hpp"

using bounded_compositions::natural;

namespace {

struct vector_case {
    const char* a;
    const char* b;
    const char* sum;
    const char* diff;  // empty when a < b
    int sign;
    std::uint32_t m;
    const char* product;
    std::uint32_t d;
    const char* quotient;
    std::uint32_t remainder;
};

const vector_case cases[] = {
#include "data/natural_vectors.inc"
};

}  // namespace

TEST_CASE("natural agrees with Python big integers") {
    for (const vector_case& c : cases) {
        INFO("a = " << c.a << ", b = " << c.b);
        const natural a = natural::parse(c.a);
        const natural b = natural::parse(c.b);
        CHECK(a.to_string() == c.a);
        CHECK((a + b).to_string() == c.sum);
        if (c.sign >= 0) {
            CHECK((a - b).to_string() == c.diff);
        } else {
            CHECK_THROWS_AS(a - b, std::domain_error);
        }
        CHECK(compare(a, b) == c.sign);
        CHECK((a < b) == (c.sign < 0));
        CHECK((a == b) == (c.sign == 0));
        natural p = a;
        CHECK(p.mul_small(c.m).to_string() == c.product);
        natural q = a;
        CHECK(q.divmod_small(c.d) == c.remainder);
        CHECK(q.to_string() == c.quotient);
    }
}

TEST_CASE("natural basics") {
    CHECK(natural().is_zero());
    CHECK(natural(0).is_zero());
    CHECK(natural().to_string() == "0");
    CHECK(natural::parse("000123").to_string() == "123");
    CHECK(natural(42).to_u64() == 42);
    CHECK(natural::parse("18446744073709551615").to_u64() == UINT64_MAX);
    CHECK_THROWS_AS(natural::parse("18446744073709551616").to_u64(), std::overflow_error);
    CHECK_THROWS_AS(natural::parse(""), std::invalid_argument);
    CHECK_THROWS_AS(natural::parse("12a"), std::invalid_argument);
    CHECK_THROWS_AS(natural::parse("-1"), std::invalid_argument);
    CHECK(natural::parse("18446744073709551616").bit_length() == 65);
    CHECK(natural(1).bit_length() == 1);
    CHECK(natural().bit_length() == 0);
    natural x(7);
    CHECK_THROWS_AS(x.divmod_small(0), std::domain_error);
}

namespace {

natural times_power_of_two(natural x, int k) {
    for (int i = 0; i < k; ++i) x.mul_small(2);
    return x;
}

}  // namespace

TEST_CASE("natural to_double is the nearest double with ties to even") {
    CHECK(natural().to_double() == 0.0);
    CHECK(natural(12345).to_double() == 12345.0);
    const natural big = natural::parse("340282366920938463463374607431768211456");  // 2^128
    CHECK(big.to_double() == std::ldexp(1.0, 128));
    // Limbs {1, 2^53 + 1}: rounding the top limb first gave 2^117.
    CHECK(natural::parse("166153499473114502559719956244594689").to_double() ==
          std::ldexp(1.0 + std::ldexp(1.0, -52), 117));
    // (2^53 + j) 2^k + e, with doubles 2^k (2^53 + 2m) on both sides of the
    // odd j. k = 10 still fits in one limb; the others cross limbs.
    const std::uint64_t base = std::uint64_t{1} << 53;
    for (int k : {10, 11, 63, 64, 65, 117, 600, 970}) {
        INFO("k = " << k);
        auto near = [&](std::uint64_t m) { return std::ldexp(static_cast<double>(base + m), k); };
        auto value = [&](std::uint64_t j, int e) {
            natural x = times_power_of_two(natural(base + j), k);
            if (e > 0) x += natural(1);
            if (e < 0) x -= natural(1);
            return x.to_double();
        };
        CHECK(value(1, 0) == near(0));  // a tie goes to the even neighbour
        CHECK(value(3, 0) == near(4));
        CHECK(value(1, 1) == near(2));  // just above a tie
        CHECK(value(3, 1) == near(4));
        CHECK(value(1, -1) == near(0));  // just below a tie
        CHECK(value(3, -1) == near(2));
        CHECK(value(2, 1) == near(2));
        CHECK(value(2, -1) == near(2));
    }
    // The largest double, (2^53 - 1) 2^971, and the tie above it, which rounds
    // to 2^1024: infinity.
    const natural largest = times_power_of_two(natural(base - 1), 971);
    CHECK(largest.to_double() == std::numeric_limits<double>::max());
    const natural tie = largest + times_power_of_two(natural(1), 970);
    CHECK(std::isinf(tie.to_double()));
    CHECK((tie - natural(1)).to_double() == std::numeric_limits<double>::max());
    CHECK(std::isinf(times_power_of_two(natural(1), 1100).to_double()));
}
