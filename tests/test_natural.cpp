#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
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

TEST_CASE("natural to_double is close for small and large values") {
    CHECK(natural(12345).to_double() == 12345.0);
    const natural big = natural::parse("340282366920938463463374607431768211456");  // 2^128
    CHECK(big.to_double() == std::ldexp(1.0, 128));
    natural huge(1);
    for (int i = 0; i < 1100; ++i) huge.mul_small(2);
    CHECK(std::isinf(huge.to_double()));
}
