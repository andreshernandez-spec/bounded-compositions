// Arithmetic on natural logarithms of non-negative numbers, for the
// logarithmic counting mode. log(0) is represented by log_zero, a large
// negative finite value, so no infinities or NaNs reach the comparisons.
#ifndef BCOMP_KERNEL_LOGSUM_HPP
#define BCOMP_KERNEL_LOGSUM_HPP

#include "bounded_compositions/kernel/config.hpp"

namespace bounded_compositions {
namespace kernel {

constexpr double log_zero = -1e300;

BCOMP_KERNEL bool log_is_zero(double a) { return a <= log_zero / 2; }

// log(e^a + e^b)
BCOMP_KERNEL double log_add(double a, double b) {
    if (a < b) {
        const double t = a;
        a = b;
        b = t;
    }
    if (log_is_zero(b)) return a;
    return a + BCOMP_LOG1P(BCOMP_EXP(b - a));
}

// log(e^a - e^b) for a >= b; log_zero when they are equal.
BCOMP_KERNEL double log_sub(double a, double b) {
    if (log_is_zero(b)) return a;
    const double d = b - a;
    if (d >= 0.0) return log_zero;
    return a + BCOMP_LOG(-BCOMP_EXPM1(d));
}

}  // namespace kernel
}  // namespace bounded_compositions

#endif
