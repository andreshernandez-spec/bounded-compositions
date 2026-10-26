// Sobol points by index, as 64-bit fractions of the unit interval.
//
// Point i is the XOR of the direction numbers selected by the bits of its Gray
// code, i ^ (i >> 1), so it can be computed for any i on its own: the same
// points, in the same order, as the usual Antonov-Saleev recurrence (and as
// scipy.stats.qmc.Sobol without scrambling). The host prepares the direction
// numbers, optionally scrambled by a random linear matrix scramble, and a
// digital shift that is XORed into every point.
#ifndef BCOMP_KERNEL_SOBOL_HPP
#define BCOMP_KERNEL_SOBOL_HPP

#include <cstdint>

#include "bounded_compositions/kernel/config.hpp"

namespace bounded_compositions {
namespace kernel {

constexpr int sobol_bits = 64;

struct sobol_view {
    const limb* directions;  // dims * 64 limbs: direction k of dimension d at [d * 64 + k]
    const limb* shift;       // dims limbs, XORed into every point
    std::int32_t dims;
};

// Writes point i to out[0..dims).
BCOMP_KERNEL void sobol_point(const sobol_view& v, std::uint64_t i, limb* out) {
    for (int d = 0; d < v.dims; ++d) out[d] = v.shift[d];
    std::uint64_t gray = i ^ (i >> 1);
    for (int k = 0; gray != 0; ++k, gray >>= 1) {
        if ((gray & 1u) == 0) continue;
        for (int d = 0; d < v.dims; ++d) out[d] ^= v.directions[d * sobol_bits + k];
    }
}

}  // namespace kernel
}  // namespace bounded_compositions

#endif
