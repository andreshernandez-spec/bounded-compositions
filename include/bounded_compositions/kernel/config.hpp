// Kernel layer configuration.
//
// Everything under bounded_compositions/kernel/ must be able to compile as GPU device
// code later, so it follows docs/kernel-rules.md: no allocation, exceptions,
// RTTI, virtual calls, recursion or mutable static state, and no standard
// library beyond <cstdint>, <cstddef> and <climits>.
#ifndef BCOMP_KERNEL_CONFIG_HPP
#define BCOMP_KERNEL_CONFIG_HPP

#include <cstdint>

// A GPU build defines BCOMP_KERNEL as `__host__ __device__ inline` (or the SYCL
// equivalent) before including any kernel header.
#ifndef BCOMP_KERNEL
#define BCOMP_KERNEL inline
#endif

// Upper bound on the 64-bit limbs of any count or rank. Kernels size their
// scratch with it, so it has to be a compile-time constant. 64 limbs is 4096
// bits; the host rejects spaces that need more.
#ifndef BCOMP_MAX_LIMBS
#define BCOMP_MAX_LIMBS 64
#endif

namespace bounded_compositions {
namespace kernel {

using limb = std::uint64_t;

constexpr int max_limbs = BCOMP_MAX_LIMBS;

// Kernels report errors by value, never by throwing.
enum class status : std::int32_t {
    ok = 0,
    out_of_range = 1,    // a rank or index outside the space
    too_many_limbs = 2,  // a value needs more than max_limbs limbs
    invalid = 3,         // a composition that violates the bounds or the total
};

}  // namespace kernel
}  // namespace bounded_compositions

#endif
