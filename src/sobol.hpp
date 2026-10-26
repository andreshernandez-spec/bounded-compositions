// Host-side preparation of Sobol direction numbers (private header).
#ifndef BCOMP_SRC_SOBOL_HPP
#define BCOMP_SRC_SOBOL_HPP

#include <cstdint>
#include <vector>

#include "bounded_compositions/kernel/sobol.hpp"

namespace bounded_compositions {
namespace detail {

// Direction numbers and digital shift for `dims` dimensions. Unscrambled
// (scramble == false) they reproduce scipy.stats.qmc.Sobol(scramble=False).
// Scrambled, every dimension's directions go through a random lower
// triangular binary matrix with unit diagonal and every point is XORed with a
// random shift, both drawn from Philox keyed by the seed; this keeps each
// coordinate's stratification over every power-of-two prefix. Throws
// std::invalid_argument if dims exceeds the 21201 dimensions of the data.
struct sobol_tables {
    std::vector<kernel::limb> directions;  // dims * 64
    std::vector<kernel::limb> shift;       // dims
    kernel::sobol_view view() const;
};

sobol_tables make_sobol(int dims, bool scramble, std::uint64_t seed);

}  // namespace detail
}  // namespace bounded_compositions

#endif
