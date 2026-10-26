// Repair: the composition of a space nearest to a real vector, for optimisers
// whose candidates leave the space (evolutionary and continuous methods).
#ifndef BCOMP_REPAIR_HPP
#define BCOMP_REPAIR_HPP

#include <cstdint>
#include <vector>

#include "bounded_compositions/space.hpp"

namespace bounded_compositions {

// Writes to out[0..parts()) the composition x of s nearest to z[0..parts())
// in squared distance, the sum over i of (x[i] - z[i])^2. Exact for every
// limit the space counts (bounds, support and group limits), in either
// counting mode; ties go to the composition first in rank order.
//
// Without limits it is a greedy allocation from the lower bounds, one unit at
// a time to the part where it adds least, which is exact for separable convex
// costs: O(total log n). With limits it is a shortest path over the states of
// the counting table, about the work of building the table times the values
// per part, holding a double per state.
//
// Throws std::invalid_argument for an empty space or a z that is not finite,
// and std::length_error if the states exceed 2^27.
void repair(const space& s, const double* z, std::int32_t* out);
std::vector<std::int32_t> repair(const space& s, const std::vector<double>& z);

}  // namespace bounded_compositions

#endif
