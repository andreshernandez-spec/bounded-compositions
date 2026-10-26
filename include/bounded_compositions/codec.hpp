// The unit-cube codec: compositions of a space and points of [0,1)^(parts - 1).
//
// decode maps u to a composition one part at a time: part i takes the value
// whose share of the completions left after parts 0..i-1 contains u[i]; the
// last part takes the rest. Any optimiser or sampler that works on a box of
// floats can then search a space of compositions without knowing about the
// fixed total. With independent uniform u the compositions come out uniform
// (up to 2^-64 rounding per part).
//
// encode returns the centre of a composition's cell, so decode(encode(x)) is
// x. A value whose share of the completions is below the resolution of the
// representation (2^-64 for 64-bit fractions, about 2^-53 for doubles) has no
// representable cell: decode never produces it and encode throws.
#ifndef BCOMP_CODEC_HPP
#define BCOMP_CODEC_HPP

#include <cstdint>

#include "bounded_compositions/space.hpp"

namespace bounded_compositions {

// u[0..parts()-1) in [0, 1); writes parts() values to out. Throws
// std::out_of_range if the space is empty or a u[i] is outside [0, 1) or NaN.
void decode(const space& s, const double* u, std::int32_t* out);
// The same with u[i] as 64-bit fractions u[i] / 2^64.
void decode(const space& s, const std::uint64_t* u, std::int32_t* out);

// Writes parts() - 1 coordinates. Throws std::invalid_argument if x is not in
// the space, and std::domain_error if its cell is too narrow to represent.
void encode(const space& s, const std::int32_t* x, double* u);
void encode(const space& s, const std::int32_t* x, std::uint64_t* u);

}  // namespace bounded_compositions

#endif
