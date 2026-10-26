// Joe and Kuo's Sobol direction numbers (see sobol_directions.cpp).
#ifndef BCOMP_SRC_SOBOL_DIRECTIONS_HPP
#define BCOMP_SRC_SOBOL_DIRECTIONS_HPP

#include <cstdint>

namespace bounded_compositions {
namespace detail {

extern const int sobol_max_dims;
extern const unsigned char sobol_degree[];
extern const std::uint32_t sobol_coeffs[];
extern const std::uint32_t sobol_offset[];
extern const std::uint32_t sobol_initial[];

}  // namespace detail
}  // namespace bounded_compositions

#endif
