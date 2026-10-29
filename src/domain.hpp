// Private: whether ranks and regions of one space carry over to another.
#ifndef BCOMP_SRC_DOMAIN_HPP
#define BCOMP_SRC_DOMAIN_HPP

#include "bounded_compositions/region.hpp"
#include "bounded_compositions/space.hpp"

namespace bounded_compositions {
namespace detail {

// Same parts, total, bounds, support and group limits: the same compositions
// in the same order, so ranks and regions carry over.
bool same_space(const space& a, const space& b);

// Throws std::invalid_argument unless r belongs to s.
void require_domain(const space& s, const region& r, const char* what);

}  // namespace detail
}  // namespace bounded_compositions

#endif
