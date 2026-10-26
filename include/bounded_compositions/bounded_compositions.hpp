// Public C++ API.
#ifndef BCOMP_BCOMP_HPP
#define BCOMP_BCOMP_HPP

#include "bounded_compositions/codec.hpp"
#include "bounded_compositions/executor.hpp"
#include "bounded_compositions/fresh.hpp"
#include "bounded_compositions/natural.hpp"
#include "bounded_compositions/region.hpp"
#include "bounded_compositions/sequence.hpp"
#include "bounded_compositions/space.hpp"

namespace bounded_compositions {

// Library version as "major.minor.patch".
const char* version() noexcept;

}  // namespace bounded_compositions

#endif
