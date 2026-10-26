// Private: exact counts from a space's table, as naturals.
#ifndef BCOMP_SRC_COUNTS_HPP
#define BCOMP_SRC_COUNTS_HPP

#include <algorithm>
#include <cstdint>
#include <vector>

#include "bounded_compositions/kernel/walk.hpp"
#include "bounded_compositions/natural.hpp"

namespace bounded_compositions {
namespace detail {

// N[row][layer][k]: ways to place k into parts row..n-1 from that layer.
inline natural completions_of(const kernel::space_view& v, int row, std::int32_t layer,
                              std::int32_t k) {
    std::vector<kernel::limb> out(static_cast<std::size_t>(v.row_limbs[row]));
    kernel::completions(v, row, layer, k, out.data());
    return natural::from_limbs(out.data(), out.size());
}

// The completions of part i's allowed values below y, in layer c with k left.
inline natural below_of(const kernel::space_view& v, int i, std::int32_t k, std::int32_t c,
                        std::int32_t y) {
    const int w = std::max(v.row_limbs[i], v.row_limbs[i + 1]);
    std::vector<kernel::limb> out(static_cast<std::size_t>(w), 0);
    kernel::add_below(v, i, k, c, y, out.data(), w);
    return natural::from_limbs(out.data(), out.size());
}

}  // namespace detail
}  // namespace bounded_compositions

#endif
