// The set of bounded compositions of a total, with exact counting.
#ifndef BCOMP_SPACE_HPP
#define BCOMP_SPACE_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "bounded_compositions/kernel/views.hpp"
#include "bounded_compositions/natural.hpp"

namespace bounded_compositions {

namespace detail {
struct space_data;
}

// All x in Z^n with x[0] + ... + x[n-1] = total and lower[i] <= x[i] <= upper[i],
// ordered lexicographically (first part slowest).
//
// A space is immutable. Copies share the same table, and every member
// function is safe to call from many threads at once.
class space {
  public:
    // Throws std::invalid_argument if there are no parts, the bound vectors
    // differ in length, or some upper[i] < lower[i]. Bounds whose sums leave
    // no composition give an empty space, not an error. Throws
    // std::length_error if the size needs more than kernel::max_limbs limbs.
    space(std::int32_t total, const std::vector<std::int32_t>& lower,
          const std::vector<std::int32_t>& upper);
    space(std::int32_t total, const std::int32_t* lower, const std::int32_t* upper,
          std::size_t parts);

    std::size_t parts() const noexcept;
    std::int32_t total() const noexcept;
    const std::vector<std::int32_t>& lower() const noexcept;
    const std::vector<std::int32_t>& upper() const noexcept;

    // Number of compositions.
    const natural& size() const noexcept;
    bool empty() const noexcept;

    // Writes the composition with rank r to out[0..parts()).
    // Throws std::out_of_range if r >= size().
    void unrank(const natural& r, std::int32_t* out) const;
    std::vector<std::int32_t> unrank(const natural& r) const;

    // Throws std::invalid_argument if x is not in the space.
    natural rank(const std::int32_t* x) const;
    natural rank(const std::vector<std::int32_t>& x) const;

    bool contains(const std::int32_t* x) const noexcept;
    bool contains(const std::vector<std::int32_t>& x) const noexcept;

    // Writes the first composition. Returns false if the space is empty.
    bool first(std::int32_t* out) const noexcept;
    // Advances x to the next composition. Returns false, leaving x unchanged,
    // if x is the last one. x must be in the space.
    bool next(std::int32_t* x) const noexcept;

    // Kernel view of this space. Valid while this space or a copy of it lives.
    kernel::space_view view() const noexcept;
    // Memory held by the counting table.
    std::size_t table_bytes() const noexcept;

  private:
    std::shared_ptr<const detail::space_data> data_;
};

}  // namespace bounded_compositions

#endif
