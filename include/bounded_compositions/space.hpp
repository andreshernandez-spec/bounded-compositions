// The set of bounded compositions of a total, with exact counting.
#ifndef BCOMP_SPACE_HPP
#define BCOMP_SPACE_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "bounded_compositions/kernel/views.hpp"
#include "bounded_compositions/natural.hpp"

namespace bounded_compositions {

namespace detail {
struct space_data;
}

// How a space counts its compositions.
enum class counting {
    // Exact multiprecision counts: sizes, ranks, regions and every sequence.
    exact,
    // Natural logarithms of the counts in doubles, for spaces too large for
    // exact tables. Supports log_size, contains/first/next, the unit-cube codec,
    // coordinate sequences and i.i.d. sequences (the sequential conditional
    // draw, uniform up to about 1e-12 relative error), but not size, ranks,
    // regions or rank sequences, which throw std::logic_error. decode agrees
    // with the exact decoder except for inputs within about 1e-12 of a cell
    // boundary, which include the exactly dyadic points of unscrambled Sobol
    // sequences; use scrambled coordinate sequences with logarithmic spaces.
    logarithmic,
};

// Limits on the parts that are nonzero (the support). The defaults allow
// every composition the bounds allow.
struct support_limits {
    // Number of parts with x[i] != 0.
    std::int32_t min_count = 0;
    std::int32_t max_count = std::numeric_limits<std::int32_t>::max();
    // Semicontinuous parts: x[i] == 0 or |x[i]| >= min_size[i]. Empty means 1
    // for every part, which allows every value.
    std::vector<std::int32_t> min_size;

    friend bool operator==(const support_limits& a, const support_limits& b) {
        return a.min_count == b.min_count && a.max_count == b.max_count && a.min_size == b.min_size;
    }
    friend bool operator!=(const support_limits& a, const support_limits& b) { return !(a == b); }
};

// All x in Z^n with x[0] + ... + x[n-1] = total and lower[i] <= x[i] <= upper[i],
// within the support limits, ordered lexicographically (first part slowest).
// The limits keep counting exact: they add a dimension for the number of
// nonzero parts so far (one layer per allowed count) and split each part's
// values into the ones below zero, zero, and the ones above.
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
          const std::vector<std::int32_t>& upper, counting mode = counting::exact);
    space(std::int32_t total, const std::int32_t* lower, const std::int32_t* upper,
          std::size_t parts, counting mode = counting::exact);
    // With support limits. Also throws std::invalid_argument if a count is
    // negative, min_count > max_count, or min_size is neither empty nor one
    // value of at least 1 per part, and std::logic_error for logarithmic
    // counting with limits that change the space, which it does not support.
    space(std::int32_t total, const std::vector<std::int32_t>& lower,
          const std::vector<std::int32_t>& upper, const support_limits& support,
          counting mode = counting::exact);
    space(std::int32_t total, const std::int32_t* lower, const std::int32_t* upper,
          std::size_t parts, const support_limits& support, counting mode = counting::exact);

    std::size_t parts() const noexcept;
    std::int32_t total() const noexcept;
    const std::vector<std::int32_t>& lower() const noexcept;
    const std::vector<std::int32_t>& upper() const noexcept;
    // For a subspace, the limits that apply to its own parts.
    const support_limits& support() const noexcept;

    counting counting_mode() const noexcept;
    // Number of compositions. Throws std::logic_error for a logarithmic space.
    const natural& size() const;
    // Natural logarithm of the number of compositions (minus infinity if
    // there are none), in either counting mode.
    double log_size() const noexcept;
    bool empty() const noexcept;

    // Writes the composition with rank r to out[0..parts()).
    // Throws std::out_of_range if r >= size(). Ranks need exact counting:
    // rank and unrank throw std::logic_error for a logarithmic space.
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

    // The compositions of the remaining parts once the first prefix.size()
    // parts are fixed to prefix: a space of parts() - prefix.size() parts that
    // shares this space's table. Its rank r is rank begin + r here, where
    // begin is the first rank of subtree(*this, prefix). Its support() is
    // what the limits leave for the remaining parts. Throws
    // std::invalid_argument if the prefix leaves no part, breaks a bound or a
    // gap, or has no completion.
    space subspace(const std::vector<std::int32_t>& prefix) const;

    // Kernel view of this space. Valid while this space or a copy of it lives.
    kernel::space_view view() const noexcept;
    // Memory held by the counting table (shared with subspaces).
    std::size_t table_bytes() const noexcept;

  private:
    explicit space(std::shared_ptr<const detail::space_data> data);

    std::shared_ptr<const detail::space_data> data_;
};

}  // namespace bounded_compositions

#endif
