// Unsigned multiprecision integer used for sizes and ranks.
#ifndef BCOMP_NATURAL_HPP
#define BCOMP_NATURAL_HPP

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

#include "bounded_compositions/kernel/config.hpp"

namespace bounded_compositions {

// Little-endian 64-bit limbs with no leading zero limbs; zero has none.
class natural {
  public:
    natural() = default;
    explicit natural(std::uint64_t value);

    // Decimal digits only. Throws std::invalid_argument otherwise.
    static natural parse(std::string_view decimal);
    // Copies limbs[0..count); leading zero limbs are dropped.
    static natural from_limbs(const kernel::limb* limbs, std::size_t count);

    std::string to_string() const;
    bool is_zero() const noexcept { return limbs_.empty(); }
    int bit_length() const noexcept;
    // Nearest double, ties to even; +inf when that is beyond the double range.
    double to_double() const noexcept;
    bool fits_u64() const noexcept { return limbs_.size() <= 1; }
    // Throws std::overflow_error unless fits_u64().
    std::uint64_t to_u64() const;

    const kernel::limb* limbs() const noexcept { return limbs_.data(); }
    std::size_t limb_count() const noexcept { return limbs_.size(); }

    natural& operator+=(const natural& other);
    // Throws std::domain_error if other > *this.
    natural& operator-=(const natural& other);
    natural& mul_small(std::uint32_t m);
    // Divides in place by 0 < d; returns the remainder. Throws std::domain_error for d == 0.
    std::uint32_t divmod_small(std::uint32_t d);

    friend natural operator+(natural a, const natural& b) { return a += b; }
    friend natural operator-(natural a, const natural& b) { return a -= b; }

    friend int compare(const natural& a, const natural& b) noexcept;
    friend bool operator==(const natural& a, const natural& b) noexcept {
        return a.limbs_ == b.limbs_;
    }
    friend bool operator!=(const natural& a, const natural& b) noexcept { return !(a == b); }
    friend bool operator<(const natural& a, const natural& b) noexcept { return compare(a, b) < 0; }
    friend bool operator<=(const natural& a, const natural& b) noexcept {
        return compare(a, b) <= 0;
    }
    friend bool operator>(const natural& a, const natural& b) noexcept { return compare(a, b) > 0; }
    friend bool operator>=(const natural& a, const natural& b) noexcept {
        return compare(a, b) >= 0;
    }

  private:
    void trim() noexcept;
    std::vector<kernel::limb> limbs_;
};

std::ostream& operator<<(std::ostream& os, const natural& value);

}  // namespace bounded_compositions

#endif
