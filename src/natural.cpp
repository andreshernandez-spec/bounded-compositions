#include "bounded_compositions/natural.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>

#include "bounded_compositions/kernel/limbs.hpp"

namespace bounded_compositions {

namespace k = kernel;

namespace {

constexpr std::uint32_t chunk_base = 1000000000u;  // 10^9, nine decimal digits
constexpr int chunk_digits = 9;

int as_int(std::size_t n) { return static_cast<int>(n); }

}  // namespace

natural::natural(std::uint64_t value) {
    if (value != 0) limbs_.push_back(value);
}

void natural::trim() noexcept {
    while (!limbs_.empty() && limbs_.back() == 0) limbs_.pop_back();
}

natural natural::from_limbs(const k::limb* limbs, std::size_t count) {
    natural out;
    out.limbs_.assign(limbs, limbs + count);
    out.trim();
    return out;
}

natural natural::parse(std::string_view decimal) {
    if (decimal.empty()) throw std::invalid_argument("natural::parse: empty string");
    for (char c : decimal) {
        if (c < '0' || c > '9') throw std::invalid_argument("natural::parse: not a decimal digit");
    }
    natural out;
    std::size_t pos = 0;
    // The first chunk takes the leftover digits so the rest are exactly nine wide.
    std::size_t len = decimal.size() % chunk_digits;
    if (len == 0) len = chunk_digits;
    while (pos < decimal.size()) {
        std::uint32_t chunk = 0;
        std::uint32_t scale = 1;
        for (std::size_t i = 0; i < len; ++i) {
            chunk = chunk * 10 + static_cast<std::uint32_t>(decimal[pos + i] - '0');
            scale *= 10;
        }
        out.mul_small(scale);
        out += natural(chunk);
        pos += len;
        len = chunk_digits;
    }
    return out;
}

std::string natural::to_string() const {
    if (is_zero()) return "0";
    natural rest = *this;
    std::vector<std::uint32_t> chunks;
    while (!rest.is_zero()) chunks.push_back(rest.divmod_small(chunk_base));
    std::string out = std::to_string(chunks.back());
    for (auto it = chunks.rbegin() + 1; it != chunks.rend(); ++it) {
        std::string part = std::to_string(*it);
        out.append(static_cast<std::size_t>(chunk_digits) - part.size(), '0');
        out += part;
    }
    return out;
}

int natural::bit_length() const noexcept {
    return k::limbs_bit_length(limbs_.data(), as_int(limbs_.size()));
}

double natural::to_double() const noexcept {
    const int bits = bit_length();
    if (bits <= 64) return is_zero() ? 0.0 : static_cast<double>(limbs_[0]);
    if (bits > 1024) return std::numeric_limits<double>::infinity();
    // The top 64 bits, with a 1 in the last if any bit below them is set. A
    // double keeps 53 of them, so that bit only tells a tie from a value
    // just above it, and rounding these 64 bits once rounds the whole value.
    const int shift = bits - 64;
    const std::size_t word = static_cast<std::size_t>(shift / 64);
    const int bit = shift % 64;
    k::limb top = limbs_[word] >> bit;
    bool below = false;
    if (bit != 0) {
        top |= limbs_[word + 1] << (64 - bit);
        below = (limbs_[word] << (64 - bit)) != 0;
    }
    for (std::size_t i = 0; i < word && !below; ++i) below = limbs_[i] != 0;
    if (below) top |= 1;
    return std::ldexp(static_cast<double>(top), shift);
}

std::uint64_t natural::to_u64() const {
    if (!fits_u64()) throw std::overflow_error("natural::to_u64: value needs more than 64 bits");
    return is_zero() ? 0 : limbs_[0];
}

natural& natural::operator+=(const natural& other) {
    const std::size_t n = std::max(limbs_.size(), other.limbs_.size()) + 1;
    limbs_.resize(n, 0);
    k::limbs_add(limbs_.data(), as_int(n), other.limbs_.data(), as_int(other.limbs_.size()),
                 limbs_.data(), as_int(n));
    trim();
    return *this;
}

natural& natural::operator-=(const natural& other) {
    if (compare(*this, other) < 0) {
        throw std::domain_error("natural: subtraction result would be negative");
    }
    k::limbs_sub(limbs_.data(), as_int(limbs_.size()), other.limbs_.data(),
                 as_int(other.limbs_.size()), limbs_.data(), as_int(limbs_.size()));
    trim();
    return *this;
}

natural& natural::mul_small(std::uint32_t m) {
    if (is_zero()) return *this;
    const k::limb carry = k::limbs_mul_add_word(limbs_.data(), as_int(limbs_.size()), m, 0);
    if (carry != 0) limbs_.push_back(carry);
    trim();
    return *this;
}

std::uint32_t natural::divmod_small(std::uint32_t d) {
    if (d == 0) throw std::domain_error("natural: division by zero");
    const std::uint32_t rem = k::limbs_divmod_u32(limbs_.data(), as_int(limbs_.size()), d);
    trim();
    return rem;
}

int compare(const natural& a, const natural& b) noexcept {
    return k::limbs_cmp(a.limbs_.data(), static_cast<int>(a.limbs_.size()), b.limbs_.data(),
                        static_cast<int>(b.limbs_.size()));
}

std::ostream& operator<<(std::ostream& os, const natural& value) { return os << value.to_string(); }

}  // namespace bounded_compositions
