#include "sobol.hpp"

#include <stdexcept>
#include <string>
#include <type_traits>

#include "bounded_compositions/kernel/philox.hpp"
#include "sobol_directions.hpp"

namespace bounded_compositions {
namespace detail {

namespace k = kernel;

static_assert(std::is_trivially_copyable_v<k::sobol_view>,
              "sobol_view must stay copyable to a device as is");

namespace {

constexpr int bits = k::sobol_bits;
constexpr k::limb tag_sobol = 0x736f626f6c2d7363u;  // "sobol-sc"

int parity(k::limb x) {
    x ^= x >> 32;
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    return static_cast<int>(x & 1u);
}

// Direction numbers v_1..v_64 of dimension d (0-based) as 64-bit fractions.
void fill_directions(int d, k::limb* v) {
    if (d == 0) {  // the first coordinate is the van der Corput sequence
        for (int j = 0; j < bits; ++j) v[j] = k::limb{1} << (bits - 1 - j);
        return;
    }
    const int s = sobol_degree[d - 1];
    const k::limb a = sobol_coeffs[d - 1];
    const std::uint32_t* initial = sobol_initial + sobol_offset[d - 1];
    k::limb m[bits];
    for (int j = 0; j < s && j < bits; ++j) m[j] = initial[j];
    // m_{j+1} = 2 a_1 m_j ^ 4 a_2 m_{j-1} ^ ... ^ 2^s m_{j+1-s} ^ m_{j+1-s}
    for (int j = s; j < bits; ++j) {
        k::limb next = m[j - s] ^ (m[j - s] << s);
        for (int i = 1; i < s; ++i) {
            if (((a >> (s - 1 - i)) & 1u) != 0) next ^= m[j - i] << i;
        }
        m[j] = next;
    }
    for (int j = 0; j < bits; ++j) v[j] = m[j] << (bits - 1 - j);
}

}  // namespace

k::sobol_view sobol_tables::view() const {
    k::sobol_view v;
    v.directions = directions.data();
    v.shift = shift.data();
    v.dims = static_cast<std::int32_t>(shift.size());
    return v;
}

sobol_tables make_sobol(int dims, bool scramble, std::uint64_t seed) {
    if (dims < 0 || dims > sobol_max_dims) {
        throw std::invalid_argument("bounded_compositions: Sobol points have at most " +
                                    std::to_string(sobol_max_dims) + " dimensions, " +
                                    std::to_string(dims) + " requested");
    }
    sobol_tables t;
    t.directions.assign(static_cast<std::size_t>(dims) * bits, 0);
    t.shift.assign(static_cast<std::size_t>(dims), 0);
    const k::limb key[2] = {seed, tag_sobol};
    for (int d = 0; d < dims; ++d) {
        k::limb* v = &t.directions[static_cast<std::size_t>(d) * bits];
        fill_directions(d, v);
        if (!scramble) continue;
        // Rows of the lower triangular matrix, most significant bit first:
        // row r keeps bit 63 - r and random bits above it.
        k::limb random[bits + 4];
        for (int b = 0; b < (bits + 4) / 4; ++b) {
            const k::limb counter[4] = {static_cast<k::limb>(d), static_cast<k::limb>(b), 0, 0};
            const k::philox_block block = k::philox4x64_10(counter, key);
            for (int j = 0; j < 4; ++j) random[4 * b + j] = block.v[j];
        }
        k::limb rows[bits];
        for (int r = 0; r < bits; ++r) {
            const int bit = bits - 1 - r;
            const k::limb above = bit == bits - 1 ? 0 : ~((k::limb{1} << (bit + 1)) - 1);
            rows[r] = (k::limb{1} << bit) | (random[r] & above);
        }
        for (int j = 0; j < bits; ++j) {
            k::limb scrambled = 0;
            for (int r = 0; r < bits; ++r) {
                scrambled |= static_cast<k::limb>(parity(rows[r] & v[j])) << (bits - 1 - r);
            }
            v[j] = scrambled;
        }
        t.shift[static_cast<std::size_t>(d)] = random[bits];
    }
    return t;
}

}  // namespace detail
}  // namespace bounded_compositions
