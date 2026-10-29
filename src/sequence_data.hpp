// Private: what a sequence holds, shared by sequence.cpp and fresh.cpp.
#ifndef BCOMP_SRC_SEQUENCE_DATA_HPP
#define BCOMP_SRC_SEQUENCE_DATA_HPP

#include <cstdint>
#include <vector>

#include "bounded_compositions/kernel/radical.hpp"
#include "bounded_compositions/natural.hpp"
#include "bounded_compositions/sequence.hpp"
#include "bounded_compositions/space.hpp"
#include "sobol.hpp"

namespace bounded_compositions {
namespace detail {

struct sequence_data {
    space dom;
    sequence_kind kind = sequence_kind::rank;
    std::uint64_t seed = 0;
    std::uint64_t capacity = 0;
    int rank_limbs = 1;
    // rank: the space's mask M, and the view of the region's order.
    std::vector<kernel::limb> mask;
    std::vector<kernel::limb> origin;
    kernel::radical_view radical{};
    kernel::limb key[2] = {0, 0};
    natural base;  // first rank of the region the sequence covers
    natural span;  // ranks in that region
    // coordinate: Sobol points decoded in `walk` (the space, or the subspace
    // below a subtree's prefix), with `prefix` written in front.
    space walk;
    std::vector<std::int32_t> prefix;
    sobol_tables sobol;
    // Points come from decoding fractions: coordinate sequences, and iid
    // sequences over a logarithmic space (Philox fractions).
    bool decoded = false;

    explicit sequence_data(const space& s) : dom(s), walk(s) {}

    // Rank of point i into out[0..rank_limbs). Not for decoded sequences.
    void rank_of(std::uint64_t i, kernel::limb* out) const;
    void write_decoded_points(std::uint64_t first, std::uint64_t count, std::int32_t* out) const;
    void write_points(std::uint64_t first, std::uint64_t count, std::int32_t* out) const;
};

}  // namespace detail
}  // namespace bounded_compositions

#endif
