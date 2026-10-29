// Extensible sequences of compositions.
//
// Point i of a sequence is a pure function of (space, kind, seed, i). So
// taking 1000 points and then 100 more gives exactly the first 1100, a
// sequence can resume from any position, and any split of the work across
// threads gives the same points.
#ifndef BCOMP_SEQUENCE_HPP
#define BCOMP_SEQUENCE_HPP

#include <atomic>
#include <cstdint>
#include <memory>

#include "bounded_compositions/executor.hpp"
#include "bounded_compositions/natural.hpp"
#include "bounded_compositions/region.hpp"
#include "bounded_compositions/space.hpp"

namespace bounded_compositions {

namespace detail {
struct sequence_data;
}
class fresh_sampler;

enum class sequence_kind {
    // Radical-inverse (one-dimensional Halton, van der Corput) order on the
    // ranks: ranks sorted by their bit reversal (kernel/radical.hpp). The
    // first 2^j points are an even grid over the ranks, each prefix stays
    // spread out, and no rank repeats: after size() points the whole space is
    // covered. A region lists its ranks in the order of its space.
    rank,
    // floor(phi(i) * size) with phi the base-2 radical inverse of the 64-bit
    // index. Repeats ranks; kept only for comparison with `rank`.
    scaled_rank,
    // Independent uniform ranks from Philox4x64-10 keyed by the seed.
    iid,
    // Sobol points (Joe-Kuo direction numbers, optionally scrambled) decoded
    // part by part with the unit-cube codec (codec.hpp): every part's values
    // are spread, not only the leading ones'. Points can repeat.
    coordinate,
};

class sequence {
  public:
    sequence(const sequence& other);
    sequence& operator=(const sequence& other);

    const space& domain() const noexcept;
    sequence_kind kind() const noexcept;
    std::uint64_t seed() const noexcept;
    // Number of points the sequence can produce. For `rank` this is the
    // region's size, or between 2^63 and 2^64 - 1 for regions wider than 64
    // bits.
    std::uint64_t capacity() const noexcept;

    // Writes points [begin, begin + count) to out, parts() values per point,
    // using ex if given. Safe to call from many threads at once. Throws
    // std::out_of_range if begin + count > capacity().
    void points(std::uint64_t begin, std::uint64_t count, std::int32_t* out,
                executor* ex = nullptr) const;
    // Rank of point i. Throws std::out_of_range if i >= capacity().
    natural rank_at(std::uint64_t i) const;

    // Reserves the next count points with one atomic step and writes them to
    // out; returns the index of the first. Concurrent calls get disjoint,
    // consecutive ranges. Throws std::out_of_range, reserving nothing, if
    // fewer than count points remain.
    std::uint64_t take(std::uint64_t count, std::int32_t* out, executor* ex = nullptr);
    std::uint64_t position() const noexcept;
    // Throws std::out_of_range if position > capacity().
    void seek(std::uint64_t position);

    // How many of the first n points of outer are points of this sequence.
    // They are its first points, so seeking past them continues this region
    // after outer without repeating any. Both must be rank sequences of the
    // same space and seed, with this sequence's region inside outer's: throws
    // std::invalid_argument otherwise, and std::out_of_range if
    // n > outer.capacity().
    std::uint64_t overlap(const sequence& outer, std::uint64_t n) const;
    // seek(overlap(outer, outer.position())).
    void resume_after(const sequence& outer);

    friend sequence rank_sequence(const space& s);
    friend sequence rank_sequence(const space& s, std::uint64_t scramble_seed);
    friend sequence rank_sequence(const space& s, const region& r);
    friend sequence rank_sequence(const space& s, const region& r, std::uint64_t scramble_seed);
    friend sequence scaled_rank_sequence(const space& s);
    friend sequence iid_sequence(const space& s, std::uint64_t seed);
    friend sequence iid_sequence(const space& s, const region& r, std::uint64_t seed);
    friend sequence coordinate_sequence(const space& s);
    friend sequence coordinate_sequence(const space& s, std::uint64_t scramble_seed);
    friend sequence coordinate_sequence(const space& s, const region& subtree);
    friend sequence coordinate_sequence(const space& s, const region& subtree,
                                        std::uint64_t scramble_seed);

  private:
    friend class fresh_sampler;

    explicit sequence(std::shared_ptr<const detail::sequence_data> data);
    // Reserves up to count points (fewer when fewer remain) with one atomic
    // step; sets got and returns the index of the first.
    std::uint64_t reserve(std::uint64_t count, std::uint64_t& got);

    std::shared_ptr<const detail::sequence_data> data_;
    std::atomic<std::uint64_t> cursor_{0};
};

// Radical-inverse order on the ranks, unscrambled.
sequence rank_sequence(const space& s);
// The same order with all rank bits XORed by a mask drawn from the seed: an
// independent replicate with the same spread.
sequence rank_sequence(const space& s, std::uint64_t scramble_seed);
sequence scaled_rank_sequence(const space& s);
sequence iid_sequence(const space& s, std::uint64_t seed);

// The same sequences restricted to the ranks of a region of s: points are
// full compositions of s whose ranks lie in r. A rank sequence over r lists
// r's ranks in the order of rank_sequence(s) with the same seed, so over any
// piece of r it gives the points of r's sequence that fall in the piece, in
// the same order, and overlap() says where a region resumes. For regions
// wider than 64 bits a sequence holds fewer points than the region: a piece's
// sequence starts with r's points in it, then continues with ranks r's
// sequence does not reach. Throws std::out_of_range if r ends past s.size().
sequence rank_sequence(const space& s, const region& r);
sequence rank_sequence(const space& s, const region& r, std::uint64_t scramble_seed);
sequence iid_sequence(const space& s, const region& r, std::uint64_t seed);

// Sobol points in parts() - 1 dimensions decoded into compositions. Unscrambled,
// the Sobol points are those of scipy.stats.qmc.Sobol(scramble=False); the
// seeded version applies a linear matrix scramble and a digital shift, which
// keep each coordinate stratified over every power-of-two prefix. The first
// part's values then match its exact distribution to within two points at
// every power-of-two prefix. Capacity is just under 2^64. Throws
// std::invalid_argument for more than 21202 parts.
sequence coordinate_sequence(const space& s);
sequence coordinate_sequence(const space& s, std::uint64_t scramble_seed);
// Restricted to a subtree: the prefix is fixed and the Sobol points cover the
// remaining parts. Throws std::invalid_argument if the region is not a subtree
// of s, or if s.subspace(prefix) throws.
sequence coordinate_sequence(const space& s, const region& subtree);
sequence coordinate_sequence(const space& s, const region& subtree, std::uint64_t scramble_seed);

}  // namespace bounded_compositions

#endif
