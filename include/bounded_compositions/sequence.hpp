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

enum class sequence_kind {
    // Radical-inverse (one-dimensional Halton, van der Corput) order on the
    // ranks, skipping indices that fall past the last rank. The first 2^j
    // points are an even grid over the ranks, each prefix stays spread out,
    // and no rank repeats: after size() points the whole space is covered.
    rank,
    // floor(phi(i) * size) with phi the base-2 radical inverse of the 64-bit
    // index. Repeats ranks; kept only for comparison with `rank`.
    scaled_rank,
    // Independent uniform ranks from Philox4x64-10 keyed by the seed.
    iid,
};

class sequence {
  public:
    sequence(const sequence& other);
    sequence& operator=(const sequence& other);

    const space& domain() const noexcept;
    sequence_kind kind() const noexcept;
    std::uint64_t seed() const noexcept;
    // Number of points the sequence can produce. For `rank` this is the
    // space's size, or slightly less than 2^64 for larger spaces.
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

    friend sequence rank_sequence(const space& s);
    friend sequence rank_sequence(const space& s, std::uint64_t scramble_seed);
    friend sequence rank_sequence(const space& s, const region& r);
    friend sequence rank_sequence(const space& s, const region& r, std::uint64_t scramble_seed);
    friend sequence scaled_rank_sequence(const space& s);
    friend sequence iid_sequence(const space& s, std::uint64_t seed);
    friend sequence iid_sequence(const space& s, const region& r, std::uint64_t seed);

  private:
    explicit sequence(std::shared_ptr<const detail::sequence_data> data);

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
// full compositions of s whose ranks lie in r, and capacity() is r's size (or
// just under 2^64 for larger regions). Over the blocks of dyadic_split(s, r,
// j), the unscrambled rank sequences produce the points of rank_sequence(s,
// r), block by block (see dyadic_split for regions wider than 64 bits).
// Throws std::out_of_range if r ends past s.size().
sequence rank_sequence(const space& s, const region& r);
sequence rank_sequence(const space& s, const region& r, std::uint64_t scramble_seed);
sequence iid_sequence(const space& s, const region& r, std::uint64_t seed);

}  // namespace bounded_compositions

#endif
