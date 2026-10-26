// Sampling without repeating points that were already evaluated, and with
// limits the counting table does not hold.
//
// Rank sequences that share a space and seed never repeat each other when a
// region resumes after an outer sequence (sequence::overlap). Everything else
// (other orders, other seeds, coordinate and iid sequences, and points from
// outside the library) goes through a visited_set: fresh_sampler skips the
// points already in it. Limits the space does not count, such as general
// linear inequalities, go through a filter: fresh_sampler skips the points it
// rejects and counts them, so a sampler per region measures that region's
// acceptance.
#ifndef BCOMP_FRESH_HPP
#define BCOMP_FRESH_HPP

#include <atomic>
#include <cstdint>
#include <memory>

#include "bounded_compositions/executor.hpp"
#include "bounded_compositions/natural.hpp"
#include "bounded_compositions/sequence.hpp"
#include "bounded_compositions/space.hpp"

namespace bounded_compositions {

// The compositions of one space that have been evaluated, stored exactly by
// rank. Safe to use from many threads at once. Needs exact counting.
class visited_set {
  public:
    // Throws std::logic_error for a space with logarithmic counting.
    explicit visited_set(const space& s);
    ~visited_set();
    visited_set(const visited_set&) = delete;
    visited_set& operator=(const visited_set&) = delete;

    const space& domain() const noexcept;
    // Adds a composition, or a rank of the space; returns false if it was
    // already there. Throws std::invalid_argument if it is not in the space.
    bool insert(const std::int32_t* x);
    bool insert(const natural& rank);
    bool contains(const std::int32_t* x) const;
    bool contains(const natural& rank) const;
    std::uint64_t size() const noexcept;

  private:
    friend class fresh_sampler;
    struct shard;

    std::uint64_t hash(const kernel::limb* key) const noexcept;
    bool insert_key(const kernel::limb* key);
    bool contains_key(const kernel::limb* key) const;
    void key_of(const natural& rank, kernel::limb* key) const;

    space dom_;
    int width_;  // limbs per stored rank
    std::unique_ptr<shard[]> shards_;
    std::atomic<std::uint64_t> size_{0};
};

// A filter for limits the space does not count: returns true to keep x. It
// is called from the threads of the executor given to take() and from
// concurrent take() calls, so it must be safe to call from several threads,
// and it must give the same answer for the same x.
using accept_fn = bool (*)(void* context, const std::int32_t* x);

// Points of a sequence's region that are not in a visited set yet, and that
// a filter accepts.
//
// take() draws from the sequence in order, skips the points already visited
// and adds the ones it returns. A rank sequence covers its region without
// repeats, so it runs out exactly when every point of the region is visited
// (for regions under 2^64 points). Coordinate and iid sequences can repeat
// points or miss some: after `patience` visited points in a row within one
// call, the sampler moves for good to the unscrambled rank sequence of the
// same region, which lists whatever is left; rejected points count towards
// the run too. Rejected points are not added to the set. Safe to call from
// many threads at once; they share the set and the sequences' positions.
class fresh_sampler {
  public:
    // Throws std::invalid_argument if the set belongs to another space.
    fresh_sampler(const sequence& source, visited_set& visited, std::uint64_t patience = 1024);
    // With a filter; accept may not be null.
    fresh_sampler(const sequence& source, visited_set& visited, accept_fn accept, void* context,
                  std::uint64_t patience = 1024);

    // Writes up to count points that were not visited, in parts() values per
    // point, adds them to the set and returns how many. Fewer than count means
    // the region has none left.
    std::uint64_t take(std::uint64_t count, std::int32_t* out, executor* ex = nullptr);

    // take() found nothing left to draw: every point of the region was handed
    // out, found visited, or is being checked by another thread's take().
    bool exhausted() const noexcept;
    // Moved from the given sequence to the region's rank sequence.
    bool switched() const noexcept;
    // Visited points passed over so far.
    std::uint64_t skipped() const noexcept;
    // Points handed out, and points the filter rejected, so far. Their ratio
    // is the acceptance of the region.
    std::uint64_t accepted() const noexcept;
    std::uint64_t rejected() const noexcept;

  private:
    static sequence fallback_for(const sequence& source);

    sequence source_;
    sequence fallback_;
    visited_set& visited_;
    std::uint64_t patience_;
    accept_fn accept_ = nullptr;
    void* context_ = nullptr;
    std::atomic<bool> switched_{false};
    std::atomic<bool> exhausted_{false};
    std::atomic<std::uint64_t> skipped_{0};
    std::atomic<std::uint64_t> accepted_{0};
    std::atomic<std::uint64_t> rejected_{0};
};

}  // namespace bounded_compositions

#endif
