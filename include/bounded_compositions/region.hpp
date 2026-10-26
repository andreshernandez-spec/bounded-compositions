// Regions: rank intervals of a space, and the subtrees of its rank tree.
//
// Lexicographic rank is a tree: the compositions sharing a prefix (fixed
// values for the first j parts) form a contiguous rank interval, a subtree.
// Any region can be sampled on its own with a region sequence (sequence.hpp).
#ifndef BCOMP_REGION_HPP
#define BCOMP_REGION_HPP

#include <cstdint>
#include <vector>

#include "bounded_compositions/natural.hpp"
#include "bounded_compositions/space.hpp"

namespace bounded_compositions {

class region {
  public:
    // First rank, one past the last rank, and their difference.
    const natural& begin() const noexcept { return begin_; }
    const natural& end() const noexcept { return end_; }
    natural size() const { return end_ - begin_; }
    bool empty() const noexcept { return begin_ == end_; }
    bool contains(const natural& rank) const noexcept { return begin_ <= rank && rank < end_; }

    // The longest prefix shared by every composition in the region (empty
    // for an empty region). It can be longer than the prefix a subtree was
    // created with, when the next parts are forced.
    const std::vector<std::int32_t>& prefix() const noexcept { return prefix_; }
    // True if the region is exactly the compositions that start with prefix().
    bool is_subtree() const noexcept { return subtree_; }

    // Regions are equal when they cover the same ranks; prefix and subtree
    // flag follow from the ranks.
    friend bool operator==(const region& a, const region& b) {
        return a.begin_ == b.begin_ && a.end_ == b.end_;
    }
    friend bool operator!=(const region& a, const region& b) { return !(a == b); }

  private:
    region(natural begin, natural end, std::vector<std::int32_t> prefix, bool subtree);

    natural begin_;
    natural end_;
    std::vector<std::int32_t> prefix_;
    bool subtree_ = false;

    friend region whole(const space& s);
    friend region subtree(const space& s, const std::vector<std::int32_t>& prefix);
    friend region interval(const space& s, const natural& begin, const natural& end);
    friend std::vector<region> children(const space& s, const region& r);
};

// The whole space: the subtree with an empty prefix.
region whole(const space& s);

// The compositions whose first prefix.size() parts equal prefix (up to all
// parts). Throws std::invalid_argument if the prefix is longer than the
// space, breaks a bound, or has no completion.
region subtree(const space& s, const std::vector<std::int32_t>& prefix);

// Ranks [begin, end). The result is marked as a subtree when it is one.
// Throws std::out_of_range unless begin <= end <= s.size().
region interval(const space& s, const natural& begin, const natural& end);

// The subtrees one level below a subtree, one for each value of the first
// part after its prefix that has completions, in rank order. Together they
// partition r, and there are at least two unless r is a single composition,
// which has none. An empty region (the whole of an empty space) has none
// either. Throws std::invalid_argument if r is a non-empty interval that is not
// a subtree.
std::vector<region> children(const space& s, const region& r);

// Splits r into `pieces` consecutive intervals whose sizes differ by at most
// one (empty pieces are dropped when r has fewer ranks than pieces).
std::vector<region> split(const space& s, const region& r, std::uint32_t pieces);

// Splits r into the 2^levels aligned blocks of its radical-inverse order:
// with m = max(1, bitlen(size - 1)), block t holds ranks
// begin + [t 2^(m-levels), (t+1) 2^(m-levels)), clipped to r, and empty blocks
// are dropped. A rank_sequence over each block produces exactly the points of
// rank_sequence over r that fall in that block, in the same order, so equal
// budgets per block reproduce the sequence of r. For regions wider than 64
// bits this holds for each block's first 2^(64 - levels) points; after them a
// block's own sequence continues on a finer grid than r's. levels is capped
// at m; throws std::invalid_argument for levels > 24.
std::vector<region> dyadic_split(const space& s, const region& r, int levels);

struct part_range {
    std::int32_t lo;
    std::int32_t hi;
};

// For each part, the smallest and largest value it takes in r. Exact for
// subtrees. For other intervals the parts up to the first one where the first
// and last compositions differ are exact; later parts get the range of the
// smallest subtree containing r, a superset.
std::vector<part_range> describe(const space& s, const region& r);

}  // namespace bounded_compositions

#endif
