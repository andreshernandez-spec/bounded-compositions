#include "bounded_compositions/region.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "bounded_compositions/kernel/walk.hpp"

namespace bounded_compositions {

namespace k = kernel;

namespace {

// P[row][kk], zero for kk < 0.
natural prefix_sum(const k::space_view& v, int row, std::int32_t kk) {
    if (kk < 0) return natural();
    return natural::from_limbs(k::table_entry(v, row, kk),
                               static_cast<std::size_t>(v.row_limbs[row]));
}

// N[row][kk]: ways to place kk into parts row..n-1.
natural count(const k::space_view& v, int row, std::int32_t kk) {
    if (kk < 0) return natural();
    return prefix_sum(v, row, kk) - prefix_sum(v, row, kk - 1);
}

struct prefix_place {
    natural begin;      // rank of the first composition with the prefix
    natural size;       // compositions with the prefix
    std::int32_t left;  // shifted total left for the parts after the prefix
};

// Throws std::invalid_argument if the prefix is not the start of some composition.
prefix_place place_prefix(const space& s, const std::vector<std::int32_t>& prefix) {
    const k::space_view v = s.view();
    const std::size_t n = s.parts();
    if (prefix.size() > n) {
        throw std::invalid_argument("bounded_compositions::subtree: prefix longer than the space");
    }
    if (s.empty()) {
        throw std::invalid_argument("bounded_compositions::subtree: the space is empty");
    }
    prefix_place out;
    out.left = v.total;
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        const std::int64_t y = static_cast<std::int64_t>(prefix[i]) - v.lo[i];
        if (y < 0 || y > v.cap[i] || y > out.left) {
            throw std::invalid_argument("bounded_compositions::subtree: prefix value " +
                                        std::to_string(prefix[i]) + " at part " +
                                        std::to_string(i) + " is out of range");
        }
        const int row = static_cast<int>(i) + 1;
        const std::int32_t yy = static_cast<std::int32_t>(y);
        out.begin += prefix_sum(v, row, out.left) - prefix_sum(v, row, out.left - yy);
        out.left -= yy;
    }
    out.size = count(v, static_cast<int>(prefix.size()), out.left);
    if (out.size.is_zero()) {
        throw std::invalid_argument("bounded_compositions::subtree: prefix has no completion");
    }
    return out;
}

std::vector<std::int32_t> common_prefix(const std::vector<std::int32_t>& a,
                                        const std::vector<std::int32_t>& b) {
    std::size_t d = 0;
    while (d < a.size() && a[d] == b[d]) ++d;
    return std::vector<std::int32_t>(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(d));
}

}  // namespace

region::region(natural begin, natural end, std::vector<std::int32_t> prefix, bool subtree)
    : begin_(std::move(begin)),
      end_(std::move(end)),
      prefix_(std::move(prefix)),
      subtree_(subtree) {}

region interval(const space& s, const natural& begin, const natural& end) {
    if (begin > end || end > s.size()) {
        throw std::out_of_range("bounded_compositions::interval: [" + begin.to_string() + ", " +
                                end.to_string() + ") is not within [0, " + s.size().to_string() +
                                ")");
    }
    if (begin == end) return region(begin, end, {}, false);
    std::vector<std::int32_t> first = s.unrank(begin);
    const std::vector<std::int32_t> last = s.unrank(end - natural(1));
    std::vector<std::int32_t> shared = common_prefix(first, last);
    // The smallest subtree holding the interval is the one of the shared prefix.
    const prefix_place p = place_prefix(s, shared);
    const bool subtree = p.begin == begin && p.begin + p.size == end;
    return region(begin, end, std::move(shared), subtree);
}

region whole(const space& s) { return interval(s, natural(), s.size()); }

region subtree(const space& s, const std::vector<std::int32_t>& prefix) {
    const prefix_place p = place_prefix(s, prefix);
    region r = interval(s, p.begin, p.begin + p.size);
    r.subtree_ = true;  // the shared prefix extends `prefix`, with the same ranks
    return r;
}

std::vector<region> children(const space& s, const region& r) {
    if (r.empty()) return {};
    if (!r.is_subtree()) {
        throw std::invalid_argument("bounded_compositions::children: region is not a subtree");
    }
    std::vector<region> out;
    const std::size_t d = r.prefix().size();
    if (d == s.parts()) return out;  // a single composition
    const k::space_view v = s.view();
    const prefix_place p = place_prefix(s, r.prefix());
    const int row = static_cast<int>(d) + 1;
    const std::int32_t top = std::min(v.cap[d], p.left);
    const natural all_below = prefix_sum(v, row, p.left);
    for (std::int32_t y = 0; y <= top; ++y) {
        const natural size = count(v, row, p.left - y);
        if (size.is_zero()) continue;
        const natural begin = r.begin() + (all_below - prefix_sum(v, row, p.left - y));
        out.push_back(interval(s, begin, begin + size));
    }
    return out;
}

std::vector<region> split(const space& s, const region& r, std::uint32_t pieces) {
    if (pieces == 0) throw std::invalid_argument("bounded_compositions::split: zero pieces");
    const natural size = r.size();
    std::vector<region> out;
    natural lo = r.begin();
    for (std::uint32_t i = 1; i <= pieces; ++i) {
        natural offset = size;
        offset.mul_small(i);
        offset.divmod_small(pieces);  // floor(i * size / pieces)
        const natural hi = r.begin() + offset;
        if (hi != lo) out.push_back(interval(s, lo, hi));
        lo = hi;
    }
    return out;
}

std::vector<region> dyadic_split(const space& s, const region& r, int levels) {
    if (levels < 0 || levels > 24) {
        throw std::invalid_argument("bounded_compositions::dyadic_split: levels must be 0..24");
    }
    if (r.empty()) return {};
    const natural size = r.size();
    const int m = std::max(1, (size - natural(1)).bit_length());
    const int used = std::min(levels, m);
    natural block(1);
    for (int b = 0; b < m - used; ++b) block.mul_small(2);
    std::vector<region> out;
    for (std::uint32_t t = 0; t < (std::uint32_t{1} << used); ++t) {
        natural lo = block;
        lo.mul_small(t);
        if (lo >= size) break;
        natural hi = lo + block;
        if (hi > size) hi = size;
        out.push_back(interval(s, r.begin() + lo, r.begin() + hi));
    }
    return out;
}

std::vector<part_range> describe(const space& s, const region& r) {
    if (r.empty()) return {};
    const k::space_view v = s.view();
    const std::size_t n = s.parts();
    const std::vector<std::int32_t>& shared = r.prefix();
    const std::size_t d = shared.size();
    std::vector<part_range> out(n);
    for (std::size_t i = 0; i < d; ++i) out[i] = {shared[i], shared[i]};
    if (d == n) return out;

    // Inside the subtree of the shared prefix, part i takes every value that
    // leaves the other free parts a feasible total.
    const prefix_place p = place_prefix(s, shared);
    std::int64_t free_caps = 0;
    for (std::size_t i = d; i < n; ++i) free_caps += v.cap[i];
    for (std::size_t i = d; i < n; ++i) {
        const std::int64_t others = free_caps - v.cap[i];
        const std::int64_t lo = std::max<std::int64_t>(0, p.left - others);
        const std::int64_t hi = std::min<std::int64_t>(v.cap[i], p.left);
        out[i] = {static_cast<std::int32_t>(lo + v.lo[i]), static_cast<std::int32_t>(hi + v.lo[i])};
    }
    if (!r.is_subtree()) {
        // Lexicographic order: part d runs from the first composition's value
        // to the last one's, with every value between them present.
        out[d] = {s.unrank(r.begin())[d], s.unrank(r.end() - natural(1))[d]};
    }
    return out;
}

}  // namespace bounded_compositions
