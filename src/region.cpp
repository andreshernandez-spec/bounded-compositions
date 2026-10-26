#include "bounded_compositions/region.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "bounded_compositions/kernel/walk.hpp"
#include "counts.hpp"

namespace bounded_compositions {

namespace k = kernel;

namespace {

using detail::below_of;
using detail::completions_of;

struct prefix_place {
    natural begin;       // rank of the first composition with the prefix
    natural size;        // compositions with the prefix
    std::int32_t left;   // shifted total left for the parts after the prefix
    std::int32_t layer;  // layer after the prefix (kernel::space_view)
};

// Throws std::invalid_argument if the prefix is not the start of some composition.
void require_exact(const space& s, const char* what) {
    if (s.counting_mode() == counting::logarithmic) {
        throw std::logic_error(std::string("bounded_compositions::") + what +
                               ": regions need exact counting; this space counts in logarithms");
    }
}

prefix_place place_prefix(const space& s, const std::vector<std::int32_t>& prefix) {
    require_exact(s, "subtree");
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
    out.layer = v.layer_base;
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        const int part = static_cast<int>(i);
        const std::int64_t y = static_cast<std::int64_t>(prefix[i]) - v.lo[i];
        const std::int32_t next =
            y < 0 || y > v.cap[i] || y > out.left
                ? -1
                : k::layer_after(v, part, out.left, out.layer, static_cast<std::int32_t>(y));
        if (next < 0) {
            throw std::invalid_argument("bounded_compositions::subtree: prefix value " +
                                        std::to_string(prefix[i]) + " at part " +
                                        std::to_string(i) + " is out of range");
        }
        const std::int32_t yy = static_cast<std::int32_t>(y);
        out.begin += below_of(v, part, out.left, out.layer, yy);
        out.left -= yy;
        out.layer = next;
    }
    out.size = completions_of(v, static_cast<int>(prefix.size()), out.layer, out.left);
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
    require_exact(s, "interval");
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
    const int part = static_cast<int>(d);
    // Values in increasing order are children in rank order.
    natural begin = r.begin();
    for (int g = 0; g < v.segs[part]; ++g) {
        std::int32_t first = 0, last = 0, next = 0;
        if (!k::move(v, part, g, p.left, p.layer, &first, &last, &next)) continue;
        for (std::int32_t y = first; y <= last; ++y) {
            const natural size = completions_of(v, part + 1, next, p.left - y);
            if (size.is_zero()) continue;
            out.push_back(interval(s, begin, begin + size));
            begin += size;
        }
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
    // rho: bits where the first and last ranks differ, so [a 2^rho, (a+1) 2^rho)
    // is the smallest aligned block holding r.
    const natural last = r.end() - natural(1);
    const std::size_t words = std::max(r.begin().limb_count(), last.limb_count());
    std::vector<k::limb> diff(words, 0);
    for (std::size_t i = 0; i < words; ++i) {
        diff[i] =
            k::limb_at(r.begin().limbs(), static_cast<int>(r.begin().limb_count()),
                       static_cast<int>(i)) ^
            k::limb_at(last.limbs(), static_cast<int>(last.limb_count()), static_cast<int>(i));
    }
    const int rho = k::limbs_bit_length(diff.data(), static_cast<int>(words));
    const int width = rho - std::min(levels, rho);  // blocks of 2^width ranks
    // The aligned block holding r's first rank: clear its low `width` bits.
    std::vector<k::limb> start(r.begin().limbs(), r.begin().limbs() + r.begin().limb_count());
    for (std::size_t i = 0; i < start.size(); ++i) {
        const int below = width - static_cast<int>(i) * 64;
        if (below >= 64) {
            start[i] = 0;
        } else if (below > 0) {
            start[i] &= ~((k::limb{1} << below) - 1);
        }
    }
    natural block(1);
    for (int b = 0; b < width; ++b) block.mul_small(2);
    std::vector<region> out;
    for (natural lo = natural::from_limbs(start.data(), start.size()); lo < r.end(); lo += block) {
        const natural from = std::max(lo, r.begin());
        const natural to = std::min(lo + block, r.end());
        out.push_back(interval(s, from, to));
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

    const prefix_place p = place_prefix(s, shared);
    if (v.plain != 0) {
        // Inside the subtree of the shared prefix, part i takes every value
        // that leaves the other free parts a feasible total.
        std::int64_t free_caps = 0;
        for (std::size_t i = d; i < n; ++i) free_caps += v.cap[i];
        for (std::size_t i = d; i < n; ++i) {
            const std::int64_t others = free_caps - v.cap[i];
            const std::int64_t lo = std::max<std::int64_t>(0, p.left - others);
            const std::int64_t hi = std::min<std::int64_t>(v.cap[i], p.left);
            out[i] = {static_cast<std::int32_t>(lo + v.lo[i]),
                      static_cast<std::int32_t>(hi + v.lo[i])};
        }
    } else {
        // Walk the states (k left, layer) reachable from the prefix that
        // still have a completion, part by part, keeping the values used.
        const std::size_t width = static_cast<std::size_t>(p.left) + 1;
        std::vector<char> reach(width * static_cast<std::size_t>(v.row_layers[d]), char{0});
        reach[static_cast<std::size_t>(p.layer) * width + static_cast<std::size_t>(p.left)] = 1;
        for (std::size_t i = d; i < n; ++i) {
            const int part = static_cast<int>(i);
            const std::size_t layers = static_cast<std::size_t>(v.row_layers[i]);
            std::vector<char> next(width * static_cast<std::size_t>(v.row_layers[i + 1]), char{0});
            std::int32_t lo = std::numeric_limits<std::int32_t>::max();
            std::int32_t hi = std::numeric_limits<std::int32_t>::min();
            for (std::size_t st = 0; st < width * layers; ++st) {
                if (reach[st] == 0) continue;
                const std::int32_t c = static_cast<std::int32_t>(st / width);
                const std::int32_t kk = static_cast<std::int32_t>(st % width);
                for (int g = 0; g < v.segs[part]; ++g) {
                    std::int32_t first = 0, last = 0, cn = 0;
                    if (!k::move(v, part, g, kk, c, &first, &last, &cn)) continue;
                    for (std::int32_t y = first; y <= last; ++y) {
                        if (!k::has_completion(v, part + 1, cn, kk - y)) continue;
                        lo = std::min(lo, y);
                        hi = std::max(hi, y);
                        next[static_cast<std::size_t>(cn) * width +
                             static_cast<std::size_t>(kk - y)] = 1;
                    }
                }
            }
            out[i] = {lo + v.lo[i], hi + v.lo[i]};
            reach.swap(next);
        }
    }
    if (!r.is_subtree()) {
        // Lexicographic order: part d runs from the first composition's value
        // to the last one's.
        out[d] = {s.unrank(r.begin())[d], s.unrank(r.end() - natural(1))[d]};
    }
    return out;
}

}  // namespace bounded_compositions
