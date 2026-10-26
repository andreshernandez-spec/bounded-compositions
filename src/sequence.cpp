#include "bounded_compositions/sequence.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "bounded_compositions/kernel/philox.hpp"
#include "bounded_compositions/kernel/radical.hpp"
#include "bounded_compositions/kernel/walk.hpp"
#include "sobol.hpp"

namespace bounded_compositions {

namespace k = kernel;

static_assert(std::is_trivially_copyable_v<k::radical_view>,
              "radical_view must stay copyable to a device as is");

namespace {

// Domain tags for the second Philox key word, so the scrambling mask and the
// i.i.d. draws never share a stream for the same seed.
constexpr k::limb tag_iid = 0x6969642d73657131u;   // "iid-seq1"
constexpr k::limb tag_mask = 0x6d61736b2d736571u;  // "mask-seq"

constexpr std::uint64_t block_points = 1024;

int as_int(std::size_t n) { return static_cast<int>(n); }

// floor(a / 2^s) as limbs.
std::vector<k::limb> shift_right(const k::limb* a, std::size_t n, int s) {
    const std::size_t word = static_cast<std::size_t>(s / 64);
    const int bit = s % 64;
    std::vector<k::limb> out(n > word ? n - word : 0, 0);
    for (std::size_t i = 0; i < out.size(); ++i) {
        k::limb lo = a[i + word] >> bit;
        if (bit != 0 && i + word + 1 < n) lo |= a[i + word + 1] << (64 - bit);
        out[i] = lo;
    }
    return out;
}

// a mod 2^s as limbs.
std::vector<k::limb> low_bits(const k::limb* a, std::size_t n, int s) {
    const std::size_t words = static_cast<std::size_t>((s + 63) / 64);
    std::vector<k::limb> out(words, 0);
    for (std::size_t i = 0; i < words && i < n; ++i) out[i] = a[i];
    if (s % 64 != 0 && words > 0) out[words - 1] &= (k::limb{1} << (s % 64)) - 1;
    return out;
}

}  // namespace

namespace detail {

struct sequence_data {
    space dom;
    sequence_kind kind = sequence_kind::rank;
    std::uint64_t seed = 0;
    std::uint64_t capacity = 0;
    int rank_limbs = 1;
    k::radical_view radical{};
    std::vector<k::limb> mask_lo;
    k::limb key[2] = {0, 0};
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

    // Rank of point i into out[0..rank_limbs). For `rank`, t is the raw index
    // of point i, already found by the caller.
    void rank_of(std::uint64_t i, std::uint64_t t, k::limb* out) const {
        const natural& size = span;
        switch (kind) {
            case sequence_kind::rank:
                k::radical_rank(radical, k::radical_x(radical, t), out, rank_limbs);
                break;
            case sequence_kind::scaled_rank: {
                std::vector<k::limb> wide(static_cast<std::size_t>(rank_limbs) + 1, 0);
                k::limbs_copy(size.limbs(), as_int(size.limb_count()), wide.data(), rank_limbs);
                k::limbs_mul_add_word(wide.data(), rank_limbs + 1, k::bit_reverse(i, 64), 0);
                k::limbs_copy(wide.data() + 1, rank_limbs, out, rank_limbs);  // drop 2^64
                break;
            }
            case sequence_kind::iid:
                if (k::uniform_below(size.limbs(), as_int(size.limb_count()), key, i, out) !=
                    k::status::ok) {
                    throw std::logic_error("bounded_compositions: uniform draw failed");
                }
                break;
            case sequence_kind::coordinate:
                // Coordinate points are decoded, not unranked.
                throw std::logic_error("bounded_compositions: rank_of on a coordinate sequence");
        }
        // Ranks above are within the region; shift them to the space.
        k::limbs_add(out, rank_limbs, base.limbs(), as_int(base.limb_count()), out, rank_limbs);
    }

    void write_decoded_points(std::uint64_t first, std::uint64_t count, std::int32_t* out) const {
        const k::space_view wv = walk.view();
        const k::sobol_view sv = sobol.view();
        const std::size_t n = dom.parts();
        const std::size_t d = prefix.size();
        const std::size_t dims = walk.parts() - 1;
        std::vector<k::limb> u(dims + 4);
        for (std::uint64_t p = 0; p < count; ++p) {
            std::int32_t* x = out + p * n;
            std::copy(prefix.begin(), prefix.end(), x);
            if (kind == sequence_kind::coordinate) {
                k::sobol_point(sv, first + p, u.data());
            } else {  // iid over a logarithmic space: Philox fractions
                for (std::size_t w = 0; w < dims; w += 4) {
                    const k::limb counter[4] = {first + p, 0, static_cast<k::limb>(w / 4), 0};
                    const k::philox_block b = k::philox4x64_10(counter, key);
                    for (std::size_t j = 0; j < 4; ++j) u[w + j] = b.v[j];
                }
            }
            if (k::decode(wv, u.data(), x + d) != k::status::ok) {
                throw std::logic_error("bounded_compositions: decode failed in a sequence");
            }
        }
    }

    void write_points(std::uint64_t first, std::uint64_t count, std::int32_t* out) const {
        if (decoded) {
            write_decoded_points(first, count, out);
            return;
        }
        const k::space_view sv = dom.view();
        const std::size_t n = dom.parts();
        std::vector<k::limb> rank(static_cast<std::size_t>(rank_limbs));
        std::uint64_t t = kind == sequence_kind::rank ? k::radical_raw_index(radical, first) : 0;
        for (std::uint64_t p = 0; p < count; ++p) {
            if (kind == sequence_kind::rank) {
                while (!k::radical_keeps(radical, t)) ++t;
            }
            rank_of(first + p, t, rank.data());
            if (k::unrank(sv, rank.data(), rank_limbs, out + p * n) != k::status::ok) {
                throw std::logic_error(
                    "bounded_compositions: sequence produced a rank out of range");
            }
            ++t;
        }
    }
};

}  // namespace detail

namespace {

// r == nullptr means the whole space.
std::shared_ptr<detail::sequence_data> base_data(const space& s, const region* r,
                                                 sequence_kind kind, std::uint64_t seed) {
    if (s.counting_mode() == counting::logarithmic) {
        // No exact counts: only point-by-point decoding over the whole space.
        if (kind == sequence_kind::rank || kind == sequence_kind::scaled_rank || r != nullptr) {
            throw std::logic_error(
                "bounded_compositions::sequence: rank sequences and regions need exact "
                "counting; a logarithmic space supports coordinate and iid sequences");
        }
        auto d = std::make_shared<detail::sequence_data>(s);
        d->kind = kind;
        d->seed = seed;
        d->decoded = true;
        d->span = natural(s.empty() ? 0 : 1);  // only its emptiness is used
        d->capacity = s.empty() ? 0 : std::numeric_limits<std::uint64_t>::max();
        return d;
    }
    if (r != nullptr && r->end() > s.size()) {
        throw std::out_of_range("bounded_compositions::sequence: region ends at rank " +
                                r->end().to_string() + ", past the size " + s.size().to_string());
    }
    auto d = std::make_shared<detail::sequence_data>(s);
    d->kind = kind;
    d->seed = seed;
    d->rank_limbs = std::max<int>(1, as_int(s.size().limb_count()));
    d->base = r != nullptr ? r->begin() : natural();
    d->span = r != nullptr ? r->size() : s.size();
    if (d->span.is_zero()) {
        d->capacity = 0;
    } else if (kind == sequence_kind::rank) {
        d->capacity = 0;  // set by set_radical
    } else {
        d->capacity = std::numeric_limits<std::uint64_t>::max();
    }
    return d;
}

// Fills the radical view for ranks [0, size) with the given m-bit mask.
void set_radical(detail::sequence_data& d, const std::vector<k::limb>& mask) {
    const natural& size = d.span;
    if (size.is_zero()) return;
    const natural last = size - natural(1);
    const int m = std::max(1, last.bit_length());
    const int w = std::min(m, 64);
    const int shift = m - w;

    const std::vector<k::limb> size_hi = shift_right(size.limbs(), size.limb_count(), shift);
    const std::vector<k::limb> size_lo = low_bits(size.limbs(), size.limb_count(), shift);
    const std::vector<k::limb> mask_hi = shift_right(mask.data(), mask.size(), shift);
    d.mask_lo = low_bits(mask.data(), mask.size(), shift);

    // threshold = size_hi + (mask_lo < size_lo); all raw indices are kept
    // when it reaches 2^w.
    const bool bump = k::limbs_cmp(d.mask_lo.data(), as_int(d.mask_lo.size()), size_lo.data(),
                                   as_int(size_lo.size())) < 0;
    std::vector<k::limb> threshold(size_hi.size() + 1, 0);
    const k::limb one[1] = {1};
    k::limbs_add(size_hi.data(), as_int(size_hi.size()), one, bump ? 1 : 0, threshold.data(),
                 as_int(threshold.size()));
    const int threshold_bits = k::limbs_bit_length(threshold.data(), as_int(threshold.size()));

    k::radical_view& v = d.radical;
    v.index_bits = w;
    v.shift = shift;
    v.mask_hi = mask_hi.empty() ? 0 : mask_hi[0];
    if (w < 64) v.mask_hi &= (k::limb{1} << w) - 1;
    v.mask_lo = d.mask_lo.data();
    v.mask_lo_limbs = as_int(d.mask_lo.size());
    v.accept_all = threshold_bits > w ? 1 : 0;  // threshold >= 2^w
    v.threshold = v.accept_all ? 0 : threshold[0];
    if (v.accept_all) {
        v.capacity = w == 64 ? std::numeric_limits<std::uint64_t>::max() : (std::uint64_t{1} << w);
    } else {
        v.capacity = v.threshold;
    }
    d.capacity = v.capacity;
}

std::vector<k::limb> mask_bits(std::uint64_t seed, int bits) {
    const k::limb key[2] = {seed, tag_mask};
    std::vector<k::limb> mask(static_cast<std::size_t>((bits + 63) / 64), 0);
    for (std::size_t w = 0; w < mask.size(); w += 4) {
        const k::limb counter[4] = {static_cast<k::limb>(w / 4), 0, 0, 0};
        const k::philox_block b = k::philox4x64_10(counter, key);
        for (std::size_t j = 0; j < 4 && w + j < mask.size(); ++j) mask[w + j] = b.v[j];
    }
    if (bits % 64 != 0) mask.back() &= (k::limb{1} << (bits % 64)) - 1;
    return mask;
}

struct points_job {
    const detail::sequence_data* data;
    std::uint64_t begin;
    std::uint64_t count;
    std::int32_t* out;
};

void points_block(void* context, std::uint64_t block) {
    const points_job& job = *static_cast<const points_job*>(context);
    const std::uint64_t offset = block * block_points;
    const std::uint64_t n = std::min(block_points, job.count - offset);
    job.data->write_points(job.begin + offset, n, job.out + offset * job.data->dom.parts());
}

void check_range(std::uint64_t begin, std::uint64_t count, std::uint64_t capacity) {
    if (begin > capacity || count > capacity - begin) {
        throw std::out_of_range("bounded_compositions::sequence: points [" + std::to_string(begin) +
                                ", " + std::to_string(begin) + " + " + std::to_string(count) +
                                ") exceed the capacity " + std::to_string(capacity));
    }
}

}  // namespace

sequence::sequence(std::shared_ptr<const detail::sequence_data> data) : data_(std::move(data)) {}

sequence::sequence(const sequence& other)
    : data_(other.data_), cursor_(other.cursor_.load(std::memory_order_relaxed)) {}

sequence& sequence::operator=(const sequence& other) {
    if (this != &other) {
        data_ = other.data_;
        cursor_.store(other.cursor_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    return *this;
}

const space& sequence::domain() const noexcept { return data_->dom; }
sequence_kind sequence::kind() const noexcept { return data_->kind; }
std::uint64_t sequence::seed() const noexcept { return data_->seed; }
std::uint64_t sequence::capacity() const noexcept { return data_->capacity; }
std::uint64_t sequence::position() const noexcept {
    return cursor_.load(std::memory_order_relaxed);
}

void sequence::seek(std::uint64_t position) {
    check_range(position, 0, data_->capacity);
    cursor_.store(position, std::memory_order_relaxed);
}

void sequence::points(std::uint64_t begin, std::uint64_t count, std::int32_t* out,
                      executor* ex) const {
    check_range(begin, count, data_->capacity);
    if (count == 0) return;
    points_job job{data_.get(), begin, count, out};
    const std::uint64_t blocks = (count + block_points - 1) / block_points;
    serial_executor serial;
    (ex != nullptr ? *ex : static_cast<executor&>(serial)).run(blocks, &points_block, &job);
}

natural sequence::rank_at(std::uint64_t i) const {
    check_range(i, 1, data_->capacity);
    if (data_->decoded) {
        std::vector<std::int32_t> x(data_->dom.parts());
        data_->write_decoded_points(i, 1, x.data());
        return data_->dom.rank(x.data());  // throws for a logarithmic space
    }
    std::vector<k::limb> rank(static_cast<std::size_t>(data_->rank_limbs));
    const std::uint64_t t =
        data_->kind == sequence_kind::rank ? k::radical_raw_index(data_->radical, i) : 0;
    data_->rank_of(i, t, rank.data());
    return natural::from_limbs(rank.data(), rank.size());
}

std::uint64_t sequence::take(std::uint64_t count, std::int32_t* out, executor* ex) {
    std::uint64_t begin = cursor_.load(std::memory_order_relaxed);
    do {
        check_range(begin, count, data_->capacity);
    } while (!cursor_.compare_exchange_weak(begin, begin + count, std::memory_order_relaxed));
    points(begin, count, out, ex);
    return begin;
}

namespace {

std::shared_ptr<detail::sequence_data> make_rank(const space& s, const region* r, bool scramble,
                                                 std::uint64_t seed) {
    auto d = base_data(s, r, sequence_kind::rank, scramble ? seed : 0);
    if (!d->span.is_zero()) {
        const int m = std::max(1, (d->span - natural(1)).bit_length());
        set_radical(*d, scramble
                            ? mask_bits(seed, m)
                            : std::vector<k::limb>(static_cast<std::size_t>((m + 63) / 64), 0));
    }
    return d;
}

std::shared_ptr<detail::sequence_data> make_iid(const space& s, const region* r,
                                                std::uint64_t seed) {
    auto d = base_data(s, r, sequence_kind::iid, seed);
    d->key[0] = seed;
    d->key[1] = tag_iid;
    return d;
}

}  // namespace

sequence rank_sequence(const space& s) { return sequence(make_rank(s, nullptr, false, 0)); }

sequence rank_sequence(const space& s, std::uint64_t scramble_seed) {
    return sequence(make_rank(s, nullptr, true, scramble_seed));
}

sequence rank_sequence(const space& s, const region& r) {
    return sequence(make_rank(s, &r, false, 0));
}

sequence rank_sequence(const space& s, const region& r, std::uint64_t scramble_seed) {
    return sequence(make_rank(s, &r, true, scramble_seed));
}

sequence scaled_rank_sequence(const space& s) {
    return sequence(base_data(s, nullptr, sequence_kind::scaled_rank, 0));
}

sequence iid_sequence(const space& s, std::uint64_t seed) {
    return sequence(make_iid(s, nullptr, seed));
}

sequence iid_sequence(const space& s, const region& r, std::uint64_t seed) {
    return sequence(make_iid(s, &r, seed));
}

namespace {

std::shared_ptr<detail::sequence_data> make_coordinate(const space& s, const region* r,
                                                       bool scramble, std::uint64_t seed) {
    if (r != nullptr && !r->empty() && !r->is_subtree()) {
        throw std::invalid_argument(
            "bounded_compositions::coordinate_sequence: the region must be a subtree");
    }
    auto d = base_data(s, r, sequence_kind::coordinate, scramble ? seed : 0);
    d->decoded = true;
    if (d->span.is_zero()) return d;
    if (r != nullptr && !r->prefix().empty()) {
        d->prefix = r->prefix();
        if (d->prefix.size() == s.parts()) {
            // A single composition: every point is the prefix itself.
            d->prefix.pop_back();
        }
        d->walk = s.subspace(d->prefix);
    }
    d->sobol = detail::make_sobol(static_cast<int>(d->walk.parts()) - 1, scramble, seed);
    return d;
}

}  // namespace

sequence coordinate_sequence(const space& s) {
    return sequence(make_coordinate(s, nullptr, false, 0));
}

sequence coordinate_sequence(const space& s, std::uint64_t scramble_seed) {
    return sequence(make_coordinate(s, nullptr, true, scramble_seed));
}

sequence coordinate_sequence(const space& s, const region& subtree) {
    return sequence(make_coordinate(s, &subtree, false, 0));
}

sequence coordinate_sequence(const space& s, const region& subtree, std::uint64_t scramble_seed) {
    return sequence(make_coordinate(s, &subtree, true, scramble_seed));
}

}  // namespace bounded_compositions
