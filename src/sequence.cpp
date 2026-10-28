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
#include "sequence_data.hpp"

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

void sequence_data::rank_of(std::uint64_t i, k::limb* out) const {
    const natural& size = span;
    switch (kind) {
        case sequence_kind::rank:
            // Absolute already: the view's origin is a rank of the space.
            k::radical_rank(radical, k::radical_select(radical, i), out, rank_limbs);
            return;
        case sequence_kind::scaled_rank: {
            std::vector<k::limb> wide(static_cast<std::size_t>(rank_limbs) + 1, 0);
            k::limbs_copy(size.limbs(), as_int(size.limb_count()), wide.data(), rank_limbs);
            k::limbs_mul_add_word(wide.data(), rank_limbs + 1, k::bit_reverse(i, 64), 0);
            k::limbs_copy(wide.data() + 1, rank_limbs, out, rank_limbs);  // drop 2^64
            break;
        }
        case sequence_kind::iid:
            // uniform_below writes only the span's limbs; a region narrower than
            // the space would keep the last point's high limbs.
            k::limbs_zero(out, rank_limbs);
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

void sequence_data::write_decoded_points(std::uint64_t first, std::uint64_t count,
                                         std::int32_t* out) const {
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

void sequence_data::write_points(std::uint64_t first, std::uint64_t count,
                                 std::int32_t* out) const {
    if (decoded) {
        write_decoded_points(first, count, out);
        return;
    }
    const k::space_view sv = dom.view();
    const std::size_t n = dom.parts();
    std::vector<k::limb> rank(static_cast<std::size_t>(rank_limbs));
    for (std::uint64_t p = 0; p < count; ++p) {
        rank_of(first + p, rank.data());
        if (k::unrank(sv, rank.data(), rank_limbs, out + p * n) != k::status::ok) {
            throw std::logic_error("bounded_compositions: sequence produced a rank out of range");
        }
    }
}

bool same_space(const space& a, const space& b) {
    return a.total() == b.total() && a.lower() == b.lower() && a.upper() == b.upper() &&
           a.support() == b.support() && a.groups() == b.groups();
}

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

natural from_vector(const std::vector<k::limb>& a) {
    return natural::from_limbs(a.data(), a.size());
}

// 2^bits.
natural power_of_two(int bits) {
    std::vector<k::limb> a(static_cast<std::size_t>(bits / 64) + 1, 0);
    a.back() = k::limb{1} << (bits % 64);
    return from_vector(a);
}

k::limb low_word(const std::vector<k::limb>& a) { return a.empty() ? 0 : a[0]; }

// Fills the view of the region [base, base + span) in the order of d.mask
// (kernel/radical.hpp).
void set_radical(detail::sequence_data& d) {
    const natural& size = d.span;
    if (size.is_zero()) return;
    const int f = std::max(0, (size - natural(1)).bit_length() - 64);
    const std::vector<k::limb>& mask = d.mask;

    // x0: the first rank of the region congruent to M modulo 2^f.
    const natural mask_low = from_vector(low_bits(mask.data(), mask.size(), f));
    const natural base_low = from_vector(low_bits(d.base.limbs(), d.base.limb_count(), f));
    const natural up =
        mask_low >= base_low ? mask_low - base_low : mask_low + power_of_two(f) - base_low;
    const natural x0 = d.base + up;
    const natural end = d.base + size;
    if (x0 >= end) throw std::logic_error("bounded_compositions: empty radical class");
    // C - 1 = (end - 1 - x0) >> f, below 2^64 (radical.hpp).
    const natural rest = end - natural(1) - x0;
    const natural last = from_vector(shift_right(rest.limbs(), rest.limb_count(), f));
    if (!last.fits_u64()) throw std::logic_error("bounded_compositions: radical class too large");

    d.origin.assign(x0.limbs(), x0.limbs() + x0.limb_count());
    if (d.origin.empty()) d.origin.push_back(0);
    k::radical_view& v = d.radical;
    v.origin = d.origin.data();
    v.origin_limbs = as_int(d.origin.size());
    v.shift = f;
    v.origin_high = low_word(shift_right(x0.limbs(), x0.limb_count(), f));
    v.mask = low_word(shift_right(mask.data(), mask.size(), f));
    v.last = last.is_zero() ? 0 : last.to_u64();
    const std::uint64_t most = std::numeric_limits<std::uint64_t>::max();
    d.capacity = v.last == most ? most : v.last + 1;
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
    data_->rank_of(i, rank.data());
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

std::uint64_t sequence::reserve(std::uint64_t count, std::uint64_t& got) {
    std::uint64_t begin = cursor_.load(std::memory_order_relaxed);
    std::uint64_t n = 0;
    do {
        n = std::min(count, data_->capacity - begin);
    } while (!cursor_.compare_exchange_weak(begin, begin + n, std::memory_order_relaxed));
    got = n;
    return begin;
}

namespace {

// key(a) <= key(b) for ranks of a space with mask M: keys are bit reversals of
// x ^ M, so they compare at the lowest bit where a and b differ.
bool key_not_after(const natural& a, const natural& b, const std::vector<k::limb>& mask) {
    const std::size_t words = std::max(a.limb_count(), b.limb_count());
    for (std::size_t i = 0; i < words; ++i) {
        k::limb diff = k::limb_at(a.limbs(), as_int(a.limb_count()), as_int(i)) ^
                       k::limb_at(b.limbs(), as_int(b.limb_count()), as_int(i));
        if (diff == 0) continue;
        int bit = 0;
        while ((diff & 1u) == 0) {
            diff >>= 1;
            ++bit;
        }
        const k::limb a_bit = k::limb_at(a.limbs(), as_int(a.limb_count()), as_int(i)) >> bit;
        const k::limb m_bit = k::limb_at(mask.data(), as_int(mask.size()), as_int(i)) >> bit;
        return ((a_bit ^ m_bit) & 1u) == 0;
    }
    return true;  // a == b
}

}  // namespace

std::uint64_t sequence::overlap(const sequence& outer, std::uint64_t n) const {
    const detail::sequence_data& in = *data_;
    const detail::sequence_data& out = *outer.data_;
    if (in.kind != sequence_kind::rank || out.kind != sequence_kind::rank ||
        !detail::same_space(in.dom, out.dom) || in.mask != out.mask) {
        throw std::invalid_argument(
            "bounded_compositions::sequence::overlap: needs two rank sequences of the same space "
            "and seed");
    }
    if (in.base < out.base || in.base + in.span > out.base + out.span) {
        throw std::invalid_argument(
            "bounded_compositions::sequence::overlap: the region is not inside the outer one");
    }
    check_range(0, n, out.capacity);
    if (n == 0 || in.capacity == 0) return 0;
    // The outer's first n points are its ranks with key up to that of point
    // n - 1; this sequence's ranks with such keys are its first ones.
    const natural last = outer.rank_at(n - 1);
    std::uint64_t lo = 0;
    std::uint64_t hi = in.capacity;
    while (lo < hi) {
        const std::uint64_t mid = lo + (hi - lo) / 2;
        if (key_not_after(rank_at(mid), last, in.mask)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

void sequence::resume_after(const sequence& outer) { seek(overlap(outer, outer.position())); }

namespace {

std::shared_ptr<detail::sequence_data> make_rank(const space& s, const region* r, bool scramble,
                                                 std::uint64_t seed) {
    auto d = base_data(s, r, sequence_kind::rank, scramble ? seed : 0);
    if (!s.empty()) {
        // One order per space and seed: the mask spans the space's rank bits.
        const int m = std::max(1, (s.size() - natural(1)).bit_length());
        d->mask = scramble ? mask_bits(seed, m)
                           : std::vector<k::limb>(static_cast<std::size_t>((m + 63) / 64), 0);
    }
    set_radical(*d);
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
