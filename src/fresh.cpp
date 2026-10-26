#include "bounded_compositions/fresh.hpp"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "bounded_compositions/kernel/limbs.hpp"
#include "bounded_compositions/kernel/walk.hpp"
#include "bounded_compositions/region.hpp"
#include "sequence_data.hpp"

namespace bounded_compositions {

namespace k = kernel;

namespace {

constexpr int shard_bits = 6;
constexpr std::size_t shard_count = std::size_t{1} << shard_bits;
// A stored rank is below the size, which fits in the stored width, so a slot
// of all ones is never a rank: it marks an empty slot.
constexpr k::limb empty_word = ~k::limb{0};
constexpr std::uint64_t batch_points = std::uint64_t{1} << 16;
constexpr std::uint64_t block_points = 1024;

// splitmix64's finaliser.
std::uint64_t mix(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9u;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBu;
    return z ^ (z >> 31);
}

std::uint64_t hash_limbs(const k::limb* key, int width) {
    std::uint64_t h = 0x9E3779B97F4A7C15u;
    for (int i = 0; i < width; ++i) h = mix(h ^ key[i]);
    return h;
}

bool is_empty(const k::limb* slot, int width) {
    for (int i = 0; i < width; ++i) {
        if (slot[i] != empty_word) return false;
    }
    return true;
}

}  // namespace

// One part of the set, with its own lock: open addressing, linear probing,
// at most half full.
struct visited_set::shard {
    mutable std::mutex lock;
    std::vector<k::limb> slots;  // capacity * width limbs
    std::uint64_t used = 0;
    std::uint64_t capacity = 0;  // a power of two, or 0
};

visited_set::visited_set(const space& s)
    : dom_(s),
      width_(std::max(
          1, static_cast<int>(s.counting_mode() == counting::exact ? s.size().limb_count() : 1))),
      shards_(new shard[shard_count]) {
    if (s.counting_mode() != counting::exact) {
        throw std::logic_error(
            "bounded_compositions::visited_set: needs exact counting; this space counts in "
            "logarithms");
    }
}

visited_set::~visited_set() = default;

const space& visited_set::domain() const noexcept { return dom_; }
std::uint64_t visited_set::size() const noexcept { return size_.load(std::memory_order_relaxed); }

std::uint64_t visited_set::hash(const k::limb* key) const noexcept {
    return hash_limbs(key, width_);
}

void visited_set::key_of(const natural& rank, k::limb* key) const {
    if (rank >= dom_.size()) {
        throw std::invalid_argument("bounded_compositions::visited_set: rank " + rank.to_string() +
                                    " is not below the size " + dom_.size().to_string());
    }
    k::limbs_copy(rank.limbs(), static_cast<int>(rank.limb_count()), key, width_);
}

bool visited_set::insert_key(const k::limb* key) {
    const std::uint64_t h = hash(key);
    shard& sh = shards_[h >> (64 - shard_bits)];
    const std::size_t w = static_cast<std::size_t>(width_);
    std::lock_guard<std::mutex> hold(sh.lock);
    if ((sh.used + 1) * 2 > sh.capacity) {
        const std::uint64_t grown = std::max<std::uint64_t>(16, sh.capacity * 2);
        std::vector<k::limb> slots(grown * w, empty_word);
        for (std::uint64_t i = 0; i < sh.capacity; ++i) {
            const k::limb* old = sh.slots.data() + i * w;
            if (is_empty(old, width_)) continue;
            std::uint64_t j = hash(old) & (grown - 1);
            while (!is_empty(slots.data() + j * w, width_)) j = (j + 1) & (grown - 1);
            std::copy(old, old + w, slots.data() + j * w);
        }
        sh.slots.swap(slots);
        sh.capacity = grown;
    }
    for (std::uint64_t i = h & (sh.capacity - 1);; i = (i + 1) & (sh.capacity - 1)) {
        k::limb* slot = sh.slots.data() + i * w;
        if (is_empty(slot, width_)) {
            std::copy(key, key + w, slot);
            ++sh.used;
            size_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        if (std::equal(key, key + w, slot)) return false;
    }
}

bool visited_set::contains_key(const k::limb* key) const {
    const std::uint64_t h = hash(key);
    const shard& sh = shards_[h >> (64 - shard_bits)];
    const std::size_t w = static_cast<std::size_t>(width_);
    std::lock_guard<std::mutex> hold(sh.lock);
    if (sh.capacity == 0) return false;
    for (std::uint64_t i = h & (sh.capacity - 1);; i = (i + 1) & (sh.capacity - 1)) {
        const k::limb* slot = sh.slots.data() + i * w;
        if (is_empty(slot, width_)) return false;
        if (std::equal(key, key + w, slot)) return true;
    }
}

bool visited_set::insert(const natural& rank) {
    std::vector<k::limb> key(static_cast<std::size_t>(width_));
    key_of(rank, key.data());
    return insert_key(key.data());
}

bool visited_set::contains(const natural& rank) const {
    std::vector<k::limb> key(static_cast<std::size_t>(width_));
    key_of(rank, key.data());
    return contains_key(key.data());
}

bool visited_set::insert(const std::int32_t* x) { return insert(dom_.rank(x)); }
bool visited_set::contains(const std::int32_t* x) const { return contains(dom_.rank(x)); }

namespace {

// Ranks (and, for decoded sequences, the points) of [begin, begin + count).
struct ranks_job {
    const detail::sequence_data* data;
    std::uint64_t begin;
    std::uint64_t count;
    int width;
    k::limb* ranks;
    std::int32_t* points;  // decoded sequences only
};

void ranks_block(void* context, std::uint64_t block) {
    const ranks_job& job = *static_cast<const ranks_job*>(context);
    const detail::sequence_data& d = *job.data;
    const std::uint64_t first = block * block_points;
    const std::uint64_t n = std::min(block_points, job.count - first);
    const std::size_t parts = d.dom.parts();
    const std::size_t w = static_cast<std::size_t>(job.width);
    if (d.decoded) {
        std::int32_t* pts = job.points + first * parts;
        d.write_decoded_points(job.begin + first, n, pts);
        const k::space_view v = d.dom.view();
        for (std::uint64_t p = 0; p < n; ++p) {
            if (k::rank(v, pts + p * parts, job.ranks + (first + p) * w, job.width) !=
                k::status::ok) {
                throw std::logic_error("bounded_compositions: a decoded point is not in the space");
            }
        }
    } else {
        for (std::uint64_t p = 0; p < n; ++p) {
            d.rank_of(job.begin + first + p, job.ranks + (first + p) * w);
        }
    }
}

// Writes the kept points to out, in order.
struct write_job {
    const detail::sequence_data* data;
    const std::vector<std::uint64_t>* kept;  // indices into the batch
    int width;
    const k::limb* ranks;
    const std::int32_t* points;
    std::int32_t* out;
};

void write_block(void* context, std::uint64_t block) {
    const write_job& job = *static_cast<const write_job*>(context);
    const detail::sequence_data& d = *job.data;
    const std::size_t parts = d.dom.parts();
    const std::size_t w = static_cast<std::size_t>(job.width);
    const k::space_view v = d.dom.view();
    const std::uint64_t first = block * block_points;
    const std::uint64_t n = std::min<std::uint64_t>(block_points, job.kept->size() - first);
    for (std::uint64_t p = first; p < first + n; ++p) {
        const std::uint64_t at = (*job.kept)[p];
        std::int32_t* x = job.out + p * parts;
        if (d.decoded) {
            std::copy(job.points + at * parts, job.points + (at + 1) * parts, x);
        } else if (k::unrank(v, job.ranks + at * w, job.width, x) != k::status::ok) {
            throw std::logic_error("bounded_compositions: sequence produced a rank out of range");
        }
    }
}

void run_blocks(executor* ex, std::uint64_t items, executor::block_fn fn, void* context) {
    if (items == 0) return;
    serial_executor serial;
    (ex != nullptr ? *ex : static_cast<executor&>(serial))
        .run((items + block_points - 1) / block_points, fn, context);
}

}  // namespace

sequence fresh_sampler::fallback_for(const sequence& source) {
    const detail::sequence_data& d = *source.data_;
    if (d.kind == sequence_kind::rank) return source;  // never used
    return rank_sequence(d.dom, interval(d.dom, d.base, d.base + d.span));
}

fresh_sampler::fresh_sampler(const sequence& source, visited_set& visited, std::uint64_t patience)
    : source_(source),
      fallback_(fallback_for(source)),
      visited_(visited),
      patience_(std::max<std::uint64_t>(1, patience)) {
    if (!detail::same_space(source.domain(), visited.domain())) {
        throw std::invalid_argument(
            "bounded_compositions::fresh_sampler: the visited set is for another space");
    }
}

bool fresh_sampler::exhausted() const noexcept {
    return exhausted_.load(std::memory_order_relaxed);
}
bool fresh_sampler::switched() const noexcept { return switched_.load(std::memory_order_relaxed); }
std::uint64_t fresh_sampler::skipped() const noexcept {
    return skipped_.load(std::memory_order_relaxed);
}

std::uint64_t fresh_sampler::take(std::uint64_t count, std::int32_t* out, executor* ex) {
    const std::size_t parts = source_.domain().parts();
    const int width = visited_.width_;
    const std::size_t w = static_cast<std::size_t>(width);
    std::vector<k::limb> ranks;
    std::vector<std::int32_t> points;
    std::vector<std::uint64_t> kept;
    std::uint64_t written = 0;
    std::uint64_t run = 0;  // visited points in a row
    while (written < count) {
        const bool on_fallback = switched_.load(std::memory_order_relaxed);
        sequence& from = on_fallback ? fallback_ : source_;
        const detail::sequence_data& d = *from.data_;
        const bool covers = d.kind == sequence_kind::rank;
        std::uint64_t got = 0;
        const std::uint64_t begin = from.reserve(std::min(count - written, batch_points), got);
        if (got == 0) {
            if (!covers) {
                switched_.store(true, std::memory_order_relaxed);
                continue;
            }
            exhausted_.store(true, std::memory_order_relaxed);
            break;
        }
        ranks.assign(got * w, 0);
        points.assign(d.decoded ? got * parts : 0, 0);
        ranks_job rj{&d, begin, got, width, ranks.data(), points.data()};
        run_blocks(ex, got, &ranks_block, &rj);

        // In order, so one caller always gets the same points.
        kept.clear();
        for (std::uint64_t p = 0; p < got; ++p) {
            if (visited_.insert_key(ranks.data() + p * w)) {
                kept.push_back(p);
                run = 0;
            } else {
                skipped_.fetch_add(1, std::memory_order_relaxed);
                if (!covers && ++run >= patience_) switched_.store(true, std::memory_order_relaxed);
            }
        }
        write_job wj{&d, &kept, width, ranks.data(), points.data(), out + written * parts};
        run_blocks(ex, kept.size(), &write_block, &wj);
        written += kept.size();
    }
    return written;
}

}  // namespace bounded_compositions
