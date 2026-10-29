// Sampling without repeats: region order, resuming, dyadic shares, the
// visited set and fresh_sampler, all against brute force.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "bounded_compositions/bounded_compositions.hpp"

namespace bc = bounded_compositions;
using bc::natural;
using composition = std::vector<std::int32_t>;

namespace {

// Random spaces of at most a few thousand compositions.
std::vector<bc::space> random_spaces(int count, unsigned seed) {
    std::mt19937 rng(seed);
    auto pick = [&](int a, int b) { return std::uniform_int_distribution<int>(a, b)(rng); };
    std::vector<bc::space> out;
    while (static_cast<int>(out.size()) < count) {
        const std::size_t n = static_cast<std::size_t>(pick(1, 5));
        composition lo(n), hi(n);
        for (std::size_t i = 0; i < n; ++i) {
            lo[i] = pick(0, 3) == 0 ? pick(-2, 2) : 0;
            hi[i] = lo[i] + pick(0, 9);
        }
        const bc::space s(pick(0, 25), lo, hi);
        if (!s.empty() && s.size() <= natural(6000)) out.push_back(s);
    }
    return out;
}

bc::sequence order_of(const bc::space& s, int seed) {
    return seed < 0 ? bc::rank_sequence(s) : bc::rank_sequence(s, static_cast<std::uint64_t>(seed));
}

bc::sequence order_of(const bc::space& s, const bc::region& r, int seed) {
    return seed < 0 ? bc::rank_sequence(s, r)
                    : bc::rank_sequence(s, r, static_cast<std::uint64_t>(seed));
}

std::vector<std::uint64_t> ranks_of(const bc::sequence& seq, std::uint64_t count) {
    std::vector<std::uint64_t> out(count);
    for (std::uint64_t i = 0; i < count; ++i) out[i] = seq.rank_at(i).to_u64();
    return out;
}

// Every prefix of N points of [lo, lo + size) has its largest gap (counting
// the ends lo - 1 and lo + size) below 2 size / N, and no rank repeats.
bool every_prefix_spread(const std::vector<std::uint64_t>& ranks, std::uint64_t lo,
                         std::uint64_t size) {
    std::set<std::int64_t> seen{-1, static_cast<std::int64_t>(size)};
    std::multiset<std::int64_t> gaps{static_cast<std::int64_t>(size) + 1};
    for (std::size_t i = 0; i < ranks.size(); ++i) {
        const std::int64_t r = static_cast<std::int64_t>(ranks[i] - lo);
        const auto [it, inserted] = seen.insert(r);
        if (!inserted) return false;
        const std::int64_t before = *std::prev(it);
        const std::int64_t after = *std::next(it);
        gaps.erase(gaps.find(after - before));
        gaps.insert(r - before);
        gaps.insert(after - r);
        if (*gaps.rbegin() * static_cast<std::int64_t>(i + 1) >=
            2 * static_cast<std::int64_t>(size)) {
            return false;
        }
    }
    return true;
}

// Regions to try in s: random intervals and the subtrees along random paths.
std::vector<bc::region> some_regions(const bc::space& s, std::mt19937& rng) {
    const std::uint64_t size = s.size().to_u64();
    std::vector<bc::region> out{bc::whole(s)};
    for (int t = 0; t < 8; ++t) {
        const std::uint64_t a = std::uniform_int_distribution<std::uint64_t>(0, size - 1)(rng);
        const std::uint64_t b = std::uniform_int_distribution<std::uint64_t>(a + 1, size)(rng);
        out.push_back(bc::interval(s, natural(a), natural(b)));
    }
    bc::region at = bc::whole(s);
    for (;;) {
        const std::vector<bc::region> kids = bc::children(s, at);
        if (kids.empty()) break;
        at = kids[std::uniform_int_distribution<std::size_t>(0, kids.size() - 1)(rng)];
        out.push_back(at);
    }
    return out;
}

}  // namespace

TEST_CASE("a region's rank sequence is the space's order restricted to it") {
    std::mt19937 rng(5);
    for (const bc::space& s : random_spaces(120, 11)) {
        for (int seed : {-1, 3, 77}) {
            const bc::sequence global = order_of(s, seed);
            const std::vector<std::uint64_t> all = ranks_of(global, global.capacity());
            for (const bc::region& r : some_regions(s, rng)) {
                const std::uint64_t a = r.begin().to_u64(), b = r.end().to_u64();
                std::vector<std::uint64_t> expected;
                for (std::uint64_t x : all) {
                    if (a <= x && x < b) expected.push_back(x);
                }
                const bc::sequence local = order_of(s, r, seed);
                REQUIRE(local.capacity() == b - a);
                const std::vector<std::uint64_t> got = ranks_of(local, local.capacity());
                REQUIRE(got == expected);
                REQUIRE(every_prefix_spread(got, a, b - a));
                // points() agrees with rank_at.
                std::vector<std::int32_t> pts(local.capacity() * s.parts());
                local.points(0, local.capacity(), pts.data());
                for (std::uint64_t i = 0; i < local.capacity(); i += 7) {
                    REQUIRE(s.rank(pts.data() + i * s.parts()) == natural(got[i]));
                }
            }
        }
    }
}

TEST_CASE("overlap counts the outer points in a region, and resuming never repeats") {
    std::mt19937 rng(17);
    auto pick = [&](std::uint64_t a, std::uint64_t b) {
        return std::uniform_int_distribution<std::uint64_t>(a, b)(rng);
    };
    for (const bc::space& s : random_spaces(150, 23)) {
        for (int seed : {-1, 8}) {
            // Descend: the whole space, a subtree, a deeper one, then an
            // interval inside it, sampling some points at each step.
            std::vector<bc::region> path{bc::whole(s)};
            for (int depth = 0; depth < 2; ++depth) {
                const std::vector<bc::region> kids = bc::children(s, path.back());
                if (kids.empty()) break;
                path.push_back(kids[pick(0, kids.size() - 1)]);
            }
            const std::uint64_t a = path.back().begin().to_u64();
            const std::uint64_t size = path.back().size().to_u64();
            const std::uint64_t lo = a + pick(0, size - 1);
            path.push_back(bc::interval(s, natural(lo), natural(lo + pick(1, a + size - lo))));

            std::set<std::uint64_t> taken;
            std::vector<bc::sequence> seqs;
            for (const bc::region& r : path) {
                bc::sequence here = order_of(s, r, seed);
                if (!seqs.empty()) {
                    const bc::sequence& outer = seqs.back();
                    // Brute force: the outer's taken points that lie in r.
                    std::vector<std::uint64_t> inside;
                    for (std::uint64_t i = 0; i < outer.position(); ++i) {
                        const natural x = outer.rank_at(i);
                        if (r.contains(x)) inside.push_back(x.to_u64());
                    }
                    REQUIRE(here.overlap(outer, outer.position()) == inside.size());
                    REQUIRE(ranks_of(here, inside.size()) == inside);  // its first points
                    here.resume_after(outer);
                    REQUIRE(here.position() == inside.size());
                }
                const std::uint64_t left = here.capacity() - here.position();
                const std::uint64_t k = left == 0 ? 0 : pick(0, left);
                std::vector<std::int32_t> pts(k * s.parts());
                here.take(k, pts.data());
                for (std::uint64_t i = 0; i < k; ++i) {
                    REQUIRE(taken.insert(s.rank(pts.data() + i * s.parts()).to_u64()).second);
                }
                seqs.push_back(here);
            }
        }
    }
    const bc::space s(20, composition(4, 0), composition(4, 9));
    const bc::region t = bc::subtree(s, composition{3});
    const bc::sequence outer = bc::rank_sequence(s, 5);
    CHECK_THROWS_AS(bc::rank_sequence(s, t).overlap(outer, 10), std::invalid_argument);  // seed
    CHECK_THROWS_AS(bc::rank_sequence(s, t, 6).overlap(outer, 10), std::invalid_argument);
    CHECK_THROWS_AS(bc::rank_sequence(s, 5).overlap(bc::rank_sequence(s, t, 5), 1),
                    std::invalid_argument);  // not inside
    CHECK_THROWS_AS(bc::iid_sequence(s, t, 5).overlap(outer, 1), std::invalid_argument);
    CHECK_THROWS_AS(bc::rank_sequence(s, t, 5).overlap(outer, outer.capacity() + 1),
                    std::out_of_range);
    CHECK(bc::rank_sequence(s, t, 5).overlap(outer, 0) == 0);
    const bc::space same(20, composition(4, 0), composition(4, 9));  // equal, separate object
    CHECK(bc::rank_sequence(same, t, 5).overlap(outer, outer.capacity()) == t.size().to_u64());
}

TEST_CASE("full dyadic blocks share every prefix of the region's sequence within one point") {
    std::mt19937 rng(29);
    for (const bc::space& s : random_spaces(80, 37)) {
        for (int seed : {-1, 4}) {
            for (const bc::region& r : some_regions(s, rng)) {
                const std::uint64_t a = r.begin().to_u64(), b = r.end().to_u64();
                std::uint64_t diff = a ^ (b - 1);
                int rho = 0;
                while (diff != 0) {
                    diff >>= 1;
                    ++rho;
                }
                const std::vector<std::uint64_t> pts = ranks_of(order_of(s, r, seed), b - a);
                for (int levels = 0; levels <= 4; ++levels) {
                    const std::vector<bc::region> blocks = bc::dyadic_split(s, r, levels);
                    REQUIRE(blocks.size() <= (std::size_t{1} << levels));
                    const std::uint64_t full = std::uint64_t{1} << (rho - std::min(levels, rho));
                    // r's parts of consecutive aligned blocks of `full` ranks.
                    natural at = r.begin();
                    for (std::size_t j = 0; j < blocks.size(); ++j) {
                        const std::uint64_t lo = blocks[j].begin().to_u64();
                        const std::uint64_t hi = blocks[j].end().to_u64();
                        REQUIRE(blocks[j].begin() == at);
                        REQUIRE(lo / full == a / full + j);
                        REQUIRE((hi - 1) / full == lo / full);
                        if (j > 0 && j + 1 < blocks.size()) REQUIRE(hi - lo == full);
                        at = blocks[j].end();
                    }
                    REQUIRE(at == r.end());
                    std::vector<std::uint64_t> counts(blocks.size(), 0);
                    for (std::uint64_t x : pts) {
                        for (std::size_t j = 0; j < blocks.size(); ++j) {
                            if (blocks[j].contains(natural(x))) ++counts[j];
                        }
                        std::uint64_t most = 0, least = ~std::uint64_t{0};
                        for (std::size_t j = 0; j < blocks.size(); ++j) {
                            if (blocks[j].size() != natural(full)) continue;
                            most = std::max(most, counts[j]);
                            least = std::min(least, counts[j]);
                        }
                        if (most > 0) REQUIRE(most - least <= 1);
                    }
                }
            }
        }
    }
}

TEST_CASE("region sequences on a space wider than 64 bits") {
    const bc::space s(100, composition(100, 0), composition(100, 100));  // 195-bit size
    for (int seed : {-1, 12}) {
        // A deep subtree is small: its sequence covers it, spread out.
        const bc::region deep = bc::subtree(s, composition(97, 0));  // C(102, 2) = 5151
        REQUIRE(deep.size() == natural(5151));
        const bc::sequence local = order_of(s, deep, seed);
        REQUIRE(local.capacity() == 5151);
        std::vector<std::uint64_t> rel(5151);
        for (std::uint64_t i = 0; i < 5151; ++i) {
            const natural x = local.rank_at(i);
            REQUIRE(deep.contains(x));
            rel[i] = (x - deep.begin()).to_u64();
        }
        CHECK(every_prefix_spread(rel, 0, 5151));

        // A wide block and a wide piece of it: the piece starts with the
        // block's points that fall in it, and overlap counts them.
        const bc::region block = bc::dyadic_split(s, bc::whole(s), 3)[2];
        const bc::region piece = bc::dyadic_split(s, block, 2)[1];
        const bc::sequence big = order_of(s, block, seed);
        const bc::sequence small = order_of(s, piece, seed);
        CHECK(big.capacity() > (std::uint64_t{1} << 63));
        CHECK(small.capacity() > (std::uint64_t{1} << 63));
        std::vector<natural> inside;
        for (std::uint64_t i = 0; i < 4096; ++i) {
            const natural x = big.rank_at(i);
            REQUIRE(block.contains(x));
            if (piece.contains(x)) inside.push_back(x);
        }
        REQUIRE(!inside.empty());
        CHECK(small.overlap(big, 4096) == inside.size());
        for (std::size_t i = 0; i < inside.size(); ++i) CHECK(small.rank_at(i) == inside[i]);
    }
    // Subtrees of the wide space start with the space's points that fall in
    // them, in order.
    for (int seed : {-1, 12}) {
        const bc::sequence global = order_of(s, seed);
        const std::uint64_t count = 1 << 15;
        std::vector<natural> pts(count);
        for (std::uint64_t i = 0; i < count; ++i) pts[i] = global.rank_at(i);
        for (std::int32_t v : {0, 1, 3, 6}) {
            const bc::region t = bc::subtree(s, composition{v});
            const bc::sequence local = order_of(s, t, seed);
            std::vector<natural> inside;
            for (const natural& x : pts) {
                if (t.contains(x)) inside.push_back(x);
            }
            REQUIRE(!inside.empty());
            CHECK(local.overlap(global, count) == inside.size());
            for (std::size_t i = 0; i < inside.size(); ++i) REQUIRE(local.rank_at(i) == inside[i]);
        }
    }
    // Unscrambled keys increase along a wide subtree's sequence: the lowest
    // bit where two ranks differ is 0 in the earlier one.
    const bc::sequence seq = bc::rank_sequence(s, bc::subtree(s, composition{3}));
    natural prev = seq.rank_at(0);
    for (std::uint64_t i = 1; i < 2000; ++i) {
        const natural x = seq.rank_at(i);
        bool earlier_has_zero = false;
        for (std::size_t w = 0; w < std::max(x.limb_count(), prev.limb_count()); ++w) {
            const std::uint64_t pa = w < prev.limb_count() ? prev.limbs()[w] : 0;
            const std::uint64_t xa = w < x.limb_count() ? x.limbs()[w] : 0;
            if (pa == xa) continue;
            std::uint64_t d = pa ^ xa;
            int bit = 0;
            while ((d & 1u) == 0) {
                d >>= 1;
                ++bit;
            }
            earlier_has_zero = ((pa >> bit) & 1u) == 0;
            break;
        }
        REQUIRE(earlier_has_zero);
        prev = x;
    }
}

TEST_CASE("visited_set matches a std::set, also from many threads") {
    const bc::space s(30, composition(6, 0), composition(6, 12));
    const std::uint64_t size = s.size().to_u64();
    bc::visited_set v(s);
    std::set<std::uint64_t> ref;
    std::mt19937_64 rng(3);
    for (int i = 0; i < 20000; ++i) {
        const std::uint64_t r = rng() % size;
        CHECK(v.contains(natural(r)) == (ref.count(r) == 1));
        CHECK(v.insert(natural(r)) == ref.insert(r).second);
    }
    CHECK(v.size() == ref.size());
    const composition x = s.unrank(natural(*ref.begin()));
    CHECK(v.contains(x.data()));
    CHECK_FALSE(v.insert(x.data()));
    CHECK_THROWS_AS(v.insert(s.size()), std::invalid_argument);
    CHECK_THROWS_AS(v.contains(s.size()), std::invalid_argument);
    CHECK_THROWS_AS(bc::visited_set(bc::space(30, composition(6, 0), composition(6, 12),
                                              bc::counting::logarithmic)),
                    std::logic_error);

    // Wide ranks: the last rank and its neighbours.
    const bc::space wide(100, composition(100, 0), composition(100, 100));
    bc::visited_set vw(wide);
    const natural top = wide.size() - natural(1);
    CHECK(vw.insert(top));
    CHECK(vw.insert(natural(0)));
    CHECK_FALSE(vw.insert(top));
    CHECK(vw.contains(top));
    CHECK_FALSE(vw.contains(top - natural(1)));

    // Threads insert overlapping ranges: each rank is new exactly once.
    bc::visited_set shared(s);
    std::vector<std::uint64_t> fresh(8, 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&, t] {
            const std::uint64_t from = static_cast<std::uint64_t>(t) * size / 16;
            for (std::uint64_t r = from; r < from + size / 4 && r < size; ++r) {
                if (shared.insert(natural(r))) ++fresh[static_cast<std::size_t>(t)];
            }
        });
    }
    for (auto& th : threads) th.join();
    std::uint64_t total = 0;
    for (std::uint64_t f : fresh) total += f;
    const std::uint64_t covered = std::min(size, 7 * size / 16 + size / 4);
    CHECK(total == shared.size());
    CHECK(shared.size() == covered);
}

namespace {

// Draws from a fresh_sampler in uneven chunks until it runs out.
std::vector<std::uint64_t> drain(bc::fresh_sampler& f, const bc::space& s, std::mt19937& rng) {
    std::vector<std::uint64_t> out;
    for (;;) {
        const std::uint64_t want = std::uniform_int_distribution<std::uint64_t>(1, 300)(rng);
        std::vector<std::int32_t> pts(want * s.parts());
        const std::uint64_t got = f.take(want, pts.data());
        for (std::uint64_t i = 0; i < got; ++i) {
            out.push_back(s.rank(pts.data() + i * s.parts()).to_u64());
        }
        if (got < want) return out;
    }
}

}  // namespace

TEST_CASE("fresh_sampler hands out exactly the region's unvisited points") {
    std::mt19937 rng(41);
    for (const bc::space& s : random_spaces(60, 43)) {
        const std::uint64_t size = s.size().to_u64();
        for (const bc::region& r : some_regions(s, rng)) {
            const std::uint64_t a = r.begin().to_u64(), b = r.end().to_u64();
            for (int kind = 0; kind < 4; ++kind) {
                bc::visited_set v(s);
                std::set<std::uint64_t> before;
                for (std::uint64_t x = 0; x < size; ++x) {
                    if (rng() % 3 == 0) {
                        v.insert(natural(x));
                        before.insert(x);
                    }
                }
                const bc::sequence source = kind == 0   ? bc::rank_sequence(s, r)
                                            : kind == 1 ? bc::rank_sequence(s, r, 9)
                                            : kind == 2 ? bc::iid_sequence(s, r, 9)
                                                        : (r.is_subtree() && s.parts() >= 2
                                                               ? bc::coordinate_sequence(s, r, 9)
                                                               : bc::iid_sequence(s, r, 10));
                bc::fresh_sampler f(source, v, 64);
                const std::vector<std::uint64_t> got = drain(f, s, rng);
                std::set<std::uint64_t> expected;
                std::uint64_t visited_inside = 0;
                for (std::uint64_t x = a; x < b; ++x) {
                    if (before.count(x) == 0) {
                        expected.insert(x);
                    } else {
                        ++visited_inside;
                    }
                }
                REQUIRE(std::set<std::uint64_t>(got.begin(), got.end()) == expected);
                REQUIRE(got.size() == expected.size());  // no repeats
                REQUIRE(f.exhausted());
                REQUIRE(v.size() == before.size() + expected.size());
                std::vector<std::int32_t> one(s.parts());
                REQUIRE(f.take(1, one.data()) == 0);
                if (kind < 2) {
                    // A rank source keeps its order and passes each visited
                    // point of the region once.
                    std::vector<std::uint64_t> order;
                    for (std::uint64_t x : ranks_of(source, source.capacity())) {
                        if (before.count(x) == 0) order.push_back(x);
                    }
                    REQUIRE(got == order);
                    REQUIRE(f.skipped() == visited_inside);
                    REQUIRE_FALSE(f.switched());
                }
            }
        }
    }
    const bc::space s(10, composition(3, 0), composition(3, 10));
    const bc::space other(11, composition(3, 0), composition(3, 10));
    bc::visited_set v(other);
    CHECK_THROWS_AS(bc::fresh_sampler(bc::rank_sequence(s), v), std::invalid_argument);
}

TEST_CASE("fresh_sampler on decoded points when the table is wider than the size") {
    // 12 parts summing to 207: the size takes 60 bits, but row 0 of the table
    // holds prefix sums over every total up to 207, which take 65.
    const bc::space s(207, composition(12, 0), composition(12, 207));
    REQUIRE(s.size() <= natural(~std::uint64_t{0}));
    for (const bc::sequence& source : {bc::coordinate_sequence(s, 3), bc::coordinate_sequence(s)}) {
        bc::visited_set v(s);
        bc::fresh_sampler f(source, v);
        std::vector<std::int32_t> out(1000 * s.parts());
        REQUIRE(f.take(1000, out.data()) == 1000);
        std::set<composition> seen;
        const auto w = static_cast<std::ptrdiff_t>(s.parts());
        for (std::ptrdiff_t p = 0; p < 1000; ++p) {
            const composition x(out.begin() + p * w, out.begin() + (p + 1) * w);
            REQUIRE(s.contains(x.data()));
            REQUIRE(v.contains(x.data()));
            seen.insert(x);
        }
        REQUIRE(seen.size() == 1000);
        REQUIRE(v.size() == 1000);
    }
}

TEST_CASE("fresh_sampler shared by threads never hands out a point twice") {
    const bc::space s(40, composition(5, 0), composition(5, 14));
    const std::uint64_t size = s.size().to_u64();
    for (int kind = 0; kind < 2; ++kind) {
        bc::visited_set v(s);
        std::set<std::uint64_t> before;
        for (std::uint64_t x = 0; x < size; x += 3) {
            v.insert(natural(x));
            before.insert(x);
        }
        const bc::sequence source = kind == 0 ? bc::rank_sequence(s, 2) : bc::iid_sequence(s, 2);
        bc::fresh_sampler f(source, v, 256);
        std::vector<std::vector<std::uint64_t>> got(8);
        bc::thread_pool pool(3);
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([&, t] {
                std::vector<std::int32_t> pts(97 * s.parts());
                for (;;) {
                    const std::uint64_t n = f.take(97, pts.data(), t % 2 == 0 ? &pool : nullptr);
                    for (std::uint64_t i = 0; i < n; ++i) {
                        got[static_cast<std::size_t>(t)].push_back(
                            s.rank(pts.data() + i * s.parts()).to_u64());
                    }
                    if (n < 97) return;
                }
            });
        }
        for (auto& th : threads) th.join();
        std::multiset<std::uint64_t> all;
        for (const auto& g : got) all.insert(g.begin(), g.end());
        CHECK(all.size() == size - before.size());
        CHECK(std::set<std::uint64_t>(all.begin(), all.end()).size() == all.size());
        for (std::uint64_t x : all) CHECK(before.count(x) == 0);
        CHECK(v.size() == size);
    }
}

namespace {

// A limit the space does not count: x[0] + 2 x[1] - x[n-1] <= bound.
struct linear_limit {
    std::int32_t bound;
    std::size_t parts;
};

bool keep_linear(void* context, const std::int32_t* x) {
    const linear_limit& l = *static_cast<const linear_limit*>(context);
    const std::int32_t second = l.parts > 1 ? x[1] : 0;
    return x[0] + 2 * second - x[l.parts - 1] <= l.bound;
}

bool keep_nothing(void*, const std::int32_t*) { return false; }

}  // namespace

TEST_CASE("fresh_sampler with a filter hands out exactly the accepted unvisited points") {
    std::mt19937 rng(61);
    for (const bc::space& s : random_spaces(50, 67)) {
        const std::uint64_t size = s.size().to_u64();
        for (const bc::region& r : some_regions(s, rng)) {
            const std::uint64_t a = r.begin().to_u64(), b = r.end().to_u64();
            linear_limit limit{static_cast<std::int32_t>(rng() % 9) - 2, s.parts()};
            for (int kind = 0; kind < 3; ++kind) {
                bc::visited_set v(s);
                std::set<std::uint64_t> before;
                for (std::uint64_t x = 0; x < size; ++x) {
                    if (rng() % 4 == 0) {
                        v.insert(natural(x));
                        before.insert(x);
                    }
                }
                const bc::sequence source = kind == 0   ? bc::rank_sequence(s, r)
                                            : kind == 1 ? bc::iid_sequence(s, r, 3)
                                                        : bc::rank_sequence(s, r, 5);
                bc::fresh_sampler f(source, v, &keep_linear, &limit, 64);
                const std::vector<std::uint64_t> got = drain(f, s, rng);
                std::set<std::uint64_t> expected;
                std::uint64_t visited_inside = 0, refused = 0;
                for (std::uint64_t x = a; x < b; ++x) {
                    if (before.count(x) != 0) {
                        ++visited_inside;
                        continue;
                    }
                    const composition y = s.unrank(natural(x));
                    if (keep_linear(&limit, y.data())) {
                        expected.insert(x);
                    } else {
                        ++refused;
                    }
                }
                REQUIRE(std::set<std::uint64_t>(got.begin(), got.end()) == expected);
                REQUIRE(got.size() == expected.size());
                REQUIRE(f.exhausted());
                REQUIRE(f.accepted() == expected.size());
                REQUIRE(v.size() == before.size() + expected.size());  // rejected ones not added
                if (kind != 1) {
                    // A rank source checks each point of the region once, in order.
                    std::vector<std::uint64_t> order;
                    for (std::uint64_t x : ranks_of(source, source.capacity())) {
                        if (expected.count(x) != 0) order.push_back(x);
                    }
                    REQUIRE(got == order);
                    REQUIRE(f.rejected() == refused);
                    REQUIRE(f.skipped() == visited_inside);
                }
            }
        }
    }
    // A filter that refuses everything ends a rank source after one pass, and
    // moves an iid source to the rank order, which then ends too.
    const bc::space s(20, composition(4, 0), composition(4, 9));
    for (const bc::sequence& source : {bc::rank_sequence(s), bc::iid_sequence(s, 1)}) {
        bc::visited_set v(s);
        bc::fresh_sampler f(source, v, &keep_nothing, nullptr, 100);
        std::vector<std::int32_t> out(10 * 4);
        CHECK(f.take(10, out.data()) == 0);
        CHECK(f.exhausted());
        CHECK(v.size() == 0);
        if (source.kind() == bc::sequence_kind::rank) CHECK(f.rejected() == s.size().to_u64());
    }
    bc::visited_set v(s);
    CHECK_THROWS_AS(bc::fresh_sampler(bc::rank_sequence(s), v, nullptr, nullptr),
                    std::invalid_argument);
}

TEST_CASE("fresh_sampler with a filter shared by threads") {
    const bc::space s(40, composition(5, 0), composition(5, 14));
    const std::uint64_t size = s.size().to_u64();
    linear_limit limit{6, 5};
    std::set<std::uint64_t> expected;
    for (std::uint64_t x = 0; x < size; ++x) {
        const composition y = s.unrank(natural(x));
        if (x % 5 != 0 && keep_linear(&limit, y.data())) expected.insert(x);
    }
    for (int kind = 0; kind < 2; ++kind) {
        bc::visited_set v(s);
        for (std::uint64_t x = 0; x < size; x += 5) v.insert(natural(x));
        const bc::sequence source = kind == 0 ? bc::rank_sequence(s, 2) : bc::iid_sequence(s, 2);
        bc::fresh_sampler f(source, v, &keep_linear, &limit, 256);
        std::vector<std::vector<std::uint64_t>> got(6);
        bc::thread_pool pool(3);
        std::vector<std::thread> threads;
        for (int t = 0; t < 6; ++t) {
            threads.emplace_back([&, t] {
                std::vector<std::int32_t> pts(53 * 5);
                for (;;) {
                    const std::uint64_t n = f.take(53, pts.data(), t % 2 == 0 ? &pool : nullptr);
                    for (std::uint64_t i = 0; i < n; ++i) {
                        got[static_cast<std::size_t>(t)].push_back(
                            s.rank(pts.data() + i * 5).to_u64());
                    }
                    if (n < 53) return;
                }
            });
        }
        for (auto& th : threads) th.join();
        std::multiset<std::uint64_t> all;
        for (const auto& g : got) all.insert(g.begin(), g.end());
        CHECK(std::set<std::uint64_t>(all.begin(), all.end()) == expected);
        CHECK(all.size() == expected.size());
        CHECK(f.accepted() == expected.size());
    }
}
