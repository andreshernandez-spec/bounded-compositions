#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "bounded_compositions/bounded_compositions.hpp"

namespace bc = bounded_compositions;
using bc::natural;
using composition = std::vector<std::int32_t>;

namespace {

void enumerate(std::size_t i, std::int64_t left, const composition& lo, const composition& hi,
               composition& cur, std::vector<composition>& out) {
    if (i + 1 == lo.size()) {
        if (left >= lo[i] && left <= hi[i]) {
            cur[i] = static_cast<std::int32_t>(left);
            out.push_back(cur);
        }
        return;
    }
    for (std::int32_t v = lo[i]; v <= hi[i]; ++v) {
        cur[i] = v;
        enumerate(i + 1, left - v, lo, hi, cur, out);
    }
}

std::vector<composition> all_of(std::int32_t total, const composition& lo, const composition& hi) {
    std::vector<composition> out;
    composition cur(lo.size());
    enumerate(0, total, lo, hi, cur, out);
    return out;
}

bool starts_with(const composition& x, const composition& prefix) {
    return std::equal(prefix.begin(), prefix.end(), x.begin());
}

// Longest prefix shared by ref[a..b).
composition shared_prefix(const std::vector<composition>& ref, std::size_t a, std::size_t b) {
    composition p = ref[a];
    for (std::size_t r = a + 1; r < b; ++r) {
        std::size_t d = 0;
        while (d < p.size() && p[d] == ref[r][d]) ++d;
        p.resize(d);
    }
    return p;
}

struct small_space {
    std::int32_t total;
    composition lo, hi;
    std::vector<composition> ref;
};

std::vector<small_space> random_spaces(int count, unsigned seed) {
    std::mt19937 rng(seed);
    auto pick = [&](int a, int b) { return std::uniform_int_distribution<int>(a, b)(rng); };
    std::vector<small_space> out;
    while (static_cast<int>(out.size()) < count) {
        small_space s;
        const std::size_t n = static_cast<std::size_t>(pick(1, 6));
        s.lo.resize(n);
        s.hi.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            s.lo[i] = pick(0, 3) == 0 ? pick(-3, 3) : 0;
            s.hi[i] = s.lo[i] + (pick(0, 6) == 0 ? 0 : pick(0, 6));
        }
        s.total = pick(-2, 15);
        s.ref = all_of(s.total, s.lo, s.hi);
        if (!s.ref.empty()) out.push_back(std::move(s));
    }
    return out;
}

// Checks one subtree against the brute-force list, which is in rank order,
// so the compositions sharing a prefix are contiguous: a node only needs its
// own entries and its two neighbours. Returns its children.
std::vector<bc::region> check_node(const bc::space& s, const std::vector<composition>& ref,
                                   const bc::region& r) {
    REQUIRE(r.is_subtree());
    const std::size_t a = static_cast<std::size_t>(r.begin().to_u64());
    const std::size_t b = static_cast<std::size_t>(r.end().to_u64());
    REQUIRE(a < b);
    REQUIRE(r.prefix() == shared_prefix(ref, a, b));
    // Exactly the compositions with that prefix.
    if (a > 0) REQUIRE_FALSE(starts_with(ref[a - 1], r.prefix()));
    if (b < ref.size()) REQUIRE_FALSE(starts_with(ref[b], r.prefix()));
    // describe is exact on subtrees.
    const std::vector<bc::part_range> ranges = bc::describe(s, r);
    for (std::size_t p = 0; p < s.parts(); ++p) {
        std::int32_t lo = ref[a][p], hi = ref[a][p];
        for (std::size_t i = a; i < b; ++i) {
            lo = std::min(lo, ref[i][p]);
            hi = std::max(hi, ref[i][p]);
        }
        REQUIRE(ranges[p].lo == lo);
        REQUIRE(ranges[p].hi == hi);
    }
    const std::vector<bc::region> kids = bc::children(s, r);
    if (b - a == 1) {
        REQUIRE(kids.empty());
        return kids;
    }
    REQUIRE(kids.size() >= 2);
    natural at = r.begin();
    for (const bc::region& kid : kids) {
        REQUIRE(kid.begin() == at);  // contiguous, in rank order
        REQUIRE(kid.prefix().size() > r.prefix().size());
        at = kid.end();
    }
    REQUIRE(at == r.end());
    return kids;
}

// Checks every subtree below r.
void check_tree(const bc::space& s, const std::vector<composition>& ref, const bc::region& r) {
    for (const bc::region& kid : check_node(s, ref, r)) check_tree(s, ref, kid);
}

}  // namespace

TEST_CASE("subtrees, children and describe match brute-force enumeration") {
    for (const small_space& ss : random_spaces(300, 31)) {
        INFO("total " << ss.total << ", parts " << ss.lo.size());
        const bc::space s(ss.total, ss.lo, ss.hi);
        const bc::region all = bc::whole(s);
        REQUIRE(all.size() == natural(ss.ref.size()));
        check_tree(s, ss.ref, all);
        // subtree() of every prefix of every composition.
        for (std::size_t r = 0; r < ss.ref.size(); r += 3) {
            for (std::size_t d = 0; d <= s.parts(); ++d) {
                const composition prefix(ss.ref[r].begin(),
                                         ss.ref[r].begin() + static_cast<long>(d));
                const bc::region t = bc::subtree(s, prefix);
                REQUIRE(starts_with(t.prefix(), prefix));
                REQUIRE(t.contains(natural(r)));
                check_node(s, ss.ref, t);
            }
        }
    }
}

TEST_CASE("intervals know whether they are subtrees, and describe bounds them") {
    std::mt19937 rng(77);
    for (const small_space& ss : random_spaces(200, 41)) {
        const bc::space s(ss.total, ss.lo, ss.hi);
        const std::size_t size = ss.ref.size();
        for (int trial = 0; trial < 20; ++trial) {
            std::size_t a = std::uniform_int_distribution<std::size_t>(0, size - 1)(rng);
            std::size_t b = std::uniform_int_distribution<std::size_t>(a + 1, size)(rng);
            const bc::region r = bc::interval(s, natural(a), natural(b));
            const composition shared = shared_prefix(ss.ref, a, b);
            REQUIRE(r.prefix() == shared);
            std::size_t with = 0;
            for (const composition& x : ss.ref) {
                if (starts_with(x, shared)) ++with;
            }
            REQUIRE(r.is_subtree() == (with == b - a));
            const std::vector<bc::part_range> ranges = bc::describe(s, r);
            for (std::size_t p = 0; p < s.parts(); ++p) {
                std::int32_t lo = ss.ref[a][p], hi = ss.ref[a][p];
                for (std::size_t i = a; i < b; ++i) {
                    lo = std::min(lo, ss.ref[i][p]);
                    hi = std::max(hi, ss.ref[i][p]);
                }
                REQUIRE(ranges[p].lo <= lo);
                REQUIRE(ranges[p].hi >= hi);
                if (p <= shared.size()) {  // fixed parts and the first free one are exact
                    REQUIRE(ranges[p].lo == lo);
                    REQUIRE(ranges[p].hi == hi);
                }
            }
        }
    }
}

TEST_CASE("split gives consecutive pieces whose sizes differ by at most one") {
    const bc::space s(30, composition(5, 0), composition(5, 12));
    const bc::region all = bc::whole(s);
    for (std::uint32_t pieces : {1u, 2u, 3u, 7u, 64u, 1000u}) {
        const std::vector<bc::region> parts = bc::split(s, all, pieces);
        natural at = all.begin();
        natural smallest = all.size(), largest;
        for (const bc::region& p : parts) {
            CHECK(p.begin() == at);
            at = p.end();
            smallest = std::min(smallest, p.size());
            largest = std::max(largest, p.size());
        }
        CHECK(at == all.end());
        CHECK(largest - smallest <= natural(1));
    }
    CHECK_THROWS_AS(bc::split(s, all, 0), std::invalid_argument);
}

TEST_CASE("invalid prefixes and intervals are rejected") {
    const bc::space s(10, composition{0, 0, 0}, composition{5, 5, 5});
    CHECK_THROWS_AS(bc::subtree(s, composition{6}), std::invalid_argument);  // above a bound
    CHECK_THROWS_AS(bc::subtree(s, composition{0, 0}),
                    std::invalid_argument);  // 10 left for one part
    CHECK_THROWS_AS(bc::subtree(s, composition{1, 1, 1, 1}), std::invalid_argument);
    CHECK_THROWS_AS(bc::interval(s, natural(5), natural(4)), std::out_of_range);
    CHECK_THROWS_AS(bc::interval(s, natural(0), s.size() + natural(1)), std::out_of_range);
    CHECK_THROWS_AS(bc::children(s, bc::interval(s, natural(1), natural(4))),
                    std::invalid_argument);
    CHECK(bc::interval(s, natural(3), natural(3)).empty());
    const bc::space none(20, composition{0, 0, 0}, composition{5, 5, 5});  // no composition
    CHECK(bc::whole(none).empty());
    CHECK(bc::children(none, bc::whole(none)).empty());
    CHECK(bc::dyadic_split(none, bc::whole(none), 3).empty());
}

TEST_CASE("a subspace is the subtree as a space of its own") {
    for (const small_space& ss : random_spaces(200, 53)) {
        if (ss.lo.size() < 2) continue;
        const bc::space s(ss.total, ss.lo, ss.hi);
        for (std::size_t r = 0; r < ss.ref.size(); r += 5) {
            for (std::size_t d = 1; d < s.parts(); ++d) {
                const composition prefix(ss.ref[r].begin(),
                                         ss.ref[r].begin() + static_cast<long>(d));
                const bc::space sub = s.subspace(prefix);
                const bc::region t = bc::subtree(s, prefix);
                REQUIRE(sub.parts() == s.parts() - d);
                REQUIRE(sub.size() == t.size());
                std::int32_t prefix_sum = 0;
                for (std::int32_t v : prefix) prefix_sum += v;
                REQUIRE(sub.total() == ss.total - prefix_sum);
                const std::uint64_t base = t.begin().to_u64();
                for (std::uint64_t q = 0; q < sub.size().to_u64(); ++q) {
                    const composition tail = sub.unrank(natural(q));
                    const composition& full = ss.ref[base + q];
                    REQUIRE(
                        std::equal(tail.begin(), tail.end(), full.begin() + static_cast<long>(d)));
                    REQUIRE(sub.rank(tail) == natural(q));
                }
            }
        }
    }
    const bc::space s(10, composition{0, 0, 0}, composition{5, 5, 5});
    CHECK_THROWS_AS(s.subspace(composition{1, 4, 5}), std::invalid_argument);  // no part left
    CHECK_THROWS_AS(s.subspace(composition{0, 0}),
                    std::invalid_argument);                  // 10 left, one part of 5
    CHECK(s.subspace(composition{0}).size() == natural(1));  // 5 + 5 only
}

TEST_CASE("equal dyadic blocks reproduce the region's rank sequence") {
    for (std::int32_t size : {7, 100, 1000, 1024, 4096, 12345}) {
        const bc::space s(size - 1, composition{0, 0}, composition{size - 1, size - 1});
        // Use the whole space and an interior interval as the region.
        for (const bc::region& r :
             {bc::whole(s),
              bc::interval(s, natural(3), natural(static_cast<std::uint64_t>(size) - 2))}) {
            if (r.empty()) continue;
            const bc::sequence global = bc::rank_sequence(s, r);
            const std::uint64_t count = global.capacity();
            std::vector<std::int32_t> pts(count * 2);
            global.points(0, count, pts.data());
            for (int levels = 0; levels <= 6; ++levels) {
                for (const bc::region& block : bc::dyadic_split(s, r, levels)) {
                    std::vector<std::int32_t> expected;
                    for (std::uint64_t i = 0; i < count; ++i) {
                        if (block.contains(natural(static_cast<std::uint64_t>(pts[2 * i])))) {
                            expected.push_back(pts[2 * i]);
                            expected.push_back(pts[2 * i + 1]);
                        }
                    }
                    const bc::sequence local = bc::rank_sequence(s, block);
                    REQUIRE(local.capacity() * 2 == expected.size());
                    std::vector<std::int32_t> got(expected.size());
                    local.points(0, local.capacity(), got.data());
                    REQUIRE(got == expected);
                }
            }
        }
    }
}

TEST_CASE("dyadic blocks reproduce the sequence on a space wider than 64 bits") {
    const bc::space s(100, composition(100, 0), composition(100, 100));  // 195-bit size
    const bc::region all = bc::whole(s);
    const bc::sequence global = bc::rank_sequence(s, all);
    const std::uint64_t count = 4096;
    std::vector<std::int32_t> pts(count * 100);
    global.points(0, count, pts.data());
    const std::vector<bc::region> blocks = bc::dyadic_split(s, all, 3);
    REQUIRE(blocks.size() == 8);
    for (const bc::region& block : blocks) {
        std::vector<std::int32_t> expected;
        for (std::uint64_t i = 0; i < count; ++i) {
            if (block.contains(s.rank(pts.data() + i * 100))) {
                expected.insert(expected.end(), pts.begin() + static_cast<long>(i * 100),
                                pts.begin() + static_cast<long>((i + 1) * 100));
            }
        }
        REQUIRE(!expected.empty());
        const bc::sequence local = bc::rank_sequence(s, block);
        std::vector<std::int32_t> got(expected.size());
        local.points(0, expected.size() / 100, got.data());
        CHECK(got == expected);
    }
}

TEST_CASE("region sequences stay inside the region and match the subspace") {
    const bc::space s(24, composition(6, 0), composition(6, 9));
    const bc::region t = bc::subtree(s, composition{2, 5});
    const bc::space sub = s.subspace(composition{2, 5});
    for (const bc::sequence& seq :
         {bc::rank_sequence(s, t), bc::rank_sequence(s, t, 9), bc::iid_sequence(s, t, 4)}) {
        const std::uint64_t count = seq.kind() == bc::sequence_kind::iid ? 500 : seq.capacity();
        if (seq.kind() != bc::sequence_kind::iid) REQUIRE(natural(seq.capacity()) == t.size());
        std::vector<std::int32_t> pts(count * 6);
        seq.points(0, count, pts.data());
        std::map<std::string, int> seen;
        for (std::uint64_t i = 0; i < count; ++i) {
            const composition x(pts.begin() + static_cast<long>(i * 6),
                                pts.begin() + static_cast<long>((i + 1) * 6));
            REQUIRE(t.contains(s.rank(x)));
            REQUIRE(x[0] == 2);
            REQUIRE(x[1] == 5);
            seen[std::to_string(s.rank(x).to_u64())] += 1;
        }
        if (seq.kind() != bc::sequence_kind::iid) REQUIRE(seen.size() == count);  // no repeats
    }
    // The region's rank sequence is the subspace's, shifted by the prefix.
    const bc::sequence in_region = bc::rank_sequence(s, t);
    const bc::sequence in_sub = bc::rank_sequence(sub);
    for (std::uint64_t i = 0; i < in_sub.capacity(); ++i) {
        CHECK(in_region.rank_at(i) == t.begin() + in_sub.rank_at(i));
    }
    CHECK_THROWS_AS(bc::rank_sequence(sub, t), std::out_of_range);  // t lies past sub's ranks
}
