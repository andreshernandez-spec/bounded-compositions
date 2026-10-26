#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "bounded_compositions/bounded_compositions.hpp"
#include "bounded_compositions/kernel/philox.hpp"
#include "bounded_compositions/kernel/radical.hpp"

namespace bc = bounded_compositions;
namespace k = bounded_compositions::kernel;
using bc::natural;

namespace {

struct philox_case {
    k::limb counter[4];
    k::limb key[2];
    k::limb block[4];
};

const philox_case philox_cases[] = {
#include "data/philox_vectors.inc"
};

// Two parts summing to size - 1 with bounds 0..size-1: exactly `size`
// compositions, and rank r is the composition (r, size - 1 - r).
bc::space line_space(std::int32_t size) {
    return bc::space(size - 1, std::vector<std::int32_t>{0, 0},
                     std::vector<std::int32_t>{size - 1, size - 1});
}

std::vector<std::uint64_t> small_ranks(const bc::sequence& seq, std::uint64_t count) {
    std::vector<std::int32_t> out(count * seq.domain().parts());
    seq.points(0, count, out.data());
    std::vector<std::uint64_t> ranks(count);
    for (std::uint64_t i = 0; i < count; ++i) {
        ranks[i] = seq.domain().rank(out.data() + i * seq.domain().parts()).to_u64();
    }
    return ranks;
}

// Largest gap between consecutive sampled ranks, counting the ends -1 and
// size, over every prefix: returns false as soon as some prefix of N points
// has a gap of 2 * size / N or more.
bool every_prefix_spread(const std::vector<std::uint64_t>& ranks, std::uint64_t size) {
    std::set<std::int64_t> seen{-1, static_cast<std::int64_t>(size)};
    std::int64_t widest = static_cast<std::int64_t>(size) + 1;
    std::multiset<std::int64_t> gaps{widest};
    for (std::size_t i = 0; i < ranks.size(); ++i) {
        const std::int64_t r = static_cast<std::int64_t>(ranks[i]);
        const auto [it, inserted] = seen.insert(r);
        if (!inserted) return false;  // repeated rank
        const std::int64_t before = *std::prev(it);
        const std::int64_t after = *std::next(it);
        gaps.erase(gaps.find(after - before));
        gaps.insert(r - before);
        gaps.insert(after - r);
        const std::int64_t n = static_cast<std::int64_t>(i + 1);
        if (*gaps.rbegin() * n >= 2 * static_cast<std::int64_t>(size)) return false;
    }
    return true;
}

}  // namespace

TEST_CASE("philox4x64-10 matches numpy") {
    for (const philox_case& c : philox_cases) {
        const k::philox_block b = k::philox4x64_10(c.counter, c.key);
        for (int i = 0; i < 4; ++i) CHECK(b.v[i] == c.block[i]);
    }
}

TEST_CASE("bit_reverse reverses the low bits") {
    CHECK(k::bit_reverse(1, 64) == k::limb{1} << 63);
    CHECK(k::bit_reverse(1, 1) == 1);
    CHECK(k::bit_reverse(0b0011, 4) == 0b1100);
    CHECK(k::bit_reverse(0b10110, 5) == 0b01101);
    CHECK(k::bit_reverse(~k::limb{0}, 64) == ~k::limb{0});
}

TEST_CASE("rank_sequence visits every rank once and every prefix stays spread") {
    std::vector<std::int32_t> sizes;
    for (std::int32_t s = 1; s <= 600; ++s) sizes.push_back(s);
    for (std::int32_t s : {1000, 1024, 1025, 3001, 4096, 5000}) sizes.push_back(s);
    for (std::int32_t size : sizes) {
        INFO("size " << size);
        const bc::space s = line_space(size);
        for (int scrambled = 0; scrambled < 2; ++scrambled) {
            const bc::sequence seq = scrambled
                                         ? bc::rank_sequence(s, 1000u + static_cast<unsigned>(size))
                                         : bc::rank_sequence(s);
            REQUIRE(seq.capacity() == static_cast<std::uint64_t>(size));
            const std::vector<std::uint64_t> ranks = small_ranks(seq, seq.capacity());
            std::vector<std::uint64_t> sorted = ranks;
            std::sort(sorted.begin(), sorted.end());
            for (std::uint64_t i = 0; i < sorted.size(); ++i) REQUIRE(sorted[i] == i);
            REQUIRE(every_prefix_spread(ranks, static_cast<std::uint64_t>(size)));
            CHECK(seq.rank_at(seq.capacity() - 1) == natural(ranks.back()));
        }
    }
}

TEST_CASE("the first 2^j points of an unscrambled rank_sequence are the even grid") {
    const bc::space s = line_space(1024);  // 2^10 ranks
    const bc::sequence seq = bc::rank_sequence(s);
    const std::vector<std::uint64_t> ranks = small_ranks(seq, 1024);
    for (int j = 0; j <= 10; ++j) {
        std::vector<std::uint64_t> head(ranks.begin(), ranks.begin() + (std::ptrdiff_t{1} << j));
        std::sort(head.begin(), head.end());
        for (std::uint64_t c = 0; c < head.size(); ++c) CHECK(head[c] == c << (10 - j));
    }
}

TEST_CASE("rank_sequence on a space wider than 64 bits") {
    // C(199, 99): ranks need 195 bits, so the index covers the top 64 bits.
    const bc::space s(100, std::vector<std::int32_t>(100, 0), std::vector<std::int32_t>(100, 100));
    for (int scrambled = 0; scrambled < 2; ++scrambled) {
        const bc::sequence seq = scrambled ? bc::rank_sequence(s, 7) : bc::rank_sequence(s);
        CHECK(seq.capacity() > (std::uint64_t{1} << 63));
        const std::uint64_t count = 3000;
        std::vector<std::int32_t> out(count * 100);
        seq.points(0, count, out.data());
        std::set<natural> ranks;
        for (std::uint64_t i = 0; i < count; ++i) {
            const natural r = s.rank(out.data() + i * 100);
            CHECK(r < s.size());
            ranks.insert(r);
            if (i % 97 == 0) CHECK(seq.rank_at(i) == r);
        }
        CHECK(ranks.size() == count);
        // Far into the sequence, random access still agrees with the counting.
        for (std::uint64_t i : {std::uint64_t{1} << 40, (std::uint64_t{1} << 62) + 12345}) {
            const natural r = seq.rank_at(i);
            CHECK(r < s.size());
            std::vector<std::int32_t> x(100);
            seq.points(i, 1, x.data());
            CHECK(s.rank(x) == r);
        }
    }
    // Unscrambled: the first 1024 points are the ranks c * 2^(195 - 10) below the size.
    const bc::sequence seq = bc::rank_sequence(s);
    natural step(1);
    for (int b = 0; b < 195 - 10; ++b) step.mul_small(2);
    std::set<natural> expected;
    for (natural r; r < s.size(); r += step) expected.insert(r);
    std::set<natural> got;
    for (std::uint64_t i = 0; i < expected.size(); ++i) got.insert(seq.rank_at(i));
    CHECK(got == expected);
}

TEST_CASE("the scaled variant repeats ranks, as measured for the old mapping") {
    const bc::space s = line_space(100);
    const std::vector<std::uint64_t> ranks = small_ranks(bc::scaled_rank_sequence(s), 100);
    CHECK(std::set<std::uint64_t>(ranks.begin(), ranks.end()).size() == 84);
}

TEST_CASE("take extends the sequence, seek resumes it, and capacity is enforced") {
    const bc::space s(40, std::vector<std::int32_t>(6, -2), std::vector<std::int32_t>(6, 12));
    for (const bc::sequence& proto : {bc::rank_sequence(s), bc::rank_sequence(s, 3),
                                      bc::scaled_rank_sequence(s), bc::iid_sequence(s, 11)}) {
        bc::sequence seq = proto;
        const std::size_t n = s.parts();
        std::vector<std::int32_t> all(1100 * n), first(1000 * n), more(100 * n);
        seq.points(0, 1100, all.data());
        CHECK(seq.take(1000, first.data()) == 0);
        CHECK(seq.take(100, more.data()) == 1000);
        CHECK(seq.position() == 1100);
        first.insert(first.end(), more.begin(), more.end());
        CHECK(first == all);
        seq.seek(500);
        std::vector<std::int32_t> again(100 * n);
        CHECK(seq.take(100, again.data()) == 500);
        CHECK(std::equal(again.begin(), again.end(), all.begin() + 500 * static_cast<long>(n)));
    }
    bc::sequence seq = bc::rank_sequence(line_space(50));
    std::vector<std::int32_t> out(60 * 2);
    seq.take(40, out.data());
    CHECK_THROWS_AS(seq.take(11, out.data()), std::out_of_range);
    CHECK(seq.position() == 40);  // the failed take reserved nothing
    CHECK(seq.take(10, out.data()) == 40);
    CHECK_THROWS_AS(seq.points(45, 6, out.data()), std::out_of_range);
    CHECK_THROWS_AS(seq.rank_at(50), std::out_of_range);
    CHECK_THROWS_AS(seq.seek(51), std::out_of_range);
    const bc::space empty(5, std::vector<std::int32_t>{3, 3}, std::vector<std::int32_t>{4, 4});
    CHECK(bc::rank_sequence(empty).capacity() == 0);
    CHECK(bc::iid_sequence(empty, 1).capacity() == 0);
    CHECK_THROWS_AS(bc::rank_sequence(empty).points(0, 1, out.data()), std::out_of_range);
}

TEST_CASE("points are identical with any executor and thread count") {
    const bc::space s(60, std::vector<std::int32_t>(12, 0), std::vector<std::int32_t>(12, 15));
    const std::uint64_t count = 20000;
    const std::size_t n = s.parts();
    for (const bc::sequence& seq :
         {bc::rank_sequence(s), bc::rank_sequence(s, 99), bc::iid_sequence(s, 5)}) {
        std::vector<std::int32_t> reference(count * n);
        seq.points(123, count, reference.data());
        for (unsigned threads : {1u, 2u, 8u, 13u}) {
            bc::thread_pool pool(threads);
            std::vector<std::int32_t> out(count * n, -1);
            seq.points(123, count, out.data(), &pool);
            CHECK(out == reference);
        }
    }
}

TEST_CASE("concurrent take on one sequence hands out disjoint consecutive ranges") {
    const bc::space s = line_space(5000);
    bc::sequence seq = bc::rank_sequence(s);
    std::vector<std::int32_t> reference(5000 * 2);
    seq.points(0, 5000, reference.data());
    std::vector<std::int32_t> out(5000 * 2, -1);
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&] {
            std::vector<std::int32_t> chunk(7 * 2);
            for (;;) {
                std::uint64_t begin;
                try {
                    begin = seq.take(7, chunk.data());
                } catch (const std::out_of_range&) {
                    return;  // fewer than 7 left
                }
                std::copy(chunk.begin(), chunk.end(), out.begin() + static_cast<long>(begin * 2));
            }
        });
    }
    for (auto& th : threads) th.join();
    const std::uint64_t taken = seq.position();
    CHECK(taken == 5000 - 5000 % 7);
    CHECK(std::equal(out.begin(), out.begin() + static_cast<long>(taken * 2), reference.begin()));
}

TEST_CASE("iid draws are uniform over the ranks") {
    // C(9, 3) = 84 compositions; 500 expected hits each.
    const bc::space s(6, std::vector<std::int32_t>(4, 0), std::vector<std::int32_t>(4, 6));
    REQUIRE(s.size() == natural(84));
    const std::uint64_t draws = 84 * 500;
    const std::vector<std::uint64_t> ranks = small_ranks(bc::iid_sequence(s, 2026), draws);
    std::vector<double> hits(84, 0.0);
    for (std::uint64_t r : ranks) hits[r] += 1.0;
    double chi2 = 0.0;
    for (double h : hits) chi2 += (h - 500.0) * (h - 500.0) / 500.0;
    // Wilson-Hilferty upper quantile at p = 1e-6 for 83 degrees of freedom.
    const double df = 83.0, z = 4.753;
    const double limit = df * std::pow(1.0 - 2.0 / (9.0 * df) + z * std::sqrt(2.0 / (9.0 * df)), 3);
    INFO("chi2 = " << chi2 << ", limit = " << limit);
    CHECK(chi2 < limit);
    // Different seeds give different streams; the same seed the same one.
    CHECK(small_ranks(bc::iid_sequence(s, 2026), 50) == small_ranks(bc::iid_sequence(s, 2026), 50));
    CHECK(small_ranks(bc::iid_sequence(s, 2026), 50) != small_ranks(bc::iid_sequence(s, 2027), 50));
}
