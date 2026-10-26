// Property fuzzer for natural and space. Built with libFuzzer
// (BCOMP_FUZZ=ON, Clang) or with fuzz_driver.cpp as a plain test.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "bounded_compositions/bounded_compositions.hpp"

using bounded_compositions::natural;

namespace {

[[noreturn]] void fail(const char* what) {
    std::fprintf(stderr, "property violated: %s\n", what);
    std::abort();
}

void expect(bool ok, const char* what) {
    if (!ok) fail(what);
}

class reader {
  public:
    reader(const std::uint8_t* data, std::size_t size) : p_(data), end_(data + size) {}
    std::uint8_t byte() { return p_ < end_ ? *p_++ : 0; }
    std::uint64_t word() {
        std::uint64_t w = 0;
        for (int i = 0; i < 8; ++i) w = (w << 8) | byte();
        return w;
    }
    std::uint32_t u32() { return static_cast<std::uint32_t>(word()); }
    natural number() {
        const std::size_t count = byte() % 9;  // up to 512 bits
        std::vector<bounded_compositions::kernel::limb> limbs(count);
        for (auto& l : limbs) {
            // Bias towards the edge values where carries happen.
            const std::uint8_t kind = byte() % 4;
            l = kind == 0 ? 0 : kind == 1 ? ~std::uint64_t{0} : word();
        }
        return natural::from_limbs(limbs.data(), limbs.size());
    }

  private:
    const std::uint8_t* p_;
    const std::uint8_t* end_;
};

void check_natural(reader& in) {
    const natural a = in.number();
    const natural b = in.number();
    expect(natural::parse(a.to_string()) == a, "parse(to_string(a)) == a");
    expect((a + b) - b == a, "(a + b) - b == a");
    expect(a + b >= a && a + b >= b, "a + b >= both");
    expect(compare(a, b) == -compare(b, a), "compare is antisymmetric");
    if (a >= b) {
        expect((a - b) + b == a, "(a - b) + b == a");
    } else {
        bool threw = false;
        try {
            (void)(a - b);
        } catch (const std::domain_error&) {
            threw = true;
        }
        expect(threw, "a - b throws when a < b");
    }
    const std::uint32_t d = in.u32() | 1u;
    natural q = a;
    const std::uint32_t r = q.divmod_small(d);
    expect(r < d, "remainder below divisor");
    natural back = q;
    back.mul_small(d);
    expect(back + natural(r) == a, "q * d + r == a");
    const std::uint32_t m = in.u32();
    natural p = a;
    p.mul_small(m);
    if (m != 0) {
        expect(p.divmod_small(m) == 0 && p == a, "(a * m) / m == a");
    } else {
        expect(p.is_zero(), "a * 0 == 0");
    }
}

void check_space(reader& in) {
    const std::size_t n = 1 + static_cast<std::size_t>(in.byte() % 6);
    std::vector<std::int32_t> lo(n), hi(n);
    for (std::size_t i = 0; i < n; ++i) {
        lo[i] = static_cast<std::int32_t>(in.byte() % 9) - 4;
        hi[i] = lo[i] + in.byte() % 7;
    }
    const std::int32_t total = static_cast<std::int32_t>(in.byte() % 40) - 10;
    // Support limits on about half the inputs.
    bounded_compositions::support_limits sup;
    if (in.byte() % 2 == 1) {
        sup.min_count = static_cast<std::int32_t>(in.byte() % (n + 1));
        sup.max_count = sup.min_count + static_cast<std::int32_t>(in.byte() % (n + 1));
        if (in.byte() % 2 == 1) {
            sup.min_size.resize(n);
            for (auto& m : sup.min_size) m = 1 + in.byte() % 4;
        }
    }
    // Group limits on about half the inputs: one or two groups of any parts.
    std::vector<bounded_compositions::group_limit> groups;
    if (in.byte() % 2 == 1) {
        const int count = 1 + in.byte() % 2;
        for (int g = 0; g < count; ++g) {
            bounded_compositions::group_limit gl;
            const std::uint8_t mask = in.byte();
            for (std::size_t i = 0; i < n; ++i) {
                if (((static_cast<unsigned>(mask) >> i) & 1u) != 0) {
                    gl.parts.push_back(static_cast<std::int32_t>(i));
                }
            }
            if (gl.parts.empty()) gl.parts.push_back(0);
            const std::int32_t from = static_cast<std::int32_t>(in.byte() % 25) - 8;
            if (in.byte() % 3 != 0) gl.min_sum = from;
            if (in.byte() % 3 != 0) gl.max_sum = from + in.byte() % 12;
            groups.push_back(gl);
        }
    }
    const bounded_compositions::space s(total, lo, hi, sup, groups);
    if (s.size() > natural(20000)) return;  // keep each input fast

    // The size against a count over the box.
    {
        std::uint64_t brute = 0;
        std::vector<std::int32_t> y = lo;
        for (;;) {
            std::int64_t sum = 0;
            std::int32_t nonzero = 0;
            bool ok = true;
            for (std::size_t i = 0; i < n; ++i) {
                sum += y[i];
                if (y[i] == 0) continue;
                ++nonzero;
                if (!sup.min_size.empty() && std::abs(y[i]) < sup.min_size[i]) ok = false;
            }
            for (const auto& g : groups) {
                std::int64_t part_sum = 0;
                for (std::int32_t p : g.parts) part_sum += y[static_cast<std::size_t>(p)];
                if (part_sum < g.min_sum || part_sum > g.max_sum) ok = false;
            }
            if (ok && sum == total && nonzero >= sup.min_count && nonzero <= sup.max_count) ++brute;
            std::size_t i = 0;
            while (i < n && y[i] == hi[i]) {
                y[i] = lo[i];
                ++i;
            }
            if (i == n) break;
            ++y[i];
        }
        expect(s.size() == natural(brute), "size equals a count over the box");
    }
    std::vector<std::int32_t> x(n);
    const bool any = s.first(x.data());
    expect(any == !s.empty(), "first succeeds exactly when non-empty");
    std::uint64_t count = 0;
    if (any) {
        do {
            expect(s.contains(x.data()), "walk stays in the space");
            expect(s.rank(x.data()) == natural(count), "rank equals position");
            expect(s.unrank(natural(count)) == x, "unrank inverts rank");
            ++count;
        } while (s.next(x.data()));
    }
    expect(natural(count) == s.size(), "walk visits size() compositions");

    // repair gives a composition no farther from z than any other.
    if (count > 0) {
        std::vector<double> z(n);
        for (std::size_t i = 0; i < n; ++i) z[i] = static_cast<double>(in.byte() % 61) / 4.0 - 6.0;
        const std::vector<std::int32_t> fixed_up = bounded_compositions::repair(s, z);
        expect(s.contains(fixed_up), "repair gives a composition of the space");
        auto dist = [&](const std::int32_t* y) {
            double d = 0.0;
            for (std::size_t i = 0; i < n; ++i) d += (y[i] - z[i]) * (y[i] - z[i]);
            return d;
        };
        const double got = dist(fixed_up.data());
        std::vector<std::int32_t> y(n);
        s.first(y.data());
        do {
            expect(dist(y.data()) >= got - 1e-9 * std::max(1.0, got), "repair is nearest");
        } while (s.next(y.data()));
    }

    // A rank sequence, scrambled or not, is a permutation of the ranks.
    const std::uint64_t seed = in.word();
    const bounded_compositions::sequence seq = (seed & 1u) != 0
                                                   ? bounded_compositions::rank_sequence(s, seed)
                                                   : bounded_compositions::rank_sequence(s);
    expect(seq.capacity() == count, "rank sequence capacity equals the size");
    std::vector<std::int32_t> pts(static_cast<std::size_t>(count) * n);
    seq.points(0, count, pts.data());
    std::vector<bool> hit(static_cast<std::size_t>(count), false);
    for (std::uint64_t i = 0; i < count; ++i) {
        const std::uint64_t r = s.rank(pts.data() + i * n).to_u64();
        expect(r < count && !hit[r], "rank sequence never repeats a rank");
        hit[r] = true;
    }

    // Children tile the whole space, and every dyadic block's own sequence
    // is the global sequence restricted to that block.
    const bounded_compositions::region all = bounded_compositions::whole(s);
    natural at;
    for (const auto& kid : bounded_compositions::children(s, all)) {
        expect(kid.begin() == at && !kid.empty(), "children are contiguous and non-empty");
        at = kid.end();
    }
    expect(at == all.end() || count == 1, "children cover the whole space");
    const bounded_compositions::sequence global = bounded_compositions::rank_sequence(s, all);
    const int levels = static_cast<int>(in.byte() % 5);
    for (const auto& block : bounded_compositions::dyadic_split(s, all, levels)) {
        const bounded_compositions::sequence local = bounded_compositions::rank_sequence(s, block);
        std::uint64_t next = 0;
        for (std::uint64_t i = 0; i < count; ++i) {
            const natural r = global.rank_at(i);
            if (!block.contains(r)) continue;
            expect(local.rank_at(next) == r, "dyadic block reproduces the global order");
            ++next;
        }
        expect(next == local.capacity(), "dyadic block has the global points of its ranks");
    }

    // Any interval lists its ranks in the order of `seq`, resumes after a
    // prefix of it, and a fresh_sampler hands out exactly its unvisited ranks.
    if (count > 0) {
        std::vector<std::uint64_t> order(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) order[i] = s.rank(pts.data() + i * n).to_u64();
        const std::uint64_t a = in.word() % count;
        const std::uint64_t b = a + 1 + in.word() % (count - a);
        const bounded_compositions::region r =
            bounded_compositions::interval(s, natural(a), natural(b));
        const bounded_compositions::sequence local =
            (seed & 1u) != 0 ? bounded_compositions::rank_sequence(s, r, seed)
                             : bounded_compositions::rank_sequence(s, r);
        const std::uint64_t cut = in.word() % (count + 1);
        std::uint64_t next = 0, before_cut = 0;
        for (std::uint64_t i = 0; i < count; ++i) {
            if (order[i] < a || order[i] >= b) continue;
            expect(local.rank_at(next) == natural(order[i]), "a region follows the space's order");
            ++next;
            if (i < cut) ++before_cut;
        }
        expect(next == local.capacity(), "a region's sequence covers it");
        expect(local.overlap(seq, cut) == before_cut, "overlap counts the outer points inside");
        bounded_compositions::visited_set seen(s);
        for (std::uint64_t i = 0; i < cut; ++i) seen.insert(natural(order[i]));
        bounded_compositions::fresh_sampler fresh(bounded_compositions::rank_sequence(s, r), seen);
        std::vector<std::int32_t> got(static_cast<std::size_t>(next) * n);
        const std::uint64_t m = fresh.take(next, got.data());
        expect(m == next - before_cut, "fresh_sampler hands out every unvisited point");
        expect(seen.size() == cut + m, "fresh_sampler marks what it hands out");
    }

    // The unit-cube codec inverts on every composition, and coordinate
    // sequence points are compositions.
    if (count == 0) return;
    std::vector<std::uint64_t> fixed(n);
    std::vector<double> unit(n);
    std::vector<std::int32_t> back(n);
    // A spread of at most 32 compositions keeps each input fast; encode does
    // a 64-step search per part.
    const std::uint64_t stride = count > 32 ? count / 32 : 1;
    for (std::uint64_t i = 0; i < count; i += stride) {
        const std::int32_t* xi = pts.data() + i * n;
        bounded_compositions::encode(s, xi, fixed.data());
        bounded_compositions::decode(s, fixed.data(), back.data());
        expect(std::equal(back.begin(), back.end(), xi), "decode(encode(x)) == x, 64-bit");
        bounded_compositions::encode(s, xi, unit.data());  // small spaces: never too narrow
        bounded_compositions::decode(s, unit.data(), back.data());
        expect(std::equal(back.begin(), back.end(), xi), "decode(encode(x)) == x, double");
    }
    const bounded_compositions::sequence coord =
        (seed & 2u) != 0 ? bounded_compositions::coordinate_sequence(s, seed)
                         : bounded_compositions::coordinate_sequence(s);
    std::vector<std::int32_t> cpts(64 * n);
    coord.points(seed % 1000, 64, cpts.data());
    for (std::size_t i = 0; i < 64; ++i) {
        expect(s.contains(cpts.data() + i * n), "coordinate points are compositions");
    }

    // The logarithmic mode: same emptiness, close sizes, valid decodes, and
    // encode inverts decode, with the same support limits.
    const bounded_compositions::space logs(total, lo, hi, sup, groups,
                                           bounded_compositions::counting::logarithmic);
    expect(!logs.empty(), "logarithmic space is non-empty too");
    expect(std::abs(logs.log_size() - s.log_size()) <= 1e-11 * std::max(1.0, s.log_size()),
           "logarithmic size matches the exact one");
    for (std::size_t j = 0; j + 1 < n; ++j) fixed[j] = in.word();
    bounded_compositions::decode(logs, fixed.data(), back.data());
    expect(logs.contains(back.data()), "logarithmic decode gives a composition");
    for (std::uint64_t i = 0; i < count; i += stride) {
        const std::int32_t* xi = pts.data() + i * n;
        bounded_compositions::encode(logs, xi, unit.data());
        bounded_compositions::decode(logs, unit.data(), back.data());
        expect(std::equal(back.begin(), back.end(), xi), "logarithmic decode(encode(x)) == x");
    }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    reader in(data, size);
    if (in.byte() % 2 == 0) {
        check_natural(in);
    } else {
        check_space(in);
    }
    return 0;
}
