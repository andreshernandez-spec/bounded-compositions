// Property fuzzer for natural and space. Built with libFuzzer
// (BCOMP_FUZZ=ON, Clang) or with fuzz_driver.cpp as a plain test.
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
    const std::size_t n = 1 + in.byte() % 6;
    std::vector<std::int32_t> lo(n), hi(n);
    for (std::size_t i = 0; i < n; ++i) {
        lo[i] = static_cast<std::int32_t>(in.byte() % 9) - 4;
        hi[i] = lo[i] + in.byte() % 7;
    }
    const std::int32_t total = static_cast<std::int32_t>(in.byte() % 40) - 10;
    const bounded_compositions::space s(total, lo, hi);
    if (s.size() > natural(20000)) return;  // keep each input fast
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
