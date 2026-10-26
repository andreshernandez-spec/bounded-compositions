// Time per unrank as the total grows, at a fixed number of parts. Unrank does
// a binary search per part, so the time should grow with log(total), not with
// the total.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "bounded_compositions/bounded_compositions.hpp"

namespace {

bounded_compositions::natural random_below(const bounded_compositions::natural& bound,
                                           std::mt19937_64& rng) {
    const int bits = bound.bit_length();
    std::vector<bounded_compositions::kernel::limb> limbs(bound.limb_count());
    for (;;) {
        for (auto& l : limbs) l = rng();
        if (bits % 64 != 0)
            limbs.back() &= (bounded_compositions::kernel::limb{1} << (bits % 64)) - 1;
        bounded_compositions::natural r =
            bounded_compositions::natural::from_limbs(limbs.data(), limbs.size());
        if (r < bound) return r;
    }
}

}  // namespace

int main() {
    const std::size_t parts = 20;
    std::printf("%8s %10s %12s %14s %12s\n", "parts", "total", "size bits", "table MiB",
                "ns/unrank");
    for (std::int32_t total : {100, 1000, 10000, 100000}) {
        const std::vector<std::int32_t> lo(parts, 0), hi(parts, total);
        const bounded_compositions::space s(total, lo, hi);
        std::mt19937_64 rng(1);
        const int count = 20000;
        std::vector<bounded_compositions::natural> ranks;
        ranks.reserve(count);
        for (int i = 0; i < count; ++i) ranks.push_back(random_below(s.size(), rng));
        std::vector<std::int32_t> out(parts);
        std::int64_t checksum = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (const auto& r : ranks) {
            s.unrank(r, out.data());
            checksum += out[0];
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / count;
        std::printf("%8zu %10d %12d %14.1f %12.1f   (checksum %lld)\n", parts, total,
                    s.size().bit_length(), static_cast<double>(s.table_bytes()) / (1 << 20), ns,
                    static_cast<long long>(checksum));
    }
}
