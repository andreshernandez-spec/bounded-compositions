// Time per point, walking one point at a time (space::unrank, decode of one
// point) against the batched walks behind sequence::points and decode of many
// points, as the table outgrows the caches. Single thread.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "bounded_compositions/bounded_compositions.hpp"

namespace bc = bounded_compositions;

namespace {

using clock_type = std::chrono::steady_clock;

double ns_per(clock_type::time_point a, clock_type::time_point b, std::size_t count) {
    return std::chrono::duration<double, std::nano>(b - a).count() / static_cast<double>(count);
}

}  // namespace

int main() {
    struct shape {
        std::size_t parts;
        std::int32_t upper;
        std::int32_t total;
        std::size_t count;
    };
    std::printf("%6s %8s %10s %12s %12s %12s %12s\n", "parts", "total", "table MiB", "unrank 1",
                "iid points", "decode 1", "decode many");
    for (const shape& sh : {shape{13, 2000, 12000, 100000}, shape{40, 2000, 40000, 20000},
                            shape{100, 500, 25000, 5000}}) {
        const std::vector<std::int32_t> lo(sh.parts, 0), hi(sh.parts, sh.upper);
        const bc::space s(sh.total, lo, hi);
        const std::size_t n = sh.parts;
        std::vector<std::int32_t> x(sh.count * n), out(sh.count * n);
        const bc::sequence iid = bc::iid_sequence(s, 1);
        iid.points(0, sh.count, x.data());
        std::vector<bc::natural> ranks;
        ranks.reserve(sh.count);
        for (std::size_t p = 0; p < sh.count; ++p) ranks.push_back(s.rank(x.data() + p * n));

        auto t0 = clock_type::now();
        for (std::size_t p = 0; p < sh.count; ++p) s.unrank(ranks[p], out.data() + p * n);
        auto t1 = clock_type::now();
        iid.points(0, sh.count, out.data());
        auto t2 = clock_type::now();
        const double unrank_one = ns_per(t0, t1, sh.count);
        const double points = ns_per(t1, t2, sh.count);

        std::mt19937_64 rng(2);
        std::vector<std::uint64_t> u(sh.count * (n - 1));
        for (auto& c : u) c = rng();
        t0 = clock_type::now();
        for (std::size_t p = 0; p < sh.count; ++p) {
            bc::decode(s, u.data() + p * (n - 1), out.data() + p * n);
        }
        t1 = clock_type::now();
        std::vector<std::int32_t> many(sh.count * n);
        bc::decode(s, u.data(), sh.count, many.data());
        t2 = clock_type::now();
        if (many != out) std::printf("decode of many points differs from decode of each\n");
        std::printf("%6zu %8d %10.1f %12.1f %12.1f %12.1f %12.1f  ns per point\n", n, sh.total,
                    static_cast<double>(s.table_bytes()) / (1024.0 * 1024.0), unrank_one, points,
                    ns_per(t0, t1, sh.count), ns_per(t1, t2, sh.count));
    }
}
