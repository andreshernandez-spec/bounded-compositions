// Batched walks (src/walk_batch.hpp) against the kernel's one-point walks.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "bounded_compositions/bounded_compositions.hpp"
#include "bounded_compositions/kernel/walk.hpp"
#include "walk_batch.hpp"

namespace bc = bounded_compositions;
namespace k = bounded_compositions::kernel;

namespace {

// Spaces of every kind the walks distinguish: plain, negative bounds, support
// limits with semicontinuous parts, groups, a subspace (whose walk starts
// from a later layer), wide counts, and the same in the logarithmic mode for
// decode.
std::vector<bc::space> spaces() {
    std::vector<bc::space> out;
    out.emplace_back(40, std::vector<std::int32_t>{0, 0, 0, 0, 0},
                     std::vector<std::int32_t>{20, 3, 15, 9, 30});
    out.emplace_back(3, std::vector<std::int32_t>{-5, -2, 0, -7},
                     std::vector<std::int32_t>{6, 4, 5, 3});
    out.emplace_back(12, std::vector<std::int32_t>{-4, -3, 0, -2, 0, 0},
                     std::vector<std::int32_t>{6, 5, 7, 6, 4, 8},
                     bc::support_limits{1, 4, {2, 1, 3, 2, 1, 2}});
    out.emplace_back(
        30, std::vector<std::int32_t>(7, 0), std::vector<std::int32_t>(7, 10), bc::support_limits{},
        std::vector<bc::group_limit>{{{0, 1, 2}, 6, 14}, {{2, 3}, 0, 8}, {{4, 5, 6}, 9, 25}});
    out.push_back(out.back().subspace({3, 2}));
    out.push_back(out[2].subspace({-4}));
    out.emplace_back(3000, std::vector<std::int32_t>(30, 0), std::vector<std::int32_t>(30, 400));
    out.emplace_back(3000, std::vector<std::int32_t>(30, 0), std::vector<std::int32_t>(30, 400),
                     bc::counting::logarithmic);
    out.emplace_back(12, std::vector<std::int32_t>{-4, -3, 0, -2, 0, 0},
                     std::vector<std::int32_t>{6, 5, 7, 6, 4, 8},
                     bc::support_limits{1, 4, {2, 1, 3, 2, 1, 2}}, bc::counting::logarithmic);
    out.emplace_back(
        30, std::vector<std::int32_t>(7, 0), std::vector<std::int32_t>(7, 10), bc::support_limits{},
        std::vector<bc::group_limit>{{{0, 1, 2}, 6, 14}, {{2, 3}, 0, 8}, {{4, 5, 6}, 9, 25}},
        bc::counting::logarithmic);
    out.push_back(out.back().subspace({3, 2}));
    out.emplace_back(9, std::vector<std::int32_t>{1}, std::vector<std::int32_t>{20});
    return out;
}

std::vector<k::limb> random_rank(const bc::natural& size, std::mt19937_64& rng) {
    const int bits = size.bit_length();
    std::vector<k::limb> limbs(size.limb_count());
    for (;;) {
        for (auto& l : limbs) l = rng();
        if (bits % 64 != 0) limbs.back() &= (k::limb{1} << (bits % 64)) - 1;
        if (bc::natural::from_limbs(limbs.data(), limbs.size()) < size) return limbs;
    }
}

}  // namespace

TEST_CASE("batched unrank and decode give the kernel's results point by point") {
    std::mt19937_64 rng(7);
    for (const bc::space& s : spaces()) {
        const k::space_view v = s.view();
        const std::size_t n = s.parts();
        const std::size_t dims = n - 1;
        for (std::size_t count :
             {std::size_t{1}, std::size_t{15}, std::size_t{16}, std::size_t{17}, std::size_t{50}}) {
            // Unused columns in the outputs and codes check the strides.
            const std::size_t stride = n + 2;
            const std::size_t u_stride = dims + 3;
            std::vector<k::limb> u(count * u_stride);
            for (std::size_t p = 0; p < count; ++p) {
                for (std::size_t i = 0; i < dims; ++i) {
                    // The ends of [0, 1) too.
                    const std::uint64_t r = rng();
                    u[p * u_stride + i] = p == 0 ? 0 : p == 1 ? ~std::uint64_t{0} : r;
                }
            }
            std::vector<std::int32_t> batch(count * stride, -99), one(count * stride, -99);
            REQUIRE(bc::detail::decode_points(v, u.data(), u_stride, count, batch.data(), stride) ==
                    k::status::ok);
            for (std::size_t p = 0; p < count; ++p) {
                REQUIRE(k::decode(v, u.data() + p * u_stride, one.data() + p * stride) ==
                        k::status::ok);
            }
            CHECK(batch == one);

            if (s.counting_mode() != bc::counting::exact) continue;
            const int width = static_cast<int>(s.size().limb_count());
            std::vector<k::limb> ranks;
            for (std::size_t p = 0; p < count; ++p) {
                // The first and last ranks too.
                std::vector<k::limb> r = random_rank(s.size(), rng);
                if (p == 0) r.assign(r.size(), 0);
                if (p == 1) {
                    const bc::natural last = s.size() - bc::natural(1);
                    r.assign(r.size(), 0);
                    for (std::size_t l = 0; l < last.limb_count(); ++l) r[l] = last.limbs()[l];
                }
                ranks.insert(ranks.end(), r.begin(), r.end());
            }
            std::fill(batch.begin(), batch.end(), -99);
            std::fill(one.begin(), one.end(), -99);
            REQUIRE(bc::detail::unrank_points(v, ranks.data(), width, count, batch.data(),
                                              stride) == k::status::ok);
            for (std::size_t p = 0; p < count; ++p) {
                REQUIRE(k::unrank(v, ranks.data() + p * static_cast<std::size_t>(width), width,
                                  one.data() + p * stride) == k::status::ok);
            }
            CHECK(batch == one);
        }
    }
}

TEST_CASE("batched walks reject a rank past the end and an empty space but not zero points") {
    const bc::space s(10, std::vector<std::int32_t>{0, 0, 0}, std::vector<std::int32_t>{5, 5, 5});
    const int width = static_cast<int>(s.size().limb_count());
    std::vector<k::limb> ranks(2 * static_cast<std::size_t>(width), 0);
    ranks[static_cast<std::size_t>(width)] = s.size().to_u64();  // the second point
    std::vector<std::int32_t> out(6);
    CHECK(bc::detail::unrank_points(s.view(), ranks.data(), width, 2, out.data(), 3) ==
          k::status::out_of_range);

    const bc::space empty(20, std::vector<std::int32_t>{0, 0}, std::vector<std::int32_t>{5, 5});
    const k::limb zero = 0;
    const k::limb u = 0;
    CHECK(bc::detail::unrank_points(empty.view(), &zero, 1, 1, out.data(), 2) ==
          k::status::out_of_range);
    CHECK(bc::detail::decode_points(empty.view(), &u, 1, 1, out.data(), 2) ==
          k::status::out_of_range);
    // No points is no error, as with a loop over none.
    CHECK(bc::detail::decode_points(empty.view(), &u, 1, 0, out.data(), 2) == k::status::ok);
    CHECK(bc::detail::unrank_points(empty.view(), &zero, 1, 0, out.data(), 2) == k::status::ok);
}

TEST_CASE("decode of many points matches decode of each") {
    const bc::space s(50, std::vector<std::int32_t>(6, -2), std::vector<std::int32_t>(6, 20));
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const std::size_t points = 300;  // more than one conversion chunk
    std::vector<double> u(points * 5);
    for (double& x : u) x = unit(rng);
    std::vector<std::int32_t> many(points * 6), each(points * 6);
    bc::decode(s, u.data(), points, many.data());
    for (std::size_t p = 0; p < points; ++p) bc::decode(s, u.data() + p * 5, each.data() + p * 6);
    CHECK(many == each);
    u[299 * 5 + 2] = 1.0;
    CHECK_THROWS_AS(bc::decode(s, u.data(), points, many.data()), std::out_of_range);
}
