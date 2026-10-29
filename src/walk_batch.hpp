// Private: kernel walks of many points, interleaved so their table reads
// overlap. A walk's searches read the table at addresses that depend on the
// previous comparison, so one point at a time leaves the core waiting on one
// read; on tables larger than the cache that wait is most of the time.
#ifndef BCOMP_SRC_WALK_BATCH_HPP
#define BCOMP_SRC_WALK_BATCH_HPP

#include <cstddef>
#include <cstdint>

#include "bounded_compositions/kernel/config.hpp"
#include "bounded_compositions/kernel/views.hpp"

namespace bounded_compositions {
namespace detail {

// kernel::unrank for `count` points: the rank of point p in
// ranks[p * rank_limbs, (p + 1) * rank_limbs), its composition written to
// out[p * out_stride, p * out_stride + s.n). The results are those of
// kernel::unrank point by point. Returns the first status other than ok, after
// which the outputs are unspecified.
kernel::status unrank_points(const kernel::space_view& s, const kernel::limb* ranks, int rank_limbs,
                             std::size_t count, std::int32_t* out, std::size_t out_stride);

// kernel::decode for `count` points, point p's 64-bit fractions in
// u[p * u_stride, p * u_stride + s.n - 1), in either counting mode, with the
// results of kernel::decode point by point.
kernel::status decode_points(const kernel::space_view& s, const kernel::limb* u,
                             std::size_t u_stride, std::size_t count, std::int32_t* out,
                             std::size_t out_stride);

}  // namespace detail
}  // namespace bounded_compositions

#endif
