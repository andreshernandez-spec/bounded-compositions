// Plain-data views that kernels read. They hold pointers into memory owned by
// the host (or, later, copied to a device) and are trivially copyable.
#ifndef BCOMP_KERNEL_VIEWS_HPP
#define BCOMP_KERNEL_VIEWS_HPP

#include <cstdint>

#include "bounded_compositions/kernel/config.hpp"

namespace bounded_compositions {
namespace kernel {

// A space after shifting every lower bound to 0: part i takes values
// 0..cap[i] and the shifted values sum to `total`.
//
// `table` holds the prefix sums P[i][k] = sum_{j <= k} N[i][j] for rows
// i = 0..n and k = 0..total, where N[i][k] counts the ways to place k into
// parts i..n-1. Row i starts at limb offset row_offset[i] and stores each
// entry in row_limbs[i] limbs. Row widths never increase with i, because
// N[i][k] >= N[i+1][k].
struct space_view {
    const limb* table;
    const std::int64_t* row_offset;  // n + 1 entries
    const std::int32_t* row_limbs;   // n + 1 entries
    const std::int32_t* lo;          // n entries, the original lower bounds
    const std::int32_t* cap;         // n entries, upper minus lower bound
    const std::int32_t* cap_suffix;  // n + 1 entries: min(total, cap[i] + ... + cap[n-1])
    const limb* size;                // number of compositions, size_limbs limbs
    std::int32_t size_limbs;         // 0 when the space is empty
    std::int32_t n;
    std::int32_t total;  // shifted total, sum of the original lower bounds removed
};

BCOMP_KERNEL const limb* table_entry(const space_view& s, int row, std::int32_t k) {
    return s.table + s.row_offset[row] + static_cast<std::int64_t>(k) * s.row_limbs[row];
}

}  // namespace kernel
}  // namespace bounded_compositions

#endif
