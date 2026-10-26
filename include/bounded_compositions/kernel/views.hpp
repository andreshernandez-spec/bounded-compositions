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
//
// In the logarithmic counting mode there is no limb table (table, row_offset,
// row_limbs and size are null). Instead log_prefix and log_suffix hold, for
// each row i, log P[i][k] and log S[i][k] = log sum_{j >= k} N[i][j] at
// i * log_stride + k for k = 0..total + 1 (the last suffix entry is log 0). A
// window of a row is summed from whichever side has less mass outside it,
// which keeps the subtraction well conditioned.
struct space_view {
    const limb* table;
    const std::int64_t* row_offset;  // n + 1 entries
    const std::int32_t* row_limbs;   // n + 1 entries
    const std::int32_t* lo;          // n entries, the original lower bounds
    const std::int32_t* cap;         // n entries, upper minus lower bound
    const std::int32_t* cap_suffix;  // n + 1 entries: min(total, cap[i] + ... + cap[n-1])
    const limb* size;                // number of compositions, size_limbs limbs
    std::int32_t size_limbs;         // 0 when the space is empty or counted in logs
    std::int32_t n;
    std::int32_t total;     // shifted total, sum of the original lower bounds removed
    std::int32_t feasible;  // 1 if the space has at least one composition
    std::int32_t log_mode;  // 1 for the logarithmic counting mode
    const double* log_prefix;
    const double* log_suffix;
    std::int64_t log_stride;
};

BCOMP_KERNEL const limb* table_entry(const space_view& s, int row, std::int32_t k) {
    return s.table + s.row_offset[row] + static_cast<std::int64_t>(k) * s.row_limbs[row];
}

// log P[row][k], with P[row][k] = 0 for k < 0 (log mode).
BCOMP_KERNEL double log_prefix_at(const space_view& s, int row, std::int32_t k) {
    return k < 0 ? -1e300 : s.log_prefix[row * s.log_stride + k];
}

// log S[row][k] for 0 <= k <= total + 1 (log mode).
BCOMP_KERNEL double log_suffix_at(const space_view& s, int row, std::int32_t k) {
    return s.log_suffix[row * s.log_stride + k];
}

}  // namespace kernel
}  // namespace bounded_compositions

#endif
