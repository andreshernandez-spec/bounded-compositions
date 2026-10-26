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
// Support limits (space.hpp) narrow this in two ways. A part's allowed values
// are up to three segments, seg[9 i + 3 g .. + 2] = (first, last, step) for
// g < segs[i], in increasing order: the values below zero, zero itself, and
// the values above, with a semicontinuous part's gap left out. A limit on the
// number of nonzero parts adds a layer c to the state, the nonzero parts so
// far, which a value moves by its segment's step (1 for nonzero values when
// the count is limited, otherwise 0). Unconstrained spaces have one segment
// [0, cap[i]] per part and one layer, and are marked plain.
//
// `table` holds the prefix sums P[i][c][k] = sum_{j <= k} N[i][j][c] for rows
// i = 0..n, layers c < layers and k = 0..total, where N[i][k][c] counts the
// ways to place k into parts i..n-1 with c nonzero parts before part i. Row i
// starts at limb offset row_offset[i], holds `layers` blocks of table_k
// entries, and stores each entry in row_limbs[i] limbs. N[n][0][c] is 1 when
// count_min <= c <= count_max and every other N[n] entry is 0. A subspace
// shares its parent's table: layer_base is the nonzero parts of its prefix and
// table_k the parent's total + 1.
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
    const std::int32_t* seg;   // 9 per part: up to three (first, last, step)
    const std::int32_t* segs;  // n entries, 1 to 3
    std::int32_t layers;
    std::int32_t layer_base;
    std::int32_t count_min;
    std::int32_t count_max;
    std::int32_t table_k;
    std::int32_t plain;
};

// P[row][layer][k], for 0 <= k < table_k.
BCOMP_KERNEL const limb* table_entry(const space_view& s, int row, std::int32_t layer,
                                     std::int32_t k) {
    return s.table + s.row_offset[row] +
           (static_cast<std::int64_t>(layer) * s.table_k + k) * s.row_limbs[row];
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
