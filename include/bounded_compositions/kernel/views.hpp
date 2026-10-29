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
// Limits (space.hpp) narrow this. A part's allowed values are segments,
// seg[3 (seg_start[i] + g) .. + 2] = (first, last, step) for g < segs[i], in
// increasing order and split at zero: the values below zero, zero itself, and
// the values above, with gaps (semicontinuous or given) left out. The step is
// what a value moves the support count by: 0 for zero and when the count is
// not limited, 1 otherwise.
//
// The state before part i is (k, layer): k is what is left to place, and the
// layer is a mixed-radix number holding the nonzero parts so far (the support
// count, layer % support_layers, which a value moves by its segment's step)
// and one coordinate per group open before part i, that is with members on
// both sides of it. A group's coordinate is its shifted running sum r, or
// r + k when part i - 1 is a member: then taking value v at part i - 1 left
// r + k unchanged, so every value of a segment leads to the same layer and
// the counts stay windows of prefix sums. row_open lists the open groups of
// each row as (group, coordinate is r + k, range, multiplier), and act lists,
// per part, the groups whose span covers it as (group, flags, position in
// this row's list, position in the next row's), flags 1 for a member, 2 when
// the group starts at the part and 4 when it ends there. grp_lo and grp_hi
// bound each group's shifted sum, checked when it ends. Row i has
// row_layers[i] layers. Unconstrained spaces have one segment [0, cap[i]] per
// part and one layer, and are marked plain.
//
// `table` holds the prefix sums P[i][c][k] = sum_{j <= k} N[i][j][c] for rows
// i = 0..n, layers c < row_layers[i] and k = 0..total, where N[i][k][c] counts
// the ways to complete parts i..n-1 from state (k, c). Row i starts at limb
// offset row_offset[i], holds row_layers[i] blocks of table_k entries, and
// stores each entry in row_limbs[i] limbs. N[n][0][c] is 1 when
// count_min <= c <= count_max and every other N[n] entry is 0. A subspace
// shares its parent's table and segments: layer_base is the layer after its
// prefix, table_k the parent's total + 1, and seg_start holds the parent's
// offsets.
//
// In the logarithmic counting mode there is no limb table (table, row_offset,
// row_limbs and size are null). Instead log_prefix and log_suffix hold, for
// each row i and layer c, log P[i][c][k] and log S[i][c][k] =
// log sum_{j >= k} N[i][j][c] at (row_layer_start[i] + c) * log_stride + k for
// k = 0..total + 1 (the last suffix entry is log 0). A window of a row is
// summed from whichever side has less mass outside it, which keeps the
// subtraction well conditioned. Whether a count is zero cannot be read from
// rounded logarithms, so positive[(row_layer_start[i] + c) * table_k + k]
// holds 1 exactly when N[i][k][c] > 0. A subspace's row arrays start at its
// first row; the offsets in them are the parent's.
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
    const std::int32_t* seg;        // (first, last, step) per segment
    const std::int32_t* seg_start;  // n entries, part i's first segment in seg
    const std::int32_t* segs;       // n entries, part i's number of segments
    std::int32_t support_layers;
    std::int32_t layer_base;
    std::int32_t count_min;
    std::int32_t count_max;
    std::int32_t table_k;
    std::int32_t plain;
    const std::uint8_t* positive;         // log mode only
    const std::int32_t* row_layers;       // n + 1 entries
    const std::int64_t* row_layer_start;  // n + 1 entries
    const std::int32_t* row_open_start;   // n + 2 entries, into row_open
    const std::int32_t* row_open;         // 4 per entry
    const std::int32_t* act_start;        // n + 1 entries, into act
    const std::int32_t* act;              // 4 per entry
    const std::int32_t* grp_lo;
    const std::int32_t* grp_hi;
};

// P[row][layer][k], for 0 <= k < table_k.
BCOMP_KERNEL const limb* table_entry(const space_view& s, int row, std::int32_t layer,
                                     std::int32_t k) {
    return s.table + s.row_offset[row] +
           (static_cast<std::int64_t>(layer) * s.table_k + k) * s.row_limbs[row];
}

// log P[row][layer][k], with P = 0 for k < 0 (log mode).
BCOMP_KERNEL double log_prefix_at(const space_view& s, int row, std::int32_t layer,
                                  std::int32_t k) {
    return k < 0 ? -1e300 : s.log_prefix[(s.row_layer_start[row] + layer) * s.log_stride + k];
}

// log S[row][layer][k] for 0 <= k <= total + 1 (log mode).
BCOMP_KERNEL double log_suffix_at(const space_view& s, int row, std::int32_t layer,
                                  std::int32_t k) {
    return s.log_suffix[(s.row_layer_start[row] + layer) * s.log_stride + k];
}

}  // namespace kernel
}  // namespace bounded_compositions

#endif
