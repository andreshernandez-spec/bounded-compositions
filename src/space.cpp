#include "bounded_compositions/space.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "bounded_compositions/kernel/limbs.hpp"
#include "bounded_compositions/kernel/walk.hpp"

namespace bounded_compositions {

namespace k = kernel;

static_assert(std::is_trivially_copyable_v<k::space_view>,
              "space_view must stay copyable to a device as is");

namespace detail {

struct space_data {
    std::int32_t total = 0;  // as given
    std::vector<std::int32_t> lower, upper;
    std::vector<std::int32_t> cap, cap_suffix;
    support_limits support;
    std::vector<group_limit> groups;
    // See kernel::space_view.
    std::vector<std::int32_t> seg, segs;
    std::vector<std::int32_t> row_layers;
    std::vector<std::int64_t> row_layer_start;
    std::vector<std::int32_t> row_open_start, row_open, act_start, act, grp_lo, grp_hi;
    std::vector<std::int64_t> row_offset;
    std::vector<std::int32_t> row_limbs;
    std::vector<k::limb> table;
    natural size;
    counting mode = counting::exact;
    std::vector<double> log_prefix, log_suffix;  // logarithmic mode only
    std::vector<std::uint8_t> positive;          // logarithmic mode only
    double log_size = -std::numeric_limits<double>::infinity();
    k::space_view view{};
    // A subspace points into its parent's arrays and keeps the parent alive.
    std::shared_ptr<const space_data> parent;
};

}  // namespace detail

namespace {

// Table entries (layers times total + 1, over all rows) a space may need.
constexpr std::int64_t max_entries = std::int64_t{1} << 28;

int as_int(std::size_t n) { return static_cast<int>(n); }

// Natural logarithm of a positive natural.
double log_of(const natural& x) {
    const int bits = x.bit_length();
    if (bits <= 1000) return std::log(x.to_double());
    // Keep the top 64 bits: x = top * 2^shift with top in [2^63, 2^64).
    const int shift = bits - 64;
    const k::limb* l = x.limbs();
    const int word = shift / 64, bit = shift % 64;
    k::limb top = l[word] >> bit;
    if (bit != 0) top |= l[word + 1] << (64 - bit);
    return std::log(static_cast<double>(top)) + shift * std::log(2.0);
}

void require_exact(const detail::space_data& d, const char* what) {
    if (d.mode == counting::logarithmic) {
        throw std::logic_error(std::string("bounded_compositions::space::") + what +
                               ": needs exact counting; this space counts in logarithms");
    }
}

// Prefix sums of the counting table, one row per part plus the empty suffix,
// row i holding row_layers[i] blocks of total + 1 entries. Row i is computed
// from row i+1 one limb wider than row i+1 (an entry is a sum over at most
// three segments of differences of row i+1, over at most total + 1 values of
// k, and 3 (total + 1) < 2^64), then trimmed to the width of its largest
// entry, the last of some layer.
void build_table(detail::space_data& d, std::int32_t shifted_total) {
    const std::size_t n = d.cap.size();
    const std::size_t width_k = static_cast<std::size_t>(shifted_total) + 1;
    std::vector<std::vector<k::limb>> rows(n + 1);
    std::vector<int> widths(n + 1);
    const k::space_view& v = d.view;  // segments, groups and count limits are set

    // N[n][0][c] = 1 for an allowed count c and N[n][k > 0][c] = 0, so P[n][c][k]
    // is that 1 or 0 for every k.
    widths[n] = 1;
    rows[n].assign(static_cast<std::size_t>(d.row_layers[n]) * width_k, 0);
    for (std::int32_t c = 0; c < d.row_layers[n]; ++c) {
        if (c < v.count_min || c > v.count_max) continue;
        const auto at = static_cast<std::ptrdiff_t>(static_cast<std::size_t>(c) * width_k);
        std::fill(rows[n].begin() + at, rows[n].begin() + at + static_cast<std::ptrdiff_t>(width_k),
                  k::limb{1});
    }

    std::vector<k::limb> count;
    for (std::size_t i = n; i-- > 0;) {
        const int part = static_cast<int>(i);
        const int wn = widths[i + 1];
        const int wc = wn + 1;
        const std::size_t un = static_cast<std::size_t>(wn);
        const std::size_t uc = static_cast<std::size_t>(wc);
        const std::size_t layers = static_cast<std::size_t>(d.row_layers[i]);
        const std::vector<k::limb>& next = rows[i + 1];
        auto at = [&](std::int32_t c, std::int32_t kk) {
            return &next[(static_cast<std::size_t>(c) * width_k + static_cast<std::size_t>(kk)) *
                         un];
        };
        std::vector<k::limb> row(layers * width_k * uc, 0);
        count.assign(uc, 0);
        for (std::int32_t c = 0; c < d.row_layers[i]; ++c) {
            for (std::int32_t kk = 0; kk <= shifted_total; ++kk) {
                // N[i][kk][c]: over each segment's allowed values [a, b],
                // P[i+1][cn][kk - a] - P[i+1][cn][kk - b - 1].
                k::limbs_zero(count.data(), wc);
                for (int g = 0; g < v.segs[part]; ++g) {
                    std::int32_t a = 0, b = 0, cn = 0;
                    if (!k::move(v, part, g, kk, c, &a, &b, &cn)) continue;
                    k::limbs_add(count.data(), wc, at(cn, kk - a), wn, count.data(), wc);
                    if (kk - b - 1 >= 0) {
                        k::limbs_sub(count.data(), wc, at(cn, kk - b - 1), wn, count.data(), wc);
                    }
                }
                k::limb* out =
                    &row[(static_cast<std::size_t>(c) * width_k + static_cast<std::size_t>(kk)) *
                         uc];
                if (kk == 0) {
                    k::limbs_copy(count.data(), wc, out, wc);
                } else {
                    k::limbs_add(out - wc, wc, count.data(), wc, out, wc);
                }
            }
        }
        int w = 1;
        for (std::size_t c = 0; c < layers; ++c) {
            const k::limb* largest = &row[(c * width_k + width_k - 1) * uc];
            w = std::max(w, k::limbs_length(largest, wc));
        }
        if (w > k::max_limbs) {
            throw std::length_error("bounded_compositions::space: counts need " +
                                    std::to_string(w) + " limbs, more than the kernel limit of " +
                                    std::to_string(k::max_limbs) + " (BCOMP_MAX_LIMBS)");
        }
        if (w < wc) {
            std::vector<k::limb> trimmed(layers * width_k * static_cast<std::size_t>(w));
            for (std::size_t e = 0; e < layers * width_k; ++e) {
                k::limbs_copy(&row[e * uc], wc, &trimmed[e * static_cast<std::size_t>(w)], w);
            }
            row.swap(trimmed);
        }
        rows[i].swap(row);
        widths[i] = w;
    }

    d.row_offset.resize(n + 1);
    d.row_limbs.assign(widths.begin(), widths.end());
    std::size_t total_limbs = 0;
    for (std::size_t i = 0; i <= n; ++i) {
        d.row_offset[i] = static_cast<std::int64_t>(total_limbs);
        total_limbs += rows[i].size();
    }
    d.table.reserve(total_limbs);
    for (std::size_t i = 0; i <= n; ++i) {
        d.table.insert(d.table.end(), rows[i].begin(), rows[i].end());
        std::vector<k::limb>().swap(rows[i]);
    }

    // size = N[0][total][0] = P[0][0][total] - P[0][0][total - 1].
    const int w0 = widths[0];
    const k::limb* last =
        &d.table[static_cast<std::size_t>(shifted_total) * static_cast<std::size_t>(w0)];
    if (shifted_total == 0) {
        d.size = natural::from_limbs(last, static_cast<std::size_t>(w0));
    } else {
        std::vector<k::limb> sz(static_cast<std::size_t>(w0));
        k::limbs_sub(last, w0, last - w0, w0, sz.data(), w0);
        d.size = natural::from_limbs(sz.data(), sz.size());
    }
}

// Logarithmic table: for each row and layer, log P and log S over
// k = 0..total + 1. N[i][k][c] sums, over part i's segments, the window
// [k - b, k - a] of row i + 1 in the layer the segment leads to, each summed
// from the side with less mass outside it (see log_window in walk.hpp).
// positive records exactly which counts are nonzero.
void build_log_table(detail::space_data& d, std::int32_t shifted_total) {
    const std::size_t n = d.cap.size();
    const std::size_t stride = static_cast<std::size_t>(shifted_total) + 2;
    const std::size_t width_k = static_cast<std::size_t>(shifted_total) + 1;
    const std::size_t all_layers = static_cast<std::size_t>(d.row_layer_start[n] + d.row_layers[n]);
    d.log_prefix.assign(all_layers * stride, k::log_zero);
    d.log_suffix.assign(all_layers * stride, k::log_zero);
    d.positive.assign(all_layers * width_k, std::uint8_t{0});
    double* P = d.log_prefix.data();
    double* S = d.log_suffix.data();
    const k::space_view& v = d.view;  // segments, groups and count limits are set
    auto start = [&](std::size_t row, std::int32_t c) {
        return static_cast<std::size_t>(d.row_layer_start[row]) + static_cast<std::size_t>(c);
    };
    // Row n: N[n][0][c] = 1 for an allowed count c, every other entry 0.
    for (std::int32_t c = 0; c < d.row_layers[n]; ++c) {
        if (c < v.count_min || c > v.count_max) continue;
        for (std::size_t kk = 0; kk < width_k; ++kk) P[start(n, c) * stride + kk] = 0.0;
        S[start(n, c) * stride] = 0.0;
        d.positive[start(n, c) * width_k] = 1;
    }
    std::vector<double> count(stride);
    std::vector<std::int32_t> ones;  // prefix counts of positive entries of row i + 1
    for (std::size_t i = n; i-- > 0;) {
        const int part = static_cast<int>(i);
        const std::size_t next_layers = static_cast<std::size_t>(d.row_layers[i + 1]);
        ones.assign(next_layers * (width_k + 1), 0);
        for (std::size_t c = 0; c < next_layers; ++c) {
            std::int32_t* o = &ones[c * (width_k + 1)];
            const std::uint8_t* pos =
                &d.positive[start(i + 1, static_cast<std::int32_t>(c)) * width_k];
            for (std::size_t kk = 0; kk < width_k; ++kk) o[kk + 1] = o[kk] + pos[kk];
        }
        for (std::int32_t c = 0; c < d.row_layers[i]; ++c) {
            for (std::int32_t kk = 0; kk <= shifted_total; ++kk) {
                double sum = k::log_zero;
                bool any = false;
                for (int g = 0; g < v.segs[part]; ++g) {
                    std::int32_t a = 0, b = 0, cn = 0;
                    if (!k::move(v, part, g, kk, c, &a, &b, &cn)) continue;
                    const double* Pn = P + start(i + 1, cn) * stride;
                    const double* Sn = S + start(i + 1, cn) * stride;
                    const std::int32_t lo = kk - b, hi = kk - a;
                    const double below = lo > 0 ? Pn[lo - 1] : k::log_zero;
                    const double above = Sn[hi + 1];
                    sum = k::log_add(sum, below <= above ? k::log_sub(Pn[hi], below)
                                                         : k::log_sub(Sn[lo], above));
                    const std::int32_t* o = &ones[static_cast<std::size_t>(cn) * (width_k + 1)];
                    any = any || o[hi + 1] > o[lo];
                }
                count[static_cast<std::size_t>(kk)] = sum;
                d.positive[start(i, c) * width_k + static_cast<std::size_t>(kk)] =
                    any ? std::uint8_t{1} : std::uint8_t{0};
            }
            double run = k::log_zero;
            for (std::int32_t kk = 0; kk <= shifted_total; ++kk) {
                run = k::log_add(run, count[static_cast<std::size_t>(kk)]);
                P[start(i, c) * stride + static_cast<std::size_t>(kk)] = run;
            }
            run = k::log_zero;
            for (std::int32_t kk = shifted_total; kk >= 0; --kk) {
                run = k::log_add(run, count[static_cast<std::size_t>(kk)]);
                S[start(i, c) * stride + static_cast<std::size_t>(kk)] = run;
            }
        }
    }
    // log N[0][total][0] = log S[0][0][total], since nothing lies above the total.
    d.log_size = S[start(0, 0) * stride + static_cast<std::size_t>(shifted_total)];
}

// Splits each part's values into segments (kernel::space_view): below zero,
// zero, above zero, leaving out a semicontinuous part's gap, with step 1 for
// nonzero values when the count is limited. Merges neighbours with equal steps.
void set_segments(detail::space_data& d, bool count_limited) {
    const std::size_t n = d.cap.size();
    d.seg.assign(9 * n, 0);
    d.segs.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t lo = d.lower[i], hi = d.upper[i];
        const std::int64_t gap = d.support.min_size.empty() ? 1 : d.support.min_size[i];
        const std::int32_t step = count_limited ? 1 : 0;
        std::int64_t ranges[3][3] = {
            {lo, std::min(hi, -gap), step},
            {std::max(lo, std::int64_t{0}), std::min(hi, std::int64_t{0}), 0},
            {std::max(lo, gap), hi, step}};
        int g = 0;
        for (const auto& q : ranges) {
            if (q[0] > q[1]) continue;
            std::int32_t* out = &d.seg[9 * i];
            if (g > 0 && out[3 * (g - 1) + 2] == q[2] &&
                out[3 * (g - 1) + 1] + d.lower[i] + 1 == q[0]) {
                out[3 * (g - 1) + 1] = static_cast<std::int32_t>(q[1] - lo);
                continue;
            }
            out[3 * g] = static_cast<std::int32_t>(q[0] - lo);
            out[3 * g + 1] = static_cast<std::int32_t>(q[1] - lo);
            out[3 * g + 2] = static_cast<std::int32_t>(q[2]);
            ++g;
        }
        d.segs[i] = g;
    }
}

// The group arrays of kernel::space_view and the layers of each row. Returns
// false if some group's limits leave no sum.
bool set_groups(detail::space_data& d, std::int32_t shifted_total, std::int32_t support_layers) {
    const std::size_t n = d.cap.size();
    const std::size_t count = d.groups.size();
    std::vector<std::size_t> first(count), last(count);
    std::vector<std::vector<char>> member(count, std::vector<char>(n, char{0}));
    d.grp_lo.assign(count, 0);
    d.grp_hi.assign(count, 0);
    bool feasible = true;
    for (std::size_t g = 0; g < count; ++g) {
        const group_limit& gl = d.groups[g];
        if (gl.parts.empty()) {
            throw std::invalid_argument("bounded_compositions::space: group " + std::to_string(g) +
                                        " has no parts");
        }
        if (gl.min_sum > gl.max_sum) {
            throw std::invalid_argument("bounded_compositions::space: group " + std::to_string(g) +
                                        " has min_sum > max_sum");
        }
        std::int64_t lower_sum = 0, cap_sum = 0;
        first[g] = n;
        last[g] = 0;
        for (std::int32_t p : gl.parts) {
            if (p < 0 || static_cast<std::size_t>(p) >= n) {
                throw std::invalid_argument("bounded_compositions::space: group " +
                                            std::to_string(g) + " names part " + std::to_string(p) +
                                            ", out of range");
            }
            const std::size_t u = static_cast<std::size_t>(p);
            if (member[g][u] != 0) {
                throw std::invalid_argument("bounded_compositions::space: group " +
                                            std::to_string(g) + " names part " + std::to_string(p) +
                                            " twice");
            }
            member[g][u] = 1;
            first[g] = std::min(first[g], u);
            last[g] = std::max(last[g], u);
            lower_sum += d.lower[u];
            cap_sum += d.cap[u];
        }
        // Shifted sums: the members' values above their lower bounds.
        const std::int64_t lo = std::max<std::int64_t>(0, std::int64_t{gl.min_sum} - lower_sum);
        const std::int64_t hi = std::min<std::int64_t>(
            {std::int64_t{gl.max_sum} - lower_sum, cap_sum, std::int64_t{shifted_total}});
        if (hi < lo) feasible = false;
        d.grp_lo[g] = static_cast<std::int32_t>(std::min<std::int64_t>(lo, shifted_total + 1));
        d.grp_hi[g] = static_cast<std::int32_t>(std::max<std::int64_t>(hi, -1));
    }

    // Row j: the groups open before part j, with members on both sides.
    d.row_open_start.assign(n + 2, 0);
    d.row_open.clear();
    d.row_layers.assign(n + 1, 0);
    d.row_layer_start.assign(n + 1, 0);
    std::vector<std::vector<std::int32_t>> position(n + 1, std::vector<std::int32_t>(count, -1));
    std::int64_t entries = 0;
    for (std::size_t j = 0; j <= n; ++j) {
        d.row_open_start[j] = static_cast<std::int32_t>(d.row_open.size() / 4);
        std::int64_t layers = support_layers;
        for (std::size_t g = 0; g < count; ++g) {
            if (!(first[g] < j && j <= last[g])) continue;
            const bool after_member = member[g][j - 1] != 0;
            const std::int64_t range =
                after_member ? std::int64_t{shifted_total} + 1 : std::int64_t{d.grp_hi[g]} + 1;
            position[j][g] = static_cast<std::int32_t>(
                d.row_open.size() / 4 - static_cast<std::size_t>(d.row_open_start[j]));
            d.row_open.insert(d.row_open.end(),
                              {static_cast<std::int32_t>(g), after_member ? 1 : 0,
                               static_cast<std::int32_t>(std::max<std::int64_t>(range, 1)),
                               static_cast<std::int32_t>(layers)});
            layers *= std::max<std::int64_t>(range, 1);
            if (layers > std::numeric_limits<std::int32_t>::max()) {
                throw std::length_error(
                    "bounded_compositions::space: the groups spanning part " + std::to_string(j) +
                    " need more than 2^31 states; keep each group's parts next to each other");
            }
        }
        d.row_layers[j] = static_cast<std::int32_t>(layers);
        d.row_layer_start[j] = j == 0 ? 0 : d.row_layer_start[j - 1] + d.row_layers[j - 1];
        entries += layers * (std::int64_t{shifted_total} + 1);
        if (entries > max_entries) {
            throw std::length_error(
                "bounded_compositions::space: the limits need more than 2^28 table entries; keep "
                "each group's parts next to each other, or use fewer groups at once");
        }
    }
    d.row_open_start[n + 1] = static_cast<std::int32_t>(d.row_open.size() / 4);

    // Part i: the groups whose span covers it.
    d.act_start.assign(n + 1, 0);
    d.act.clear();
    for (std::size_t i = 0; i < n; ++i) {
        d.act_start[i] = static_cast<std::int32_t>(d.act.size() / 4);
        for (std::size_t g = 0; g < count; ++g) {
            if (!(first[g] <= i && i <= last[g])) continue;
            const std::int32_t flags =
                (member[g][i] != 0 ? 1 : 0) | (first[g] == i ? 2 : 0) | (last[g] == i ? 4 : 0);
            d.act.insert(d.act.end(),
                         {static_cast<std::int32_t>(g), flags, position[i][g], position[i + 1][g]});
        }
    }
    d.act_start[n] = static_cast<std::int32_t>(d.act.size() / 4);
    return feasible;
}

void check_support(const support_limits& support, std::size_t parts) {
    if (support.min_count < 0 || support.max_count < 0 || support.min_count > support.max_count) {
        throw std::invalid_argument(
            "bounded_compositions::space: support counts must satisfy 0 <= min_count <= "
            "max_count");
    }
    if (!support.min_size.empty()) {
        if (support.min_size.size() != parts) {
            throw std::invalid_argument(
                "bounded_compositions::space: min_size needs one value per part");
        }
        for (std::int32_t m : support.min_size) {
            if (m < 1) {
                throw std::invalid_argument(
                    "bounded_compositions::space: min_size values must be at least 1");
            }
        }
    }
}

std::shared_ptr<const detail::space_data> make_data(std::int32_t total, const std::int32_t* lower,
                                                    const std::int32_t* upper, std::size_t parts,
                                                    const support_limits& support,
                                                    const std::vector<group_limit>& groups,
                                                    counting mode) {
    if (parts == 0) throw std::invalid_argument("bounded_compositions::space: no parts");
    if (parts > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() - 1)) {
        throw std::invalid_argument("bounded_compositions::space: too many parts");
    }
    auto d = std::make_shared<detail::space_data>();
    d->total = total;
    d->mode = mode;
    d->lower.assign(lower, lower + parts);
    d->upper.assign(upper, upper + parts);
    d->cap.resize(parts);

    std::int64_t lower_sum = 0;
    std::int64_t cap_sum = 0;
    for (std::size_t i = 0; i < parts; ++i) {
        if (upper[i] < lower[i]) {
            throw std::invalid_argument(
                "bounded_compositions::space: upper bound below lower bound at part " +
                std::to_string(i));
        }
        const std::int64_t cap = static_cast<std::int64_t>(upper[i]) - lower[i];
        if (cap > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument(
                "bounded_compositions::space: bound range too wide at part " + std::to_string(i));
        }
        d->cap[i] = static_cast<std::int32_t>(cap);
        lower_sum += lower[i];
        cap_sum += cap;
    }

    const std::int64_t shifted = static_cast<std::int64_t>(total) - lower_sum;
    bool feasible = shifted >= 0 && shifted <= cap_sum;
    if (shifted > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(
            "bounded_compositions::space: total minus lower bounds exceeds int32");
    }
    const std::int32_t shifted_total = feasible ? static_cast<std::int32_t>(shifted) : 0;

    check_support(support, parts);
    d->support = support;
    d->groups = groups;
    const std::int32_t n32 = static_cast<std::int32_t>(parts);
    const bool count_limited = support.min_count > 0 || support.max_count < n32;
    const std::int32_t support_layers = count_limited ? std::min(support.max_count, n32) + 1 : 1;
    set_segments(*d, count_limited);
    if (!set_groups(*d, shifted_total, support_layers)) feasible = false;
    bool plain = !count_limited && groups.empty();
    for (std::size_t i = 0; i < parts && plain; ++i) {
        plain = d->segs[i] == 1 && d->seg[9 * i] == 0 && d->seg[9 * i + 1] == d->cap[i];
    }

    // cap_suffix[i] = min(shifted_total, cap[i] + ... + cap[n-1]).
    d->cap_suffix.assign(parts + 1, 0);
    std::int64_t run = 0;
    for (std::size_t i = parts; i-- > 0;) {
        run += d->cap[i];
        d->cap_suffix[i] = static_cast<std::int32_t>(std::min<std::int64_t>(run, shifted_total));
    }

    k::space_view& v = d->view;
    v.seg = d->seg.data();
    v.segs = d->segs.data();
    v.support_layers = support_layers;
    v.layer_base = 0;
    v.count_min = count_limited ? support.min_count : 0;
    v.count_max =
        count_limited ? std::min(support.max_count, n32) : std::numeric_limits<std::int32_t>::max();
    v.table_k = shifted_total + 1;
    v.plain = plain ? 1 : 0;
    v.row_layers = d->row_layers.data();
    v.row_layer_start = d->row_layer_start.data();
    v.row_open_start = d->row_open_start.data();
    v.row_open = d->row_open.data();
    v.act_start = d->act_start.data();
    v.act = d->act.data();
    v.grp_lo = d->grp_lo.data();
    v.grp_hi = d->grp_hi.data();

    const bool logarithmic = mode == counting::logarithmic;
    if (feasible && logarithmic) {
        build_log_table(*d, shifted_total);
        if (k::log_is_zero(d->log_size)) {  // the limits leave nothing
            feasible = false;
            d->log_size = -std::numeric_limits<double>::infinity();
            std::vector<double>().swap(d->log_prefix);
            std::vector<double>().swap(d->log_suffix);
            std::vector<std::uint8_t>().swap(d->positive);
        }
    }
    if (feasible && !logarithmic) {
        build_table(*d, shifted_total);
        if (d->size.is_zero()) {  // the limits leave nothing
            feasible = false;
            std::vector<k::limb>().swap(d->table);
        }
    }

    v.feasible = feasible ? 1 : 0;
    v.log_mode = logarithmic ? 1 : 0;
    v.log_prefix = logarithmic && feasible ? d->log_prefix.data() : nullptr;
    v.log_suffix = logarithmic && feasible ? d->log_suffix.data() : nullptr;
    v.positive = logarithmic && feasible ? d->positive.data() : nullptr;
    v.log_stride = static_cast<std::int64_t>(shifted_total) + 2;
    v.n = static_cast<std::int32_t>(parts);
    v.total = shifted_total;
    v.lo = d->lower.data();
    v.cap = d->cap.data();
    v.cap_suffix = d->cap_suffix.data();
    if (feasible && !logarithmic) {
        v.table = d->table.data();
        v.row_offset = d->row_offset.data();
        v.row_limbs = d->row_limbs.data();
        v.size = d->size.limbs();
        v.size_limbs = static_cast<std::int32_t>(d->size.limb_count());
    } else {
        v.table = nullptr;
        v.row_offset = nullptr;
        v.row_limbs = nullptr;
        v.size = nullptr;
        v.size_limbs = 0;
    }
    if (feasible && !logarithmic) d->log_size = log_of(d->size);
    return d;
}

void check_length(std::size_t got, std::size_t want) {
    if (got != want) {
        throw std::invalid_argument("bounded_compositions::space: expected " +
                                    std::to_string(want) + " values, got " + std::to_string(got));
    }
}

void check_same_length(std::size_t lower, std::size_t upper) {
    if (lower != upper) {
        throw std::invalid_argument(
            "bounded_compositions::space: lower and upper differ in length");
    }
}

// A limit shifted by what the prefix already holds, saturating at the
// int32 range so that an absent limit stays absent.
std::int32_t shift_limit(std::int32_t limit, std::int64_t taken) {
    if (limit == std::numeric_limits<std::int32_t>::min() ||
        limit == std::numeric_limits<std::int32_t>::max()) {
        return limit;
    }
    const std::int64_t out = std::int64_t{limit} - taken;
    return static_cast<std::int32_t>(
        std::clamp<std::int64_t>(out, std::numeric_limits<std::int32_t>::min() + std::int64_t{1},
                                 std::numeric_limits<std::int32_t>::max() - std::int64_t{1}));
}

}  // namespace

space::space(std::int32_t total, const std::vector<std::int32_t>& lower,
             const std::vector<std::int32_t>& upper, counting mode) {
    check_same_length(lower.size(), upper.size());
    data_ = make_data(total, lower.data(), upper.data(), lower.size(), support_limits{}, {}, mode);
}

space::space(std::int32_t total, const std::int32_t* lower, const std::int32_t* upper,
             std::size_t parts, counting mode)
    : data_(make_data(total, lower, upper, parts, support_limits{}, {}, mode)) {}

space::space(std::int32_t total, const std::vector<std::int32_t>& lower,
             const std::vector<std::int32_t>& upper, const support_limits& support, counting mode) {
    check_same_length(lower.size(), upper.size());
    data_ = make_data(total, lower.data(), upper.data(), lower.size(), support, {}, mode);
}

space::space(std::int32_t total, const std::int32_t* lower, const std::int32_t* upper,
             std::size_t parts, const support_limits& support, counting mode)
    : data_(make_data(total, lower, upper, parts, support, {}, mode)) {}

space::space(std::int32_t total, const std::vector<std::int32_t>& lower,
             const std::vector<std::int32_t>& upper, const support_limits& support,
             const std::vector<group_limit>& groups, counting mode) {
    check_same_length(lower.size(), upper.size());
    data_ = make_data(total, lower.data(), upper.data(), lower.size(), support, groups, mode);
}

space::space(std::int32_t total, const std::int32_t* lower, const std::int32_t* upper,
             std::size_t parts, const support_limits& support,
             const std::vector<group_limit>& groups, counting mode)
    : data_(make_data(total, lower, upper, parts, support, groups, mode)) {}

std::size_t space::parts() const noexcept { return data_->lower.size(); }
std::int32_t space::total() const noexcept { return data_->total; }
const std::vector<std::int32_t>& space::lower() const noexcept { return data_->lower; }
const std::vector<std::int32_t>& space::upper() const noexcept { return data_->upper; }
const support_limits& space::support() const noexcept { return data_->support; }
const std::vector<group_limit>& space::groups() const noexcept { return data_->groups; }
counting space::counting_mode() const noexcept { return data_->mode; }

const natural& space::size() const {
    require_exact(*data_, "size");
    return data_->size;
}

double space::log_size() const noexcept { return data_->log_size; }
bool space::empty() const noexcept { return data_->view.feasible == 0; }
k::space_view space::view() const noexcept { return data_->view; }

std::size_t space::table_bytes() const noexcept {
    const detail::space_data* d = data_.get();
    while (d->parent) d = d->parent.get();
    return d->table.size() * sizeof(k::limb) +
           (d->log_prefix.size() + d->log_suffix.size()) * sizeof(double) + d->positive.size();
}

space::space(std::shared_ptr<const detail::space_data> data) : data_(std::move(data)) {}

space space::subspace(const std::vector<std::int32_t>& prefix) const {
    const std::size_t n = parts();
    const std::size_t d = prefix.size();
    if (d >= n) {
        throw std::invalid_argument(
            "bounded_compositions::space::subspace: prefix must leave at least one part");
    }
    if (empty()) throw std::invalid_argument("bounded_compositions::space::subspace: empty space");
    const k::space_view& v = data_->view;
    std::int32_t left = v.total;
    std::int32_t layer = v.layer_base;
    std::int32_t nonzero = 0;
    std::int64_t prefix_sum = 0;
    for (std::size_t i = 0; i < d; ++i) {
        const std::int64_t y = static_cast<std::int64_t>(prefix[i]) - v.lo[i];
        const std::int32_t next =
            y < 0 || y > v.cap[i] || y > left
                ? -1
                : k::layer_after(v, static_cast<int>(i), left, layer, static_cast<std::int32_t>(y));
        if (next < 0) {
            throw std::invalid_argument("bounded_compositions::space::subspace: prefix value " +
                                        std::to_string(prefix[i]) + " at part " +
                                        std::to_string(i) + " is out of range");
        }
        left -= static_cast<std::int32_t>(y);
        layer = next;
        if (prefix[i] != 0) ++nonzero;
        prefix_sum += prefix[i];
    }
    const int row = static_cast<int>(d);
    // The limits on the parts left.
    support_limits rest = data_->support;
    rest.min_count = std::max(0, rest.min_count - nonzero);
    if (rest.max_count != std::numeric_limits<std::int32_t>::max()) rest.max_count -= nonzero;
    if (!rest.min_size.empty()) {
        rest.min_size.erase(rest.min_size.begin(),
                            rest.min_size.begin() + static_cast<std::ptrdiff_t>(d));
    }
    std::vector<group_limit> rest_groups;
    for (const group_limit& gl : data_->groups) {
        group_limit out;
        std::int64_t taken = 0;
        for (std::int32_t p : gl.parts) {
            if (static_cast<std::size_t>(p) < d) {
                taken += prefix[static_cast<std::size_t>(p)];
            } else {
                out.parts.push_back(p - static_cast<std::int32_t>(d));
            }
        }
        if (out.parts.empty()) continue;
        out.min_sum = shift_limit(gl.min_sum, taken);
        out.max_sum = shift_limit(gl.max_sum, taken);
        rest_groups.push_back(std::move(out));
    }

    auto sub = std::make_shared<detail::space_data>();
    sub->parent = data_;
    sub->mode = data_->mode;
    sub->total = static_cast<std::int32_t>(data_->total - prefix_sum);
    sub->lower.assign(data_->lower.begin() + static_cast<std::ptrdiff_t>(d), data_->lower.end());
    sub->upper.assign(data_->upper.begin() + static_cast<std::ptrdiff_t>(d), data_->upper.end());
    sub->support = rest;
    sub->groups = std::move(rest_groups);
    k::space_view& sv = sub->view;
    sv = v;  // same tables; rows and parts shifted by d
    sv.seg = v.seg + 9 * d;
    sv.segs = v.segs + d;
    sv.lo = v.lo + d;
    sv.cap = v.cap + d;
    sv.cap_suffix = v.cap_suffix + d;
    sv.row_layers = v.row_layers + d;
    sv.row_layer_start = v.row_layer_start + d;
    sv.row_open_start = v.row_open_start + d;
    sv.act_start = v.act_start + d;
    sv.layer_base = layer;
    sv.n = static_cast<std::int32_t>(n - d);
    sv.total = left;
    if (data_->mode == counting::logarithmic) {
        const double log_count = k::log_window(v, row, layer, left, left, nullptr);
        if (!k::has_completion(v, row, layer, left)) {
            throw std::invalid_argument(
                "bounded_compositions::space::subspace: prefix has no completion");
        }
        sub->log_size = log_count;
        return space(std::move(sub));
    }
    // size = N[d][left][layer]
    std::vector<k::limb> count(static_cast<std::size_t>(v.row_limbs[row]));
    k::completions(v, row, layer, left, count.data());
    sub->size = natural::from_limbs(count.data(), count.size());
    if (sub->size.is_zero()) {
        throw std::invalid_argument(
            "bounded_compositions::space::subspace: prefix has no completion");
    }
    sv.row_offset = v.row_offset + d;
    sv.row_limbs = v.row_limbs + d;
    sv.size = sub->size.limbs();
    sv.size_limbs = static_cast<std::int32_t>(sub->size.limb_count());
    sub->log_size = log_of(sub->size);
    return space(std::move(sub));
}

void space::unrank(const natural& r, std::int32_t* out) const {
    require_exact(*data_, "unrank");
    const k::status st = k::unrank(data_->view, r.limbs(), as_int(r.limb_count()), out);
    if (st != k::status::ok) {
        throw std::out_of_range("bounded_compositions::space::unrank: rank " + r.to_string() +
                                " is not below the size " + data_->size.to_string());
    }
}

std::vector<std::int32_t> space::unrank(const natural& r) const {
    std::vector<std::int32_t> out(parts());
    unrank(r, out.data());
    return out;
}

natural space::rank(const std::int32_t* x) const {
    require_exact(*data_, "rank");
    const k::space_view& v = data_->view;
    // An empty space has no table; the kernel rejects x before writing.
    const int width = v.size_limbs == 0 ? 1 : v.row_limbs[0];
    std::vector<k::limb> out(static_cast<std::size_t>(width));
    const k::status st = k::rank(v, x, out.data(), as_int(out.size()));
    if (st != k::status::ok) {
        throw std::invalid_argument(
            "bounded_compositions::space::rank: not a composition of this space");
    }
    return natural::from_limbs(out.data(), out.size());
}

natural space::rank(const std::vector<std::int32_t>& x) const {
    check_length(x.size(), parts());
    return rank(x.data());
}

bool space::contains(const std::int32_t* x) const noexcept { return k::contains(data_->view, x); }

bool space::contains(const std::vector<std::int32_t>& x) const noexcept {
    return x.size() == parts() && contains(x.data());
}

bool space::first(std::int32_t* out) const noexcept {
    return k::first(data_->view, out) == k::status::ok;
}

bool space::next(std::int32_t* x) const noexcept {
    if (data_->view.plain != 0) return k::next(data_->view, x, nullptr) == k::status::ok;
    std::vector<std::int32_t> scratch(parts());
    return k::next(data_->view, x, scratch.data()) == k::status::ok;
}

}  // namespace bounded_compositions
