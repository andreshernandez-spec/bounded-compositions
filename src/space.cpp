#include "bounded_compositions/space.hpp"

#include <algorithm>
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
    std::vector<std::int64_t> row_offset;
    std::vector<std::int32_t> row_limbs;
    std::vector<k::limb> table;
    natural size;
    k::space_view view{};
    // A subspace points into its parent's arrays and keeps the parent alive.
    std::shared_ptr<const space_data> parent;
};

}  // namespace detail

namespace {

int as_int(std::size_t n) { return static_cast<int>(n); }

// Prefix sums of the counting table, one row per part plus the empty suffix.
// Row i is computed from row i+1 one limb wider than row i+1 (an entry is a
// sum of at most total + 1 entries of row i+1, and total + 1 < 2^32), then
// trimmed to the width of its largest entry, which is its last.
void build_table(detail::space_data& d, std::int32_t shifted_total) {
    const std::size_t n = d.cap.size();
    const std::size_t width_k = static_cast<std::size_t>(shifted_total) + 1;
    std::vector<std::vector<k::limb>> rows(n + 1);
    std::vector<int> widths(n + 1);

    widths[n] = 1;  // N[n][0] = 1 and N[n][k > 0] = 0, so P[n][k] = 1 for every k
    rows[n].assign(width_k, 1);

    std::vector<k::limb> count;
    for (std::size_t i = n; i-- > 0;) {
        const int wn = widths[i + 1];
        const int wc = wn + 1;
        const std::vector<k::limb>& next = rows[i + 1];
        std::vector<k::limb> row(width_k * static_cast<std::size_t>(wc), 0);
        count.assign(static_cast<std::size_t>(wc), 0);
        const std::int32_t cap = d.cap[i];
        for (std::int32_t kk = 0; kk <= shifted_total; ++kk) {
            // N[i][kk] = P[i+1][kk] - P[i+1][kk - min(cap, kk) - 1].
            const k::limb* hi = &next[static_cast<std::size_t>(kk) * static_cast<std::size_t>(wn)];
            const std::int32_t j = kk - std::min(cap, kk) - 1;
            if (j < 0) {
                k::limbs_copy(hi, wn, count.data(), wc);
            } else {
                const k::limb* lo =
                    &next[static_cast<std::size_t>(j) * static_cast<std::size_t>(wn)];
                k::limbs_sub(hi, wn, lo, wn, count.data(), wc);
            }
            k::limb* out = &row[static_cast<std::size_t>(kk) * static_cast<std::size_t>(wc)];
            if (kk == 0) {
                k::limbs_copy(count.data(), wc, out, wc);
            } else {
                k::limbs_add(out - wc, wc, count.data(), wc, out, wc);
            }
        }
        const k::limb* largest =
            &row[static_cast<std::size_t>(shifted_total) * static_cast<std::size_t>(wc)];
        const int w = std::max(1, k::limbs_length(largest, wc));
        if (w > k::max_limbs) {
            throw std::length_error("bounded_compositions::space: counts need " +
                                    std::to_string(w) + " limbs, more than the kernel limit of " +
                                    std::to_string(k::max_limbs) + " (BCOMP_MAX_LIMBS)");
        }
        if (w < wc) {
            std::vector<k::limb> trimmed(width_k * static_cast<std::size_t>(w));
            for (std::size_t kk = 0; kk < width_k; ++kk) {
                k::limbs_copy(&row[kk * static_cast<std::size_t>(wc)], wc,
                              &trimmed[kk * static_cast<std::size_t>(w)], w);
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

    // size = N[0][total] = P[0][total] - P[0][total - 1].
    const int w0 = widths[0];
    const k::limb* last =
        &d.table[static_cast<std::size_t>(shifted_total) * static_cast<std::size_t>(w0)];
    if (shifted_total == 0) {
        d.size = natural::from_limbs(last, static_cast<std::size_t>(w0));
    } else {
        std::vector<k::limb> s(static_cast<std::size_t>(w0));
        k::limbs_sub(last, w0, last - w0, w0, s.data(), w0);
        d.size = natural::from_limbs(s.data(), s.size());
    }
}

std::shared_ptr<const detail::space_data> make_data(std::int32_t total, const std::int32_t* lower,
                                                    const std::int32_t* upper, std::size_t parts) {
    if (parts == 0) throw std::invalid_argument("bounded_compositions::space: no parts");
    if (parts > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() - 1)) {
        throw std::invalid_argument("bounded_compositions::space: too many parts");
    }
    auto d = std::make_shared<detail::space_data>();
    d->total = total;
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
    const bool feasible = shifted >= 0 && shifted <= cap_sum;
    if (shifted > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(
            "bounded_compositions::space: total minus lower bounds exceeds int32");
    }
    const std::int32_t shifted_total = feasible ? static_cast<std::int32_t>(shifted) : 0;

    // cap_suffix[i] = min(shifted_total, cap[i] + ... + cap[n-1]).
    d->cap_suffix.assign(parts + 1, 0);
    std::int64_t run = 0;
    for (std::size_t i = parts; i-- > 0;) {
        run += d->cap[i];
        d->cap_suffix[i] = static_cast<std::int32_t>(std::min<std::int64_t>(run, shifted_total));
    }

    if (feasible) build_table(*d, shifted_total);

    k::space_view& v = d->view;
    v.n = static_cast<std::int32_t>(parts);
    v.total = shifted_total;
    v.lo = d->lower.data();
    v.cap = d->cap.data();
    v.cap_suffix = d->cap_suffix.data();
    if (feasible) {
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
    return d;
}

void check_length(std::size_t got, std::size_t want) {
    if (got != want) {
        throw std::invalid_argument("bounded_compositions::space: expected " +
                                    std::to_string(want) + " values, got " + std::to_string(got));
    }
}

}  // namespace

space::space(std::int32_t total, const std::vector<std::int32_t>& lower,
             const std::vector<std::int32_t>& upper) {
    if (lower.size() != upper.size()) {
        throw std::invalid_argument(
            "bounded_compositions::space: lower and upper differ in length");
    }
    data_ = make_data(total, lower.data(), upper.data(), lower.size());
}

space::space(std::int32_t total, const std::int32_t* lower, const std::int32_t* upper,
             std::size_t parts)
    : data_(make_data(total, lower, upper, parts)) {}

std::size_t space::parts() const noexcept { return data_->lower.size(); }
std::int32_t space::total() const noexcept { return data_->total; }
const std::vector<std::int32_t>& space::lower() const noexcept { return data_->lower; }
const std::vector<std::int32_t>& space::upper() const noexcept { return data_->upper; }
const natural& space::size() const noexcept { return data_->size; }
bool space::empty() const noexcept { return data_->size.is_zero(); }
k::space_view space::view() const noexcept { return data_->view; }

std::size_t space::table_bytes() const noexcept {
    const detail::space_data* d = data_.get();
    while (d->parent) d = d->parent.get();
    return d->table.size() * sizeof(k::limb);
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
    std::int64_t prefix_sum = 0;
    for (std::size_t i = 0; i < d; ++i) {
        const std::int64_t y = static_cast<std::int64_t>(prefix[i]) - v.lo[i];
        if (y < 0 || y > v.cap[i] || y > left) {
            throw std::invalid_argument("bounded_compositions::space::subspace: prefix value " +
                                        std::to_string(prefix[i]) + " at part " +
                                        std::to_string(i) + " is out of range");
        }
        left -= static_cast<std::int32_t>(y);
        prefix_sum += prefix[i];
    }
    // size = N[d][left] = P[d][left] - P[d][left - 1]
    const int row = static_cast<int>(d);
    const int w = v.row_limbs[row];
    const k::limb* hi = k::table_entry(v, row, left);
    natural size = natural::from_limbs(hi, static_cast<std::size_t>(w));
    if (left > 0) size -= natural::from_limbs(hi - w, static_cast<std::size_t>(w));
    if (size.is_zero()) {
        throw std::invalid_argument(
            "bounded_compositions::space::subspace: prefix has no completion");
    }

    auto sub = std::make_shared<detail::space_data>();
    sub->parent = data_;
    sub->total = static_cast<std::int32_t>(data_->total - prefix_sum);
    sub->lower.assign(data_->lower.begin() + static_cast<std::ptrdiff_t>(d), data_->lower.end());
    sub->upper.assign(data_->upper.begin() + static_cast<std::ptrdiff_t>(d), data_->upper.end());
    sub->size = std::move(size);
    k::space_view& sv = sub->view;
    sv = v;  // same table; rows and parts shifted by d
    sv.row_offset = v.row_offset + d;
    sv.row_limbs = v.row_limbs + d;
    sv.lo = v.lo + d;
    sv.cap = v.cap + d;
    sv.cap_suffix = v.cap_suffix + d;
    sv.size = sub->size.limbs();
    sv.size_limbs = static_cast<std::int32_t>(sub->size.limb_count());
    sv.n = static_cast<std::int32_t>(n - d);
    sv.total = left;
    return space(std::move(sub));
}

void space::unrank(const natural& r, std::int32_t* out) const {
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
    return k::next(data_->view, x) == k::status::ok;
}

}  // namespace bounded_compositions
