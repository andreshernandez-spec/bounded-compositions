// Measurements behind docs/sampling.md. Writes JSON records to the path given
// as the first argument; tools/make_sampling_report.py turns them into the
// report. Build with the release preset and -DBCOMP_BUILD_BENCH=ON.
//
// Every number is checked where it is made: point counts, membership of every
// point, and the sizes asked for.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "bounded_compositions/bounded_compositions.hpp"
#include "bounded_compositions/kernel/walk.hpp"

namespace bc = bounded_compositions;
namespace k = bounded_compositions::kernel;
using composition = std::vector<std::int32_t>;

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("measurement check failed: " + what);
}

// Minimal JSON writer: values are written as given, strings quoted.
std::string quote(const std::string& s) { return "\"" + s + "\""; }
std::string num(double v) {
    std::ostringstream os;
    os.precision(10);
    os << v;
    return os.str();
}
template <typename T>
std::string list(const std::vector<T>& v, std::function<std::string(const T&)> f) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + f(v[i]);
    return out + "]";
}

struct space_spec {
    std::string name;
    std::int32_t total;
    composition lo, hi;
};

std::vector<space_spec> spaces() {
    return {
        {"small", 30, composition(6, 0), composition(6, 12)},
        {"medium", 40, composition(8, 0), composition{20, 15, 25, 10, 30, 12, 18, 40}},
        {"near_cap", 590, composition(30, 0), composition(30, 20)},
        {"wide", 100, composition(100, 0), composition(100, 100)},
    };
}

// P(x_i = lower[i] + y) for every part, from the logarithmic tables
// (relative error about 1e-12): propagate the distribution over the states
// (layer, what is left to place) row by row.
std::vector<std::vector<double>> exact_marginals(const bc::space& logs) {
    const k::space_view v = logs.view();
    const std::size_t n = static_cast<std::size_t>(v.n);
    const std::size_t width = static_cast<std::size_t>(v.total) + 1;
    std::vector<double> dist(static_cast<std::size_t>(v.row_layers[0]) * width, 0.0);
    dist[static_cast<std::size_t>(v.layer_base) * width + static_cast<std::size_t>(v.total)] = 1.0;
    std::vector<std::vector<double>> marg(n);
    for (std::size_t i = 0; i < n; ++i) {
        const int row = static_cast<int>(i);
        marg[i].assign(static_cast<std::size_t>(v.cap[i]) + 1, 0.0);
        std::vector<double> next(static_cast<std::size_t>(v.row_layers[i + 1]) * width, 0.0);
        for (std::int32_t c = 0; c < v.row_layers[i]; ++c) {
            for (std::int32_t kk = 0; kk <= v.total; ++kk) {
                const double here =
                    dist[static_cast<std::size_t>(c) * width + static_cast<std::size_t>(kk)];
                if (here == 0.0) continue;
                const double whole = k::log_window(v, row, c, kk, kk, nullptr);
                for (int g = 0; g < v.segs[i]; ++g) {
                    std::int32_t a = 0, b = 0, cn = 0;
                    if (!k::move(v, row, g, kk, c, &a, &b, &cn)) continue;
                    for (std::int32_t y = a; y <= b; ++y) {
                        const double lc = k::log_window(v, row + 1, cn, kk - y, kk - y, nullptr);
                        if (k::log_is_zero(lc)) continue;
                        const double prob = here * std::exp(lc - whole);
                        marg[i][static_cast<std::size_t>(y)] += prob;
                        next[static_cast<std::size_t>(cn) * width +
                             static_cast<std::size_t>(kk - y)] += prob;
                    }
                }
            }
        }
        dist.swap(next);
    }
    for (const auto& m : marg) {
        double s = 0.0;
        for (double x : m) s += x;
        require(std::abs(s - 1.0) < 1e-9, "marginals sum to one");
    }
    return marg;
}

// The same space with its parts in reverse order.
bc::space reversed(const bc::space& s) {
    const std::size_t n = s.parts();
    const composition lo(s.lower().rbegin(), s.lower().rend());
    const composition hi(s.upper().rbegin(), s.upper().rend());
    bc::support_limits sup = s.support();
    std::reverse(sup.min_size.begin(), sup.min_size.end());
    std::vector<bc::group_limit> groups = s.groups();
    for (bc::group_limit& g : groups) {
        for (std::int32_t& p : g.parts) p = static_cast<std::int32_t>(n) - 1 - p;
    }
    return bc::space(s.total(), lo, hi, sup, groups);
}

// The first part's marginal from exact subtree sizes.
std::vector<double> first_part_marginal(const bc::space& s) {
    std::vector<double> m(static_cast<std::size_t>(s.upper()[0] - s.lower()[0]) + 1, 0.0);
    for (const bc::region& kid : bc::children(s, bc::whole(s))) {
        m[static_cast<std::size_t>(kid.prefix()[0] - s.lower()[0])] =
            kid.size().to_double() / s.size().to_double();
    }
    return m;
}

// The propagated marginals of the first and the last part against exact
// counts: the last part is the first of the reversed space.
void check_marginals(const bc::space& s, const std::vector<std::vector<double>>& marg) {
    const std::vector<double> head = first_part_marginal(s);
    const std::vector<double> tail = first_part_marginal(reversed(s));
    require(head.size() == marg.front().size() && tail.size() == marg.back().size(),
            "marginal lengths");
    for (std::size_t y = 0; y < head.size(); ++y) {
        require(std::abs(head[y] - marg.front()[y]) < 1e-9, "first part's marginal is exact");
    }
    for (std::size_t y = 0; y < tail.size(); ++y) {
        require(std::abs(tail[y] - marg.back()[y]) < 1e-9, "last part's marginal is exact");
    }
}

struct named_sequence {
    std::string name;
    std::function<bc::sequence(const bc::space&)> make;
};

std::vector<named_sequence> sequences() {
    return {
        {"rank", [](const bc::space& s) { return bc::rank_sequence(s); }},
        {"rank_scrambled", [](const bc::space& s) { return bc::rank_sequence(s, 1); }},
        {"coordinate", [](const bc::space& s) { return bc::coordinate_sequence(s); }},
        {"coordinate_scrambled", [](const bc::space& s) { return bc::coordinate_sequence(s, 1); }},
        {"iid", [](const bc::space& s) { return bc::iid_sequence(s, 1); }},
        {"scaled_rank", [](const bc::space& s) { return bc::scaled_rank_sequence(s); }},
    };
}

std::vector<std::uint64_t> checkpoints(const bc::natural& size) {
    std::vector<std::uint64_t> n = {16,   32,   64,   100,  128,  256,  512,   1000,
                                    1024, 1100, 2048, 3000, 4096, 8192, 10000, 16384};
    std::vector<std::uint64_t> out;
    for (std::uint64_t x : n) {
        if (bc::natural(x) <= size) out.push_back(x);
    }
    return out;
}

// Gap ratio, marginal error and distinct points of each sequence at every
// checkpoint, one record each.
void spread_rows(const std::string& name, const bc::space& s,
                 const std::vector<std::vector<double>>& marg,
                 const std::vector<named_sequence>& qs, std::vector<std::string>& rows) {
    const std::size_t n = s.parts();
    const std::vector<std::uint64_t> cps = checkpoints(s.size());
    const std::uint64_t most = cps.back();
    const double size = s.size().to_double();
    for (const named_sequence& q : qs) {
        const bc::sequence seq = q.make(s);
        std::vector<std::int32_t> pts(most * n);
        seq.points(0, most, pts.data());
        std::vector<std::vector<double>> counts(n);
        for (std::size_t i = 0; i < n; ++i) counts[i].assign(marg[i].size(), 0.0);
        std::vector<bc::natural> ranks;
        std::set<bc::natural> distinct;
        std::uint64_t done = 0;
        for (std::uint64_t cp : cps) {
            for (; done < cp; ++done) {
                const std::int32_t* x = pts.data() + done * n;
                require(s.contains(x), "every point is a composition");
                for (std::size_t i = 0; i < n; ++i)
                    counts[i][static_cast<std::size_t>(x[i] - s.lower()[i])] += 1.0;
                const bc::natural r = s.rank(x);
                ranks.push_back(r);
                distinct.insert(r);
            }
            require(ranks.size() == cp, "checkpoint size");
            // Largest gap between sampled ranks, with ends -1 and size.
            std::vector<bc::natural> sorted(distinct.begin(), distinct.end());
            double widest = sorted.front().to_double() + 1.0;
            for (std::size_t i = 1; i < sorted.size(); ++i) {
                widest = std::max(widest, (sorted[i] - sorted[i - 1]).to_double());
            }
            widest = std::max(widest, (s.size() - sorted.back()).to_double());
            std::vector<double> by_part(n);
            double worst = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double d = 0.0;
                for (std::size_t vv = 0; vv < marg[i].size(); ++vv) {
                    d = std::max(d,
                                 std::abs(counts[i][vv] / static_cast<double>(cp) - marg[i][vv]));
                }
                by_part[i] = d;
                worst = std::max(worst, d);
            }
            std::ostringstream row;
            row << "{\"space\": " << quote(name) << ", \"sequence\": " << quote(q.name)
                << ", \"n\": " << cp << ", \"distinct\": " << distinct.size()
                << ", \"gap_ratio\": " << num(widest * static_cast<double>(cp) / size)
                << ", \"marginal_max\": " << num(worst)
                << ", \"marginal_by_part\": " << list<double>(by_part, num) << "}";
            rows.push_back(row.str());
        }
    }
}

std::string spread_records(std::vector<std::string>& extra_parts) {
    std::vector<std::string> rows;
    for (const space_spec& p : spaces()) {
        const bc::space s(p.total, p.lo, p.hi);
        const std::size_t n = s.parts();
        const auto marg =
            exact_marginals(bc::space(p.total, p.lo, p.hi, bc::counting::logarithmic));
        check_marginals(s, marg);
        spread_rows(p.name, s, marg, sequences(), rows);
        std::ostringstream info;
        info << "{\"name\": " << quote(p.name) << ", \"parts\": " << n << ", \"total\": " << p.total
             << ", \"lower\": "
             << list<std::int32_t>(p.lo, [](const std::int32_t& v) { return std::to_string(v); })
             << ", \"upper\": "
             << list<std::int32_t>(p.hi, [](const std::int32_t& v) { return std::to_string(v); })
             << ", \"size\": " << quote(s.size().to_string())
             << ", \"size_bits\": " << s.size().bit_length()
             << ", \"table_bytes\": " << s.table_bytes() << "}";
        extra_parts.push_back(info.str());
    }
    return list<std::string>(rows, [](const std::string& r) { return r; });
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::string throughput_records() {
    std::vector<std::string> rows;
    const space_spec p = spaces()[3];  // wide
    const bc::space s(p.total, p.lo, p.hi);
    const std::uint64_t count = 200000;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    std::vector<unsigned> threads = {1, 2, 4, 8};
    if (hw > 8) threads.push_back(hw);
    const std::vector<named_sequence> qs = {sequences()[0], sequences()[3], sequences()[4]};
    std::vector<std::int32_t> pts(count * s.parts());
    for (const named_sequence& q : qs) {
        const bc::sequence seq = q.make(s);
        for (unsigned t : threads) {
            bc::thread_pool pool(t);
            std::vector<double> runs;
            const int repeats = 3;
            for (int r = 0; r < repeats; ++r) {
                std::fill(pts.begin(), pts.end(), -1);
                const auto t0 = std::chrono::steady_clock::now();
                seq.points(0, count, pts.data(), &pool);
                runs.push_back(seconds_since(t0));
                require(s.contains(pts.data() + (count - 1) * s.parts()), "last point written");
            }
            std::sort(runs.begin(), runs.end());
            std::ostringstream row;
            require(static_cast<int>(runs.size()) == repeats, "every repeat ran");
            row << "{\"space\": " << quote(p.name) << ", \"sequence\": " << quote(q.name)
                << ", \"threads\": " << t << ", \"runs\": " << repeats << ", \"points\": " << count
                << ", \"seconds\": " << num(runs[1])
                << ", \"points_per_second\": " << num(static_cast<double>(count) / runs[1]) << "}";
            rows.push_back(row.str());
        }
    }
    return list<std::string>(rows, [](const std::string& r) { return r; });
}

std::string build_records() {
    std::vector<std::string> rows;
    struct shape {
        std::size_t parts;
        std::int32_t total;
        std::int32_t cap;
        bool exact;
    };
    const std::vector<shape> shapes = {
        {10, 100, 100, true},      {100, 100, 100, true},     {100, 1000, 1000, true},
        {1000, 1000, 1000, true},  {10, 100, 100, false},     {100, 100, 100, false},
        {100, 1000, 1000, false},  {1000, 1000, 1000, false}, {2000, 2200, 2200, false},
        {3000, 3000, 3000, false},
    };
    for (const shape& sh : shapes) {
        const composition lo(sh.parts, 0), hi(sh.parts, sh.cap);
        const auto t0 = std::chrono::steady_clock::now();
        const bc::space s(sh.total, lo, hi,
                          sh.exact ? bc::counting::exact : bc::counting::logarithmic);
        const double secs = seconds_since(t0);
        require(!s.empty(), "built space is non-empty");
        std::ostringstream row;
        row << "{\"parts\": " << sh.parts << ", \"total\": " << sh.total << ", \"cap\": " << sh.cap
            << ", \"mode\": " << quote(sh.exact ? "exact" : "logarithmic")
            << ", \"seconds\": " << num(secs) << ", \"bytes\": " << s.table_bytes()
            << ", \"log_size\": " << num(s.log_size()) << "}";
        rows.push_back(row.str());
    }
    return list<std::string>(rows, [](const std::string& r) { return r; });
}

// Median nanoseconds per call of f over `calls` calls, three runs.
double per_call_ns(std::uint64_t calls, const std::function<void(std::uint64_t)>& f) {
    std::vector<double> runs;
    for (int r = 0; r < 3; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (std::uint64_t i = 0; i < calls; ++i) f(i);
        runs.push_back(seconds_since(t0) * 1e9 / static_cast<double>(calls));
    }
    std::sort(runs.begin(), runs.end());
    return runs[1];
}

std::string latency_records() {
    std::vector<std::string> rows;
    const space_spec p = spaces()[3];  // wide
    const bc::space exact(p.total, p.lo, p.hi);
    const bc::space logs(p.total, p.lo, p.hi, bc::counting::logarithmic);
    const std::size_t n = exact.parts();
    const std::uint64_t calls = 20000;
    std::vector<bc::natural> ranks(calls);
    const bc::sequence iid = bc::iid_sequence(exact, 7);
    for (std::uint64_t i = 0; i < calls; ++i) ranks[i] = iid.rank_at(i);
    std::vector<std::vector<std::uint64_t>> us(calls, std::vector<std::uint64_t>(n - 1));
    for (std::uint64_t i = 0; i < calls; ++i) {
        for (std::size_t j = 0; j + 1 < n; ++j) us[i][j] = (i + 1) * 0x9E3779B97F4A7C15u * (j + 3);
    }
    composition x(n);
    std::int64_t check = 0;
    auto add = [&](const std::string& op, double ns) {
        rows.push_back("{\"op\": " + quote(op) + ", \"space\": " + quote(p.name) +
                       ", \"ns\": " + num(ns) + "}");
    };
    add("unrank", per_call_ns(calls, [&](std::uint64_t i) {
            exact.unrank(ranks[i], x.data());
            check += x[0];
        }));
    add("rank", per_call_ns(calls, [&](std::uint64_t i) {
            exact.unrank(ranks[i], x.data());
            check += static_cast<std::int64_t>(exact.rank(x.data()).bit_length());
        }));
    add("decode_exact", per_call_ns(calls, [&](std::uint64_t i) {
            bc::decode(exact, us[i].data(), x.data());
            check += x[0];
        }));
    add("decode_log", per_call_ns(calls, [&](std::uint64_t i) {
            bc::decode(logs, us[i].data(), x.data());
            check += x[0];
        }));
    std::vector<std::uint64_t> enc(n - 1);
    std::vector<double> encd(n - 1);
    add("encode_exact", per_call_ns(500, [&](std::uint64_t i) {
            exact.unrank(ranks[i], x.data());
            bc::encode(exact, x.data(), enc.data());
            check += static_cast<std::int64_t>(enc[0] & 1u);
        }));
    add("encode_log", per_call_ns(500, [&](std::uint64_t i) {
            exact.unrank(ranks[i], x.data());
            bc::encode(logs, x.data(), encd.data());
            check += encd[0] > 0.5 ? 1 : 0;
        }));
    require(check != 0, "latency loops did work");
    return list<std::string>(rows, [](const std::string& r) { return r; });
}

// Largest gap between the sorted ranks inside r, counting the ends
// begin - 1 and end, times count / size.
double region_gap_ratio(std::vector<bc::natural> ranks, const bc::region& r) {
    std::sort(ranks.begin(), ranks.end());
    double widest = (ranks.front() - r.begin()).to_double() + 1.0;
    for (std::size_t i = 1; i < ranks.size(); ++i) {
        widest = std::max(widest, (ranks[i] - ranks[i - 1]).to_double());
    }
    widest = std::max(widest, (r.end() - ranks.back()).to_double());
    return widest * static_cast<double>(ranks.size()) / r.size().to_double();
}

// Every first-level subtree resumed after the whole space's first points:
// repeats, and the spread of each subtree's points afterwards.
std::string resume_records() {
    std::vector<std::string> rows;
    const std::uint64_t first = 4096, more = 256;
    for (const space_spec& p : spaces()) {
        const bc::space s(p.total, p.lo, p.hi);
        const std::size_t n = s.parts();
        bc::sequence global = bc::rank_sequence(s);
        std::vector<std::int32_t> pts(first * n);
        global.take(first, pts.data());
        std::set<bc::natural> seen;
        std::vector<bc::natural> before;
        for (std::uint64_t i = 0; i < first; ++i) {
            before.push_back(s.rank(pts.data() + i * n));
            seen.insert(before.back());
        }
        require(seen.size() == first, "the whole space's points are distinct");
        std::uint64_t repeats = 0, added = 0, measured = 0;
        double worst = 0.0;
        const std::vector<bc::region> kids = bc::children(s, bc::whole(s));
        for (const bc::region& kid : kids) {
            bc::sequence local = bc::rank_sequence(s, kid);
            local.resume_after(global);
            std::vector<bc::natural> inside;
            for (const bc::natural& r : before) {
                if (kid.contains(r)) inside.push_back(r);
            }
            require(local.position() == inside.size(), "resume skips the points already taken");
            const std::uint64_t k = std::min(more, local.capacity() - local.position());
            std::vector<std::int32_t> extra(k * n);
            local.take(k, extra.data());
            for (std::uint64_t i = 0; i < k; ++i) {
                const bc::natural r = s.rank(extra.data() + i * n);
                require(kid.contains(r), "resumed points stay in the subtree");
                if (!seen.insert(r).second) ++repeats;
                inside.push_back(r);
            }
            added += k;
            if (inside.size() >= 2) {
                worst = std::max(worst, region_gap_ratio(inside, kid));
                ++measured;
            }
        }
        std::ostringstream row;
        row << "{\"space\": " << quote(p.name) << ", \"first\": " << first
            << ", \"per_region\": " << more << ", \"regions\": " << kids.size()
            << ", \"measured\": " << measured << ", \"added\": " << added
            << ", \"repeats\": " << repeats << ", \"worst_gap_ratio\": " << num(worst) << "}";
        rows.push_back(row.str());
    }
    return list<std::string>(rows, [](const std::string& r) { return r; });
}

// fresh_sampler on a small subtree after the whole space's first points were
// evaluated: what each source hands out before the subtree runs dry.
std::string fresh_records() {
    std::vector<std::string> rows;
    const space_spec p = spaces()[0];  // small
    const bc::space s(p.total, p.lo, p.hi);
    const std::size_t n = s.parts();
    const std::uint64_t first = 16384;
    const composition prefix{5, 5, 5};
    const bc::region t = bc::subtree(s, prefix);
    const std::uint64_t size = t.size().to_u64();
    const bc::sequence global = bc::rank_sequence(s);
    std::uint64_t inside = 0;
    for (std::uint64_t i = 0; i < first; ++i) {
        if (t.contains(global.rank_at(i))) ++inside;
    }
    const std::vector<named_sequence> sources = {
        {"rank", [&](const bc::space&) { return bc::rank_sequence(s, t); }},
        {"coordinate_scrambled",
         [&](const bc::space&) { return bc::coordinate_sequence(s, t, 1); }},
        {"iid", [&](const bc::space&) { return bc::iid_sequence(s, t, 1); }},
    };
    for (const named_sequence& q : sources) {
        bc::visited_set visited(s);
        for (std::uint64_t i = 0; i < first; ++i) visited.insert(global.rank_at(i));
        bc::fresh_sampler fresh(q.make(s), visited);
        std::vector<std::int32_t> out(size * n);
        const std::uint64_t got = fresh.take(size, out.data());
        std::set<bc::natural> distinct;
        for (std::uint64_t i = 0; i < got; ++i) {
            const bc::natural r = s.rank(out.data() + i * n);
            require(t.contains(r), "fresh points stay in the subtree");
            distinct.insert(r);
        }
        require(distinct.size() == got, "fresh points are distinct");
        require(visited.size() == first + got, "every fresh point was unvisited");
        std::ostringstream row;
        row << "{\"space\": " << quote(p.name) << ", \"source\": " << quote(q.name)
            << ", \"prefix\": "
            << list<std::int32_t>(prefix, [](const std::int32_t& v) { return std::to_string(v); })
            << ", \"region_size\": " << size << ", \"first\": " << first
            << ", \"visited_inside\": " << inside << ", \"handed_out\": " << got
            << ", \"skipped\": " << fresh.skipped()
            << ", \"switched\": " << (fresh.switched() ? "true" : "false")
            << ", \"exhausted\": " << (fresh.exhausted() ? "true" : "false") << "}";
        rows.push_back(row.str());
    }
    return list<std::string>(rows, [](const std::string& r) { return r; });
}

// Section 6 of the report: one base space, 30 parts summing to 100 with each
// part 0..20, under each kind of limit.
struct limited_spec {
    std::string name;
    bc::support_limits support;
    std::vector<bc::group_limit> groups;
};

const std::int32_t limited_parts = 30, limited_total = 100, limited_cap = 20;

bc::group_limit run_of(std::int32_t first, std::int32_t count, std::int32_t min_sum,
                       std::int32_t max_sum) {
    bc::group_limit g;
    for (std::int32_t p = first; p < first + count; ++p) g.parts.push_back(p);
    g.min_sum = min_sum;
    g.max_sum = max_sum;
    return g;
}

std::vector<limited_spec> limited_specs() {
    bc::support_limits count;
    count.min_count = 5;
    count.max_count = 12;
    bc::support_limits size;
    size.min_size.assign(static_cast<std::size_t>(limited_parts), 3);
    bc::support_limits both = count;
    both.min_size = size.min_size;
    // Six groups of five adjacent parts; the two halves of three groups each
    // nest them; the even parts and parts 10 to 19 cross.
    std::vector<bc::group_limit> six;
    for (std::int32_t g = 0; g < 6; ++g) six.push_back(run_of(5 * g, 5, 10, 30));
    std::vector<bc::group_limit> nested = six;
    nested.push_back(run_of(0, 15, 40, 60));
    nested.push_back(run_of(15, 15, 40, 60));
    bc::group_limit even;
    for (std::int32_t p = 0; p < limited_parts; p += 2) even.parts.push_back(p);
    even.min_sum = 40;
    even.max_sum = 60;
    return {
        {"base", {}, {}},
        {"count", count, {}},
        {"min_size", size, {}},
        {"support", both, {}},
        {"groups", {}, six},
        {"nested", {}, nested},
        {"crossing", {}, {even, run_of(10, 10, 20, 40)}},
        {"combined", both, six},
    };
}

// per_call_ns over enough calls for about 0.2 s, at least 3 and at most most.
double adaptive_ns(std::uint64_t most, const std::function<void(std::uint64_t)>& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f(0);
    const double want = 0.2 / std::max(seconds_since(t0), 1e-9);
    const std::uint64_t calls =
        std::min<std::uint64_t>(most, std::max<std::uint64_t>(3, static_cast<std::uint64_t>(want)));
    return per_call_ns(calls, f);
}

std::string ints(const std::vector<std::int32_t>& v) {
    return list<std::int32_t>(v, [](const std::int32_t& x) { return std::to_string(x); });
}

// Each limited space: its table, the spread of its sequences, and the cost of
// single operations.
void limited_records(std::string& spaces_out, std::string& spread_out, std::string& ops_out) {
    std::vector<std::string> infos, spread, ops;
    const composition lo(static_cast<std::size_t>(limited_parts), 0);
    const composition hi(static_cast<std::size_t>(limited_parts), limited_cap);
    const std::vector<named_sequence> all = sequences();
    const std::vector<named_sequence> qs = {all[0], all[1], all[3], all[4]};
    for (const limited_spec& p : limited_specs()) {
        std::fprintf(stderr, "  %s\n", p.name.c_str());
        const int builds = 3;
        std::vector<double> exact_runs, log_runs;
        for (int r = 0; r < builds; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            const bc::space e(limited_total, lo, hi, p.support, p.groups);
            exact_runs.push_back(seconds_since(t0));
            t0 = std::chrono::steady_clock::now();
            const bc::space l(limited_total, lo, hi, p.support, p.groups,
                              bc::counting::logarithmic);
            log_runs.push_back(seconds_since(t0));
        }
        std::sort(exact_runs.begin(), exact_runs.end());
        std::sort(log_runs.begin(), log_runs.end());
        require(static_cast<int>(exact_runs.size()) == builds, "every build ran");
        const bc::space s(limited_total, lo, hi, p.support, p.groups);
        const bc::space logs(limited_total, lo, hi, p.support, p.groups, bc::counting::logarithmic);
        require(!s.empty() && !logs.empty(), "limited space is non-empty");
        const std::size_t n = s.parts();
        const k::space_view v = s.view();
        std::int32_t layers = 0;
        for (std::size_t r = 0; r <= n; ++r) layers = std::max(layers, v.row_layers[r]);
        std::vector<std::string> groups;
        for (const bc::group_limit& g : p.groups) {
            groups.push_back("{\"parts\": " + ints(g.parts) +
                             ", \"min_sum\": " + std::to_string(g.min_sum) +
                             ", \"max_sum\": " + std::to_string(g.max_sum) + "}");
        }
        std::ostringstream info;
        info << "{\"name\": " << quote(p.name) << ", \"parts\": " << n
             << ", \"total\": " << limited_total << ", \"lower\": " << ints(lo)
             << ", \"upper\": " << ints(hi) << ", \"min_count\": " << p.support.min_count
             << ", \"max_count\": " << p.support.max_count
             << ", \"min_size\": " << ints(p.support.min_size) << ", \"groups\": "
             << list<std::string>(groups, [](const std::string& r) { return r; })
             << ", \"size\": " << quote(s.size().to_string())
             << ", \"size_bits\": " << s.size().bit_length() << ", \"layers\": " << layers
             << ", \"table_bytes\": " << s.table_bytes()
             << ", \"log_table_bytes\": " << logs.table_bytes() << ", \"builds\": " << builds
             << ", \"build_seconds\": " << num(exact_runs[1])
             << ", \"log_build_seconds\": " << num(log_runs[1]) << "}";
        infos.push_back(info.str());

        const auto marg = exact_marginals(logs);
        check_marginals(s, marg);
        spread_rows(p.name, s, marg, qs, spread);

        const std::uint64_t most = 20000;
        const bc::sequence iid = bc::iid_sequence(s, 7);
        std::vector<bc::natural> ranks(most);
        for (std::uint64_t i = 0; i < most; ++i) ranks[i] = iid.rank_at(i);
        std::vector<std::vector<std::uint64_t>> us(most, std::vector<std::uint64_t>(n - 1));
        for (std::uint64_t i = 0; i < most; ++i) {
            for (std::size_t j = 0; j + 1 < n; ++j)
                us[i][j] = (i + 1) * 0x9E3779B97F4A7C15u * (j + 3);
        }
        // Repair targets around the space: each part uniform in [0, 2 total / n].
        std::mt19937 rng(11);
        std::uniform_real_distribution<double> unit(0.0, 2.0 * limited_total / limited_parts);
        std::vector<std::vector<double>> zs(most, std::vector<double>(n));
        for (auto& z : zs) {
            for (double& zi : z) zi = unit(rng);
        }
        composition x(n);
        for (std::uint64_t i = 0; i < 20; ++i) {
            bc::repair(s, zs[i].data(), x.data());
            require(s.contains(x), "repair gives a composition of the space");
        }
        std::int64_t check = 0;
        auto add = [&](const std::string& op, double ns) {
            ops.push_back("{\"op\": " + quote(op) + ", \"space\": " + quote(p.name) +
                          ", \"ns\": " + num(ns) + "}");
        };
        add("unrank", adaptive_ns(most, [&](std::uint64_t i) {
                s.unrank(ranks[i], x.data());
                check += x[0];
            }));
        add("rank", adaptive_ns(most, [&](std::uint64_t i) {
                s.unrank(ranks[i], x.data());
                check += static_cast<std::int64_t>(s.rank(x.data()).bit_length());
            }));
        add("decode_exact", adaptive_ns(most, [&](std::uint64_t i) {
                bc::decode(s, us[i].data(), x.data());
                check += x[0];
            }));
        add("decode_log", adaptive_ns(most, [&](std::uint64_t i) {
                bc::decode(logs, us[i].data(), x.data());
                check += x[0];
            }));
        add("repair", adaptive_ns(most, [&](std::uint64_t i) {
                bc::repair(s, zs[i].data(), x.data());
                check += x[0];
            }));
        require(check != 0, "operation loops did work");
    }
    auto join = [](const std::vector<std::string>& v) {
        return list<std::string>(v, [](const std::string& r) { return r; });
    };
    spaces_out = join(infos);
    spread_out = join(spread);
    ops_out = join(ops);
}

// A weighted budget, weights[i] x[i] summed over the parts at most limit: a
// linear limit the table does not count, for the filter.
struct budget {
    std::vector<std::int64_t> weights;
    std::int64_t limit;
};

bool within_budget(void* context, const std::int32_t* x) {
    const budget& b = *static_cast<const budget*>(context);
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < b.weights.size(); ++i) sum += b.weights[i] * x[i];
    return sum <= b.limit;
}

// Weights 1, 2, ..., parts.
budget rising_budget(std::size_t parts, std::int64_t limit) {
    budget b{std::vector<std::int64_t>(parts), limit};
    for (std::size_t i = 0; i < parts; ++i) b.weights[i] = static_cast<std::int64_t>(i) + 1;
    return b;
}

std::string budget_fields(const budget& b) {
    return "\"weights\": " +
           list<std::int64_t>(b.weights, [](const std::int64_t& w) { return std::to_string(w); }) +
           ", \"limit\": " + std::to_string(b.limit);
}

// Calls f on every composition of subtree t of s.
void for_each_in(const bc::space& s, const bc::region& t,
                 const std::function<void(const std::int32_t*)>& f) {
    const composition& prefix = t.prefix();
    composition x(s.parts());
    std::copy(prefix.begin(), prefix.end(), x.begin());
    if (prefix.size() == s.parts()) {
        f(x.data());
        return;
    }
    const bc::space sub = s.subspace(prefix);
    composition y(sub.parts());
    if (!sub.first(y.data())) return;
    do {
        std::copy(y.begin(), y.end(), x.begin() + static_cast<std::ptrdiff_t>(prefix.size()));
        f(x.data());
    } while (sub.next(y.data()));
}

// The budget's acceptance in each first-level subtree of the medium space:
// exact by enumeration, and estimated from the first points of each
// subtree's sequences, one replicate per seed.
std::string acceptance_records() {
    const space_spec p = spaces()[1];  // medium
    const bc::space s(p.total, p.lo, p.hi);
    const std::size_t n = s.parts();
    budget b = rising_budget(n, 160);
    const std::vector<std::uint64_t> draws = {16, 32, 64, 128, 256, 512, 1024};
    const std::uint64_t seeds = 16;
    const std::vector<std::string> kinds = {"rank_scrambled", "coordinate_scrambled", "iid"};
    auto counts = [](const std::vector<std::uint64_t>& v) {
        return list<std::uint64_t>(v, [](const std::uint64_t& c) { return std::to_string(c); });
    };
    std::vector<std::string> rows;
    std::vector<std::int32_t> pts(draws.back() * n);
    for (const bc::region& kid : bc::children(s, bc::whole(s))) {
        std::uint64_t seen = 0, kept = 0;
        for_each_in(s, kid, [&](const std::int32_t* x) {
            ++seen;
            if (within_budget(&b, x)) ++kept;
        });
        require(bc::natural(seen) == kid.size(), "enumerated the whole subtree");
        std::vector<std::string> estimates;
        for (const std::string& kind : kinds) {
            std::vector<std::string> replicates;
            for (std::uint64_t seed = 1; seed <= seeds; ++seed) {
                const bc::sequence seq = kind == "rank_scrambled" ? bc::rank_sequence(s, kid, seed)
                                         : kind == "iid" ? bc::iid_sequence(s, kid, seed)
                                                         : bc::coordinate_sequence(s, kid, seed);
                require(seq.capacity() >= draws.back(), "subtree holds the draws");
                seq.points(0, draws.back(), pts.data());
                std::vector<std::uint64_t> accepted;
                std::uint64_t a = 0, d = 0;
                for (std::uint64_t want : draws) {
                    for (; d < want; ++d) {
                        const std::int32_t* x = pts.data() + d * n;
                        require(kid.contains(s.rank(x)), "draws stay in the subtree");
                        if (within_budget(&b, x)) ++a;
                    }
                    accepted.push_back(a);
                }
                replicates.push_back(counts(accepted));
            }
            require(replicates.size() == seeds, "every seed ran");
            estimates.push_back(
                quote(kind) + ": " +
                list<std::string>(replicates, [](const std::string& r) { return r; }));
        }
        std::string est = "{";
        for (std::size_t i = 0; i < estimates.size(); ++i) est += (i ? ", " : "") + estimates[i];
        rows.push_back(
            "{\"prefix\": " + ints(kid.prefix()) + ", \"size\": " + std::to_string(seen) +
            ", \"accepted\": " + std::to_string(kept) + ", \"estimates\": " + est + "}}");
    }
    return "{\"space\": " + quote(p.name) + ", " + budget_fields(b) +
           ", \"draws\": " + counts(draws) + ", \"seeds\": " + std::to_string(seeds) +
           ", \"subtrees\": " + list<std::string>(rows, [](const std::string& r) { return r; }) +
           "}";
}

// fresh_sampler with the budget as its filter, on the subtree of
// fresh_records after the same points were visited: every acceptable point
// not visited comes out exactly once.
std::string fresh_filter_records() {
    std::vector<std::string> rows;
    const space_spec p = spaces()[0];  // small
    const bc::space s(p.total, p.lo, p.hi);
    const std::size_t n = s.parts();
    const std::uint64_t first = 16384;
    const composition prefix{5, 5, 5};
    budget b = rising_budget(n, 100);
    const bc::region t = bc::subtree(s, prefix);
    const std::uint64_t size = t.size().to_u64();
    const bc::sequence global = bc::rank_sequence(s);
    std::set<bc::natural> before;
    for (std::uint64_t i = 0; i < first; ++i) before.insert(global.rank_at(i));
    std::uint64_t inside = 0, new_kept = 0, new_dropped = 0;
    for_each_in(s, t, [&](const std::int32_t* x) {
        if (before.count(s.rank(x)) != 0) {
            ++inside;
        } else if (within_budget(&b, x)) {
            ++new_kept;
        } else {
            ++new_dropped;
        }
    });
    require(inside + new_kept + new_dropped == size, "enumerated the whole subtree");
    const std::vector<named_sequence> sources = {
        {"rank", [&](const bc::space&) { return bc::rank_sequence(s, t); }},
        {"coordinate_scrambled",
         [&](const bc::space&) { return bc::coordinate_sequence(s, t, 1); }},
        {"iid", [&](const bc::space&) { return bc::iid_sequence(s, t, 1); }},
    };
    for (const named_sequence& q : sources) {
        bc::visited_set visited(s);
        for (const bc::natural& r : before) visited.insert(r);
        bc::fresh_sampler fresh(q.make(s), visited, &within_budget, &b);
        std::vector<std::int32_t> out(size * n);
        const std::uint64_t got = fresh.take(size, out.data());
        std::set<bc::natural> distinct;
        for (std::uint64_t i = 0; i < got; ++i) {
            const std::int32_t* x = out.data() + i * n;
            const bc::natural r = s.rank(x);
            require(t.contains(r) && within_budget(&b, x),
                    "filtered points are accepted and inside");
            distinct.insert(r);
        }
        require(distinct.size() == got, "filtered points are distinct");
        require(got == new_kept && fresh.accepted() == got, "every acceptable new point came out");
        require(fresh.exhausted(), "the subtree ran out");
        if (q.name == "rank") {
            require(fresh.rejected() == new_dropped && fresh.skipped() == inside,
                    "the rank sequence draws each point once");
        }
        std::ostringstream row;
        row << "{\"space\": " << quote(p.name) << ", \"source\": " << quote(q.name)
            << ", \"prefix\": " << ints(prefix) << ", \"region_size\": " << size
            << ", \"first\": " << first << ", \"visited_inside\": " << inside
            << ", \"new_accepted\": " << new_kept << ", \"new_rejected\": " << new_dropped << ", "
            << budget_fields(b) << ", \"handed_out\": " << got
            << ", \"rejected\": " << fresh.rejected() << ", \"skipped\": " << fresh.skipped()
            << ", \"switched\": " << (fresh.switched() ? "true" : "false") << "}";
        rows.push_back(row.str());
    }
    return list<std::string>(rows, [](const std::string& r) { return r; });
}

std::string machine_record() {
    std::string model = "unknown";
    std::ifstream cpu("/proc/cpuinfo");
    for (std::string line; std::getline(cpu, line);) {
        if (line.rfind("model name", 0) == 0) {
            model = line.substr(line.find(':') + 2);
            break;
        }
    }
    std::ostringstream os;
    os << "{\"cpu\": " << quote(model)
       << ", \"hardware_threads\": " << std::thread::hardware_concurrency()
#if defined(__clang__)
       << ", \"compiler\": " << quote(std::string("clang ") + __clang_version__)
#elif defined(__GNUC__)
       << ", \"compiler\": " << quote(std::string("gcc ") + __VERSION__)
#else
       << ", \"compiler\": " << quote("unknown")
#endif
#if defined(NDEBUG)
       << ", \"optimised\": true"
#else
       << ", \"optimised\": false"
#endif
       << ", \"library_version\": " << quote(bc::version()) << "}";
    return os.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <records.json>\n", argv[0]);
        return 2;
    }
    std::vector<std::string> space_info;
    std::fprintf(stderr, "spread...\n");
    const std::string spread = spread_records(space_info);
    std::fprintf(stderr, "regions...\n");
    const std::string resume = resume_records();
    const std::string fresh = fresh_records();
    std::fprintf(stderr, "limited spaces...\n");
    std::string limited_spaces, limited_spread, limited_ops;
    limited_records(limited_spaces, limited_spread, limited_ops);
    std::fprintf(stderr, "filters...\n");
    const std::string acceptance = acceptance_records();
    const std::string fresh_filter = fresh_filter_records();
    std::fprintf(stderr, "throughput...\n");
    const std::string throughput = throughput_records();
    std::fprintf(stderr, "build cost...\n");
    const std::string build = build_records();
    std::fprintf(stderr, "latency...\n");
    const std::string latency = latency_records();
    std::ofstream out(argv[1]);
    out << "{\n\"machine\": " << machine_record() << ",\n\"spaces\": "
        << list<std::string>(space_info, [](const std::string& r) { return r; })
        << ",\n\"spread\": " << spread << ",\n\"resume\": " << resume << ",\n\"fresh\": " << fresh
        << ",\n\"limited_spaces\": " << limited_spaces
        << ",\n\"limited_spread\": " << limited_spread << ",\n\"limited_ops\": " << limited_ops
        << ",\n\"acceptance\": " << acceptance << ",\n\"fresh_filter\": " << fresh_filter
        << ",\n\"throughput\": " << throughput << ",\n\"build\": " << build
        << ",\n\"latency\": " << latency << "\n}\n";
    require(static_cast<bool>(out), "records written");
    std::fprintf(stderr, "wrote %s\n", argv[1]);
    return 0;
}
