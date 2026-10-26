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

// P(x_i = v) for every part, from the logarithmic tables (relative error
// about 1e-12): propagate the distribution of what is left to place.
std::vector<std::vector<double>> exact_marginals(const space_spec& p) {
    const bc::space logs(p.total, p.lo, p.hi, bc::counting::logarithmic);
    const k::space_view v = logs.view();
    const std::size_t n = p.lo.size();
    auto log_count = [&](int row, std::int32_t kk) {
        return k::log_window(v, row, kk, kk, nullptr);
    };
    std::vector<double> dist(static_cast<std::size_t>(v.total) + 1, 0.0);
    dist[static_cast<std::size_t>(v.total)] = 1.0;
    std::vector<std::vector<double>> marg(n);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        marg[i].assign(static_cast<std::size_t>(v.cap[i]) + 1, 0.0);
        std::vector<double> next(dist.size(), 0.0);
        for (std::int32_t kk = 0; kk <= v.total; ++kk) {
            const double here = dist[static_cast<std::size_t>(kk)];
            if (here == 0.0) continue;
            const double total_log = log_count(static_cast<int>(i), kk);
            for (std::int32_t y = 0; y <= std::min(v.cap[i], kk); ++y) {
                const double lc = log_count(static_cast<int>(i) + 1, kk - y);
                if (k::log_is_zero(lc)) continue;
                const double prob = here * std::exp(lc - total_log);
                marg[i][static_cast<std::size_t>(y)] += prob;
                next[static_cast<std::size_t>(kk - y)] += prob;
            }
        }
        dist.swap(next);
    }
    // The last part takes whatever is left.
    marg[n - 1].assign(static_cast<std::size_t>(v.cap[n - 1]) + 1, 0.0);
    for (std::int32_t kk = 0; kk <= v.total; ++kk) {
        if (kk <= v.cap[n - 1])
            marg[n - 1][static_cast<std::size_t>(kk)] += dist[static_cast<std::size_t>(kk)];
    }
    for (const auto& m : marg) {
        double s = 0.0;
        for (double x : m) s += x;
        require(std::abs(s - 1.0) < 1e-9, "marginals sum to one");
    }
    return marg;
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

std::string spread_records(std::vector<std::string>& extra_parts) {
    std::vector<std::string> rows;
    for (const space_spec& p : spaces()) {
        const bc::space s(p.total, p.lo, p.hi);
        const std::size_t n = s.parts();
        const auto marg = exact_marginals(p);
        const std::vector<std::uint64_t> cps = checkpoints(s.size());
        const std::uint64_t most = cps.back();
        const double size = s.size().to_double();
        for (const named_sequence& q : sequences()) {
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
                        counts[i][static_cast<std::size_t>(x[i] - p.lo[i])] += 1.0;
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
                        d = std::max(
                            d, std::abs(counts[i][vv] / static_cast<double>(cp) - marg[i][vv]));
                    }
                    by_part[i] = d;
                    worst = std::max(worst, d);
                }
                std::ostringstream row;
                row << "{\"space\": " << quote(p.name) << ", \"sequence\": " << quote(q.name)
                    << ", \"n\": " << cp << ", \"distinct\": " << distinct.size()
                    << ", \"gap_ratio\": " << num(widest * static_cast<double>(cp) / size)
                    << ", \"marginal_max\": " << num(worst)
                    << ", \"marginal_by_part\": " << list<double>(by_part, num) << "}";
                rows.push_back(row.str());
            }
        }
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
        << ",\n\"throughput\": " << throughput << ",\n\"build\": " << build
        << ",\n\"latency\": " << latency << "\n}\n";
    require(static_cast<bool>(out), "records written");
    std::fprintf(stderr, "wrote %s\n", argv[1]);
    return 0;
}
