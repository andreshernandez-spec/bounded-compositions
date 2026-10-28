// Python bindings, bounded_compositions._core. Names follow Python (Space,
// Sequence, ...); behaviour follows the C++ API. Compositions go in as any
// integer array-like and come out as int32 NumPy arrays; sizes and ranks are
// Python ints. Table builds, sampling and batch calls release the GIL.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "bounded_compositions/bounded_compositions.hpp"

namespace nb = nanobind;
namespace bc = bounded_compositions;
using namespace nb::literals;

namespace {

using limb = bc::kernel::limb;
constexpr std::int32_t no_limit = std::numeric_limits<std::int32_t>::max();

nb::module_ numpy() { return nb::module_::import_("numpy"); }

// Python truth, which also takes NumPy's bool scalars (nb::cast<bool> does not).
bool truthy(nb::handle h) {
    const int t = PyObject_IsTrue(h.ptr());
    if (t < 0) throw nb::python_error();
    return t != 0;
}

// ---------------------------------------------------------------- integers

// A non-negative Python integer (or anything with __index__) as a natural.
bc::natural to_natural(nb::handle value, const char* what) {
    nb::object v = nb::module_::import_("operator").attr("index")(value);
    if (nb::cast<bool>(v.attr("__lt__")(0))) {
        throw nb::value_error((std::string(what) + " must be non-negative").c_str());
    }
    const std::size_t bits = nb::cast<std::size_t>(v.attr("bit_length")());
    const std::size_t limbs = (bits + 63) / 64;
    const nb::bytes raw = nb::borrow<nb::bytes>(v.attr("to_bytes")(limbs * 8, "little"));
    const auto* bytes = reinterpret_cast<const unsigned char*>(raw.c_str());
    std::vector<limb> out(limbs, 0);
    for (std::size_t i = 0; i < limbs * 8; ++i) out[i / 8] |= limb{bytes[i]} << (8 * (i % 8));
    return bc::natural::from_limbs(out.data(), out.size());
}

nb::object to_int(const bc::natural& n) {
    std::string raw(n.limb_count() * 8, '\0');
    for (std::size_t i = 0; i < raw.size(); ++i) {
        raw[i] = static_cast<char>((n.limbs()[i / 8] >> (8 * (i % 8))) & 0xffu);
    }
    return nb::module_::import_("builtins")
        .attr("int")
        .attr("from_bytes")(nb::bytes(raw.data(), raw.size()), "little");
}

// ------------------------------------------------------------------ arrays

using i32_array = nb::ndarray<std::int32_t, nb::c_contig, nb::device::cpu>;
using f64_array = nb::ndarray<double, nb::c_contig, nb::device::cpu>;
using u64_array = nb::ndarray<std::uint64_t, nb::c_contig, nb::device::cpu>;

// An integer array-like as a C-contiguous int32 NumPy array of 1 or 2
// dimensions (ndim 0 accepts either). Values outside int32 raise, they never
// wrap.
nb::object int32_array(nb::handle obj, int ndim, const char* what) {
    nb::module_ np = numpy();
    nb::object a = np.attr("asarray")(obj);
    const std::string kind = nb::cast<std::string>(a.attr("dtype").attr("kind"));
    const bool none = nb::cast<std::size_t>(a.attr("size")) == 0;  // [] comes as float64
    if (kind != "i" && kind != "u" && !none) {
        throw nb::type_error((std::string(what) + " must hold integers").c_str());
    }
    const int dims = nb::cast<int>(a.attr("ndim"));
    if ((ndim != 0 && dims != ndim) || dims < 1 || dims > 2) {
        throw nb::value_error((std::string(what) + " has the wrong number of dimensions").c_str());
    }
    if (nb::cast<std::size_t>(a.attr("size")) > 0 &&
        nb::cast<int>(a.attr("dtype").attr("itemsize")) >= 4 &&
        !(kind == "i" && nb::cast<int>(a.attr("dtype").attr("itemsize")) == 4)) {
        const bool low =
            kind == "i" &&
            truthy(np.attr("any")(np.attr("less")(a, std::numeric_limits<int>::min())));
        const bool high =
            truthy(np.attr("any")(np.attr("greater")(a, std::numeric_limits<int>::max())));
        if (low || high) {
            throw nb::value_error((std::string(what) + " holds values outside int32").c_str());
        }
    }
    return np.attr("ascontiguousarray")(a, "dtype"_a = "int32");
}

// A float array-like as a C-contiguous float64 array, or a uint64 array kept
// as uint64 (64-bit fractions) when allow_u64 is set.
nb::object float_array(nb::handle obj, bool allow_u64, const char* what) {
    nb::module_ np = numpy();
    nb::object a = np.attr("asarray")(obj);
    const std::string name = nb::cast<std::string>(a.attr("dtype").attr("name"));
    const int dims = nb::cast<int>(a.attr("ndim"));
    if (dims < 1 || dims > 2) {
        throw nb::value_error((std::string(what) + " must have 1 or 2 dimensions").c_str());
    }
    if (allow_u64 && name == "uint64") return np.attr("ascontiguousarray")(a);
    const std::string kind = nb::cast<std::string>(a.attr("dtype").attr("kind"));
    if (kind != "f" && kind != "i" && kind != "u") {
        throw nb::type_error((std::string(what) + " must hold real numbers").c_str());
    }
    return np.attr("ascontiguousarray")(a, "dtype"_a = "float64");
}

std::size_t rows_of(nb::handle a) {
    return nb::cast<int>(a.attr("ndim")) == 1
               ? 1
               : nb::cast<std::size_t>(a.attr("shape").attr("__getitem__")(0));
}
std::size_t cols_of(nb::handle a) {
    const nb::object shape = a.attr("shape");
    return nb::cast<std::size_t>(shape.attr("__getitem__")(-1));
}

void require_cols(nb::handle a, std::size_t want, const char* what) {
    if (cols_of(a) != want) {
        throw nb::value_error((std::string(what) + ": expected " + std::to_string(want) +
                               " values per row, got " + std::to_string(cols_of(a)))
                                  .c_str());
    }
}

// An uninitialised C-contiguous array of the given dtype, 1-D when rows is
// absent.
nb::object empty(std::optional<std::size_t> rows, std::size_t cols, const char* dtype) {
    nb::module_ np = numpy();
    if (rows) return np.attr("empty")(nb::make_tuple(*rows, cols), "dtype"_a = dtype);
    return np.attr("empty")(cols, "dtype"_a = dtype);
}

std::int32_t* i32_data(nb::handle a) { return nb::cast<i32_array>(a).data(); }
double* f64_data(nb::handle a) { return nb::cast<f64_array>(a).data(); }
std::uint64_t* u64_data(nb::handle a) { return nb::cast<u64_array>(a).data(); }

nb::object composition(const bc::space& s, const std::int32_t* x) {
    nb::object out = empty(std::nullopt, s.parts(), "int32");
    std::copy(x, x + s.parts(), i32_data(out));
    return out;
}

// ----------------------------------------------------------------- workers

// Thread pools by size, created on first use. A child process after fork()
// has none of the parent's threads, so it starts a new cache and leaks the
// old one rather than joining threads that do not exist.
struct pool_cache {
    std::mutex mutex;
    std::map<unsigned, std::unique_ptr<bc::thread_pool>> pools;
};
pool_cache* pools = new pool_cache;

// workers as in SciPy: 1 runs on the calling thread, -1 on every hardware
// thread.
bc::executor* executor_for(int workers) {
    if (workers == 0 || workers < -1) {
        throw nb::value_error("workers must be a positive number of threads or -1");
    }
    const unsigned threads = workers == -1 ? std::max(1u, std::thread::hardware_concurrency())
                                           : static_cast<unsigned>(workers);
    if (threads == 1) return nullptr;
    std::lock_guard<std::mutex> lock(pools->mutex);
    std::unique_ptr<bc::thread_pool>& pool = pools->pools[threads];
    if (!pool) pool = std::make_unique<bc::thread_pool>(threads);
    return pool.get();
}

// Runs fn(context, block) for blocks [0, blocks) on ex, or inline.
void run_blocks(bc::executor* ex, std::uint64_t blocks, bc::executor::block_fn fn, void* context) {
    if (ex == nullptr) {
        for (std::uint64_t b = 0; b < blocks; ++b) fn(context, b);
    } else {
        ex->run(blocks, fn, context);
    }
}

constexpr std::uint64_t rows_per_block = 64;

std::uint64_t blocks_for(std::size_t rows) {
    return (static_cast<std::uint64_t>(rows) + rows_per_block - 1) / rows_per_block;
}

// ------------------------------------------------------------------ spaces

std::vector<std::int32_t> int32_vector(nb::handle obj, const char* what) {
    nb::object a = int32_array(obj, 1, what);
    const std::int32_t* p = i32_data(a);
    return std::vector<std::int32_t>(p, p + nb::cast<std::size_t>(a.attr("size")));
}

std::string counting_name(bc::counting mode) {
    return mode == bc::counting::exact ? "exact" : "logarithmic";
}

bc::counting counting_mode(const std::string& name) {
    if (name == "exact") return bc::counting::exact;
    if (name == "logarithmic") return bc::counting::logarithmic;
    throw nb::value_error("counting must be 'exact' or 'logarithmic'");
}

std::string group_repr(const bc::group_limit& g) {
    std::string out = "Group(parts=[";
    for (std::size_t i = 0; i < g.parts.size(); ++i) {
        out += (i ? ", " : "") + std::to_string(g.parts[i]);
    }
    out += "]";
    if (g.min_sum != std::numeric_limits<std::int32_t>::min()) {
        out += ", min_sum=" + std::to_string(g.min_sum);
    }
    if (g.max_sum != no_limit) out += ", max_sum=" + std::to_string(g.max_sum);
    return out + ")";
}

void construct_space(bc::space* self, std::int32_t total, nb::handle lower, nb::handle upper,
                     std::int32_t min_count, std::optional<std::int32_t> max_count,
                     nb::handle min_size, const std::vector<bc::group_limit>& groups,
                     const std::string& counting) {
    const std::vector<std::int32_t> lo = int32_vector(lower, "lower");
    const std::vector<std::int32_t> hi = int32_vector(upper, "upper");
    bc::support_limits support;
    support.min_count = min_count;
    support.max_count = max_count.value_or(no_limit);
    if (!min_size.is_none()) support.min_size = int32_vector(min_size, "min_size");
    const bc::counting mode = counting_mode(counting);
    nb::gil_scoped_release release;  // the table build can take seconds
    new (self) bc::space(total, lo, hi, support, groups, mode);
}

// Iterates over a space in rank order with first() and next().
struct space_iterator {
    bc::space s;
    std::vector<std::int32_t> x;
    bool started = false;
    bool done = false;
};

// ----------------------------------------------------------------- filters

// A Python callable as a fresh_sampler filter. It is called with the GIL
// held, from whichever thread checks the point, on a copy of the point.
struct py_filter {
    nb::object fn;
    std::size_t parts = 0;
};

bool call_filter(void* context, const std::int32_t* x) {
    const auto* f = static_cast<const py_filter*>(context);
    nb::gil_scoped_acquire acquire;
    nb::object point = empty(std::nullopt, f->parts, "int32");
    std::copy(x, x + f->parts, i32_data(point));
    return truthy(f->fn(point));
}

// fresh_sampler keeps the filter's context pointer, so the filter lives here
// at a fixed address. The visited set is kept alive by the Python binding.
// One return per branch: MSVC does not elide the copy of a non-copyable
// sampler out of a conditional expression.
bc::fresh_sampler make_sampler(const bc::sequence& source, bc::visited_set& visited, py_filter* f,
                               std::uint64_t patience) {
    if (f != nullptr) return bc::fresh_sampler(source, visited, &call_filter, f, patience);
    return bc::fresh_sampler(source, visited, patience);
}

struct fresh {
    std::size_t parts;
    std::unique_ptr<py_filter> filter;
    bc::fresh_sampler sampler;

    fresh(const bc::sequence& source, bc::visited_set& visited, std::unique_ptr<py_filter> f,
          std::uint64_t patience)
        : parts(source.domain().parts()),
          filter(std::move(f)),
          sampler(make_sampler(source, visited, filter.get(), patience)) {}
};

// The filter can refer back to its sampler (a lambda whose globals hold
// it), so the garbage collector has to see the reference.
int fresh_traverse(PyObject* self, visitproc visit, void* arg) {
    Py_VISIT(Py_TYPE(self));
    if (!nb::inst_ready(self)) return 0;
    const fresh* f = nb::inst_ptr<fresh>(self);
    if (f->filter) Py_VISIT(f->filter->fn.ptr());
    return 0;
}

int fresh_clear(PyObject* self) {
    fresh* f = nb::inst_ptr<fresh>(self);
    if (f->filter) f->filter->fn = nb::none();
    return 0;
}

PyType_Slot fresh_slots[] = {{Py_tp_traverse, reinterpret_cast<void*>(fresh_traverse)},
                             {Py_tp_clear, reinterpret_cast<void*>(fresh_clear)},
                             {0, nullptr}};

// ------------------------------------------------------------- batch jobs

struct decode_job {
    const bc::space* s;
    const double* uf;
    const std::uint64_t* uu;
    std::int32_t* out;
    std::size_t rows;
};

void decode_block(void* context, std::uint64_t block) {
    const auto* j = static_cast<const decode_job*>(context);
    const std::size_t n = j->s->parts();
    const std::size_t d = n - 1;
    const std::size_t end = std::min<std::size_t>(j->rows, (block + 1) * rows_per_block);
    for (std::size_t r = block * rows_per_block; r < end; ++r) {
        if (j->uf != nullptr) {
            bc::decode(*j->s, j->uf + r * d, j->out + r * n);
        } else {
            bc::decode(*j->s, j->uu + r * d, j->out + r * n);
        }
    }
}

struct repair_job {
    const bc::space* s;
    const double* z;
    std::int32_t* out;
    std::size_t rows;
};

void repair_block(void* context, std::uint64_t block) {
    const auto* j = static_cast<const repair_job*>(context);
    const std::size_t n = j->s->parts();
    const std::size_t end = std::min<std::size_t>(j->rows, (block + 1) * rows_per_block);
    for (std::size_t r = block * rows_per_block; r < end; ++r) {
        bc::repair(*j->s, j->z + r * n, j->out + r * n);
    }
}

std::string kind_name(bc::sequence_kind k) {
    switch (k) {
        case bc::sequence_kind::rank:
            return "rank";
        case bc::sequence_kind::scaled_rank:
            return "scaled_rank";
        case bc::sequence_kind::iid:
            return "iid";
        case bc::sequence_kind::coordinate:
            return "coordinate";
    }
    return "unknown";
}

// Calls f for one composition, or for each row of a 2-D array; returns a bool
// or a bool array to match.
template <typename F>
nb::object per_composition(const bc::space& s, nb::handle obj, const char* what, F f) {
    nb::object a = int32_array(obj, 0, what);
    require_cols(a, s.parts(), what);
    const std::int32_t* x = i32_data(a);
    if (nb::cast<int>(a.attr("ndim")) == 1) return nb::bool_(f(x));
    const std::size_t rows = rows_of(a);
    nb::object flags = numpy().attr("zeros")(rows, "dtype"_a = "uint8");
    std::uint8_t* p = nb::cast<nb::ndarray<std::uint8_t, nb::c_contig>>(flags).data();
    for (std::size_t r = 0; r < rows; ++r) p[r] = f(x + r * s.parts()) ? 1 : 0;
    return flags.attr("astype")("bool");
}

// A Python or NumPy integer (a rank), as opposed to an array-like.
bool is_integer_scalar(nb::handle obj) {
    if (nb::isinstance<nb::int_>(obj)) return true;
    return nb::cast<bool>(
        nb::module_::import_("builtins").attr("isinstance")(obj, numpy().attr("integer")));
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "Compiled core of bounded_compositions.";

    m.def(
        "version", []() { return std::string(bc::version()); },
        "Library version as 'major.minor.patch'.");
    m.def(
        "_forget_thread_pools", []() { pools = new pool_cache; },
        "After fork(): drop the parent's thread pools without joining them.");

    // ---------------------------------------------------------------- Group
    nb::class_<bc::group_limit>(m, "Group",
                                "A limit on the sum of some parts: min_sum <= sum of "
                                "x[i] over parts <= max_sum.")
        .def(
            "__init__",
            [](bc::group_limit* self, nb::handle parts, std::optional<std::int32_t> min_sum,
               std::optional<std::int32_t> max_sum) {
                std::vector<std::int32_t> p = int32_vector(parts, "parts");
                new (self) bc::group_limit{
                    std::move(p), min_sum.value_or(std::numeric_limits<std::int32_t>::min()),
                    max_sum.value_or(no_limit)};
            },
            "parts"_a, "min_sum"_a = nb::none(), "max_sum"_a = nb::none())
        .def_prop_ro("parts", [](const bc::group_limit& g) { return g.parts; })
        .def_prop_ro("min_sum",
                     [](const bc::group_limit& g) -> std::optional<std::int32_t> {
                         if (g.min_sum == std::numeric_limits<std::int32_t>::min()) {
                             return std::nullopt;
                         }
                         return g.min_sum;
                     })
        .def_prop_ro("max_sum",
                     [](const bc::group_limit& g) -> std::optional<std::int32_t> {
                         if (g.max_sum == no_limit) return std::nullopt;
                         return g.max_sum;
                     })
        .def("__eq__", [](const bc::group_limit& a, const bc::group_limit& b) { return a == b; })
        .def("__repr__", &group_repr)
        .def("__getstate__",
             [](const bc::group_limit& g) { return nb::make_tuple(g.parts, g.min_sum, g.max_sum); })
        .def("__setstate__", [](bc::group_limit* self, const nb::tuple& t) {
            new (self) bc::group_limit{nb::cast<std::vector<std::int32_t>>(t[0]),
                                       nb::cast<std::int32_t>(t[1]), nb::cast<std::int32_t>(t[2])};
        });

    // ---------------------------------------------------------------- Space
    nb::class_<bc::space>(
        m, "Space",
        "The integer vectors x with x[0] + ... + x[n-1] == total and lower[i] <= x[i] <= "
        "upper[i], within the support and group limits, in lexicographic order (first part "
        "slowest).\n\n"
        "min_count and max_count limit the number of nonzero parts; min_size[i] makes part i "
        "semicontinuous (x[i] == 0 or abs(x[i]) >= min_size[i]); groups limit sums of parts. "
        "counting='logarithmic' stores logarithms of the counts, for spaces too large for "
        "exact tables; it has no size, ranks, regions or rank sequences.")
        .def("__init__", &construct_space, "total"_a, "lower"_a, "upper"_a, nb::kw_only(),
             "min_count"_a = 0, "max_count"_a = nb::none(), "min_size"_a = nb::none(),
             "groups"_a = std::vector<bc::group_limit>(), "counting"_a = "exact")
        .def_prop_ro("parts", &bc::space::parts)
        .def_prop_ro("total", &bc::space::total)
        .def_prop_ro("lower", [](const bc::space& s) { return composition(s, s.lower().data()); })
        .def_prop_ro("upper", [](const bc::space& s) { return composition(s, s.upper().data()); })
        .def_prop_ro("min_count", [](const bc::space& s) { return s.support().min_count; })
        .def_prop_ro("max_count",
                     [](const bc::space& s) -> std::optional<std::int32_t> {
                         if (s.support().max_count == no_limit) return std::nullopt;
                         return s.support().max_count;
                     })
        .def_prop_ro("min_size",
                     [](const bc::space& s) -> std::optional<std::vector<std::int32_t>> {
                         if (s.support().min_size.empty()) return std::nullopt;
                         return s.support().min_size;
                     })
        .def_prop_ro("groups", &bc::space::groups)
        .def_prop_ro("counting",
                     [](const bc::space& s) { return counting_name(s.counting_mode()); })
        .def_prop_ro(
            "size", [](const bc::space& s) { return to_int(s.size()); },
            "Number of compositions (exact counting only).")
        .def_prop_ro("log_size", &bc::space::log_size,
                     "Natural logarithm of the number of compositions, in either mode.")
        .def_prop_ro("empty", &bc::space::empty)
        .def_prop_ro("table_bytes", &bc::space::table_bytes)
        .def(
            "rank",
            [](const bc::space& s, nb::handle x) {
                nb::object a = int32_array(x, 1, "x");
                require_cols(a, s.parts(), "x");
                return to_int(s.rank(i32_data(a)));
            },
            "x"_a, "Rank of composition x. Raises ValueError if x is not in the space.")
        .def(
            "unrank",
            [](const bc::space& s, nb::handle r) {
                const bc::natural rank = to_natural(r, "rank");
                nb::object out = empty(std::nullopt, s.parts(), "int32");
                s.unrank(rank, i32_data(out));
                return out;
            },
            "rank"_a, "The composition with the given rank.")
        .def(
            "contains",
            [](const bc::space& s, nb::handle x) {
                nb::object a = int32_array(x, 0, "x");
                if (cols_of(a) != s.parts()) {
                    if (nb::cast<int>(a.attr("ndim")) == 1) return nb::object(nb::bool_(false));
                    require_cols(a, s.parts(), "x");
                }
                return per_composition(s, a, "x",
                                       [&](const std::int32_t* p) { return s.contains(p); });
            },
            "x"_a, "Whether x (one composition, or each row of a 2-D array) is in the space.")
        .def(
            "first",
            [](const bc::space& s) -> nb::object {
                nb::object out = empty(std::nullopt, s.parts(), "int32");
                if (!s.first(i32_data(out))) return nb::none();
                return out;
            },
            "The first composition, or None for an empty space.")
        .def(
            "next",
            [](const bc::space& s, nb::handle x) -> nb::object {
                nb::object a = int32_array(x, 1, "x");
                require_cols(a, s.parts(), "x");
                if (!s.contains(i32_data(a))) throw nb::value_error("x is not in the space");
                nb::object out = numpy().attr("array")(a);
                if (!s.next(i32_data(out))) return nb::none();
                return out;
            },
            "x"_a, "The composition after x, or None if x is the last.")
        .def(
            "subspace",
            [](const bc::space& s, nb::handle prefix) {
                return s.subspace(int32_vector(prefix, "prefix"));
            },
            "prefix"_a,
            "The remaining parts once the first len(prefix) parts are fixed. Its rank r is rank "
            "subtree(space, prefix).begin + r here.")
        .def("__iter__",
             [](const bc::space& s) {
                 return space_iterator{s, std::vector<std::int32_t>(s.parts()), false, false};
             })
        .def("__repr__",
             [](const bc::space& s) {
                 return "Space(total=" + std::to_string(s.total()) +
                        ", parts=" + std::to_string(s.parts()) + ", counting='" +
                        counting_name(s.counting_mode()) + "')";
             })
        .def("__getstate__",
             [](const bc::space& s) {
                 return nb::make_tuple(s.total(), s.lower(), s.upper(), s.support().min_count,
                                       s.support().max_count, s.support().min_size, s.groups(),
                                       counting_name(s.counting_mode()));
             })
        .def("__setstate__", [](bc::space* self, const nb::tuple& t) {
            bc::support_limits support;
            support.min_count = nb::cast<std::int32_t>(t[3]);
            support.max_count = nb::cast<std::int32_t>(t[4]);
            support.min_size = nb::cast<std::vector<std::int32_t>>(t[5]);
            new (self)
                bc::space(nb::cast<std::int32_t>(t[0]), nb::cast<std::vector<std::int32_t>>(t[1]),
                          nb::cast<std::vector<std::int32_t>>(t[2]), support,
                          nb::cast<std::vector<bc::group_limit>>(t[6]),
                          counting_mode(nb::cast<std::string>(t[7])));
        });

    nb::class_<space_iterator>(m, "_SpaceIterator")
        .def("__iter__", [](nb::handle self) { return nb::borrow(self); })
        .def("__next__", [](space_iterator& it) {
            if (it.done) throw nb::stop_iteration();
            const bool more = it.started ? it.s.next(it.x.data()) : it.s.first(it.x.data());
            it.started = true;
            if (!more) {
                it.done = true;
                throw nb::stop_iteration();
            }
            return composition(it.s, it.x.data());
        });

    // --------------------------------------------------------------- Region
    nb::class_<bc::region>(m, "Region",
                           "An interval of ranks of a space. Subtrees, the compositions that "
                           "share their first parts, are regions.")
        .def_prop_ro("begin", [](const bc::region& r) { return to_int(r.begin()); })
        .def_prop_ro("end", [](const bc::region& r) { return to_int(r.end()); })
        .def_prop_ro("size", [](const bc::region& r) { return to_int(r.size()); })
        .def_prop_ro("empty", &bc::region::empty)
        .def_prop_ro("prefix", &bc::region::prefix,
                     "The longest prefix shared by every composition of the region.")
        .def_prop_ro("is_subtree", &bc::region::is_subtree)
        .def(
            "contains",
            [](const bc::region& r, nb::handle rank) {
                return r.contains(to_natural(rank, "rank"));
            },
            "rank"_a)
        .def("__eq__", [](const bc::region& a, const bc::region& b) { return a == b; })
        .def("__repr__", [](const bc::region& r) {
            return "Region(begin=" + r.begin().to_string() + ", end=" + r.end().to_string() + ")";
        });

    m.def("whole", &bc::whole, "space"_a, "The whole space as a region.");
    m.def(
        "subtree",
        [](const bc::space& s, nb::handle prefix) {
            return bc::subtree(s, int32_vector(prefix, "prefix"));
        },
        "space"_a, "prefix"_a, "The compositions whose first parts equal prefix.");
    m.def(
        "interval",
        [](const bc::space& s, nb::handle begin, nb::handle end) {
            return bc::interval(s, to_natural(begin, "begin"), to_natural(end, "end"));
        },
        "space"_a, "begin"_a, "end"_a, "Ranks [begin, end).");
    m.def("children", &bc::children, "space"_a, "region"_a,
          "The subtrees one level below a subtree, in rank order.");
    m.def("split", &bc::split, "space"_a, "region"_a, "pieces"_a,
          "Consecutive intervals whose sizes differ by at most one.");
    m.def("dyadic_split", &bc::dyadic_split, "space"_a, "region"_a, "levels"_a,
          "Pieces along the grid of the rank order (see the C++ documentation).");
    m.def(
        "describe",
        [](const bc::space& s, const bc::region& r) {
            const std::vector<bc::part_range> ranges = bc::describe(s, r);
            nb::object out = empty(ranges.size(), 2, "int32");
            std::int32_t* p = i32_data(out);
            for (std::size_t i = 0; i < ranges.size(); ++i) {
                p[2 * i] = ranges[i].lo;
                p[2 * i + 1] = ranges[i].hi;
            }
            return out;
        },
        "space"_a, "region"_a,
        "Smallest and largest value of each part in the region, as a (parts, 2) array.");

    // ------------------------------------------------------------- Sequence
    nb::class_<bc::sequence>(
        m, "Sequence",
        "Point i is a pure function of (space, kind, seed, i): taking 1000 points and then 100 "
        "more gives exactly the first 1100, and any number of workers gives the same points.")
        .def_prop_ro("space", &bc::sequence::domain)
        .def_prop_ro("kind", [](const bc::sequence& q) { return kind_name(q.kind()); })
        .def_prop_ro("seed", &bc::sequence::seed)
        .def_prop_ro("capacity", &bc::sequence::capacity,
                     "Number of points the sequence can produce.")
        .def_prop_ro("position", &bc::sequence::position, "Index of the next point take() returns.")
        .def("seek", &bc::sequence::seek, "position"_a)
        .def(
            "rank_at", [](const bc::sequence& q, std::uint64_t i) { return to_int(q.rank_at(i)); },
            "i"_a, "Rank of point i.")
        .def(
            "points",
            [](const bc::sequence& q, std::uint64_t begin, std::uint64_t count, int workers) {
                bc::executor* ex = executor_for(workers);
                nb::object out = empty(count, q.domain().parts(), "int32");
                std::int32_t* p = i32_data(out);
                nb::gil_scoped_release release;
                q.points(begin, count, p, ex);
                return out;
            },
            "begin"_a, "count"_a, nb::kw_only(), "workers"_a = 1,
            "Points [begin, begin + count) as a (count, parts) array.")
        .def(
            "take",
            [](bc::sequence& q, std::uint64_t count, int workers) {
                bc::executor* ex = executor_for(workers);
                nb::object out = empty(count, q.domain().parts(), "int32");
                std::int32_t* p = i32_data(out);
                nb::gil_scoped_release release;
                q.take(count, p, ex);
                return out;
            },
            "count"_a, nb::kw_only(), "workers"_a = 1,
            "The next count points, as a (count, parts) array. Raises IndexError, taking "
            "nothing, if fewer remain.")
        .def("overlap", &bc::sequence::overlap, "outer"_a, "n"_a,
             "How many of the first n points of outer are points of this sequence.")
        .def("resume_after", &bc::sequence::resume_after, "outer"_a,
             "Seeks past the points outer has already taken in this sequence's region.")
        .def("__repr__", [](const bc::sequence& q) {
            return "Sequence(kind='" + kind_name(q.kind()) + "', seed=" + std::to_string(q.seed()) +
                   ", position=" + std::to_string(q.position()) + ")";
        });

    m.def(
        "rank_sequence",
        [](const bc::space& s, const bc::region* r, std::optional<std::uint64_t> seed) {
            if (r == nullptr) return seed ? bc::rank_sequence(s, *seed) : bc::rank_sequence(s);
            return seed ? bc::rank_sequence(s, *r, *seed) : bc::rank_sequence(s, *r);
        },
        "space"_a, "region"_a.none() = nb::none(), nb::kw_only(), "seed"_a = nb::none(),
        "Radical-inverse order on the ranks: no repeats, every prefix spread out. A seed XORs "
        "the rank bits with a mask drawn from it (a scrambled replicate).");
    m.def("scaled_rank_sequence", &bc::scaled_rank_sequence, "space"_a,
          "floor(phi(i) * size), the earlier mapping, kept for comparison.");
    m.def(
        "iid_sequence",
        [](const bc::space& s, std::uint64_t seed, const bc::region* r) {
            return r == nullptr ? bc::iid_sequence(s, seed) : bc::iid_sequence(s, *r, seed);
        },
        "space"_a, "seed"_a, "region"_a.none() = nb::none(),
        "Independent uniform compositions from Philox4x64-10 keyed by seed.");
    m.def(
        "coordinate_sequence",
        [](const bc::space& s, const bc::region* r, std::optional<std::uint64_t> seed) {
            if (r == nullptr) {
                return seed ? bc::coordinate_sequence(s, *seed) : bc::coordinate_sequence(s);
            }
            return seed ? bc::coordinate_sequence(s, *r, *seed) : bc::coordinate_sequence(s, *r);
        },
        "space"_a, "region"_a.none() = nb::none(), nb::kw_only(), "seed"_a = nb::none(),
        "Sobol points decoded part by part; seed scrambles them.");

    // ---------------------------------------------------------------- codec
    m.def(
        "decode",
        [](const bc::space& s, nb::handle u, int workers) {
            if (s.parts() < 2) throw nb::value_error("decode needs at least two parts");
            nb::object a = float_array(u, true, "u");
            require_cols(a, s.parts() - 1, "u");
            const bool one = nb::cast<int>(a.attr("ndim")) == 1;
            const std::size_t rows = rows_of(a);
            const bool fractions = nb::cast<std::string>(a.attr("dtype").attr("name")) == "uint64";
            nb::object out =
                one ? empty(std::nullopt, s.parts(), "int32") : empty(rows, s.parts(), "int32");
            decode_job job{&s, fractions ? nullptr : f64_data(a), fractions ? u64_data(a) : nullptr,
                           i32_data(out), rows};
            bc::executor* ex = executor_for(workers);
            nb::gil_scoped_release release;
            try {
                run_blocks(ex, blocks_for(rows), &decode_block, &job);
            } catch (const std::out_of_range& e) {
                throw nb::value_error(e.what());  // a u outside [0, 1) is a bad value here
            }
            return out;
        },
        "space"_a, "u"_a, nb::kw_only(), "workers"_a = 1,
        "Points of [0, 1)^(parts - 1) (one, or rows of a 2-D array) to compositions. uint64 "
        "input is read as 64-bit fractions u / 2^64.");
    m.def(
        "encode",
        [](const bc::space& s, nb::handle x, nb::handle dtype) {
            if (s.parts() < 2) throw nb::value_error("encode needs at least two parts");
            const std::string name =
                nb::cast<std::string>(numpy().attr("dtype")(dtype).attr("name"));
            if (name != "float64" && name != "uint64") {
                throw nb::value_error("dtype must be float64 or uint64");
            }
            nb::object a = int32_array(x, 0, "x");
            require_cols(a, s.parts(), "x");
            const bool one = nb::cast<int>(a.attr("ndim")) == 1;
            const std::size_t rows = rows_of(a);
            const std::size_t d = s.parts() - 1;
            nb::object out =
                one ? empty(std::nullopt, d, name.c_str()) : empty(rows, d, name.c_str());
            const std::int32_t* p = i32_data(a);
            if (name == "float64") {
                double* o = f64_data(out);
                for (std::size_t r = 0; r < rows; ++r) bc::encode(s, p + r * s.parts(), o + r * d);
            } else {
                std::uint64_t* o = u64_data(out);
                for (std::size_t r = 0; r < rows; ++r) bc::encode(s, p + r * s.parts(), o + r * d);
            }
            return out;
        },
        "space"_a, "x"_a, "dtype"_a = "float64",
        "The centre of each composition's cell in [0, 1)^(parts - 1), so that "
        "decode(encode(x)) == x.");

    // --------------------------------------------------------------- repair
    m.def(
        "repair",
        [](const bc::space& s, nb::handle z, int workers) {
            nb::object a = float_array(z, false, "z");
            require_cols(a, s.parts(), "z");
            const bool one = nb::cast<int>(a.attr("ndim")) == 1;
            const std::size_t rows = rows_of(a);
            nb::object out =
                one ? empty(std::nullopt, s.parts(), "int32") : empty(rows, s.parts(), "int32");
            repair_job job{&s, f64_data(a), i32_data(out), rows};
            bc::executor* ex = executor_for(workers);
            nb::gil_scoped_release release;
            run_blocks(ex, blocks_for(rows), &repair_block, &job);
            return out;
        },
        "space"_a, "z"_a, nb::kw_only(), "workers"_a = 1,
        "The composition nearest to each real vector in squared distance, ties to the first in "
        "rank order. Exact for every limit the space counts.");

    // ----------------------------------------------------------- VisitedSet
    nb::class_<bc::visited_set>(m, "VisitedSet",
                                "Evaluated compositions of one space, stored exactly by rank. "
                                "Safe to share between threads.")
        .def(nb::init<const bc::space&>(), "space"_a)
        .def_prop_ro("space", &bc::visited_set::domain)
        .def(
            "insert",
            [](bc::visited_set& v, nb::handle item) -> nb::object {
                if (is_integer_scalar(item)) return nb::bool_(v.insert(to_natural(item, "rank")));
                return per_composition(v.domain(), item, "x",
                                       [&](const std::int32_t* x) { return v.insert(x); });
            },
            "item"_a,
            "Adds a rank, a composition or each row of a 2-D array; returns whether each was "
            "new.")
        .def(
            "contains",
            [](const bc::visited_set& v, nb::handle item) -> nb::object {
                if (is_integer_scalar(item)) {
                    return nb::bool_(v.contains(to_natural(item, "rank")));
                }
                return per_composition(v.domain(), item, "x",
                                       [&](const std::int32_t* x) { return v.contains(x); });
            },
            "item"_a)
        .def("__len__", [](const bc::visited_set& v) { return v.size(); });

    // --------------------------------------------------------- FreshSampler
    nb::class_<fresh>(m, "FreshSampler", nb::type_slots(fresh_slots),
                      "Points of a sequence's region not yet in a visited set, and accepted by "
                      "an optional filter accept(x) -> bool. Rejected points are counted, not "
                      "marked visited. The filter runs with the GIL held; an exception from it "
                      "propagates, and the points drawn in that batch are skipped.")
        .def(
            "__init__",
            [](fresh* self, const bc::sequence& source, bc::visited_set& visited, nb::object accept,
               std::uint64_t patience) {
                std::unique_ptr<py_filter> f;
                if (!accept.is_none()) {
                    if (!PyCallable_Check(accept.ptr()))
                        throw nb::type_error("accept must be callable");
                    f = std::make_unique<py_filter>(py_filter{accept, source.domain().parts()});
                }
                new (self) fresh(source, visited, std::move(f), patience);
            },
            "source"_a, "visited"_a, "accept"_a = nb::none(), "patience"_a = 1024,
            nb::keep_alive<1, 3>())
        .def(
            "take",
            [](fresh& f, std::uint64_t count, int workers) {
                bc::executor* ex = executor_for(workers);
                nb::object out = empty(count, f.parts, "int32");
                std::int32_t* p = i32_data(out);
                std::uint64_t got = 0;
                {
                    nb::gil_scoped_release release;
                    got = f.sampler.take(count, p, ex);
                }
                nb::object first = nb::module_::import_("builtins").attr("slice")(got);
                return out.attr("__getitem__")(first);
            },
            "count"_a, nb::kw_only(), "workers"_a = 1,
            "Up to count new points; fewer means the region has none left.")
        .def_prop_ro("exhausted", [](const fresh& f) { return f.sampler.exhausted(); })
        .def_prop_ro("switched", [](const fresh& f) { return f.sampler.switched(); })
        .def_prop_ro("skipped", [](const fresh& f) { return f.sampler.skipped(); })
        .def_prop_ro("accepted", [](const fresh& f) { return f.sampler.accepted(); })
        .def_prop_ro("rejected", [](const fresh& f) { return f.sampler.rejected(); });
}
