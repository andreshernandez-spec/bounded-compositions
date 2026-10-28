// The ranks in data/sequence_vectors.inc come from the Python bindings
// (tools/gen_sequence_vectors.py); C++ must give the same point for the same
// space, kind, seed and region.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "bounded_compositions/bounded_compositions.hpp"

namespace bc = bounded_compositions;
using composition = std::vector<std::int32_t>;

namespace {

struct vector_case {
    int space;
    const char* kind;
    std::uint64_t seed;
    bool has_seed;
    int prefix_length;
    std::int32_t prefix[2];
    std::uint64_t index;
    const char* rank;
};

const vector_case cases[] = {
#include "data/sequence_vectors.inc"
};

// The spaces of tools/gen_sequence_vectors.py, by id.
bc::space space_by_id(int id) {
    if (id == 0) return bc::space(100, composition(100, 0), composition(100, 100));
    if (id == 1) {
        return bc::space(5, composition{-3, 0, -1, 0, -2, 0, 0, -1},
                         composition{4, 6, 3, 5, 2, 7, 4, 3});
    }
    bc::support_limits support;
    support.min_count = 3;
    support.max_count = 8;
    support.min_size.assign(12, 2);
    bc::group_limit a{{0, 1, 2}, 5, 15};
    bc::group_limit b{{6, 7, 8, 9, 10, 11}, 10, 30};
    bc::group_limit c;
    c.parts = {2, 3, 8};
    c.max_sum = 12;
    return bc::space(40, composition(12, 0), composition(12, 10), support, {a, b, c});
}

bc::sequence make(const bc::space& s, const vector_case& v) {
    const std::string kind = v.kind;
    if (v.prefix_length == 0) {
        if (kind == "rank") return v.has_seed ? bc::rank_sequence(s, v.seed) : bc::rank_sequence(s);
        if (kind == "scaled_rank") return bc::scaled_rank_sequence(s);
        if (kind == "iid") return bc::iid_sequence(s, v.seed);
        return v.has_seed ? bc::coordinate_sequence(s, v.seed) : bc::coordinate_sequence(s);
    }
    const bc::region r = bc::subtree(s, composition(v.prefix, v.prefix + v.prefix_length));
    if (kind == "rank") {
        return v.has_seed ? bc::rank_sequence(s, r, v.seed) : bc::rank_sequence(s, r);
    }
    if (kind == "iid") return bc::iid_sequence(s, r, v.seed);
    return v.has_seed ? bc::coordinate_sequence(s, r, v.seed) : bc::coordinate_sequence(s, r);
}

}  // namespace

TEST_CASE("C++ gives the points the Python bindings gave") {
    const bc::space spaces[] = {space_by_id(0), space_by_id(1), space_by_id(2)};
    int checked = 0;
    for (const vector_case& v : cases) {
        INFO("space " << v.space << ", " << v.kind << ", seed " << v.seed << ", index " << v.index);
        const bc::sequence q = make(spaces[v.space], v);
        CHECK(q.rank_at(v.index).to_string() == v.rank);
        ++checked;
    }
    CHECK(checked > 400);
}
