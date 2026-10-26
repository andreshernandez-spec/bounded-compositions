#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "bounded_compositions/executor.hpp"

using bounded_compositions::executor;
using bounded_compositions::serial_executor;
using bounded_compositions::thread_pool;

namespace {

struct counting_job {
    std::vector<std::atomic<int>> calls;
    explicit counting_job(std::size_t n) : calls(n) {}
};

void count_block(void* context, std::uint64_t block) {
    static_cast<counting_job*>(context)->calls[block].fetch_add(1);
}

void throw_on_block_5(void* context, std::uint64_t block) {
    static_cast<counting_job*>(context)->calls[block].fetch_add(1);
    if (block == 5) throw std::runtime_error("block 5");
}

}  // namespace

TEST_CASE("every block runs exactly once") {
    serial_executor serial;
    thread_pool one(1), many(8);
    for (executor* ex : {static_cast<executor*>(&serial), static_cast<executor*>(&one),
                         static_cast<executor*>(&many)}) {
        for (std::size_t blocks :
             {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{10000}}) {
            counting_job job(blocks);
            ex->run(blocks, &count_block, &job);
            for (const auto& c : job.calls) REQUIRE(c.load() == 1);
        }
    }
}

TEST_CASE("an exception stops the run, is rethrown, and the pool stays usable") {
    thread_pool pool(4);
    counting_job job(1000);
    CHECK_THROWS_AS(pool.run(1000, &throw_on_block_5, &job), std::runtime_error);
    CHECK(job.calls[5].load() == 1);
    counting_job again(1000);
    pool.run(1000, &count_block, &again);
    for (const auto& c : again.calls) CHECK(c.load() == 1);
}

TEST_CASE("a pool survives many short runs") {
    thread_pool pool(6);
    for (int r = 0; r < 500; ++r) {
        counting_job job(17);
        pool.run(17, &count_block, &job);
        for (const auto& c : job.calls) REQUIRE(c.load() == 1);
    }
    CHECK(pool.threads() == 6);
    CHECK(thread_pool(0).threads() >= 1);
}
