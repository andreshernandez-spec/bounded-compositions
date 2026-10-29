// Thread start failures in a pool, injected by defining pthread_create here:
// the executable's definition comes before libc's, so std::thread reaches it.
// Linux (glibc) only, and not under sanitizers, which intercept it themselves.
#include <catch2/catch_test_macros.hpp>

#include <dlfcn.h>
#include <pthread.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <system_error>

#include "bounded_compositions/executor.hpp"

namespace {

// Calls to let through before the next one fails; negative: never fail.
std::atomic<int> starts_left{-1};

void add_one(void* context, std::uint64_t) {
    static_cast<std::atomic<int>*>(context)->fetch_add(1);
}

}  // namespace

extern "C" int pthread_create(pthread_t* thread, const pthread_attr_t* attr, void* (*start)(void*),
                              void* arg) {
    using create_fn = int (*)(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
    static const create_fn real = reinterpret_cast<create_fn>(dlsym(RTLD_NEXT, "pthread_create"));
    int left = starts_left.load();
    while (left >= 0 && !starts_left.compare_exchange_weak(left, left - 1)) {
    }
    if (left == 0) return EAGAIN;
    return real(thread, attr, start, arg);
}

TEST_CASE("a pool whose later thread fails to start stops the others and throws") {
    for (int started : {0, 1, 3}) {
        starts_left = started;
        CHECK_THROWS_AS(bounded_compositions::thread_pool(4), std::system_error);
    }
    starts_left = -1;
    bounded_compositions::thread_pool pool(4);
    std::atomic<int> calls{0};
    pool.run(100, &add_one, &calls);
    CHECK(calls == 100);
}
