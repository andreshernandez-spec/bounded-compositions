#include "bounded_compositions/executor.hpp"

namespace bounded_compositions {

void serial_executor::run(std::uint64_t blocks, block_fn fn, void* context) {
    for (std::uint64_t b = 0; b < blocks; ++b) fn(context, b);
}

thread_pool::thread_pool(unsigned threads) {
    if (threads == 0) threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 1;
    workers_.reserve(threads);
    try {
        for (unsigned i = 0; i < threads; ++i) workers_.emplace_back([this] { work(); });
    } catch (...) {
        // No destructor runs for a constructor that throws, and destroying a
        // joinable thread terminates: stop the workers already started first.
        stop();
        throw;
    }
}

thread_pool::~thread_pool() { stop(); }

void thread_pool::stop() noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    start_cv_.notify_all();
    for (std::thread& t : workers_) t.join();
}

void thread_pool::run(std::uint64_t blocks, block_fn fn, void* context) {
    if (blocks == 0) return;
    std::lock_guard<std::mutex> serial(run_mutex_);
    std::unique_lock<std::mutex> lock(mutex_);
    fn_ = fn;
    context_ = context;
    blocks_ = blocks;
    error_ = nullptr;
    next_.store(0, std::memory_order_relaxed);
    failed_.store(false, std::memory_order_relaxed);
    running_ = static_cast<unsigned>(workers_.size());
    ++generation_;
    start_cv_.notify_all();
    // Workers finish under mutex_, so everything they wrote happens before
    // this wait returns.
    done_cv_.wait(lock, [this] { return running_ == 0; });
    if (error_) std::rethrow_exception(error_);
}

void thread_pool::work() {
    std::uint64_t seen = 0;
    for (;;) {
        block_fn fn;
        void* context;
        std::uint64_t blocks;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            start_cv_.wait(lock, [&] { return stopping_ || generation_ != seen; });
            if (stopping_) return;
            seen = generation_;
            fn = fn_;
            context = context_;
            blocks = blocks_;
        }
        for (;;) {
            if (failed_.load(std::memory_order_relaxed)) break;
            const std::uint64_t b = next_.fetch_add(1, std::memory_order_relaxed);
            if (b >= blocks) break;
            try {
                fn(context, b);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!error_) error_ = std::current_exception();
                failed_.store(true, std::memory_order_relaxed);
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (--running_ == 0) done_cv_.notify_one();
    }
}

}  // namespace bounded_compositions
