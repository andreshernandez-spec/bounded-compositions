// Executors run blocks of independent work, possibly on several threads.
#ifndef BCOMP_EXECUTOR_HPP
#define BCOMP_EXECUTOR_HPP

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace bounded_compositions {

class executor {
  public:
    using block_fn = void (*)(void* context, std::uint64_t block);

    executor() = default;
    executor(const executor&) = delete;
    executor& operator=(const executor&) = delete;
    virtual ~executor() = default;

    // Calls fn(context, b) once for every b in [0, blocks), possibly from
    // several threads at once, and returns when all calls have finished. If a
    // call throws, no further blocks start and the first exception is
    // rethrown here.
    virtual void run(std::uint64_t blocks, block_fn fn, void* context) = 0;
};

// Runs every block on the calling thread, in order.
class serial_executor final : public executor {
  public:
    void run(std::uint64_t blocks, block_fn fn, void* context) override;
};

// A fixed set of worker threads that claim blocks from a shared atomic
// counter, so the load balances itself. Calls to run() from different threads
// are serialised.
class thread_pool final : public executor {
  public:
    // threads == 0 means std::thread::hardware_concurrency() (at least 1).
    // If a thread fails to start, the ones already started are stopped and
    // joined, and the std::system_error is rethrown.
    explicit thread_pool(unsigned threads = 0);
    ~thread_pool() override;

    unsigned threads() const noexcept { return static_cast<unsigned>(workers_.size()); }
    void run(std::uint64_t blocks, block_fn fn, void* context) override;

  private:
    void work();
    void stop() noexcept;

    std::vector<std::thread> workers_;
    std::mutex run_mutex_;  // one run() at a time
    std::mutex mutex_;      // guards the job fields below
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::uint64_t generation_ = 0;
    bool stopping_ = false;
    block_fn fn_ = nullptr;
    void* context_ = nullptr;
    std::uint64_t blocks_ = 0;
    unsigned running_ = 0;
    std::exception_ptr error_;
    std::atomic<std::uint64_t> next_{0};
    std::atomic<bool> failed_{false};
};

}  // namespace bounded_compositions

#endif
