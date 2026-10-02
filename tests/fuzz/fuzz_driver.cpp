// Runs the fuzz properties on seeded random inputs, for compilers without
// libFuzzer. Usage: bounded_compositions_fuzz_smoke [iterations]
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size);

int main(int argc, char** argv) {
    const long iterations = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 20000;
    std::mt19937_64 rng(20261002);
    std::vector<std::uint8_t> input;
    for (long i = 0; i < iterations; ++i) {
        input.resize(rng() % 256);
        for (auto& b : input) b = static_cast<std::uint8_t>(rng());
        LLVMFuzzerTestOneInput(input.data(), input.size());
    }
    std::printf("fuzz smoke: %ld inputs passed\n", iterations);
    return 0;
}
