# Kernel rules

Code under `include/bounded_compositions/kernel/` computes one point at a time: limb
arithmetic, walking the counting table, sequence indices, random bits. It is
meant to compile unchanged as GPU device code later (CUDA, HIP or SYCL), and
it is shared by every host thread. So it follows stricter rules than the rest
of the library.

These rules specialize the repository's [architecture](architecture.md) and
[coding style](coding-style.md). They apply to kernel headers regardless of
which build system or caller uses them.

## Rules

1. Every function is declared `BCOMP_KERNEL`. Today that expands to `inline`; a
   GPU build defines it as `__host__ __device__ inline` before including any
   kernel header.
2. No allocation, no exceptions, no RTTI, no virtual functions, no
   `thread_local`, no non-constant `static`. Errors are returned as
   `kernel::status`.
3. No recursion. Device stacks are small and fixed; every walk is a loop.
4. Includes: `<cstdint>`, `<cstddef>`, `<climits>`, and other kernel headers.
   The only `std::` names allowed are the fixed-width integer types,
   `size_t`, `ptrdiff_t`, `intptr_t` and `uintptr_t`. The one exception is
   `config.hpp`, which includes `<cmath>` and wraps `std::exp`, `std::log`,
   `std::log1p` and `std::expm1` in the macros `BCOMP_EXP`, `BCOMP_LOG`,
   `BCOMP_LOG1P` and `BCOMP_EXPM1` for the logarithmic counting mode. Other
   kernel headers use the macros, which a GPU build can redefine.
5. Inputs and outputs are views: structs of pointers, sizes and strides that
   are trivially copyable. Copying a view does not copy its pointees or make
   host pointers valid on a device: a future device backend must arrange
   accessible storage and rebind the pointers.
6. Scratch space has a compile-time bound, `kernel::max_limbs` (set by
   `BCOMP_MAX_LIMBS`, default 64 limbs, so 4096 bits). The host rejects any
   space that would need more. The compiled host and its kernel consumers
   must use the same limb budget.
7. No templates.

## How they are checked

- `tools/check_kernel.py` scans the headers for rules 2 and 4 and runs as a
  CTest test (`kernel_rules`).
- The CMake build compiles each kernel header alone with exceptions and RTTI
  turned off (`bounded_compositions_kernel_standalone`).
- Host code checks rule 5 with `static_assert(std::is_trivially_copyable_v<...>)`.
- Rules 1, 3 and 7 are checked in review.

The script is registered with CTest when a Python interpreter is found; it
can also be run directly as described in the coding-style guide. Meson runs
it as a test too, when Python is found, and compiles each kernel header alone
the same way. A CI job also runs the script on its own.
