# Architecture

This document defines the architecture of the implemented library. Changes
must preserve its invariants or revise the affected contract, documentation
and tests together. The [coding style](coding-style.md) governs how code is
written; [kernel rules](kernel-rules.md) add restrictions for device-compatible
headers. The [sampling report](sampling.md) explains algorithms and measurements.

This is a pre-alpha implementation. The [code review](reviews/2026-10-29.md)
records where the code does not yet meet these contracts. An invariant is a
requirement, not evidence that every implementation path already satisfies it.

## Layers and dependencies

| Layer | Location | Responsibility |
|---|---|---|
| Kernel | `include/bounded_compositions/kernel/` | Per-point arithmetic, table walks, random bits and sequence indices on borrowed views. |
| Host | `src/`, declared by `include/bounded_compositions/*.hpp` | Validation, constraint planning, table construction, ownership, value types, sequences, regions, repair and executors. |
| Python bindings | `python/src/core.cpp`, `python/bounded_compositions/__init__.py` | Conversion, array shapes/dtypes, Python lifetimes, GIL management and batch dispatch. |
| Adapters | `python/bounded_compositions/scipy.py`, `optuna.py`, `package/` | Integration with external sampling and optimization interfaces. |

Dependencies flow from adapters to bindings to the host, and from the host to
the kernel. Kernel headers must not depend on host types, Python, threading
or external libraries. Host code must not depend on Python or an optimizer.
The C++ core uses C++17 and the standard library; test and Python build
dependencies are separate.

The kernel layer is header-only. The host is compiled, with private headers
under `src/`. Public headers expose non-template library types; standard-library
types in their interfaces do not make the library header-only. Private host
and binding helpers may use templates.

GPU execution, a C ABI and a library-owned optimizer are future work. Device
compatibility of kernel headers does not establish that a GPU backend exists.
Host-side codec, repair and batch preparation can allocate; the allocation
restriction applies to the kernel layer.

## Vocabulary and domain identity

| Term | Meaning |
|---|---|
| Part | A coordinate of an integer vector. Parts may be negative. |
| Composition | A vector within the bounds and counted limits whose parts sum to the total. |
| Space | That feasible set, ordered lexicographically, first part slowest. |
| Size | The number of compositions, represented by `natural` in exact mode. |
| Rank | A zero-based position in the space's lexicographic order. |
| Prefix | Fixed values for the first parts. |
| Subspace | The remaining parts after fixing a prefix, sharing the parent's table. |
| Region | A half-open rank interval `[begin, end)` in a particular space. |
| Subtree | A region containing exactly the compositions that share a prefix. |
| Sequence | An indexed mapping from immutable configuration to compositions, with a separate cursor. |
| Support | The parts whose values are nonzero. |

Ranks and cached prefixes have meaning relative to a space. Treat a region
as belonging to the space that created it, including copies of that space;
do not transfer its cached metadata to another space. The current `region`
value does not retain domain identity. Consumers must validate compatibility
before using metadata; incomplete checks are review R1. Region equality
currently compares endpoints only and is not a domain-compatibility check.

The current `detail::same_space` check compares total, ordered bounds, support
limits and ordered group descriptions. It does not prove equivalence of
differently expressed constraints. Operations using exact ranks also require
exact counting. Changes to compatibility must preserve agreement among
regions, sequences and visited sets.

## Construction and representations

`space` validates bounds and limits on the host, shifts lower bounds to zero,
plans constraint states, builds a table and publishes immutable data:

```text
y[i] = x[i] - lower[i]
cap[i] = upper[i] - lower[i]
shifted_total = original_total - sum(lower)
sum(y) = shifted_total, with 0 <= y[i] <= cap[i]
```

These representations must remain distinct:

| Quantity | Current representation and rule |
|---|---|
| Parts, bounds, public total and group limits | `int32_t`; aggregate calculations need wider intermediates. |
| Bound sums and shifts | `int64_t` during construction; check before narrowing to dimensions or metadata. |
| Shifted total | `space_view::total`, stored as `int32_t`; differs from `space::total()`. |
| Counts and ranks | `natural`, little-endian 64-bit limbs; zero has no significant limbs. |
| Table row width | `row_limbs[row]`, sufficient for the largest prefix sum across that row's layers and totals. |
| Rank storage width | Sufficient for the domain size; it may be smaller than a table row's width. |
| Table width | `table_k`, the parent table's shifted total plus one, including in a subspace view. |
| Sequence indices and cursors | `uint64_t`; capacity remains bounded when the space has more compositions. |
| Cube coordinates | Doubles in `[0, 1)` or integers representing fractions `u / 2^64`. |

Widen operands before arithmetic, then check before narrowing, multiplying
allocation dimensions or computing offsets. Saturation must not change a
feasible set; reject an unsupported representation explicitly. Subspace
metadata must reconstruct the same set as its table, including after Python
pickling. Review R3 and R8 identify violations.

Exact tables store prefix sums of suffix counts in flat row-major limb arrays:

```text
N[row][k][layer] = ways to complete parts row..n-1 with k left in state layer
P[row][layer][k] = sum(N[row][j][layer] for j <= k)
```

Each row has its own width and offset. Kernel output and temporary scratch
must honor the declared row/output widths; the mathematical rank's width
alone is not always enough for its calculation. `BCOMP_MAX_LIMBS` bounds
kernel stack scratch, defaulting to 64 limbs. The compiled host and kernel
consumers must use the same setting.

Support limits split allowed values into at most three segments: negative,
zero and positive, with semicontinuous gaps removed. Layers encode the nonzero
count and sums of groups spanning the row. A group coordinate may store its
running sum plus the remaining total, so every value in a segment can lead
to the same next layer and counts remain prefix-sum windows. The layout is
defined beside [space_view](../include/bounded_compositions/kernel/views.hpp);
builders and walkers must share that interpretation.

Logarithmic tables store both prefix and suffix log sums, evaluating a window
from the better-conditioned side. A separate `positive` array records nonzero
counts exactly; rounded logs cannot determine feasibility reliably. Subspace
views retain the parent's storage strides and offsets while shifting their
starting row and constraint state.

Construction currently limits tables to `2^28` entries and rejects rows beyond
the limb budget. Repair's state search has a separate `2^27`-state limit.
Smaller requests can still exhaust memory: entry widths, logarithmic arrays
and temporary storage also cost memory.

## Counting modes and numerical contracts

| Operation | Exact | Logarithmic |
|---|---|---|
| `contains`, `first`, `next`, `subspace` | Supported | Supported using exact feasibility flags |
| `size`, `rank`, `unrank` | Exact integer results | Unsupported; throws |
| `log_size` | Computed from exact size | Computed from log counts |
| Regions, rank and scaled-rank sequences | Supported | Unsupported; throws |
| IID sequence | Uniform rank draws | Conditional draws using approximate log probabilities |
| Coordinate sequence | Sobol fractions decoded with exact counts | Sobol fractions decoded with log probabilities |
| Codec | Exact counts, finite coordinate resolution | Approximate counts and finite coordinate resolution |
| Repair | Search over counted states with floating-point costs | Same optimization over counted states |
| Visited set and fresh sampler | Supported | Unsupported; requires exact ranks |

Logarithmic probabilities are approximate, especially near conditional cell
boundaries. Unscrambled Sobol points can land exactly on such boundaries;
prefer scrambled coordinate sequences with logarithmic spaces. Agreement
with exact decoding near a boundary is not guaranteed.

The codec targets `decode(encode(x)) == x` whenever an input code exists for
the composition. A narrow cell may contain a grid point even when its center
cannot be represented; width alone does not establish unreachability. Encoding
then returns a code inside the cell. Python encode/decode require at
least two parts; the C++ pointer interface permits one part with zero cube
coordinates.

Repair minimizes squared distance, with ties resolved by the first composition
in rank order. It compares costs in doubles after subtracting the median of z
and taking each part's cost relative to its value nearest its target, which
change every composition's distance by the same amount; `repair.hpp` states
the accuracy. Redundant counted limits do not change its answer when those
costs are exact. `natural` converts to the nearest double, ties to even.

## Ownership and lifetimes

| Object | Owned or shared data | Borrowed data |
|---|---|---|
| `space` | `shared_ptr<const space_data>`; copies share tables | Its view borrows storage from that ownership chain |
| Subspace | Parent reference and remaining-part metadata | Parent tables and planning arrays |
| `sequence` | Immutable shared configuration and its own atomic cursor | Executor/output for a call |
| `region` | Endpoints, cached prefix and subtree flag | Domain association is a caller obligation today |
| `visited_set` | A space copy and a sharded exact set of ranks | Input composition buffers |
| `fresh_sampler` | Copies of source and fallback sequences | Visited set, filter context, executor/output |
| `thread_pool` | Workers, synchronization and current job state | Callback/context until `run` finishes |

Sequence copies share configuration and copy the current cursor value; their
cursors then advance independently. A fresh sampler does not advance its
caller's source sequence. Its visited set and filter context must outlive it.
Python bindings retain the visited set and own Python filter objects.

Views contain pointers, sizes and strides and remain trivially copyable. They
do not own memory. Copying a view to a device does not copy its pointees or
make host pointers accessible: a future backend must arrange accessible
storage and rebind pointers.

## Sampling and concurrency

An indexed point depends on domain, region, kind, scrambling mode, seed and
index. Preserve all factory options for reproducibility: supplying seed zero
requests scrambling, while omitting a seed requests an unscrambled rank or
coordinate sequence. `sequence::seed() == 0` alone cannot distinguish them.

Rank sequences visit distinct ranks up to capacity and restrict the domain's
order to a region. Above 64-bit region sizes they cover a bounded subset;
cursor exhaustion does not enumerate the full region. Scaled-rank, coordinate
and IID sequences may repeat. Fresh samplers claim compositions through a
visited set. After enough consecutive misses within a call, a repeating source
switches to the region's rank sequence. Filters are deterministic for a given
composition; rejected points are not marked visited.

- Space data and sequence configuration remain immutable during use. Indexed
  reads use separate outputs or disjoint output ranges.
- Concurrent `sequence::take` calls reserve disjoint index ranges atomically.
  Seeking, assignment and destruction require coordination with callers;
  rewinding deliberately permits repeated indices.
- Visited-set claims lock individual shards. Concurrent fresh calls prevent
  duplicate claims, but scheduling can change which caller receives a point.
- Executors return after callbacks finish. Pool `run` calls serialize. A
  callback must not recursively use the same pool or wait for work needing
  that pool. Output publication uses executor synchronization; relaxed cursor
  and progress atomics are not output-publication barriers.
- Filters must be safe on worker threads and concurrent calls. Python filters
  acquire the GIL. Python pools are cached by worker count, so recursively
  requesting the same cached pool can also deadlock indirectly.

C++ work starts threads through a supplied executor. Python's `workers=1`
uses the caller, positive counts use cached pools, and `-1` selects hardware
concurrency. After `fork`, the child discards inherited cached pools and
creates new ones when needed.

## Errors and state changes

Validate user input on the host before accessing tables. Kernel calls have
explicit preconditions and return statuses. Host APIs translate invalid values,
ranges, unsupported modes, representation limits and internal invariant failures
to documented exceptions. Bindings also validate shapes and dtypes.

Stateful operations must document their commit point. Current behavior:

- A range error in `sequence::take` reserves nothing. After reservation, a
  generation/executor failure does not roll the cursor back; output may be partial.
- Fresh sampling can consume indices and claim ranks before a later failure.
  Filter exceptions propagate; consumed batches are not automatically replayed.
  There is no transaction across source, set and output.
- A pool waits for workers and rethrows a recorded callback exception. Workers
  already executing or racing with failure detection may finish more work.
- SciPy's `CompositionEngine.random` encodes before moving its cursor, so a
  draw that raises leaves the engine where it was.

Construction must clean up partially acquired resources. Pool construction
currently violates this (R2). `noexcept` must match the operations performed:
constrained `space::next` currently allocates within `noexcept`, so an
allocation failure would terminate.

## Verification and conformance

| Invariant | Required verification |
|---|---|
| Count, rank, unrank and enumeration agree | Brute-force spaces, inverse properties, integer boundaries |
| Subspace metadata describes its shared table | Prefix comparisons, reconstruction and pickle round trips |
| Regions partition and samplers stay inside them | Interval/subtree properties, domain mismatch rejection |
| Results do not depend on worker count | Serial/pool comparisons, concurrent cursor reservations |
| Claims do not repeat | Shared visited-set and fresh-sampler stress cases |
| Codec and repair satisfy numerical contracts | Representable boundary cells, rounding ties, redundant limits, large offsets |
| Resources survive failures | Worker/allocation failure injection and throwing callbacks |
| Kernel rules hold | Checker, standalone builds, trivially-copyable assertions and review |
| Both build systems exercise the code | Aligned sources/tests and nonempty test registration |

Existing CMake tests, sanitizer presets, Python tests and fuzz properties
cover much of this. Meson can currently skip every test if Catch2 is absent
and omits kernel guards (R9). Fuzzer coverage reaches the harness but omits
the compiled host (R10). Formatting has no CI gate yet (R11).

Known runtime deviations at reviewed commit `44b026a` are R1 region validation,
R2 pool construction, R3 subspace metadata, R4 repair precision, R5 codec
reachability, R6 SciPy state, R7 natural rounding and R8 range arithmetic.
The linked review retains reproductions and status. Fixes update that status
and regression coverage; documenting a defect does not waive its invariant.
