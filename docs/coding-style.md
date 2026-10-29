# Coding style

This is the repository's coding and review standard. New code and changes
must follow it together with the [architecture](architecture.md). Kernel
code also follows [kernel rules](kernel-rules.md). Existing deviations are
recorded in the [code review](reviews/2026-10-29.md); they remain work to fix.

## Formatting and language baseline

Use C++17. Builds in newer modes check compatibility; they do not authorize
newer features in the library. Keep compiler-specific code behind an explicit
portable interface and retain the MSVC-compatible limb arithmetic. The C++
core must not acquire a third-party runtime dependency.

[`.clang-format`](../.clang-format) is authoritative for authored C++:
four-space indentation, a 100-column limit, left-aligned pointer declarators
and the configured Google-derived layout. The reference formatter used for
the 2026-10-29 review is clang-format 23.1.2. Treat a formatter upgrade and any
resulting broad reformatting as an explicit maintenance change.

Simple single-line guards and loops may omit braces, matching existing code.
Use braces for multi-statement bodies and nested control flow where association
would be unclear. Keep formatting-only changes separate from behavior changes
when practical and avoid reformatting unrelated code.

Python uses four spaces, `snake_case` functions/variables, `PascalCase` public
classes and explicit imports. Separate standard-library, third-party and local
imports. Target 100 columns for new code where practical; URLs, literal messages
and compact fixture data may need more. Use the existing NumPy-style sections
for substantial public adapter docstrings. Python currently has no configured
automatic formatter or linter, so these rules are checked in review.

CMake and Meson follow the existing two-space indentation. Prefer structured
arguments over embedded shell strings. Keep shared source and test lists aligned.

## Names and vocabulary

Use part, composition, space, size, rank, prefix, region, subtree, support,
group, sequence, codec, executor and visited set consistently. Core code uses
generic combinatorial terms; application-specific language belongs in adapter
examples or downstream projects.

| Entity | Convention | Examples |
|---|---|---|
| C++ library types, functions, fields and constants | `snake_case` | `support_limits`, `rank_sequence`, `max_entries` |
| Private C++ class fields | Trailing underscore | `data_`, `cursor_`, `run_mutex_` |
| Macros and include guards | `BCOMP_` prefix and uppercase | `BCOMP_KERNEL`, `BCOMP_SPACE_HPP` |
| Private host types | `detail` namespace or private headers | `detail::space_data` |
| Translation-unit helpers | Anonymous namespace | A builder or validation helper |
| Python public types | `PascalCase` | `Space`, `FreshSampler`, `CompositionEngine` |
| Python functions and parameters | `snake_case` | `suggest_composition`, `min_count` |

Short mathematical names such as `i`, `n`, `k`, `x` and `y` are appropriate
in small scopes when the formula or comment defines them. Use descriptive
names at function, storage and ownership boundaries.

Distinguish original/shifted coordinates, element/byte counts, offsets and
widths, and mathematical rank/table-row widths. Prefer `shifted_total`,
`table_width`, `rank_limbs`, `row_limbs`, `segment_counts` and `active_groups`
for new interfaces. Existing view fields `total`, `table_k`, `seg`, `segs`
and `act` retain their documented meanings until a coordinated rename. Do not
rename a public or kernel-layout field in isolation.

Use named flags, accessors or small plain-data records for tuple fields and
bit meanings reused across modules. Flat storage is compatible with readable
accessors; repeated unexplained indices and numeric masks are not an API.

## Files and interfaces

Public declarations live in `include/bounded_compositions/`; implementation
and private headers live in `src/`. Keep binding conversion/lifecycle code
in `python/src/` and optimizer-specific behavior in Python adapters. Follow
the architecture's dependency direction.

Include a source file's own header first, then standard headers, then project
and private headers in separate groups. Files without a corresponding header,
including bindings, keep external dependency headers in their own group.
Include what the file uses instead of relying on transitive includes. Public
headers use `BCOMP_..._HPP` guards and must be independently includable.

Do not introduce library-defined public templates or any kernel templates.
Private helpers may use templates for a concrete implementation need.
Standard-library containers in host interfaces are allowed.

Split files around responsibilities with useful interfaces, not line count
alone. Shared state-transition semantics need one definition; exact and
logarithmic builders must not diverge in their interpretation of constraints.
When adding sources/tests, update CMake, Meson and reduced-limb builds that
compile the affected implementation. Keep warning exceptions explicit beside
the affected target and avoid propagating repository warnings to consumers.

## Numeric and memory safety

Follow the architecture's representation table. Widen operands before
arithmetic; assigning an overflowed narrow expression to a wider type does
not repair it. Check conversions, offsets and allocation dimensions before
casting or allocating. Saturation may represent an intentional unbounded
value only when it preserves the feasible set.

Counts, sizes and ranks use `natural` or corresponding limb views. Kernel
scratch honors row/output widths and the compile-time limb budget; size
width cannot replace row width. Floating-point conversion and comparison
must meet their rounding and tie contracts. Test boundaries and remove large
candidate-independent offsets from optimization costs before comparing them.

Raw-pointer interfaces document element types, required lengths, layouts,
mutability, aliasing, ownership and lifetimes. A pointer plus size does not
transfer ownership. Prefer existing value/vector host overloads when they
clarify validation; kernel views remain borrowed plain data with caller-owned
scratch and output.

## Errors, resources and concurrency

Validate user inputs on the host before table access. Fallible kernel entry
points return statuses and document preconditions; host code translates failures to public
exceptions. Do not hide an internal invariant failure as an empty result or
another feasible set.

Use RAII for allocations, locks and resources. Partial construction must stop
and join started workers before destroying their shared state. Audit allocations
and called operations before declaring `noexcept`. An intentional termination
policy must be explicit; `noexcept` is not error handling.

Document when outputs, cursors or visited sets may change before an exception.
State whether an operation is transactional or leaves partial results and
reservations. Preserve the distinction between a failed range check and a
failure after successful reservation.

Shared data is immutable or protected by a named synchronization mechanism.
Write points to disjoint output ranges. Explain the invariant protected by a
lock and why an atomic memory order suffices. Relaxed counters do not publish
output. Thread-safety contracts state which operations they cover and restrictions
on seeking, assignment and destruction.

Callbacks document determinism, thread safety, GIL behavior, reentrancy and
context lifetime. They must not recursively use a pool or block on work that
needs that pool. Account for Python's cache of pools by worker count when
reviewing indirect calls from filters.

## Documentation and generated files

Public headers document behavior, preconditions, results, exceptions,
ownership, concurrent use and state changes. Python docstrings also state
shapes, dtypes, counting modes and worker behavior. Keep semantics aligned
across languages and explain deliberate differences.

Explain why transformations, recurrences and state encodings are correct, and
which units/representations values use. Avoid comments that repeat assignments.
Quantitative accuracy or performance claims need a scope and reproducible
evidence. Separate implemented behavior, required invariants, known defects
and future work. Documenting a defect does not waive its contract.

Update these guides and public headers with affected behavior changes. Keep
review findings in a linked, versioned record with reproduction and resolution
status; do not rely on ignored planning files or local paths.

Edit generated artifacts through their sources:

| Artifact | Source or generator |
|---|---|
| `src/sobol_directions.cpp` | `tools/gen_sobol_directions.py`, retaining the checked input SHA-256 and license |
| `tests/data/natural_vectors.inc`, `space_sizes.inc`, `philox_vectors.inc`, `sobol_vectors.inc` | `tools/gen_test_vectors.py` |
| `tests/data/sequence_vectors.inc` | `tools/gen_sequence_vectors.py` using a built Python package |
| `docs/sampling.md` and report plots | `tools/make_sampling_report.py`, `docs/sampling.md.in`, `docs/sampling/records.json` |

Follow each generator's usage instructions. Regenerating expectations from
the implementation under test does not independently validate behavior:
explain the oracle and inspect vector changes. Do not run general source
formatters over generated Sobol data or hand-edit the generated sampling report.

## Checks and review obligations

From the repository root, check authored C++ formatting in a POSIX shell:

```sh
git ls-files -z -- '*.cpp' '*.hpp' ':(exclude)src/sobol_directions.cpp' |
  xargs -0 clang-format --dry-run --Werror
```

CI runs this check with clang-format 23.1.2, the reference version. Python
style, semantic naming, contracts and layering are checked in review.

For behavior changes, run the development build and relevant tests:

```sh
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
python tools/check_kernel.py include/bounded_compositions/kernel
```

Build/install the Python package with its test extra before testing bindings
or adapters, using Python 3.10 or later:

```sh
python -m pip install ".[test]"
python -m pytest
```

Use sanitizer presets for memory/concurrency changes and Meson when changing
shared sources or build behavior, with `-Dtests=enabled` so that a missing
Catch2 fails the setup instead of leaving no tests. Select further checks
for the invariant changed rather than repeating unrelated suites. Documentation
edits need link, whitespace and contract checks, without rebuilding unchanged code.

Test observable contracts with small brute-force or independent references
where possible. Fixed defects need regressions, especially for numeric limits,
domain mismatches, partial construction and failure after reservation. Check
deterministic indexed results across worker counts. Avoid tests that only
mirror the implementation or assert documentation wording.

Reviews check the affected invariant, public contract, numeric widths,
lifetimes/error paths, names, tests, build-system lists and documentation or
generated sources where applicable. An intentional architecture change updates
the guides, public contract and relevant tests together. Record unresolved
violations with concrete impact and reproduction steps.
