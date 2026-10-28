# bounded-compositions

Count, rank, enumerate and sample bounded integer compositions:

    { x in Z^n : x[0] + ... + x[n-1] = K,  lo[i] <= x[i] <= hi[i] }

optionally with limits on the number of nonzero parts, semicontinuous parts
(each nonzero `x[i]` at least `s[i]` in size) and the sums of groups of parts,
in either counting mode.

The goal is sampling that can be extended in place (the first N points are
well spread for every N, and asking for more continues the same sequence), and
that splits into regions that can be sampled independently.
[docs/sampling.md](docs/sampling.md) explains the methods and measures them.

**Status: pre-alpha. Nothing usable yet.**

## Design

- C++17, no third-party dependencies.
- A compiled host library, plus a small header-only kernel layer for the
  per-point work, kept compatible with GPU compilation
  ([docs/kernel-rules.md](docs/kernel-rules.md)).
- Thread-safe: a sequence's data is immutable, and point `i` depends only on
  the space, the seed and `i`. The set of evaluated points is shared through
  locks.
- No repeats: a region lists its compositions in the space's order, so it
  resumes where earlier sampling stopped; a set of evaluated points covers
  other orders and sequence kinds.
- Limits the table does not count (general linear inequalities, say) go
  through a filter that rejects points and counts the acceptance per region.
  `repair` maps any real vector to the nearest composition of a space.
- Exact multiprecision counting, or logarithmic counting for spaces too large
  for exact tables.

## Build

```sh
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

Presets: `dev`, `release`, `asan-ubsan`, `tsan`, `cxx20`, `cxx23`. Tests use
Catch2 v3; it is fetched automatically if not installed.

The Python package `bounded_compositions` (nanobind, NumPy in and out) builds
with pip:

```sh
python -m pip install ".[test]"
python -m pytest
```

`bounded_compositions.scipy` maps compositions to the unit cube for SciPy: a
`QMCEngine` whose samples decode to a sequence's compositions, and
`box_objective`, which hands any box-bounded optimiser a function of
compositions:

```python
import bounded_compositions as bc
from bounded_compositions.scipy import CompositionEngine, box_objective, decode_box
from scipy.optimize import differential_evolution

def f(x):  # any function of a composition
    return float(((x - [30, 30, 0, 20, 20, 0, 0, 0, 0, 0]) ** 2).sum())

space = bc.Space(100, [0] * 10, [40] * 10, min_count=3, max_count=6)
objective, bounds = box_objective(space, f)
init = CompositionEngine(space, rng=1).random(64)
res = differential_evolution(objective, bounds, init=init)
best = decode_box(space, res.x)
```

Meson consumers can use the `bounded-compositions` dependency (`bounded_compositions_dep`).

## License

BSD-3-Clause. The Sobol direction numbers in `src/sobol_directions.cpp` are
Joe and Kuo's, under their own BSD-style licence:
[licenses/sobol-direction-numbers.txt](licenses/sobol-direction-numbers.txt).
