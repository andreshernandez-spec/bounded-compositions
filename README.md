# bounded-compositions

Count, rank, enumerate and sample bounded integer compositions:

    { x in Z^n : x[0] + ... + x[n-1] = K,  lo[i] <= x[i] <= hi[i] }

optionally with limits on the number of nonzero parts, semicontinuous parts
(`x[i] == 0` or `abs(x[i]) >= s[i]`), the sums of groups of parts, and gaps
(values a part may not take), in either counting mode.

The goal is sampling that can be extended in place (the first N points are
well spread for every N, and asking for more continues the same sequence), and
that splits into regions that can be sampled independently.
[docs/sampling.md](docs/sampling.md) explains the methods and measures them.

**Status: pre-alpha.** Counting, ranking, enumeration, sampling, repair and
Python adapters are implemented. APIs and numerical contracts are still being
reviewed; the [code review](docs/reviews/2026-10-29.md) records known defects.

## Design

- C++17 core, no third-party dependencies. Tests and Python bindings have
  separate dependencies.
- A compiled host library, plus a small header-only kernel layer for the
  per-point work, kept compatible with GPU compilation
  ([docs/kernel-rules.md](docs/kernel-rules.md)).
- A sequence shares immutable data. Its point `i` is determined by its space,
  region, kind, scrambling options, seed and `i`; its cursor reserves index
  ranges atomically. Evaluated points are shared through a locked set.
- Rank sequences visit distinct compositions up to their capacity and preserve
  their order across regions. `fresh_sampler` uses a visited set to remove
  repeats across other orders, seeds and sequence kinds.
- Limits the table does not count (general linear inequalities, say) go
  through a filter that rejects points and counts the acceptance per region.
  `repair` maps any real vector to the nearest composition of a space.
- Exact multiprecision counting, or logarithmic counting for spaces too large
  for exact tables.

The [architecture](docs/architecture.md) defines the layers, ownership,
representations and behavioral contracts. The [coding style](docs/coding-style.md)
defines naming, formatting, documentation and review rules. Changes to the code
must follow both; known deviations are recorded in the review linked above.

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

`bounded_compositions.optuna` has two ways to search compositions with Optuna.
`CompositionSampler` gives trial number t point t of a sequence, so the
workers of a distributed study use distinct sequence indices without sharing
a counter (give them the same space, seed and sampling options). The default
rank sequence visits distinct compositions until its capacity; later trials
use independent draws, which can repeat. `suggest_composition` works with any
sampler (TPE, CMA-ES, GP, NSGA-II) through the unit-cube codec:

```python
import optuna
from bounded_compositions.optuna import CompositionSampler, suggest_composition, suggest_parts

study = optuna.create_study(sampler=CompositionSampler(space, seed=1))
study.optimize(lambda trial: f(suggest_parts(trial, space)), n_trials=100)

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
study.optimize(lambda trial: f(suggest_composition(trial, "x", space)), n_trials=100)
```

Meson consumers can use the `bounded-compositions` dependency (`bounded_compositions_dep`).

## License

BSD-3-Clause. The Sobol direction numbers in `src/sobol_directions.cpp` are
Joe and Kuo's, under their own BSD-style licence:
[licenses/sobol-direction-numbers.txt](licenses/sobol-direction-numbers.txt).
