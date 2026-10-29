---
author: Andres Hernandez
title: Bounded Compositions Sampler
description: Samples integer vectors with a fixed sum and bounds on each part, from extensible low-discrepancy sequences, and searches them with any Optuna sampler.
tags: [sampler, quasi-Monte Carlo, composition, constrained, integer]
optuna_versions: [5.0.0]
license: BSD 3-Clause License
---

## Abstract

Many search spaces are compositions: integer vectors whose parts add up to a
fixed total, each part within its own bounds, such as a budget split across
channels in whole units or portfolio weights on a 1% grid. Optuna samples each
parameter on its own, so it cannot keep the sum. This package draws whole
compositions instead, optionally with limits on the number of nonzero parts,
semicontinuous parts (zero or at least some size) and the sums of groups of
parts.

`CompositionSampler` gives trial number t point t of a sequence that spreads
its points over the compositions: every prefix of the sequence is spread out,
and the rank sequence never repeats a point until the space is exhausted. The
storage hands out trial numbers, so the workers of a distributed study draw
distinct points without sharing a counter. `suggest_composition` works with
any Optuna sampler (TPE, CMA-ES, GP, NSGA-II) by searching the unit cube and
decoding each point to a composition.

## APIs

- `Space(total, lower, upper, *, min_count=0, max_count=None, min_size=None, groups=(), gaps=None)`:
  the compositions of `total` with `lower[i] <= x[i] <= upper[i]`, between
  `min_count` and `max_count` nonzero parts, part `i` zero or at least
  `min_size[i]` in size, each `Group(parts, min_sum, max_sum)` keeping its
  parts' sum in range, and no part in one of its gaps, given as
  `(part, first, last)` triples.
- `CompositionSampler(space, *, prefix="x", kind="rank", scramble=True, seed=None, independent_sampler=None)`:
  trial number t gets point t of a `"rank"`, `"coordinate"` (Sobol, spreads
  every part) or `"iid"` sequence. Give every worker of a distributed study
  the same `seed`. Other parameters go to `independent_sampler`
  (`RandomSampler(seed)` by default).
- `suggest_parts(trial, space, prefix="x")`: suggests the parts
  `prefix0, prefix1, ...` with the space's bounds, for `CompositionSampler`,
  and returns the composition.
- `suggest_composition(trial, name, space)`: a composition searched by any
  sampler; stored as the trial's user attribute `name`.

## Installation

The sampler is the Optuna adapter of the bounded-compositions library, which
has a compiled core:

```shell
pip install -r https://raw.githubusercontent.com/andreshernandez-spec/bounded-compositions/main/package/samplers/bounded_compositions/requirements.txt
```

## Example

```python
import numpy as np
import optuna
import optunahub

hub = optunahub.load_module(
    package="samplers/bounded_compositions",
    repo_owner="andreshernandez-spec",
    repo_name="bounded-compositions",
)

space = hub.Space(100, [0] * 10, [40] * 10, min_count=3, max_count=6)
target = np.array([30, 30, 0, 20, 20, 0, 0, 0, 0, 0])


def objective(trial):
    x = hub.suggest_parts(trial, space)
    return float(((x - target) ** 2).sum())


study = optuna.create_study(sampler=hub.CompositionSampler(space, seed=1))
study.optimize(objective, n_trials=200)
```

With any other sampler:

```python
def objective(trial):
    x = hub.suggest_composition(trial, "x", space)
    return float(((x - target) ** 2).sum())


study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
study.optimize(objective, n_trials=200)
```

## Others

How the sequences are built and how they measure is in the library's
[sampling report](https://github.com/andreshernandez-spec/bounded-compositions/blob/main/docs/sampling.md).
