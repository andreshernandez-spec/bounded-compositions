"""Minimal example: 200 compositions of 100 into 10 parts of at most 40, with
3 to 6 nonzero parts, spread over the space, and the one nearest a target
among them. CompositionSampler fills the space evenly; it does not model the
objective."""

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
print(study.best_value, [study.best_params[n] for n in hub.part_names(space)])
