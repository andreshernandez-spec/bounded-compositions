"""The OptunaHub package in package/samplers/bounded_compositions loads and
samples, as optunahub.load_module would load it from the repository."""

import pathlib

import pytest

import bounded_compositions as bc

optuna = pytest.importorskip("optuna", minversion="4.0")
optunahub = pytest.importorskip("optunahub")

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_hub_package_loads_and_samples():
    hub = optunahub.load_local_module(package="samplers/bounded_compositions",
                                      registry_root=str(ROOT / "package"))
    from bounded_compositions.optuna import CompositionSampler

    assert hub.CompositionSampler is CompositionSampler and hub.Space is bc.Space
    space = hub.Space(30, [0] * 6, [12] * 6, min_count=2)
    sampler = hub.CompositionSampler(space, seed=1)
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda t: float(hub.suggest_parts(t, space)[0]), n_trials=10)
    assert all(space.contains([t.params[n] for n in hub.part_names(space)]) for t in study.trials)
    readme = (ROOT / "package/samplers/bounded_compositions/README.md").read_text()
    assert readme.startswith("---\nauthor:") and "\ntags: [sampler," in readme
