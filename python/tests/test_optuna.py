import importlib.util
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import bounded_compositions as bc

optuna = pytest.importorskip("optuna", minversion="4.0")
from bounded_compositions.optuna import (  # noqa: E402
    CompositionSampler,
    part_names,
    suggest_composition,
    suggest_parts,
)

optuna.logging.set_verbosity(optuna.logging.ERROR)


def parts_of(trial, space, prefix="x"):
    return [trial.params[n] for n in part_names(space, prefix)]


@pytest.mark.parametrize("kind", ["rank", "coordinate", "iid"])
def test_trial_t_gets_point_t(small, kind):
    sampler = CompositionSampler(small, kind=kind, seed=3)
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda t: float(suggest_parts(t, small) @ np.arange(6)), n_trials=40)
    expected = sampler.sequence.points(0, 40)
    for t in study.trials:
        assert parts_of(t, small) == expected[t.number].tolist()
        assert sum(parts_of(t, small)) == 30
    assert np.array_equal(CompositionSampler(small, kind=kind, seed=3).sequence.points(0, 40), expected)
    if kind == "rank":
        assert len({tuple(parts_of(t, small)) for t in study.trials}) == 40


def test_failed_trials_leave_gaps_and_other_parameters_sample_independently(small):
    sampler = CompositionSampler(small, prefix="w", seed=5)
    study = optuna.create_study(sampler=sampler)

    def objective(trial):
        x = suggest_parts(trial, small, prefix="w")
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        if trial.number % 3 == 1:
            raise ValueError("failed on purpose")
        return float(x[0]) * lr

    study.optimize(objective, n_trials=30, catch=(ValueError,))
    states = {t.state for t in study.trials}
    assert optuna.trial.TrialState.FAIL in states and optuna.trial.TrialState.COMPLETE in states
    for t in study.trials:
        assert parts_of(t, small, "w") == sampler.composition(t.number).tolist()
        assert 1e-4 <= t.params["lr"] <= 1e-1


def test_past_the_end_of_the_space():
    s = bc.Space(3, [0, 0, 0], [3, 3, 3])  # 10 compositions
    sampler = CompositionSampler(s, seed=1)
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda t: float(suggest_parts(t, s)[0]), n_trials=16)
    xs = [tuple(parts_of(t, s)) for t in study.trials]
    assert len(set(xs[:10])) == 10
    assert all(s.contains(list(x)) for x in xs)


def test_fixed_parts_and_logarithmic_spaces():
    s = bc.Space(10, [0, 2, 0], [5, 2, 8])
    sampler = CompositionSampler(s, seed=2)
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda t: float(suggest_parts(t, s)[2]), n_trials=10)
    assert all(parts_of(t, s)[1] == 2 and s.contains(parts_of(t, s)) for t in study.trials)
    with pytest.raises(ValueError):
        CompositionSampler(bc.Space(10, [0, 0], [8, 8], counting="logarithmic"))


def test_a_part_suggested_with_other_bounds_is_sampled_independently(small, caplog):
    sampler = CompositionSampler(small, seed=6)
    study = optuna.create_study(sampler=sampler)
    optuna.logging.enable_propagation()
    try:
        study.optimize(lambda t: float(t.suggest_int("x0", 11, 12)), n_trials=10)
    finally:
        optuna.logging.disable_propagation()
    outside = [t for t in study.trials if not 11 <= sampler.composition(t.number)[0] <= 12]
    assert outside and all(11 <= t.params["x0"] <= 12 for t in study.trials)
    assert "suggest_parts" in caplog.text


WORKER = textwrap.dedent("""
    import sys
    import optuna
    import bounded_compositions as bc
    from bounded_compositions.optuna import CompositionSampler, suggest_parts
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    space = bc.Space(30, [0] * 6, [12] * 6)
    storage = optuna.storages.RDBStorage(sys.argv[1], engine_kwargs={"connect_args": {"timeout": 60}})
    study = optuna.load_study(study_name="shared", storage=storage,
                              sampler=CompositionSampler(space, seed=11))
    study.optimize(lambda t: float(suggest_parts(t, space)[0]), n_trials=8)
""")


def test_workers_sharing_sqlite_draw_distinct_points(small, tmp_path):
    url = f"sqlite:///{(tmp_path / 'study.db').as_posix()}"
    optuna.create_study(study_name="shared", storage=url)
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    procs = [subprocess.Popen([sys.executable, str(script), url]) for _ in range(3)]
    assert all(p.wait(timeout=300) == 0 for p in procs)
    trials = optuna.load_study(study_name="shared", storage=url).trials
    assert sorted(t.number for t in trials) == list(range(24))
    expected = CompositionSampler(small, seed=11).sequence.points(0, 24)
    for t in trials:
        assert parts_of(t, small) == expected[t.number].tolist()
    assert len({tuple(parts_of(t, small)) for t in trials}) == 24


def _samplers():
    yield "tpe", lambda: optuna.samplers.TPESampler(seed=1)
    if importlib.util.find_spec("cmaes"):
        yield "cmaes", lambda: optuna.samplers.CmaEsSampler(seed=1)
    if importlib.util.find_spec("torch"):
        yield "gp", lambda: optuna.samplers.GPSampler(seed=1)
    yield "nsga2", lambda: optuna.samplers.NSGAIISampler(seed=1)


@pytest.mark.parametrize("name,make", list(_samplers()), ids=[n for n, _ in _samplers()])
def test_suggest_composition_with_any_sampler(name, make):
    s = bc.Space(30, [0] * 6, [12] * 6, min_count=2, max_count=4)
    target = np.array([12, 0, 0, 9, 9, 0])

    def objective(trial):
        x = suggest_composition(trial, "w", s)
        return float(((x - target) ** 2).sum())

    study = optuna.create_study(sampler=make())
    study.optimize(objective, n_trials=25)
    for t in study.trials:
        x = t.user_attrs["w"]
        assert s.contains(x) and len(t.params) == 5
        assert all(0.0 <= v <= 1.0 for v in t.params.values())
    assert study.best_value < max(t.value for t in study.trials)
