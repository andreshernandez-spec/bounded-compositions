import numpy as np
import pytest

import bounded_compositions as bc

pytest.importorskip("scipy", minversion="1.15")
from scipy.optimize import differential_evolution  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from bounded_compositions.scipy import CompositionEngine, box_objective, decode_box  # noqa: E402


@pytest.mark.parametrize("kind", ["rank", "coordinate", "iid"])
def test_codes_decode_to_the_sequence(wide, kind):
    eng = CompositionEngine(wide, kind=kind, rng=1)
    assert isinstance(eng, qmc.QMCEngine) and eng.d == 99
    u = eng.random(300, workers=2)
    assert u.shape == (300, 99) and (u >= 0).all() and (u < 1).all()
    assert eng.num_generated == 300
    x = eng.reset().compositions(300)
    assert (bc.decode(wide, u) == x).all()
    assert (eng.sequence.points(0, 300) == x).all()


def test_a_failed_draw_leaves_the_engine_where_it_was(small, monkeypatch):
    import bounded_compositions.scipy as adapter

    eng = CompositionEngine(small, rng=7)
    first = eng.random(10)

    def no_code(*args, **kwargs):
        raise ValueError("no double decodes to the value of part 0")

    monkeypatch.setattr(adapter, "encode", no_code)
    with pytest.raises(ValueError):
        eng.random(5)
    assert eng.num_generated == 10 and eng.sequence.position == 10
    monkeypatch.undo()
    whole = CompositionEngine(small, rng=7).random(15)
    assert np.array_equal(np.vstack([first, eng.random(5)]), whole)


def test_extending_and_fast_forward(small):
    eng = CompositionEngine(small, rng=7)
    whole = eng.random(100)
    eng.reset()
    assert np.array_equal(np.vstack([eng.random(30), eng.random(70)]), whole)
    eng.reset().fast_forward(40)
    assert eng.num_generated == 40
    assert np.array_equal(eng.random(60), whole[40:])
    eng.reset()
    assert np.array_equal(eng.random(100, workers=3), whole)


def test_rng_and_scrambling(small):
    a = CompositionEngine(small, rng=3).random(50)
    assert np.array_equal(CompositionEngine(small, rng=3).random(50), a)
    assert not np.array_equal(CompositionEngine(small, rng=4).random(50), a)
    plain = CompositionEngine(small, scramble=False, rng=3).random(50)
    assert np.array_equal(CompositionEngine(small, scramble=False, rng=99).random(50), plain)
    assert np.array_equal(bc.decode(small, plain), bc.rank_sequence(small).points(0, 50))
    gen = np.random.default_rng(5)
    assert CompositionEngine(small, rng=gen).random(5).shape == (5, 5)


def test_rank_engine_covers_the_space_without_repeats():
    s = bc.Space(12, [0] * 4, [6] * 4)
    eng = CompositionEngine(s, rng=2)
    x = eng.compositions(s.size)
    assert len({tuple(r) for r in x}) == s.size
    with pytest.raises(IndexError):
        eng.random(1)


def test_bad_arguments(small):
    with pytest.raises(ValueError):
        CompositionEngine(bc.Space(3, [0], [5]))
    with pytest.raises(ValueError):
        CompositionEngine(small, kind="halton")


def test_box_objective_clips_and_vectorizes(small):
    calls = []

    def f(x):
        calls.append(np.asarray(x).shape)
        return float(np.asarray(x).sum())

    objective, bounds = box_objective(small, f)
    assert bounds == [(0.0, 1.0)] * 5
    assert objective(np.ones(5)) == 30.0 and objective(-np.ones(5)) == 30.0
    assert (decode_box(small, np.ones(5)) == bc.decode(small, np.full(5, np.nextafter(1.0, 0.0)))).all()
    u = np.random.default_rng(1).random((5, 7))
    assert objective(u).shape == (7,) and calls[-1] == (6,)
    vec, _ = box_objective(small, lambda x: x[:, 0].astype(float), vectorized=True)
    expected = bc.decode(small, np.ascontiguousarray(u.T))[:, 0]
    assert np.array_equal(vec(u), expected)


def test_differential_evolution_searches_compositions():
    s = bc.Space(30, [0] * 6, [12] * 6, min_count=3, max_count=4)
    target = np.array([12, 0, 9, 0, 9, 0])
    assert s.contains(target)

    def f(x):
        return float(((x - target) ** 2).sum())

    objective, bounds = box_objective(s, f)
    init = CompositionEngine(s, kind="coordinate", rng=0).random(60)
    start = min(f(x) for x in bc.decode(s, init))
    res = differential_evolution(objective, bounds, init=init, maxiter=300, polish=False, rng=1)
    best = decode_box(s, res.x)
    assert s.contains(best) and f(best) == res.fun
    # Within one unit moved of the optimum, from 24 at the start.
    assert start >= 20 and res.fun <= 2
