import numpy as np
import pytest

import bounded_compositions as bc
from conftest import brute_force, random_cases


@pytest.mark.parametrize("with_limits", [False, True])
def test_repair_is_the_first_nearest(with_limits):
    rng = np.random.default_rng(4)
    for total, lower, upper, kw in random_cases(150, 20 + with_limits, with_limits):
        s = bc.Space(total, lower, upper, **kw)
        xs = np.array(brute_force(total, lower, upper, **kw), dtype=float).reshape(-1, len(lower))
        if len(xs) == 0:
            continue
        z = rng.uniform(np.array(lower) - 3, np.array(upper) + 3, size=(4, len(lower)))
        z[1] = np.round(z[1] * 2) / 2  # exact ties
        got = bc.repair(s, z)
        for zi, gi in zip(z, got):
            d = ((xs - zi) ** 2).sum(axis=1)
            best = xs[np.flatnonzero(d <= d.min() + 1e-9 * max(1.0, d.min()))[0]]
            assert gi.tolist() == best.astype(int).tolist()


def test_rows_with_workers(wide):
    z = np.random.default_rng(5).uniform(0, 2, size=(300, 100))
    rows = bc.repair(wide, z, workers=4)
    assert (rows == bc.repair(wide, z)).all()
    assert (rows.sum(axis=1) == 100).all()
    with pytest.raises(ValueError):
        bc.repair(wide, np.full(100, np.nan))
