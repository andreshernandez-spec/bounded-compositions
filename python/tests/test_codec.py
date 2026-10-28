import numpy as np
import pytest

import bounded_compositions as bc
from conftest import brute_force, random_cases


@pytest.mark.parametrize("with_limits", [False, True])
def test_decode_of_encode_is_identity(with_limits):
    for total, lower, upper, kw in random_cases(150, 10 + with_limits, with_limits):
        if len(lower) < 2:
            continue
        s = bc.Space(total, lower, upper, **kw)
        xs = np.array(brute_force(total, lower, upper, **kw), dtype=np.int32).reshape(-1, len(lower))
        if len(xs) == 0:
            continue
        for dtype in (np.float64, np.uint64):
            u = bc.encode(s, xs, dtype=dtype)
            assert u.dtype == dtype and u.shape == (len(xs), len(lower) - 1)
            assert (bc.decode(s, u) == xs).all()
            assert (bc.decode(s, u[0]) == xs[0]).all()


def test_rows_with_workers_match_one_by_one(wide):
    u = np.random.default_rng(3).random((500, 99))
    rows = bc.decode(wide, u, workers=4)
    assert rows.shape == (500, 100)
    assert all((bc.decode(wide, u[i]) == rows[i]).all() for i in range(0, 500, 50))
    logs = bc.Space(100, [0] * 100, [100] * 100, counting="logarithmic")
    assert (bc.decode(logs, u, workers=2) == rows).mean() > 0.99


def test_decode_rejects_bad_input(small):
    with pytest.raises(ValueError):
        bc.decode(small, [0.5] * 4)
    with pytest.raises(ValueError):
        bc.decode(small, [0.5, 0.5, 0.5, 0.5, 1.0])
    with pytest.raises(ValueError):
        bc.decode(small, [0.5, 0.5, 0.5, 0.5, float("nan")])
    with pytest.raises(ValueError):
        bc.encode(small, [1, 1, 1, 1, 1, 1])
    with pytest.raises(ValueError):
        bc.encode(small, [5, 5, 5, 5, 5, 5], dtype=np.float32)
