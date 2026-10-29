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


def test_cells_too_narrow_for_doubles_have_no_code(wide):
    # Part 0 takes 100 in one composition out of about 2^195.
    with pytest.raises(ValueError):
        bc.encode(wide, [100] + [0] * 99)
    with pytest.raises(ValueError):
        bc.encode(wide, [100] + [0] * 99, dtype=np.uint64)
    assert bc.encode(wide, [0] * 99 + [100]).shape == (99,)


def test_narrow_cells_that_decode_reaches_have_codes():
    # x[0] = 0 leaves one composition out of 2^99; code 0 decodes to it.
    s = bc.Space(99, [0] * 100, [99] + [1] * 99)
    x = bc.decode(s, np.zeros(99))
    assert x[:3].tolist() == [0, 1, 1]
    for dtype in (np.uint64, np.float64):
        u = bc.encode(s, x, dtype=dtype)
        assert u[0] == 0 and (bc.decode(s, u) == x).all()


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
