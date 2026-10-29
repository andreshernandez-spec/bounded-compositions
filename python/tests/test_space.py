import math
import pickle

import numpy as np
import pytest

import bounded_compositions as bc
from conftest import brute_force, random_cases


@pytest.mark.parametrize("with_limits", [False, True])
def test_matches_brute_force(with_limits):
    for total, lower, upper, kw in random_cases(300, 1 + with_limits, with_limits):
        s = bc.Space(total, lower, upper, **kw)
        expected = brute_force(total, lower, upper, **kw)
        assert s.size == len(expected)
        assert s.empty == (not expected)
        assert [x.tolist() for x in s] == expected
        for r, x in enumerate(expected):
            assert s.rank(x) == r
            assert s.unrank(r).tolist() == x
            assert s.contains(x)
        if expected:
            assert s.first().tolist() == expected[0]
            assert s.next(expected[-1]) is None
            logs = bc.Space(total, lower, upper, counting="logarithmic", **kw)
            assert logs.log_size == pytest.approx(math.log(len(expected)), abs=1e-9)


def test_python_ints_beyond_64_bits(wide):
    assert wide.size.bit_length() == 195
    last = wide.size - 1
    x = wide.unrank(last)
    assert x.tolist() == [100] + [0] * 99  # rank 0 is [0, ..., 0, 100]
    assert wide.rank(x) == last
    with pytest.raises(IndexError):
        wide.unrank(wide.size)
    with pytest.raises(ValueError):
        wide.unrank(-1)
    assert wide.unrank(np.uint64(5)).tolist() == wide.unrank(5).tolist()


def test_contains_rows(small):
    rows = np.array([[5, 5, 5, 5, 5, 5], [30, 0, 0, 0, 0, 0], [12, 12, 6, 0, 0, 0]])
    assert small.contains(rows).tolist() == [True, False, True]
    assert not small.contains([5, 5, 5])


def test_subspace_is_the_subtree(small):
    prefix = [3, 7]
    sub = small.subspace(prefix)
    t = bc.subtree(small, prefix)
    assert sub.size == t.size
    for r in range(0, sub.size, 37):
        assert small.unrank(t.begin + r)[2:].tolist() == sub.unrank(r).tolist()


def test_limits_are_kept():
    g = bc.Group([0, 2], min_sum=3, max_sum=9)
    s = bc.Space(20, [0] * 5, [8] * 5, min_count=2, max_count=4, min_size=[2] * 5, groups=[g])
    assert (s.min_count, s.max_count, s.min_size, s.groups) == (2, 4, [2] * 5, [g])
    plain = bc.Space(20, [0] * 5, [8] * 5)
    assert (plain.min_count, plain.max_count, plain.min_size, plain.groups) == (0, None, None, [])


def test_pickle_round_trip():
    s = bc.Space(20, [0] * 5, [8] * 5, min_count=2, groups=[bc.Group([1, 2], max_sum=6)])
    back = pickle.loads(pickle.dumps(s))
    assert back.size == s.size and back.groups == s.groups and back.min_count == 2


def test_subspaces_pickle_as_the_same_space():
    x = [-2000000000, 1300000000, 1300000000, -2000000000]
    group = bc.Group([0, 1, 2], min_sum=600000000, max_sum=600000000)
    s = bc.Space(-1400000000, x, x, groups=[group])
    with pytest.raises(ValueError, match="does not fit in int32"):
        s.subspace([-2000000000])  # the remaining group sums to 2.6e9
    fixed = [-(2**31), 2**31 - 1, 2**31 - 1]
    with pytest.raises(ValueError, match="does not fit in int32"):
        bc.Space(2**31 - 2, fixed, fixed).subspace([-(2**31)])
    g = bc.Group([0, 2], max_sum=9)
    s = bc.Space(12, [-2] * 5, [6] * 5, min_count=2, groups=[g, bc.Group([1, 3], min_sum=1)])
    for prefix in ([3], [-2, 6], [0, 0, 4]):
        sub = s.subspace(prefix)
        back = pickle.loads(pickle.dumps(sub))
        assert back.size == sub.size
        assert [y.tolist() for y in back] == [y.tolist() for y in sub]


def test_bad_input():
    with pytest.raises(TypeError):
        bc.Space(5, [0, 0], [2.5, 3])
    with pytest.raises(ValueError):
        bc.Space(5, [0, 0], [2**40, 3])
    with pytest.raises(ValueError):
        bc.Space(5, [0, 0], [3])
    with pytest.raises(ValueError):
        bc.Space(5, [0, 0], [3, 3], counting="approximate")
    s = bc.Space(5, [0, 0], [3, 3])
    with pytest.raises(ValueError):
        s.rank([1, 1, 3])
    with pytest.raises(ValueError):
        s.rank([1, 1])
    logs = bc.Space(5, [0, 0], [3, 3], counting="logarithmic")
    with pytest.raises(RuntimeError):
        logs.size


def test_infeasible_space_is_empty():
    s = bc.Space(50, [0] * 3, [5] * 3)
    assert s.empty and s.size == 0 and s.first() is None and list(s) == []
