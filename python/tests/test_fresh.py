import multiprocessing
import sys

import numpy as np
import pytest

import bounded_compositions as bc
from conftest import brute_force


def test_visited_set_takes_ranks_and_compositions(small):
    v = bc.VisitedSet(small)
    x = small.unrank(123)
    assert v.insert(x) and not v.insert(123) and not v.insert(np.int64(123))
    assert v.contains(123) and v.contains(x) and len(v) == 1
    rows = np.array([small.unrank(i) for i in (1, 123, 5)])
    assert v.insert(rows).tolist() == [True, False, True]
    assert v.contains(rows).tolist() == [True, True, True]
    with pytest.raises(ValueError):
        v.insert([1, 2, 3])


@pytest.mark.parametrize("workers", [1, 3])
def test_fresh_sampler_covers_a_region_once(small, workers):
    t = bc.subtree(small, [5, 5])
    v = bc.VisitedSet(small)
    global_seq = bc.rank_sequence(small)
    for x in global_seq.take(4096):
        v.insert(x)
    inside_before = sum(v.contains(t.begin + i) for i in range(t.size))
    for source in (bc.rank_sequence(small, t), bc.iid_sequence(small, 1, t),
                   bc.coordinate_sequence(small, t, seed=1)):
        vv = bc.VisitedSet(small)
        for x in global_seq.points(0, 4096):
            vv.insert(x)
        f = bc.FreshSampler(source, vv, patience=64)
        pts = f.take(t.size, workers=workers)
        ranks = [small.rank(x) for x in pts]
        assert len(set(ranks)) == len(ranks) == t.size - inside_before
        assert all(t.contains(r) for r in ranks)
        assert f.exhausted and f.accepted == len(pts)


def test_filter_counts_and_hands_out_what_passes(small):
    t = bc.subtree(small, [5, 5, 5])
    xs = [x for x in brute_force(30, [0] * 6, [12] * 6) if x[:3] == [5, 5, 5]]
    passing = [x for x in xs if x[3] <= x[4]]
    f = bc.FreshSampler(bc.rank_sequence(small, t), bc.VisitedSet(small), accept=lambda x: x[3] <= x[4])
    pts = f.take(len(xs), workers=2)
    assert sorted(p.tolist() for p in pts) == sorted(passing)
    assert (f.accepted, f.rejected) == (len(passing), len(xs) - len(passing))


def test_coordinate_source_when_the_table_is_wider_than_the_size():
    # The size takes 60 bits; row 0 of the table holds prefix sums over every
    # total up to 207, which take 65.
    s = bc.Space(207, [0] * 12, [207] * 12)
    assert s.size < 2**64
    v = bc.VisitedSet(s)
    pts = bc.FreshSampler(bc.coordinate_sequence(s, seed=3), v).take(1000)
    assert len({tuple(x) for x in pts.tolist()}) == 1000 == len(v)
    assert all(v.contains(x) for x in pts)


def test_filter_errors_propagate(small):
    def boom(_):
        raise KeyError("from the filter")

    f = bc.FreshSampler(bc.rank_sequence(small), bc.VisitedSet(small), accept=boom)
    with pytest.raises(KeyError):
        f.take(5, workers=2)
    with pytest.raises(TypeError):
        bc.FreshSampler(bc.rank_sequence(small), bc.VisitedSet(small), accept=3)


def _child_take(queue):
    s = bc.Space(100, [0] * 100, [100] * 100)
    queue.put(bc.rank_sequence(s).points(0, 200, workers=2).sum())


@pytest.mark.skipif(sys.platform != "linux", reason="fork start method")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_thread_pools_survive_fork(wide):
    expected = bc.rank_sequence(wide).points(0, 200, workers=2).sum()  # the parent's pool exists
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    p = ctx.Process(target=_child_take, args=(queue,))
    p.start()
    p.join(60)
    assert p.exitcode == 0
    assert queue.get(timeout=5) == expected
