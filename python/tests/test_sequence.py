import threading

import numpy as np
import pytest

import bounded_compositions as bc


def sequences(s):
    return [
        bc.rank_sequence(s),
        bc.rank_sequence(s, seed=3),
        bc.scaled_rank_sequence(s),
        bc.iid_sequence(s, 7),
        bc.coordinate_sequence(s),
        bc.coordinate_sequence(s, seed=2**64 - 1),
    ]


def test_extending_gives_the_same_points(wide):
    for q in sequences(wide):
        first = q.take(1000)
        more = q.take(100)
        assert q.position == 1100
        assert (np.vstack([first, more]) == q.points(0, 1100)).all()
        q.seek(0)
        assert (q.take(1100) == q.points(0, 1100)).all()
        assert all(wide.contains(x) for x in more)


def test_workers_do_not_change_the_points(wide):
    for q in sequences(wide):
        one = q.points(0, 3000)
        assert (q.points(0, 3000, workers=4) == one).all()
        assert (q.points(0, 3000, workers=-1) == one).all()
    with pytest.raises(ValueError):
        bc.rank_sequence(wide).points(0, 10, workers=0)


def test_rank_at_is_the_rank_of_the_point(wide):
    for q in sequences(wide):
        pts = q.points(0, 50)
        assert [q.rank_at(i) for i in range(50)] == [wide.rank(x) for x in pts]


def test_rank_sequence_covers_without_repeats(small):
    q = bc.rank_sequence(small, seed=11)
    assert q.capacity == small.size
    ranks = [small.rank(x) for x in q.take(q.capacity)]
    assert sorted(ranks) == list(range(small.size))
    with pytest.raises(IndexError):
        q.take(1)


def test_seeds_use_all_64_bits(wide):
    a = bc.iid_sequence(wide, 2**63).points(0, 20)
    b = bc.iid_sequence(wide, 2**63 + 1).points(0, 20)
    c = bc.iid_sequence(wide, 1).points(0, 20)
    assert not (a == b).all() and not (a == c).all()
    assert bc.iid_sequence(wide, 2**64 - 1).seed == 2**64 - 1
    with pytest.raises((TypeError, OverflowError)):
        bc.iid_sequence(wide, 2**64)
    with pytest.raises((TypeError, OverflowError)):
        bc.iid_sequence(wide, -1)


def test_region_sequences_stay_inside_and_resume(small):
    outer = bc.rank_sequence(small)
    taken = outer.take(500)
    seen = {small.rank(x) for x in taken}
    for kid in bc.children(small, bc.whole(small)):
        inner = bc.rank_sequence(small, kid)
        inner.resume_after(outer)
        new = inner.take(min(50, inner.capacity - inner.position))
        for x in new:
            r = small.rank(x)
            assert kid.contains(r) and r not in seen
        for q in (bc.iid_sequence(small, 5, kid), bc.coordinate_sequence(small, kid, seed=5)):
            assert all(kid.contains(small.rank(x)) for x in q.take(20))


def test_kinds_and_repr(small):
    assert [q.kind for q in sequences(small)] == [
        "rank", "rank", "scaled_rank", "iid", "coordinate", "coordinate"]
    assert "rank" in repr(bc.rank_sequence(small))


def test_concurrent_takes_are_disjoint(wide):
    q = bc.rank_sequence(wide)
    got = []
    lock = threading.Lock()

    def work():
        for _ in range(20):
            start = q.position
            pts = q.take(50, workers=2)
            with lock:
                got.append(pts)
            assert start <= q.position

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ranks = sorted(wide.rank(x) for block in got for x in block)
    expected = sorted(q.rank_at(i) for i in range(4 * 20 * 50))
    assert ranks == expected
