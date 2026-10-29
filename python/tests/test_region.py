import pytest

import bounded_compositions as bc
from conftest import brute_force


def test_children_partition_a_subtree(small):
    t = bc.subtree(small, [4])
    kids = bc.children(small, t)
    assert kids[0].begin == t.begin and kids[-1].end == t.end
    assert all(a.end == b.begin for a, b in zip(kids, kids[1:]))
    assert all(k.is_subtree and k.prefix[:1] == [4] for k in kids)


def test_split_and_dyadic_split_cover_the_region(small):
    r = bc.interval(small, 1000, 50000)
    pieces = bc.split(small, r, 7)
    assert max(p.size for p in pieces) - min(p.size for p in pieces) <= 1
    assert pieces[0].begin == r.begin and pieces[-1].end == r.end
    dy = bc.dyadic_split(small, r, 3)
    assert dy[0].begin == r.begin and dy[-1].end == r.end
    assert all(a.end == b.begin for a, b in zip(dy, dy[1:]))
    with pytest.raises(ValueError):
        bc.dyadic_split(small, r, 25)


def test_describe_is_exact_for_subtrees():
    s = bc.Space(9, [0, 0, 0, 0], [4, 4, 4, 4])
    for prefix in ([], [1], [2, 3], [4, 0]):
        t = bc.subtree(s, prefix)
        xs = [x for x in brute_force(9, [0] * 4, [4] * 4) if x[: len(prefix)] == prefix]
        lohi = bc.describe(s, t)
        for i in range(4):
            assert lohi[i].tolist() == [min(x[i] for x in xs), max(x[i] for x in xs)]


def test_region_basics(small):
    r = bc.interval(small, 10, 20)
    assert (r.begin, r.end, r.size, r.empty) == (10, 20, 10, False)
    assert r.contains(10) and not r.contains(20)
    assert r == bc.interval(small, 10, 20) and not r == bc.whole(small)
    with pytest.raises(IndexError):
        bc.interval(small, 0, small.size + 1)
    with pytest.raises(ValueError):
        bc.subtree(small, [13])


def test_a_region_belongs_to_its_space():
    a = bc.Space(1, [0] * 3, [1] * 3)
    b = bc.Space(2, [0] * 3, [2] * 3)
    r = bc.subtree(a, [1])
    assert r.space.total == 1
    for call in (
        lambda: bc.coordinate_sequence(b, r),
        lambda: bc.rank_sequence(b, r),
        lambda: bc.describe(b, r),
        lambda: bc.split(b, r, 2),
    ):
        with pytest.raises(ValueError, match="another space"):
            call()
    again = bc.Space(1, [0] * 3, [1] * 3)
    assert bc.coordinate_sequence(again, r).points(0, 1)[0].tolist() == [1, 0, 0]
    assert r == bc.subtree(again, [1]) and r != bc.interval(b, r.begin, r.end)
