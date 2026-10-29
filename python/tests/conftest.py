import itertools
import random

import pytest


def brute_force(total, lower, upper, min_count=0, max_count=None, min_size=None, groups=(),
                gaps=None):
    """Every composition, in lexicographic order, by plain enumeration."""
    out = []
    for x in itertools.product(*(range(a, b + 1) for a, b in zip(lower, upper))):
        if sum(x) != total:
            continue
        if any(first <= x[part] <= last for part, first, last in gaps or ()):
            continue
        nonzero = sum(v != 0 for v in x)
        if nonzero < min_count or (max_count is not None and nonzero > max_count):
            continue
        if min_size is not None and any(v != 0 and abs(v) < m for v, m in zip(x, min_size)):
            continue
        if any(not _group_ok(x, g) for g in groups):
            continue
        out.append(list(x))
    return out


def _group_ok(x, g):
    s = sum(x[p] for p in g.parts)
    return (g.min_sum is None or s >= g.min_sum) and (g.max_sum is None or s <= g.max_sum)


def random_cases(count, seed, with_limits):
    """Small spaces with bounds (some negative) and, optionally, every kind of
    limit, drawn around a random vector so most are non-empty."""
    import bounded_compositions as bc

    rng = random.Random(seed)
    cases = []
    while len(cases) < count:
        n = rng.randint(1, 5)
        lower = [rng.randint(-3, 1) if rng.random() < 0.25 else 0 for _ in range(n)]
        upper = [a + rng.randint(0, 5) for a in lower]
        x0 = [rng.randint(a, b) for a, b in zip(lower, upper)]
        kw = {}
        if with_limits:
            nonzero = sum(v != 0 for v in x0)
            if rng.random() < 0.5:
                kw["min_count"] = max(0, nonzero - rng.randint(0, 2))
                kw["max_count"] = min(n, nonzero + rng.randint(0, 2))
            if rng.random() < 0.3:
                kw["min_size"] = [rng.randint(1, 3) for _ in range(n)]
            groups = []
            for _ in range(rng.randint(0, 2)):
                parts = [p for p in range(n) if rng.random() < 0.5] or [rng.randrange(n)]
                s0 = sum(x0[p] for p in parts)
                groups.append(bc.Group(parts, min_sum=s0 - rng.randint(0, 3), max_sum=s0 + rng.randint(0, 3)))
            if groups:
                kw["groups"] = groups
        cases.append((sum(x0), lower, upper, kw))
    return cases


@pytest.fixture
def small():
    import bounded_compositions as bc

    return bc.Space(30, [0] * 6, [12] * 6)


@pytest.fixture
def wide():
    import bounded_compositions as bc

    return bc.Space(100, [0] * 100, [100] * 100)
