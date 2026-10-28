"""The bindings against tests/data/sequence_vectors.inc, which the C++ tests
check too: the same seed gives the same points in both languages."""

import pathlib
import re
import sys

import bounded_compositions as bc

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import gen_sequence_vectors as gen  # noqa: E402

ROW = re.compile(r'\{(\d+), "(\w+)", (\d+)u, (true|false), (\d+), \{([\d, -]*)\}, (\d+)u, "(\d+)"\},')


def test_points_match_the_shared_vectors():
    spaces = gen.spaces(bc)
    rows = [ROW.fullmatch(line) for line in (ROOT / "tests/data/sequence_vectors.inc").read_text().splitlines()
            if line.startswith("{")]
    assert rows and all(rows)
    for m in rows:
        sid, kind, seed, has_seed, n, prefix, index, rank = m.groups()
        s = spaces[int(sid)]
        prefix = [int(v) for v in prefix.split(",")] if int(n) else []
        region = bc.subtree(s, prefix) if prefix else None
        q = gen.make(bc, s, kind, int(seed) if has_seed == "true" else None, region)
        assert q.rank_at(int(index)) == int(rank), m.group(0)
    assert len(rows) > 400
