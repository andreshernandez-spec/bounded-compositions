"""Count, rank, enumerate and sample bounded integer compositions.

A Space is the set of integer vectors x with x[0] + ... + x[n-1] == total and
lower[i] <= x[i] <= upper[i], optionally with limits on the nonzero parts and
on sums of groups of parts. Its compositions are ordered lexicographically,
so a rank is a position in that order and the compositions sharing their
first parts form a contiguous Region.

Sequences produce compositions one index at a time: point i depends only on
the space, the kind of sequence, its seed and i. Seeds are the 64-bit seeds of
the C++ library, so the same seed gives the same points in both.

bounded_compositions.scipy adapts spaces to SciPy: a QMCEngine of
unit-cube codes and box objectives for scipy.optimize.
"""

import os

from . import _core
from ._core import (
    FreshSampler,
    Group,
    Region,
    Sequence,
    Space,
    VisitedSet,
    children,
    coordinate_sequence,
    decode,
    describe,
    dyadic_split,
    encode,
    iid_sequence,
    interval,
    rank_sequence,
    repair,
    scaled_rank_sequence,
    split,
    subtree,
    version,
    whole,
)

__version__ = version()

# Thread pools do not survive fork(); a child starts its own.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_core._forget_thread_pools)

__all__ = [
    "FreshSampler",
    "Group",
    "Region",
    "Sequence",
    "Space",
    "VisitedSet",
    "children",
    "coordinate_sequence",
    "decode",
    "describe",
    "dyadic_split",
    "encode",
    "iid_sequence",
    "interval",
    "rank_sequence",
    "repair",
    "scaled_rank_sequence",
    "split",
    "subtree",
    "version",
    "whole",
]
