"""SciPy adapters: compositions as points of the unit cube.

A space of n parts maps to [0, 1)^(n - 1) through the codec: decode(space, u)
is a composition, and encode(space, x) the centre of x's cell, so
decode(space, encode(space, x)) == x. CompositionEngine is a
scipy.stats.qmc.QMCEngine whose samples are the codes of a sequence's
compositions, and box_objective turns a function of compositions into one of
the unit cube for the optimisers of scipy.optimize.

Needs SciPy 1.15 or later (the rng keyword of SPEC 7).
"""

import numpy as np
from scipy.stats import qmc

from ._core import coordinate_sequence, decode, encode, iid_sequence, rank_sequence

__all__ = ["CompositionEngine", "box_objective", "decode_box"]

# The largest double below 1: decode needs u < 1, optimisers' bounds are closed.
_BELOW_ONE = np.nextafter(1.0, 0.0)


class CompositionEngine(qmc.QMCEngine):
    """The compositions of a sequence, as codes in [0, 1)^(parts - 1).

    Sample i is encode(space, x_i) for the sequence's i-th composition x_i, so
    decode(space, engine.random(n)) gives the sequence's first n points and
    compositions(n) gives them directly. Drawing n points and then m more gives
    the first n + m, as with SciPy's Sobol and Halton engines.

    Parameters
    ----------
    space : Space
        At least two parts.
    kind : {"rank", "coordinate", "iid"}
        "rank" (the default) lists the ranks in radical-inverse order, with no
        repeats; "coordinate" decodes Sobol points, which spreads every part;
        "iid" draws independent uniform compositions.
    scramble : bool
        Scramble the rank or coordinate sequence with a seed drawn from rng.
        Unscrambled sequences do not depend on rng. "iid" always uses rng.
    rng : numpy.random.Generator or seed, optional
        As in SciPy: a Generator, or anything numpy.random.default_rng takes.

    Notes
    -----
    A composition whose cell is narrower than a double can represent (below
    about 2^-53 of the unit interval, which needs a space of more than 2^53
    compositions and an extreme composition) has no code: random() raises
    ValueError for it, and compositions() returns it. Decoding never produces
    such compositions, so box optimisers cannot reach them either.
    """

    def __init__(self, space, *, kind="rank", scramble=True, rng=None):
        if space.parts < 2:
            raise ValueError("the space needs at least two parts")
        if kind not in ("rank", "coordinate", "iid"):
            raise ValueError("kind must be 'rank', 'coordinate' or 'iid'")
        super().__init__(d=space.parts - 1, rng=rng)
        self.space = space
        self.kind = kind
        self.scramble = scramble
        seed = None
        if kind == "iid" or scramble:
            seed = int(self.rng.integers(0, 2**64, dtype=np.uint64))
        if kind == "rank":
            self._sequence = rank_sequence(space, seed=seed)
        elif kind == "coordinate":
            self._sequence = coordinate_sequence(space, seed=seed)
        else:
            self._sequence = iid_sequence(space, seed)

    @property
    def sequence(self):
        """The Sequence the samples come from; its position is the engine's."""
        return self._sequence

    def _random(self, n=1, *, workers=1):
        return encode(self.space, self._sequence.take(n, workers=workers), workers=workers)

    def compositions(self, n=1, *, workers=1):
        """The next n compositions, as an (n, parts) int32 array. They count
        as drawn, like random()."""
        x = self._sequence.take(n, workers=workers)
        self.num_generated += n
        return x

    def reset(self) -> "CompositionEngine":
        self._sequence.seek(0)
        self.num_generated = 0
        return self

    def fast_forward(self, n) -> "CompositionEngine":
        self._sequence.seek(self._sequence.position + n)
        self.num_generated += n
        return self


def decode_box(space, u, *, workers=1):
    """decode for points of the closed cube [0, 1]^(parts - 1), such as an
    optimiser's result: values outside [0, 1) move to the nearest point inside,
    as box_objective does."""
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, _BELOW_ONE)
    return decode(space, u, workers=workers)


def box_objective(space, func, *, vectorized=False):
    """func of compositions as a function of the unit cube, with its bounds.

    Returns (objective, bounds) for box-bounded optimisers such as
    scipy.optimize.differential_evolution, dual_annealing or direct:
    objective(u) is func(decode_box(space, u)), and bounds is
    [(0, 1)] * (parts - 1). decode_box moves values outside [0, 1) to the
    nearest point inside, since the optimisers' bounds are closed and decode
    needs u < 1; use it to turn the result into a composition too.

    objective also takes a 2-D array of shape (parts - 1, S), the layout of
    differential_evolution(..., vectorized=True): it then returns S values,
    calling func once per composition, or once on the (S, parts) array of
    compositions if vectorized is True.
    """
    if space.parts < 2:
        raise ValueError("the space needs at least two parts")

    def objective(u):
        u = np.asarray(u, dtype=np.float64)
        if u.ndim == 1:
            return func(decode_box(space, u))
        x = decode_box(space, np.ascontiguousarray(u.T))
        if vectorized:
            return np.asarray(func(x))
        return np.array([func(row) for row in x])

    return objective, [(0.0, 1.0)] * (space.parts - 1)
