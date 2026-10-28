"""Optuna adapters: a sampler that draws whole compositions, and a helper that
searches compositions with any sampler.

Optuna samples each parameter on its own, and a composition's parts are tied
by their sum, so they have to be drawn together.

CompositionSampler draws them as one relative sample: trial number t gets
point t of a sequence. The storage hands out trial numbers, so the workers of
a distributed study draw distinct points with no shared counter, as long as
they use the same seed. The objective suggests the parts with suggest_parts.

suggest_composition works with any sampler (TPE, CMA-ES, GP, NSGA-II): it
suggests parts - 1 floats in [0, 1] and decodes them with the unit-cube codec.

Needs Optuna 4 or later.
"""

import numpy as np
import optuna
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler, RandomSampler

from ._core import coordinate_sequence, decode, iid_sequence, rank_sequence

__all__ = ["CompositionSampler", "part_names", "suggest_composition", "suggest_parts"]

_logger = optuna.logging.get_logger(__name__)

# The largest double below 1: decode needs u < 1, suggest_float can return 1.
_BELOW_ONE = np.nextafter(1.0, 0.0)


def part_names(space, prefix="x"):
    """The parameter names of the parts: prefix0, prefix1, ..."""
    return [f"{prefix}{i}" for i in range(space.parts)]


def suggest_parts(trial, space, prefix="x"):
    """Suggests every part as an integer with the space's bounds and returns
    the composition as an int32 array. With a CompositionSampler of the same
    space and prefix, the parts come from its sequence and sum to the total."""
    names = part_names(space, prefix)
    lower, upper = space.lower, space.upper
    return np.array([trial.suggest_int(n, int(lo), int(hi)) for n, lo, hi in zip(names, lower, upper)],
                    dtype=np.int32)


def suggest_composition(trial, name, space):
    """A composition of space searched by the study's sampler, whatever it is.

    Suggests the floats name_u0 ... name_u{parts - 2} in [0, 1] and decodes
    them; the composition is also stored as the trial's user attribute name.
    Samplers that model the parameters jointly (CMA-ES, GP) see a smoother
    landscape than TPE, since what coordinate i means depends on the ones
    before it.
    """
    if space.parts < 2:
        x = space.first()
        if x is None:
            raise ValueError("the space is empty")
    else:
        u = [trial.suggest_float(f"{name}_u{i}", 0.0, 1.0) for i in range(space.parts - 1)]
        x = decode(space, np.clip(np.asarray(u, dtype=np.float64), 0.0, _BELOW_ONE))
    trial.set_user_attr(name, x.tolist())
    return x


class CompositionSampler(BaseSampler):
    """Gives trial t the composition at point t of a sequence.

    Parameters
    ----------
    space : Space
        Exact counting.
    prefix : str
        The parts are the integer parameters prefix0, prefix1, ... (see
        suggest_parts).
    kind : {"rank", "coordinate", "iid"}
        "rank" (the default) has no repeats until the space is exhausted;
        "coordinate" spreads every part; "iid" draws independent uniform
        compositions.
    scramble : bool
        Scramble the rank or coordinate sequence with a seed drawn from seed.
    seed : int, optional
        Seeds the scrambling, the independent draws past the end of the
        sequence, and the default independent sampler. Every worker of a
        distributed study needs the same seed, or they draw from different
        sequences.
    independent_sampler : BaseSampler, optional
        For parameters that are not parts. RandomSampler(seed) by default.

    Notes
    -----
    Failed and pruned trials leave gaps in the sequence. Once a rank
    sequence has listed the whole space, later trials get independent
    uniform compositions. A part suggested with other bounds that do not
    hold the sequence's value is sampled independently, with a warning, and
    the sum is no longer kept.
    """

    def __init__(self, space, *, prefix="x", kind="rank", scramble=True, seed=None,
                 independent_sampler=None):
        if kind not in ("rank", "coordinate", "iid"):
            raise ValueError("kind must be 'rank', 'coordinate' or 'iid'")
        if space.counting != "exact":
            raise ValueError("CompositionSampler needs a space with exact counting")
        self._space = space
        self._names = part_names(space, prefix)
        self._search_space = {
            n: IntDistribution(int(lo), int(hi))
            for n, lo, hi in zip(self._names, space.lower, space.upper)
            if lo < hi  # Optuna fills fixed parts itself
        }
        rng = np.random.default_rng(seed)
        scramble_seed = int(rng.integers(0, 2**64, dtype=np.uint64))
        if seed is None and (scramble or kind == "iid"):
            _logger.warning(
                "No seed is provided for `CompositionSampler` and the seed is set randomly. "
                "Workers of a distributed study need the same seed to draw from the same "
                "sequence.")
        if kind == "rank":
            self._sequence = rank_sequence(space, seed=scramble_seed if scramble else None)
        elif kind == "coordinate":
            self._sequence = coordinate_sequence(space, seed=scramble_seed if scramble else None)
        else:
            self._sequence = iid_sequence(space, scramble_seed)
        self._after_end = iid_sequence(space, int(rng.integers(0, 2**64, dtype=np.uint64)))
        self._independent_sampler = independent_sampler or RandomSampler(seed=seed)

    @property
    def space(self):
        return self._space

    @property
    def sequence(self):
        """The sequence trial numbers index into."""
        return self._sequence

    def composition(self, number):
        """The composition trial `number` gets."""
        if number < self._sequence.capacity:
            return self._sequence.points(number, 1)[0]
        return self._after_end.points(number, 1)[0]

    def infer_relative_search_space(self, study, trial):
        return dict(self._search_space)

    def sample_relative(self, study, trial, search_space):
        if not search_space:
            return {}
        x = self.composition(trial.number)
        return {n: int(v) for n, v in zip(self._names, x) if n in search_space}

    def sample_independent(self, study, trial, param_name, param_distribution):
        if param_name in self._search_space:
            _logger.warning(
                f"The part {param_name} of trial {trial.number} was suggested with "
                f"{param_distribution}, which does not hold its value in the sequence, so "
                f"{self._independent_sampler.__class__.__name__} samples it and the parts no "
                "longer keep their sum. Suggest the parts with suggest_parts.")
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution)

    def before_trial(self, study, trial):
        self._independent_sampler.before_trial(study, trial)

    def after_trial(self, study, trial, state, values):
        self._independent_sampler.after_trial(study, trial, state, values)

    def reseed_rng(self):
        # Never reseed the sequence: parallel workers must keep drawing from
        # the same one.
        self._independent_sampler.reseed_rng()
