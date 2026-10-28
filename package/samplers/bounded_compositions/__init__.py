"""OptunaHub package of bounded-compositions: Optuna samplers for integer
compositions (vectors with a fixed sum and bounds on each part). The code
lives in the bounded-compositions library; this package re-exports its
Optuna adapter."""

from bounded_compositions.optuna import (
    CompositionSampler,
    part_names,
    suggest_composition,
    suggest_parts,
)
from bounded_compositions import Group, Space

__all__ = [
    "CompositionSampler",
    "Group",
    "Space",
    "part_names",
    "suggest_composition",
    "suggest_parts",
]
