# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa
# Modifications: ASA refactor (thin artificial selection library).

from collections.abc import Mapping
from enum import Enum
from typing import Any

ProgramIdx = int


class FrontierType(str, Enum):
    """Strategy for tracking Pareto frontiers.

    INSTANCE:  per validation example (baseline)
    OBJECTIVE: per objective metric (e.g. accuracy, latency, cost)
    HYBRID:    both instance and objective fronts maintained
    CARTESIAN: per (example, objective) pair
    """

    INSTANCE = "instance"
    OBJECTIVE = "objective"
    HYBRID = "hybrid"
    CARTESIAN = "cartesian"


def is_dominated(
    y: ProgramIdx,
    programs: set[ProgramIdx],
    program_at_pareto_front: Mapping[Any, set[ProgramIdx]],
) -> bool:
    """Return True if y is dominated — every front containing y also contains some member of programs."""
    y_fronts = [front for front in program_at_pareto_front.values() if y in front]
    for front in y_fronts:
        if not any(other in programs for other in front):
            return False
    return True


def remove_dominated_programs(
    program_at_pareto_front: Mapping[Any, set[ProgramIdx]],
    scores: Mapping[ProgramIdx, float] | None = None,
) -> dict[Any, set[ProgramIdx]]:
    """Return a pruned front where programs dominated by others are removed."""
    freq: dict[ProgramIdx, int] = {}
    for front in program_at_pareto_front.values():
        for p in front:
            freq[p] = freq.get(p, 0) + 1

    dominated: set[ProgramIdx] = set()
    programs = list(freq.keys())

    if scores is None:
        scores = dict.fromkeys(programs, 1)

    programs = sorted(programs, key=lambda x: scores[x])

    found_to_remove = True
    while found_to_remove:
        found_to_remove = False
        for y in programs:
            if y in dominated:
                continue
            if is_dominated(y, set(programs).difference({y}).difference(dominated), program_at_pareto_front):
                dominated.add(y)
                found_to_remove = True
                break

    dominators = [p for p in programs if p not in dominated]
    for front in program_at_pareto_front.values():
        if not front:
            continue
        assert any(p in front for p in dominators)

    return {key: {p for p in front if p in dominators} for key, front in program_at_pareto_front.items()}
