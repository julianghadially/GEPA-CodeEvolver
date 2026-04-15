# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa
# Modifications: ASA refactor (thin artificial selection library).
"""ASA — Artificial Selection Algorithm.

A thin state-tracking library for agent-directed candidate evolution. Maintains a candidate pool,
per-row validation subscores, multi-objective scores, and Pareto frontiers. The caller picks
parents and runs evaluation; ASA tracks state and exposes frontiers as a memory substrate.
"""

from asa.frontier import FrontierType, ProgramIdx, is_dominated, remove_dominated_programs
from asa.result import ASAResult
from asa.state import ASAState, DataId, ObjectiveScores

__all__ = [
    "ASAState",
    "ASAResult",
    "FrontierType",
    "ProgramIdx",
    "DataId",
    "ObjectiveScores",
    "is_dominated",
    "remove_dominated_programs",
]
