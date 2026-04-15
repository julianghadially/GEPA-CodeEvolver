from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from asa.frontier import ProgramIdx
from asa.state import ASAState, DataId, ObjectiveScores


@dataclass(frozen=True)
class ASAResult:
    """Immutable snapshot of an ASA run.

    - candidates: list of candidates (component_name -> component_text)
    - parents: for each candidate, a list of parent indices
    - val_aggregate_scores: per-candidate average val score
    - val_subscores: per-candidate mapping val_id -> score
    - per_val_instance_best_candidates: val_id -> set of candidates winning on that instance
    - val_objective_scores: per-candidate mapping objective -> aggregate score
    - per_objective_best_candidates: objective -> set of candidates winning on that objective
    - objective_pareto_front: objective -> best score
    - discovery_eval_counts: total metric calls when each candidate was added
    - best_outputs_valset: optional per-instance best outputs
    """

    candidates: list[dict[str, str]]
    parents: list[list[ProgramIdx]]
    val_aggregate_scores: list[float]
    val_subscores: list[dict[DataId, float]]
    per_val_instance_best_candidates: dict[DataId, set[ProgramIdx]]
    val_objective_scores: list[ObjectiveScores]
    per_objective_best_candidates: dict[str, set[ProgramIdx]]
    objective_pareto_front: ObjectiveScores
    discovery_eval_counts: list[int]
    best_outputs_valset: dict[DataId, list[tuple[ProgramIdx, Any]]] | None = None
    total_metric_calls: int | None = None

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)

    @property
    def num_val_instances(self) -> int:
        return len(self.per_val_instance_best_candidates)

    @property
    def best_idx(self) -> int:
        return max(range(len(self.val_aggregate_scores)), key=lambda i: self.val_aggregate_scores[i])

    @property
    def best_candidate(self) -> dict[str, str]:
        return self.candidates[self.best_idx]

    @staticmethod
    def from_state(state: ASAState) -> ASAResult:
        return ASAResult(
            candidates=[dict(c) for c in state.program_candidates],
            parents=[list(p) for p in state.parent_program_for_candidates],
            val_aggregate_scores=[state.get_program_average_val_score(i) for i in range(len(state.program_candidates))],
            val_subscores=[dict(s) for s in state.prog_candidate_val_subscores],
            per_val_instance_best_candidates={
                val_id: set(front) for val_id, front in state.program_at_pareto_front_valset.items()
            },
            val_objective_scores=[dict(s) for s in state.prog_candidate_objective_scores],
            per_objective_best_candidates={
                obj: set(front) for obj, front in state.program_at_pareto_front_objectives.items()
            },
            objective_pareto_front=dict(state.objective_pareto_front),
            discovery_eval_counts=list(state.num_metric_calls_by_discovery),
            best_outputs_valset=(
                {val_id: list(entries) for val_id, entries in state.best_outputs_valset.items()}
                if state.best_outputs_valset is not None
                else None
            ),
            total_metric_calls=state.total_num_evals,
        )
