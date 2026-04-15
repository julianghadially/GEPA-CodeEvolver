from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, ClassVar, TypeAlias

from asa.frontier import FrontierType, ProgramIdx

DataId: TypeAlias = int | str
ObjectiveScores: TypeAlias = dict[str, float]


class ASAState:
    """Candidate pool, per-row val subscores, multi-objective scores, and Pareto frontier as memory substrate.

    ASA (Artificial Selection Algorithm) is a thin state-tracking library. It maintains the candidate
    pool and Pareto frontiers but does NOT choose parents, run reflection, or orchestrate iteration —
    the caller (typically an agent) drives those decisions and reads the frontier as a memory substrate.
    """

    _SCHEMA_VERSION: ClassVar[int] = 1

    program_candidates: list[dict[str, str]]
    parent_program_for_candidates: list[list[ProgramIdx]]
    prog_candidate_val_subscores: list[dict[DataId, float]]
    prog_candidate_objective_scores: list[ObjectiveScores]
    num_metric_calls_by_discovery: list[int]

    pareto_front_valset: dict[DataId, float]
    program_at_pareto_front_valset: dict[DataId, set[ProgramIdx]]
    objective_pareto_front: ObjectiveScores
    program_at_pareto_front_objectives: dict[str, set[ProgramIdx]]
    pareto_front_cartesian: dict[tuple[DataId, str], float]
    program_at_pareto_front_cartesian: dict[tuple[DataId, str], set[ProgramIdx]]

    best_outputs_valset: dict[DataId, list[tuple[ProgramIdx, Any]]] | None
    full_program_trace: list[dict[str, Any]]

    i: int
    total_num_evals: int
    frontier_type: FrontierType
    track_best_outputs: bool

    def __init__(
        self,
        seed_candidate: Mapping[str, str],
        seed_val_scores: Mapping[DataId, float],
        seed_objective_scores_by_val_id: Mapping[DataId, ObjectiveScores] | None = None,
        seed_outputs_by_val_id: Mapping[DataId, Any] | None = None,
        frontier_type: FrontierType = FrontierType.INSTANCE,
        track_best_outputs: bool = False,
    ):
        if frontier_type in (FrontierType.OBJECTIVE, FrontierType.HYBRID, FrontierType.CARTESIAN):
            if not seed_objective_scores_by_val_id:
                raise ValueError(
                    f"frontier_type={frontier_type.value!r} requires seed_objective_scores_by_val_id to be provided."
                )

        self.program_candidates = [dict(seed_candidate)]
        self.parent_program_for_candidates = [[]]
        self.prog_candidate_val_subscores = [dict(seed_val_scores)]

        seed_objective_aggregate = self._aggregate_objective_scores(seed_objective_scores_by_val_id)
        self.prog_candidate_objective_scores = [seed_objective_aggregate]
        self.num_metric_calls_by_discovery = [0]

        self.pareto_front_valset = dict(seed_val_scores)
        self.program_at_pareto_front_valset = {val_id: {0} for val_id in seed_val_scores.keys()}
        self.objective_pareto_front = dict(seed_objective_aggregate)
        self.program_at_pareto_front_objectives = {obj: {0} for obj in seed_objective_aggregate.keys()}

        if frontier_type == FrontierType.CARTESIAN:
            assert seed_objective_scores_by_val_id is not None
            self.pareto_front_cartesian = {
                (val_id, obj): score
                for val_id, obj_scores in seed_objective_scores_by_val_id.items()
                for obj, score in obj_scores.items()
            }
            self.program_at_pareto_front_cartesian = {
                (val_id, obj): {0}
                for val_id, obj_scores in seed_objective_scores_by_val_id.items()
                for obj in obj_scores.keys()
            }
        else:
            self.pareto_front_cartesian = {}
            self.program_at_pareto_front_cartesian = {}

        self.frontier_type = frontier_type
        self.track_best_outputs = track_best_outputs
        if track_best_outputs and seed_outputs_by_val_id is not None:
            self.best_outputs_valset = {val_id: [(0, output)] for val_id, output in seed_outputs_by_val_id.items()}
        else:
            self.best_outputs_valset = None

        self.full_program_trace = []
        self.i = -1
        self.total_num_evals = len(seed_val_scores)

    # ---- Aggregation ----

    @staticmethod
    def _aggregate_objective_scores(
        val_objective_scores: Mapping[DataId, ObjectiveScores] | None,
    ) -> ObjectiveScores:
        if not val_objective_scores:
            return {}
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for obj_dict in val_objective_scores.values():
            for obj, score in obj_dict.items():
                totals[obj] = totals.get(obj, 0.0) + score
                counts[obj] = counts.get(obj, 0) + 1
        return {obj: totals[obj] / counts[obj] for obj in totals.keys() if counts[obj] > 0}

    # ---- Mutation ----

    def add_candidate(
        self,
        candidate: Mapping[str, str],
        val_scores_by_id: Mapping[DataId, float],
        parent_ids: list[ProgramIdx],
        objective_scores_by_val_id: Mapping[DataId, ObjectiveScores] | None = None,
        outputs_by_val_id: Mapping[DataId, Any] | None = None,
    ) -> ProgramIdx:
        """Append a new candidate and incrementally update every Pareto frontier."""
        if self.frontier_type in (FrontierType.OBJECTIVE, FrontierType.HYBRID, FrontierType.CARTESIAN):
            if not objective_scores_by_val_id:
                raise ValueError(f"frontier_type={self.frontier_type.value!r} requires objective_scores_by_val_id.")

        new_idx = len(self.program_candidates)
        self.program_candidates.append(dict(candidate))
        self.parent_program_for_candidates.append(list(parent_ids))
        self.prog_candidate_val_subscores.append(dict(val_scores_by_id))
        obj_aggregate = self._aggregate_objective_scores(objective_scores_by_val_id)
        self.prog_candidate_objective_scores.append(obj_aggregate)
        self.num_metric_calls_by_discovery.append(self.total_num_evals)

        for val_id, score in val_scores_by_id.items():
            output = outputs_by_val_id.get(val_id) if outputs_by_val_id else None
            self._update_instance_front(val_id, score, new_idx, output)

        self._update_objective_front(obj_aggregate, new_idx)

        if self.frontier_type == FrontierType.CARTESIAN:
            assert objective_scores_by_val_id is not None
            for val_id, obj_scores in objective_scores_by_val_id.items():
                for obj, obj_score in obj_scores.items():
                    self._update_cartesian_front(val_id, obj, obj_score, new_idx)

        return new_idx

    def _update_instance_front(self, val_id: DataId, score: float, program_idx: ProgramIdx, output: Any | None) -> None:
        prev = self.pareto_front_valset.get(val_id, float("-inf"))
        if score > prev:
            self.pareto_front_valset[val_id] = score
            self.program_at_pareto_front_valset[val_id] = {program_idx}
            if self.best_outputs_valset is not None and output is not None:
                self.best_outputs_valset[val_id] = [(program_idx, output)]
        elif score == prev:
            self.program_at_pareto_front_valset.setdefault(val_id, set()).add(program_idx)
            if self.best_outputs_valset is not None and output is not None:
                self.best_outputs_valset.setdefault(val_id, []).append((program_idx, output))

    def _update_objective_front(self, obj_scores: ObjectiveScores, program_idx: ProgramIdx) -> None:
        if not obj_scores:
            return
        for obj, score in obj_scores.items():
            prev = self.objective_pareto_front.get(obj, float("-inf"))
            if score > prev:
                self.objective_pareto_front[obj] = score
                self.program_at_pareto_front_objectives[obj] = {program_idx}
            elif score == prev:
                self.program_at_pareto_front_objectives.setdefault(obj, set()).add(program_idx)

    def _update_cartesian_front(self, val_id: DataId, objective: str, score: float, program_idx: ProgramIdx) -> None:
        key = (val_id, objective)
        prev = self.pareto_front_cartesian.get(key, float("-inf"))
        if score > prev:
            self.pareto_front_cartesian[key] = score
            self.program_at_pareto_front_cartesian[key] = {program_idx}
        elif score == prev:
            self.program_at_pareto_front_cartesian.setdefault(key, set()).add(program_idx)

    # ---- Budget + iteration audit ----

    def increment_evals(self, count: int) -> None:
        self.total_num_evals += count

    def record_iteration(self, metadata: Mapping[str, Any]) -> None:
        self.i += 1
        entry = {"i": self.i, **dict(metadata)}
        self.full_program_trace.append(entry)

    # ---- Memory-substrate reads ----

    def get_pareto_front(self) -> dict[DataId, set[ProgramIdx]]:
        """Instance Pareto front: val_id → set of candidates currently winning on that instance."""
        return {val_id: set(front) for val_id, front in self.program_at_pareto_front_valset.items()}

    def get_objective_front(self) -> dict[str, set[ProgramIdx]]:
        """Objective Pareto front: objective name → set of candidates winning on that objective."""
        return {obj: set(front) for obj, front in self.program_at_pareto_front_objectives.items()}

    def get_cartesian_front(self) -> dict[tuple[DataId, str], set[ProgramIdx]]:
        return {key: set(front) for key, front in self.program_at_pareto_front_cartesian.items()}

    def get_frontier_members(self) -> set[ProgramIdx]:
        """Union of all frontier members across every maintained front. 'Who is still winning somewhere.'"""
        members: set[ProgramIdx] = set()
        for front in self.program_at_pareto_front_valset.values():
            members.update(front)
        for front in self.program_at_pareto_front_objectives.values():
            members.update(front)
        for front in self.program_at_pareto_front_cartesian.values():
            members.update(front)
        return members

    def get_unique_wins(self, program_idx: ProgramIdx) -> set[DataId]:
        """Val instances where this candidate is the only one on the instance front."""
        return {val_id for val_id, front in self.program_at_pareto_front_valset.items() if front == {program_idx}}

    def get_unique_objective_wins(self, program_idx: ProgramIdx) -> set[str]:
        """Objectives where this candidate is the only one on the objective front."""
        return {obj for obj, front in self.program_at_pareto_front_objectives.items() if front == {program_idx}}

    def get_program_average_val_score(self, program_idx: ProgramIdx) -> float:
        scores = self.prog_candidate_val_subscores[program_idx]
        if not scores:
            return float("-inf")
        return sum(scores.values()) / len(scores)

    def get_best_program(self) -> ProgramIdx:
        """Candidate with the highest average val subscore."""
        return max(
            range(len(self.program_candidates)),
            key=self.get_program_average_val_score,
        )

    # ---- Consistency ----

    def is_consistent(self) -> bool:
        n = len(self.program_candidates)
        assert len(self.parent_program_for_candidates) == n
        assert len(self.prog_candidate_val_subscores) == n
        assert len(self.prog_candidate_objective_scores) == n
        assert len(self.num_metric_calls_by_discovery) == n
        assert set(self.pareto_front_valset.keys()) == set(self.program_at_pareto_front_valset.keys())
        assert set(self.objective_pareto_front.keys()) == set(self.program_at_pareto_front_objectives.keys())
        for front in self.program_at_pareto_front_valset.values():
            for p in front:
                assert p < n
        return True

    # ---- JSON persistence ----

    def save(self, path: str) -> None:
        """Write state to a JSON file."""
        with open(path, "w") as f:
            json.dump(self._to_json_dict(), f, indent=2, default=_json_default)

    @classmethod
    def load(cls, path: str) -> ASAState:
        with open(path) as f:
            data = json.load(f)
        return cls._from_json_dict(data)

    def _to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self._SCHEMA_VERSION,
            "frontier_type": self.frontier_type.value,
            "track_best_outputs": self.track_best_outputs,
            "i": self.i,
            "total_num_evals": self.total_num_evals,
            "program_candidates": self.program_candidates,
            "parent_program_for_candidates": self.parent_program_for_candidates,
            "prog_candidate_val_subscores": [
                _encode_valscore_dict(scores) for scores in self.prog_candidate_val_subscores
            ],
            "prog_candidate_objective_scores": self.prog_candidate_objective_scores,
            "num_metric_calls_by_discovery": self.num_metric_calls_by_discovery,
            "pareto_front_valset": _encode_valscore_dict(self.pareto_front_valset),
            "program_at_pareto_front_valset": _encode_front_dict(self.program_at_pareto_front_valset),
            "objective_pareto_front": self.objective_pareto_front,
            "program_at_pareto_front_objectives": {
                obj: sorted(front) for obj, front in self.program_at_pareto_front_objectives.items()
            },
            "pareto_front_cartesian": [
                [val_id, obj, score] for (val_id, obj), score in self.pareto_front_cartesian.items()
            ],
            "program_at_pareto_front_cartesian": [
                [val_id, obj, sorted(front)] for (val_id, obj), front in self.program_at_pareto_front_cartesian.items()
            ],
            "best_outputs_valset": _encode_best_outputs(self.best_outputs_valset),
            "full_program_trace": self.full_program_trace,
        }

    @classmethod
    def _from_json_dict(cls, data: dict[str, Any]) -> ASAState:
        version = data.get("schema_version", 0)
        if version != cls._SCHEMA_VERSION:
            raise ValueError(f"Unsupported ASAState schema version {version}; expected {cls._SCHEMA_VERSION}")

        state = cls.__new__(cls)
        state.frontier_type = FrontierType(data["frontier_type"])
        state.track_best_outputs = data["track_best_outputs"]
        state.i = data["i"]
        state.total_num_evals = data["total_num_evals"]
        state.program_candidates = [dict(c) for c in data["program_candidates"]]
        state.parent_program_for_candidates = [list(p) for p in data["parent_program_for_candidates"]]
        state.prog_candidate_val_subscores = [_decode_valscore_dict(s) for s in data["prog_candidate_val_subscores"]]
        state.prog_candidate_objective_scores = [dict(s) for s in data["prog_candidate_objective_scores"]]
        state.num_metric_calls_by_discovery = list(data["num_metric_calls_by_discovery"])
        state.pareto_front_valset = _decode_valscore_dict(data["pareto_front_valset"])
        state.program_at_pareto_front_valset = _decode_front_dict(data["program_at_pareto_front_valset"])
        state.objective_pareto_front = dict(data["objective_pareto_front"])
        state.program_at_pareto_front_objectives = {
            obj: set(front) for obj, front in data["program_at_pareto_front_objectives"].items()
        }
        state.pareto_front_cartesian = {
            (_coerce_val_id(val_id), obj): score for val_id, obj, score in data["pareto_front_cartesian"]
        }
        state.program_at_pareto_front_cartesian = {
            (_coerce_val_id(val_id), obj): set(front)
            for val_id, obj, front in data["program_at_pareto_front_cartesian"]
        }
        state.best_outputs_valset = _decode_best_outputs(data.get("best_outputs_valset"))
        state.full_program_trace = list(data["full_program_trace"])
        return state


# ---- JSON helpers ----


def _json_default(x: Any) -> Any:
    try:
        return {**x}
    except Exception:
        return repr(x)


def _coerce_val_id(val_id: Any) -> DataId:
    # JSON object keys are always strings; if the original was an int, coerce back.
    if isinstance(val_id, str) and val_id.lstrip("-").isdigit():
        return int(val_id)
    return val_id


def _encode_valscore_dict(d: Mapping[DataId, float]) -> dict[str, float]:
    return {str(k): v for k, v in d.items()}


def _decode_valscore_dict(d: Mapping[str, float]) -> dict[DataId, float]:
    return {_coerce_val_id(k): v for k, v in d.items()}


def _encode_front_dict(d: Mapping[DataId, set[ProgramIdx]]) -> dict[str, list[ProgramIdx]]:
    return {str(k): sorted(front) for k, front in d.items()}


def _decode_front_dict(d: Mapping[str, list[ProgramIdx]]) -> dict[DataId, set[ProgramIdx]]:
    return {_coerce_val_id(k): set(front) for k, front in d.items()}


def _encode_best_outputs(
    best: dict[DataId, list[tuple[ProgramIdx, Any]]] | None,
) -> dict[str, list[list[Any]]] | None:
    if best is None:
        return None
    return {str(val_id): [[prog_idx, output] for prog_idx, output in entries] for val_id, entries in best.items()}


def _decode_best_outputs(
    data: dict[str, list[list[Any]]] | None,
) -> dict[DataId, list[tuple[ProgramIdx, Any]]] | None:
    if data is None:
        return None
    return {
        _coerce_val_id(val_id): [(int(entry[0]), entry[1]) for entry in entries] for val_id, entries in data.items()
    }
