import pytest

from asa import ASAResult, ASAState, FrontierType
from asa.frontier import is_dominated, remove_dominated_programs


def test_seed_initialization_instance_frontier():
    state = ASAState(
        seed_candidate={"m": "seed"},
        seed_val_scores={0: 0.5, 1: 0.7},
    )
    assert state.program_candidates == [{"m": "seed"}]
    assert state.parent_program_for_candidates == [[]]
    assert state.prog_candidate_val_subscores == [{0: 0.5, 1: 0.7}]
    assert state.get_pareto_front() == {0: {0}, 1: {0}}
    assert state.get_frontier_members() == {0}
    assert state.get_best_program() == 0
    assert state.total_num_evals == 2
    assert state.i == -1
    assert state.is_consistent()


def test_objective_frontier_requires_objective_scores():
    with pytest.raises(ValueError):
        ASAState(
            seed_candidate={"m": "seed"},
            seed_val_scores={0: 0.5},
            frontier_type=FrontierType.OBJECTIVE,
        )


def test_seed_initialization_objective_frontier():
    state = ASAState(
        seed_candidate={"m": "seed"},
        seed_val_scores={0: 0.5, 1: 0.9},
        seed_objective_scores_by_val_id={0: {"acc": 0.5, "lat": 0.3}, 1: {"acc": 0.9, "lat": 0.1}},
        frontier_type=FrontierType.OBJECTIVE,
    )
    assert state.prog_candidate_objective_scores[0] == pytest.approx({"acc": 0.7, "lat": 0.2})
    assert state.get_objective_front() == {"acc": {0}, "lat": {0}}


def test_cartesian_frontier_initialization():
    state = ASAState(
        seed_candidate={"m": "seed"},
        seed_val_scores={0: 0.5},
        seed_objective_scores_by_val_id={0: {"acc": 0.5, "lat": 0.3}},
        frontier_type=FrontierType.CARTESIAN,
    )
    assert state.get_cartesian_front() == {(0, "acc"): {0}, (0, "lat"): {0}}


def test_add_candidate_dominating_replaces_instance_front():
    state = ASAState({"m": "a"}, {0: 0.5, 1: 0.5})
    new_idx = state.add_candidate({"m": "b"}, {0: 0.9, 1: 0.9}, parent_ids=[0])
    assert new_idx == 1
    assert state.get_pareto_front() == {0: {1}, 1: {1}}
    assert state.get_frontier_members() == {1}
    assert state.get_best_program() == 1


def test_add_candidate_unique_instance_win_keeps_old_on_front():
    state = ASAState({"m": "a"}, {0: 0.5, 1: 0.5})
    state.add_candidate({"m": "b"}, {0: 0.1, 1: 0.9}, parent_ids=[0])
    # candidate 0 still wins val 0; candidate 1 uniquely wins val 1
    assert state.get_pareto_front() == {0: {0}, 1: {1}}
    assert state.get_frontier_members() == {0, 1}
    assert state.get_unique_wins(0) == {0}
    assert state.get_unique_wins(1) == {1}


def test_add_candidate_tie_extends_front():
    state = ASAState({"m": "a"}, {0: 0.5, 1: 0.5})
    state.add_candidate({"m": "b"}, {0: 0.5, 1: 0.5}, parent_ids=[0])
    assert state.get_pareto_front() == {0: {0, 1}, 1: {0, 1}}
    assert state.get_unique_wins(0) == set()
    assert state.get_unique_wins(1) == set()


def test_objective_front_updates_on_add():
    state = ASAState(
        {"m": "a"},
        {0: 0.5, 1: 0.9},
        seed_objective_scores_by_val_id={0: {"acc": 0.5, "lat": 0.3}, 1: {"acc": 0.9, "lat": 0.1}},
        frontier_type=FrontierType.HYBRID,
    )
    # candidate 1 beats acc (0.95 > 0.7) but loses lat (0.05 < 0.2)
    state.add_candidate(
        {"m": "b"},
        {0: 0.9, 1: 1.0},
        parent_ids=[0],
        objective_scores_by_val_id={0: {"acc": 0.9, "lat": 0.1}, 1: {"acc": 1.0, "lat": 0.0}},
    )
    assert state.get_objective_front() == {"acc": {1}, "lat": {0}}
    assert state.get_unique_objective_wins(0) == {"lat"}
    assert state.get_unique_objective_wins(1) == {"acc"}


def test_record_iteration_appends_trace():
    state = ASAState({"m": "a"}, {0: 0.5})
    state.record_iteration({"note": "first", "accepted": True})
    assert state.i == 0
    assert state.full_program_trace == [{"i": 0, "note": "first", "accepted": True}]


def test_increment_evals():
    state = ASAState({"m": "a"}, {0: 0.5})
    assert state.total_num_evals == 1
    state.increment_evals(5)
    assert state.total_num_evals == 6


def test_discovery_eval_count_matches_total_at_add():
    state = ASAState({"m": "a"}, {0: 0.5})
    state.increment_evals(4)
    state.add_candidate({"m": "b"}, {0: 0.9}, parent_ids=[0])
    # discovery count is snapshotted from total_num_evals at add time
    assert state.num_metric_calls_by_discovery == [0, 5]


def test_best_outputs_valset_tracked():
    state = ASAState(
        {"m": "a"},
        {0: 0.5, 1: 0.5},
        seed_outputs_by_val_id={0: "out_a0", 1: "out_a1"},
        track_best_outputs=True,
    )
    state.add_candidate(
        {"m": "b"},
        {0: 0.9, 1: 0.3},
        parent_ids=[0],
        outputs_by_val_id={0: "out_b0", 1: "out_b1"},
    )
    assert state.best_outputs_valset is not None
    assert state.best_outputs_valset[0] == [(1, "out_b0")]
    assert state.best_outputs_valset[1] == [(0, "out_a1")]


def test_json_roundtrip_instance_frontier(tmp_path):
    state = ASAState({"m": "a"}, {0: 0.5, 1: 0.7})
    state.add_candidate({"m": "b"}, {0: 0.8, 1: 0.4}, parent_ids=[0])
    state.record_iteration({"accepted": True})
    state.increment_evals(3)

    path = tmp_path / "state.json"
    state.save(str(path))
    loaded = ASAState.load(str(path))

    assert loaded.program_candidates == state.program_candidates
    assert loaded.parent_program_for_candidates == state.parent_program_for_candidates
    assert loaded.prog_candidate_val_subscores == state.prog_candidate_val_subscores
    assert loaded.get_pareto_front() == state.get_pareto_front()
    assert loaded.total_num_evals == state.total_num_evals
    assert loaded.i == state.i
    assert loaded.full_program_trace == state.full_program_trace


def test_json_roundtrip_cartesian_frontier(tmp_path):
    state = ASAState(
        {"m": "a"},
        {0: 0.5, 1: 0.7},
        seed_objective_scores_by_val_id={0: {"acc": 0.5, "lat": 0.3}, 1: {"acc": 0.7, "lat": 0.2}},
        frontier_type=FrontierType.CARTESIAN,
    )
    state.add_candidate(
        {"m": "b"},
        {0: 0.9, 1: 0.6},
        parent_ids=[0],
        objective_scores_by_val_id={0: {"acc": 0.9, "lat": 0.4}, 1: {"acc": 0.6, "lat": 0.1}},
    )

    path = tmp_path / "state.json"
    state.save(str(path))
    loaded = ASAState.load(str(path))

    assert loaded.get_cartesian_front() == state.get_cartesian_front()
    assert loaded.pareto_front_cartesian == state.pareto_front_cartesian
    assert loaded.frontier_type == FrontierType.CARTESIAN


def test_asa_result_from_state():
    state = ASAState({"m": "a"}, {0: 0.5, 1: 0.7})
    state.add_candidate({"m": "b"}, {0: 0.8, 1: 0.6}, parent_ids=[0])

    result = ASAResult.from_state(state)
    assert result.num_candidates == 2
    assert result.num_val_instances == 2
    assert result.best_idx == 1
    assert result.best_candidate == {"m": "b"}
    assert result.val_aggregate_scores[0] == pytest.approx(0.6)
    assert result.val_aggregate_scores[1] == pytest.approx(0.7)
    # candidate 1 uniquely wins val 0 (0.8 > 0.5); candidate 0 uniquely wins val 1 (0.7 > 0.6)
    assert result.per_val_instance_best_candidates == {0: {1}, 1: {0}}


def test_is_dominated_simple():
    front = {0: {0}, 1: {1}}
    assert is_dominated(0, {1}, front) is False  # 0 uniquely holds val 0
    assert is_dominated(1, {0}, front) is False  # 1 uniquely holds val 1


def test_remove_dominated_programs_prunes():
    # program 2 only appears on the shared front with {0,1}, so it's dominated
    # by the pair. Programs 0 and 1 each uniquely win one instance.
    front = {"a": {0}, "b": {1}, "shared": {0, 1, 2}}
    pruned = remove_dominated_programs(front)
    assert pruned == {"a": {0}, "b": {1}, "shared": {0, 1}}
