"""Tests for FullEvaluationPolicy with max_val_set_size sampling."""

import random

import pytest

from gepa.core.data_loader import ListDataLoader
from gepa.core.state import GEPAState
from gepa.strategies.eval_policy import FullEvaluationPolicy


@pytest.fixture()
def loader_20():
    """ListDataLoader with 20 items keyed 0..19."""
    return ListDataLoader(list(range(20)))


@pytest.fixture()
def dummy_state():
    """Minimal GEPAState stub — only needs to exist for the protocol signature."""

    class _Stub:
        i = 0

    return _Stub()  # type: ignore[return-value]


class TestFullEvaluationPolicyDefaults:
    def test_returns_all_ids_when_no_max(self, loader_20, dummy_state):
        policy = FullEvaluationPolicy()
        ids = policy.get_eval_batch(loader_20, dummy_state)
        assert ids == list(range(20))

    def test_returns_all_ids_when_max_exceeds_size(self, loader_20, dummy_state):
        policy = FullEvaluationPolicy(max_val_set_size=100, rng=random.Random(0))
        ids = policy.get_eval_batch(loader_20, dummy_state)
        assert sorted(ids) == list(range(20))

    def test_returns_all_ids_when_max_equals_size(self, loader_20, dummy_state):
        policy = FullEvaluationPolicy(max_val_set_size=20, rng=random.Random(0))
        ids = policy.get_eval_batch(loader_20, dummy_state)
        assert sorted(ids) == list(range(20))


class TestFullEvaluationPolicySampling:
    def test_returns_correct_count(self, loader_20, dummy_state):
        policy = FullEvaluationPolicy(max_val_set_size=5, rng=random.Random(42))
        ids = policy.get_eval_batch(loader_20, dummy_state)
        assert len(ids) == 5
        assert all(0 <= i < 20 for i in ids)
        assert len(set(ids)) == 5  # no duplicates

    def test_different_samples_each_call(self, loader_20, dummy_state):
        policy = FullEvaluationPolicy(max_val_set_size=5, rng=random.Random(42))
        ids1 = policy.get_eval_batch(loader_20, dummy_state)
        ids2 = policy.get_eval_batch(loader_20, dummy_state)
        # With 20 choose 5, two consecutive samples from the same RNG are extremely
        # unlikely to be identical.
        assert ids1 != ids2

    def test_reproducible_with_same_seed(self, loader_20, dummy_state):
        ids1 = FullEvaluationPolicy(max_val_set_size=5, rng=random.Random(99)).get_eval_batch(
            loader_20, dummy_state
        )
        ids2 = FullEvaluationPolicy(max_val_set_size=5, rng=random.Random(99)).get_eval_batch(
            loader_20, dummy_state
        )
        assert ids1 == ids2

    def test_builds_coverage_over_multiple_calls(self, loader_20, dummy_state):
        policy = FullEvaluationPolicy(max_val_set_size=5, rng=random.Random(7))
        seen: set[int] = set()
        for _ in range(20):
            ids = policy.get_eval_batch(loader_20, dummy_state)
            seen.update(ids)
        # After 20 draws of 5 from 20, we should have covered all IDs
        assert seen == set(range(20))


class TestOptimizeMaxValSetSizeValidation:
    def test_zero_raises(self):
        from gepa.api import optimize

        with pytest.raises(ValueError, match="max_val_set_size must be a positive integer"):
            optimize(
                seed_candidate={"comp": "hello"},
                trainset=[{"input": "x", "output": "y"}],
                valset=[{"input": "x", "output": "y"}],
                max_val_set_size=0,
                max_metric_calls=1,
            )

    def test_negative_raises(self):
        from gepa.api import optimize

        with pytest.raises(ValueError, match="max_val_set_size must be a positive integer"):
            optimize(
                seed_candidate={"comp": "hello"},
                trainset=[{"input": "x", "output": "y"}],
                valset=[{"input": "x", "output": "y"}],
                max_val_set_size=-5,
                max_metric_calls=1,
            )

    def test_conflict_with_custom_policy_raises(self):
        from gepa.api import optimize

        with pytest.raises(ValueError, match="cannot be used together"):
            optimize(
                seed_candidate={"comp": "hello"},
                trainset=[{"input": "x", "output": "y"}],
                valset=[{"input": "x", "output": "y"}],
                max_val_set_size=5,
                val_evaluation_policy=FullEvaluationPolicy(),
                max_metric_calls=1,
            )
