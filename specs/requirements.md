# ASA Requirements

## Purpose

ASA (Artificial Selection Algorithm) is a thin Python state-tracking library for agent-directed candidate evolution. It maintains a candidate pool, per-row validation subscores, multi-objective scores, and Pareto frontiers. It does **not** choose parents, run reflection, evaluate candidates, or orchestrate iteration — the caller (typically a coding agent in CodeEvolver) drives all of those decisions and uses ASA purely as a state substrate.

ASA is a refactor of the upstream GEPA library (github.com/gepa-ai/gepa), which originally bundled state tracking, an optimization engine, reflective mutation proposers, adapters, batch samplers, and evaluation policies. ASA keeps only the state layer. MIT license attribution to Lakshya A Agrawal and the GEPA contributors is retained.

## Design rationale

1. **Agent-directed selection beats random Pareto sampling for coding.** Unlike prompt tuning, coding has strong structural priors an agent can exploit — it can read two candidates, reason about which traits compose, and pick parents like an artificial breeder rather than sampling uniformly from a frontier. The library therefore exposes frontiers for the agent to *read*, not a selector to *call*.
2. **The Pareto frontier is valuable as memory, not as a sampling mechanism.** An agent choosing parents from conversational memory is prone to diversity collapse — anchoring on recent successes and forgetting candidates that uniquely solve niche val instances or niche objectives. The incrementally-maintained frontier is a cheap, structural guarantee against that forgetting and answers "what are we at risk of losing?" without the agent having to simulate it.
3. **Merges are delegated to the agent.** Cross-candidate merging is subsumed by the agent's own ability to inspect two programs and compose their traits. The upstream merge proposer and its scheduling machinery are removed.
4. **Pool entry is unconditional.** Upstream GEPA required subsample improvement before adding a candidate to the pool. In ASA the caller decides whether to add a candidate — ASA never rejects. Frontier membership (not pool membership) is what determines whether a candidate is "still winning somewhere."

## Scope

**In scope**
- Candidate pool and genealogy tracking.
- Per-row validation subscores (`prog_candidate_val_subscores`).
- Multi-objective aggregate scores (`prog_candidate_objective_scores`).
- Incremental Pareto frontier maintenance for four frontier types: instance, objective, hybrid, cartesian.
- Memory-substrate read helpers for the agent (frontier views, unique wins, best program).
- Budget counter and iteration audit trace.
- JSON persistence (save/load).
- Result export (`ASAResult`).
- Dominance utilities (`is_dominated`, `remove_dominated_programs`) for agents that want to prune frontiers before inspection.

**Out of scope (removed from upstream)**
- Optimization engine / iteration loop.
- Reflective mutation proposer and instruction-proposal prompt templates.
- Merge proposer and merge scheduling.
- `GEPAAdapter` protocol and all adapter implementations (DSPy, RAG, MCP, etc.).
- Batch samplers and evaluation policies.
- Data loaders and evaluation cache.
- Stop conditions, budget hooks, callback event system.
- Experiment trackers (mlflow, wandb).
- Parent selection logic (Pareto selector, epsilon-greedy, current-best).

## Package layout

```
src/asa/
  __init__.py       # public API re-exports
  state.py          # ASAState + incremental frontier maintenance + JSON serde
  frontier.py       # FrontierType enum, is_dominated, remove_dominated_programs
  result.py         # ASAResult export
  logging.py        # LoggerProtocol shim
  py.typed
```

## Public API

```python
import asa
from asa import ASAState, ASAResult, FrontierType
```

### Construction

```python
state = ASAState(
    seed_candidate: Mapping[str, str],
    seed_val_scores: Mapping[DataId, float],
    seed_objective_scores_by_val_id: Mapping[DataId, dict[str, float]] | None = None,
    seed_outputs_by_val_id: Mapping[DataId, Any] | None = None,
    frontier_type: FrontierType = FrontierType.INSTANCE,
    track_best_outputs: bool = False,
)
```

Non-instance frontier types require `seed_objective_scores_by_val_id`.

### Mutation

```python
idx: int = state.add_candidate(
    candidate: Mapping[str, str],
    val_scores_by_id: Mapping[DataId, float],
    parent_ids: list[int],
    objective_scores_by_val_id: Mapping[DataId, dict[str, float]] | None = None,
    outputs_by_val_id: Mapping[DataId, Any] | None = None,
) -> int

state.increment_evals(count: int) -> None
state.record_iteration(metadata: Mapping[str, Any]) -> None
```

`add_candidate` always appends; all frontiers are updated incrementally. Objective aggregates across val instances are computed internally as arithmetic means.

### Memory-substrate reads

```python
state.get_pareto_front()          -> dict[DataId, set[int]]              # instance front
state.get_objective_front()       -> dict[str, set[int]]                 # objective front
state.get_cartesian_front()       -> dict[tuple[DataId, str], set[int]]  # cartesian front
state.get_frontier_members()      -> set[int]                            # union of all frontier members
state.get_unique_wins(idx)        -> set[DataId]                         # instances only this candidate wins
state.get_unique_objective_wins(idx) -> set[str]                         # objectives only this candidate wins
state.get_program_average_val_score(idx) -> float
state.get_best_program()          -> int                                 # argmax of average val score
```

### Raw fields

```python
state.program_candidates              # list[dict[str, str]]
state.parent_program_for_candidates   # list[list[int]]
state.prog_candidate_val_subscores    # list[dict[DataId, float]]
state.prog_candidate_objective_scores # list[dict[str, float]]
state.num_metric_calls_by_discovery   # list[int]
state.total_num_evals                 # int
state.i                               # int (iteration counter; -1 at init)
state.full_program_trace              # list[dict[str, Any]]
state.frontier_type                   # FrontierType
state.best_outputs_valset             # optional dict[DataId, list[tuple[int, Any]]]
```

### Persistence and export

```python
state.save(path: str)                      # JSON
ASAState.load(path: str) -> ASAState       # JSON

ASAResult.from_state(state) -> ASAResult   # immutable snapshot
```

JSON serialization rules:
- Dict keys (`DataId`) are coerced to strings on write and back to int on read when they round-trip as digit strings.
- Sets are encoded as sorted lists.
- Cartesian tuple keys `(val_id, objective)` are encoded as three-element lists `[val_id, objective, value]`.
- Arbitrary `best_outputs_valset` values pass through JSON with a `default=str` fallback.
- Schema version is pinned at `ASAState._SCHEMA_VERSION = 1`; loads reject mismatched versions.

### Dominance utilities

```python
from asa.frontier import is_dominated, remove_dominated_programs

is_dominated(y, programs, front) -> bool
remove_dominated_programs(front, scores=None) -> dict[key, set[int]]
```

Useful when the agent wants to prune a frontier view before inspecting it.

## Intended caller loop (CodeEvolver)

```python
state = asa.ASAState(seed, seed_val_scores, seed_objective_scores_by_val_id)
while not done:
    parent_idx = agent.choose_parent(state)  # reads frontier as memory substrate
    candidate, val_scores, obj_scores = agent.evolve(
        parent=state.program_candidates[parent_idx],
        state=state,
    )
    if candidate is not None:
        state.add_candidate(
            candidate,
            val_scores,
            parent_ids=[parent_idx],
            objective_scores_by_val_id=obj_scores,
        )
    state.record_iteration({"accepted": candidate is not None})
state.save("run/state.json")
```

Everything between `choose_parent` and `add_candidate` — reflection, subsample evaluation, retries, validation, accept/reject — belongs to the agent.

## Non-functional requirements

- **Language target:** Python 3.10+, typed with `pyright` (0 errors required).
- **Lint/format:** `ruff` (line length 120, double quotes, no relative imports).
- **Tests:** `pytest`; state + frontier unit tests cover seed init, incremental front updates across all four frontier types, memory-substrate reads, JSON round-trip, and dominance pruning.
- **Line budget:** `src/asa/` stays under ~900 LOC across the five files.
- **Dependencies:** no runtime dependencies beyond the Python standard library.
- **License:** MIT; upstream GEPA copyright attribution preserved in every source file header.

## Verification

```bash
pytest tests/           # unit tests
ruff check src/         # lint
ruff format --check src/
pyright src/            # type check
```

Smoke test — construct an `ASAState` with multi-objective seed, add a dominating candidate and a uniquely-winning candidate, confirm frontier views and `get_best_program` behave as expected, and round-trip through JSON.
