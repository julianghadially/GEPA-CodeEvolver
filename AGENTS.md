# ASA — Artificial Selection Algorithm

ASA is a thin Python state-tracking library for agent-directed candidate evolution. It maintains a candidate pool, per-row validation subscores, multi-objective scores, and Pareto frontiers — but does not choose parents, run reflection, or orchestrate iteration. The caller (typically an agent) drives those decisions and reads the frontier as a memory substrate.

This is a refactor of the upstream GEPA library (github.com/gepa-ai/gepa). Copyright for the original code remains with Lakshya A Agrawal and the GEPA contributors (MIT license).

## Setup

```bash
pip install -e .
```

Python 3.10+. Build backend: setuptools.

## Project Structure

- `src/asa/` — main package source
  - `state.py` — `ASAState`: candidate pool, val subscores, objective scores, frontier maintenance, JSON persistence
  - `frontier.py` — `FrontierType` enum, `is_dominated`, `remove_dominated_programs`
  - `result.py` — `ASAResult` immutable export
  - `logging.py` — `LoggerProtocol` shim
- `tests/` — pytest test suite
- `specs/` — design documents

## Public API

```python
import asa

state = asa.ASAState(
    seed_candidate={"module_1": "..."},
    seed_val_scores={0: 0.5, 1: 0.7},
    frontier_type=asa.FrontierType.INSTANCE,  # or OBJECTIVE / HYBRID / CARTESIAN
)

new_idx = state.add_candidate(
    candidate={"module_1": "..."},
    val_scores_by_id={0: 0.8, 1: 0.6},
    parent_ids=[0],
)

# Memory-substrate reads
state.get_pareto_front()          # {val_id: {prog_idx, ...}}
state.get_objective_front()       # {objective: {prog_idx, ...}}
state.get_frontier_members()      # set of prog_idx still winning somewhere
state.get_unique_wins(idx)        # val_ids only this candidate wins
state.get_best_program()          # highest average val score

state.save("run/state.json")
asa.ASAState.load("run/state.json")
asa.ASAResult.from_state(state)
```

See `specs/` for the architectural rationale: why ASA replaced the upstream GEPA engine/proposer/adapter system with a thin state library, and why the Pareto frontier is retained as agent memory rather than as a sampling mechanism.

## Build & Test

```bash
pytest tests/
ruff check src/
ruff format src/
pyright src/
```

## Code Style

- Linter/formatter: ruff (line length 120, double quotes, space indent)
- Type checking: pyright
- Python target: 3.10+
- No relative imports (enforced by ruff)
