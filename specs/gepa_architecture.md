# GEPA Architecture — Process Flow

This document maps the step-by-step process flow of a GEPA optimization run, with special attention to where the adapter boundary is crossed (i.e., what the engine delegates to the adapter vs. what it handles internally).

---

## Entry Point

`gepa.api.optimize()` is the top-level function. It:
1. Resolves the adapter (user-provided or `DefaultAdapter`)
2. Wraps trainset/valset into `DataLoader` instances
3. Constructs the `ReflectiveMutationProposer` (and optionally `MergeProposer`)
4. Constructs `GEPAEngine` with all strategy objects
5. Calls `engine.run()` → returns `GEPAResult`

---

## Initialization Phase (`engine.run()` start)

| Step | Engine Process | Adapter Call |
|------|---------------|-------------|
| 1 | `initialize_gepa_state()` — checks for saved state in `run_dir`, resumes if found | |
| 2 | If no saved state: evaluate seed candidate on **full valset** | **`adapter.evaluate(valset, seed_candidate, capture_traces=False)`** |
| 3 | Construct `GEPAState` with seed candidate scores, initialize Pareto front with seed as sole member | |
| 4 | Log base program score, fire `on_optimization_start` callback | |

---

## Main Loop (per iteration)

The engine loops `while not _should_stop(state)`. Each iteration follows this flow:

### Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  MAIN LOOP ITERATION                                                │
│                                                                     │
│  ┌─ Save state to disk ────────────────────────────────────────┐   │
│  │  state.save()                                                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                          ▼                                          │
│  ┌─ OPTIONAL: Merge Branch ────────────────────────────────────┐   │
│  │  If merge is scheduled and last iter found a new program:    │   │
│  │  → merge_proposer.propose(state)                             │   │
│  │  → Subsample eval on valset   ← adapter.evaluate (no traces) │   │
│  │  → Accept if score >= max(parents) → full eval + add         │   │
│  │  → Skip reflective mutation this iteration                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                          ▼                                          │
│  ┌─ REFLECTIVE MUTATION (main path) ───────────────────────────┐   │
│  │                                                              │   │
│  │  1. SELECT CANDIDATE from Pareto front                       │   │
│  │  2. SAMPLE MINIBATCH from trainset                           │   │
│  │  3. EVALUATE parent on minibatch ← adapter.evaluate (traces) │   │
│  │  4. SELECT COMPONENTS to update                              │   │
│  │  5. BUILD REFLECTIVE DATASET    ← adapter.make_reflective..  │   │
│  │  6. PROPOSE NEW TEXTS           ← adapter.propose_new_texts  │   │
│  │                                    OR default LM proposal     │   │
│  │  7. BUILD NEW CANDIDATE with proposed texts                  │   │
│  │  8. EVALUATE new candidate      ← adapter.evaluate (no trace)│   │
│  │  9. ACCEPT/REJECT (subsample score comparison)               │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                          ▼                                          │
│  ┌─ If accepted: Full Eval + Pareto Update ────────────────────┐   │
│  │  → Evaluate on full valset    ← adapter.evaluate (no traces) │   │
│  │  → update_state_with_new_program() — update Pareto front     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Detailed Step-by-Step

Below is each step in detail, showing what the engine does internally vs. what it delegates to the adapter.

#### Step 0: Save State & Increment Iteration

| Engine | Adapter |
|--------|---------|
| `state.save(run_dir)` — persist GEPAState to disk | — |
| `state.i += 1` — increment iteration counter | — |
| Fire `on_iteration_start` callback | — |

#### Step 1: Attempt Merge (optional, if enabled)

Only runs if `use_merge=True`, a merge is scheduled (`merges_due > 0`), and the previous iteration accepted a new candidate.

| Engine | Adapter |
|--------|---------|
| `merge_proposer.propose(state)` — find two compatible Pareto front candidates with a common ancestor | — |
| Construct merged candidate by combining predictors from each parent | — |
| Sample valset subset covering both parents' strengths | — |
| Evaluate merged candidate on subsample | **`adapter.evaluate(subsample, merged_candidate, capture_traces=False)`** |
| Compare: `sum(new_scores) >= max(sum(parent1_scores), sum(parent2_scores))` | — |
| If accepted → full eval on valset, add to state, update Pareto front | **`adapter.evaluate(valset, merged_candidate, capture_traces=False)`** |
| **Skip to next iteration** (no reflective mutation this round) | — |

#### Step 2: Select Candidate from Pareto Front

| Engine | Adapter |
|--------|---------|
| `candidate_selector.select_candidate_idx(state)` — typically `ParetoCandidateSelector` which randomly samples from Pareto front programs weighted by their scores | — |

Strategies available:
- **Pareto** (default): random sample from programs on the Pareto front
- **CurrentBest**: always pick the highest-scoring program
- **EpsilonGreedy**: with probability ε pick random, otherwise pick best

#### Step 3: Sample Training Minibatch

| Engine | Adapter |
|--------|---------|
| `batch_sampler.next_minibatch_ids(trainset, state)` — typically `EpochShuffledBatchSampler` (default minibatch_size=3) | — |
| `trainset.fetch(subsample_ids)` → concrete examples | — |

The batch sampler shuffles training IDs each epoch and walks through them in minibatch-sized windows, padding with least-seen examples if needed.

#### Step 4: Evaluate Parent Candidate on Minibatch (WITH traces)

| Engine | Adapter |
|--------|---------|
| Fire `on_evaluation_start` callback | — |
| — | **`adapter.evaluate(minibatch, parent_candidate, capture_traces=True)`** |
| — | Returns `EvaluationBatch(outputs, scores, trajectories)` |
| `state.increment_evals(len(minibatch))` | — |
| Fire `on_evaluation_end` callback | — |
| If no trajectories returned → skip this iteration (`return None`) | — |
| If all scores are perfect and `skip_perfect_score=True` → skip | — |

This is the only evaluation that captures traces — it's needed to build the reflective dataset.

#### Step 5: Select Components to Update

| Engine | Adapter |
|--------|---------|
| `module_selector(state, trajectories, scores, candidate_idx, candidate)` | — |
| Returns list of component names to update | — |

Strategies:
- **RoundRobin** (default): cycles through components one at a time
- **All**: updates every component each iteration

#### Step 6: Build Reflective Dataset

| Engine | Adapter |
|--------|---------|
| — | **`adapter.make_reflective_dataset(parent_candidate, eval_batch, components_to_update)`** |
| — | Returns `{component_name: [{Inputs: ..., Generated Outputs: ..., Feedback: ...}, ...]}` |
| Fire `on_reflective_dataset_built` callback | — |

The adapter examines the trajectories and outputs from Step 4 and constructs a concise dataset that the reflection LM can use. This is where domain-specific knowledge about the program's execution is encoded.

#### Step 7: Propose New Texts (Reflection)

The engine calls `propose_new_texts()` which resolves through a priority chain:

| Priority | Source | What Happens |
|----------|--------|-------------|
| 1st | **`adapter.propose_new_texts`** (if adapter implements it) | Adapter gets `(candidate, reflective_dataset, components_to_update)` and returns `{component: new_text}` |
| 2nd | `custom_candidate_proposer` (if user provides one) | Same signature as above |
| 3rd | Default: `InstructionProposalSignature` | For each component: build prompt from template + current instruction + reflective dataset, call `reflection_lm`, extract new instruction from response |

**Default proposal prompt structure** (InstructionProposalSignature):
```
I provided an assistant with the following instructions:
<current_instruction>

Here are examples of inputs, outputs, and feedback:
<reflective_dataset_formatted_as_markdown>

Write a new instruction for the assistant.
```

The reflection LM response is parsed to extract text within ``` blocks.

| Engine | Adapter |
|--------|---------|
| Fire `on_proposal_start` callback | — |
| Call `propose_new_texts(candidate, reflective_dataset, components_to_update)` | **`adapter.propose_new_texts(...)` if implemented** |
| Fire `on_proposal_end` callback | — |

#### Step 8: Build and Evaluate New Candidate

| Engine | Adapter |
|--------|---------|
| Copy parent candidate, replace updated component texts with proposed texts | — |
| Fire `on_evaluation_start` callback | — |
| Evaluate new candidate on the **same minibatch** (no traces needed) | **`adapter.evaluate(minibatch, new_candidate, capture_traces=False)`** |
| Fire `on_evaluation_end` callback | — |
| `state.increment_evals(count)` | — |

#### Step 9: Accept/Reject Decision (Subsample Gate)

| Engine | Adapter |
|--------|---------|
| Compare `sum(new_scores)` vs `sum(old_scores)` on the minibatch | — |
| If adapter has `outcome_reflection`: call it with both scores | **`adapter.outcome_reflection(candidate, eval_batch, old_score, new_score)`** (optional) |
| If `new_sum <= old_sum` → **REJECT**, fire `on_candidate_rejected`, go to next iteration | — |
| If `new_sum > old_sum` → **ACCEPT**, proceed to full eval | — |

#### Step 10: Full Valset Evaluation & Pareto Update (accepted candidates only)

| Engine | Adapter |
|--------|---------|
| Evaluate accepted candidate on **full validation set** | **`adapter.evaluate(valset, new_candidate, capture_traces=False)`** |
| `state.update_state_with_new_program()` — add to candidate pool, update Pareto front | — |
| Determine if this is the new best program overall | — |
| Fire `on_pareto_front_updated`, `on_valset_evaluated`, `on_candidate_accepted` callbacks | — |
| If merge is enabled: schedule a merge attempt for next iteration | — |

---

## Termination

| Step | Engine Process | Adapter Call |
|------|---------------|-------------|
| 1 | `_should_stop(state)` returns True (budget exhausted, stop file found, etc.) | — |
| 2 | Final `state.save(run_dir)` | — |
| 3 | Determine best candidate via `val_evaluation_policy.get_best_program(state)` | — |
| 4 | Fire `on_optimization_end` callback | — |
| 5 | Return `GEPAResult.from_state(state)` | — |

---

## Complete Adapter Interface Summary

These are all the points where `GEPAEngine` crosses the adapter boundary:

| Adapter Method | When Called | Capture Traces? | Purpose |
|---|---|---|---|
| `adapter.evaluate(valset, seed, False)` | Initialization | No | Baseline score for seed candidate |
| `adapter.evaluate(minibatch, parent, True)` | Step 4 — every iteration | **Yes** | Get trajectories + scores for reflection |
| `adapter.make_reflective_dataset(candidate, eval_batch, components)` | Step 6 — every iteration | — | Build reflection input for the LM |
| `adapter.propose_new_texts(candidate, dataset, components)` | Step 7 — every iteration (if implemented) | — | Custom proposal logic |
| `adapter.evaluate(minibatch, new_candidate, False)` | Step 8 — every iteration | No | Subsample gate: is new candidate better? |
| `adapter.outcome_reflection(candidate, batch, old, new)` | Step 9 — every iteration (if implemented) | — | Optional hook after scoring |
| `adapter.evaluate(valset, accepted_candidate, False)` | Step 10 — accepted only | No | Full eval for Pareto tracking |
| `adapter.evaluate(subsample, merged_candidate, False)` | Merge — when scheduled | No | Merged candidate subsample eval |
| `adapter.evaluate(valset, merged_candidate, False)` | Merge accepted | No | Full eval for merged candidate |

---

## Key Data Structures

### Candidate
```python
candidate: dict[str, str] = {
    "component_name_1": "instruction text for component 1",
    "component_name_2": "instruction text for component 2",
}
```

### EvaluationBatch (adapter returns this)
```python
EvaluationBatch(
    outputs: list[RolloutOutput],       # raw outputs, opaque to GEPA
    scores: list[float],                 # per-example scores, higher=better
    trajectories: list[Trajectory],      # per-example traces (only when capture_traces=True)
    objective_scores: list[dict[str,float]] | None,  # multi-objective (optional)
)
```

### Reflective Dataset (adapter returns this)
```python
{
    "component_name": [
        {"Inputs": {...}, "Generated Outputs": {...}, "Feedback": "..."},
        {"Inputs": {...}, "Generated Outputs": {...}, "Feedback": "..."},
    ]
}
```

### GEPAState (key fields)
```python
program_candidates: list[dict[str, str]]           # all candidates ever accepted
parent_program_for_candidate: list[list[int|None]]  # lineage tracking
prog_candidate_val_subscores: list[dict[id, float]] # per-example scores per candidate
pareto_front_valset: dict[id, float]                # best score per val example
program_at_pareto_front_valset: dict[id, set[int]]  # which candidates are on the front
```

---

## Scoring Semantics

- **Subsample gate** (accept/reject): uses `sum(scores)` on the minibatch
- **Pareto tracking**: uses per-example scores, tracking which candidate is best for each validation example
- **"Best program"**: the candidate with the highest `mean(scores)` across all evaluated validation examples
- **Merge acceptance**: `sum(merged_scores) >= max(sum(parent1_scores), sum(parent2_scores))`

---

## Strategy Injection Points

| Strategy | Interface | Default | Where Used |
|----------|-----------|---------|------------|
| Candidate selection | `CandidateSelector` | `ParetoCandidateSelector` | Step 2 |
| Minibatch sampling | `BatchSampler` | `EpochShuffledBatchSampler(size=3)` | Step 3 |
| Component selection | `ReflectionComponentSelector` | `RoundRobinReflectionComponentSelector` | Step 5 |
| Instruction proposal | `ProposalFn` or `adapter.propose_new_texts` | `InstructionProposalSignature` + `reflection_lm` | Step 7 |
| Validation eval policy | `EvaluationPolicy` | `FullEvaluationPolicy` | Step 10 |
| Stopping condition | `StopperProtocol` | `MaxMetricCallsStopper` | Loop guard |
