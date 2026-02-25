# Workflows API Reference

HLQuantum workflows let you orchestrate multi-step quantum–classical pipelines with branching, looping, parallelism, and automatic checkpointing.

## Key Concepts

| Symbol            | Purpose                                                             |
| ----------------- | ------------------------------------------------------------------- |
| `Workflow`        | Ordered collection of nodes; runs async, supports save/resume.      |
| `WorkflowRunner`  | Executes circuits and classical functions with optional throttling. |
| `TaskNode`        | Runs a circuit, layer, or callable.                                 |
| `ClassicalNode`   | Runs a classical function, receiving the workflow context dict.     |
| `MapNode`         | Applies a transform to a single context key.                        |
| `PipelineNode`    | Chains multiple classical functions sequentially.                   |
| `ParallelNode`    | Executes multiple nodes concurrently via `asyncio.gather`.          |
| `LoopNode`        | Repeats a node _N_ times, threading context between iterations.     |
| `ConditionalNode` | Branches on a condition callable (if/else).                         |

Convenience factory functions — `Parallel()`, `Loop()`, `Branch()`, `Classical()`, `Map()`, `Pipeline()` — wrap the node classes with automatic type detection.

## Context Propagation

Each node result is stored in the workflow context under two keys:

- `"previous_result"` — always the most recent result.
- `"result_<node_id>"` — keyed by node ID for targeted access.

Classical functions receive the full context dict as their first argument.

## Quick Example — Hybrid Quantum → Classical

```python
import asyncio
from hlquantum.circuit import Circuit
from hlquantum.workflows import Workflow, Branch, WorkflowRunner

wf = Workflow(name="HybridPipeline")

# Step 1 — quantum circuit
wf.add(Circuit(2).h(0).cx(0, 1).measure_all(), name="bell")

# Step 2 — classical: extract counts
wf.add(lambda ctx: ctx["previous_result"].counts, name="extract")

# Step 3 — classical: compute correlation
def correlation(ctx):
    counts = ctx["previous_result"]
    total = sum(counts.values())
    return (counts.get("00", 0) + counts.get("11", 0)) / total

wf.add(correlation, name="correlate")

# Step 4 — branch
wf.add(Branch(
    lambda ctx: ctx["previous_result"] > 0.9,
    lambda ctx: "entangled",
    lambda ctx: "not entangled",
), name="classify")

results = asyncio.run(wf.run())
```

## Parallelism & Loops

```python
from hlquantum.workflows import Parallel, Loop

wf = Workflow()
wf.add(Parallel(circuit_a, circuit_b))  # run concurrently
wf.add(Loop(circuit_c, iterations=5))   # repeat 5 times
```

## Save & Resume

```python
wf = Workflow(state_file="checkpoint.json")
wf.add(circuit, name="expensive_step")

# First run — saves progress after each node
results = asyncio.run(wf.run())

# Later — skips already-completed nodes
results = asyncio.run(wf.run(resume=True))
```

## Mermaid Visualisation

```python
print(wf.to_mermaid())
# graph TD
#     N0[bell (Task)]
#     N1[extract (Clas)]
#     N0 --> N1
#     ...
```

## Throttling

```python
runner = WorkflowRunner(backend=my_backend, throttling_delay=0.5)
results = asyncio.run(wf.run(runner=runner))
```

---

## Engine

::: hlquantum.workflows.engine

## Nodes

::: hlquantum.workflows.nodes
