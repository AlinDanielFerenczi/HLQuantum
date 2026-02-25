"""
State-Machine Loop Demo
~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrates a workflow that behaves like a state machine: a quantum
circuit is executed inside a loop, and an asynchronous classical function
inspects each iteration's results to decide when to stop.

The pipeline:

1. **Initialise** — set the target fidelity threshold.
2. **Loop** — run a noisy Bell-state circuit repeatedly (up to N times).
   Each iteration checks whether the measured correlation has reached
   the target fidelity.  If it has, the loop records convergence.
3. **Report** — summarise how many iterations it took to converge.

Everything runs asynchronously via ``asyncio.run()``.
"""

import asyncio
import random

from hlquantum.circuit import Circuit
from hlquantum.backends.base import Backend
from hlquantum.result import ExecutionResult
from hlquantum.workflows import Workflow, Loop, WorkflowRunner


# ── Noisy simulator backend ────────────────────────────────────────────────

class NoisyBellBackend(Backend):
    """Simulates a Bell state with variable noise that improves over time.

    On each call the noise shrinks slightly, mimicking a calibration loop
    where hardware quality improves as the system warms up.
    """

    def __init__(self):
        self._call_count = 0

    @property
    def name(self) -> str:
        return "NoisyBellSim"

    def run(self, circuit, shots=1000, **kwargs):
        self._call_count += 1
        # Noise drops from ~20 % down toward ~2 % over successive calls
        noise = max(0.02, 0.20 - 0.03 * self._call_count + random.uniform(-0.02, 0.02))
        correlated = int(shots * (1 - noise))
        erroneous = shots - correlated
        c00 = correlated // 2
        c11 = correlated - c00
        c01 = erroneous // 2
        c10 = erroneous - c01
        return ExecutionResult(
            counts={"00": c00, "11": c11, "01": c01, "10": c10},
            shots=shots,
            backend_name=self.name,
        )


# ── Async classical helpers ─────────────────────────────────────────────────

async def initialise(ctx):
    """Set the convergence target in the workflow context."""
    target = 0.95
    print(f"  Target fidelity: {target:.0%}")
    return {"target_fidelity": target, "converged": False, "history": []}


async def evaluate_iteration(ctx):
    """Compute correlation from the loop body's quantum result and check convergence."""
    # The loop body returns an ExecutionResult
    result = ctx.get("previous_result")

    # Inside a Loop the body result may be wrapped in a list; unwrap if needed
    if isinstance(result, list):
        result = result[-1]

    counts = result.counts if isinstance(result, ExecutionResult) else result
    total = sum(counts.values())
    correlated = counts.get("00", 0) + counts.get("11", 0)
    fidelity = correlated / total if total else 0.0

    # Retrieve state carried forward through context
    state = ctx.get("result_init", {})
    target = state.get("target_fidelity", 0.95)
    history = list(ctx.get("history", []))
    history.append(fidelity)

    iteration = len(history)
    converged = fidelity >= target
    symbol = "✓" if converged else "…"
    print(f"  Iteration {iteration}: fidelity = {fidelity:.4f}  {symbol}")

    # Simulate a short async calibration delay
    await asyncio.sleep(0.05)

    return {
        "fidelity": fidelity,
        "converged": converged,
        "history": history,
        "iteration": iteration,
    }


async def report(ctx):
    """Print a summary of the convergence loop."""
    state = ctx.get("previous_result", {})
    history = state.get("history", [])
    converged = state.get("converged", False)
    print()
    if converged:
        print(f"  Converged after {len(history)} iteration(s).")
    else:
        print(f"  Did NOT converge within {len(history)} iterations.")
    print(f"  Fidelity history: {['%.4f' % f for f in history]}")
    return {"history": history, "converged": converged}


# ── Build and run the state-machine workflow ────────────────────────────────

async def main():
    backend = NoisyBellBackend()
    runner = WorkflowRunner(backend=backend)

    bell = Circuit(2).h(0).cx(0, 1).measure_all()

    wf = Workflow(name="CalibrationStateMachine")

    # State 1 — initialise parameters
    wf.add(initialise, name="init")

    # State 2 — loop: run the quantum circuit then evaluate
    # The Loop node repeats its body N times; each iteration's result
    # feeds into the next via the context.
    max_iterations = 8
    wf.add(Loop(bell, iterations=max_iterations), name="calibration_loop")

    # State 3 — evaluate the final iteration
    wf.add(evaluate_iteration, name="evaluate")

    # State 4 — report
    wf.add(report, name="report")

    print("Running calibration state machine …\n")
    results = await wf.run(runner=runner)

    # Mermaid diagram of the state machine
    print("\n── State-machine diagram (Mermaid) ──")
    print(wf.to_mermaid())


if __name__ == "__main__":
    asyncio.run(main())
