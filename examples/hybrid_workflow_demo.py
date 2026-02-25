"""
Hybrid Workflow Demo
~~~~~~~~~~~~~~~~~~~~

Shows how to run classical functions that process the results of a
quantum circuit inside a single workflow.  The workflow:

1. Prepares and executes a Bell-state circuit.
2. Extracts the measurement counts from the quantum result.
3. Computes the correlation (fraction of correlated outcomes 00/11).
4. Decides whether to flag the result as "entangled" based on a threshold.

Each step is a node in an HLQuantum ``Workflow``, so the entire
quantum → classical pipeline is managed, checkpointed, and visualisable
as a single unit.
"""

import asyncio

from hlquantum.circuit import Circuit
from hlquantum.backends.base import Backend
from hlquantum.result import ExecutionResult
from hlquantum.workflows import Workflow, Branch, WorkflowRunner


# ── Lightweight simulator for the demo (replace with a real backend) ────────

class DemoBackend(Backend):
    """Tiny simulator that returns plausible Bell-state counts."""

    @property
    def name(self) -> str:
        return "DemoSim"

    def run(self, circuit, shots=1000, **kwargs):
        # Simulate a noisy Bell state: ~48% |00⟩, ~48% |11⟩, ~2% each |01⟩/|10⟩
        c00 = int(shots * 0.48)
        c11 = int(shots * 0.48)
        c01 = int(shots * 0.02)
        c10 = shots - c00 - c11 - c01
        return ExecutionResult(
            counts={"00": c00, "11": c11, "01": c01, "10": c10},
            shots=shots,
            backend_name=self.name,
        )


# ── Classical post-processing functions ─────────────────────────────────────

def extract_counts(ctx):
    """Pull the counts dict out of the previous quantum result."""
    result = ctx["previous_result"]  # an ExecutionResult
    counts = result.counts
    print(f"  Measurement counts: {counts}")
    return counts


def compute_correlation(ctx):
    """Compute the fraction of correlated outcomes (|00⟩ + |11⟩)."""
    counts = ctx["previous_result"]  # dict from the previous node
    total = sum(counts.values())
    correlated = counts.get("00", 0) + counts.get("11", 0)
    ratio = correlated / total if total else 0.0
    print(f"  Correlation ratio : {ratio:.4f}")
    return ratio


def label_entangled(ctx):
    """Return a human-readable label when entanglement is detected."""
    ratio = ctx["previous_result"]
    print(f"  ✓ Entanglement detected (correlation {ratio:.2%})")
    return {"entangled": True, "correlation": ratio}


def label_not_entangled(ctx):
    """Return a label when the state does NOT look entangled."""
    ratio = ctx["previous_result"]
    print(f"  ✗ No entanglement  (correlation {ratio:.2%})")
    return {"entangled": False, "correlation": ratio}


# ── Build and run the workflow ──────────────────────────────────────────────

async def main():
    backend = DemoBackend()
    runner = WorkflowRunner(backend=backend)

    wf = Workflow(name="HybridQuantumClassical")

    # Step 1 – quantum: prepare a Bell state and measure
    bell = Circuit(2).h(0).cx(0, 1).measure_all()
    wf.add(bell, name="bell_circuit")

    # Step 2 – classical: extract counts from the ExecutionResult
    wf.add(extract_counts, name="extract_counts")

    # Step 3 – classical: compute the correlation ratio
    wf.add(compute_correlation, name="compute_correlation")

    # Step 4 – branch: decide based on the ratio (threshold = 90 %)
    wf.add(
        Branch(
            condition=lambda ctx: ctx.get("previous_result", 0) > 0.90,
            true_node=label_entangled,
            false_node=label_not_entangled,
        ),
        name="classify",
    )

    print("Running hybrid workflow …\n")
    results = await wf.run(runner=runner)

    print("\n── Results ──")
    for i, r in enumerate(results):
        print(f"  Node {i}: {r}")

    # Show the workflow as a Mermaid diagram
    print("\n── Mermaid diagram ──")
    print(wf.to_mermaid())


if __name__ == "__main__":
    asyncio.run(main())
