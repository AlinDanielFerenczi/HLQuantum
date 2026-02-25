"""
parallel_algorithms_demo.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run multiple built-in algorithms **in parallel** inside an HLQuantum
workflow.  This showcases:

* User-friendly algorithm aliases (``frequency_transform``,
  ``quantum_search``, ``find_hidden_pattern``, ``check_balance``)
* The :class:`Workflow` / :class:`Parallel` helpers for concurrent
  circuit execution
* Inspecting individual results from a parallel step

Requirements
------------
    pip install hlquantum
"""

import asyncio

from hlquantum.result import ExecutionResult
from hlquantum.algorithms import (
    frequency_transform,    # QFT
    quantum_search,         # Grover's search
    find_hidden_pattern,    # Bernstein-Vazirani
    check_balance,          # Deutsch-Jozsa
)
from hlquantum.algorithms.deutsch_jozsa import balanced_oracle
from hlquantum.backends.base import Backend
from hlquantum.workflows import Workflow, Parallel, WorkflowRunner

# ──────────────────────────────────────────────────────────────────────────────
# Mock backend (swap in a real backend for actual hardware execution)
# ──────────────────────────────────────────────────────────────────────────────

class MockBackend(Backend):
    """Lightweight mock that returns deterministic counts for demonstration."""

    @property
    def name(self) -> str:
        return "Mock"

    def run(self, circuit, shots=1000, **kwargs):
        # Return a plausible single-bitstring result for each circuit
        n = circuit.num_qubits
        bitstring = "0" * n
        return ExecutionResult(counts={bitstring: shots}, shots=shots)


# ──────────────────────────────────────────────────────────────────────────────
# Build circuits from the built-in algorithms
# ──────────────────────────────────────────────────────────────────────────────

def build_circuits():
    """Return a dict of named circuits produced by user-friendly algorithms."""

    # 1. Frequency transform (QFT on 3 qubits)
    qft_circuit = frequency_transform(num_qubits=3)

    # 2. Quantum search (Grover – look for |101⟩ in a 3-qubit space)
    grover_circuit = quantum_search(num_qubits=3, target_states=["101"])

    # 3. Find hidden pattern (Bernstein-Vazirani – secret "1011")
    bv_circuit = find_hidden_pattern(secret_bitstring="1011")

    # 4. Check balance (Deutsch-Jozsa with a balanced oracle on 2 qubits)
    dj_circuit = check_balance(num_qubits=2, oracle=balanced_oracle)

    return {
        "frequency_transform (QFT)": qft_circuit,
        "quantum_search (Grover)": grover_circuit,
        "find_hidden_pattern (BV)": bv_circuit,
        "check_balance (DJ)": dj_circuit,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Assemble and run the workflow
# ──────────────────────────────────────────────────────────────────────────────

async def main():
    circuits = build_circuits()

    # Print a summary of each circuit
    print("Built-in algorithm circuits")
    print("=" * 50)
    for label, circ in circuits.items():
        print(f"  {label:40s}  qubits={circ.num_qubits}  gates={len(circ)}")
    print()

    # Create a workflow that runs all four circuits in parallel
    wf = Workflow(name="ParallelAlgorithms")
    wf.add(
        Parallel(*circuits.values()),
        name="Run 4 algorithms in parallel",
    )

    # Execute the workflow
    backend = MockBackend()
    runner = WorkflowRunner(backend=backend)
    results = await wf.run(runner)

    # `results[0]` is a list of results from the parallel node
    parallel_results = results[0]

    # Display results
    print()
    print("Results")
    print("=" * 50)
    for label, res in zip(circuits.keys(), parallel_results):
        print(f"  {label}")
        print(f"    counts       : {res.counts}")
        print(f"    most probable : {res.most_probable}")
        print()

    print("All algorithms executed in parallel successfully.")


if __name__ == "__main__":
    asyncio.run(main())
