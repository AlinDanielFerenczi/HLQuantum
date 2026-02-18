"""
Example demonstrating the new Quantum Pipeline and Workflow features.
"""

import asyncio
from hlquantum.circuit import Circuit
from hlquantum.layers import Sequential, CircuitLayer, GroverLayer, QFTLayer
from hlquantum.workflows import Workflow, Parallel, Loop, Branch, WorkflowRunner
from hlquantum.backends.base import Backend
from hlquantum.result import ExecutionResult

class MockBackend(Backend):
    @property
    def name(self) -> str: return "Mock"
    def run(self, circuit, shots=1000, **kwargs):
        return ExecutionResult(counts={"0": shots}, shots=shots)

async def main():
    # 1. Pipeline Example (Similar to ML Layers)
    print("--- Pipeline Example ---")
    # Create a manual circuit to wrap in a layer
    qc = Circuit(3).h(0).cx(0, 1)

    # Build a sequential model
    model = Sequential([
        CircuitLayer(qc),
        QFTLayer(3),
        GroverLayer(3, target_states=["101"])
    ])

    # Compile the model into a single circuit
    final_circuit = model.build()
    print(f"Compiled circuit has {len(final_circuit)} gates.")
    print(final_circuit)

    # 2. Workflow Example (Parallelism, Loops, Branching)
    print("\n--- Workflow Example ---")

    def check_result():
        # In a real scenario, this would check some execution metadata
        import random
        success = random.choice([True, False])
        print(f"Condition check: {'Success' if success else 'Retry'}")
        return success

    # Define a complex workflow
    wf = Workflow()

    # Add a parallel step
    wf.add(Parallel(
        Circuit(2).h(0).cx(0, 1).measure_all(),
        Circuit(2).h(1).cx(1, 0).measure_all()
    ))

    # Add a loop step
    wf.add(Loop(
        Circuit(1).x(0).measure(0),
        iterations=3
    ))

    # Add a conditional branch
    wf.add(Branch(
        condition=check_result,
        true_node=Circuit(1).h(0).measure(0),
        false_node=Circuit(1).x(0).measure(0)
    ))

    print("Executing workflow...")
    runner = WorkflowRunner(backend=MockBackend())
    results = await wf.run(runner)
    print(f"Workflow executed. Got {len(results)} top-level results.")

    print("\nVerification script complete.")

if __name__ == "__main__":
    asyncio.run(main())
