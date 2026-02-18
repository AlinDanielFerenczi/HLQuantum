"""
Demo of advanced hlquantum features: Visualization, Async Workflows, and Variational Ansätze.
"""

import asyncio
import time
from hlquantum.circuit import Circuit, Parameter
from hlquantum.layers import RealAmplitudes, Sequential
from hlquantum.workflows import Workflow, WorkflowRunner, Parallel
from hlquantum.backends.base import Backend
from hlquantum.result import ExecutionResult

class MockBackend(Backend):
    @property
    def name(self) -> str: return "Mock"
    def run(self, circuit, shots=1000, **kwargs):
        return ExecutionResult(counts={"0": shots}, shots=shots)

async def main():
    backend = MockBackend()
    # 1. Variational Ansatz & Parameter Binding
    print("--- 1. Variational Ansatz ---")
    ansatz = RealAmplitudes(num_qubits=2, reps=1)
    ansatz_circuit = ansatz.build()
    print(f"Ansatz Circuit with parameters:\n{ansatz_circuit}")
    print(f"Parameters: {ansatz_circuit.parameters}")
    
    # Bind some values
    values = {p: 0.1 * i for i, p in enumerate(ansatz_circuit.parameters)}
    bound_qc = ansatz_circuit.bind_parameters(values)
    print(f"\nBound Circuit (first 5 gates):\n{bound_qc.gates[:5]}")

    # 2. Async Workflow & Visualization
    print("\n--- 2. Async Workflow & Visualization ---")
    wf = Workflow(name="AdvancedSim")
    
    # Add some nodes
    wf.add(Circuit(1).h(0).measure(0), name="Preparation")
    
    # Large parallel step
    parallel_circs = [Circuit(1).rx(0, 0.1 * i).measure(0) for i in range(5)]
    wf.add(Parallel(*parallel_circs), name="ParallelScans")
    
    wf.add(Circuit(1).x(0).measure(0), name="FinalResult")

    # Show Mermaid diagram
    print("\nMermaid Diagram Definition:")
    print(wf.to_mermaid())

    # Run asynchronously
    print("\nExecuting workflow asynchronously...")
    runner = WorkflowRunner(backend=backend, throttling_delay=0.1)
    start = time.time()
    results = await wf.run(runner)
    end = time.time()
    
    print(f"\nExecution finished in {end-start:.2f}s")
    print(f"Got {len(results)} high-level results.")

if __name__ == "__main__":
    asyncio.run(main())
