import os
import time
import asyncio
from hlquantum.circuit import Circuit
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
    
    # 1. Throttling Example
    print("--- Throttling Example ---")
    runner = WorkflowRunner(backend=backend, throttling_delay=0.5) # 0.5s delay
    qc = Circuit(2).h(0).cx(0, 1).measure_all()

    print("Executing 3 circuits with 0.5s throttling...")
    start = time.time()
    for i in range(3):
        print(f" Running circuit {i+1}...")
        await runner.run_circuit(qc)
    end = time.time()
    print(f"Total time: {end-start:.2f}s (Expected >= 1.5s)")

    # 2. Save/Resume Example
    print("\n--- Save/Resume Example ---")
    STATE_FILE = "workflow_checkpoint.json"

    # Clean start
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

    print("Starting first run, might be interrupted (simulated)...")
    # We'll just run it partially or show how it works
    wf_first = Workflow(state_file=STATE_FILE)
    wf_first.add(Circuit(1).h(0).measure(0), node_id="node1")
    wf_first.add(lambda: print(" Interrupted here!"), node_id="interrupt")
    await wf_first.run(runner)

    print("\nResuming workflow...")
    # New workflow object, same state file
    wf_second = Workflow(state_file=STATE_FILE)
    wf_second.add(Circuit(1).h(0).measure(0), node_id="node1")
    wf_second.add(lambda: print(" This was already done!"), node_id="interrupt")
    wf_second.add(Circuit(1).x(0).measure(0), node_id="node2")
    await wf_second.run(runner, resume=True)

    print("\nResilience demo complete.")
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

if __name__ == "__main__":
    asyncio.run(main())
