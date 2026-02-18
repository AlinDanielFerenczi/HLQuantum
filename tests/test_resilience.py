"""Tests for hlquantum throttling and resilience."""

import os
import json
import pytest
import time
from hlquantum.circuit import Circuit
from hlquantum.workflows import Workflow, WorkflowRunner, TaskNode
from hlquantum.backends.base import Backend
from hlquantum.result import ExecutionResult

class MockBackend(Backend):
    @property
    def name(self) -> str:
        return "MockBackend"
    
    def run(self, circuit, shots=1000, include_statevector=False, **kwargs):
        return ExecutionResult(counts={"0": shots}, shots=shots)

import asyncio

class TestResilience:
    def test_throttling(self):
        backend = MockBackend()
        runner = WorkflowRunner(backend=backend, throttling_delay=0.1)
        qc = Circuit(1).h(0)
        
        start_time = time.time()
        asyncio.run(runner.run_circuit(qc))
        end_time = time.time()
        
        assert end_time - start_time >= 0.1

    def test_state_persistence(self, tmp_path):
        state_file = tmp_path / "workflow_state.json"
        wf = Workflow(state_file=str(state_file))
        
        # Add two nodes
        wf.add(Circuit(1).h(0), node_id="node1")
        wf.add(Circuit(1).x(0), node_id="node2")
        
        # Run only the first node and mock failure/stop
        backend = MockBackend()
        runner = WorkflowRunner(backend=backend)
        asyncio.run(wf.nodes[0].execute(runner))
        wf.completed_nodes.append("node1")
        wf._save_state()
        
        # Verify state file
        with open(state_file, 'r') as f:
            data = json.load(f)
            assert "node1" in data["completed_nodes"]
            
        # Create new workflow with same state file and resume
        wf2 = Workflow(state_file=str(state_file))
        wf2.add(Circuit(1).h(0), node_id="node1")
        wf2.add(Circuit(1).x(0), node_id="node2")
        
        # Verify it skips node1
        # (We'll check if results have None for skipped nodes)
        results = asyncio.run(wf2.run(runner, resume=True))
        assert results[0] is None
        assert results[1] is not None
        assert "node2" in wf2.completed_nodes
