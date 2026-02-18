from __future__ import annotations
import time
import json
import os
import asyncio
from typing import Any, List, Optional, Dict, Union
from hlquantum.circuit import Circuit
from hlquantum.runner import run as hl_run
from hlquantum.workflows.nodes import WorkflowNode, TaskNode


class WorkflowRunner:
    """Handles execution of workflow nodes with optional throttling and concurrency."""

    def __init__(self, backend=None, throttling_delay: float = 0.0) -> None:
        self.backend = backend
        self.throttling_delay = throttling_delay

    async def run_circuit(self, circuit: Circuit) -> Any:
        """Executes a single circuit asynchronously."""
        if self.throttling_delay > 0:
            await asyncio.sleep(self.throttling_delay)
        
        # hl_run is likely synchronous, so we run it in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: hl_run(circuit, backend=self.backend))

    async def run_parallel(self, nodes: List[WorkflowNode]) -> List[Any]:
        """Executes nodes in parallel using asyncio.gather."""
        tasks = [node.execute(self) for node in nodes]
        return await asyncio.gather(*tasks)


class Workflow:
    """A collection of nodes forming a quantum state machine or execution flow."""

    def __init__(self, state_file: Optional[str] = None, name: str = "QuantumWorkflow") -> None:
        self.nodes: List[WorkflowNode] = []
        self.state_file = state_file
        self.name = name
        self.completed_nodes: List[str] = []
        
        if self.state_file and os.path.exists(self.state_file):
            self._load_state()

    def add(self, action: Any, node_id: Optional[str] = None, name: Optional[str] = None) -> Workflow:
        if not isinstance(action, WorkflowNode):
            action = TaskNode(action, node_id=node_id, name=name)
        elif node_id or name:
            if node_id: action.node_id = node_id
            if name: action.name = name
        self.nodes.append(action)
        return self

    def _save_state(self) -> None:
        if self.state_file:
            with open(self.state_file, 'w') as f:
                json.dump({"completed_nodes": self.completed_nodes}, f)

    def _load_state(self) -> None:
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                self.completed_nodes = data.get("completed_nodes", [])
        except (json.JSONDecodeError, IOError):
            pass

    async def run(self, runner: Optional[WorkflowRunner] = None, resume: bool = False, verbose: bool = True) -> List[Any]:
        if runner is None:
            runner = WorkflowRunner()
        
        results = []
        total = len(self.nodes)
        
        if verbose:
            print(f"Starting Workflow: {self.name} [{total} nodes]")

        for i, node in enumerate(self.nodes):
            if resume and node.node_id in self.completed_nodes:
                if verbose:
                    print(f"[{i+1}/{total}] Skipping: {node.name} (id: {node.node_id[:8]}...)")
                results.append(None)
                continue
                
            if verbose:
                print(f"[{i+1}/{total}] Running: {node.name}...")
            
            result = await node.execute(runner)
            results.append(result)
            
            self.completed_nodes.append(node.node_id)
            self._save_state()
            
        if verbose:
            print(f"Workflow {self.name} completed successfully.")
            
        return results

    def to_mermaid(self) -> str:
        """Generate a Mermaid diagram definition for the workflow."""
        from hlquantum.workflows.nodes import ParallelNode
        lines = ["graph TD"]
        for i, node in enumerate(self.nodes):
            node_id = f"N{i}"
            label = f"{node.name} ({node.node_id[:4]})"
            
            if isinstance(node, ParallelNode):
                lines.append(f"    subgraph SUB{i} [{label}]")
                for j, subnode in enumerate(node.nodes):
                    lines.append(f"        {node_id}_{j}[{subnode.name}]")
                lines.append("    end")
            else:
                lines.append(f"    {node_id}[{label}]")
                
            if i > 0:
                prev_id = f"N{i-1}"
                lines.append(f"    {prev_id} --> {node_id}")
        
        return "\n".join(lines)
