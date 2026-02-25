from __future__ import annotations
import time
import json
import os
import asyncio
from typing import Any, List, Optional, Dict, Union
from hlquantum.circuit import Circuit
from hlquantum.runner import run as hl_run
from hlquantum.workflows.nodes import WorkflowNode, TaskNode, ClassicalNode


class WorkflowRunner:
    """Handles execution of workflow nodes with optional throttling and concurrency.

    Supports both quantum circuit execution and classical function execution.
    """

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

    async def run_classical(self, func, *args, **kwargs) -> Any:
        """Executes a classical function asynchronously.

        If the function is a coroutine, it is awaited directly.
        Otherwise it is run in a thread-pool executor so the event loop
        is not blocked by long-running computations.
        """
        import inspect
        if self.throttling_delay > 0:
            await asyncio.sleep(self.throttling_delay)
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def run_parallel(self, nodes: List[WorkflowNode], context: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Executes nodes in parallel using asyncio.gather."""
        ctx = context or {}
        tasks = [node.execute(self, context=ctx) for node in nodes]
        return await asyncio.gather(*tasks)


class Workflow:
    """A collection of nodes forming a quantum state machine or execution flow."""

    def __init__(self, state_file: Optional[str] = None, name: str = "HybridWorkflow") -> None:
        self.nodes: List[WorkflowNode] = []
        self.state_file = state_file
        self.name = name
        self.completed_nodes: List[str] = []
        self.context: Dict[str, Any] = {}
        
        if self.state_file and os.path.exists(self.state_file):
            self._load_state()

    def add(self, action: Any, node_id: Optional[str] = None, name: Optional[str] = None) -> Workflow:
        """Add a node to the workflow.

        *action* can be a ``WorkflowNode``, a ``Circuit``, a ``Layer``,
        or any callable (classical function).  Callables that are not
        Circuits or Layers are automatically wrapped in a ``ClassicalNode``
        if they accept a dict argument, otherwise in a ``TaskNode``.
        """
        if not isinstance(action, WorkflowNode):
            if isinstance(action, (Circuit,)):
                action = TaskNode(action, node_id=node_id, name=name)
            elif callable(action):
                # Wrap plain callables as ClassicalNode for first-class support
                action = ClassicalNode(action, node_id=node_id, name=name)
            else:
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
        """Execute all nodes in sequence, propagating context between them.

        Each node result is stored in ``self.context`` under two keys:
        * ``"previous_result"`` — always the most recent result
        * ``"result_<node_id>"`` — keyed by node id for targeted access
        """
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
                node_type = type(node).__name__
                print(f"[{i+1}/{total}] Running: {node.name} ({node_type})...")
            
            result = await node.execute(runner, context=self.context)
            results.append(result)
            
            # Update context so downstream nodes can access prior results
            self.context["previous_result"] = result
            self.context[f"result_{node.node_id}"] = result
            self.context["results"] = results
            
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
