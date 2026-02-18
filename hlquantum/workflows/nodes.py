"""
hlquantum.workflows.nodes
~~~~~~~~~~~~~~~~~~~~~~~~~

Nodes for defining quantum workflows and state machines.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
from hlquantum.circuit import Circuit
from hlquantum.layers.base import Layer


class WorkflowNode(ABC):
    """Base class for a node in a workflow."""

    def __init__(self, node_id: Optional[str] = None, name: Optional[str] = None) -> None:
        self.node_id = node_id or f"{self.__class__.__name__}_{id(self)}"
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def execute(self, runner: "WorkflowRunner") -> Any:
        pass


class TaskNode(WorkflowNode):
    """Executes a single circuit or layer."""

    def __init__(self, action: Union[Circuit, Layer, Callable], node_id: Optional[str] = None, name: Optional[str] = None) -> None:
        super().__init__(node_id, name)
        self.action = action

    async def execute(self, runner: "WorkflowRunner") -> Any:
        if isinstance(self.action, Circuit):
            return await runner.run_circuit(self.action)
        elif isinstance(self.action, Layer):
            return await runner.run_circuit(self.action.build())
        elif callable(self.action):
            import inspect
            if inspect.iscoroutinefunction(self.action):
                return await self.action()
            return self.action()
        else:
            raise TypeError(f"Unsupported action type: {type(self.action)}")


class ParallelNode(WorkflowNode):
    """Executes multiple nodes in parallel."""

    def __init__(self, nodes: List[WorkflowNode], node_id: Optional[str] = None, name: Optional[str] = None) -> None:
        super().__init__(node_id, name)
        self.nodes = nodes

    async def execute(self, runner: "WorkflowRunner") -> List[Any]:
        # The runner will handle actual parallelism if supported
        return await runner.run_parallel(self.nodes)


class LoopNode(WorkflowNode):
    """Executes a node in a loop."""

    def __init__(self, body: WorkflowNode, iterations: int, node_id: Optional[str] = None, name: Optional[str] = None) -> None:
        super().__init__(node_id, name)
        self.body = body
        self.iterations = iterations

    async def execute(self, runner: "WorkflowRunner") -> List[Any]:
        results = []
        for _ in range(self.iterations):
            results.append(await self.body.execute(runner))
        return results


class ConditionalNode(WorkflowNode):
    """Executes one of two nodes based on a condition."""

    def __init__(self, condition: Callable[[], bool], true_node: WorkflowNode, false_node: Optional[WorkflowNode] = None, node_id: Optional[str] = None, name: Optional[str] = None) -> None:
        super().__init__(node_id, name)
        self.condition = condition
        self.true_node = true_node
        self.false_node = false_node

    async def execute(self, runner: "WorkflowRunner") -> Any:
        import inspect
        cond_val = await self.condition() if inspect.iscoroutinefunction(self.condition) else self.condition()
        if cond_val:
            return await self.true_node.execute(runner)
        elif self.false_node:
            return await self.false_node.execute(runner)
        return None
