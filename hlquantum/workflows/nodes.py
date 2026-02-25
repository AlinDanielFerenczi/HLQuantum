"""
hlquantum.workflows.nodes
~~~~~~~~~~~~~~~~~~~~~~~~~

Nodes for defining hybrid quantum-classical workflows and pipelines.
Supports quantum circuits, classical functions, data transformations,
and mixed pipelines that chain quantum and classical steps together.
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
    async def execute(self, runner: "WorkflowRunner", context: Optional[Dict[str, Any]] = None) -> Any:
        pass


class TaskNode(WorkflowNode):
    """Executes a single circuit, layer, or callable."""

    def __init__(self, action: Union[Circuit, Layer, Callable], node_id: Optional[str] = None, name: Optional[str] = None) -> None:
        super().__init__(node_id, name)
        self.action = action

    async def execute(self, runner: "WorkflowRunner", context: Optional[Dict[str, Any]] = None) -> Any:
        if isinstance(self.action, Circuit):
            return await runner.run_circuit(self.action)
        elif isinstance(self.action, Layer):
            return await runner.run_circuit(self.action.build())
        elif callable(self.action):
            import inspect
            sig = inspect.signature(self.action)
            if sig.parameters and context is not None:
                if inspect.iscoroutinefunction(self.action):
                    return await self.action(context)
                return self.action(context)
            else:
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

    async def execute(self, runner: "WorkflowRunner", context: Optional[Dict[str, Any]] = None) -> List[Any]:
        # The runner will handle actual parallelism if supported
        return await runner.run_parallel(self.nodes, context=context)


class LoopNode(WorkflowNode):
    """Executes a node in a loop."""

    def __init__(self, body: WorkflowNode, iterations: int, node_id: Optional[str] = None, name: Optional[str] = None) -> None:
        super().__init__(node_id, name)
        self.body = body
        self.iterations = iterations

    async def execute(self, runner: "WorkflowRunner", context: Optional[Dict[str, Any]] = None) -> List[Any]:
        results = []
        ctx = context or {}
        for i in range(self.iterations):
            result = await self.body.execute(runner, context=ctx)
            results.append(result)
            ctx = {**ctx, "previous_result": result, "loop_index": i}
        return results


class ConditionalNode(WorkflowNode):
    """Executes one of two nodes based on a condition."""

    def __init__(self, condition: Callable[..., bool], true_node: WorkflowNode, false_node: Optional[WorkflowNode] = None, node_id: Optional[str] = None, name: Optional[str] = None) -> None:
        super().__init__(node_id, name)
        self.condition = condition
        self.true_node = true_node
        self.false_node = false_node

    async def execute(self, runner: "WorkflowRunner", context: Optional[Dict[str, Any]] = None) -> Any:
        import inspect
        ctx = context or {}
        sig = inspect.signature(self.condition)
        if sig.parameters:
            cond_val = await self.condition(ctx) if inspect.iscoroutinefunction(self.condition) else self.condition(ctx)
        else:
            cond_val = await self.condition() if inspect.iscoroutinefunction(self.condition) else self.condition()
        if cond_val:
            return await self.true_node.execute(runner, context=ctx)
        elif self.false_node:
            return await self.false_node.execute(runner, context=ctx)
        return None


class ClassicalNode(WorkflowNode):
    """Executes a classical (non-quantum) function within a workflow.

    The function receives the workflow context dict containing results from
    prior nodes.  It may be synchronous or async.

    Parameters
    ----------
    func : Callable
        A callable that accepts a context dict and returns any value.
    node_id : str, optional
        Unique identifier for this node.
    name : str, optional
        Human-readable name.
    """

    def __init__(self, func: Callable[..., Any], node_id: Optional[str] = None, name: Optional[str] = None) -> None:
        super().__init__(node_id, name)
        if not callable(func):
            raise TypeError(f"ClassicalNode requires a callable, got {type(func)}")
        self.func = func

    async def execute(self, runner: "WorkflowRunner", context: Optional[Dict[str, Any]] = None) -> Any:
        import inspect
        ctx = context or {}
        sig = inspect.signature(self.func)
        # Pass context if the function accepts an argument
        if sig.parameters:
            if inspect.iscoroutinefunction(self.func):
                return await self.func(ctx)
            return self.func(ctx)
        else:
            if inspect.iscoroutinefunction(self.func):
                return await self.func()
            return self.func()


class MapNode(WorkflowNode):
    """Applies a classical transformation to a specific key from the context.

    Useful for post-processing quantum results or transforming data between
    pipeline stages.

    Parameters
    ----------
    func : Callable
        A function that takes a single value and returns a transformed value.
    input_key : str
        The context key whose value will be passed to *func*.
    node_id : str, optional
        Unique identifier.
    name : str, optional
        Human-readable name.
    """

    def __init__(self, func: Callable[[Any], Any], input_key: str = "previous_result", node_id: Optional[str] = None, name: Optional[str] = None) -> None:
        super().__init__(node_id, name)
        if not callable(func):
            raise TypeError(f"MapNode requires a callable, got {type(func)}")
        self.func = func
        self.input_key = input_key

    async def execute(self, runner: "WorkflowRunner", context: Optional[Dict[str, Any]] = None) -> Any:
        import inspect
        ctx = context or {}
        value = ctx.get(self.input_key)
        if inspect.iscoroutinefunction(self.func):
            return await self.func(value)
        return self.func(value)


class PipelineNode(WorkflowNode):
    """Chains multiple classical functions into a single node.

    Each function receives the output of the previous one, forming a
    classical processing pipeline.  The first function in the chain
    receives the value from *input_key* in the workflow context.

    Parameters
    ----------
    funcs : list of Callable
        Ordered list of functions to chain.
    input_key : str
        Context key to seed the first function.
    node_id : str, optional
        Unique identifier.
    name : str, optional
        Human-readable name.
    """

    def __init__(self, funcs: List[Callable], input_key: str = "previous_result", node_id: Optional[str] = None, name: Optional[str] = None) -> None:
        super().__init__(node_id, name)
        for f in funcs:
            if not callable(f):
                raise TypeError(f"PipelineNode requires callables, got {type(f)}")
        self.funcs = funcs
        self.input_key = input_key

    async def execute(self, runner: "WorkflowRunner", context: Optional[Dict[str, Any]] = None) -> Any:
        import inspect
        ctx = context or {}
        value = ctx.get(self.input_key)
        for func in self.funcs:
            if inspect.iscoroutinefunction(func):
                value = await func(value)
            else:
                value = func(value)
        return value
