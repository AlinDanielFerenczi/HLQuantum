from hlquantum.workflows.engine import Workflow, WorkflowRunner
from hlquantum.workflows.nodes import (
    WorkflowNode, TaskNode, ParallelNode, LoopNode, ConditionalNode,
    ClassicalNode, MapNode, PipelineNode,
)

__all__ = [
    "Workflow", "WorkflowRunner",
    "WorkflowNode", "TaskNode", "ParallelNode", "LoopNode", "ConditionalNode",
    "ClassicalNode", "MapNode", "PipelineNode",
    "Parallel", "Loop", "Branch", "Classical", "Map", "Pipeline",
]

# High-level helpers
def Parallel(*nodes) -> ParallelNode:
    """Create a parallel node from circuits, callables, or workflow nodes."""
    from hlquantum.workflows.nodes import TaskNode, ClassicalNode
    processed_nodes = []
    for n in nodes:
        if not isinstance(n, WorkflowNode):
            if callable(n):
                processed_nodes.append(ClassicalNode(n))
            else:
                processed_nodes.append(TaskNode(n))
        else:
            processed_nodes.append(n)
    return ParallelNode(processed_nodes)

def Loop(body, iterations: int) -> LoopNode:
    """Create a loop node that repeats *body* for *iterations* times."""
    if not isinstance(body, WorkflowNode):
        if callable(body):
            body = ClassicalNode(body)
        else:
            body = TaskNode(body)
    return LoopNode(body, iterations)

def Branch(condition, true_node, false_node=None) -> ConditionalNode:
    """Create a conditional branch node."""
    if not isinstance(true_node, WorkflowNode):
        if callable(true_node):
            true_node = ClassicalNode(true_node)
        else:
            true_node = TaskNode(true_node)
    if false_node and not isinstance(false_node, WorkflowNode):
        if callable(false_node):
            false_node = ClassicalNode(false_node)
        else:
            false_node = TaskNode(false_node)
    return ConditionalNode(condition, true_node, false_node)

def Classical(func, node_id=None, name=None) -> ClassicalNode:
    """Convenience wrapper to create a ClassicalNode from a function."""
    return ClassicalNode(func, node_id=node_id, name=name)

def Map(func, input_key: str = "previous_result", node_id=None, name=None) -> MapNode:
    """Convenience wrapper to create a MapNode."""
    return MapNode(func, input_key=input_key, node_id=node_id, name=name)

def Pipeline(funcs, input_key: str = "previous_result", node_id=None, name=None) -> PipelineNode:
    """Convenience wrapper to create a PipelineNode from a list of functions."""
    return PipelineNode(funcs, input_key=input_key, node_id=node_id, name=name)
