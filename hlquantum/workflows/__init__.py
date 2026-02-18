from hlquantum.workflows.engine import Workflow, WorkflowRunner
from hlquantum.workflows.nodes import WorkflowNode, TaskNode, ParallelNode, LoopNode, ConditionalNode

__all__ = ["Workflow", "WorkflowRunner", "WorkflowNode", "TaskNode", "ParallelNode", "LoopNode", "ConditionalNode"]

# High-level helpers
def Parallel(*nodes) -> ParallelNode:
    from hlquantum.workflows.nodes import TaskNode
    processed_nodes = []
    for n in nodes:
        if not isinstance(n, WorkflowNode):
            processed_nodes.append(TaskNode(n))
        else:
            processed_nodes.append(n)
    return ParallelNode(processed_nodes)

def Loop(body, iterations: int) -> LoopNode:
    if not isinstance(body, WorkflowNode):
        body = TaskNode(body)
    return LoopNode(body, iterations)

def Branch(condition, true_node, false_node=None) -> ConditionalNode:
    if not isinstance(true_node, WorkflowNode):
        true_node = TaskNode(true_node)
    if false_node and not isinstance(false_node, WorkflowNode):
        false_node = TaskNode(false_node)
    return ConditionalNode(condition, true_node, false_node)
