"""Tests for hlquantum layers and workflows."""

import asyncio
import pytest
from hlquantum.circuit import Circuit
from hlquantum.layers import Sequential, CircuitLayer, GroverLayer, QFTLayer
from hlquantum.workflows import (
    Workflow, Parallel, Loop, Branch, Classical, Map, Pipeline,
    ClassicalNode, MapNode, PipelineNode, WorkflowRunner,
)

class TestLayers:
    def test_circuit_piping(self):
        c1 = Circuit(2).h(0)
        c2 = Circuit(2).x(1)
        c3 = c1 | c2
        assert len(c3) == 2
        assert c3.num_qubits == 2

    def test_sequential_build(self):
        c1 = Circuit(2).h(0)
        model = Sequential([
            CircuitLayer(c1),
            QFTLayer(2)
        ])
        qc = model.build()
        assert qc.num_qubits == 2
        assert len(qc) > 1

class TestWorkflows:
    def test_workflow_structure(self):
        wf = Workflow()
        wf.add(Circuit(1).h(0))
        wf.add(Loop(Circuit(1).x(0), 2))
        assert len(wf.nodes) == 2

    def test_parallel_nodes(self):
        p = Parallel(Circuit(1).h(0), Circuit(1).x(0))
        assert len(p.nodes) == 2


class TestClassicalWorkflows:
    """Tests for classical function support in workflows."""

    def test_classical_node_creation(self):
        """ClassicalNode wraps a plain function."""
        def my_func(ctx):
            return ctx.get("x", 0) + 1
        node = ClassicalNode(my_func, name="add_one")
        assert node.name == "add_one"

    def test_classical_node_rejects_non_callable(self):
        with pytest.raises(TypeError):
            ClassicalNode(42)

    def test_map_node_creation(self):
        node = MapNode(lambda x: x * 2, input_key="data")
        assert node.input_key == "data"

    def test_pipeline_node_creation(self):
        node = PipelineNode([lambda x: x + 1, lambda x: x * 2])
        assert len(node.funcs) == 2

    def test_pipeline_node_rejects_non_callable(self):
        with pytest.raises(TypeError):
            PipelineNode([lambda x: x, 42])

    def test_workflow_add_classical_function(self):
        """Adding a plain callable to a workflow wraps it in ClassicalNode."""
        wf = Workflow()
        wf.add(lambda ctx: ctx.get("x", 0) + 1, name="increment")
        assert len(wf.nodes) == 1
        assert isinstance(wf.nodes[0], ClassicalNode)

    def test_workflow_mixed_quantum_classical(self):
        """Workflow can contain both quantum circuits and classical functions."""
        wf = Workflow()
        wf.add(Circuit(1).h(0), name="quantum_step")
        wf.add(lambda ctx: "processed", name="classical_step")
        wf.add(Circuit(1).x(0), name="another_quantum")
        assert len(wf.nodes) == 3

    def test_classical_helper(self):
        """The Classical() helper produces a ClassicalNode."""
        node = Classical(lambda ctx: 42, name="answer")
        assert isinstance(node, ClassicalNode)
        assert node.name == "answer"

    def test_map_helper(self):
        """The Map() helper produces a MapNode."""
        node = Map(lambda x: x * 2, input_key="data")
        assert isinstance(node, MapNode)

    def test_pipeline_helper(self):
        """The Pipeline() helper produces a PipelineNode."""
        node = Pipeline([str.upper, str.strip])
        assert isinstance(node, PipelineNode)

    def test_parallel_with_classical(self):
        """Parallel() correctly wraps callables as ClassicalNode."""
        p = Parallel(lambda: 1, lambda: 2)
        assert len(p.nodes) == 2
        assert all(isinstance(n, ClassicalNode) for n in p.nodes)

    def test_loop_with_classical(self):
        """Loop() correctly wraps a callable as ClassicalNode."""
        loop = Loop(lambda ctx: 1, 3)
        assert isinstance(loop.body, ClassicalNode)
        assert loop.iterations == 3

    def test_branch_with_classical(self):
        """Branch() correctly wraps callables."""
        branch = Branch(lambda: True, lambda ctx: "yes", lambda ctx: "no")
        assert isinstance(branch.true_node, ClassicalNode)
        assert isinstance(branch.false_node, ClassicalNode)

    def test_classical_node_execution(self):
        """ClassicalNode executes a function and passes context."""
        def double(ctx):
            return ctx.get("previous_result", 0) * 2

        node = ClassicalNode(double)
        runner = WorkflowRunner()
        result = asyncio.run(
            node.execute(runner, context={"previous_result": 5})
        )
        assert result == 10

    def test_map_node_execution(self):
        """MapNode transforms a value from context."""
        node = MapNode(lambda x: x ** 2, input_key="value")
        runner = WorkflowRunner()
        result = asyncio.run(
            node.execute(runner, context={"value": 7})
        )
        assert result == 49

    def test_pipeline_node_execution(self):
        """PipelineNode chains multiple functions."""
        node = PipelineNode(
            [lambda x: (x or 0) + 1, lambda x: x * 3, lambda x: x - 2],
            input_key="seed",
        )
        runner = WorkflowRunner()
        result = asyncio.run(
            node.execute(runner, context={"seed": 4})
        )
        assert result == 13  # (4+1)*3 - 2

    def test_classical_workflow_run(self):
        """Full workflow with only classical functions propagates context."""
        wf = Workflow(name="ClassicalPipeline")

        def step1(ctx):
            return 10

        def step2(ctx):
            return ctx.get("previous_result", 0) + 5

        def step3(ctx):
            return ctx.get("previous_result", 0) * 2

        wf.add(step1, name="init")
        wf.add(step2, name="add_five")
        wf.add(step3, name="double")

        results = asyncio.run(
            wf.run(verbose=False)
        )
        assert results == [10, 15, 30]

    def test_hybrid_context_propagation(self):
        """Context is propagated so classical nodes can access prior results."""
        wf = Workflow(name="ContextTest")
        wf.add(lambda ctx: {"data": [1, 2, 3]}, name="produce_data")
        wf.add(
            Map(lambda val: sum(val["data"]), input_key="previous_result"),
        )
        results = asyncio.run(
            wf.run(verbose=False)
        )
        assert results[0] == {"data": [1, 2, 3]}
        assert results[1] == 6

    def test_async_classical_node(self):
        """ClassicalNode works with async functions."""
        async def async_compute(ctx):
            return ctx.get("previous_result", 0) + 100

        node = ClassicalNode(async_compute)
        runner = WorkflowRunner()
        result = asyncio.run(
            node.execute(runner, context={"previous_result": 42})
        )
        assert result == 142

    def test_workflow_default_name(self):
        """Default workflow name reflects hybrid support."""
        wf = Workflow()
        assert wf.name == "HybridWorkflow"
