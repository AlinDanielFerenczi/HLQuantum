"""Tests for hlquantum layers and workflows."""

import pytest
from hlquantum.circuit import Circuit
from hlquantum.layers import Sequential, CircuitLayer, GroverLayer, QFTLayer
from hlquantum.workflows import Workflow, Parallel, Loop, Branch

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
