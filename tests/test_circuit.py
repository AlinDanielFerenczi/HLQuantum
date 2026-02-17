"""Tests for hlquantum.circuit."""

import pytest

from hlquantum.circuit import Gate, QuantumCircuit


class TestQuantumCircuit:
    def test_create(self):
        qc = QuantumCircuit(3)
        assert qc.num_qubits == 3
        assert len(qc) == 0

    def test_invalid_num_qubits(self):
        with pytest.raises(ValueError):
            QuantumCircuit(0)

    def test_h_gate(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        assert len(qc) == 1
        assert qc.gates[0].name == "h"
        assert qc.gates[0].targets == (0,)

    def test_cx_gate(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        gate = qc.gates[0]
        assert gate.name == "cx"
        assert gate.controls == (0,)
        assert gate.targets == (1,)

    def test_rx_gate_with_params(self):
        qc = QuantumCircuit(1)
        qc.rx(0, 1.57)
        gate = qc.gates[0]
        assert gate.name == "rx"
        assert gate.params == (1.57,)

    def test_measure_all(self):
        qc = QuantumCircuit(3)
        qc.measure_all()
        assert len(qc) == 3
        for i, gate in enumerate(qc.gates):
            assert gate.name == "mz"
            assert gate.targets == (i,)

    def test_fluent_api(self):
        qc = QuantumCircuit(2)
        result = qc.h(0).cx(0, 1).measure_all()
        assert result is qc
        assert len(qc) == 4

    def test_qubit_out_of_range(self):
        qc = QuantumCircuit(2)
        with pytest.raises(IndexError):
            qc.h(5)

    def test_ccx(self):
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        gate = qc.gates[0]
        assert gate.controls == (0, 1)
        assert gate.targets == (2,)

    def test_repr(self):
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1)
        assert "num_qubits=2" in repr(qc)
        assert "gates=2" in repr(qc)


class TestGate:
    def test_repr(self):
        g = Gate(name="cx", targets=(1,), controls=(0,))
        r = repr(g)
        assert "cx" in r
        assert "controls" in r
