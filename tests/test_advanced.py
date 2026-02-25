"""Tests for hlquantum transpiler and mitigation."""

from hlquantum import QuantumCircuit, run
from hlquantum.transpiler import RemoveRedundantGates, MergeRotations, transpile
from hlquantum.mitigation import ThresholdMitigation

class TestTranspiler:
    def test_remove_redundant_gates(self):
        c = QuantumCircuit(1)
        c.h(0).h(0).x(0).x(0)
        
        opt = RemoveRedundantGates().run(c)
        assert len(opt.gates) == 0
        
        c2 = QuantumCircuit(2)
        c2.cx(0, 1).cx(0, 1)
        opt2 = RemoveRedundantGates().run(c2)
        assert len(opt2.gates) == 0

    def test_merge_rotations(self):
        c = QuantumCircuit(1)
        c.rx(0, 0.5).rx(0, 0.5)
        
        opt = MergeRotations().run(c)
        assert len(opt.gates) == 1
        assert opt.gates[0].params[0] == 1.0

    def test_global_transpile(self):
        c = QuantumCircuit(1)
        c.h(0).h(0).rx(0, 0.5).rx(0, 0.5)
        
        opt = transpile(c)
        assert len(opt.gates) == 1
        assert opt.gates[0].name == "rx"

class TestMitigation:
    def test_threshold_mitigation(self):
        from hlquantum.result import ExecutionResult
        
        counts = {"00": 900, "11": 95, "01": 5} # 5 is 0.5%
        res = ExecutionResult(counts=counts, shots=1000)
        
        mit = ThresholdMitigation(threshold=0.01) # filter < 1%
        mitigated = mit.apply(res)
        
        assert "01" not in mitigated.counts
        assert "11" in mitigated.counts
        assert mitigated.counts["00"] == 900
