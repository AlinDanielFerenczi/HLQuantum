"""Tests for hlquantum.result."""

from hlquantum.result import ExecutionResult


class TestExecutionResult:
    def test_probabilities(self):
        r = ExecutionResult(counts={"00": 500, "11": 500}, shots=1000)
        probs = r.probabilities
        assert probs["00"] == 0.5
        assert probs["11"] == 0.5

    def test_most_probable(self):
        r = ExecutionResult(counts={"00": 700, "11": 300}, shots=1000)
        assert r.most_probable == "00"

    def test_most_probable_empty(self):
        r = ExecutionResult()
        assert r.most_probable is None

    def test_expectation_value(self):
        r = ExecutionResult(counts={"00": 1000}, shots=1000)
        assert r.expectation_value() == 1.0

        r2 = ExecutionResult(counts={"11": 1000}, shots=1000)
        assert r2.expectation_value() == 1.0

        r3 = ExecutionResult(counts={"01": 1000}, shots=1000)
        assert r3.expectation_value() == -1.0

    def test_get_state_vector(self):
        import numpy as np
        sv = [1.0, 0.0, 0.0, 0.0]
        r = ExecutionResult(state_vector=sv)
        res = r.get_state_vector()
        assert isinstance(res, np.ndarray)
        assert res[0] == 1.0

    def test_repr(self):
        r = ExecutionResult(
            counts={"00": 500, "11": 500}, shots=1000, backend_name="test",
        )
        s = repr(r)
        assert "1000" in s
        assert "test" in s
