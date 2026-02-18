"""
hlquantum.backends.cirq_backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Google Cirq backend for HLQuantum.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from hlquantum.backends.base import Backend
from hlquantum.circuit import Gate, QuantumCircuit
from hlquantum.gpu import GPUConfig
from hlquantum.result import ExecutionResult

logger = logging.getLogger(__name__)


def _require_cirq():
    try:
        import cirq
        return cirq
    except ImportError as exc:
        raise ImportError(
            "Cirq is required for the CirqBackend but is not installed.\n"
            "Install it with:  pip install cirq\n"
            "See https://quantumai.google/cirq for details."
        ) from exc


def _require_qsimcirq():
    try:
        import qsimcirq
        return qsimcirq
    except ImportError as exc:
        raise ImportError(
            "qsimcirq is required for GPU-accelerated Cirq simulation.\n"
            "Install it with:  pip install qsimcirq\n"
            "See https://github.com/quantumlib/qsim for details."
        ) from exc


class CirqBackend(Backend):
    """Execute HLQuantum circuits using Google Cirq."""

    def __init__(
        self,
        simulator: Optional[Any] = None,
        noise_model: Optional[Any] = None,
        gpu_config: Optional[GPUConfig] = None,
    ) -> None:
        self._user_simulator = simulator
        self._noise_model = noise_model
        self._gpu_config = gpu_config or GPUConfig()

    @property
    def supports_gpu(self) -> bool:
        return True

    @property
    def name(self) -> str:
        if self._gpu_config.enabled:
            return "cirq (qsim GPU)"
        if self._user_simulator is not None:
            return f"cirq ({type(self._user_simulator).__name__})"
        return "cirq (Simulator)"

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1000,
        include_statevector: bool = False,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Simulate *circuit* with Cirq and return an :class:`ExecutionResult`."""
        cirq = _require_cirq()

        cirq_circuit, qubits, measured_keys = self._translate(circuit, cirq)

        # Resolve simulator
        sim = self._build_simulator(cirq)

        # Apply CUDA_VISIBLE_DEVICES if configured
        if self._gpu_config.enabled:
            self._gpu_config.apply_env()

        logger.info(
            "Simulating %d-qubit circuit (%d gates) for %d shots on %s (GPU: %s, SV: %s)",
            circuit.num_qubits,
            len(circuit),
            shots,
            self.name,
            "enabled" if self._gpu_config.enabled else "disabled",
            include_statevector,
        )

        # 1. Get counts if shots > 0
        counts: Dict[str, int] = {}
        raw_result = None
        if shots > 0:
            raw_result = sim.run(cirq_circuit, repetitions=shots, **kwargs)
            if measured_keys:
                import numpy as np
                arrays = [raw_result.measurements[k] for k in measured_keys]
                combined = np.concatenate(arrays, axis=1)
                for row in combined:
                    bitstring = "".join(str(int(b)) for b in row)
                    counts[bitstring] = counts.get(bitstring, 0) + 1
            else:
                counts[""] = shots

        # 2. Get statevector if requested
        state_vector = None
        if include_statevector:
            # For a "pure" state vector, we often want it without measurements
            if measured_keys:
                # Create a version without measurements
                sv_ops = [op for op in cirq_circuit.all_operations() if not cirq.is_measurement(op)]
                sv_circuit = cirq.Circuit(sv_ops)
                sv_res = sim.simulate(sv_circuit)
            else:
                sv_res = sim.simulate(cirq_circuit)

            # Extract state vector (handling density matrix simulator if needed)
            if hasattr(sv_res, "final_state_vector"):
                state_vector = sv_res.final_state_vector
            elif hasattr(sv_res, "final_density_matrix"):
                state_vector = sv_res.final_density_matrix

        return ExecutionResult(
            counts=counts,
            shots=shots,
            backend_name=self.name,
            raw=raw_result,
            state_vector=state_vector,
            metadata={"gpu_config": repr(self._gpu_config)},
        )

    def _build_simulator(self, cirq: Any) -> Any:
        if self._gpu_config.enabled:
            qsimcirq = _require_qsimcirq()
            qsim_options = {"use_gpu": True, **self._gpu_config.extra}
            logger.info("Configuring qsim GPU simulator: %s", qsim_options)
            return qsimcirq.QSimSimulator(qsim_options)
        if self._user_simulator is not None:
            return self._user_simulator
        if self._noise_model is not None:
            return cirq.DensityMatrixSimulator(noise=self._noise_model)
        return cirq.Simulator()

    @staticmethod
    def _translate(circuit: QuantumCircuit, cirq: Any):
        qubits = cirq.LineQubit.range(circuit.num_qubits)
        ops = []
        measured_keys: list[str] = []

        for gate in circuit.gates:
            name = gate.name
            if name == "h":      ops.append(cirq.H(qubits[gate.targets[0]]))
            elif name == "x":    ops.append(cirq.X(qubits[gate.targets[0]]))
            elif name == "y":    ops.append(cirq.Y(qubits[gate.targets[0]]))
            elif name == "z":    ops.append(cirq.Z(qubits[gate.targets[0]]))
            elif name == "s":    ops.append(cirq.S(qubits[gate.targets[0]]))
            elif name == "t":    ops.append(cirq.T(qubits[gate.targets[0]]))
            elif name == "rx":   ops.append(cirq.rx(gate.params[0]).on(qubits[gate.targets[0]]))
            elif name == "ry":   ops.append(cirq.ry(gate.params[0]).on(qubits[gate.targets[0]]))
            elif name == "rz":   ops.append(cirq.rz(gate.params[0]).on(qubits[gate.targets[0]]))
            elif name == "cx":   ops.append(cirq.CNOT(qubits[gate.controls[0]], qubits[gate.targets[0]]))
            elif name == "cz":   ops.append(cirq.CZ(qubits[gate.controls[0]], qubits[gate.targets[0]]))
            elif name == "swap": ops.append(cirq.SWAP(qubits[gate.targets[0]], qubits[gate.targets[1]]))
            elif name == "ccx":  ops.append(cirq.CCX(qubits[gate.controls[0]], qubits[gate.controls[1]], qubits[gate.targets[0]]))
            elif name == "mz":
                key = f"m{gate.targets[0]}"
                measured_keys.append(key)
                ops.append(cirq.measure(qubits[gate.targets[0]], key=key))
            else:
                raise ValueError(f"CirqBackend does not support gate: {name!r}")

        return cirq.Circuit(ops), qubits, measured_keys
