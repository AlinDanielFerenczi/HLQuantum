"""
hlquantum.backends.pennylane_backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PennyLane (Xanadu) backend for HLQuantum.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from hlquantum.backends.base import Backend
from hlquantum.circuit import Gate, QuantumCircuit
from hlquantum.gpu import GPUConfig
from hlquantum.result import ExecutionResult

logger = logging.getLogger(__name__)


def _require_pennylane():
    try:
        import pennylane as qml
        return qml
    except ImportError as exc:
        raise ImportError(
            "PennyLane is required for the PennyLaneBackend but is not installed.\n"
            "Install it with:  pip install pennylane\n"
            "See https://pennylane.ai for details."
        ) from exc


class PennyLaneBackend(Backend):
    """Execute HLQuantum circuits using Xanadu PennyLane."""

    def __init__(
        self,
        device_name: Optional[str] = None,
        device_kwargs: Optional[Dict[str, Any]] = None,
        gpu_config: Optional[GPUConfig] = None,
    ) -> None:
        self._gpu_config = gpu_config or GPUConfig()
        self._explicit_device = device_name
        self._device_name = self._resolve_device()
        self._device_kwargs = device_kwargs or {}

    @property
    def supports_gpu(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"pennylane ({self._device_name})"

    def _resolve_device(self) -> str:
        if self._explicit_device is not None:
            return self._explicit_device
        if self._gpu_config.enabled:
            return "lightning.gpu"
        return "default.qubit"

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1000,
        include_statevector: bool = False,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute *circuit* with PennyLane and return an :class:`ExecutionResult`."""
        qml = _require_pennylane()
        import numpy as np

        num_qubits = circuit.num_qubits
        gates = circuit.gates

        # Apply CUDA_VISIBLE_DEVICES if configured
        if self._gpu_config.enabled:
            self._gpu_config.apply_env()

        # Shared device kwargs
        dev_kwargs = {**self._device_kwargs, **self._gpu_config.extra}

        logger.info(
            "Executing %d-qubit circuit (%d gates) for %d shots on %s (GPU: %s, SV: %s)",
            num_qubits,
            len(gates),
            shots,
            self.name,
            "enabled" if self._gpu_config.enabled else "disabled",
            include_statevector,
        )

        # 1. Get counts if shots > 0
        counts: Dict[str, int] = {}
        raw_result = None
        if shots > 0:
            measured: List[int] = [g.targets[0] for g in gates if g.name == "mz"]
            if not measured:
                measured = list(range(num_qubits))

            dev = qml.device(self._device_name, wires=num_qubits, shots=shots, **dev_kwargs)

            @qml.qnode(dev)
            def qnode_counts():
                for gate in gates:
                    self._apply_gate(gate, qml)
                return qml.counts(wires=measured)

            raw_result = qnode_counts()

            for key, count in raw_result.items():
                if isinstance(key, (int, np.integer)):
                    bitstring = str(int(key))
                elif isinstance(key, (tuple, list, np.ndarray)):
                    bitstring = "".join(str(int(b)) for b in key)
                else:
                    bitstring = str(key)
                counts[bitstring] = int(count)

        # 2. Get statevector if requested
        state_vector = None
        if include_statevector:
            # State vector in PL usually requires shots=None
            sv_dev = qml.device(self._device_name, wires=num_qubits, shots=None, **dev_kwargs)

            @qml.qnode(sv_dev)
            def qnode_state():
                # Apply gates EXCEPT measurements (PL statevector is usually pre-measurement)
                for gate in gates:
                    if gate.name != "mz":
                        self._apply_gate(gate, qml)
                return qml.state()

            state_vector = qnode_state()

        return ExecutionResult(
            counts=counts,
            shots=shots,
            backend_name=self.name,
            raw=raw_result,
            state_vector=state_vector,
            metadata={"gpu_config": repr(self._gpu_config)},
        )

    @staticmethod
    def _apply_gate(gate: Gate, qml: Any) -> None:
        name = gate.name
        t0 = gate.targets[0]
        if name == "h":      qml.Hadamard(wires=t0)
        elif name == "x":    qml.PauliX(wires=t0)
        elif name == "y":    qml.PauliY(wires=t0)
        elif name == "z":    qml.PauliZ(wires=t0)
        elif name == "s":    qml.S(wires=t0)
        elif name == "t":    qml.T(wires=t0)
        elif name == "rx":   qml.RX(gate.params[0], wires=t0)
        elif name == "ry":   qml.RY(gate.params[0], wires=t0)
        elif name == "rz":   qml.RZ(gate.params[0], wires=t0)
        elif name == "cx":   qml.CNOT(wires=[gate.controls[0], t0])
        elif name == "cz":   qml.CZ(wires=[gate.controls[0], t0])
        elif name == "swap": qml.SWAP(wires=[gate.targets[0], gate.targets[1]])
        elif name == "ccx":  qml.Toffoli(wires=[gate.controls[0], gate.controls[1], t0])
        elif name == "mz":   pass
        else: raise ValueError(f"PennyLaneBackend does not support gate: {name!r}")
