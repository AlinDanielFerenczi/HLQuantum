"""
hlquantum.backends.braket_backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Amazon Braket backend for HLQuantum.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from hlquantum.backends.base import Backend
from hlquantum.circuit import Gate, QuantumCircuit
from hlquantum.result import ExecutionResult

logger = logging.getLogger(__name__)


def _require_braket():
    try:
        from braket.circuits import Circuit as BraketCircuit
        from braket.devices import LocalSimulator
        return BraketCircuit, LocalSimulator
    except ImportError as exc:
        raise ImportError(
            "Amazon Braket SDK is required for the BraketBackend but is not installed.\n"
            "Install it with:  pip install amazon-braket-sdk\n"
            "See https://github.com/amazon-braket/amazon-braket-sdk-python for details."
        ) from exc


class BraketBackend(Backend):
    """Execute HLQuantum circuits using Amazon Braket."""

    def __init__(self, device: Optional[Any] = None, s3_destination: Optional[tuple] = None) -> None:
        self._user_device = device
        self._s3_destination = s3_destination

    @property
    def name(self) -> str:
        if self._user_device is not None:
            label = getattr(self._user_device, "name", str(self._user_device))
            return f"braket ({label})"
        return "braket (LocalSimulator)"

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1000,
        include_statevector: bool = False,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute *circuit* with Amazon Braket and return an :class:`ExecutionResult`."""
        BraketCircuit, LocalSimulator = _require_braket()
        braket_circuit = self._translate(circuit, BraketCircuit)

        device = self._user_device if self._user_device is not None else LocalSimulator()

        logger.info(
            "Running %d-qubit circuit (%d gates) for %d shots on %s (SV: %s)",
            circuit.num_qubits,
            len(circuit),
            shots,
            self.name,
            include_statevector,
        )

        counts: Dict[str, int] = {}
        raw_result = None
        state_vector = None

        # 1. Get counts if shots > 0
        if shots > 0:
            run_kwargs: Dict[str, Any] = {"shots": shots, **kwargs}
            if self._s3_destination is not None:
                run_kwargs["s3_destination_folder"] = self._s3_destination

            task = device.run(braket_circuit, **run_kwargs)
            raw_result = task.result()

            for bitstring, count in raw_result.measurement_counts.items():
                counts[bitstring] = count

        # 2. Get statevector if requested
        if include_statevector:
            try:
                from braket.circuits import result_types as braket_rt
                sv_circuit = self._translate(circuit, BraketCircuit)
                sv_circuit.state_vector()
                sv_kwargs = {**kwargs, "shots": 0}
                sv_task = device.run(sv_circuit, **sv_kwargs)
                sv_result = sv_task.result()
                # Access state vector from result_types
                if sv_result.result_types:
                    state_vector = sv_result.result_types[0]["value"]
            except Exception as exc:
                logger.warning("Could not retrieve state vector: %s", exc)
            if raw_result is None:
                raw_result = sv_result

        return ExecutionResult(
            counts=counts,
            shots=shots,
            backend_name=self.name,
            raw=raw_result,
            state_vector=state_vector,
        )

    @staticmethod
    def _translate(circuit: QuantumCircuit, BraketCircuit: type) -> Any:
        bc = BraketCircuit()
        for gate in circuit.gates:
            name = gate.name
            t0 = gate.targets[0]
            if name == "h":      bc.h(t0)
            elif name == "x":    bc.x(t0)
            elif name == "y":    bc.y(t0)
            elif name == "z":    bc.z(t0)
            elif name == "s":    bc.s(t0)
            elif name == "t":    bc.t(t0)
            elif name == "rx":   bc.rx(t0, gate.params[0])
            elif name == "ry":   bc.ry(t0, gate.params[0])
            elif name == "rz":   bc.rz(t0, gate.params[0])
            elif name == "cx":   bc.cnot(gate.controls[0], t0)
            elif name == "cz":   bc.cz(gate.controls[0], t0)
            elif name == "swap": bc.swap(gate.targets[0], gate.targets[1])
            elif name == "ccx":  bc.ccnot(gate.controls[0], gate.controls[1], t0)
            elif name == "mz":   pass
            else: raise ValueError(f"BraketBackend does not support gate: {name!r}")
        return bc
