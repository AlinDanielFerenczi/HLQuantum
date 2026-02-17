"""
hlquantum.backends.cudaq_backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA-Q backend for HLQuantum.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from hlquantum.backends.base import Backend
from hlquantum.circuit import Gate, QuantumCircuit
from hlquantum.gpu import GPUConfig, GPUPrecision
from hlquantum.result import ExecutionResult

logger = logging.getLogger(__name__)


def _require_cudaq():
    try:
        import cudaq
        return cudaq
    except ImportError as exc:
        raise ImportError(
            "CUDA-Q is required for the CudaQBackend but is not installed.\n"
            "Install it with:  pip install cudaq\n"
            "See https://nvidia.github.io/cuda-quantum for details."
        ) from exc


class CudaQBackend(Backend):
    """Execute HLQuantum circuits on NVIDIA CUDA-Q."""

    def __init__(
        self,
        target: Optional[str] = None,
        gpu_config: Optional[GPUConfig] = None,
    ) -> None:
        self._gpu_config = gpu_config or GPUConfig()
        self._explicit_target = target
        self._target = self._resolve_target()

    @property
    def supports_gpu(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"cudaq ({self._target})"

    def _resolve_target(self) -> str:
        if self._explicit_target is not None:
            return self._explicit_target
        if not self._gpu_config.enabled:
            return "default"
        if self._gpu_config.multi_gpu:
            return "nvidia-mqpu"
        if self._gpu_config.precision == GPUPrecision.FP64:
            return "nvidia-fp64"
        return "nvidia"

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1000,
        include_statevector: bool = False,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Sample *circuit* on CUDA-Q and return an :class:`ExecutionResult`."""
        cudaq = _require_cudaq()

        # Apply CUDA_VISIBLE_DEVICES if configured
        if self._gpu_config.enabled:
            self._gpu_config.apply_env()

        # Set the target
        target_kwargs: Dict[str, Any] = {}
        if self._gpu_config.enabled and self._gpu_config.extra:
            target_kwargs.update(self._gpu_config.extra)

        cudaq.set_target(self._target, **target_kwargs)

        num_qubits = circuit.num_qubits
        gates = circuit.gates

        # Build a cudaq kernel dynamically
        kernel_builder, qubits = cudaq.make_kernel()
        qreg = kernel_builder.qalloc(num_qubits)

        for gate in gates:
            self._apply_gate(kernel_builder, qreg, gate, cudaq)

        logger.info(
            "Executing %d-qubit circuit (%d gates) for %d shots on %s (GPU: %s, SV: %s)",
            num_qubits,
            len(gates),
            shots,
            self._target,
            "enabled" if self._gpu_config.enabled else "disabled",
            include_statevector,
        )

        # 1. Get counts if shots > 0
        counts: Dict[str, int] = {}
        raw_result = None
        if shots > 0:
            raw_result = cudaq.sample(kernel_builder, shots_count=shots)
            for bitstring in raw_result:
                counts[bitstring] = raw_result.count(bitstring)

        # 2. Get statevector if requested
        state_vector = None
        if include_statevector:
            # Note: get_state typically works best without measurement gates
            # If the circuit has measurements, we might need a separate kernel without them
            if any(g.name == "mz" for g in gates):
                sv_builder, _ = cudaq.make_kernel()
                sv_qreg = sv_builder.qalloc(num_qubits)
                for gate in gates:
                    if gate.name != "mz":
                        self._apply_gate(sv_builder, sv_qreg, gate, cudaq)
                state_vector = cudaq.get_state(sv_builder)
            else:
                state_vector = cudaq.get_state(kernel_builder)

        return ExecutionResult(
            counts=counts,
            shots=shots,
            backend_name=self.name,
            raw=raw_result,
            state_vector=state_vector,
            metadata={"gpu_config": repr(self._gpu_config)},
        )

    @staticmethod
    def _apply_gate(builder: Any, qreg: Any, gate: Gate, cudaq: Any) -> None:
        name = gate.name
        if name == "mz":
            builder.mz(qreg[gate.targets[0]])
        elif name in ("h", "x", "y", "z", "s", "t"):
            getattr(builder, name)(qreg[gate.targets[0]])
        elif name in ("rx", "ry", "rz"):
            getattr(builder, name)(gate.params[0], qreg[gate.targets[0]])
        elif name == "cx":
            builder.cx(qreg[gate.controls[0]], qreg[gate.targets[0]])
        elif name == "cz":
            builder.cz(qreg[gate.controls[0]], qreg[gate.targets[0]])
        elif name == "swap":
            builder.swap(qreg[gate.targets[0]], qreg[gate.targets[1]])
        elif name == "ccx":
            builder.ccx(qreg[gate.controls[0]], qreg[gate.controls[1]], qreg[gate.targets[0]])
        else:
            raise ValueError(f"CudaQBackend does not support gate: {name!r}")
