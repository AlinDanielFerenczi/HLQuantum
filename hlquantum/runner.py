"""Convenience helpers for executing quantum circuits."""

from __future__ import annotations

from typing import Any, List, Optional, Union

from hlquantum.backends.base import Backend
from hlquantum.circuit import QuantumCircuit
from hlquantum.kernel import Kernel
from hlquantum.result import ExecutionResult
from hlquantum.transpiler import transpile as default_transpile
from hlquantum.mitigation import apply_mitigation

_default_backend: Optional[Backend] = None


def set_default_backend(backend: Backend) -> None:
    global _default_backend
    _default_backend = backend


def get_default_backend() -> Backend:
    global _default_backend
    if _default_backend is None:
        from hlquantum.backends.cudaq_backend import CudaQBackend
        _default_backend = CudaQBackend()
    return _default_backend


def run(
    circuit_or_kernel: Union[QuantumCircuit, Kernel],
    *,
    shots: int = 1000,
    include_statevector: bool = False,
    transpile: bool = False,
    mitigation: Optional[Union[List[Any], Any]] = None,
    backend: Optional[Backend] = None,
    **kwargs,
) -> ExecutionResult:
    """Execute a circuit or kernel and return the result."""
    if isinstance(circuit_or_kernel, Kernel):
        circuit = circuit_or_kernel.circuit
    elif isinstance(circuit_or_kernel, QuantumCircuit):
        circuit = circuit_or_kernel
    else:
        raise TypeError(f"Expected Circuit or Kernel, got {type(circuit_or_kernel).__name__}")

    if transpile:
        circuit = default_transpile(circuit)

    be = backend or get_default_backend()
    be.validate(circuit)
    
    result = be.run(
        circuit, 
        shots=shots, 
        include_statevector=include_statevector, 
        **kwargs
    )

    if mitigation:
        methods = mitigation if isinstance(mitigation, list) else [mitigation]
        result = apply_mitigation(result, methods)

    return result

