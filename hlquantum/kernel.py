"""Quantum kernel decorator and wrapper."""

from __future__ import annotations

import functools
from typing import Callable, Optional

from hlquantum.circuit import QuantumCircuit


class Kernel:
    """Wraps a function into a reusable quantum kernel."""

    def __init__(self, fn: Callable, num_qubits: int) -> None:
        self._fn = fn
        self.num_qubits = num_qubits
        self.name = fn.__name__
        functools.update_wrapper(self, fn)

    @property
    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        self._fn(qc)
        return qc

    def __call__(self, qc: Optional[QuantumCircuit] = None) -> QuantumCircuit:
        if qc is None:
            qc = QuantumCircuit(self.num_qubits)
        self._fn(qc)
        return qc

    def __repr__(self) -> str:
        return f"Kernel({self.name!r}, num_qubits={self.num_qubits})"


def kernel(num_qubits: int) -> Callable:
    """Decorator to transform a function into a Kernel."""
    def decorator(fn: Callable) -> Kernel:
        return Kernel(fn, num_qubits)
    return decorator

