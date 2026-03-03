"""Quantum mechanical operators (Hamiltonians/Observables)."""

from __future__ import annotations

import cmath
from typing import Any, Callable, Dict, Optional, Union

import numpy as np


class Operator:
    """A quantum mechanical operator."""

    def __init__(self, matrix: np.ndarray, num_qubits: Optional[int] = None):
        """Initialize operator from a matrix."""
        self.matrix = np.array(matrix, dtype=np.complex128)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Operator matrix must be square.")
        
        if num_qubits is not None:
            self.num_qubits = num_qubits
            if self.matrix.shape[0] != (2 ** num_qubits):
                raise ValueError(f"Matrix dimension doesn't match {num_qubits} qubits.")
        else:
            n = np.log2(self.matrix.shape[0])
            self.num_qubits = int(n) if n.is_integer() else None

    def evaluate(self, t: float = 0.0) -> np.ndarray:
        """Evaluate operator at time t."""
        return self.matrix

    def exp(self, coefficient: Union[complex, float] = 1.0) -> np.ndarray:
        """Matrix exponential exp(coeff * matrix)."""
        import scipy.linalg
        return scipy.linalg.expm(coefficient * self.matrix)

    def __add__(self, other: Union[Operator, float, complex]) -> Operator:
        if isinstance(other, (float, int, complex)):
            return Operator(self.matrix + other * np.eye(self.matrix.shape[0]), self.num_qubits)
        if isinstance(other, Operator):
            if self.matrix.shape != other.matrix.shape:
                raise ValueError("Incompatible dimensions.")
            return Operator(self.matrix + other.matrix, self.num_qubits)
        if isinstance(other, ScalarOperator):
            return TimeDependentOperator(lambda t: self.evaluate(t) + other.evaluate(t))
        return NotImplemented

    def __radd__(self, other: Union[float, complex]) -> Operator:
        return self.__add__(other)

    def __sub__(self, other: Union[Operator, float, complex]) -> Operator:
        if isinstance(other, (float, int, complex)):
            return Operator(self.matrix - other * np.eye(self.matrix.shape[0]), self.num_qubits)
        if isinstance(other, Operator):
            if self.matrix.shape != other.matrix.shape:
                raise ValueError("Incompatible dimensions.")
            return Operator(self.matrix - other.matrix, self.num_qubits)
        if isinstance(other, ScalarOperator):
            return TimeDependentOperator(lambda t: self.evaluate(t) - other.evaluate(t))
        return NotImplemented

    def __rsub__(self, other: Union[float, complex]) -> Operator:
        if isinstance(other, (float, int, complex)):
            return Operator(other * np.eye(self.matrix.shape[0]) - self.matrix, self.num_qubits)
        return NotImplemented

    def __mul__(self, other: Union[Operator, float, complex]) -> Operator:
        if isinstance(other, (float, int, complex)):
            return Operator(self.matrix * other, self.num_qubits)
        if isinstance(other, Operator):
            if self.matrix.shape != other.matrix.shape:
                raise ValueError("Incompatible dimensions.")
            return Operator(self.matrix @ other.matrix, self.num_qubits)
        if isinstance(other, ScalarOperator):
            return TimeDependentOperator(lambda t: self.evaluate(t) * other.evaluate(t))
        return NotImplemented

    def __rmul__(self, other: Union[float, complex]) -> Operator:
        return self.__mul__(other)


class ScalarOperator:
    """A time-dependent scalar coefficient."""
    def __init__(self, func: Callable[[float], Union[float, complex]]):
        self.func = func

    def evaluate(self, t: float) -> Union[float, complex]:
        return self.func(t)

    def __mul__(self, other: Operator) -> TimeDependentOperator:
        if isinstance(other, Operator):
            return TimeDependentOperator(lambda t: self.func(t) * other.evaluate(t))
        return NotImplemented


class TimeDependentOperator(Operator):
    """An operator that changes over time."""
    def __init__(self, eval_func: Callable[[float], np.ndarray]):
        self.eval_func = eval_func
        m0 = np.array(self.eval_func(0.0))
        super().__init__(m0)

    def evaluate(self, t: float = 0.0) -> np.ndarray:
        return np.array(self.eval_func(t), dtype=np.complex128)

    def __add__(self, other: Union[Operator, float, complex]) -> TimeDependentOperator:
        if isinstance(other, (float, int, complex)):
            return TimeDependentOperator(lambda t: self.evaluate(t) + other * np.eye(self.matrix.shape[0]))
        if isinstance(other, Operator):
            return TimeDependentOperator(lambda t: self.evaluate(t) + other.evaluate(t))
        return NotImplemented

    def __sub__(self, other: Union[Operator, float, complex]) -> TimeDependentOperator:
        if isinstance(other, (float, int, complex)):
            return TimeDependentOperator(lambda t: self.evaluate(t) - other * np.eye(self.matrix.shape[0]))
        if isinstance(other, Operator):
            return TimeDependentOperator(lambda t: self.evaluate(t) - other.evaluate(t))
        return NotImplemented

    def __mul__(self, other: Union[Operator, float, complex]) -> TimeDependentOperator:
        if isinstance(other, (float, int, complex)):
            return TimeDependentOperator(lambda t: self.evaluate(t) * other)
        if hasattr(other, 'evaluate'):
            return TimeDependentOperator(lambda t: self.evaluate(t) @ other.evaluate(t))
        return NotImplemented


def spin_x() -> Operator:
    """Pauli X."""
    return Operator([[0, 1], [1, 0]], num_qubits=1)

def spin_y() -> Operator:
    """Pauli Y."""
    return Operator([[0, -1j], [1j, 0]], num_qubits=1)

def spin_z() -> Operator:
    """Pauli Z."""
    return Operator([[1, 0], [0, -1]], num_qubits=1)

def spin_plus() -> Operator:
    """Raising operator |0><1|."""
    return Operator([[0, 1], [0, 0]], num_qubits=1)

def spin_minus() -> Operator:
    """Lowering operator |1><0|."""
    return Operator([[0, 0], [1, 0]], num_qubits=1)


def annihilate(dim: int) -> Operator:
    """Bosonic annihilation operator."""
    m = np.zeros((dim, dim), dtype=np.complex128)
    for n in range(1, dim):
        m[n-1, n] = np.sqrt(n)
    return Operator(m, num_qubits=None)

def create(dim: int) -> Operator:
    """Bosonic creation operator."""
    m = np.zeros((dim, dim), dtype=np.complex128)
    for n in range(1, dim):
        m[n, n-1] = np.sqrt(n)
    return Operator(m, num_qubits=None)

def number(dim: int) -> Operator:
    """Number operator N = a^† a."""
    return create(dim) * annihilate(dim)

def parity(dim: int) -> Operator:
    """Parity operator P = exp(i * pi * a^† a)."""
    n_op = number(dim)
    return Operator(n_op.exp(1j * np.pi), num_qubits=None)

def displace(dim: int, alpha: complex) -> Operator:
    """Displacement operator D(alpha)."""
    a = annihilate(dim)
    a_dag = create(dim)
    arg = alpha * a_dag - np.conj(alpha) * a
    return Operator(arg.exp(), num_qubits=None)

def squeeze(dim: int, z: complex) -> Operator:
    """Squeezing operator S(z)."""
    a = annihilate(dim)
    a_dag = create(dim)
    arg = 0.5 * (np.conj(z) * (a * a) - z * (a_dag * a_dag))
    return Operator(arg.exp(), num_qubits=None)

