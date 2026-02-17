"""
HLQuantum — A high-level Python package for quantum computing.

HLQuantum (High Level Quantum) provides a unified, backend-agnostic
interface for building and executing quantum circuits on real and
simulated quantum hardware. It ships with support for CUDA-Q, Qiskit,
Cirq, Amazon Braket, and PennyLane — with optional GPU acceleration
where available.
"""

__version__ = "0.1.0"

from hlquantum.circuit import QuantumCircuit as Circuit  # Friendly alias
from hlquantum.gpu import GPUConfig, GPUPrecision, detect_gpus
from hlquantum.kernel import kernel
from hlquantum.runner import run
from hlquantum import transpiler
from hlquantum import mitigation
from hlquantum import algorithms

# Explicit export of friendly names
QuantumCircuit = Circuit

__all__ = [
    "__version__",
    "Circuit",
    "QuantumCircuit",
    "GPUConfig",
    "GPUPrecision",
    "detect_gpus",
    "kernel",
    "run",
    "transpiler",
    "mitigation",
    "algorithms",
]
