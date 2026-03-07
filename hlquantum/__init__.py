"""HLQuantum — High-level quantum computing in Python."""

__version__ = "0.1.3"

from hlquantum.circuit import QuantumCircuit
from hlquantum.result import ExecutionResult
from hlquantum.gpu import GPUConfig, GPUPrecision, detect_gpus
from hlquantum.kernel import kernel
from hlquantum.runner import run
from hlquantum import transpiler
from hlquantum import mitigation
from hlquantum import algorithms
from hlquantum import operators
from hlquantum import dynamics
from hlquantum.operators import Operator, ScalarOperator

try:
    from hlquantum import mcp
except ImportError:
    mcp = None  # Extra dependencies not installed

__all__ = [
    "__version__",
    "QuantumCircuit",
    "ExecutionResult",
    "GPUConfig",
    "GPUPrecision",
    "detect_gpus",
    "kernel",
    "run",
    "transpiler",
    "mitigation",
    "algorithms",
    "operators",
    "dynamics",
    "Operator",
    "ScalarOperator",
]

if mcp is not None:
    __all__.append("mcp")

