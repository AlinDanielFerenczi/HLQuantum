"""
Example: building a Qiskit algorithm and exposing it via an hlquantum MCP server.

Requirements:
    pip install hlquantum[mcp,qiskit]

Usage:
    python examples\mcp_qiskit_server_demo.py     (direct)
    python -m examples.mcp_qiskit_server_demo     (as module, from project root)

MCP config (e.g. claude_desktop_config.json):
    {
      "mcpServers": {
        "hlquantum-qiskit-demo": {
          "command": "python",
          "args": ["<absolute-path>/examples/mcp_qiskit_server_demo.py"]
        }
      }
    }
"""

from __future__ import annotations

import json
from typing import Optional

import hlquantum as hlq
from hlquantum.circuit import Circuit
from hlquantum.mcp import QuantumMCPServer

try:
    from qiskit import QuantumCircuit as QiskitCircuit
    from qiskit.circuit.library import QFT
    _QISKIT_AVAILABLE = True
except ImportError:
    _QISKIT_AVAILABLE = False


# ── Algorithm builders ────────────────────────────────────────────────────────

def _build_qiskit_bell_pair() -> QiskitCircuit:
    qc = QiskitCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _build_qiskit_qft(num_qubits: int) -> QiskitCircuit:
    qft = QFT(num_qubits, do_swaps=True, approximation_degree=0).decompose()
    qc = QiskitCircuit(num_qubits, num_qubits)
    qc.compose(qft, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def run_algorithm(
    algorithm: str,
    num_qubits: int = 2,
    shots: int = 1024,
    backend_name: Optional[str] = None,
) -> dict:
    if not _QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit is not installed. Add it with: pip install hlquantum[qiskit]")

    if algorithm == "bell_pair":
        qiskit_circuit = _build_qiskit_bell_pair()
        actual_qubits = 2
    elif algorithm == "qft":
        if not 1 <= num_qubits <= 8:
            raise ValueError("num_qubits must be between 1 and 8 for QFT demo.")
        qiskit_circuit = _build_qiskit_qft(num_qubits)
        actual_qubits = num_qubits
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose 'bell_pair' or 'qft'.")

    # Import Qiskit circuit into hlquantum
    hlq_circuit: Circuit = Circuit.from_qiskit(qiskit_circuit)

    if backend_name:
        import importlib
        backend_map = {
            "qiskit_aer": ("hlquantum.backends.qiskit_backend", "QiskitBackend"),
            "cudaq":       ("hlquantum.backends.cudaq_backend",  "CudaQBackend"),
        }
        if backend_name not in backend_map:
            raise ValueError(f"Unknown backend '{backend_name}'. Choose from: {list(backend_map.keys())}")
        module_path, class_name = backend_map[backend_name]
        backend = getattr(importlib.import_module(module_path), class_name)()
    else:
        backend = None  # uses hlquantum default

    result = hlq.run(hlq_circuit, shots=shots, backend=backend)

    return {
        "algorithm": algorithm,
        "num_qubits": actual_qubits,
        "backend": result.backend_name,
        "shots": result.shots,
        "counts": result.counts,
        "most_likely": result.most_probable,
    }


# ── MCP server ────────────────────────────────────────────────────────────────

server = QuantumMCPServer(name="hlquantum-qiskit-algorithms")


@server.tool(
    description=(
        "Run a quantum algorithm built with Qiskit and executed through hlquantum. "
        "Supported algorithms: 'bell_pair' (2-qubit entangled pair) and "
        "'qft' (Quantum Fourier Transform, 1-8 qubits). "
        "Returns measurement counts and the most likely bitstring as JSON."
    )
)
def run_qiskit_algorithm(
    algorithm: str = "bell_pair",
    num_qubits: int = 2,
    shots: int = 1024,
) -> str:
    try:
        result = run_algorithm(algorithm=algorithm, num_qubits=num_qubits, shots=shots)
        return json.dumps(result, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@server.resource(
    uri="hlquantum://algorithms/list",
    name="Available Algorithms",
    description="Lists the quantum algorithms available in this MCP server.",
    mime_type="application/json",
)
def list_algorithms() -> str:
    algorithms = [
        {"id": "bell_pair", "description": "Maximally entangled 2-qubit Bell pair.", "qubits": 2},
        {"id": "qft", "description": "Quantum Fourier Transform via Qiskit's circuit library.", "qubits": "1–8"},
    ]
    return json.dumps({"algorithms": algorithms}, indent=2)


@server.prompt(
    name="suggest_algorithm",
    description="Given a high-level quantum task, suggests which algorithm to use and how to call it.",
)
def suggest_algorithm_prompt(task: str) -> str:
    return (
        f"The user wants to: {task}\n\n"
        "Available algorithms:\n"
        "  • bell_pair  – maximal qubit entanglement (2 qubits)\n"
        "  • qft        – Quantum Fourier Transform (1–8 qubits)\n\n"
        "Call `run_qiskit_algorithm` with the appropriate algorithm, num_qubits, and shots."
    )


if __name__ == "__main__":
    server.run()
