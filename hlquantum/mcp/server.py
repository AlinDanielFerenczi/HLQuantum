import os
from typing import List, Dict, Any, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None  # type: ignore

import hlquantum as hlq


class QuantumMCPServer:
    """
    Wraps FastMCP to expose custom quantum algorithms as MCP tools.
    Requires: pip install hlquantum[mcp]
    """

    def __init__(self, name: str = "quantum-mcp", enable_raw_tools: bool = False):
        if FastMCP is None:
            raise ImportError(
                "The 'mcp' dependency is required. "
                "Install it via `pip install hlquantum[mcp]` or `pip install mcp`."
            )
        self.server = FastMCP(name)
        if enable_raw_tools:
            self._register_raw_tools()

    def _register_raw_tools(self):
        """Register core hlquantum operations as tools."""

        @self.server.tool(description="Execute a quantum circuit from a list of gate definitions.")
        def run_raw_circuit(num_qubits: int, gates: List[Dict[str, Any]], shots: int = 1000) -> str:
            """
            `gates` example: [{"name": "h", "targets": [0]}, {"name": "cx", "targets": [1], "controls": [0]}]
            """
            try:
                qc = hlq.QuantumCircuit(num_qubits)
                for g in gates:
                    func = getattr(qc, g["name"])
                    args = g.get("targets", []) + g.get("controls", [])
                    params = g.get("params", [])
                    func(*args, *params)
                
                # Automatically add measurements if not present
                if not any(g["name"].startswith("m") for g in gates):
                    qc.measure_all()
                
                result = hlq.run(qc, shots=shots)
                return str(result)
            except Exception as e:
                return f"Error: {e}"

        @self.server.tool(description="Get available hlquantum backends and GPU capabilities.")
        def get_system_info() -> str:
            from hlquantum.runner import get_default_backend
            info = {
                "gpus": hlq.detect_gpus(),
                "default_backend": str(get_default_backend())
            }
            return str(info)

    def tool(self, description: str = None):
        return self.server.tool(description=description)

    def resource(self, uri: str, name: str = None, description: str = None, mime_type: str = "text/plain"):
        return self.server.resource(uri, name=name, description=description, mime_type=mime_type)

    def prompt(self, name: str = None, description: str = None):
        return self.server.prompt(name=name, description=description)

    def run(self, transport: str = "stdio"):
        self.server.run(transport=transport)


def serve():
    """Entry point for `python -m hlquantum.mcp`. Enabled raw tools via HLQUANTUM_MCP_RAW=1 environment variable."""
    if FastMCP is None:
        raise ImportError(
            "The 'mcp' dependency is required. "
            "Install it via `pip install hlquantum[mcp]` or `pip install mcp`."
        )

    enable_raw = os.getenv("HLQUANTUM_MCP_RAW", "0") == "1"
    server = QuantumMCPServer("hlquantum-mcp", enable_raw_tools=enable_raw)

    @server.tool(description="Simulate an n-qubit uniform superposition circuit and return measurement counts.")
    def simulate_test_circuit(num_qubits: int = 2) -> str:
        circuit = hlq.QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            circuit.h(i)
            circuit.measure(i)
        try:
            result = hlq.run(circuit)
            return f"Counts: {result.get_counts()}"
        except Exception as e:
            return f"Error: {e}"

    server.run()
