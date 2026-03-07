try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None  # type: ignore

import hlquantum as hlq


class QuantumMCPServer:
    """
    Wraps FastMCP to expose custom quantum algorithms as MCP tools.
    Requires: pip install hlquantum[mcp]

    Example:
        server = QuantumMCPServer("my-quantum-algorithms")

        @server.tool(description="Run a VQE on a random Hamiltonian")
        def run_vqe(depth: int) -> str:
            return str(hlq.algorithms.vqe.run(depth=depth))

        if __name__ == "__main__":
            server.run()
    """

    def __init__(self, name: str = "quantum-mcp"):
        if FastMCP is None:
            raise ImportError(
                "The 'mcp' dependency is required. "
                "Install it via `pip install hlquantum[mcp]` or `pip install mcp`."
            )
        self.server = FastMCP(name)

    def tool(self, description: str = None):
        return self.server.tool(description=description)

    def resource(self, uri: str, name: str = None, description: str = None, mime_type: str = "text/plain"):
        return self.server.resource(uri, name=name, description=description, mime_type=mime_type)

    def prompt(self, name: str = None, description: str = None):
        return self.server.prompt(name=name, description=description)

    def run(self, transport: str = "stdio"):
        self.server.run(transport=transport)


def serve():
    """Entry point for `python -m hlquantum.mcp`. Starts the default built-in server."""
    if FastMCP is None:
        raise ImportError(
            "The 'mcp' dependency is required. "
            "Install it via `pip install hlquantum[mcp]` or `pip install mcp`."
        )

    mcp = FastMCP("hlquantum-mcp")

    @mcp.tool(description="Simulate an n-qubit uniform superposition circuit and return measurement counts.")
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

    mcp.run()
