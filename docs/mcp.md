# AI-Driven Quantum (MCP)

HLQuantum supports the **Model Context Protocol (MCP)**, a standard for connecting AI models to external tools and data. With HLQuantum's MCP integration, you can expose your quantum algorithms and the entire quantum toolstack to AI assistants like Claude, Gemini, or custom agents.

## Installation

To use MCP features, install HLQuantum with the `mcp` extra:

```bash
pip install hlquantum[mcp]
```

## 1. Quick Start: The Default Server

HLQuantum comes with a built-in MCP server that provides standard quantum tools to an agent.

### Basic Simulation
Launch the server via the command line:

```bash
python -m hlquantum.mcp
```

By default, this server provides a `simulate_test_circuit` tool that an AI can use to verify backend connectivity.

### Enabling Raw Library Tools
For more advanced agentic workflows, you can enable "raw" tools that give the agent direct control over circuit instruction building and system inspections. Set the `HLQUANTUM_MCP_RAW` environment variable to `1`:

**Windows (PowerShell):**
```powershell
$env:HLQUANTUM_MCP_RAW="1"
python -m hlquantum.mcp
```

**Linux / macOS:**
```bash
HLQUANTUM_MCP_RAW=1 python -m hlquantum.mcp
```

#### Available Raw Tools:
- **`run_raw_circuit`**: Allows the agent to submit a list of gate operations and execute them.
- **`get_system_info`**: Provides details on detected GPUs and the default execution backend.

---

## 2. Building a Custom MCP Server

You can easily wrap your own quantum algorithms and expose them as MCP tools using the `QuantumMCPServer` wrapper.

### Example: Custom Grover Search

```python
from hlquantum.mcp import QuantumMCPServer
import hlquantum as hlq

# Create the server
# enable_raw_tools=True adds the standard circuit/system tools automatically
server = QuantumMCPServer(name="quantum-search-agent", enable_raw_tools=True)

@server.tool(description="Runs a Grover search for a specific bitstring pattern.")
def search_pattern(num_qubits: int, pattern: str):
    """
    Agent-facing tool to perform a quantum search.
    """
    # Create the algorithm using hlquantum's built-in Grover
    circuit = hlq.algorithms.grover(num_qubits, [pattern])
    
    # Run and return as string for the agent to parse
    result = hlq.run(circuit, shots=1024)
    return str(result)

if __name__ == "__main__":
    # Communicate over standard I/O (stdio)
    server.run()
```

---

## 3. Configuring your AI Client

Once your server script is ready, you need to point your AI client to it.

### Claude Desktop Configuration
Add your server to the `mcpServers` section of your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hlquantum": {
      "command": "python",
      "args": ["C:/path/to/your/custom_server.py"]
    }
  }
}
```

Now, Claude will be able to see and call your `search_pattern` tool directly!

## API Reference

### `QuantumMCPServer`

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` | `"quantum-mcp"` | Internal name for the MCP server. |
| `enable_raw_tools` | `bool` | `False` | Whether to automatically register `run_raw_circuit` and `get_system_info`. |

### Methods

- **`tool(description: str)`**: Decorator to register a Python function as an MCP tool.
- **`resource(uri: str, name: str)`**: Register a data resource (e.g., hardware specifications).
- **`prompt(name: str)`**: Define a prompt template that the agent can use to structure its quantum queries.
- **`run(transport="stdio")`**: Start the server.
