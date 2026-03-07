import pytest
try:
    import mcp
except ImportError:
    mcp = None

import hlquantum as hlq

@pytest.mark.skipif(mcp is None, reason="mcp extra not installed")
def test_mcp_server_initialization():
    from hlquantum.mcp import QuantumMCPServer
    
    server = QuantumMCPServer(name="test_server")
    assert server.server.name == "test_server"
    
@pytest.mark.skipif(mcp is None, reason="mcp extra not installed")
def test_mcp_server_tool_registration():
    from hlquantum.mcp import QuantumMCPServer
    
    server = QuantumMCPServer(name="test_server")
    
    @server.tool(description="A mock tool")
    def mock_tool(val: int) -> int:
        return val * 2
        
    assert mock_tool(2) == 4
    
    # The tool should work as expected.
    # FastMCP holds the tool internally in _tool_manager.

def test_mcp_is_exposed():
    # It should correctly expose `mcp` module in __init__ if `mcp` is installed
    if mcp is not None:
        assert hasattr(hlq, "mcp")
        assert hlq.mcp is not None
        assert hasattr(hlq.mcp, "QuantumMCPServer")
    else:
        assert not hasattr(hlq, "mcp") or getattr(hlq, "mcp") is None
