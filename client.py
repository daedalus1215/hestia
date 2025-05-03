import asyncio
import threading 
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any

class OllamaMCP:
    """
    A simple integration between Ollama and 
    FastMCP
    """
    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params
        self.initialized = threading.Event()
        self.tools: list[Any] = []
    def _run_background(self):
        asyncio.run(self._async_run())
    async def _async_run(self):
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    tools_result = await session.list_tools()
                    self.tools = tools_result.tools
                    print(tools_result)
        except Exception as e:  
            print(f"Error initiating the MCP server {str(e)}")
if __name__ == "__main__":
    server_parameters = StdioServerParameters(
        command="uv",
        args=["run", "python", "server.py"],
        cwd=str(Path.cwd())
    )
    ollamamcp = OllamaMCP(server_params=server_parameters)
    ollamamcp._run_background()