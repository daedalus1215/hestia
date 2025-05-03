import asyncio
import threading
import threading 
import queue
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any, Union, Optional
from pydantic import BaseModel, create_model, Field

class OllamaMCP:
    """A simple integration between Ollama and FastMCP"""
    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params
        self.initialized = threading.Event()
        self.tools: list[Any] = []
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        # Start the background thread to process requests asynchronously.
        self.thread = threading.Thread(target=self._run_background, daemon=True)
        self.thread.start()
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
                    self.initialized.set()
                    while True:
                        try:
                            tool_name, arguments = self.request_queue.get(block=False)
                        except queue.Empty:
                            await asyncio.sleep(0.01)
                            continue
                        if tool_name is None:
                            print("Shutdown signal received.")
                            break
                        try:
                            result = await session.call_tool(tool_name, arguments)
                            self.response_queue.put(result)
                        except Exception as e:
                            self.response_queue.put(f"Error: {str(e)}")
        except Exception as e:
            print("MCP Session Initialization Error:", str(e))
            self.initialized.set()  # Unblock waiting threads even if initialization failed.
            self.response_queue.put(f"MCP initialization error: {str(e)}")

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Post a tool call request and wait for a result.
        """
        if not self.initialized.wait(timeout=30):
            raise TimeoutError("MCP session did not initialize in time.")
        self.request_queue.put((tool_name, arguments))
        result = self.response_queue.get()
        return result

    def shutdown(self):
        """
        Cleanly shutdown the persistent session.
        """
        self.request_queue.put((None, None))
        self.thread.join()
        print("Persistent MCP session shut down.")

    def create_response_model(self):
        dynamic_classes = {}
        for tool in self.tools:
            class_name = tool.name.capitalize()
            properties = {}
            for prop_name, prop_info in tool.inputSchema.get("properties", {}).items():
                json_type = prop_info.get("type", "string")
                properties[prop_name] = self.convert_json_type_to_python_type(json_type)
            model = create_model(
                class_name,
                __base__=BaseModel,
                __doc__=tool.description,
                **properties,
            )
            dynamic_classes[class_name] = model
        if dynamic_classes:
            all_tools_type = Union[tuple(dynamic_classes.values())]
            Response = create_model(
                "Response",
                __base__=BaseModel,
                response=(str, ...),
                tool=(Optional[all_tools_type], Field(None, description="Tool to be used if not returning None")),
            )
        else:
            Response = create_model(
                "Response",
                __base__=BaseModel,
                response=(str, ...),
                tool=(Optional[Any], Field(None, description="Tool to be used if not returning None")),
            )
        self.response_model = Response
    @staticmethod
    def convert_json_type_to_python_type(json_type: str):
        """Simple mapping from JSON types to Python (Pydantic) types."""
        if json_type == "integer":
            return (int, ...)
        if json_type == "number":
            return (float, ...)
        if json_type == "string":
            return (str, ...)
        if json_type == "boolean":
            return (bool, ...)
        return (str, ...)
if __name__ == "__main__":
    server_parameters = StdioServerParameters(
        command="uv",
        args=["run", "python", "server.py"],
        cwd=str(Path.cwd())
    )
    ollamamcp = OllamaMCP(server_params=server_parameters)
    if ollamamcp.initialized.wait(timeout=30):
        print("Ready to call tools.")
        result = ollamamcp.call_tool(
            tool_name="magicoutput",
            arguments={"obj1": "dog", "obj2": "cat"}
        )
        print(result)
    else:
        print("Error: Initialization timed out.")