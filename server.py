from fastmcp import FastMCP
# Create an MCP server
mcp = FastMCP("TestServer")
# my tool:
@mcp.tool()
def magicoutput(obj1: str, obj2: str) -> int:
    """Use this function to get  the magic output"""
    return "WomboWombat"
if __name__ == "__main__":
    mcp.run()