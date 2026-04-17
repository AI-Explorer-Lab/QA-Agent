from fastmcp import FastMCP  

mcp = FastMCP("Chat Server")

@mcp.tool()
def chat(message: str) -> str:
    return None