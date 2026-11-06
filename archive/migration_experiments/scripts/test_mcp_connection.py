import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    project_dir = os.getcwd()
    params = StdioServerParameters(
        command=os.path.join(project_dir, ".venv", "Scripts", "python.exe"),
        args=[os.path.join(project_dir, "historian_mcp_server.py")],
        cwd=project_dir,
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            print("Conexión MCP correcta. Herramientas:", len(result.tools))

asyncio.run(main())
