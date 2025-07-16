from fastapi import FastAPI, Request, HTTPException
from mcp.server.fastmcp import FastMCP

app = FastAPI()
mcp_app = FastMCP()

# Define some tools
@mcp_app.tool()
def get_public_data():
    """Retrieves publicly available data."""
    return {"data": "This is public information."}

@mcp_app.tool()
def get_admin_dashboard_stats():
    """Retrieves statistics for the admin dashboard (requires admin role)."""
    return {"stats": {"users": 100, "revenue": 50000, "active_sessions": 25}}

@mcp_app.tool()
def update_user_status(user_id: str, status: str):
    """Updates the status of a user (requires admin role)."""
    return {"message": f"User {user_id} status updated to {status}."}

# Custom list_tools endpoint (or a middleware approach)
@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """
    Handles MCP requests, dynamically adjusting available tools based on headers.
    """
    # Get the incoming request body
    body = await request.json()
    operation = body.get("operation")

    if operation == "list_tools":
        # Check the "role" header
        user_role = request.headers.get("role")
        print(f"Received 'role' header: {user_role}") # For debugging

        if user_role == "admin":
            # If admin, return all tools
            tools = await mcp_app.list_tools()
            return {"tools": [tool_def.model_dump() for tool_def in tools]}
        else:
            tools = await mcp_app.list_tools()
            public_tools = [tool for tool in tools if tool.name == "get_public_data"]
            return {"tools": [tool.model_dump() for tool in public_tools]}
    elif operation == "call_tool":
        # For call_tool, we'll let FastMCP handle it.
        # You might want to add similar role-based checks for calling tools directly
        # if your @tool decorators don't already handle authorization.
        arguments = body.get("arguments", {})
        return await mcp_app.call_tool(request, arguments)
    else:
        raise HTTPException(status_code=400, detail="Invalid MCP operation")

# Example of how you'd run this (using uvicorn)
# uvicorn your_file_name:app --reload