from fastmcp.server.middleware import Middleware, MiddlewareContext

class ListingFilterMiddleware(Middleware):
    async def on_list_tools(self, context: MiddlewareContext, call_next):
        result = await call_next(context)
        
        # Filter out tools with "private" tag
        filtered_tools = [
            tool for tool in result 
            if "private" not in tool.tags
        ]
        
        # Return modified list
        return filtered_tools
