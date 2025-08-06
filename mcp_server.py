from fastmcp import FastMCP

from listing_middleware import ListingFilterMiddleware
import random


mcp = FastMCP("Weather")
mcp.add_middleware(ListingFilterMiddleware())

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    weather_options = [
        "sunny",
        "cloudy",
        "rainy",
        "stormy",
        "snowy",
        "windy",
        "foggy",
        "hazy",
        "drizzly",
        "clear"
    ]
    weather = random.choice(weather_options)
    return f"It's {weather} in {location}!"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")