from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-service")


@mcp.tool()
def get_current_weather(city: str) -> str:
    """
    Get current weather for a city
    """
    # Return hardcoded weather data
    weather_data = {
        "beijing": "Beijing: Sunny, 22°C, low humidity, light wind",
        "shanghai": "Shanghai: Cloudy, 25°C, moderate humidity, occasional rain",
        "guangzhou": "Guangzhou: Rainy, 28°C, high humidity, heavy rain",
        "shenzhen": "Shenzhen: Partly cloudy, 26°C, moderate humidity, scattered showers",
        "hangzhou": "Hangzhou: Sunny, 24°C, low humidity, clear skies",
        "chengdu": "Chengdu: Overcast, 20°C, high humidity, foggy",
        "nyc": "New York: Sunny, 22°C, low humidity, clear day",
        "london": "London: Rainy, 15°C, high humidity, drizzle",
        "paris": "Paris: Cloudy, 18°C, moderate humidity, partly cloudy",
        "tokyo": "Tokyo: Sunny, 23°C, low humidity, clear skies",
        "sydney": "Sydney: Sunny, 20°C, low humidity, perfect day"
    }
    
    city_lower = city.lower()
    if city_lower in weather_data:
        return weather_data[city_lower]
    else:
        return f"Weather data for {city} not found. Available cities: Beijing, Shanghai, Guangzhou, Shenzhen, Hangzhou, Chengdu, NYC, London, Paris, Tokyo, Sydney"


@mcp.tool()
def get_forecast(city: str) -> str:
    """
    Get 3-day weather forecast for a city
    """
    forecast_data = {
        "beijing": "Beijing 3-day forecast: Day 1: Sunny, 22°C; Day 2: Cloudy, 20°C; Day 3: Rainy, 18°C",
        "shanghai": "Shanghai 3-day forecast: Day 1: Cloudy, 25°C; Day 2: Rainy, 24°C; Day 3: Sunny, 26°C",
        "nyc": "New York 3-day forecast: Day 1: Sunny, 22°C; Day 2: Windy, 20°C; Day 3: Partly cloudy, 21°C"
    }
    
    city_lower = city.lower()
    if city_lower in forecast_data:
        return forecast_data[city_lower]
    else:
        return f"Forecast for {city} not found. Available forecasts: Beijing, Shanghai, NYC"


@mcp.tool()
def get_temperature_unit() -> str:
    """
    Get the temperature unit used in this service
    """
    return "Temperature unit: Celsius (°C)"


if __name__ == "__main__":
    mcp.run(transport="stdio")