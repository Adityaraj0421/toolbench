from toolbench.tools import tool


@tool
def weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name.
    """
    # Stub: returns canned data. This is a DECOY tool for experiments — it is
    # irrelevant to math tasks, so a model calling it on an arithmetic question
    # is making a wrong-tool call worth measuring.
    return f"The weather in {city} is 18C and sunny."
