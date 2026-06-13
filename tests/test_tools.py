import pytest
from toolbench.tools import tool, ToolRegistry


@tool
def sample(city: str, days: int = 3, fast: bool = False) -> str:
    """Get a forecast.

    Args:
        city: City name.
        days: Number of days.
    """
    return f"{city}:{days}:{fast}"


def test_schema_derivation():
    fn = sample.tool.schema()["function"]
    assert fn["name"] == "sample"
    assert fn["description"] == "Get a forecast."
    props = fn["parameters"]["properties"]
    assert props["city"] == {"type": "string", "description": "City name."}
    assert props["days"]["type"] == "integer"
    assert props["fast"]["type"] == "boolean"
    assert fn["parameters"]["required"] == ["city"]


def test_registry_execute_and_error():
    reg = ToolRegistry([sample])
    assert reg.execute("sample", {"city": "NYC"}) == "NYC:3:False"
    assert reg.execute("nope", {}).startswith("ERROR: unknown tool")
    assert reg.execute("sample", {}).startswith("ERROR:")


def test_variant_override_keeps_function():
    reg = ToolRegistry([sample], overrides={"sample": {"description": "Math."}})
    assert reg.schemas()[0]["function"]["description"] == "Math."
    assert reg.execute("sample", {"city": "LA"}) == "LA:3:False"


def test_unsupported_type_fails():
    with pytest.raises(TypeError):
        @tool
        def bad(x: dict) -> str:
            """Bad.

            Args:
                x: a dict.
            """
            return "x"
