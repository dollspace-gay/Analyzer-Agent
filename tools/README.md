# Tools Directory

This directory contains tool plugins for the Protocol AI system.

## Creating a New Tool

All tools must inherit from the `Tool` base class and implement the `execute()` method.

### Basic Template

```python
"""
Tool Name

Description of what this tool does.
"""

import sys
sys.path.insert(0, '..')

from protocol_ai import Tool, ToolResult
from typing import Dict, Any


class YourToolName(Tool):
    """Brief description of your tool."""

    def __init__(self):
        """Initialize the tool."""
        name = "your_tool_name"
        description = "What this tool does"

        # Define parameter schema
        parameters = {
            "param1": {
                "type": "string",  # or "number", "boolean", "array", "object"
                "description": "Description of parameter",
                "required": True,  # or False for optional params
            },
            "param2": {
                "type": "number",
                "description": "Another parameter",
                "required": False,
                "default": 0
            }
        }

        super().__init__(name, description, parameters)

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool.

        Args:
            **kwargs: Tool parameters

        Returns:
            ToolResult with success status and output/error
        """
        # Extract parameters
        param1 = kwargs.get('param1')
        param2 = kwargs.get('param2', 0)

        try:
            # Your tool logic here
            result = self._do_something(param1, param2)

            return ToolResult(
                success=True,
                tool_name=self.name,
                output=result,
                metadata={
                    "param1": param1,
                    "param2": param2
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Error: {str(e)}"
            )

    def _do_something(self, param1, param2):
        """Your tool's main logic."""
        # Implementation here
        return "result"
```

## Security Guidelines

### For Sandboxed Tools (Code Execution, File Operations)

1. **ALWAYS** restrict file operations to `./sandbox/` directory
2. **NEVER** allow path traversal (`../`, absolute paths)
3. **VALIDATE** all file paths before operations
4. **LIMIT** resource usage (memory, CPU, execution time)
5. **SANITIZE** all inputs and outputs

### Path Validation Example

```python
def _validate_sandbox_path(self, path: str) -> tuple[bool, str]:
    """
    Validate that path is within sandbox.

    Returns:
        (is_valid, absolute_path or error_message)
    """
    from pathlib import Path

    sandbox_root = Path("./sandbox").resolve()
    try:
        requested_path = Path(path).resolve()

        # Check if path is within sandbox
        if sandbox_root in requested_path.parents or requested_path == sandbox_root:
            return True, str(requested_path)
        else:
            return False, f"Path outside sandbox: {path}"

    except Exception as e:
        return False, f"Invalid path: {str(e)}"
```

## Available Tools

### calculator_tool.py
- **Purpose**: Basic arithmetic operations
- **Operations**: add, subtract, multiply, divide
- **Example**:
  ```python
  result = await registry.execute_tool(
      "calculator",
      operation="multiply",
      a=7,
      b=6
  )
  # result.output == 42.0
  ```

## Tool Loading

Tools are automatically discovered and loaded by the `ToolLoader`:

```python
from protocol_ai import ToolRegistry, ToolLoader

registry = ToolRegistry()
loader = ToolLoader(tools_dir="./tools")
loader.load_tools(registry)

# All tools in ./tools/*.py are now registered
```

## Testing Your Tool

Create a test file in the root directory:

```python
import asyncio
from protocol_ai import ToolRegistry, ToolLoader

async def test_my_tool():
    registry = ToolRegistry()
    loader = ToolLoader()
    loader.load_tools(registry)

    # Execute your tool
    result = await registry.execute_tool(
        "your_tool_name",
        param1="test",
        param2=123
    )

    print(f"Success: {result.success}")
    print(f"Output: {result.output}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_my_tool())
```

## Planned Tools

- [Agent-7k1] **Web Search Tool**: Search engines integration
- [Agent-bc4] **Code Execution Tool**: Sandboxed Python execution
- [Agent-3qc] **File System Tool**: Read/write files in sandbox
- [Agent-wjm] **Sandbox Manager**: Clean and manage sandbox environment

## Best Practices

1. **Error Handling**: Always catch exceptions and return descriptive error messages
2. **Validation**: Validate all inputs before processing
3. **Logging**: Use meaningful metadata in ToolResult
4. **Async**: Use `async def` for I/O-bound operations
5. **Documentation**: Clear docstrings for all methods
6. **Type Hints**: Use type hints for better IDE support
7. **Testing**: Test both success and failure cases
