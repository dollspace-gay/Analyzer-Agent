"""
Example Calculator Tool

Demonstrates how to implement a Tool subclass.
This tool performs basic arithmetic operations.
"""

import sys
sys.path.insert(0, '..')

from protocol_ai import Tool, ToolResult
from typing import Dict, Any


class CalculatorTool(Tool):
    """
    Simple calculator tool for basic arithmetic operations.

    Supports: addition, subtraction, multiplication, division
    """

    def __init__(self):
        """Initialize the calculator tool."""
        name = "calculator"
        description = "Performs basic arithmetic operations (add, subtract, multiply, divide)"

        # Define parameter schema
        parameters = {
            "operation": {
                "type": "string",
                "description": "Operation to perform: 'add', 'subtract', 'multiply', or 'divide'",
                "required": True,
                "enum": ["add", "subtract", "multiply", "divide"]
            },
            "a": {
                "type": "number",
                "description": "First operand",
                "required": True
            },
            "b": {
                "type": "number",
                "description": "Second operand",
                "required": True
            }
        }

        super().__init__(name, description, parameters)

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the calculator operation.

        Args:
            operation: Type of operation
            a: First number
            b: Second number

        Returns:
            ToolResult with calculation result
        """
        operation = kwargs.get('operation')
        a = float(kwargs.get('a', 0))
        b = float(kwargs.get('b', 0))

        try:
            if operation == 'add':
                result = a + b
            elif operation == 'subtract':
                result = a - b
            elif operation == 'multiply':
                result = a * b
            elif operation == 'divide':
                if b == 0:
                    return ToolResult(
                        success=False,
                        tool_name=self.name,
                        error="Division by zero is not allowed"
                    )
                result = a / b
            else:
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    error=f"Unknown operation: {operation}"
                )

            return ToolResult(
                success=True,
                tool_name=self.name,
                output=result,
                metadata={
                    "operation": operation,
                    "operands": {"a": a, "b": b}
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Calculation error: {str(e)}"
            )
