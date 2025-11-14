"""
Test script for Tools System

Tests the complete tools framework including:
- Tool base class
- ToolRegistry registration and lookup
- ToolLoader plugin discovery
- Tool execution and validation
- Execution logging
"""

import asyncio
import sys
from protocol_ai import Tool, ToolResult, ToolRegistry, ToolLoader


# ============================================================================
# Mock Tool for Testing
# ============================================================================

class MockEchoTool(Tool):
    """Simple echo tool for testing."""

    def __init__(self):
        super().__init__(
            name="echo",
            description="Echoes back the input message",
            parameters={
                "message": {
                    "type": "string",
                    "description": "Message to echo",
                    "required": True
                }
            }
        )

    async def execute(self, **kwargs) -> ToolResult:
        """Echo the message back."""
        message = kwargs.get('message', '')
        return ToolResult(
            success=True,
            tool_name=self.name,
            output=f"Echo: {message}"
        )


class MockFailTool(Tool):
    """Tool that always fails for testing error handling."""

    def __init__(self):
        super().__init__(
            name="fail",
            description="Always fails for testing",
            parameters={}
        )

    async def execute(self, **kwargs) -> ToolResult:
        """Always raise an exception."""
        raise RuntimeError("Intentional failure for testing")


# ============================================================================
# Test Functions
# ============================================================================

async def test_tool_registration():
    """Test tool registration and lookup."""
    print("=" * 60)
    print("TEST: Tool Registration")
    print("=" * 60)

    registry = ToolRegistry()

    # Test registration
    echo_tool = MockEchoTool()
    registry.register(echo_tool)

    # Test lookup
    found_tool = registry.get_tool("echo")
    assert found_tool is not None, "Tool should be found"
    assert found_tool.name == "echo", "Tool name should match"

    print("[OK] Tool registered and found successfully")

    # Test duplicate registration
    try:
        registry.register(MockEchoTool())
        print("[ERROR] Should have raised ValueError for duplicate registration")
    except ValueError as e:
        print(f"[OK] Duplicate registration prevented: {e}")

    # Test get all tools
    all_tools = registry.get_all_tools()
    assert len(all_tools) == 1, "Should have 1 tool"
    print(f"[OK] Registry has {len(all_tools)} tool(s)")

    # Test unregister
    registry.unregister("echo")
    assert registry.get_tool("echo") is None, "Tool should be unregistered"
    print("[OK] Tool unregistered successfully")

    print()


async def test_tool_execution():
    """Test tool execution with validation."""
    print("=" * 60)
    print("TEST: Tool Execution")
    print("=" * 60)

    registry = ToolRegistry()
    echo_tool = MockEchoTool()
    registry.register(echo_tool)

    # Test successful execution
    result = await registry.execute_tool("echo", message="Hello, World!")
    assert result.success, "Execution should succeed"
    assert result.output == "Echo: Hello, World!", "Output should match"
    print(f"[OK] Tool executed successfully: {result.output}")
    print(f"     Execution time: {result.execution_time:.4f}s")

    # Test validation failure (missing required parameter)
    result = await registry.execute_tool("echo")
    assert not result.success, "Should fail validation"
    assert "Missing required parameter" in result.error, "Should have validation error"
    print(f"[OK] Validation correctly caught missing parameter")

    # Test tool not found
    result = await registry.execute_tool("nonexistent")
    assert not result.success, "Should fail for nonexistent tool"
    assert "not found" in result.error, "Should have not found error"
    print(f"[OK] Correctly handled nonexistent tool")

    # Test disabled tool
    echo_tool.enabled = False
    result = await registry.execute_tool("echo", message="test")
    assert not result.success, "Should fail for disabled tool"
    assert "disabled" in result.error, "Should have disabled error"
    print(f"[OK] Correctly prevented execution of disabled tool")

    print()


async def test_tool_error_handling():
    """Test error handling in tool execution."""
    print("=" * 60)
    print("TEST: Error Handling")
    print("=" * 60)

    registry = ToolRegistry()
    fail_tool = MockFailTool()
    registry.register(fail_tool)

    # Test exception handling
    result = await registry.execute_tool("fail")
    assert not result.success, "Should fail on exception"
    assert "Intentional failure" in result.error, "Should capture exception message"
    print(f"[OK] Exception caught and returned as error: {result.error}")

    print()


async def test_execution_logging():
    """Test execution log tracking."""
    print("=" * 60)
    print("TEST: Execution Logging")
    print("=" * 60)

    registry = ToolRegistry()
    echo_tool = MockEchoTool()
    registry.register(echo_tool)

    # Execute multiple times
    await registry.execute_tool("echo", message="First")
    await registry.execute_tool("echo", message="Second")
    await registry.execute_tool("echo", message="Third")

    # Check log
    log = registry.get_execution_log()
    assert len(log) == 3, "Should have 3 log entries"
    print(f"[OK] Execution log has {len(log)} entries")

    # Check log contents
    for i, entry in enumerate(log, 1):
        print(f"     Entry {i}: {entry['tool']} - {entry['success']} - {entry['execution_time']:.4f}s")

    # Test limited log retrieval
    recent_log = registry.get_execution_log(limit=2)
    assert len(recent_log) == 2, "Should return only 2 recent entries"
    print(f"[OK] Limited log retrieval works (last 2 entries)")

    # Test log clearing
    registry.clear_execution_log()
    assert len(registry.get_execution_log()) == 0, "Log should be empty"
    print(f"[OK] Execution log cleared")

    print()


async def test_tool_schema():
    """Test tool schema generation for LLM consumption."""
    print("=" * 60)
    print("TEST: Tool Schema Generation")
    print("=" * 60)

    registry = ToolRegistry()
    echo_tool = MockEchoTool()
    registry.register(echo_tool)

    # Get schemas
    schemas = registry.get_tools_schema()
    assert len(schemas) == 1, "Should have 1 schema"

    schema = schemas[0]
    print(f"[OK] Schema generated for tool: {schema['name']}")
    print(f"     Description: {schema['description']}")
    print(f"     Parameters: {list(schema['parameters'].keys())}")
    print(f"     Enabled: {schema['enabled']}")

    # Disable and check schema
    echo_tool.enabled = False
    schemas = registry.get_tools_schema()
    assert len(schemas) == 0, "Should have no schemas for disabled tools"
    print(f"[OK] Disabled tools excluded from schema")

    print()


async def test_tool_discovery():
    """Test tool discovery and loading."""
    print("=" * 60)
    print("TEST: Tool Discovery & Loading")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")

    # Discover tools
    tool_modules = loader.discover_tools()
    print(f"[OK] Discovered {len(tool_modules)} tool module(s): {tool_modules}")

    # Load tools
    loaded_count = loader.load_tools(registry)
    print(f"[OK] Loaded {loaded_count} tool(s)")

    # List loaded tools
    all_tools = registry.get_all_tools()
    print(f"\nLoaded Tools:")
    for tool in all_tools:
        print(f"  - {tool.name}: {tool.description}")

    # Test using loaded tool (if calculator is loaded)
    if registry.get_tool("calculator"):
        result = await registry.execute_tool("calculator", operation="add", a=5, b=3)
        if result.success:
            print(f"\n[OK] Calculator tool executed: 5 + 3 = {result.output}")
        else:
            print(f"[ERROR] Calculator failed: {result.error}")

    print()


# ============================================================================
# Integration Test
# ============================================================================

async def test_full_integration():
    """Test complete tools workflow."""
    print("=" * 60)
    print("INTEGRATION TEST: Complete Workflow")
    print("=" * 60)

    # 1. Create registry
    registry = ToolRegistry()
    print("\n1. Created ToolRegistry")

    # 2. Load tools from directory
    loader = ToolLoader(tools_dir="./tools")
    loaded_count = loader.load_tools(registry)
    print(f"2. Loaded {loaded_count} tools from ./tools/")

    # 3. Get enabled tools schema (for LLM)
    schemas = registry.get_tools_schema()
    print(f"3. Generated schemas for {len(schemas)} enabled tools")

    # 4. Execute a tool
    if schemas:
        tool_name = schemas[0]['name']
        print(f"\n4. Executing tool: {tool_name}")

        if tool_name == "calculator":
            result = await registry.execute_tool(
                tool_name,
                operation="multiply",
                a=7,
                b=6
            )
            print(f"   Result: {result.output if result.success else result.error}")
            print(f"   Success: {result.success}")
            print(f"   Time: {result.execution_time:.4f}s")

    # 5. Check execution log
    log = registry.get_execution_log()
    print(f"\n5. Execution log has {len(log)} entries")

    print("\n[OK] Integration test complete!")


# ============================================================================
# Main Test Runner
# ============================================================================

async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TOOLS SYSTEM TEST SUITE")
    print("=" * 60 + "\n")

    # Run all tests
    await test_tool_registration()
    await test_tool_execution()
    await test_tool_error_handling()
    await test_execution_logging()
    await test_tool_schema()
    await test_tool_discovery()
    await test_full_integration()

    # Summary
    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
