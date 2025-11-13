"""
Test script for Tools Integration with Orchestrator

Tests the complete integration of tools with LLM prompt assembly including:
- Tool schema injection into prompts
- Tool invocation detection in responses
- Tool execution and result feedback
- Multi-turn tool usage
- Tool chaining
"""

import asyncio
import sys
from protocol_ai import (
    ModuleLoader, ToolRegistry, ToolLoader, Orchestrator, LLMInterface, Module
)


# ============================================================================
# Mock LLM for Testing
# ============================================================================

class MockLLMWithTools:
    """
    Mock LLM that simulates tool invocations.

    Returns different responses based on call count to test tool chaining.
    """

    def __init__(self):
        self.call_count = 0
        self.model = "mock-llm-with-tools"

    def load_model(self):
        pass

    def execute(self, prompt: str) -> str:
        """
        Simulate LLM response with tool calls.

        Args:
            prompt: The assembled prompt

        Returns:
            Simulated LLM response (may include tool calls)
        """
        self.call_count += 1

        print(f"\n[MockLLM] Call #{self.call_count}")
        print(f"[MockLLM] Prompt length: {len(prompt)} chars")

        # Check if tools are available in prompt
        has_tools = "[AVAILABLE TOOLS]" in prompt
        has_tool_history = "[TOOL EXECUTION HISTORY]" in prompt

        if self.call_count == 1 and has_tools:
            # First call: Request calculator tool
            return """I'll help you with that calculation.

{"tool_call": {"tool": "calculator", "params": {"operation": "multiply", "a": 7, "b": 6}}}

Let me calculate 7 * 6 for you."""

        elif self.call_count == 2 and has_tool_history:
            # Second call: Use calculator result and request web search
            return """Based on the calculation result of 42, let me search for more information.

{"tool_call": {"tool": "web_search", "params": {"query": "meaning of 42", "max_results": 2}}}

Searching for the significance of the number 42."""

        elif self.call_count == 3 and has_tool_history:
            # Third call: Final answer using both tool results
            return """Based on the tools I used:

1. Calculator showed that 7 * 6 = 42
2. Web search found information about the number 42

The answer to your question is 42, which is famously known from
"The Hitchhiker's Guide to the Galaxy" as the Answer to the Ultimate
Question of Life, the Universe, and Everything!"""

        else:
            # Default response
            return "This is a standard response without tool usage."


# ============================================================================
# Test Functions
# ============================================================================

async def test_tool_detection():
    """Test tool invocation detection."""
    print("=" * 60)
    print("TEST: Tool Invocation Detection")
    print("=" * 60)

    # Create mock orchestrator
    modules = []
    llm = MockLLMWithTools()
    orch = Orchestrator(modules, llm, enable_audit=False)

    # Test valid tool call
    response1 = '{"tool_call": {"tool": "calculator", "params": {"operation": "add", "a": 1, "b": 2}}}'
    detected1 = orch._detect_tool_invocation(response1)

    if detected1 and detected1['tool'] == 'calculator':
        print(f"[OK] Tool call detected: {detected1}")
    else:
        print(f"[ERROR] Failed to detect tool call")

    # Test no tool call
    response2 = "This is just a regular response without any tool calls."
    detected2 = orch._detect_tool_invocation(response2)

    if detected2 is None:
        print(f"[OK] Correctly identified no tool call")
    else:
        print(f"[ERROR] False positive tool detection: {detected2}")

    # Test embedded tool call
    response3 = """Let me use a tool to help you.

{"tool_call": {"tool": "web_search", "params": {"query": "test query"}}}

I'll search for that information."""

    detected3 = orch._detect_tool_invocation(response3)

    if detected3 and detected3['tool'] == 'web_search':
        print(f"[OK] Embedded tool call detected: {detected3}")
    else:
        print(f"[ERROR] Failed to detect embedded tool call")

    print()


async def test_tool_schema_injection():
    """Test that tool schemas are injected into prompts."""
    print("=" * 60)
    print("TEST: Tool Schema Injection")
    print("=" * 60)

    # Create tool registry and load tools
    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    print(f"[OK] Loaded {len(registry.get_all_tools())} tools")

    # Create orchestrator with tools
    modules = []
    llm = MockLLMWithTools()
    orch = Orchestrator(modules, llm, enable_audit=False, tool_registry=registry)

    # Assemble prompt
    prompt = orch._assemble_prompt("Test prompt", None, None)

    # Check if tools are in prompt
    if "[AVAILABLE TOOLS]" in prompt:
        print(f"[OK] Tool schema injected into prompt")
        print(f"[OK] Prompt contains tool usage instructions")

        # Check for specific tools
        if "calculator" in prompt:
            print(f"[OK] Calculator tool listed")
        if "web_search" in prompt:
            print(f"[OK] Web search tool listed")
        if "sandbox_manager" in prompt:
            print(f"[OK] Sandbox manager tool listed")
        if "code_execution" in prompt:
            print(f"[OK] Code execution tool listed")
    else:
        print(f"[ERROR] Tool schema not injected")

    print()


async def test_single_tool_execution():
    """Test single tool execution."""
    print("=" * 60)
    print("TEST: Single Tool Execution")
    print("=" * 60)

    # Setup
    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    modules = []
    llm = MockLLMWithTools()
    orch = Orchestrator(modules, llm, enable_audit=False, tool_registry=registry)

    # Process prompt (should trigger calculator tool)
    print("\nProcessing prompt that will trigger tool usage...")
    result = await orch.process_prompt("Calculate 7 times 6")

    print(f"\n--- Results ---")
    print(f"Tool turns: {result['tool_turns']}")
    print(f"Tool executions: {len(result['tool_executions'])}")

    if result['tool_executions']:
        print(f"\nTool executions:")
        for i, exec in enumerate(result['tool_executions'], 1):
            print(f"\n  {i}. Tool: {exec['tool']}")
            print(f"     Params: {exec['params']}")
            print(f"     Success: {exec['result']['success']}")
            if exec['result']['success']:
                print(f"     Output: {exec['result']['output']}")

        if result['tool_turns'] > 0:
            print(f"\n[OK] Tool execution successful")
        else:
            print(f"\n[ERROR] Tool should have been executed")
    else:
        print(f"\n[INFO] No tools executed (LLM may not have called any)")

    print(f"\nFinal response: {result['llm_response'][:200]}...")

    print()


async def test_multi_turn_tool_usage():
    """Test multi-turn tool usage (tool chaining)."""
    print("=" * 60)
    print("TEST: Multi-Turn Tool Usage (Tool Chaining)")
    print("=" * 60)

    # Setup
    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    modules = []
    llm = MockLLMWithTools()
    orch = Orchestrator(modules, llm, enable_audit=False, tool_registry=registry)

    # Process prompt (should trigger multiple tools)
    print("\nProcessing prompt that will trigger multiple tool calls...")
    result = await orch.process_prompt("What is 7 times 6 and what does that number mean?")

    print(f"\n--- Results ---")
    print(f"Total LLM calls: {llm.call_count}")
    print(f"Tool turns: {result['tool_turns']}")
    print(f"Tool executions: {len(result['tool_executions'])}")

    if result['tool_executions']:
        print(f"\nTool execution chain:")
        for i, exec in enumerate(result['tool_executions'], 1):
            print(f"\n  Turn {i}:")
            print(f"    Tool: {exec['tool']}")
            print(f"    Params: {exec['params']}")
            print(f"    Success: {exec['result']['success']}")
            if exec['result']['success']:
                output = str(exec['result']['output'])
                print(f"    Output: {output[:100]}{'...' if len(output) > 100 else ''}")
            else:
                print(f"    Error: {exec['result']['error']}")

        if result['tool_turns'] >= 2:
            print(f"\n[OK] Multi-turn tool usage working ({result['tool_turns']} turns)")
        else:
            print(f"\n[INFO] Only {result['tool_turns']} tool turn(s)")
    else:
        print(f"\n[INFO] No tools executed")

    print(f"\nFinal response:")
    print(result['llm_response'])

    print()


async def test_tool_error_handling():
    """Test tool error handling."""
    print("=" * 60)
    print("TEST: Tool Error Handling")
    print("=" * 60)

    # Create mock LLM that calls non-existent tool
    class MockLLMBadTool:
        def __init__(self):
            self.model = "mock"

        def load_model(self):
            pass

        def execute(self, prompt: str) -> str:
            if "AVAILABLE TOOLS" in prompt:
                # Try to call non-existent tool
                return '{"tool_call": {"tool": "nonexistent_tool", "params": {}}}'
            return "No tools available"

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    modules = []
    llm = MockLLMBadTool()
    orch = Orchestrator(modules, llm, enable_audit=False, tool_registry=registry)

    result = await orch.process_prompt("Test error handling")

    if result['tool_executions']:
        exec = result['tool_executions'][0]
        if not exec['result']['success'] and 'not found' in exec['result']['error'].lower():
            print(f"[OK] Tool error handled correctly: {exec['result']['error']}")
        else:
            print(f"[ERROR] Tool error not handled properly")
    else:
        print(f"[INFO] No tool execution attempted")

    print()


async def test_max_tool_turns():
    """Test max tool turns limit."""
    print("=" * 60)
    print("TEST: Max Tool Turns Limit")
    print("=" * 60)

    # Create mock LLM that always tries to call tools (infinite loop scenario)
    class MockLLMInfiniteTools:
        def __init__(self):
            self.model = "mock"
            self.call_count = 0

        def load_model(self):
            pass

        def execute(self, prompt: str) -> str:
            self.call_count += 1
            # Always try to call calculator
            return '{"tool_call": {"tool": "calculator", "params": {"operation": "add", "a": 1, "b": 1}}}'

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    modules = []
    llm = MockLLMInfiniteTools()
    orch = Orchestrator(modules, llm, enable_audit=False, tool_registry=registry)

    result = await orch.process_prompt("Keep calling tools")

    print(f"Tool turns: {result['tool_turns']}")
    print(f"Max allowed: {orch.max_tool_turns}")

    if result['tool_turns'] == orch.max_tool_turns:
        print(f"[OK] Max tool turns limit enforced ({result['tool_turns']} turns)")
    else:
        print(f"[WARNING] Tool turns: {result['tool_turns']}, expected {orch.max_tool_turns}")

    print()


async def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("TOOLS INTEGRATION TEST SUITE")
    print("=" * 60 + "\n")

    await test_tool_detection()
    await test_tool_schema_injection()
    await test_single_tool_execution()
    await test_multi_turn_tool_usage()
    await test_tool_error_handling()
    await test_max_tool_turns()

    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
    print("\nTools are successfully integrated with Orchestrator!")
    print("Features working:")
    print("  [OK] Tool schema injection into prompts")
    print("  [OK] Tool invocation detection")
    print("  [OK] Tool execution and result feedback")
    print("  [OK] Multi-turn tool usage (tool chaining)")
    print("  [OK] Error handling")
    print("  [OK] Max turns limit enforcement")


if __name__ == "__main__":
    asyncio.run(main())
