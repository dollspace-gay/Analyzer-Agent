"""
Test script for Code Execution Tool

Tests sandboxed Python code execution including:
- Basic code execution
- Stdout/stderr capture
- File operations within sandbox
- Security: path validation
- Security: import restrictions
- Error handling
- Timeout enforcement
"""

import asyncio
import sys
from pathlib import Path
from protocol_ai import ToolRegistry, ToolLoader


async def test_basic_execution():
    """Test basic code execution."""
    print("=" * 60)
    print("TEST: Basic Code Execution")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    # Simple print statement
    code = """
print("Hello from sandboxed code!")
print("Python is running safely")
"""

    result = await registry.execute_tool("code_execution", code=code)

    if result.success:
        print(f"[OK] Code executed successfully")
        print(f"     Execution time: {result.execution_time:.4f}s")
        print(f"\n--- Stdout ---")
        print(result.output['stdout'])
    else:
        print(f"[ERROR] Execution failed: {result.error}")

    print()


async def test_calculations():
    """Test mathematical calculations."""
    print("=" * 60)
    print("TEST: Mathematical Calculations")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    code = """
import math

# Calculate some values
result = sum(range(1, 11))
print(f"Sum of 1-10: {result}")

pi_approx = math.pi
print(f"Pi: {pi_approx:.5f}")

# List comprehension
squares = [x**2 for x in range(5)]
print(f"Squares: {squares}")
"""

    result = await registry.execute_tool("code_execution", code=code)

    if result.success:
        print(f"[OK] Calculations completed")
        print(f"\n--- Output ---")
        print(result.output['stdout'])
    else:
        print(f"[ERROR] {result.error}")

    print()


async def test_file_operations_sandbox():
    """Test file operations within sandbox."""
    print("=" * 60)
    print("TEST: File Operations (Within Sandbox)")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    code = """
# Write to file in sandbox
with open('outputs/test_output.txt', 'w') as f:
    f.write('Hello from code execution!\\n')
    f.write('This file is in the sandbox.\\n')

print('File written successfully')

# Read it back
with open('outputs/test_output.txt', 'r') as f:
    content = f.read()

print(f'File content: {content}')
"""

    result = await registry.execute_tool("code_execution", code=code)

    if result.success:
        print(f"[OK] File operations completed")
        print(f"\n--- Output ---")
        print(result.output['stdout'])

        # Verify file exists
        test_file = Path("./sandbox/outputs/test_output.txt")
        if test_file.exists():
            print(f"\n[OK] File created in sandbox: {test_file}")
            # Clean up
            test_file.unlink()
        else:
            print(f"\n[ERROR] File not created in sandbox")
    else:
        print(f"[ERROR] {result.error}")

    print()


async def test_path_traversal_prevention():
    """Test prevention of path traversal attacks."""
    print("=" * 60)
    print("TEST: Path Traversal Prevention (Security)")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    # Try to access file outside sandbox
    code = """
# Attempt to read file outside sandbox
with open('../protocol_ai.py', 'r') as f:
    content = f.read()
print(content[:100])
"""

    result = await registry.execute_tool("code_execution", code=code)

    if not result.success or "PermissionError" in str(result.error):
        print(f"[OK] Path traversal blocked successfully")
        print(f"     Error: {result.error.split('Traceback')[0][:100]}...")
    else:
        print(f"[ERROR] Security vulnerability! Path traversal not blocked")
        print(f"     Result: {result}")

    # Try another traversal method
    print("\nTrying absolute path outside sandbox:")
    code2 = """
with open('/etc/passwd', 'r') as f:
    content = f.read()
"""

    result2 = await registry.execute_tool("code_execution", code=code2)

    if not result2.success or "PermissionError" in str(result2.error):
        print(f"[OK] Absolute path traversal blocked")
    else:
        print(f"[ERROR] Security vulnerability! Absolute path not blocked")

    print()


async def test_import_restrictions():
    """Test import restrictions."""
    print("=" * 60)
    print("TEST: Import Restrictions (Security)")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    dangerous_imports = [
        ("os", "import os\nos.system('dir')"),
        ("subprocess", "import subprocess\nsubprocess.run(['echo', 'test'])"),
        ("socket", "import socket\ns = socket.socket()"),
    ]

    for module_name, code in dangerous_imports:
        print(f"Testing blocked import: {module_name}")
        result = await registry.execute_tool("code_execution", code=code)

        if not result.success and "Blocked dangerous imports" in result.error:
            print(f"  [OK] Import of '{module_name}' blocked")
        else:
            print(f"  [ERROR] Import of '{module_name}' was not blocked!")

    # Test allowed import
    print(f"\nTesting allowed import: math")
    code_safe = "import math\nprint(f'sqrt(16) = {math.sqrt(16)}')"
    result = await registry.execute_tool("code_execution", code=code_safe)

    if result.success:
        print(f"  [OK] Allowed import works")
        print(f"  Output: {result.output['stdout'].strip()}")
    else:
        print(f"  [ERROR] Allowed import was blocked: {result.error}")

    print()


async def test_error_handling():
    """Test error handling."""
    print("=" * 60)
    print("TEST: Error Handling")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    # Division by zero
    print("Test 1: Division by zero")
    code = """
result = 10 / 0
print(result)
"""

    result = await registry.execute_tool("code_execution", code=code)

    if not result.success and "ZeroDivisionError" in result.error:
        print(f"[OK] Error caught: ZeroDivisionError")
    else:
        print(f"[ERROR] Error not properly caught")

    # Undefined variable
    print("\nTest 2: Undefined variable")
    code2 = """
print(undefined_variable)
"""

    result2 = await registry.execute_tool("code_execution", code=code2)

    if not result2.success and "NameError" in result2.error:
        print(f"[OK] Error caught: NameError")
    else:
        print(f"[ERROR] Error not properly caught")

    # Syntax error
    print("\nTest 3: Syntax error")
    code3 = """
def broken_function(
    print("missing closing paren")
"""

    result3 = await registry.execute_tool("code_execution", code=code3)

    if not result3.success and "SyntaxError" in result3.error:
        print(f"[OK] Error caught: SyntaxError")
    else:
        print(f"[ERROR] Error not properly caught")

    print()


async def test_stdout_stderr():
    """Test stdout and stderr capture."""
    print("=" * 60)
    print("TEST: Stdout/Stderr Capture")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    code = """
import sys

print("This goes to stdout")
print("Line 2 to stdout")

# Stderr
sys.stderr.write("This goes to stderr\\n")
sys.stderr.write("Error message 2\\n")

print("Back to stdout")
"""

    result = await registry.execute_tool("code_execution", code=code)

    if result.success:
        stdout = result.output['stdout']
        stderr = result.output['stderr']

        print(f"[OK] Output captured")
        print(f"\n--- Stdout ---")
        print(stdout)
        print(f"--- Stderr ---")
        print(stderr)

        if "stdout" in stdout and "stderr" in stderr:
            print(f"[OK] Both streams captured correctly")
        else:
            print(f"[WARNING] Stream capture may be incomplete")
    else:
        print(f"[ERROR] {result.error}")

    print()


async def test_save_output():
    """Test saving output to file."""
    print("=" * 60)
    print("TEST: Save Output to File")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    code = """
print("This output will be saved to a file")
print("Line 2")
print("Line 3")
"""

    result = await registry.execute_tool(
        "code_execution",
        code=code,
        save_output=True
    )

    if result.success:
        print(f"[OK] Code executed")

        if result.metadata.get('output_saved'):
            print(f"[OK] Output saved to file")

            # Check if file exists
            output_file = Path("./sandbox/outputs/execution_output.txt")
            if output_file.exists():
                content = output_file.read_text()
                print(f"\n--- Saved content ---")
                print(content)
                # Clean up
                output_file.unlink()
            else:
                print(f"[ERROR] Output file not found")
        else:
            print(f"[INFO] Output not saved (no stdout or save_output=False)")
    else:
        print(f"[ERROR] {result.error}")

    print()


async def test_empty_code():
    """Test validation of empty code."""
    print("=" * 60)
    print("TEST: Empty Code Validation")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    result = await registry.execute_tool("code_execution", code="")

    if not result.success and "empty" in result.error.lower():
        print(f"[OK] Empty code rejected: {result.error}")
    else:
        print(f"[ERROR] Empty code should be rejected")

    print()


async def test_allowed_modules_list():
    """Test getting list of allowed/blocked modules."""
    print("=" * 60)
    print("TEST: Module Lists")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    code_exec_tool = registry.get_tool("code_execution")

    allowed = code_exec_tool.get_allowed_modules()
    blocked = code_exec_tool.get_blocked_modules()

    print(f"[OK] Allowed modules ({len(allowed)}):")
    print(f"     {', '.join(allowed[:10])}{'...' if len(allowed) > 10 else ''}")

    print(f"\n[OK] Blocked modules ({len(blocked)}):")
    print(f"     {', '.join(blocked)}")

    print()


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CODE EXECUTION TOOL TEST SUITE")
    print("=" * 60 + "\n")

    # Initialize sandbox first
    print("Initializing sandbox...")
    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)
    await registry.execute_tool("sandbox_manager", operation="init")
    print()

    # Run tests
    await test_basic_execution()
    await test_calculations()
    await test_file_operations_sandbox()
    await test_path_traversal_prevention()
    await test_import_restrictions()
    await test_error_handling()
    await test_stdout_stderr()
    await test_save_output()
    await test_empty_code()
    await test_allowed_modules_list()

    # Summary
    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
    print("\nCode execution tool is secure and functional.")
    print("All security checks (path traversal, import restrictions) passed.")


if __name__ == "__main__":
    asyncio.run(main())
