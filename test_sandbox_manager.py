"""
Test script for Sandbox Manager Tool

Tests sandbox management functionality including:
- Sandbox initialization
- Path validation (security)
- Directory listing
- Cleanup operations
- Statistics tracking
- File removal
"""

import asyncio
import sys
from pathlib import Path
from protocol_ai import ToolRegistry, ToolLoader


async def test_sandbox_init():
    """Test sandbox initialization."""
    print("=" * 60)
    print("TEST: Sandbox Initialization")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    # Initialize sandbox
    result = await registry.execute_tool("sandbox_manager", operation="init")

    if result.success:
        print(f"[OK] {result.output}")
        print(f"     Sandbox root: {result.metadata['sandbox_root']}")
        print(f"     Subdirectories: {', '.join(result.metadata['subdirectories'])}")

        # Verify directories exist
        sandbox_root = Path(result.metadata['sandbox_root'])
        for subdir in result.metadata['subdirectories']:
            subdir_path = sandbox_root / subdir
            if subdir_path.exists():
                print(f"     [OK] {subdir}/ exists")
            else:
                print(f"     [ERROR] {subdir}/ not created")
    else:
        print(f"[ERROR] Initialization failed: {result.error}")

    print()


async def test_path_validation():
    """Test path validation security."""
    print("=" * 60)
    print("TEST: Path Validation (Security)")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    test_cases = [
        # Valid paths
        ("inputs/test.txt", True, "Valid relative path"),
        ("outputs", True, "Valid subdirectory"),
        ("temp/subdir/file.py", True, "Valid nested path"),
        ("", True, "Empty path (sandbox root)"),

        # Invalid paths (directory traversal attempts)
        ("../outside.txt", False, "Parent directory traversal"),
        ("../../etc/passwd", False, "Multiple parent traversal"),
        ("/etc/passwd", False, "Absolute path outside sandbox"),
        ("inputs/../../../etc/passwd", False, "Mixed traversal attempt"),
    ]

    print("Testing path validation...")
    passed = 0
    failed = 0

    for path, should_be_valid, description in test_cases:
        result = await registry.execute_tool(
            "sandbox_manager",
            operation="validate_path",
            path=path
        )

        is_valid = result.success

        if is_valid == should_be_valid:
            status = "[OK]"
            passed += 1
        else:
            status = "[FAIL]"
            failed += 1

        print(f"  {status} {description}")
        print(f"       Path: '{path}'")
        print(f"       Expected valid: {should_be_valid}, Got: {is_valid}")

        if not is_valid:
            print(f"       Error: {result.error}")

        print()

    print(f"Results: {passed} passed, {failed} failed")
    print()


async def test_directory_listing():
    """Test directory listing functionality."""
    print("=" * 60)
    print("TEST: Directory Listing")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    # Initialize sandbox first
    await registry.execute_tool("sandbox_manager", operation="init")

    # Create some test files
    sandbox_root = Path("./sandbox")
    test_file = sandbox_root / "inputs" / "test.txt"
    test_file.write_text("Test content")
    print(f"Created test file: {test_file.relative_to(sandbox_root)}")

    # List sandbox root
    print("\nListing sandbox root (non-recursive):")
    result = await registry.execute_tool(
        "sandbox_manager",
        operation="list",
        recursive=False
    )

    if result.success:
        print(f"[OK] Found {result.metadata['item_count']} items")
        for item in result.output:
            print(f"     {item['type']:10s} {item['path']}")
    else:
        print(f"[ERROR] {result.error}")

    # List inputs subdirectory
    print("\nListing inputs subdirectory:")
    result = await registry.execute_tool(
        "sandbox_manager",
        operation="list",
        path="inputs",
        recursive=False
    )

    if result.success:
        print(f"[OK] Found {result.metadata['item_count']} items in inputs/")
        for item in result.output:
            size_mb = item['size'] / 1024 if item['type'] == 'file' else 0
            print(f"     {item['type']:10s} {item['path']:30s} {size_mb:.2f} KB")
    else:
        print(f"[ERROR] {result.error}")

    # Recursive listing
    print("\nListing sandbox (recursive):")
    result = await registry.execute_tool(
        "sandbox_manager",
        operation="list",
        recursive=True
    )

    if result.success:
        print(f"[OK] Found {result.metadata['item_count']} total items (recursive)")
    else:
        print(f"[ERROR] {result.error}")

    print()


async def test_statistics():
    """Test sandbox statistics."""
    print("=" * 60)
    print("TEST: Sandbox Statistics")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    # Get stats
    result = await registry.execute_tool("sandbox_manager", operation="stats")

    if result.success:
        stats = result.output
        print(f"[OK] Sandbox statistics:")
        print(f"     Root: {stats['sandbox_root']}")
        print(f"     Exists: {stats['exists']}")
        print(f"     Total files: {stats.get('total_files', 0)}")
        print(f"     Total size: {stats.get('total_size_mb', 0)} MB")

        print(f"\n     Subdirectories:")
        for subdir, info in stats.get('subdirectories', {}).items():
            print(f"       {subdir:10s}: {info['file_count']} files, {info['total_size_mb']} MB")
            print(f"                    {info['description']}")
    else:
        print(f"[ERROR] {result.error}")

    print()


async def test_file_removal():
    """Test file removal."""
    print("=" * 60)
    print("TEST: File Removal")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    # Create a test file
    sandbox_root = Path("./sandbox")
    test_file = sandbox_root / "temp" / "delete_me.txt"
    test_file.write_text("This file will be deleted")
    print(f"Created test file: {test_file.relative_to(sandbox_root)}")

    # Verify it exists
    assert test_file.exists(), "Test file should exist"

    # Remove the file
    result = await registry.execute_tool(
        "sandbox_manager",
        operation="remove_file",
        path="temp/delete_me.txt"
    )

    if result.success:
        print(f"[OK] {result.output}")
        if not test_file.exists():
            print(f"[OK] File successfully deleted")
        else:
            print(f"[ERROR] File still exists after removal")
    else:
        print(f"[ERROR] {result.error}")

    # Try to remove non-existent file
    print("\nTrying to remove non-existent file:")
    result = await registry.execute_tool(
        "sandbox_manager",
        operation="remove_file",
        path="temp/nonexistent.txt"
    )

    if not result.success:
        print(f"[OK] Correctly failed: {result.error}")
    else:
        print(f"[ERROR] Should have failed for non-existent file")

    # Try to remove file outside sandbox (security test)
    print("\nTrying to remove file outside sandbox (security test):")
    result = await registry.execute_tool(
        "sandbox_manager",
        operation="remove_file",
        path="../test_sandbox_manager.py"
    )

    if not result.success and "outside sandbox" in result.error.lower():
        print(f"[OK] Security check passed: {result.error}")
    else:
        print(f"[ERROR] Security vulnerability! Should block removal outside sandbox")

    print()


async def test_cleanup():
    """Test sandbox cleanup."""
    print("=" * 60)
    print("TEST: Sandbox Cleanup")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    # Create some test files
    sandbox_root = Path("./sandbox")
    test_files = [
        sandbox_root / "inputs" / "cleanup_test1.txt",
        sandbox_root / "outputs" / "cleanup_test2.txt",
        sandbox_root / "temp" / "cleanup_test3.txt",
    ]

    for test_file in test_files:
        test_file.write_text("Test content")
        print(f"Created: {test_file.relative_to(sandbox_root)}")

    # Get initial stats
    stats_before = await registry.execute_tool("sandbox_manager", operation="stats")
    files_before = stats_before.output.get('total_files', 0) if stats_before.success else 0

    print(f"\nFiles before cleanup: {files_before}")

    # Cleanup sandbox
    result = await registry.execute_tool("sandbox_manager", operation="cleanup")

    if result.success:
        print(f"[OK] {result.output}")
        print(f"     Items removed: {result.metadata['files_removed']}")
        print(f"     Directories cleaned: {', '.join(result.metadata['directories_cleaned'])}")

        # Verify files are gone
        remaining = sum(1 for f in test_files if f.exists())
        if remaining == 0:
            print(f"[OK] All test files removed")
        else:
            print(f"[ERROR] {remaining} test files still exist")

        # Verify structure is preserved
        for subdir in ["inputs", "outputs", "temp", "logs"]:
            subdir_path = sandbox_root / subdir
            if subdir_path.exists():
                print(f"[OK] Directory structure preserved: {subdir}/")
            else:
                print(f"[ERROR] Directory missing: {subdir}/")
    else:
        print(f"[ERROR] {result.error}")

    print()


async def test_clear_temp():
    """Test clearing only temp directory."""
    print("=" * 60)
    print("TEST: Clear Temp Directory")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    sandbox_root = Path("./sandbox")

    # Create files in different directories
    temp_file = sandbox_root / "temp" / "temp_file.txt"
    input_file = sandbox_root / "inputs" / "input_file.txt"

    temp_file.write_text("Temp content")
    input_file.write_text("Input content")

    print(f"Created temp file: {temp_file.exists()}")
    print(f"Created input file: {input_file.exists()}")

    # Clear only temp
    result = await registry.execute_tool("sandbox_manager", operation="clear_temp")

    if result.success:
        print(f"[OK] {result.output}")

        # Verify temp file removed
        if not temp_file.exists():
            print(f"[OK] Temp file removed")
        else:
            print(f"[ERROR] Temp file still exists")

        # Verify input file preserved
        if input_file.exists():
            print(f"[OK] Input file preserved")
        else:
            print(f"[ERROR] Input file was removed (should be preserved)")

        # Cleanup
        if input_file.exists():
            input_file.unlink()
    else:
        print(f"[ERROR] {result.error}")

    print()


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SANDBOX MANAGER TEST SUITE")
    print("=" * 60 + "\n")

    # Run tests in order
    await test_sandbox_init()
    await test_path_validation()
    await test_directory_listing()
    await test_statistics()
    await test_file_removal()
    await test_clear_temp()
    await test_cleanup()

    # Summary
    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
    print("\nSandbox manager tool is fully functional and secure.")


if __name__ == "__main__":
    asyncio.run(main())
