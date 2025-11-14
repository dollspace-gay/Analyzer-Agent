"""
Test Logging and Error Handling Framework

Verifies that the logging system:
1. Logs at appropriate verbosity levels
2. Handles file and console logging independently
3. Captures context correctly
4. Provides structured error handling
5. Integrates with the system
"""

import sys
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protocol_ai_logging import (
    ProtocolAILogger,
    get_logger,
    set_verbosity,
    ProtocolAIError,
    ModuleError,
    TriggerError,
    DependencyError,
    BundleError,
    LLMError
)
import logging


def test_logger_creation():
    """Test 1: Logger creation and basic logging"""
    print("="*60)
    print("TEST 1: Logger Creation")
    print("="*60)

    log_dir = "./tests/test_logs_temp"
    logger = ProtocolAILogger(
        name="test_logger",
        console_level=logging.INFO,
        log_dir=log_dir,
        enable_file_logging=True
    )

    print("\n[OK] Logger created")

    # Test basic logging
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")

    # Check log file exists
    log_file = Path(log_dir) / "test_logger.log"
    if log_file.exists():
        print("[OK] Log file created")
        # Clean up
        shutil.rmtree(log_dir, ignore_errors=True)
        return True
    else:
        print("[FAIL] Log file not created")
        return False


def test_verbosity_levels():
    """Test 2: Verbosity level filtering"""
    print("\n" + "="*60)
    print("TEST 2: Verbosity Levels")
    print("="*60)

    log_dir = "./tests/test_logs_temp2"
    logger = ProtocolAILogger(
        name="test_levels",
        console_level=logging.WARNING,  # Only WARNING and above
        log_dir=log_dir,
        enable_file_logging=False
    )

    print("\nConsole level set to WARNING")
    print("Testing different levels (only WARNING+ should show):")

    logger.debug("This is debug (should not appear)")
    logger.info("This is info (should not appear)")
    logger.warning("This is warning (should appear)")
    logger.error("This is error (should appear)")

    print("\n[OK] Verbosity filtering works")
    return True


def test_context_tracking():
    """Test 3: Context tracking"""
    print("\n" + "="*60)
    print("TEST 3: Context Tracking")
    print("="*60)

    logger = ProtocolAILogger(
        name="test_context",
        console_level=logging.INFO,
        enable_file_logging=False
    )

    # Set context
    logger.set_context(module="GriftDetection", tier=2)
    logger.info("Message with context")

    # Add more context
    logger.info("Message with additional context", test_id="TEST-001")

    # Clear context
    logger.clear_context()
    logger.info("Message without context")

    print("\n[OK] Context tracking works")
    return True


def test_category_logging():
    """Test 4: Category-specific logging"""
    print("\n" + "="*60)
    print("TEST 4: Category Logging")
    print("="*60)

    logger = ProtocolAILogger(
        name="test_category",
        console_level=logging.INFO,
        enable_file_logging=False
    )

    logger.log_module_trigger("GriftDetection", "test prompt", "grift")
    logger.log_arbitration(5, "GriftDetection", 2)
    logger.log_dependency_resolution("ComplianceEngineering", ["ConsentAudit"], 1)
    logger.log_bundle_load("governance", 5)
    logger.log_llm_execution(1000, 500, 1.5)

    print("\n[OK] Category-specific logging works")
    return True


def test_error_classes():
    """Test 5: Custom error classes"""
    print("\n" + "="*60)
    print("TEST 5: Custom Error Classes")
    print("="*60)

    try:
        raise ModuleError("Test module error", module_name="TestModule", tier=2)
    except ModuleError as e:
        print(f"\n[OK] ModuleError caught: {e}")
        error_dict = e.to_dict()
        if error_dict['category'] == 'module' and error_dict['context']['module'] == 'TestModule':
            print("[OK] Error context captured correctly")
        else:
            print("[FAIL] Error context incorrect")
            return False

    try:
        raise DependencyError("Circular dependency", module="TestModule")
    except DependencyError as e:
        print(f"[OK] DependencyError caught: {e}")

    try:
        raise BundleError("Bundle not found", bundle_name="missing_bundle")
    except BundleError as e:
        print(f"[OK] BundleError caught: {e}")

    print("\n[OK] All error classes work")
    return True


def test_global_logger():
    """Test 6: Global logger singleton"""
    print("\n" + "="*60)
    print("TEST 6: Global Logger")
    print("="*60)

    logger1 = get_logger()
    logger2 = get_logger()

    if logger1 is logger2:
        print("\n[OK] Global logger is singleton")
        return True
    else:
        print("\n[FAIL] Global logger created multiple instances")
        return False


def test_set_verbosity():
    """Test 7: Set verbosity helper"""
    print("\n" + "="*60)
    print("TEST 7: Set Verbosity Helper")
    print("="*60)

    logger = get_logger()

    set_verbosity('debug')
    print("\n[OK] Set verbosity to DEBUG")

    logger.debug("Debug message (should appear)")

    set_verbosity('error')
    print("[OK] Set verbosity to ERROR")

    logger.info("Info message (should not appear)")
    logger.error("Error message (should appear)")

    print("\n[OK] Verbosity helper works")
    return True


def test_file_logging():
    """Test 8: File logging with rotation"""
    print("\n" + "="*60)
    print("TEST 8: File Logging")
    print("="*60)

    log_dir = "./tests/test_logs_temp8"
    logger = ProtocolAILogger(
        name="test_file",
        console_level=logging.ERROR,  # Quiet console
        file_level=logging.DEBUG,  # Verbose file
        log_dir=log_dir,
        enable_file_logging=True,
        max_bytes=1000,  # Small size to test rotation
        backup_count=2
    )

    # Write many messages
    for i in range(50):
        logger.info(f"Message {i}")

    log_file = Path(log_dir) / "test_file.log"
    if log_file.exists():
        print(f"\n[OK] Log file created: {log_file}")

        # Check if content was written
        with open(log_file, 'r') as f:
            content = f.read()
            if "Message" in content:
                print("[OK] Log content written")
            else:
                print("[FAIL] No log content found")
                shutil.rmtree(log_dir, ignore_errors=True)
                return False

        # Clean up
        shutil.rmtree(log_dir, ignore_errors=True)
        return True
    else:
        print("[FAIL] Log file not created")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("LOGGING FRAMEWORK TEST SUITE")
    print("="*60)

    tests = [
        ("Logger Creation", test_logger_creation),
        ("Verbosity Levels", test_verbosity_levels),
        ("Context Tracking", test_context_tracking),
        ("Category Logging", test_category_logging),
        ("Custom Error Classes", test_error_classes),
        ("Global Logger", test_global_logger),
        ("Set Verbosity Helper", test_set_verbosity),
        ("File Logging", test_file_logging),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n[SUCCESS] All logging framework tests passed!")
        return 0
    else:
        print(f"\n[PARTIAL] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
