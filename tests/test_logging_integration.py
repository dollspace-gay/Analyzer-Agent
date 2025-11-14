"""
Test Logging Integration with Protocol AI

Demonstrates how the logging framework integrates with the main system
and improves debuggability for module conflicts, dependency issues, and LLM failures.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protocol_ai import ModuleLoader, Orchestrator
from protocol_ai_logging import get_logger, set_verbosity, ModuleError, DependencyError
import logging


class MockLLM:
    """Mock LLM with logging"""
    def __init__(self):
        self.logger = get_logger()

    def execute(self, prompt):
        import time
        start = time.time()

        # Simulate execution
        response = "Mock LLM response"

        duration = time.time() - start
        self.logger.log_llm_execution(len(prompt), len(response), duration)

        return response


def test_logging_with_module_loading():
    """Test 1: Logging during module loading"""
    print("="*60)
    print("TEST 1: Module Loading with Logging")
    print("="*60)

    set_verbosity('info')
    logger = get_logger()

    logger.info("Starting module loader test", category="test")

    try:
        loader = ModuleLoader(modules_dir="./modules")
        modules = loader.load_modules()

        logger.info(f"Successfully loaded {len(modules)} modules", category="system")

        # Log each module
        for module in modules:
            logger.debug(
                f"Module loaded: {module.name}",
                category="module",
                tier=module.tier,
                triggers=len(module.triggers)
            )

        print("\n[OK] Module loading with logging")
        return True

    except Exception as e:
        logger.error(f"Failed to load modules: {e}", category="error", exc_info=True)
        return False


async def test_logging_with_orchestrator():
    """Test 2: Logging during orchestration"""
    print("\n" + "="*60)
    print("TEST 2: Orchestrator with Logging")
    print("="*60)

    set_verbosity('info')
    logger = get_logger()

    try:
        loader = ModuleLoader(modules_dir="./modules")
        modules = loader.load_modules()

        llm = MockLLM()
        orchestrator = Orchestrator(
            modules=modules,
            llm_interface=llm,
            enable_audit=False
        )

        logger.info("Processing test prompt", category="system")

        # Process a prompt
        result = await orchestrator.process_prompt(
            "Analyze this organization's fundraising practices for potential grift"
        )

        triggered = result.get('triggered_modules', [])
        logger.log_arbitration(len(triggered), result.get('selected_module'), 2)

        print("\n[OK] Orchestrator logging")
        return True

    except Exception as e:
        logger.exception("Orchestrator failed", category="error")
        return False


def test_error_logging():
    """Test 3: Structured error logging"""
    print("\n" + "="*60)
    print("TEST 3: Structured Error Logging")
    print("="*60)

    set_verbosity('info')
    logger = get_logger()

    try:
        # Simulate a module error
        raise ModuleError(
            "Failed to load module definition",
            module_name="BrokenModule",
            file_path="./modules/broken.yaml"
        )
    except ModuleError as e:
        logger.error(
            f"Module error occurred: {e.message}",
            category="error",
            **e.context
        )
        print("\n[OK] Structured error logged")

    try:
        # Simulate a dependency error
        raise DependencyError(
            "Circular dependency detected",
            module="ModuleA",
            depends_on="ModuleB"
        )
    except DependencyError as e:
        logger.error(
            f"Dependency error: {e.message}",
            category="error",
            **e.context
        )
        print("[OK] Dependency error logged")

    return True


def test_context_based_logging():
    """Test 4: Context-based debugging"""
    print("\n" + "="*60)
    print("TEST 4: Context-Based Debugging")
    print("="*60)

    set_verbosity('debug')
    logger = get_logger()

    # Simulate debugging a specific module
    logger.set_context(module="GriftDetection", tier=2)

    logger.debug("Analyzing prompt for grift indicators")
    logger.debug("Checking for opacity pattern")
    logger.debug("Checking for urgency tactics")
    logger.debug("Checking for vague promises")

    logger.info("Module triggered - 3/5 patterns matched")

    logger.clear_context()

    print("\n[OK] Context-based debugging")
    return True


def test_performance_logging():
    """Test 5: Performance metric logging"""
    print("\n" + "="*60)
    print("TEST 5: Performance Logging")
    print("="*60)

    set_verbosity('debug')
    logger = get_logger()

    import time

    # Simulate operations with timing
    operations = [
        ("module_loading", 0.1),
        ("trigger_analysis", 0.05),
        ("dependency_resolution", 0.02),
        ("arbitration", 0.01),
        ("prompt_assembly", 0.03),
        ("llm_execution", 1.5),
    ]

    for operation, duration in operations:
        time.sleep(duration / 10)  # Simulate but faster
        logger.log_performance_metric(operation, duration)

    print("\n[OK] Performance logging")
    return True


def test_verbosity_control():
    """Test 6: Dynamic verbosity control"""
    print("\n" + "="*60)
    print("TEST 6: Dynamic Verbosity Control")
    print("="*60)

    logger = get_logger()

    print("\nSetting to ERROR (should only see errors):")
    set_verbosity('error')
    logger.debug("Debug message")
    logger.info("Info message")
    logger.error("Error message")

    print("\nSetting to DEBUG (should see everything):")
    set_verbosity('debug')
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")

    print("\n[OK] Dynamic verbosity control")
    return True


async def main():
    """Run all integration tests"""
    print("="*60)
    print("LOGGING INTEGRATION TEST SUITE")
    print("="*60)

    tests = [
        ("Module Loading with Logging", lambda: test_logging_with_module_loading()),
        ("Orchestrator with Logging", test_logging_with_orchestrator),
        ("Structured Error Logging", lambda: test_error_logging()),
        ("Context-Based Debugging", lambda: test_context_based_logging()),
        ("Performance Logging", lambda: test_performance_logging()),
        ("Dynamic Verbosity Control", lambda: test_verbosity_control()),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{test_name}' failed: {e}")
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
        print("\n[SUCCESS] All integration tests passed!")
        return 0
    else:
        print(f"\n[PARTIAL] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
