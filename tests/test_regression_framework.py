"""
Test Regression Framework

Verifies that the regression test framework:
1. Captures failed tests correctly
2. Stores them in structured format
3. Can run regression tests
4. Auto-marks tests as fixed when they pass
5. Tracks regression counts
6. Integrates with existing test suites
"""

import asyncio
import sys
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protocol_ai import Module, ModuleLoader, Orchestrator
from tests.regression_framework import (
    RegressionTest,
    RegressionTestCapture,
    RegressionTestRunner
)


class MockLLM:
    """Mock LLM for testing"""
    def __init__(self):
        self.should_trigger = set()  # Modules to simulate triggering

    def execute(self, prompt):
        return "Mock LLM response"


class MockOrchestrator:
    """Mock orchestrator for controlled testing"""
    def __init__(self, triggered_modules=None):
        self.triggered_modules = triggered_modules or []

    async def process_prompt(self, prompt):
        return {
            'triggered_modules': self.triggered_modules,
            'llm_response': 'Mock response'
        }


def test_capture_failure():
    """Test 1: Capturing a failed test"""
    print("="*60)
    print("TEST 1: Capture Failed Test")
    print("="*60)

    # Use temporary directory for testing
    test_dir = "./tests/regression_test_temp"
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    try:
        capture = RegressionTestCapture(regression_dir=test_dir)

        # Capture a failure
        test_id = capture.capture_failure(
            test_id="TEST-001",
            name="Test Grift Detection",
            category="grift_tests",
            prompt="Test prompt for grift detection",
            expected_modules=["GriftDetection"],
            actual_modules=[],
            expected_behavior="Should detect grift patterns",
            failure_reason="GriftDetection module did not trigger"
        )

        print(f"\n[OK] Captured failure with ID: {test_id}")

        # Verify it was saved
        if test_id in capture.regression_tests:
            test = capture.regression_tests[test_id]
            print(f"[OK] Test stored: {test.name}")
            print(f"[OK] Expected modules: {test.expected_modules}")
            print(f"[OK] Failure reason: {test.failure_reason}")
            return True
        else:
            print("[FAIL] Test not stored")
            return False

    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)


def test_regression_count():
    """Test 2: Track regression count"""
    print("\n" + "="*60)
    print("TEST 2: Track Regression Count")
    print("="*60)

    test_dir = "./tests/regression_test_temp2"
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    try:
        capture = RegressionTestCapture(regression_dir=test_dir)

        # Capture same failure multiple times
        for i in range(3):
            capture.capture_failure(
                test_id="TEST-002",
                name="Recurring Failure",
                category="test",
                prompt="Test prompt",
                expected_modules=["TestModule"],
                actual_modules=[],
                expected_behavior="Should trigger",
                failure_reason=f"Failure attempt {i+1}"
            )

        test = capture.regression_tests["TEST-002"]
        print(f"\n[OK] Regression count: {test.regression_count}")

        if test.regression_count == 2:  # 0-indexed, so 2 regressions after first capture
            print("[OK] Regression count tracked correctly")
            return True
        else:
            print(f"[FAIL] Expected regression_count=2, got {test.regression_count}")
            return False

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_mark_fixed():
    """Test 3: Mark test as fixed"""
    print("\n" + "="*60)
    print("TEST 3: Mark Test as Fixed")
    print("="*60)

    test_dir = "./tests/regression_test_temp3"
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    try:
        capture = RegressionTestCapture(regression_dir=test_dir)

        # Capture failure
        capture.capture_failure(
            test_id="TEST-003",
            name="Test to be fixed",
            category="test",
            prompt="Test prompt",
            expected_modules=["TestModule"],
            actual_modules=[],
            expected_behavior="Should trigger",
            failure_reason="Module not triggering"
        )

        # Mark as fixed
        capture.mark_fixed("TEST-003", "Fixed by adding trigger keyword")

        test = capture.regression_tests["TEST-003"]

        if test.fixed_date and test.fix_description:
            print(f"\n[OK] Test marked as fixed")
            print(f"[OK] Fix date: {test.fixed_date}")
            print(f"[OK] Fix description: {test.fix_description}")
            return True
        else:
            print("[FAIL] Test not marked as fixed correctly")
            return False

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_get_active_regressions():
    """Test 4: Get active (unfixed) regressions"""
    print("\n" + "="*60)
    print("TEST 4: Get Active Regressions")
    print("="*60)

    test_dir = "./tests/regression_test_temp4"
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    try:
        capture = RegressionTestCapture(regression_dir=test_dir)

        # Capture 3 failures
        for i in range(3):
            capture.capture_failure(
                test_id=f"TEST-00{i+4}",
                name=f"Test {i+4}",
                category="test",
                prompt="Test",
                expected_modules=["TestModule"],
                actual_modules=[],
                expected_behavior="Should work",
                failure_reason="Failed"
            )

        # Fix one of them
        capture.mark_fixed("TEST-004", "Fixed")

        # Get active regressions
        active = capture.get_active_regressions()

        print(f"\n[OK] Total tests: 3")
        print(f"[OK] Active regressions: {len(active)}")

        if len(active) == 2:
            print("[OK] Correct number of active regressions")
            return True
        else:
            print(f"[FAIL] Expected 2 active, got {len(active)}")
            return False

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


async def test_run_regression_test():
    """Test 5: Run a regression test"""
    print("\n" + "="*60)
    print("TEST 5: Run Regression Test")
    print("="*60)

    test_dir = "./tests/regression_test_temp5"
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    try:
        capture = RegressionTestCapture(regression_dir=test_dir)

        # Capture a failure
        capture.capture_failure(
            test_id="TEST-007",
            name="Test Run",
            category="test",
            prompt="Test grift detection",
            expected_modules=["GriftDetection"],
            actual_modules=[],
            expected_behavior="Should detect grift",
            failure_reason="Module not triggered"
        )

        # Create mock orchestrator that now triggers the module
        mock_orch = MockOrchestrator(triggered_modules=["GriftDetection"])

        # Run regression test
        runner = RegressionTestRunner(capture, mock_orch)
        test = capture.regression_tests["TEST-007"]
        result = await runner.run_regression_test(test)

        print(f"\n[OK] Test executed")
        print(f"[OK] Test passed: {result['passed']}")

        if result['passed']:
            print("[OK] Regression test passed (module now triggers)")
            return True
        else:
            print("[FAIL] Regression test did not pass")
            return False

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


async def test_auto_fix_detection():
    """Test 6: Auto-detect when test is fixed"""
    print("\n" + "="*60)
    print("TEST 6: Auto-Fix Detection")
    print("="*60)

    test_dir = "./tests/regression_test_temp6"
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    try:
        capture = RegressionTestCapture(regression_dir=test_dir)

        # Capture failure
        capture.capture_failure(
            test_id="TEST-008",
            name="Auto-fix test",
            category="test",
            prompt="Test",
            expected_modules=["TestModule"],
            actual_modules=[],
            expected_behavior="Should work",
            failure_reason="Failed"
        )

        # Verify not fixed yet
        test = capture.regression_tests["TEST-008"]
        if test.fixed_date:
            print("[FAIL] Test should not be marked as fixed yet")
            return False

        # Create orchestrator that triggers the module
        mock_orch = MockOrchestrator(triggered_modules=["TestModule"])

        # Run regression tests (should auto-detect fix)
        runner = RegressionTestRunner(capture, mock_orch)
        await runner.run_all_regressions(only_active=True)

        # Check if auto-marked as fixed
        test = capture.regression_tests["TEST-008"]
        if test.fixed_date:
            print(f"\n[OK] Test auto-marked as fixed")
            print(f"[OK] Fix description: {test.fix_description}")
            return True
        else:
            print("[FAIL] Test was not auto-marked as fixed")
            return False

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_statistics():
    """Test 7: Get regression statistics"""
    print("\n" + "="*60)
    print("TEST 7: Regression Statistics")
    print("="*60)

    test_dir = "./tests/regression_test_temp7"
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    try:
        capture = RegressionTestCapture(regression_dir=test_dir)

        # Create some regression tests
        for i in range(5):
            test_id = f"TEST-{9+i:03d}"
            capture.capture_failure(
                test_id=test_id,
                name=f"Test {9+i}",
                category="test",
                prompt="Test",
                expected_modules=["TestModule"],
                actual_modules=[],
                expected_behavior="Should work",
                failure_reason="Failed"
            )

        # Fix 2 of them
        capture.mark_fixed("TEST-009", "Fixed")
        capture.mark_fixed("TEST-010", "Fixed")

        # Get statistics
        stats = capture.get_statistics()

        print(f"\n[OK] Total regressions: {stats['total_regressions']}")
        print(f"[OK] Active: {stats['active_regressions']}")
        print(f"[OK] Fixed: {stats['fixed_regressions']}")
        print(f"[OK] Fix rate: {stats['fix_rate']*100:.1f}%")

        if stats['total_regressions'] == 5 and stats['fixed_regressions'] == 2:
            print("[OK] Statistics correct")
            return True
        else:
            print("[FAIL] Statistics incorrect")
            return False

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


async def main():
    """Run all tests"""
    print("="*60)
    print("REGRESSION FRAMEWORK TEST SUITE")
    print("="*60)

    tests = [
        ("Capture Failed Test", lambda: test_capture_failure()),
        ("Track Regression Count", lambda: test_regression_count()),
        ("Mark Test as Fixed", lambda: test_mark_fixed()),
        ("Get Active Regressions", lambda: test_get_active_regressions()),
        ("Run Regression Test", test_run_regression_test),
        ("Auto-Fix Detection", test_auto_fix_detection),
        ("Regression Statistics", lambda: test_statistics()),
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
        print("\n[SUCCESS] All regression framework tests passed!")
        return 0
    else:
        print(f"\n[PARTIAL] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
