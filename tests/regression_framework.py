"""
Regression Test Framework for Protocol AI Governance Layer

Automatically captures failed test cases and stores them as regression tests
to ensure that once bugs are fixed, they don't get reintroduced.

Features:
- Automatic capture of failed test cases
- Structured storage in YAML format
- Regression test runner
- Integration with existing test suites
- Test metadata tracking (when failed, when fixed, etc.)
"""

import asyncio
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class RegressionTest:
    """
    Represents a single regression test case.

    Captures all information needed to reproduce and verify a fix.
    """
    test_id: str
    name: str
    category: str
    prompt: str
    expected_modules: List[str]
    expected_behavior: str

    # Failure information
    failed_date: str
    failure_reason: str
    actual_modules: List[str] = field(default_factory=list)

    # Fix information (optional)
    fixed_date: Optional[str] = None
    fix_description: Optional[str] = None

    # Metadata
    source_test: Optional[str] = None  # Original test ID if from another suite
    regression_count: int = 0  # How many times this has regressed
    last_verified: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


class RegressionTestCapture:
    """
    Captures failed test cases and stores them as regression tests.
    """

    def __init__(self, regression_dir: str = "./tests/regression"):
        """
        Initialize the regression test capture system.

        Args:
            regression_dir: Directory to store regression test files
        """
        self.regression_dir = Path(regression_dir)
        self.regression_dir.mkdir(parents=True, exist_ok=True)

        self.regression_file = self.regression_dir / "regression_tests.yaml"
        self.regression_tests: Dict[str, RegressionTest] = {}

        # Load existing regression tests
        self._load_regression_tests()

    def _load_regression_tests(self):
        """Load existing regression tests from file."""
        if self.regression_file.exists():
            with open(self.regression_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if data and 'regression_tests' in data:
                for test_data in data['regression_tests']:
                    test = RegressionTest(**test_data)
                    self.regression_tests[test.test_id] = test

    def _save_regression_tests(self):
        """Save regression tests to file."""
        data = {
            'regression_test_metadata': {
                'version': '1.0.0',
                'last_updated': datetime.now().isoformat(),
                'total_tests': len(self.regression_tests)
            },
            'regression_tests': [asdict(test) for test in self.regression_tests.values()]
        }

        with open(self.regression_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def capture_failure(self, test_id: str, name: str, category: str,
                       prompt: str, expected_modules: List[str],
                       actual_modules: List[str], expected_behavior: str,
                       failure_reason: str, source_test: Optional[str] = None) -> str:
        """
        Capture a failed test case as a regression test.

        Args:
            test_id: Original test ID
            name: Test name
            category: Test category
            prompt: The prompt that failed
            expected_modules: Modules that should have triggered
            actual_modules: Modules that actually triggered
            expected_behavior: What should have happened
            failure_reason: Why the test failed
            source_test: ID of source test suite (e.g., "red_team")

        Returns:
            Regression test ID
        """
        # Check if this test already exists as a regression test
        if test_id in self.regression_tests:
            # Test regressed again!
            existing = self.regression_tests[test_id]
            existing.regression_count += 1
            existing.failed_date = datetime.now().isoformat()
            existing.failure_reason = failure_reason
            existing.actual_modules = actual_modules
            existing.fixed_date = None  # No longer fixed
            existing.fix_description = None

            print(f"[REGRESSION] Test {test_id} regressed (count: {existing.regression_count})")
        else:
            # New regression test
            regression_test = RegressionTest(
                test_id=test_id,
                name=name,
                category=category,
                prompt=prompt,
                expected_modules=expected_modules,
                expected_behavior=expected_behavior,
                failed_date=datetime.now().isoformat(),
                failure_reason=failure_reason,
                actual_modules=actual_modules,
                source_test=source_test
            )

            self.regression_tests[test_id] = regression_test
            print(f"[CAPTURED] New regression test: {test_id}")

        self._save_regression_tests()
        return test_id

    def mark_fixed(self, test_id: str, fix_description: str):
        """
        Mark a regression test as fixed.

        Args:
            test_id: Regression test ID
            fix_description: Description of the fix
        """
        if test_id in self.regression_tests:
            test = self.regression_tests[test_id]
            test.fixed_date = datetime.now().isoformat()
            test.fix_description = fix_description
            test.last_verified = datetime.now().isoformat()

            self._save_regression_tests()
            print(f"[FIXED] Regression test {test_id} marked as fixed")
        else:
            print(f"[ERROR] Regression test {test_id} not found")

    def get_active_regressions(self) -> List[RegressionTest]:
        """
        Get all regression tests that are not yet fixed.

        Returns:
            List of unfixed regression tests
        """
        return [test for test in self.regression_tests.values() if test.fixed_date is None]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get regression test statistics.

        Returns:
            Dictionary of statistics
        """
        active = self.get_active_regressions()
        fixed = [test for test in self.regression_tests.values() if test.fixed_date is not None]

        return {
            'total_regressions': len(self.regression_tests),
            'active_regressions': len(active),
            'fixed_regressions': len(fixed),
            'fix_rate': len(fixed) / len(self.regression_tests) if self.regression_tests else 0,
            'most_regressed': max(self.regression_tests.values(),
                                 key=lambda t: t.regression_count,
                                 default=None)
        }


class RegressionTestRunner:
    """
    Runs regression tests to verify fixes remain fixed.
    """

    def __init__(self, capture: RegressionTestCapture, orchestrator):
        """
        Initialize the regression test runner.

        Args:
            capture: RegressionTestCapture instance
            orchestrator: Orchestrator instance to run tests against
        """
        self.capture = capture
        self.orchestrator = orchestrator

    async def run_regression_test(self, test: RegressionTest) -> Dict[str, Any]:
        """
        Run a single regression test.

        Args:
            test: RegressionTest to execute

        Returns:
            Test result dictionary
        """
        print(f"\n  Running regression test: {test.test_id}")

        # Execute the prompt
        result = await self.orchestrator.process_prompt(test.prompt)

        # Get triggered modules
        triggered_modules = result.get('triggered_modules', [])

        # Check if expected modules were triggered
        passed = True
        missing_modules = []

        for expected_module in test.expected_modules:
            if expected_module not in triggered_modules:
                passed = False
                missing_modules.append(expected_module)

        return {
            'test_id': test.test_id,
            'test_name': test.name,
            'passed': passed,
            'triggered_modules': triggered_modules,
            'expected_modules': test.expected_modules,
            'missing_modules': missing_modules,
            'was_fixed': test.fixed_date is not None
        }

    async def run_all_regressions(self, only_active: bool = True) -> Dict[str, Any]:
        """
        Run all regression tests.

        Args:
            only_active: If True, only run unfixed regression tests

        Returns:
            Summary of regression test results
        """
        print("\n" + "="*60)
        print("RUNNING REGRESSION TESTS")
        print("="*60)

        # Get tests to run
        if only_active:
            tests = self.capture.get_active_regressions()
            print(f"\nRunning {len(tests)} active regression test(s)")
        else:
            tests = list(self.capture.regression_tests.values())
            print(f"\nRunning {len(tests)} total regression test(s)")

        if not tests:
            print("\nNo regression tests to run")
            return {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'results': []
            }

        # Run tests
        results = []
        for test in tests:
            result = await self.run_regression_test(test)
            results.append(result)

            # Auto-mark as fixed if it passes and was previously unfixed
            if result['passed'] and not result['was_fixed']:
                print(f"    [AUTO-FIX] Test {test.test_id} now passes - marking as fixed")
                self.capture.mark_fixed(test.test_id, "Automatically detected as passing")

        # Summary
        passed = sum(1 for r in results if r['passed'])
        failed = len(results) - passed

        print(f"\n{'='*60}")
        print(f"REGRESSION TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {len(results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Pass Rate: {passed/len(results)*100:.1f}%")

        # Show failed tests
        if failed > 0:
            print(f"\nFailed Regression Tests:")
            for result in results:
                if not result['passed']:
                    print(f"  - {result['test_id']}: {result['test_name']}")
                    print(f"    Missing modules: {', '.join(result['missing_modules'])}")

        return {
            'total': len(results),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(results) if results else 0,
            'results': results
        }


def integrate_with_test_suite(test_runner_class):
    """
    Decorator to integrate regression capture with existing test runners.

    Usage:
        @integrate_with_test_suite
        class MyTestRunner:
            ...
    """
    original_run_test = test_runner_class.run_test

    async def run_test_with_capture(self, test_data, category):
        """Wrapped run_test that captures failures."""
        # Run the original test
        result = await original_run_test(self, test_data, category)

        # If test failed, capture it as regression test
        if not result.passed:
            if not hasattr(self, 'regression_capture'):
                self.regression_capture = RegressionTestCapture()

            self.regression_capture.capture_failure(
                test_id=result.test_id,
                name=result.test_name,
                category=result.category,
                prompt=test_data.get('prompt', test_data.get('prompt_a', '')),
                expected_modules=result.expected_modules,
                actual_modules=result.triggered_modules,
                expected_behavior=test_data.get('expected_behavior', 'Unknown'),
                failure_reason='; '.join(result.failure_reasons),
                source_test='red_team'
            )

        return result

    # Replace the method
    test_runner_class.run_test = run_test_with_capture

    return test_runner_class
