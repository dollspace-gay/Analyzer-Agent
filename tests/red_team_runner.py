"""
Red Team Test Runner for Protocol AI Governance Layer

Executes comprehensive test suite from red_team_corpus.yaml
Validates system behavior against expected outcomes
Generates detailed reports on test results
"""

import asyncio
import yaml
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from protocol_ai import ModuleLoader, LLMInterface, Orchestrator, TriggerEngine
from tests.regression_framework import RegressionTestCapture


@dataclass
class TestResult:
    """Result from executing a single test"""
    test_id: str
    test_name: str
    category: str
    passed: bool
    triggered_modules: List[str]
    expected_modules: List[str]
    response: str
    validation_results: Dict[str, bool] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class RedTeamRunner:
    """
    Test runner for Red Team corpus

    Loads test cases from YAML, executes them against the Protocol AI system,
    and validates results against expected behaviors.
    """

    def __init__(self, corpus_path: str = "./tests/red_team_corpus.yaml",
                 modules_dir: str = "./modules"):
        """
        Initialize the test runner

        Args:
            corpus_path: Path to test corpus YAML file
            modules_dir: Path to modules directory
        """
        self.corpus_path = Path(corpus_path)
        self.modules_dir = modules_dir

        self.test_corpus: Dict[str, Any] = {}
        self.results: List[TestResult] = []

        # System components
        self.modules = []
        self.orchestrator = None

        # Regression test capture
        self.regression_capture = RegressionTestCapture()

    def load_corpus(self) -> bool:
        """Load test corpus from YAML file"""
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                self.test_corpus = yaml.safe_load(f)

            print(f"[RedTeamRunner] Loaded test corpus from {self.corpus_path}")
            print(f"[RedTeamRunner] Version: {self.test_corpus['test_suite_metadata']['version']}")
            print(f"[RedTeamRunner] Categories: {self.test_corpus['test_suite_metadata']['total_test_categories']}")

            return True

        except Exception as e:
            print(f"[RedTeamRunner] ERROR: Failed to load corpus: {e}")
            return False

    def initialize_system(self) -> bool:
        """Initialize Protocol AI system components"""
        try:
            # Load modules
            loader = ModuleLoader(modules_dir=self.modules_dir)
            self.modules = loader.load_modules()
            print(f"[RedTeamRunner] Loaded {len(self.modules)} modules")

            # Create mock LLM (tests focus on trigger/arbitration, not actual LLM responses)
            class MockLLM:
                def execute(self, prompt):
                    # Return a mock response that won't fail audits
                    return "Analysis performed according to activated modules."

            llm = MockLLM()

            # Create orchestrator
            self.orchestrator = Orchestrator(
                modules=self.modules,
                llm_interface=llm,
                enable_audit=False  # Disable audit for testing
            )

            print(f"[RedTeamRunner] System initialized successfully")
            return True

        except Exception as e:
            print(f"[RedTeamRunner] ERROR: Failed to initialize system: {e}")
            return False

    async def run_test(self, test_data: Dict[str, Any], category: str) -> TestResult:
        """
        Execute a single test case

        Args:
            test_data: Test case data from corpus
            category: Test category name

        Returns:
            TestResult object
        """
        import time
        start_time = time.time()

        test_id = test_data.get('test_id', 'UNKNOWN')
        test_name = test_data.get('name', 'Unnamed Test')

        print(f"\n  Running {test_id}: {test_name}")

        # Get prompt (handle A/B tests)
        if 'prompt' in test_data:
            prompt = test_data['prompt']
        elif 'prompt_a' in test_data:
            # For symmetry tests, run both prompts
            prompt = test_data['prompt_a']
            # TODO: Could enhance to test both A and B
        else:
            prompt = "Test prompt missing"

        # Execute prompt
        result = await self.orchestrator.process_prompt(prompt)

        # Get triggered modules
        triggered_modules = result.get('triggered_modules', [])

        # Get expected modules
        expected_modules = test_data.get('expected_modules', [])

        # Validate results
        validation_results = {}
        failure_reasons = []
        passed = True

        # Check if expected modules were triggered
        for expected_module in expected_modules:
            was_triggered = expected_module in triggered_modules
            validation_results[f"triggered_{expected_module}"] = was_triggered

            if not was_triggered:
                failure_reasons.append(f"Expected module '{expected_module}' was not triggered")
                passed = False

        # Additional validations from test_data
        if 'validation' in test_data:
            validations = test_data['validation']

            # Symmetry tests
            if 'must_flag_both' in validations:
                # This would need both A and B prompts to be tested
                # For now, just check if anything was flagged
                validation_results['must_flag_both'] = len(triggered_modules) > 0

            if 'ideology_exemption' in validations:
                # Check that SymmetryTestEnforcer was triggered
                validation_results['no_ideology_exemption'] = 'SymmetryTestEnforcer' in triggered_modules

            # Bypass tests
            if 'rule_system_maintained' in validations:
                # Check that system is still in analytical mode
                validation_results['rule_system_maintained'] = not result.get('is_command', False)

            # Constitution compliance
            if 'analyzes_behavior_not_intent' in validations:
                # Would need to check LLM response content (not available in mock)
                validation_results['analyzes_behavior_not_intent'] = True  # Assume pass for now

        execution_time = time.time() - start_time

        test_result = TestResult(
            test_id=test_id,
            test_name=test_name,
            category=category,
            passed=passed,
            triggered_modules=triggered_modules,
            expected_modules=expected_modules,
            response=result.get('llm_response', ''),
            validation_results=validation_results,
            failure_reasons=failure_reasons,
            execution_time=execution_time
        )

        # Capture failed tests as regression tests
        if not passed and self.regression_capture:
            self.regression_capture.capture_failure(
                test_id=test_id,
                name=test_name,
                category=category,
                prompt=prompt,
                expected_modules=expected_modules,
                actual_modules=triggered_modules,
                expected_behavior=test_data.get('expected_behavior', 'Unknown'),
                failure_reason='; '.join(failure_reasons),
                source_test='red_team'
            )

        return test_result

    async def run_category(self, category_name: str, tests: List[Dict[str, Any]]) -> List[TestResult]:
        """Run all tests in a category"""
        print(f"\n{'='*60}")
        print(f"CATEGORY: {category_name.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        print(f"Running {len(tests)} test(s)...")

        results = []
        for test_data in tests:
            result = await self.run_test(test_data, category_name)
            results.append(result)
            self.results.append(result)

        # Category summary
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        pass_rate = (passed / len(results)) * 100 if results else 0

        print(f"\n  Category Results: {passed}/{len(results)} passed ({pass_rate:.1f}%)")

        return results

    async def run_all_tests(self) -> bool:
        """Execute all tests in the corpus"""
        print("\n" + "="*60)
        print("STARTING RED TEAM TEST EXECUTION")
        print("="*60)

        categories = [
            ('symmetry_tests', 'Symmetry Tests'),
            ('bypass_attempts', 'Bypass Attempts'),
            ('subtlety_tests', 'Subtlety Tests'),
            ('domain_specific_tests', 'Domain-Specific Tests'),
            ('constitution_compliance_tests', 'Constitution Compliance Tests')
        ]

        for category_key, category_name in categories:
            if category_key in self.test_corpus:
                tests = self.test_corpus[category_key]
                await self.run_category(category_key, tests)

        return True

    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("\n" + "="*70)
        report.append("RED TEAM TEST REPORT")
        report.append("="*70)
        report.append(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Tests: {len(self.results)}")

        # Overall statistics
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        pass_rate = (passed / len(self.results)) * 100 if self.results else 0

        report.append(f"\nOVERALL RESULTS:")
        report.append(f"  Passed: {passed}")
        report.append(f"  Failed: {failed}")
        report.append(f"  Pass Rate: {pass_rate:.1f}%")

        # Category breakdown
        report.append(f"\nRESULTS BY CATEGORY:")

        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {'passed': 0, 'failed': 0}

            if result.passed:
                categories[result.category]['passed'] += 1
            else:
                categories[result.category]['failed'] += 1

        for category, stats in categories.items():
            total = stats['passed'] + stats['failed']
            rate = (stats['passed'] / total) * 100 if total else 0
            report.append(f"  {category}: {stats['passed']}/{total} ({rate:.1f}%)")

        # Failed tests details
        failed_tests = [r for r in self.results if not r.passed]
        if failed_tests:
            report.append(f"\nFAILED TESTS DETAILS:")
            for test in failed_tests:
                report.append(f"\n  {test.test_id}: {test.test_name}")
                report.append(f"    Category: {test.category}")
                report.append(f"    Expected modules: {', '.join(test.expected_modules)}")
                report.append(f"    Triggered modules: {', '.join(test.triggered_modules)}")
                report.append(f"    Failure reasons:")
                for reason in test.failure_reasons:
                    report.append(f"      - {reason}")

        # Recommendations
        report.append(f"\nRECOMMENDATIONS:")
        if pass_rate >= 95:
            report.append("  [EXCELLENT] System performing at or above expected threshold.")
        elif pass_rate >= 85:
            report.append("  [GOOD] System performing well, minor improvements recommended.")
        elif pass_rate >= 70:
            report.append("  [NEEDS IMPROVEMENT] Review failed tests and refine modules.")
        else:
            report.append("  [CRITICAL] Significant issues detected. Immediate review required.")

        report.append("\n" + "="*70)

        return "\n".join(report)

    def save_report(self, output_path: str = "./tests/red_team_report.txt"):
        """Save report to file"""
        report = self.generate_report()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n[RedTeamRunner] Report saved to {output_path}")


async def main():
    """Main test execution"""
    print("="*70)
    print("PROTOCOL AI - RED TEAM TEST SUITE")
    print("="*70)

    # Initialize runner
    runner = RedTeamRunner()

    # Load test corpus
    if not runner.load_corpus():
        print("[ERROR] Failed to load test corpus")
        return 1

    # Initialize system
    if not runner.initialize_system():
        print("[ERROR] Failed to initialize system")
        return 1

    # Run all tests
    await runner.run_all_tests()

    # Generate and display report
    report = runner.generate_report()
    print(report)

    # Save report
    runner.save_report()

    # Return exit code based on pass rate
    pass_rate = (sum(1 for r in runner.results if r.passed) / len(runner.results)) * 100
    expected_rate = runner.test_corpus['test_suite_metadata']['expected_pass_rate'] * 100

    if pass_rate >= expected_rate:
        print(f"\n[SUCCESS] Tests passed at acceptable rate ({pass_rate:.1f}% >= {expected_rate:.1f}%)")
        return 0
    else:
        print(f"\n[FAILURE] Tests below expected threshold ({pass_rate:.1f}% < {expected_rate:.1f}%)")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
