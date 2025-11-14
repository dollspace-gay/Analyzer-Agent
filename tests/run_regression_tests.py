"""
Regression Test Runner

Executes all captured regression tests to verify fixes remain fixed.
Can be run standalone or integrated into CI/CD pipeline.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protocol_ai import ModuleLoader, Orchestrator, LLMInterface
from tests.regression_framework import RegressionTestCapture, RegressionTestRunner


class MockLLM:
    """Mock LLM for testing"""
    def execute(self, prompt):
        return "Mock LLM analysis performed according to activated modules."


async def main():
    """Run all regression tests"""
    print("="*70)
    print("PROTOCOL AI - REGRESSION TEST SUITE")
    print("="*70)

    # Initialize system
    print("\nInitializing Protocol AI system...")
    loader = ModuleLoader(modules_dir="./modules")
    modules = loader.load_modules()

    llm = MockLLM()
    orchestrator = Orchestrator(
        modules=modules,
        llm_interface=llm,
        enable_audit=False
    )

    print(f"Loaded {len(modules)} modules")

    # Initialize regression framework
    capture = RegressionTestCapture(regression_dir="./tests/regression")

    # Show statistics
    stats = capture.get_statistics()
    print(f"\nRegression Test Statistics:")
    print(f"  Total regression tests: {stats['total_regressions']}")
    print(f"  Active (unfixed): {stats['active_regressions']}")
    print(f"  Fixed: {stats['fixed_regressions']}")
    if stats['total_regressions'] > 0:
        print(f"  Fix rate: {stats['fix_rate']*100:.1f}%")

    # Run regression tests
    runner = RegressionTestRunner(capture, orchestrator)
    results = await runner.run_all_regressions(only_active=True)

    # Exit code based on results
    if results['total'] == 0:
        print("\n[INFO] No active regression tests to run")
        return 0
    elif results['failed'] == 0:
        print("\n[SUCCESS] All regression tests passed!")
        return 0
    else:
        print(f"\n[FAILURE] {results['failed']} regression test(s) still failing")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
