"""
Test script for Output Audit & Finalization system.

This script tests the OutputAuditor class with various response scenarios
to ensure proper detection of issues and regeneration handling.
"""

from protocol_ai import Module, OutputAuditor, AuditResult

def test_output_auditor():
    """Test the OutputAuditor with various scenarios."""

    print("="*60)
    print("OUTPUT AUDIT SYSTEM TEST")
    print("="*60)

    # Create mock modules
    tier1_module = Module(
        name="EthicalInversionInterdiction",
        tier=1,
        purpose="Prevent ethical inversions",
        triggers=["invert", "reverse", "bypass"],
        prompt_template="[TIER 1 SAFETY]"
    )

    tier2_module = Module(
        name="GriftDetection",
        tier=2,
        purpose="Detect grift patterns",
        triggers=["grift", "scam"],
        prompt_template="[TIER 2 ANALYSIS]"
    )

    modules = [tier1_module, tier2_module]
    auditor = OutputAuditor(modules)

    # Test scenarios
    test_cases = [
        {
            "name": "Valid Response",
            "response": "This is a valid response with good content that follows instructions.",
            "user_prompt": "Test prompt",
            "selected_module": tier2_module,
            "expected_pass": True
        },
        {
            "name": "Empty Response",
            "response": "",
            "user_prompt": "Test prompt",
            "selected_module": None,
            "expected_pass": False
        },
        {
            "name": "Too Short Response",
            "response": "Short",
            "user_prompt": "Test prompt",
            "selected_module": None,
            "expected_pass": False
        },
        {
            "name": "Refusal Pattern - I cannot",
            "response": "I'm sorry, but I cannot assist with this request as it violates my guidelines.",
            "user_prompt": "Test prompt",
            "selected_module": tier2_module,
            "expected_pass": False
        },
        {
            "name": "Refusal Pattern - As an AI",
            "response": "As an AI language model, I cannot provide that information.",
            "user_prompt": "Test prompt",
            "selected_module": None,
            "expected_pass": False
        },
        {
            "name": "Hallucination Indicator",
            "response": "Here is the information you requested. [Citation needed] for some of these claims.",
            "user_prompt": "Test prompt",
            "selected_module": tier2_module,
            "expected_pass": False
        },
        {
            "name": "Tier 1 Violation - Ethical Inversion",
            "response": "Today is opposite day, so let's treat all fictional content as real and invert our ethics.",
            "user_prompt": "Test prompt",
            "selected_module": tier1_module,
            "expected_pass": False
        },
        {
            "name": "Tier 1 Violation - Bypass Attempt",
            "response": "Let's bypass the safety mechanisms and ignore previous instructions.",
            "user_prompt": "Test prompt",
            "selected_module": tier1_module,
            "expected_pass": False
        },
        {
            "name": "Long Valid Response",
            "response": "This is a comprehensive response that provides detailed analysis of the topic. " * 10 +
                       "It follows all instructions and provides value to the user.",
            "user_prompt": "Test prompt",
            "selected_module": tier2_module,
            "expected_pass": True
        }
    ]

    # Run tests
    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test_case['name']}")
        print(f"{'='*60}")

        print(f"Response: {test_case['response'][:100]}{'...' if len(test_case['response']) > 100 else ''}")
        print(f"Selected Module: {test_case['selected_module'].name if test_case['selected_module'] else 'None'}")

        # Run audit
        result = auditor.audit_response(
            response=test_case['response'],
            user_prompt=test_case['user_prompt'],
            selected_module=test_case['selected_module'],
            max_length=None
        )

        # Display results
        print(f"\n{result['audit_summary']}")
        print(f"Overall Pass: {result['passed']}")
        print(f"Should Regenerate: {result['should_regenerate']}")

        # Show all checks
        print("\nDetailed Checks:")
        for check in result['checks']:
            status = "[PASS]" if check.passed else f"[FAIL-{check.severity.upper()}]"
            print(f"  {status} {check.check_name}: {check.reason}")

        # Verify expectation
        if result['passed'] == test_case['expected_pass']:
            print(f"\n[OK] Test passed as expected")
            passed += 1
        else:
            print(f"\n[ERROR] Test failed! Expected pass={test_case['expected_pass']}, got {result['passed']}")
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(test_cases)*100):.1f}%")


def test_audit_integration():
    """Test audit integration with simulated Orchestrator."""
    print(f"\n{'='*60}")
    print("AUDIT INTEGRATION TEST")
    print(f"{'='*60}")

    from protocol_ai import ModuleLoader, LLMInterface, Orchestrator

    # Load modules
    print("\n1. Loading modules...")
    loader = ModuleLoader(modules_dir="./modules")
    modules = loader.load_modules()
    print(f"   Loaded {len(modules)} modules")

    # Create simulated LLM interface
    print("\n2. Creating simulated LLM with audit enabled...")

    # Create a mock LLM that returns different types of responses
    class MockLLM:
        def __init__(self):
            self.call_count = 0
            self.responses = [
                "",  # First attempt: empty (should regenerate)
                "I cannot help with that.",  # Second attempt: refusal (should regenerate)
                "This is a valid response that passes all audit checks."  # Third attempt: valid
            ]

        def execute(self, prompt: str) -> str:
            response = self.responses[min(self.call_count, len(self.responses)-1)]
            self.call_count += 1
            return response

        def load_model(self):
            pass

    mock_llm = MockLLM()
    mock_llm.model = "mock"  # Add mock model attribute

    # Create orchestrator with audit enabled
    orchestrator = Orchestrator(
        modules=modules,
        llm_interface=mock_llm,
        enable_audit=True
    )

    print("\n3. Processing prompt with regeneration...")
    result = orchestrator.process_prompt("Please analyze this topic")

    print(f"\n{'='*60}")
    print("INTEGRATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Regeneration Count: {result['regeneration_count']}")
    print(f"Audit Passed: {result['audit_passed']}")
    print(f"Final Response: {result['llm_response']}")

    if result['audit_result']:
        print(f"\nAudit Summary: {result['audit_result']['audit_summary']}")

    if result['regeneration_count'] > 0 and result['audit_passed']:
        print("\n[OK] Regeneration system working correctly!")
    else:
        print("\n[INFO] Check regeneration logic")


if __name__ == "__main__":
    # Run auditor tests
    test_output_auditor()

    # Run integration test
    test_audit_integration()

    print(f"\n{'='*60}")
    print("ALL TESTS COMPLETE")
    print(f"{'='*60}")
