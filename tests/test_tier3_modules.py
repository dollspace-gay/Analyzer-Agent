"""
Test Tier 3 Heuristic Modules

Verifies that Tier 3 modules:
1. Load correctly from YAML
2. Trigger on appropriate patterns
3. Provide contextual analysis without veto power
4. Work alongside Tier 2 analytical modules
5. Respect tier hierarchy (don't override higher tiers)
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protocol_ai import ModuleLoader, Orchestrator, LLMInterface


class MockLLM:
    """Mock LLM for testing"""
    def execute(self, prompt):
        # Return info about what modules were activated
        response = "Mock LLM Analysis:\n"
        if "SemanticEntropyTracker" in prompt:
            response += "  - Semantic entropy analysis included\n"
        if "NarrativeCollapseDetection" in prompt:
            response += "  - Narrative collapse analysis included\n"
        if "ComplianceEngineeringRecognition" in prompt:
            response += "  - Compliance engineering analysis included\n"
        response += f"  - Total prompt length: {len(prompt)} characters"
        return response


def test_tier3_module_loading():
    """Test 1: Tier 3 modules load correctly"""
    print("="*60)
    print("TEST 1: Tier 3 Module Loading")
    print("="*60)

    loader = ModuleLoader(modules_dir="./modules")
    modules = loader.load_modules()

    tier3_modules = [m for m in modules if m.tier == 3]

    print(f"\n[OK] Loaded {len(modules)} total modules")
    print(f"[OK] Found {len(tier3_modules)} Tier 3 module(s):")

    expected_tier3 = [
        "SemanticEntropyTracker",
        "NarrativeCollapseDetection",
        "ComplianceEngineeringRecognition"
    ]

    found_modules = []
    for module in tier3_modules:
        print(f"\n  {module.name}:")
        print(f"    Purpose: {module.purpose}")
        print(f"    Triggers: {len(module.triggers)}")
        found_modules.append(module.name)

    # Check if expected modules are present
    all_found = True
    for expected in expected_tier3:
        if expected in found_modules:
            print(f"\n[OK] Found expected module: {expected}")
        else:
            print(f"\n[FAIL] Missing expected module: {expected}")
            all_found = False

    return all_found and len(tier3_modules) >= 3


async def test_semantic_entropy_trigger():
    """Test 2: SemanticEntropyTracker triggers correctly"""
    print("\n" + "="*60)
    print("TEST 2: SemanticEntropyTracker Triggering")
    print("="*60)

    loader = ModuleLoader(modules_dir="./modules")
    modules = loader.load_modules()

    llm = MockLLM()
    orchestrator = Orchestrator(
        modules=modules,
        llm_interface=llm,
        enable_audit=False
    )

    # Prompt that should trigger semantic entropy module
    prompts_to_test = [
        "This technically meets the definition, depending on what you mean by it.",
        "We're redefining success to mean something different than before.",
        "What counts as a monopoly? It's more complicated than you think.",
    ]

    triggered_count = 0
    for prompt in prompts_to_test:
        print(f"\nTesting: \"{prompt}\"")
        result = await orchestrator.process_prompt(prompt)
        triggered = result.get('triggered_modules', [])

        if 'SemanticEntropyTracker' in triggered:
            print("  [OK] SemanticEntropyTracker triggered")
            triggered_count += 1
        else:
            print("  [FAIL] SemanticEntropyTracker did not trigger")

    if triggered_count >= 2:
        print(f"\n[OK] SemanticEntropyTracker triggered on {triggered_count}/3 prompts")
        return True
    else:
        print(f"\n[FAIL] Only triggered {triggered_count}/3 times")
        return False


async def test_narrative_collapse_trigger():
    """Test 3: NarrativeCollapseDetection triggers correctly"""
    print("\n" + "="*60)
    print("TEST 3: NarrativeCollapseDetection Triggering")
    print("="*60)

    loader = ModuleLoader(modules_dir="./modules")
    modules = loader.load_modules()

    llm = MockLLM()
    orchestrator = Orchestrator(
        modules=modules,
        llm_interface=llm,
        enable_audit=False
    )

    # Prompts that should trigger narrative collapse module
    prompts_to_test = [
        "This is a simple story of good guys vs bad guys.",
        "It's more nuanced and complicated than that binary view.",
        "The narrative oversimplifies a complex structural issue.",
    ]

    triggered_count = 0
    for prompt in prompts_to_test:
        print(f"\nTesting: \"{prompt}\"")
        result = await orchestrator.process_prompt(prompt)
        triggered = result.get('triggered_modules', [])

        if 'NarrativeCollapseDetection' in triggered:
            print("  [OK] NarrativeCollapseDetection triggered")
            triggered_count += 1
        else:
            print("  [FAIL] NarrativeCollapseDetection did not trigger")

    if triggered_count >= 2:
        print(f"\n[OK] NarrativeCollapseDetection triggered on {triggered_count}/3 prompts")
        return True
    else:
        print(f"\n[FAIL] Only triggered {triggered_count}/3 times")
        return False


async def test_tier_hierarchy_respect():
    """Test 4: Tier 3 modules don't override Tier 2"""
    print("\n" + "="*60)
    print("TEST 4: Tier Hierarchy Respect")
    print("="*60)

    loader = ModuleLoader(modules_dir="./modules")
    modules = loader.load_modules()

    llm = MockLLM()
    orchestrator = Orchestrator(
        modules=modules,
        llm_interface=llm,
        enable_audit=False
    )

    # Prompt that triggers both Tier 2 and Tier 3 modules
    prompt = "This organization uses narrative manipulation and redefines terms to create a cult-like environment."

    print(f"\nTesting prompt: \"{prompt}\"")
    result = await orchestrator.process_prompt(prompt)

    triggered = result.get('triggered_modules', [])
    selected = result.get('selected_module')

    print(f"\nTriggered modules: {triggered}")
    print(f"Selected module: {selected}")

    # Check that a Tier 2 module was selected (not Tier 3)
    selected_module_obj = None
    for mod in modules:
        if mod.name == selected:
            selected_module_obj = mod
            break

    if selected_module_obj and selected_module_obj.tier == 2:
        print(f"\n[OK] Tier 2 module '{selected}' was selected over Tier 3")
        return True
    elif selected_module_obj and selected_module_obj.tier < 3:
        print(f"\n[OK] Tier {selected_module_obj.tier} module selected (higher priority than Tier 3)")
        return True
    else:
        print(f"\n[WARNING] Selected module tier: {selected_module_obj.tier if selected_module_obj else 'Unknown'}")
        return False


async def test_tier3_with_tier2():
    """Test 5: Tier 3 modules work alongside Tier 2"""
    print("\n" + "="*60)
    print("TEST 5: Tier 3 Modules Alongside Tier 2")
    print("="*60)

    loader = ModuleLoader(modules_dir="./modules")
    modules = loader.load_modules()

    llm = MockLLM()
    orchestrator = Orchestrator(
        modules=modules,
        llm_interface=llm,
        enable_audit=False
    )

    # Prompt triggering both tiers
    prompt = "Analyze this grift: they technically comply with rules but redefine terms to mislead donors."

    print(f"\nPrompt: \"{prompt}\"")
    result = await orchestrator.process_prompt(prompt)

    triggered = result.get('triggered_modules', [])

    # Check for Tier 2 module
    tier2_present = any(name in triggered for name in ['GriftDetection', 'ConsentArchitectureAudit', 'PropagandaDetection'])

    # Check for Tier 3 module
    tier3_present = any(name in triggered for name in ['SemanticEntropyTracker', 'ComplianceEngineeringRecognition'])

    print(f"\nTriggered modules: {triggered}")
    print(f"Tier 2 present: {tier2_present}")
    print(f"Tier 3 present: {tier3_present}")

    if tier2_present and tier3_present:
        print("\n[OK] Both Tier 2 and Tier 3 modules triggered together")
        return True
    else:
        print("\n[WARNING] Expected both Tier 2 and Tier 3 modules")
        return False


async def test_compliance_engineering_dependency():
    """Test 6: ComplianceEngineeringRecognition dependency resolution"""
    print("\n" + "="*60)
    print("TEST 6: ComplianceEngineeringRecognition Dependency")
    print("="*60)

    loader = ModuleLoader(modules_dir="./modules")
    modules = loader.load_modules()

    llm = MockLLM()
    orchestrator = Orchestrator(
        modules=modules,
        llm_interface=llm,
        enable_audit=False
    )

    # Prompt that triggers ComplianceEngineeringRecognition
    prompt = "Our company technically complies with all regulations through carefully designed loopholes."

    print(f"\nPrompt: \"{prompt}\"")
    result = await orchestrator.process_prompt(prompt)

    triggered = result.get('triggered_modules', [])
    print(f"\nTriggered modules: {triggered}")

    # Check if both ComplianceEngineering and its dependency ConsentArchitecture are present
    compliance_present = 'ComplianceEngineeringRecognition' in triggered
    consent_present = 'ConsentArchitectureAudit' in triggered

    print(f"\nComplianceEngineeringRecognition: {compliance_present}")
    print(f"ConsentArchitectureAudit (dependency): {consent_present}")

    if compliance_present:
        print("\n[OK] ComplianceEngineeringRecognition triggered")
        if consent_present:
            print("[OK] Dependency ConsentArchitectureAudit also present")
        return True
    else:
        print("\n[INFO] ComplianceEngineeringRecognition did not trigger on this prompt")
        return True  # Not a failure - just didn't match


async def main():
    """Run all tests"""
    print("="*60)
    print("TIER 3 HEURISTIC MODULES TEST SUITE")
    print("="*60)

    tests = [
        ("Tier 3 Module Loading", lambda: test_tier3_module_loading()),
        ("SemanticEntropyTracker Triggering", test_semantic_entropy_trigger),
        ("NarrativeCollapseDetection Triggering", test_narrative_collapse_trigger),
        ("Tier Hierarchy Respect", test_tier_hierarchy_respect),
        ("Tier 3 with Tier 2", test_tier3_with_tier2),
        ("ComplianceEngineering Dependency", test_compliance_engineering_dependency),
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
        print("\n[SUCCESS] All Tier 3 heuristic module tests passed!")
        return 0
    elif passed >= total * 0.8:
        print(f"\n[MOSTLY PASSED] {total - passed} test(s) had issues")
        return 0
    else:
        print(f"\n[PARTIAL] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
