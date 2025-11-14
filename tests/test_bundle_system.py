"""
Test Bundle System

Verifies that the bundle system correctly:
1. Loads bundle definitions from YAML files
2. Activates specified modules when bundle is loaded
3. Applies bundle configuration (trigger sensitivity, etc.)
4. Includes bundle instructions in prompts
5. Responds to bundle commands (/load_bundle, /clear_bundle)
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protocol_ai import ModuleLoader, BundleLoader, Orchestrator, LLMInterface


class MockLLM:
    """Mock LLM for testing - just echoes what modules are active"""
    def execute(self, prompt):
        # Return a simple response showing what was in the prompt
        response = "Mock LLM Response:\n"
        if "[BUNDLE:" in prompt:
            response += "  - Bundle instructions detected\n"
        if "ACTIVATE MODULE:" in prompt:
            response += "  - Module instructions detected\n"
        response += f"  - Prompt length: {len(prompt)} characters\n"
        return response


async def test_bundle_loader():
    """Test 1: BundleLoader can load bundle files"""
    print("="*60)
    print("TEST 1: Bundle Loading")
    print("="*60)

    loader = BundleLoader(bundles_dir="./bundles")
    bundles = loader.load_bundles()

    print(f"\n[OK] Loaded {len(bundles)} bundle(s)")

    for bundle_name, bundle in bundles.items():
        print(f"\n  Bundle: {bundle_name}")
        print(f"    Name: {bundle.name}")
        print(f"    Version: {bundle.version}")
        print(f"    Description: {bundle.description}")
        print(f"    Active Modules: {len(bundle.active_modules)}")

    expected_bundles = ['governance', 'coding', 'minimal', 'red_team']
    for expected in expected_bundles:
        if expected in bundles:
            print(f"  [OK] Found expected bundle: {expected}")
        else:
            print(f"  [FAIL] Missing expected bundle: {expected}")

    return len(bundles) > 0


async def test_bundle_activation():
    """Test 2: Loading a bundle activates correct modules"""
    print("\n" + "="*60)
    print("TEST 2: Bundle Activation")
    print("="*60)

    # Load modules and bundles
    module_loader = ModuleLoader(modules_dir="./modules")
    modules = module_loader.load_modules()

    bundle_loader = BundleLoader(bundles_dir="./bundles")
    bundle_loader.load_bundles()

    # Create orchestrator with bundle support
    llm = MockLLM()
    orchestrator = Orchestrator(
        modules=modules,
        llm_interface=llm,
        enable_audit=False,
        bundle_loader=bundle_loader
    )

    print(f"\n[OK] System initialized with {len(modules)} modules")

    # Test loading governance bundle
    print("\nLoading 'governance' bundle...")
    result = await orchestrator.process_prompt("/load_bundle governance")

    print(f"\n{result['llm_response']}")

    # Check that bundle is active
    if orchestrator.active_bundle:
        print(f"\n[OK] Bundle active: {orchestrator.active_bundle.name}")
        print(f"[OK] Bundle modules: {len(orchestrator.bundle_modules)}")

        # List activated modules
        for module in orchestrator.bundle_modules:
            print(f"  - {module.name} (Tier {module.tier})")

        return True
    else:
        print("\n[FAIL] Bundle not activated")
        return False


async def test_bundle_instructions():
    """Test 3: Bundle instructions are included in prompts"""
    print("\n" + "="*60)
    print("TEST 3: Bundle Instructions in Prompts")
    print("="*60)

    # Load system
    module_loader = ModuleLoader(modules_dir="./modules")
    modules = module_loader.load_modules()

    bundle_loader = BundleLoader(bundles_dir="./bundles")
    bundle_loader.load_bundles()

    llm = MockLLM()
    orchestrator = Orchestrator(
        modules=modules,
        llm_interface=llm,
        enable_audit=False,
        bundle_loader=bundle_loader
    )

    # Load governance bundle
    await orchestrator.process_prompt("/load_bundle governance")

    print("\nProcessing test prompt with governance bundle active...")
    result = await orchestrator.process_prompt(
        "Analyze this organization's power structure"
    )

    # Check if bundle instructions are present
    # We can't directly access the assembled prompt, but we can check the MockLLM response
    response = result['llm_response']
    print(f"\nLLM Response:\n{response}")

    if "Bundle instructions detected" in response:
        print("\n[OK] Bundle instructions were included in prompt")
        return True
    else:
        print("\n[WARNING] Bundle instructions may not have been included")
        return False


async def test_bundle_clear():
    """Test 4: Clearing bundles works correctly"""
    print("\n" + "="*60)
    print("TEST 4: Bundle Clear")
    print("="*60)

    # Load system
    module_loader = ModuleLoader(modules_dir="./modules")
    modules = module_loader.load_modules()

    bundle_loader = BundleLoader(bundles_dir="./bundles")
    bundle_loader.load_bundles()

    llm = MockLLM()
    orchestrator = Orchestrator(
        modules=modules,
        llm_interface=llm,
        enable_audit=False,
        bundle_loader=bundle_loader
    )

    # Load bundle
    await orchestrator.process_prompt("/load_bundle governance")
    print(f"\n[OK] Bundle loaded: {orchestrator.active_bundle.name}")

    # Clear bundle
    result = await orchestrator.process_prompt("/clear_bundle")
    print(f"\n{result['llm_response']}")

    # Check that bundle is cleared
    if orchestrator.active_bundle is None:
        print("\n[OK] Bundle cleared successfully")
        return True
    else:
        print("\n[FAIL] Bundle still active after clear")
        return False


async def test_bundle_configuration():
    """Test 5: Bundle configuration applies to TriggerEngine"""
    print("\n" + "="*60)
    print("TEST 5: Bundle Configuration")
    print("="*60)

    # Load system
    module_loader = ModuleLoader(modules_dir="./modules")
    modules = module_loader.load_modules()

    bundle_loader = BundleLoader(bundles_dir="./bundles")
    bundle_loader.load_bundles()

    llm = MockLLM()
    orchestrator = Orchestrator(
        modules=modules,
        llm_interface=llm,
        enable_audit=False,
        bundle_loader=bundle_loader
    )

    # Check initial TriggerEngine settings
    initial_mode = orchestrator.trigger_engine.matching_mode
    initial_synonyms = orchestrator.trigger_engine.enable_synonyms
    print(f"\nInitial TriggerEngine settings:")
    print(f"  Matching mode: {initial_mode}")
    print(f"  Synonyms enabled: {initial_synonyms}")

    # Load coding bundle (should use 'simple' mode)
    await orchestrator.process_prompt("/load_bundle coding")

    coding_mode = orchestrator.trigger_engine.matching_mode
    coding_synonyms = orchestrator.trigger_engine.enable_synonyms
    print(f"\nAfter loading 'coding' bundle:")
    print(f"  Matching mode: {coding_mode}")
    print(f"  Synonyms enabled: {coding_synonyms}")

    # Clear and load red_team bundle (should use 'advanced' mode)
    await orchestrator.process_prompt("/clear_bundle")
    await orchestrator.process_prompt("/load_bundle red_team")

    red_team_mode = orchestrator.trigger_engine.matching_mode
    red_team_synonyms = orchestrator.trigger_engine.enable_synonyms
    print(f"\nAfter loading 'red_team' bundle:")
    print(f"  Matching mode: {red_team_mode}")
    print(f"  Synonyms enabled: {red_team_synonyms}")

    # Verify configurations were applied
    if coding_mode == "simple" and not coding_synonyms:
        print("\n[OK] Coding bundle configuration applied correctly")
    else:
        print("\n[WARNING] Coding bundle configuration may not have applied")

    if red_team_mode == "advanced" and red_team_synonyms:
        print("[OK] Red team bundle configuration applied correctly")
        return True
    else:
        print("[WARNING] Red team bundle configuration may not have applied")
        return False


async def test_status_command():
    """Test 6: /status command shows bundle information"""
    print("\n" + "="*60)
    print("TEST 6: Status Command with Bundle")
    print("="*60)

    # Load system
    module_loader = ModuleLoader(modules_dir="./modules")
    modules = module_loader.load_modules()

    bundle_loader = BundleLoader(bundles_dir="./bundles")
    bundle_loader.load_bundles()

    llm = MockLLM()
    orchestrator = Orchestrator(
        modules=modules,
        llm_interface=llm,
        enable_audit=False,
        bundle_loader=bundle_loader
    )

    # Load bundle
    await orchestrator.process_prompt("/load_bundle governance")

    # Check status
    result = await orchestrator.process_prompt("/status")
    print(f"\n{result['llm_response']}")

    response = result['llm_response']
    if "Governance Analysis" in response:
        print("\n[OK] Status shows bundle information")
        return True
    else:
        print("\n[WARNING] Status may not show bundle info correctly")
        return False


async def main():
    """Run all tests"""
    print("="*60)
    print("BUNDLE SYSTEM TEST SUITE")
    print("="*60)

    tests = [
        ("Bundle Loading", test_bundle_loader),
        ("Bundle Activation", test_bundle_activation),
        ("Bundle Instructions", test_bundle_instructions),
        ("Bundle Clear", test_bundle_clear),
        ("Bundle Configuration", test_bundle_configuration),
        ("Status Command", test_status_command),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
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
        print("\n[SUCCESS] All bundle system tests passed!")
        return 0
    else:
        print(f"\n[PARTIAL] {total - passed} test(s) failed or incomplete")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
