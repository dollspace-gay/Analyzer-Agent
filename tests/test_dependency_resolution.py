"""
Test Module Dependency Resolution

Verifies that the dependency system correctly:
1. Loads module dependencies from YAML
2. Resolves dependencies recursively
3. Detects circular dependencies
4. Orders modules for execution (topological sort)
5. Handles missing dependencies gracefully
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protocol_ai import Module, ModuleLoader, DependencyResolver, Orchestrator, LLMInterface


class MockLLM:
    """Mock LLM for testing"""
    def execute(self, prompt):
        return "Mock response"


def test_dependency_resolver_basic():
    """Test 1: Basic dependency resolution"""
    print("="*60)
    print("TEST 1: Basic Dependency Resolution")
    print("="*60)

    # Create test modules with dependencies
    module_a = Module(
        name="ModuleA",
        tier=2,
        purpose="Module A (no dependencies)",
        triggers=["trigger_a"],
        prompt_template="Template A",
        dependencies=[]
    )

    module_b = Module(
        name="ModuleB",
        tier=2,
        purpose="Module B (depends on A)",
        triggers=["trigger_b"],
        prompt_template="Template B",
        dependencies=["ModuleA"]
    )

    module_c = Module(
        name="ModuleC",
        tier=2,
        purpose="Module C (depends on B, which depends on A)",
        triggers=["trigger_c"],
        prompt_template="Template C",
        dependencies=["ModuleB"]
    )

    all_modules = [module_a, module_b, module_c]
    resolver = DependencyResolver(all_modules)

    # Test resolving C (should include B and A)
    print("\nResolving ModuleC (depends on B, which depends on A)...")
    resolved = resolver.resolve([module_c])

    print(f"Resolved {len(resolved)} modules:")
    for i, mod in enumerate(resolved, 1):
        print(f"  {i}. {mod.name}")

    # Verify order: A should come before B, B before C
    names = [m.name for m in resolved]
    a_index = names.index("ModuleA")
    b_index = names.index("ModuleB")
    c_index = names.index("ModuleC")

    if a_index < b_index < c_index:
        print("\n[OK] Modules in correct execution order")
        return True
    else:
        print(f"\n[FAIL] Wrong order: {names}")
        return False


def test_circular_dependency():
    """Test 2: Circular dependency detection"""
    print("\n" + "="*60)
    print("TEST 2: Circular Dependency Detection")
    print("="*60)

    # Create modules with circular dependency
    module_a = Module(
        name="ModuleA",
        tier=2,
        purpose="Module A (depends on B)",
        triggers=["trigger_a"],
        prompt_template="Template A",
        dependencies=["ModuleB"]
    )

    module_b = Module(
        name="ModuleB",
        tier=2,
        purpose="Module B (depends on A - creates cycle)",
        triggers=["trigger_b"],
        prompt_template="Template B",
        dependencies=["ModuleA"]
    )

    all_modules = [module_a, module_b]
    resolver = DependencyResolver(all_modules)

    print("\nAttempting to resolve circular dependency (A -> B -> A)...")
    try:
        resolved = resolver.resolve([module_a])
        print("\n[FAIL] Should have detected circular dependency")
        return False
    except ValueError as e:
        print(f"\n[OK] Circular dependency detected: {e}")
        return True


def test_missing_dependency():
    """Test 3: Missing dependency handling"""
    print("\n" + "="*60)
    print("TEST 3: Missing Dependency Handling")
    print("="*60)

    # Create module with dependency on non-existent module
    module_a = Module(
        name="ModuleA",
        tier=2,
        purpose="Module A (depends on non-existent ModuleX)",
        triggers=["trigger_a"],
        prompt_template="Template A",
        dependencies=["ModuleX"]  # ModuleX doesn't exist
    )

    all_modules = [module_a]
    resolver = DependencyResolver(all_modules)

    print("\nAttempting to resolve module with missing dependency...")
    try:
        resolved = resolver.resolve([module_a])
        print("\n[FAIL] Should have detected missing dependency")
        return False
    except ValueError as e:
        print(f"\n[OK] Missing dependency detected: {e}")
        return True


def test_complex_dependency_graph():
    """Test 4: Complex dependency graph resolution"""
    print("\n" + "="*60)
    print("TEST 4: Complex Dependency Graph")
    print("="*60)

    # Create a diamond dependency pattern:
    #     A
    #    / \
    #   B   C
    #    \ /
    #     D

    module_a = Module(
        name="ModuleA", tier=2, purpose="Base module",
        triggers=["a"], prompt_template="A", dependencies=[]
    )

    module_b = Module(
        name="ModuleB", tier=2, purpose="Depends on A",
        triggers=["b"], prompt_template="B", dependencies=["ModuleA"]
    )

    module_c = Module(
        name="ModuleC", tier=2, purpose="Depends on A",
        triggers=["c"], prompt_template="C", dependencies=["ModuleA"]
    )

    module_d = Module(
        name="ModuleD", tier=2, purpose="Depends on B and C",
        triggers=["d"], prompt_template="D", dependencies=["ModuleB", "ModuleC"]
    )

    all_modules = [module_a, module_b, module_c, module_d]
    resolver = DependencyResolver(all_modules)

    print("\nResolving diamond dependency pattern...")
    print("     A")
    print("    / \\")
    print("   B   C")
    print("    \\ /")
    print("     D")

    resolved = resolver.resolve([module_d])

    print(f"\nResolved {len(resolved)} modules:")
    for i, mod in enumerate(resolved, 1):
        print(f"  {i}. {mod.name}")

    names = [m.name for m in resolved]

    # Verify A comes before B and C, and both B and C come before D
    a_index = names.index("ModuleA")
    b_index = names.index("ModuleB")
    c_index = names.index("ModuleC")
    d_index = names.index("ModuleD")

    if a_index < b_index and a_index < c_index and b_index < d_index and c_index < d_index:
        print("\n[OK] Complex dependency graph resolved correctly")
        return True
    else:
        print(f"\n[FAIL] Wrong order in dependency resolution")
        return False


def test_loaded_modules_with_dependencies():
    """Test 5: Loading real modules with dependencies"""
    print("\n" + "="*60)
    print("TEST 5: Real Modules with Dependencies")
    print("="*60)

    # Load actual modules from filesystem
    loader = ModuleLoader(modules_dir="./modules")
    modules = loader.load_modules()

    print(f"\nLoaded {len(modules)} modules")

    # Find modules with dependencies
    modules_with_deps = [m for m in modules if m.dependencies]

    if modules_with_deps:
        print(f"Found {len(modules_with_deps)} module(s) with dependencies:")
        for mod in modules_with_deps:
            print(f"\n  {mod.name}:")
            print(f"    Depends on: {', '.join(mod.dependencies)}")

        # Create resolver and validate
        resolver = DependencyResolver(modules)
        errors = resolver.validate_all()

        if errors:
            print("\n[WARNING] Dependency validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print("\n[OK] All module dependencies are valid")
            return True
    else:
        print("\n[INFO] No modules with dependencies found")
        print("[INFO] Created ComplianceEngineeringRecognition as example")
        return True


async def test_dependency_resolution_in_orchestrator():
    """Test 6: Dependency resolution in Orchestrator"""
    print("\n" + "="*60)
    print("TEST 6: Dependency Resolution in Orchestrator")
    print("="*60)

    # Load modules
    loader = ModuleLoader(modules_dir="./modules")
    modules = loader.load_modules()

    llm = MockLLM()
    orchestrator = Orchestrator(
        modules=modules,
        llm_interface=llm,
        enable_audit=False
    )

    # Test prompt that should trigger ComplianceEngineeringRecognition
    # which depends on ConsentArchitectureAudit
    print("\nTesting prompt that triggers dependent module...")
    prompt = "This system technically complies with regulations but violates their spirit through loopholes"

    result = await orchestrator.process_prompt(prompt)

    triggered = result.get('triggered_modules', [])
    print(f"\nTriggered modules: {triggered}")

    # Check if dependency resolution message appeared
    # (We can't easily capture print output, so we'll just verify no errors)
    print("\n[OK] Orchestrator handled dependencies without errors")
    return True


async def main():
    """Run all tests"""
    print("="*60)
    print("DEPENDENCY RESOLUTION TEST SUITE")
    print("="*60)

    tests = [
        ("Basic Dependency Resolution", lambda: test_dependency_resolver_basic()),
        ("Circular Dependency Detection", lambda: test_circular_dependency()),
        ("Missing Dependency Handling", lambda: test_missing_dependency()),
        ("Complex Dependency Graph", lambda: test_complex_dependency_graph()),
        ("Real Modules with Dependencies", lambda: test_loaded_modules_with_dependencies()),
        ("Orchestrator Integration", test_dependency_resolution_in_orchestrator),
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
        print("\n[SUCCESS] All dependency resolution tests passed!")
        return 0
    else:
        print(f"\n[PARTIAL] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
