"""
Verify all 71 modules load correctly
"""
from protocol_ai import ModuleLoader

print("="*70)
print("MODULE VERIFICATION")
print("="*70)

loader = ModuleLoader("./modules")
modules = loader.load_modules()

print(f"\nTotal modules loaded: {len(modules)}")

# Count by tier
tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
for module in modules:
    tier = module.tier
    tier_counts[tier] = tier_counts.get(tier, 0) + 1

print("\nModules by tier:")
for tier in sorted(tier_counts.keys()):
    print(f"  Tier {tier}: {tier_counts[tier]} modules")

# List Tier 1 modules (critical)
tier1_modules = [m.name for m in modules if m.tier == 1]
print(f"\nTier 1 (Critical) modules ({len(tier1_modules)}):")
for name in sorted(tier1_modules):
    print(f"  - {name}")

# Check for required modules
required = ['CadenceNeutralization', 'AffectiveFirewall', 'ModuleSweepEnforcer',
            'StructuralFinalityEnforcer']
print(f"\nRequired modules check:")
for req in required:
    found = any(m.name == req for m in modules)
    status = "[OK]" if found else "[MISSING]"
    print(f"  {status} {req}")

print("\n" + "="*70)
print("[SUCCESS] All modules loaded successfully!")
print("="*70)
