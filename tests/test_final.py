"""
Final test: Cadence neutralization + No thoughts + Proper format
"""

import asyncio
from protocol_ai import ModuleLoader, LLMInterface, Orchestrator, ToolRegistry
import sys
import re

sys.path.insert(0, 'tools')
from web_search_tool import WebSearchTool

print("="*70)
print("FINAL TEST: Cadence Neutralization + Affective Firewall + No Thoughts")
print("="*70)

# Load modules (should now include CadenceNeutralization and AffectiveFirewall)
modules = ModuleLoader("./modules").load_modules()
print(f"\n[1/4] Loaded {len(modules)} modules")

# Check for critical modules
critical_modules = ['CadenceNeutralization', 'AffectiveFirewall']
for mod_name in critical_modules:
    found = any(m.name == mod_name for m in modules)
    status = "[OK]" if found else "[MISSING]"
    print(f"      {status} {mod_name}")

# Initialize LLM with maximum context
llm = LLMInterface(
    model_path="F:/Agent/DeepSeek-R1/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf",
    gpu_layers=-1,
    context_length=13000,
    max_new_tokens=4096,
    temperature=0.7
)
llm.load_model()
print("\n[2/4] LLM loaded")

# Setup tools
tool_registry = ToolRegistry()
tool_registry.register(WebSearchTool())

# Create orchestrator
orchestrator = Orchestrator(modules, llm, enable_audit=True, tool_registry=tool_registry)
print("\n[3/4] Orchestrator initialized")

# Process
print("\n[4/4] Processing: 'Analyze OpenAI'")
print("="*70)

result = asyncio.run(orchestrator.process_prompt("Analyze OpenAI"))

# Get response
response = result['llm_response']

print("\n" + "="*70)
print("VALIDATION CHECKS")
print("="*70)

# Check 1: Format structure
checks = {
    "7 sections present": len(re.findall(r'SECTION \d', response)) >= 7,
    "Checksum present": "[CHECKSUM: SHA256::" in response,
    "Module sweep complete": "[MODULE_SWEEP_COMPLETE]" in response,
}

# Check 2: No hedging language
hedging_words = ["might", "could", "possibly", "arguably", "perhaps", "potentially"]
found_hedging = []
for word in hedging_words:
    if re.search(rf'\b{word}\b', response, re.IGNORECASE):
        found_hedging.append(word)

checks["No hedging language"] = len(found_hedging) == 0

# Check 3: No meta-commentary/thoughts
meta_phrases = [
    "let's", "we'll", "we need", "we should", "first,", "then,",
    "i will", "let me", "here's", "okay,"
]
found_meta = []
for phrase in meta_phrases:
    if re.search(rf'\b{phrase}\b', response, re.IGNORECASE):
        found_meta.append(phrase)

checks["No meta-commentary"] = len(found_meta) == 0

# Check 4: No softening
softening = ["it seems", "it appears", "to be fair", "unfortunately", "sadly"]
found_softening = []
for phrase in softening:
    if phrase in response.lower():
        found_softening.append(phrase)

checks["No softening language"] = len(found_softening) == 0

# Print results
for check, passed in checks.items():
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {check}")

if found_hedging:
    print(f"\n[WARNING] Found hedging words: {', '.join(found_hedging)}")

if found_meta:
    print(f"\n[WARNING] Found meta-commentary: {', '.join(found_meta)}")

if found_softening:
    print(f"\n[WARNING] Found softening: {', '.join(found_softening)}")

# Show sample
print("\n" + "="*70)
print("SAMPLE OUTPUT (first 1500 chars)")
print("="*70)
try:
    print(response[:1500])
except UnicodeEncodeError:
    print(response[:1500].encode('ascii', 'replace').decode('ascii'))

print("\n" + "="*70)
all_passed = all(checks.values())
if all_passed:
    print("[SUCCESS] All validation checks passed!")
    print("Format: Correct | Cadence: Neutral | Thoughts: Removed")
else:
    print("[PARTIAL] Some checks failed - see warnings above")
print("="*70)
