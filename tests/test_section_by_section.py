"""
Test the section-by-section analysis system with Mistral model.

Each section gets its own generation turn for more detailed output.
"""

from protocol_ai import ModuleLoader, LLMInterface, Orchestrator, TriggerEngine, ToolRegistry
from section_by_section_analysis import section_by_section_analysis
import asyncio
import sys

sys.path.insert(0, 'tools')
from web_search_tool import WebSearchTool

print("="*70)
print("SECTION-BY-SECTION ANALYSIS TEST")
print("="*70)

# Load modules
loaded_modules = ModuleLoader("./modules").load_modules()
print(f"\nLoaded {len(loaded_modules)} modules")

# Initialize LLM with Mistral model
llm = LLMInterface(
    model_path="F:/Agent/DeepSeek-R1/Mistral-7B-Instruct-v0.3-Q8_0.gguf",
    gpu_layers=-1,
    context_length=8000,
    max_new_tokens=1024,  # Per section limit
    temperature=0.7
)
llm.load_model()

# Setup tools
tool_registry = ToolRegistry()
tool_registry.register(WebSearchTool())

# Initialize orchestrator
orchestrator = Orchestrator(
    loaded_modules,
    llm,
    enable_audit=False,
    tool_registry=tool_registry
)

print("\n" + "="*70)
print("RUNNING SECTION-BY-SECTION ANALYSIS")
print("="*70)

# Get context
async def get_context():
    user_prompt = "Analyze OpenAI"

    # Get triggered modules
    trigger_engine = TriggerEngine()
    triggered_modules = trigger_engine.analyze_prompt(user_prompt, loaded_modules)

    print(f"\nTriggered {len(triggered_modules)} modules")

    # Get web context
    web_context = await orchestrator._search_web_context(user_prompt)

    return user_prompt, web_context, triggered_modules

user_prompt, web_context, triggered_modules = asyncio.run(get_context())

# Run section-by-section analysis
result = section_by_section_analysis(
    orchestrator=orchestrator,
    user_prompt=user_prompt,
    web_context=web_context,
    modules=triggered_modules
)

print("\n" + "="*70)
print("RESULTS")
print("="*70)

for i, section in enumerate(result['sections'], 1):
    print(f"\nSection {i}: {len(section)} chars")
    # Show first 150 chars of each section
    preview = section[:150].replace('\n', ' ')
    print(f"Preview: {preview}...")

print(f"\nFull report saved to: {result['output_file']}")
print(f"Total report length: {len(result['full_report'])} chars")

# Quality check
print("\n" + "="*70)
print("QUALITY CHECK")
print("="*70)

full_report = result['full_report']

# Check that all 7 sections exist
sections_found = []
for i in range(1, 8):
    if f"**SECTION {i}:" in full_report:
        sections_found.append(i)

print(f"\nSections present: {len(sections_found)}/7")
if len(sections_found) == 7:
    print("[OK] All sections generated")
else:
    missing = [i for i in range(1, 8) if i not in sections_found]
    print(f"[FAIL] Missing sections: {missing}")

# Check for prohibited patterns
prohibited_patterns = {
    "Let's take": r"(?i)let'?s\s+take",
    "I must/need": r"(?i)\bi\s+(must|need|should)\s+",
    "[Content not generated]": r"\[Content not generated\]",
    "Meta-commentary": r"(?i)okay,?\s+(let's|here's|i've)",
}

issues = []
for name, pattern in prohibited_patterns.items():
    import re
    if re.search(pattern, full_report):
        issues.append(name)

if issues:
    print(f"\n[WARN] Found prohibited patterns: {', '.join(issues)}")
else:
    print("\n[OK] No prohibited patterns detected")

# Check section lengths
print("\nSection lengths:")
for i, section in enumerate(result['sections'], 1):
    length = len(section)
    status = "[OK]" if length > 200 else "[SHORT]"
    print(f"  Section {i}: {length} chars {status}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
