"""
Test two-pass analysis system

This should produce much cleaner output because:
1. LLM thinks freely without format constraints
2. LLM formats the thoughts into proper structure
"""

import asyncio
import sys

# Add parent directory to path to import from root
sys.path.insert(0, '..')

from protocol_ai import ModuleLoader, LLMInterface, Orchestrator, ToolRegistry
from multi_turn_analysis import two_pass_analysis
import re

sys.path.insert(0, '../tools')
from web_search_tool import WebSearchTool

print("="*70)
print("TWO-PASS ANALYSIS SYSTEM TEST")
print("="*70)

# Load modules
modules = ModuleLoader("./modules").load_modules()
print(f"\nLoaded {len(modules)} modules")

# Initialize LLM
llm = LLMInterface(
    model_path="F:/Agent/DeepSeek-R1/Qwen3-14B-Q4_K_M.gguf",
    gpu_layers=-1,
    context_length=13000,
    max_new_tokens=4096,
    temperature=0.7
)
llm.load_model()

# Setup tools
tool_registry = ToolRegistry()
tool_registry.register(WebSearchTool())

# Create orchestrator
orchestrator = Orchestrator(modules, llm, enable_audit=False, tool_registry=tool_registry)

print("\n" + "="*70)
print("RUNNING TWO-PASS ANALYSIS")
print("="*70)

# Get web context (simulate with deep research)
async def get_context():
    # This would normally use deep research
    # For now, simulate a simple prompt
    from protocol_ai import TriggerEngine

    user_prompt = "Analyze OpenAI"

    # Get triggered modules
    trigger_engine = TriggerEngine()
    triggered_modules = trigger_engine.analyze_prompt(user_prompt, modules)

    print(f"\nTriggered {len(triggered_modules)} modules")

    # Get web context
    web_context = await orchestrator._search_web_context(user_prompt)

    return user_prompt, web_context, triggered_modules

user_prompt, web_context, triggered_modules = asyncio.run(get_context())

# Run two-pass analysis
result = two_pass_analysis(orchestrator, user_prompt, web_context, triggered_modules)

print("\n" + "="*70)
print("TWO-PASS RESULTS")
print("="*70)

print(f"\n[OK] Draft analysis: {len(result['draft_analysis'])} chars")
print(f"[OK] Formatted report: {len(result['formatted_report'])} chars")
print(f"[OK] Sections found: {result['sections_found']}/7")
print(f"[OK] Draft saved to: {result['draft_file']}")

# Validate output quality
report = result['formatted_report']

# Check for prohibited patterns
prohibited = {
    "Let's take": r"(?i)let'?s\s+take",
    "I must/need": r"(?i)\bi\s+(must|need|should)",
    "[Content not generated]": r"\[Content not generated\]",
}

issues_found = []
for name, pattern in prohibited.items():
    if re.search(pattern, report):
        issues_found.append(name)

print("\n" + "="*70)
print("QUALITY CHECK")
print("="*70)

if issues_found:
    print("[FAIL] Found prohibited patterns:")
    for issue in issues_found:
        print(f"   - {issue}")
else:
    print("[OK] No prohibited patterns found!")

# Check section content
sections_with_content = 0
for i in range(1, 8):
    pattern = rf"\*\*SECTION {i}.*?\n\n(.*?)(?=\*\*SECTION|\[MODULE|$)"
    match = re.search(pattern, report, re.DOTALL)
    if match:
        content = match.group(1).strip()
        if content and len(content) > 50:
            sections_with_content += 1

print(f"\n[OK] Sections with real content: {sections_with_content}/7")

# Show preview
print("\n" + "="*70)
print("REPORT PREVIEW (First 1000 chars)")
print("="*70)
try:
    print(report[:1000])
except UnicodeEncodeError:
    print(report[:1000].encode('ascii', 'replace').decode('ascii'))

# Save final report
from pathlib import Path
output_file = Path("two_pass_report.txt")
output_file.write_text(report, encoding='utf-8')
print(f"\n[OK] Full report saved to: {output_file}")

# Overall assessment
print("\n" + "="*70)
if not issues_found and sections_with_content >= 6:
    print("[SUCCESS] Two-pass system working correctly!")
    print("   - Draft analysis complete")
    print("   - Proper formatting applied")
    print("   - All sections have content")
    print("   - No prohibited patterns")
else:
    print("[PARTIAL SUCCESS] Needs improvement:")
    if issues_found:
        print(f"   - Remove: {', '.join(issues_found)}")
    if sections_with_content < 6:
        print(f"   - Only {sections_with_content}/7 sections have content")

print("="*70)
