"""
Test improved format enforcement and output cleaning
"""

import asyncio
from protocol_ai import ModuleLoader, LLMInterface, Orchestrator, ToolRegistry
import sys
import re

sys.path.insert(0, 'tools')
from web_search_tool import WebSearchTool

print("="*70)
print("TESTING IMPROVED FORMAT ENFORCEMENT")
print("="*70)

# Load modules
modules = ModuleLoader("./modules").load_modules()
print(f"\nLoaded {len(modules)} modules")

# Initialize LLM
llm = LLMInterface(
    model_path="F:/Agent/DeepSeek-R1/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf",
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
orchestrator = Orchestrator(modules, llm, enable_audit=True, tool_registry=tool_registry)

print("\nProcessing: 'Analyze OpenAI'")
print("="*70)

result = asyncio.run(orchestrator.process_prompt("Analyze OpenAI"))

# Get response
response = result['llm_response']

print("\n" + "="*70)
print("FORMAT VALIDATION")
print("="*70)

# Check for prohibited patterns
prohibited_patterns = {
    "Let's take": r"(?i)let'?s\s+take",
    "I must/need/should": r"(?i)\bi\s+(must|need|should)",
    "But note": r"(?i)but\s+note",
    "Since the background": r"(?i)since\s+the\s+background",
    "- Evaluate/Identify/Combine": r"(?i)^-\s*(evaluate|identify|combine|analyze)",
    "[Content not generated]": r"\[Content not generated\]",
    "Single word caps lines": r"^[A-Z]{2,}$",
}

found_issues = {}
for name, pattern in prohibited_patterns.items():
    matches = re.findall(pattern, response, re.MULTILINE)
    if matches:
        found_issues[name] = matches

# Check for required patterns
required_patterns = {
    "Section 1 has content": r"\*\*SECTION 1.*?\n\n.*?\w{20,}",
    "Section 3 has content": r"\*\*SECTION 3.*?\n\n.*?\w{20,}",
    "Section 6 has content": r"\*\*SECTION 6.*?\n\n.*?\w{20,}",
    "Triggered modules per section": r"\[Triggered Modules:",
    "Concrete evidence": r"(\$\d+|\d{4}|[A-Z][a-z]+ \d+,? \d{4})",
}

found_required = {}
for name, pattern in required_patterns.items():
    matches = re.findall(pattern, response, re.DOTALL)
    found_required[name] = len(matches) > 0

# Print results
print("\n🔍 PROHIBITED PATTERNS CHECK:")
if found_issues:
    for issue, matches in found_issues.items():
        print(f"  ❌ FOUND: {issue}")
        print(f"     Examples: {matches[:3]}")
else:
    print("  ✅ No prohibited patterns found!")

print("\n✅ REQUIRED PATTERNS CHECK:")
for name, found in found_required.items():
    status = "✅" if found else "❌"
    print(f"  {status} {name}")

# Count sections with actual content
sections_with_content = 0
for i in range(1, 8):
    pattern = rf"\*\*SECTION {i}.*?\n\n(.*?)(?=\*\*SECTION|\[MODULE|$)"
    match = re.search(pattern, response, re.DOTALL)
    if match and match.group(1).strip() and "[Content not generated]" not in match.group(1):
        sections_with_content += 1

print(f"\n📊 SECTIONS WITH CONTENT: {sections_with_content}/7")

# Show sample from problematic sections
print("\n" + "="*70)
print("SECTION 1 PREVIEW:")
print("="*70)
section1_match = re.search(r"\*\*SECTION 1.*?\n\n(.*?)(?=\*\*SECTION|$)", response, re.DOTALL)
if section1_match:
    content = section1_match.group(1)[:500]
    try:
        print(content)
    except UnicodeEncodeError:
        print(content.encode('ascii', 'replace').decode('ascii'))
else:
    print("[NO SECTION 1 FOUND]")

print("\n" + "="*70)
print("SECTION 6 PREVIEW:")
print("="*70)
section6_match = re.search(r"\*\*SECTION 6.*?\n\n(.*?)(?=\*\*SECTION|\[MODULE|$)", response, re.DOTALL)
if section6_match:
    content = section6_match.group(1)[:500]
    try:
        print(content)
    except UnicodeEncodeError:
        print(content.encode('ascii', 'replace').decode('ascii'))
else:
    print("[NO SECTION 6 FOUND]")

# Overall assessment
print("\n" + "="*70)
all_good = (
    not found_issues and
    all(found_required.values()) and
    sections_with_content >= 6
)

if all_good:
    print("✅ [SUCCESS] Format enforcement working correctly!")
else:
    print("❌ [NEEDS WORK] Format enforcement needs improvement")
    if found_issues:
        print("   - Remove prohibited patterns from output")
    if not all(found_required.values()):
        print("   - Add required patterns to output")
    if sections_with_content < 6:
        print(f"   - Generate content for all sections (only {sections_with_content}/7 have content)")

print("="*70)
