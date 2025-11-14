"""
Quick test of Qwen3-14B output quality
Single-pass analysis to see if it's cleaner than DeepSeek-R1
"""

import asyncio
from protocol_ai import ModuleLoader, LLMInterface, Orchestrator, ToolRegistry
import sys
import re

sys.path.insert(0, 'tools')
from web_search_tool import WebSearchTool

print("="*70)
print("QWEN3-14B QUICK OUTPUT TEST")
print("="*70)

# Load modules
modules = ModuleLoader("./modules").load_modules()
print(f"\nLoaded {len(modules)} modules")

# Initialize LLM with Qwen
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
orchestrator = Orchestrator(modules, llm, enable_audit=True, tool_registry=tool_registry)

print("\nProcessing: 'Analyze OpenAI'")
print("="*70)

result = asyncio.run(orchestrator.process_prompt("Analyze OpenAI"))

# Get response
response = result['llm_response']

print("\n" + "="*70)
print("QWEN OUTPUT QUALITY CHECK")
print("="*70)

# Check for prohibited meta-commentary patterns
prohibited = {
    "Let's take...": r"(?i)let'?s\s+take",
    "I must/need/should...": r"(?i)\bi\s+(must|need|should)\s+",
    "But note:": r"(?i)but\s+note:",
    "However, the user...": r"(?i)however,\s+the\s+user",
    "Since the background...": r"(?i)since\s+the\s+background",
    "Then, you will...": r"(?i)then,?\s+you\s+will",
    "Okay, I...": r"(?i)okay,?\s+i\s+",
    "[Content not generated]": r"\[Content not generated\]",
    "- Evaluate/Identify...": r"(?i)^-\s*(evaluate|identify|combine)",
}

print("\n🔍 Meta-Commentary Check:")
issues = []
for name, pattern in prohibited.items():
    matches = re.findall(pattern, response, re.MULTILINE)
    if matches:
        issues.append(name)
        print(f"  ❌ FOUND: {name}")
        print(f"     Matches: {matches[:2]}")

if not issues:
    print("  ✅ NO meta-commentary found!")

# Check sections
print("\n📋 Section Check:")
sections_found = len(re.findall(r'\*\*SECTION \d+', response))
print(f"  Total sections: {sections_found}/7")

sections_with_content = 0
empty_sections = []
for i in range(1, 8):
    pattern = rf"\*\*SECTION {i}.*?\n\n(.*?)(?=\*\*SECTION|\[MODULE|$)"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        content = match.group(1).strip()
        if content and len(content) > 100 and "[Content not generated]" not in content:
            sections_with_content += 1
        else:
            empty_sections.append(i)

print(f"  Sections with content: {sections_with_content}/7")
if empty_sections:
    print(f"  ❌ Empty sections: {empty_sections}")
else:
    print(f"  ✅ All sections have content")

# Check for concrete evidence
print("\n🎯 Evidence Check:")
evidence_patterns = {
    "Dollar amounts": r'\$[\d,]+',
    "Dates/Years": r'\b(19|20)\d{2}\b',
    "Named people": r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
}

for name, pattern in evidence_patterns.items():
    matches = re.findall(pattern, response)
    count = len(matches)
    status = "✅" if count > 0 else "❌"
    print(f"  {status} {name}: {count} found")

# Overall score
print("\n" + "="*70)
print("OVERALL ASSESSMENT")
print("="*70)

score = 0
max_score = 4

if not issues:
    score += 1
    print("✅ No meta-commentary (1/1)")
else:
    print(f"❌ Meta-commentary found: {len(issues)} issues (0/1)")

if sections_with_content >= 6:
    score += 1
    print(f"✅ Sections have content: {sections_with_content}/7 (1/1)")
else:
    print(f"❌ Too few sections with content: {sections_with_content}/7 (0/1)")

if sections_found == 7:
    score += 1
    print("✅ All 7 sections present (1/1)")
else:
    print(f"❌ Missing sections: {7-sections_found} (0/1)")

# Check for any evidence
total_evidence = sum(len(re.findall(p, response)) for p in evidence_patterns.values())
if total_evidence > 5:
    score += 1
    print(f"✅ Concrete evidence present: {total_evidence} items (1/1)")
else:
    print(f"❌ Insufficient evidence: {total_evidence} items (0/1)")

print("\n" + "="*70)
print(f"SCORE: {score}/{max_score}")
if score == max_score:
    print("🎉 EXCELLENT - Qwen is producing clean, structured output!")
elif score >= 3:
    print("✅ GOOD - Minor issues but much better than DeepSeek-R1")
elif score >= 2:
    print("⚠️  FAIR - Still has issues, needs improvement")
else:
    print("❌ POOR - Output quality not acceptable")
print("="*70)

# Show preview
print("\n" + "="*70)
print("SECTION 1 PREVIEW (first 800 chars):")
print("="*70)
section1 = re.search(r"\*\*SECTION 1.*?\n\n(.*?)(?=\*\*SECTION|$)", response, re.DOTALL)
if section1:
    preview = section1.group(1)[:800]
    print(preview.encode('ascii', 'replace').decode('ascii'))
else:
    print("[SECTION 1 NOT FOUND]")

# Save output
from pathlib import Path
output_file = Path("qwen_output.txt")
output_file.write_text(response, encoding='utf-8')
print(f"\n✅ Full output saved to: {output_file}")
