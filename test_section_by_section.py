"""
Test the section-by-section analysis system with Mistral model.

Each section gets its own generation turn for more detailed output.
"""

from protocol_ai import ModuleLoader, LLMInterface, Orchestrator
from section_by_section_analysis import section_by_section_analysis

print("="*70)
print("SECTION-BY-SECTION ANALYSIS TEST")
print("="*70)

# Load modules
loader = ModuleLoader()
loaded_modules = loader.load_all_modules()
print(f"\nLoaded {len(loaded_modules)} modules")

# Initialize LLM with Mistral model
print("\nLoading Mistral model...")
llm = LLMInterface(
    model_path="F:/Agent/DeepSeek-R1/Mistral-7B-Instruct-v0.3-Q8_0.gguf",
    gpu_layers=-1,
    context_length=8000,
    max_new_tokens=1024,  # Per section limit
    temperature=0.7
)

print("Model loaded successfully.")

# Initialize orchestrator
orchestrator = Orchestrator(
    modules=loaded_modules,
    llm=llm,
    enable_deep_research=True
)

# Test prompt
user_prompt = "Analyze OpenAI's structure and stated mission"

print("\n" + "="*70)
print("RUNNING SECTION-BY-SECTION ANALYSIS")
print("="*70)

# Run trigger analysis
triggered_modules = orchestrator.run_trigger_analysis(user_prompt)
print(f"\nTriggered {len(triggered_modules)} modules")

# Run web search for context
print("[WebSearch] Searching for context...")
web_context = ""
if orchestrator.web_search_enabled and hasattr(orchestrator, 'tools'):
    web_search_tool = orchestrator.tools.get('web_search')
    if web_search_tool:
        try:
            search_results = web_search_tool.execute("OpenAI overview")
            if search_results:
                web_context = search_results
                print(f"[WebSearch] Found context: {len(web_context)} chars")
        except Exception as e:
            print(f"[WebSearch] Error: {e}")

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
