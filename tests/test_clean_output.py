"""
Test that web context is removed from final output
"""

import asyncio
from protocol_ai import ModuleLoader, LLMInterface, Orchestrator, ToolRegistry
import sys

sys.path.insert(0, 'tools')
from web_search_tool import WebSearchTool

# Load modules
modules = ModuleLoader("./modules").load_modules()
print(f"Loaded {len(modules)} modules\n")

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

# Create orchestrator (deep research enabled by default)
orchestrator = Orchestrator(modules, llm, enable_audit=True, tool_registry=tool_registry)

print("\nProcessing: 'Analyze OpenAI'")
print("="*70)

result = asyncio.run(orchestrator.process_prompt("Analyze OpenAI"))

# Get response
response = result['llm_response']
raw_response = result.get('raw_llm_response', response)

print("\n" + "="*70)
print("RAW LLM OUTPUT (first 1000 chars)")
print("="*70)
try:
    print(raw_response[:1000])
except UnicodeEncodeError:
    print(raw_response[:1000].encode('ascii', 'replace').decode('ascii'))

print("\n" + "="*70)
print("OUTPUT VALIDATION")
print("="*70)

# Check if web context is in output
has_web_context = "[WEB CONTEXT]" in response or "[Web Context:" in response
print(f"Web context in output: {has_web_context}")

if has_web_context:
    print("[FAILED] Web context should NOT be in final output")
else:
    print("[SUCCESS] Web context properly removed from output")

# Check if output starts with triggered modules or section
starts_correct = response.startswith("[Triggered Modules:") or response.startswith("**SECTION 1")
print(f"Starts with modules/section: {starts_correct}")

if starts_correct:
    print("[SUCCESS] Output starts correctly")
else:
    print("[FAILED] Output should start with [Triggered Modules:] or **SECTION 1")

# Show first 500 chars
print("\n" + "="*70)
print("FIRST 500 CHARACTERS OF OUTPUT:")
print("="*70)
try:
    print(response[:500])
except UnicodeEncodeError:
    print(response[:500].encode('ascii', 'replace').decode('ascii'))

print("\n" + "="*70)
if not has_web_context and starts_correct:
    print("[SUCCESS] Output format is clean!")
else:
    print("[FAILED] Output needs cleaning")
print("="*70)
