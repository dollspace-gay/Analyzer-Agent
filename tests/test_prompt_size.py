"""
Diagnostic: Check prompt size and LLM output
"""

import asyncio
from protocol_ai import ModuleLoader, LLMInterface, Orchestrator, ToolRegistry
import sys

sys.path.insert(0, 'tools')
from web_search_tool import WebSearchTool

# Load modules
modules = ModuleLoader("./modules").load_modules()

# Initialize LLM with LARGER context and tokens
llm = LLMInterface(
    model_path="F:/Agent/DeepSeek-R1/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
    gpu_layers=-1,
    context_length=8192,  # DOUBLED
    max_new_tokens=2048,  # QUADRUPLED
    temperature=0.7
)
llm.load_model()

# Setup tools
tool_registry = ToolRegistry()
tool_registry.register(WebSearchTool())

# Create orchestrator
orchestrator = Orchestrator(modules, llm, enable_audit=True, tool_registry=tool_registry)

# Process
print("Processing with LARGER context (8192) and tokens (2048)...")
print("="*70)

result = asyncio.run(orchestrator.process_prompt("Analyze OpenAI"))

print("\n" + "="*70)
print("RAW LLM OUTPUT (first 2000 chars):")
print("="*70)
raw = result.get('raw_llm_response', result['llm_response'])
try:
    print(raw[:2000])
except UnicodeEncodeError:
    print(raw[:2000].encode('ascii', 'replace').decode('ascii'))

print("\n" + "="*70)
print("FORMATTED RESPONSE (first 2000 chars):")
print("="*70)
formatted = result['llm_response']
try:
    print(formatted[:2000])
except UnicodeEncodeError:
    print(formatted[:2000].encode('ascii', 'replace').decode('ascii'))

print("\n" + "="*70)
print("SECTIONS CHECK:")
print("="*70)
for i in range(1, 8):
    has_section = f"**SECTION {i}" in formatted
    print(f"Section {i}: {'✓ Found' if has_section else '✗ Missing'}")
