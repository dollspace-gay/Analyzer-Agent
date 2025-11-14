"""
Test script for web search and report formatter integration
"""

import asyncio
from protocol_ai import ModuleLoader, LLMInterface, Orchestrator, ToolRegistry
from pathlib import Path
import sys

# Add tools directory to path
sys.path.insert(0, 'tools')
from web_search_tool import WebSearchTool

print("="*60)
print("Testing Web Search + Report Formatter Integration")
print("="*60)

# Load modules
print("\n1. Loading modules...")
module_loader = ModuleLoader("./modules")
modules = module_loader.load_modules()
print(f"   Loaded {len(modules)} modules")

# Initialize LLM (with minimal config for testing)
print("\n2. Initializing LLM...")
llm = LLMInterface(
    model_path="F:/Agent/DeepSeek-R1/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
    gpu_layers=-1,
    context_length=4096,
    max_new_tokens=512,
    temperature=0.7
)
llm.load_model()
print("   LLM loaded successfully")

# Initialize Tool Registry and register web search
print("\n3. Initializing Tool Registry...")
tool_registry = ToolRegistry()
web_search_tool = WebSearchTool()
tool_registry.register(web_search_tool)
print(f"   Registered web search tool")

# Initialize Orchestrator
print("\n4. Initializing Orchestrator...")
orchestrator = Orchestrator(modules, llm, enable_audit=True, tool_registry=tool_registry)
print(f"   Web search tool available: {'web_search' in tool_registry.tools}")
print(f"   Report formatter available: {orchestrator.report_formatter is not None}")

# Test prompt
print("\n5. Testing with prompt...")
test_prompt = "Analyze OpenAI"
print(f"   Prompt: {test_prompt}")

# Process prompt
print("\n6. Processing...")
result = asyncio.run(orchestrator.process_prompt(test_prompt))

# Display results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\nTriggered Modules: {result['triggered_modules']}")
print(f"Selected Module: {result['selected_module']}")
print(f"Regeneration Count: {result['regeneration_count']}")
print(f"Audit Passed: {result['audit_passed']}")

if result.get('web_context'):
    print(f"\nWeb Context Retrieved: YES")
    print(result['web_context'][:200] + "...")
else:
    print(f"\nWeb Context Retrieved: NO")

print(f"\n{'='*60}")
print("FORMATTED RESPONSE")
print(f"{'='*60}")
# Handle Unicode encoding issues on Windows
try:
    print(result['llm_response'][:1000])
except UnicodeEncodeError:
    print(result['llm_response'][:1000].encode('ascii', 'replace').decode('ascii'))
print("\n... (truncated)")

# Check for standardized format elements
response = result['llm_response']
has_checksum = "[CHECKSUM: SHA256::" in response
has_module_sweep = "[MODULE_SWEEP_COMPLETE]" in response
has_sections = "**SECTION" in response

print(f"\n{'='*60}")
print("FORMAT VALIDATION")
print(f"{'='*60}")
print(f"Has Checksum: {has_checksum}")
print(f"Has Module Sweep Complete: {has_module_sweep}")
print(f"Has Section Headers: {has_sections}")

if has_checksum and has_module_sweep and has_sections:
    print("\n[SUCCESS] Report formatted correctly with standardized structure!")
else:
    print("\n[ERROR] Report may not be using standardized format")

print(f"\n{'='*60}")
print("TEST COMPLETE")
print(f"{'='*60}")
