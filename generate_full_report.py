"""
Generate full OpenAI analysis report to verify format
"""

import asyncio
from protocol_ai import ModuleLoader, LLMInterface, Orchestrator, ToolRegistry
import sys

sys.path.insert(0, 'tools')
from web_search_tool import WebSearchTool

# Load modules
modules = ModuleLoader("./modules").load_modules()
print(f"Loaded {len(modules)} modules")

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

# Process
print("\nProcessing: 'Analyze OpenAI'")
print("="*70)

result = asyncio.run(orchestrator.process_prompt("Analyze OpenAI"))

# Save full response
with open("full_report_output.txt", "w", encoding="utf-8") as f:
    f.write(result['llm_response'])

print("\n" + "="*70)
print("FULL REPORT saved to: full_report_output.txt")
print("="*70)
print(f"\nReport length: {len(result['llm_response'])} characters")
print(f"Report lines: {len(result['llm_response'].splitlines())} lines")
