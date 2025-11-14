"""
Verify deep research is automatically enabled
"""
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

# Create orchestrator WITHOUT specifying enable_deep_research (should default to True)
orchestrator = Orchestrator(modules, llm, enable_audit=True, tool_registry=tool_registry)

# Check if deep research is enabled
print("\n" + "="*70)
print("DEEP RESEARCH STATUS CHECK")
print("="*70)
print(f"Deep research enabled: {orchestrator.enable_deep_research}")

if orchestrator.enable_deep_research:
    print("[SUCCESS] Deep research is AUTOMATICALLY enabled by default")
    print("The orchestrator will now use multi-source research + RAG by default")
else:
    print("[FAILED] Deep research is NOT enabled")

print("="*70)
