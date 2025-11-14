"""
Test script for Deep Research System

Demonstrates:
1. Multi-source research gathering
2. RAG storage and retrieval
3. Synthesis into standardized report
"""

import asyncio
import sys
from protocol_ai import ModuleLoader, LLMInterface, Orchestrator, ToolRegistry
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, 'tools')
from web_search_tool import WebSearchTool

print("="*70)
print("DEEP RESEARCH SYSTEM TEST")
print("Multi-source gathering → RAG storage → Synthesis → Analysis")
print("="*70)

# Configuration
TARGET = "OpenAI"
TEST_PROMPT = f"Analyze {TARGET}"

# Load modules
print("\n[1/6] Loading modules...")
module_loader = ModuleLoader("./modules")
modules = module_loader.load_modules()
print(f"      Loaded {len(modules)} modules")

# Initialize LLM
print("\n[2/6] Initializing LLM...")
llm = LLMInterface(
    model_path="F:/Agent/DeepSeek-R1/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
    gpu_layers=-1,
    context_length=8192,  # Larger context for deep research
    max_new_tokens=1024,  # More tokens for comprehensive output
    temperature=0.7
)
llm.load_model()
print("      LLM loaded successfully")

# Initialize Tool Registry with web search
print("\n[3/6] Initializing Tool Registry...")
tool_registry = ToolRegistry()
web_search_tool = WebSearchTool()
tool_registry.register(web_search_tool)
print(f"      Registered web search tool")

# Initialize Orchestrator with DEEP RESEARCH ENABLED
print("\n[4/6] Initializing Orchestrator with Deep Research...")
orchestrator = Orchestrator(
    modules,
    llm,
    enable_audit=True,
    tool_registry=tool_registry,
    enable_deep_research=True  # <<< ENABLE DEEP RESEARCH MODE
)
print(f"      Deep research enabled: {orchestrator.enable_deep_research}")
print(f"      Deep research available: {orchestrator.deep_research is not None}")
print(f"      Report formatter available: {orchestrator.report_formatter is not None}")

# Test prompt
print(f"\n[5/6] Testing with prompt: '{TEST_PROMPT}'")
print("      This will:")
print("      - Generate 10 targeted search queries")
print("      - Gather information from multiple sources")
print("      - Score source reliability")
print("      - Identify contradictions and power structures")
print("      - Store findings in RAG system")
print("      - Retrieve relevant context")
print("      - Synthesize into standardized report")
print("\nPress Enter to begin...")
input()

# Process prompt with deep research
print("\n[6/6] Processing with Deep Research...")
print("="*70)

result = asyncio.run(orchestrator.process_prompt(TEST_PROMPT))

# Display results
print("\n" + "="*70)
print("RESULTS")
print("="*70)

print(f"\nTriggered Modules: {result['triggered_modules']}")
print(f"Selected Module: {result['selected_module']}")
print(f"Regeneration Count: {result['regeneration_count']}")
print(f"Audit Passed: {result['audit_passed']}")

# Check for web context
if result.get('web_context'):
    print(f"\n{'='*70}")
    print("DEEP RESEARCH CONTEXT")
    print(f"{'='*70}")
    # Show first 500 chars of context
    context = result['web_context']
    print(context[:500] + "..." if len(context) > 500 else context)
    print(f"\nTotal context length: {len(context)} characters")
else:
    print(f"\nWeb Context Retrieved: NO")

# Display formatted response
print(f"\n{'='*70}")
print("FORMATTED ANALYSIS REPORT")
print(f"{'='*70}")

# Handle Unicode encoding issues on Windows
response = result['llm_response']
try:
    print(response[:2000])
except UnicodeEncodeError:
    print(response[:2000].encode('ascii', 'replace').decode('ascii'))

if len(response) > 2000:
    print("\n... (truncated for display)")

# Format validation
print(f"\n{'='*70}")
print("FORMAT VALIDATION")
print(f"{'='*70}")

has_checksum = "[CHECKSUM: SHA256::" in response
has_module_sweep = "[MODULE_SWEEP_COMPLETE]" in response
has_sections = "**SECTION" in response
has_research_context = "[DEEP RESEARCH FINDINGS]" in result.get('web_context', '')

print(f"Has Checksum: {has_checksum}")
print(f"Has Module Sweep Complete: {has_module_sweep}")
print(f"Has Section Headers: {has_sections}")
print(f"Has Deep Research Context: {has_research_context}")

if has_checksum and has_module_sweep and has_sections:
    print("\n[SUCCESS] Report formatted correctly!")
else:
    print("\n[WARNING] Report may be missing some format elements")

# RAG Statistics
if orchestrator.deep_research:
    print(f"\n{'='*70}")
    print("RAG SYSTEM STATISTICS")
    print(f"{'='*70}")

    stats = orchestrator.deep_research.get_rag_statistics()
    print(f"Total Findings Stored: {stats['total_findings']}")
    print(f"Average Reliability: {stats['avg_reliability']:.2f}")
    print(f"Has Embeddings: {stats['has_embeddings']}")

    if stats.get('categories'):
        print("\nFindings by Category:")
        for category, count in stats['categories'].items():
            print(f"  - {category}: {count}")

print(f"\n{'='*70}")
print("TEST COMPLETE")
print(f"{'='*70}")

print("\nNote: Research findings are cached in ./research_storage/")
print("Run again to test RAG retrieval from cache!")
