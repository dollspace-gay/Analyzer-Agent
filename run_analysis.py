#!/usr/bin/env python3
"""
Protocol AI - Command Line Interface

Quick command-line runner for analysis without GUI.

Usage:
    python run_analysis.py "Analyze OpenAI"
    python run_analysis.py "Analyze effective altruism"
"""

import sys
import asyncio
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent / 'tools'))

from protocol_ai import ModuleLoader, LLMInterface, Orchestrator, ToolRegistry
from web_search_tool import WebSearchTool


def main():
    """Main entry point for command-line analysis."""

    # Check for query argument
    if len(sys.argv) < 2:
        print("Usage: python run_analysis.py \"Your analysis query\"")
        print("\nExamples:")
        print("  python run_analysis.py \"Analyze OpenAI\"")
        print("  python run_analysis.py \"Analyze effective altruism\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    print("="*70)
    print("Protocol AI - Governance Layer Analysis")
    print("="*70)
    print(f"\nQuery: {query}\n")

    # Load modules
    print("[1/4] Loading modules...")
    modules = ModuleLoader("./modules").load_modules()
    print(f"      Loaded {len(modules)} modules")

    # Initialize LLM
    print("\n[2/4] Loading LLM...")
    llm = LLMInterface(
        model_path="F:/Agent/DeepSeek-R1/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf",
        gpu_layers=-1,
        context_length=13000,
        max_new_tokens=4096,
        temperature=0.7
    )
    llm.load_model()
    print("      Model loaded")

    # Setup tools
    print("\n[3/4] Setting up tools...")
    tool_registry = ToolRegistry()
    tool_registry.register(WebSearchTool())
    print("      Web search registered")

    # Create orchestrator (deep research enabled by default)
    print("\n[4/4] Initializing orchestrator...")
    orchestrator = Orchestrator(modules, llm, enable_audit=True, tool_registry=tool_registry)
    print(f"      Deep research: {orchestrator.enable_deep_research}")

    # Process query
    print("\n" + "="*70)
    print("Processing Query...")
    print("="*70 + "\n")

    result = asyncio.run(orchestrator.process_prompt(query))

    # Display result
    print("\n" + "="*70)
    print("ANALYSIS RESULT")
    print("="*70 + "\n")

    try:
        print(result['llm_response'])
    except UnicodeEncodeError:
        # Handle Windows console encoding issues
        print(result['llm_response'].encode('ascii', 'replace').decode('ascii'))

    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)

    # Save to file
    output_file = Path("analysis_output.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result['llm_response'])

    print(f"\nOutput saved to: {output_file.absolute()}")


if __name__ == "__main__":
    main()
