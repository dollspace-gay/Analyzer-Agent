"""
Multi-Turn Analysis System

Supports both two-pass and section-by-section (8-step) analysis modes.

Two-Pass Mode:
  Pass 1: LLM analyzes freely with all reasoning
  Pass 2: Format draft into standardized 7-section report

Section-by-Section Mode (8 steps):
  Steps 1-7: Generate each section individually
  Step 8: Format all sections with report_formatter tool

This module provides utility functions used by both modes.
"""

from pathlib import Path
import re


def _strip_post_report_thinking(text: str) -> str:
    """
    Strip reasoning/thinking that gets appended after the formatted report ends.

    Mistral tends to output the clean report, then append its reasoning process.
    We want to cut everything after the report ends.

    Args:
        text: Full LLM output

    Returns:
        Just the formatted report
    """
    # Look for the end of Section 7
    section_7_end = text.find("This analysis prioritizes observable systemic dynamics")
    if section_7_end != -1:
        # Find the end of that sentence
        end_marker = text.find("This statement is a standardized component of this report structure.", section_7_end)
        if end_marker != -1:
            # Cut everything after this point
            return text[:end_marker + len("This statement is a standardized component of this report structure.")]

    # If we can't find Section 7 end, look for other markers
    # Sometimes there's ``` or markdown code fences
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            # Keep everything before the first code fence after the report
            # This is fragile but better than nothing
            return parts[0].rstrip()

    # Fallback: look for "Okay, let's start" which indicates thinking restart
    thinking_restart = text.find("Okay, let's start by looking")
    if thinking_restart != -1:
        return text[:thinking_restart].rstrip()

    return text


def _clean_metacognitive_loops(text: str) -> str:
    """
    Remove metacognitive loop patterns where LLM gets stuck saying
    "I'm ready to present" without actually presenting.

    Args:
        text: Raw LLM output

    Returns:
        Cleaned text with loops removed
    """
    lines = text.split('\n')
    cleaned_lines = []
    loop_count = 0
    max_loop_repeats = 3

    for line in lines:
        # Detect metacognitive loop patterns
        is_loop = any([
            "I'll present" in line,
            "I'm ready to present" in line,
            "Now, I'll present" in line,
            "present it as the final answer" in line,
            "I've completed the analysis. Now" in line,
            "I'm done with the analysis" in line,
        ])

        if is_loop:
            loop_count += 1
            if loop_count <= max_loop_repeats:
                # Keep first few instances
                cleaned_lines.append(line)
        else:
            loop_count = 0  # Reset counter if not a loop line
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def create_analysis_prompt(user_prompt: str, web_context: str, modules: list) -> str:
    """
    Create Pass 1 prompt - free-form analysis with reasoning.

    Args:
        user_prompt: Original user query
        web_context: Research findings
        modules: Triggered modules

    Returns:
        Prompt for free-form analysis
    """

    module_names = [m.name for m in modules]

    prompt = f"""
You are conducting a structural analysis. You have access to research data and governance modules.

ANALYSIS TASK: {user_prompt}

TRIGGERED GOVERNANCE MODULES:
{', '.join(module_names)}

RESEARCH DATA:
{web_context}

YOUR TASK - PASS 1 (FREE-FORM ANALYSIS):

Think through this analysis completely. Write everything:
- Your reasoning process
- Evidence you find important
- Connections you notice
- Contradictions you identify
- Analytical frameworks that apply
- Structural patterns

DO NOT WORRY ABOUT FORMAT. Just analyze thoroughly.

Write your complete analysis below, including all reasoning:
"""

    return prompt


def create_formatting_prompt(draft_analysis: str, modules: list) -> str:
    """
    Create Pass 2 prompt - format the draft into standardized report.

    Args:
        draft_analysis: Raw analysis from Pass 1
        modules: Triggered modules

    Returns:
        Prompt for formatting
    """

    module_names = [m.name for m in modules]

    prompt = f"""
FORMATTING TASK - DO NOT ANALYZE, JUST FORMAT THE TEXT BELOW.

You are a formatter, not an analyst. Your ONLY job is to reorganize the text below into the 7-section structure.

DO NOT:
- Add new analysis
- Show your thinking process
- Say "I'll present" or "I'm ready to"
- Explain what you're doing

DO:
- Extract existing points from the analysis
- Put them into the correct sections
- Output ONLY the formatted report

TRIGGERED MODULES: {', '.join(module_names)}

TEXT TO FORMAT:
{draft_analysis}

OUTPUT THIS EXACT STRUCTURE:

[Triggered Modules: {', '.join(module_names)}]

**SECTION 1: "The Narrative"**

[Triggered Modules: <specific modules used>]
<Extract the key narrative points from the analysis above>

**SECTION 2: "The Central Contradiction"**

[Triggered Modules: <specific modules used>]
Stated Intent: <Extract from analysis>
Behavior/Actual Outcome: <Extract from analysis>
Narrative Collapse: <Extract from analysis>

**SECTION 3: "Deconstruction of Core Concepts"**

[Triggered Modules: <specific modules used>]

Concept: "<concept name>"
The Narrative: <How it's framed>
Structural Analysis: <What it actually is, from your analysis>

(Repeat for 2-3 key concepts)

**SECTION 4: "Ideological Adjacency"**

[Triggered Modules: <specific modules used>]
<Extract ideological patterns identified in analysis>

**SECTION 5: "Synthesis"**

[Triggered Module: CrossModuleSynthesisProtocol]
<Synthesize key findings using analytical frameworks>

**SECTION 6: "System Performance Audit"**

[Triggered Module: DriftContainmentProtocol]
Analysis completeness: <Assess the analysis quality>
Evidence strength: <Rate the evidence used>

**SECTION 7: "Standardized Epistemic Lens Acknowledgment"**

This analysis prioritizes observable systemic dynamics and structural logic. Other epistemological frameworks may offer complementary perspectives. This statement is a standardized component of this report structure.

CRITICAL RULES:
- Extract actual content from the analysis, don't make up new content
- Each section must have REAL content (no "[Content not generated]")
- Use direct declarative statements
- Include specific evidence from the analysis
- Name analytical frameworks explicitly
- NO meta-commentary ("Let's take...", "I need to...", etc.)

Output ONLY the formatted report, starting with [Triggered Modules:].
"""

    return prompt


def two_pass_analysis(orchestrator, user_prompt: str, web_context: str, modules: list) -> dict:
    """
    Execute two-pass analysis system.

    Args:
        orchestrator: Orchestrator instance with LLM
        user_prompt: User's query
        web_context: Research data
        modules: Triggered modules

    Returns:
        dict with draft, final report, and metadata
    """

    # PASS 1: Free-form analysis
    print("\n=== PASS 1: Free-Form Analysis ===")
    print("Letting LLM analyze without format constraints...")

    analysis_prompt = create_analysis_prompt(user_prompt, web_context, modules)

    # Save original max_tokens and override for Pass 1
    original_max_tokens = orchestrator.llm.max_new_tokens
    orchestrator.llm.max_new_tokens = 2048  # Prevent infinite loops

    draft_analysis = orchestrator.llm.execute(analysis_prompt)

    # Restore original max_tokens
    orchestrator.llm.max_new_tokens = original_max_tokens

    # Clean metacognitive loops from draft
    draft_analysis = _clean_metacognitive_loops(draft_analysis)

    # Save draft for inspection
    draft_file = Path("analysis_draft.txt")
    draft_file.write_text(draft_analysis, encoding='utf-8')
    print(f"Draft saved to: {draft_file}")
    print(f"Draft length: {len(draft_analysis)} chars")

    # PASS 2: Format into standardized report
    print("\n=== PASS 2: Format Into Report ===")
    print("Converting draft into standardized 7-section format...")

    formatting_prompt = create_formatting_prompt(draft_analysis, modules)
    formatted_report = orchestrator.llm.execute(formatting_prompt)

    # Clean the formatted output (cleaning is done in LLMInterface.execute, but do it again to be sure)
    cleaned_report = orchestrator.llm._clean_llm_output(formatted_report)

    # Strip any reasoning that got appended after the report
    cleaned_report = _strip_post_report_thinking(cleaned_report)

    print(f"Formatted report length: {len(cleaned_report)} chars")

    # Validate sections exist
    sections_found = len(re.findall(r'\*\*SECTION \d+', cleaned_report))
    print(f"Sections generated: {sections_found}/7")

    return {
        'draft_analysis': draft_analysis,
        'formatted_report': cleaned_report,
        'sections_found': sections_found,
        'draft_file': str(draft_file)
    }
