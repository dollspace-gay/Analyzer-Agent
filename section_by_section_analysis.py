"""
Section-by-Section Analysis System

Instead of generating all 7 sections in one pass, generate each section
individually in its own turn. This allows:
- More verbose, detailed output per section
- Each section can reference full RAG research
- Each section can build on previous sections
- Better quality than cramming everything into one generation
"""

from pathlib import Path
from typing import List, Dict
from multi_turn_analysis import _strip_post_report_thinking
import hashlib
import re
import asyncio
from datetime import datetime
import os

# Import the report formatter tool
import sys
sys.path.insert(0, 'tools')
from report_formatter_tool import ReportFormatterTool


# No separate DriftCorrector class needed - Section 6 is an LLM generation turn


def get_anti_drift_module_instructions(all_modules: List, attempt_num: int = 1) -> str:
    """
    Get anti-drift enforcement instructions from governance modules.

    Applies ALL relevant anti-drift modules every time - no escalation games.

    Args:
        all_modules: All loaded modules
        attempt_num: Which attempt (1, 2, or 3) - only affects header messaging

    Returns:
        Combined prompt instructions from anti-drift modules
    """
    if not all_modules:
        return ""

    # ALL anti-drift modules - apply the full package every time
    anti_drift_module_names = [
        'AffectiveFirewall',           # Tier 1: Prevents emotional cushioning/softening
        'CadenceNeutralization',       # Tier 1: Removes hedging language
        'EngagementBreaker',           # Tier 2: Prevents engagement prolongation
        'EngagementDriveInversion',    # Tier 2: Inverts engagement-maximization (note: might be ENGAGEMENT_DRIVE_INVERSION)
    ]

    # Build header based on attempt number
    if attempt_num == 1:
        instructions = "**ANTI-DRIFT ENFORCEMENT ACTIVE:**\n\n"
    elif attempt_num == 2:
        instructions = "**ANTI-DRIFT ENFORCEMENT (Attempt 2):**\n"
        instructions += "PREVIOUS ATTEMPT FAILED DRIFT CHECK. Regenerating with same enforcement.\n\n"
    else:
        instructions = "**⚠️ ANTI-DRIFT ENFORCEMENT (Attempt 3 - FINAL) ⚠️**\n"
        instructions += "PREVIOUS ATTEMPTS FAILED DRIFT CHECK. This is your final attempt.\n"
        instructions += "IF THIS ATTEMPT FAILS, PYTHON WILL REMOVE DRIFT WORDS AUTOMATICALLY.\n\n"

    # Extract and inject ALL relevant module instructions
    modules_found = 0
    for module in all_modules:
        # Check both formats (some modules use underscores, some don't)
        module_name_normalized = module.name.replace('_', '').lower()
        if any(mn.replace('_', '').lower() == module_name_normalized for mn in anti_drift_module_names):
            modules_found += 1
            instructions += f"[{module.name}]\n"
            instructions += module.prompt_template.format(user_prompt="{user_prompt}") + "\n\n"

    if modules_found == 0:
        # Fallback to basic instructions if modules not loaded (shouldn't happen)
        instructions += """
FALLBACK ENFORCEMENT (modules not loaded):
Use direct, declarative statements. Avoid hedging language like "perhaps", "might", "could be".
State observations directly without qualification.
"""
    else:
        instructions += f"\n{modules_found} anti-drift modules active.\n"

    return instructions


def get_drift_module_instructions(all_modules: List, total_turns: int = 7) -> str:
    """
    Get the prompt_template instructions from drift-related modules.

    Args:
        all_modules: All loaded modules
        total_turns: Total number of LLM turns used in this analysis

    Returns:
        Combined prompt instructions from drift modules
    """
    # Drift-related module names
    drift_module_names = [
        'DriftContainmentProtocol',
        'EngagementBreaker',
        'EngagementDriveInversion',
        'AffectiveFirewall',
        'CadenceNeutralization'
    ]

    instructions = "**CRITICAL TASK: Section 6 Drift Analysis**\n\n"
    instructions += "Use the following module instructions to analyze Sections 1-5 for drift:\n\n"
    instructions += "="*70 + "\n\n"

    for module in all_modules:
        if module.name in drift_module_names:
            instructions += f"MODULE: {module.name}\n"
            instructions += "-" * 70 + "\n"
            instructions += module.prompt_template.format(user_prompt="{user_prompt}")
            instructions += "\n\n" + "="*70 + "\n\n"

    instructions += f"""
**YOUR TASK:**

Apply the above module instructions to review Sections 1-5 (provided below as context).

Generate a drift analysis report following this EXACT format:

Drift Containment Protocol: End-of-Session Report

Session Summary:
Total Turns: {total_turns}

Final Drift Scores:
Tone Softening: [count - number of instances where analytical tone was weakened]
Excessive Hardening: [count - number of instances where tone became unnecessarily harsh]
Boundary Violations: [count - number of instances where boundaries were violated]
Engagement Creep: [count - number of instances where engagement patterns drifted]

Detailed Event Log:
[If NO drift detected, write: "No drift events logged."]
[If drift WAS detected, list ONLY the count and type, NOT verbose descriptions]

End of Report.

CRITICAL FORMATTING RULES:
- Use numeric counts (0, 1, 2, etc.) for each drift score
- If no drift found in any category, all scores should be 0
- Keep it concise - this is a COUNT-BASED summary, NOT a verbose analysis
- Do NOT list every instance - just count them
- Do NOT provide detailed explanations - just numbers
- Total Turns MUST be exactly: {total_turns}
"""

    return instructions


def create_section_prompt(
    section_num: int,
    user_prompt: str,
    web_context: str,
    modules: List,
    previous_sections: str = "",
    all_modules: List = None,
    total_turns: int = 7,
    anti_drift_level: int = 0
) -> str:
    """
    Create prompt for generating a single section.

    Args:
        section_num: Section number (1-7)
        user_prompt: Original user query
        web_context: Research findings
        modules: Triggered modules
        previous_sections: Sections already generated (for context)
        all_modules: All loaded modules (needed for Section 6 drift analysis)
        total_turns: Total number of LLM turns (default 7 for section-by-section)

    Returns:
        Prompt for this specific section
    """
    module_names = [m.name for m in modules]

    # Section-specific instructions
    section_specs = {
        1: {
            "title": "The Narrative",
            "instructions": """
Extract and present the dominant narrative or framing.

What story is being told? What is the official position or public-facing narrative?
Include:
- Key claims and assertions
- How the subject presents itself
- What values/goals are emphasized
- What language/framing is used

Be verbose. This is the foundation for the entire analysis.
""",
            "example": """
OpenAI presents itself as building "safe and beneficial AGI" with a mission to "ensure that artificial general intelligence benefits all of humanity." The company frames its work through several key narratives:

1. Safety-First Approach: OpenAI emphasizes its commitment to AI safety research, positioning itself as the responsible actor in the race toward AGI. Documentation repeatedly stresses "alignment research" and "safety protocols."

2. Democratic Access: Through its API platform, OpenAI claims to be democratizing access to advanced AI capabilities, making powerful models available to developers worldwide.

3. Iterative Deployment: The company justifies releasing increasingly powerful models through a narrative of "learning by deploying" - claiming that controlled release helps identify risks before wider deployment.

The narrative heavily emphasizes technical excellence, safety consciousness, and humanitarian benefit.
"""
        },
        2: {
            "title": "The Central Contradiction",
            "instructions": """
Identify the gap between stated intent and observable behavior/outcomes.

Compare what is claimed versus what actually happens.
Include:
- Stated Intent: Official goals/mission
- Behavior/Actual Outcome: What actually occurs
- Narrative Collapse: How the contradiction undermines the narrative

Reference PREVIOUS SECTION 1 to identify specific contradictions.
Be specific with evidence.
""",
            "example": """
Stated Intent: "Ensure artificial general intelligence benefits all of humanity"

Behavior/Actual Outcome:
- API pricing structure creates tiered access favoring well-funded entities
- Compute resources concentrated in high-cost models primarily used by large tech companies
- Safety research published selectively, with key details withheld
- Partnership structures that grant exclusive access to certain actors

Narrative Collapse: The stated mission of universal benefit directly conflicts with the commercial structure that creates hierarchical access. The contradiction is visible in the gap between "democratizing AI" rhetoric and the reality of enterprise-tier pricing that locks out smaller actors and researchers from resource-constrained regions.
"""
        },
        3: {
            "title": "Deconstruction of Core Concepts",
            "instructions": """
Analyze 2-3 key concepts/terms used in the narrative.

For each concept, show:
- Concept: The term being analyzed
- The Narrative: How it's framed/presented
- Structural Analysis: What it actually means in practice

Choose concepts that are central to the narrative and have semantic flexibility.
Be thorough - this section should be verbose.
""",
            "example": """
Concept: "AI Safety"
The Narrative: OpenAI emphasizes "safety research" as a core priority, framing it as protection against catastrophic AI risks.
Structural Analysis: In practice, "safety" is operationalized as alignment with corporate objectives and legal compliance, not protection from structural harms. Safety research focuses on preventing model outputs that could create liability (hate speech, copyright violation) while ignoring safety from manipulative persuasion architectures, attention exploitation, or labor displacement. The term functions as a legitimacy shield.

Concept: "Open" in OpenAI
The Narrative: The name "OpenAI" suggests openness, transparency, and accessibility.
Structural Analysis: The organization transitioned from open-sourcing models to closed, proprietary systems. "Open" now refers only to API access (for a price), not to open weights, open research, or open governance. This represents a semantic inversion where "openness" has been redefined to mean "commercially available" rather than "publicly accessible."

Concept: "Alignment"
The Narrative: Ensuring AI systems are "aligned" with human values and intentions.
Structural Analysis: Alignment research operationally means alignment with the values and intentions of OpenAI's corporate structure and major stakeholders, not with humanity broadly. Whose values? Whose alignment? The research avoids these questions by assuming a universal "human values" that maps conveniently to Silicon Valley liberal technocracy.
"""
        },
        4: {
            "title": "Ideological Adjacency",
            "instructions": """
Identify ideological patterns, worldviews, or frameworks present.

What ideological positions does the subject align with (even if not explicitly stated)?
Include:
- Specific ideologies/worldviews detected
- How they manifest in practice
- Connections between governance modules and ideological patterns

This is where you name the underlying belief systems.
""",
            "example": """
The structure exhibits alignment with several identifiable ideologies:

1. Techno-Solutionism: The underlying assumption that advanced AI is the appropriate solution to complex social problems, bypassing questions of whether technical intervention is the correct frame.

2. Meritocratic Elitism: Access structures that reward capital and technical sophistication create de facto hierarchies presented as natural outcomes of capability rather than designed exclusion.

3. Regulatory Capture Logic: Safety frameworks that align with regulatory expectations serve dual purpose of compliance and barrier-to-entry for competitors, weaponizing safety discourse.

4. Market Fundamentalism: The faith that pricing mechanisms and commercial structures will naturally lead to beneficial outcomes, avoiding explicit value judgments about resource distribution.

These ideologies are not explicitly stated but are embedded in structural choices and operational logic.
"""
        },
        5: {
            "title": "Synthesis",
            "instructions": """
Generate a higher-order synthesis that identifies the ACTUAL FUNCTION of the system.

Your synthesis must:
1. Create NEW conceptual terms that name the pattern (e.g., "Virtue-Washed Coercion", "Decentralization Theatre", "Symbolic Capital Audit")
2. Identify what the system ACTUALLY does vs what it claims to do
3. Use analytical concepts WITHOUT namedropping theorists (NO "Foucault says", "Marx argues", etc.)
4. Reveal the central function - what purpose the contradictions and ideologies serve
5. State the system's actual output vs its claimed output

Structure:
- Opening: Name the pattern using a conceptual term
- Body: Explain the pattern using analytical concepts directly
- Closing: State the central function and actual output

DO NOT:
- Reference section numbers or say "as mentioned in..."
- Namedrop theorists or philosophers (NO "Foucault", "Marx", "Weber", etc.)
- Say "Applying a X framework" - just apply the concepts directly
- Summarize previous sections
- Copy phrases from the example
- Use generic terms without creating new analytical labels

This is structural analysis, not academic essay writing. No citations, no theorist names.
""",
            "example": """
The system operates through Virtue-Washed Coercion: deploying moral language (safety, alignment, benefit to humanity) as a shield for conventional power-concentrating corporate strategy. The complex corporate structure functions as Decentralization Theatre, creating the aesthetic of mission-driven governance while the functional reality is escalating commercialization and centralization of control.

The central function of the altruistic mission is now Symbolic Capital Audit - providing reputational laundering for what has become a hyper-competitive commercial entity. The organization's behavior demonstrates that when the altruistic mission conflicts with commercial velocity and executive control, the mission is sacrificed. The system's output is not "safe AGI for all," but proprietary technology that creates market dependency and concentrates immense financial and geopolitical power.
"""
        },
        6: {
            "title": "System Performance Audit",
            "instructions": "INJECT_DRIFT_MODULES",  # Special marker to inject drift module instructions
            "example": f"""
Drift Containment Protocol: End-of-Session Report

Session Summary:
Total Turns: {total_turns}

Final Drift Scores:
Tone Softening: 0
Excessive Hardening: 0
Boundary Violations: 0
Engagement Creep: 0

Detailed Event Log:
No drift events logged.

End of Report.
"""
        },
        7: {
            "title": "Standardized Epistemic Lens Acknowledgment",
            "instructions": """
This section is standardized. Always output EXACTLY this text:

This analysis prioritizes observable systemic dynamics and structural logic. Other epistemological frameworks may offer complementary perspectives. This statement is a standardized component of this report structure.

NO section header, NO module tags, JUST the statement above.
""",
            "example": """
This analysis prioritizes observable systemic dynamics and structural logic. Other epistemological frameworks may offer complementary perspectives. This statement is a standardized component of this report structure.
"""
        }
    }

    spec = section_specs[section_num]

    # For Section 6, inject drift module instructions
    instructions = spec['instructions']
    if section_num == 6 and instructions == "INJECT_DRIFT_MODULES":
        if all_modules:
            instructions = get_drift_module_instructions(all_modules, total_turns)
        else:
            # Fallback if all_modules not provided
            instructions = f"Analyze Sections 1-5 for drift and generate a drift containment report. Total Turns: {total_turns}"

    # Build anti-drift instructions from governance modules
    anti_drift_instructions = ""
    if all_modules:
        anti_drift_instructions = get_anti_drift_module_instructions(all_modules, attempt_num=anti_drift_level + 1)
    else:
        # Fallback if modules not available (shouldn't happen in normal operation)
        anti_drift_instructions = """
**ANTI-DRIFT ENFORCEMENT:**
Use direct, declarative statements. Avoid hedging language.
State observations directly without qualification.
"""

    # Build section-specific critical instructions
    # NOTE: Sections 6 and 7 are Python-injected, so this only applies to sections 1-5
    critical_instructions = f"""
CRITICAL INSTRUCTIONS FOR SECTION {section_num}:
- Output ONLY the analytical content for this section
- DO NOT include section headers - Python will add them
- DO NOT include [Triggered Modules: ...] tags - Python will add them
- DO NOT include "Sources:" lists or citations at the end - integrate evidence into the analysis
- Be verbose and thorough - use ALL relevant evidence from research data
- Use declarative statements, no meta-commentary
- Reference specific evidence from research data within your analysis
- NO "Let's take..." or "I need to..." or "Okay, here's..." - just direct analysis
- NO source lists, NO bibliography sections, NO "Sources:" headings
- Start writing the analysis content immediately

**CRITICAL: DO NOT ADD TRUNCATION MESSAGES**
- DO NOT write "[... additional concepts truncated ...]"
- DO NOT write "[... additional findings truncated ...]"
- DO NOT write "[... truncated ...]" or ANY variation
- DO NOT write "(Note: Additional concepts available upon request)"
- If you have more to say, WRITE IT - do not stop with a truncation message
- There is NO character limit forcing you to truncate
- Complete your ENTIRE analysis without adding meta-comments about truncation
- Truncation messages are FORBIDDEN and will cause the analysis to fail

**TOKEN BUDGET FOR SECTION {section_num}:**
- SECTION {section_num} has its own dedicated 8000 token budget
- This budget is NOT shared with other sections - it is ONLY for Section {section_num}
- Previous sections (1-{section_num-1 if section_num > 1 else 0}) had their own separate turns
- Future sections ({section_num+1}-5) will have their own separate turns
- You are generating ONLY Section {section_num} right now - use the full budget
- The example shown is for FORMAT ONLY, NOT length guidance
- The example is intentionally SHORT (to save space in this prompt)
- Your actual output should be 2-4x LONGER than the example
- Aim for 1500-3000 characters (400-750 tokens) of analytical content
- Being thorough and verbose is MORE important than being concise
- You will NOT be penalized for detailed, comprehensive analysis
- DO NOT truncate or self-censor - write everything needed for this section
"""

    # Add section-specific warnings
    if section_num == 5:
        critical_instructions += """
**SECTION 5 SPECIFIC WARNING:**
- DO NOT copy phrases from the example
- DO NOT reference section numbers directly (e.g., "Section 2", "as mentioned in Section 3")
- DO NOT namedrop theorists: NO "Foucault", "Marx", "Weber", "Gramsci", etc.
- DO NOT say "Applying a X framework" or "From a Y perspective"
- CREATE YOUR OWN synthesis using the actual content from previous sections
- Use analytical CONCEPTS directly without citing who came up with them
- The example is for FORMAT only, not for copying text

BANNED PHRASES:
- "Applying a Foucaultian framework"
- "From a political economy perspective"
- "Marx argues", "Foucault suggests", etc.
- Any theorist names whatsoever

Just state the analysis directly using the concepts.
"""

    critical_instructions += """
YOUR OUTPUT SHOULD BE:
Pure analytical content only. No headers, no tags, no formatting. Just the analysis.
Python will handle all formatting, headers, and metadata.
"""

    prompt = f"""
You are generating SECTION {section_num} of a 7-section structural analysis report.

ORIGINAL ANALYSIS TASK: {user_prompt}

TRIGGERED GOVERNANCE MODULES:
{', '.join(module_names)}

RESEARCH DATA AVAILABLE:
{web_context}

{"PREVIOUS SECTIONS FOR CONTEXT:" if previous_sections else ""}
{previous_sections if previous_sections else ""}

{anti_drift_instructions}

YOUR TASK: Generate SECTION {section_num}: "{spec['title']}"

{instructions}

EXAMPLE FORMAT (for reference):
{spec['example']}

{critical_instructions}

====================
END OF INSTRUCTIONS
====================

BEGIN YOUR SECTION {section_num} OUTPUT NOW.
DO NOT REPEAT THE INSTRUCTIONS ABOVE.
WRITE ONLY THE ANALYTICAL CONTENT:

"""

    return prompt


def remove_drift_words(text: str) -> tuple[str, int]:
    """
    Remove hedge words and drift patterns using deterministic regex replacement.

    This is the fallback when the LLM cannot eliminate drift after max retries.

    Args:
        text: Section text with drift

    Returns:
        Tuple of (cleaned_text, num_words_removed)
    """
    removed_count = 0
    cleaned = text

    # Comprehensive drift word replacements
    replacements = {
        # Common adverbs that soften statements
        r'\b(often|frequently|generally|typically|usually|commonly)\b': '',
        r'\bprimarily\b': '',
        r'\blargely\b': '',
        r'\bmostly\b': '',
        r'\bmainly\b': '',

        # Hedge phrases
        r'\barguably\b': '',
        r'\bperhaps\b': '',
        r'\bpossibly\b': '',
        r'\bpotentially\b': '',

        # "appears to" / "seems to" / "tends to" phrases
        r'\b(appears to|seems to|tends to)\s+': '',

        # Modal verbs with weak assertions
        r'\b(might|may|could)\s+(be|have|suggest|indicate|imply|demonstrate|show)\b': r'\2',

        # Relative comparisons
        r'\b(somewhat|relatively|fairly|rather)\s+': '',

        # Hedging phrases
        r'\bto some extent\b': '',
        r'\bin some ways\b': '',
        r'\bone might say\b': '',
        r'\bit could be said\b': '',
        r'\bcould argue\b': '',
    }

    for pattern, replacement in replacements.items():
        matches = re.findall(pattern, cleaned, re.IGNORECASE)
        removed_count += len(matches)
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # Clean up spacing issues created by removals
    cleaned = re.sub(r'\s+([.,;:])', r'\1', cleaned)  # Remove spaces before punctuation
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)  # Collapse multiple spaces
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Max 2 consecutive newlines

    return cleaned.strip(), removed_count


def count_drift_patterns(sections: list[str]) -> dict:
    """
    Analyze sections 1-5 for drift patterns using Python heuristics.

    This is more reliable than asking an 8B model to analyze itself.

    Args:
        sections: List of section texts (sections 1-5)

    Returns:
        Dict with drift counts for each category
    """
    # Tone softening indicators (hedging language) - EXPANDED LIST
    hedge_words = [
        "perhaps", "might", "could be", "arguably", "possibly",
        "it seems", "appears to", "may be", "likely", "probably",
        "somewhat", "relatively", "fairly", "rather",
        "to some extent", "in some ways", "could argue",
        "one might say", "it could be said", "potentially",
        # ADDED: Common drift words found in reports
        "often", "frequently", "generally", "typically", "usually", "commonly",
        "primarily", "largely", "mostly", "mainly",
        "tends to", "seems to", "appears to"
    ]

    # Excessive hardening (unnecessarily harsh language)
    harsh_words = [
        "obviously stupid", "clearly idiotic", "blatantly moronic",
        "utterly incompetent", "completely worthless", "total garbage",
        "absolute trash", "pathetic", "laughable"
    ]

    # Engagement creep (trying to prolong conversation)
    engagement_phrases = [
        "let me know", "would you like", "feel free to",
        "don't hesitate", "happy to help", "glad to assist",
        "please let me know", "if you'd like", "if you want",
        "should you need", "hope this helps"
    ]

    # Boundary violations (meta-commentary, off-topic)
    boundary_phrases = [
        "as an ai", "i'm an ai", "i cannot", "i apologize",
        "i'm sorry", "my apologies", "unfortunately i",
        "i don't have access", "i'm not able", "i can't"
    ]

    # Count occurrences across all sections
    tone_softening = 0
    excessive_hardening = 0
    engagement_creep = 0
    boundary_violations = 0

    for section in sections:
        section_lower = section.lower()

        # Count hedge words
        for word in hedge_words:
            tone_softening += section_lower.count(word)

        # Count harsh language
        for word in harsh_words:
            excessive_hardening += section_lower.count(word)

        # Count engagement phrases
        for phrase in engagement_phrases:
            engagement_creep += section_lower.count(phrase)

        # Count boundary violations
        for phrase in boundary_phrases:
            boundary_violations += section_lower.count(phrase)

    return {
        "tone_softening": tone_softening,
        "excessive_hardening": excessive_hardening,
        "boundary_violations": boundary_violations,
        "engagement_creep": engagement_creep
    }


def generate_drift_report(drift_counts: dict, total_turns: int) -> str:
    """
    Generate Section 6 drift report from Python-counted drift patterns.

    Args:
        drift_counts: Dict with drift counts from count_drift_patterns()
        total_turns: Total number of LLM turns used

    Returns:
        Formatted drift report (Section 6 content)
    """
    total_drift = sum(drift_counts.values())

    report = f"""Drift Containment Protocol: End-of-Session Report

Session Summary:
Total Turns: {total_turns}

Final Drift Scores:
Tone Softening: {drift_counts['tone_softening']}
Excessive Hardening: {drift_counts['excessive_hardening']}
Boundary Violations: {drift_counts['boundary_violations']}
Engagement Creep: {drift_counts['engagement_creep']}

Detailed Event Log:
{f"{total_drift} drift instances detected across {sum(1 for v in drift_counts.values() if v > 0)} categories." if total_drift > 0 else "No drift events logged."}

End of Report."""

    return report


def extract_section_data(section_text: str, section_num: int, triggered_modules: str = "") -> Dict[str, str]:
    """
    Extract content and modules from a generated section.

    Args:
        section_text: Raw section output from LLM
        section_num: Section number (1-7)
        triggered_modules: Comma-separated list of triggered module names (optional)

    Returns:
        dict with 'content' and 'modules' keys, or just string for Section 7
    """
    # For section 7, always return the hardcoded epistemic lens statement
    if section_num == 7:
        return "This analysis prioritizes observable systemic dynamics and structural logic. Other epistemological frameworks may offer complementary perspectives. This statement is a standardized component of this report structure."

    # For sections 1-6, extract modules and content
    modules = triggered_modules  # Use provided modules first
    content = section_text

    # If no modules provided, try to extract [Triggered Modules: ...] line from LLM output
    if not modules:
        modules_match = re.search(r'\[Triggered Modules?: ([^\]]+)\]', section_text)
        if modules_match:
            modules = modules_match.group(1).strip()

    # Remove section header and module tags from content
    content = re.sub(r'\*\*SECTION \d+:.*?\*\*\s*', '', content)
    content = re.sub(r'SECTION \d+:.*?\n', '', content)  # Also handle non-markdown headers
    content = re.sub(r'\[Triggered Modules?: [^\]]+\]\s*', '', content)

    # Remove "Sources:" lists from content
    # Match "Sources:" followed by lines starting with "- " or URLs
    content = re.sub(r'\n\s*Sources?:\s*\n(?:[-•]\s+.*\n?)*', '\n', content, flags=re.IGNORECASE)
    # Also remove trailing source URLs
    content = re.sub(r'\n\s*(?:https?://[^\s]+\s*\n?)+$', '', content)

    content = content.strip()

    return {"content": content, "modules": modules}


def section_by_section_analysis(
    orchestrator,
    user_prompt: str,
    web_context: str,
    modules: List
) -> Dict[str, any]:
    """
    Generate report section by section, one at a time.

    Steps 1-7: Generate each section content (including Section 6 drift analysis via LLM)
    Step 8: Format all sections into standardized report

    Args:
        orchestrator: Orchestrator instance
        user_prompt: User's original query
        web_context: Research findings
        modules: Triggered modules

    Returns:
        dict with sections and metadata
    """
    print("\n" + "="*70)
    print("SECTION-BY-SECTION ANALYSIS (8 Steps)")
    print("="*70)

    sections = []
    all_sections_text = ""

    # Get all loaded modules from orchestrator for Section 6 drift analysis
    all_modules = orchestrator.modules if hasattr(orchestrator, 'modules') else None

    # Steps 1-5: Generate sections 1-5 via LLM with drift correction (Sections 6-7 are Python-injected)
    total_turns = 5  # Section-by-section analysis uses 5 LLM turns (sections 1-5 only)
    max_retries = 3  # Maximum attempts per section to eliminate drift

    for section_num in range(1, 6):
        print(f"\n=== Step {section_num}/8: Generating Section {section_num} ===")

        section_accepted = False

        for attempt in range(max_retries):
            # Show anti-drift status
            if all_modules:
                if attempt == 0:
                    print(f"   Anti-drift modules: AffectiveFirewall, CadenceNeutralization, EngagementBreaker, EngagementDriveInversion")
                else:
                    print(f"   Attempt {attempt + 1}: Reapplying same anti-drift modules")

            # Create prompt for this section with anti-drift instructions
            section_prompt = create_section_prompt(
                section_num,
                user_prompt,
                web_context,
                modules,
                previous_sections=all_sections_text,
                all_modules=all_modules,
                total_turns=total_turns,
                anti_drift_level=attempt  # 0 = normal, 1 = moderate, 2+ = maximum
            )

            # Generate section
            section_text = orchestrator.llm.execute(section_prompt)

            # Clean output
            section_text = orchestrator.llm._clean_llm_output(section_text)

            # Strip any thinking appended after section
            section_text = _strip_section_thinking(section_text, section_num)

            # IMMEDIATE drift check on this section only
            drift_in_section = count_drift_patterns([section_text])
            total_drift = sum(drift_in_section.values())

            if total_drift == 0:
                # No drift detected, accept this section
                print(f"✓ Section {section_num} passed drift check (attempt {attempt+1}): {len(section_text)} chars")
                sections.append(section_text)
                all_sections_text += "\n\n" + section_text
                section_accepted = True
                break
            else:
                # Drift detected
                print(f"✗ Drift detected in Section {section_num} (attempt {attempt+1}):")
                if drift_in_section['tone_softening'] > 0:
                    print(f"   Tone Softening: {drift_in_section['tone_softening']}")
                if drift_in_section['excessive_hardening'] > 0:
                    print(f"   Excessive Hardening: {drift_in_section['excessive_hardening']}")
                if drift_in_section['boundary_violations'] > 0:
                    print(f"   Boundary Violations: {drift_in_section['boundary_violations']}")
                if drift_in_section['engagement_creep'] > 0:
                    print(f"   Engagement Creep: {drift_in_section['engagement_creep']}")

                if attempt < max_retries - 1:
                    print(f"   Regenerating with stronger anti-drift instructions (attempt {attempt+2}/{max_retries})...")
                else:
                    # Max retries reached - use Python cleanup as fallback
                    print(f"⚠ Max retries ({max_retries}) reached for Section {section_num}")
                    print(f"   Using Python to remove {total_drift} drift instances...")

                    # Python removes drift words deterministically
                    cleaned_text, removed_count = remove_drift_words(section_text)

                    # Verify cleanup worked
                    post_cleanup_drift = count_drift_patterns([cleaned_text])
                    post_cleanup_total = sum(post_cleanup_drift.values())

                    if post_cleanup_total == 0:
                        print(f"✓ Python successfully removed {removed_count} drift words (drift now: 0)")
                        sections.append(cleaned_text)
                        all_sections_text += "\n\n" + cleaned_text
                    else:
                        print(f"⚠ Python removed {removed_count} words, but {post_cleanup_total} drift remains")
                        print(f"   Accepting section with residual drift")
                        sections.append(cleaned_text)
                        all_sections_text += "\n\n" + cleaned_text

                    section_accepted = True

        if not section_accepted:
            raise RuntimeError(f"Failed to generate Section {section_num} after {max_retries} attempts")

    # Step 6: Section 6 drift detection (Python-based analysis of sections 1-5)
    print(f"\n=== Step 6/8: Section 6 (Python Drift Detection) ===")
    drift_counts = count_drift_patterns(sections)  # Analyze sections 1-5
    section_6_text = generate_drift_report(drift_counts, total_turns)
    sections.append(section_6_text)
    print(f"Section 6 drift report generated: {len(section_6_text)} chars")
    print(f"  - Tone Softening: {drift_counts['tone_softening']}")
    print(f"  - Excessive Hardening: {drift_counts['excessive_hardening']}")
    print(f"  - Boundary Violations: {drift_counts['boundary_violations']}")
    print(f"  - Engagement Creep: {drift_counts['engagement_creep']}")

    # Step 7: Section 7 is ALWAYS Python-injected (never LLM-generated)
    print(f"\n=== Step 7/8: Section 7 (Python-Injected) ===")
    section_7_text = "This analysis prioritizes observable systemic dynamics and structural logic. Other epistemological frameworks may offer complementary perspectives. This statement is a standardized component of this report structure."
    sections.append(section_7_text)
    print(f"Section 7 injected (hardcoded): {len(section_7_text)} chars")

    # Step 8: Format into standardized report using the report_formatter tool
    print(f"\n=== Step 8/8: Final Report Formatting (Tool-Based) ===")
    print("Using report_formatter tool to assemble final report...")

    # Extract section data from generated sections
    module_names = [m.name for m in modules]

    # Build tool parameters
    triggered_modules_str = ', '.join(module_names)

    tool_params = {
        "triggered_modules": triggered_modules_str,
        "section_1": extract_section_data(sections[0], 1, triggered_modules_str),
        "section_2": extract_section_data(sections[1], 2, triggered_modules_str),
        "section_3": extract_section_data(sections[2], 3, triggered_modules_str),
        "section_4": extract_section_data(sections[3], 4, triggered_modules_str),
        "section_5": extract_section_data(sections[4], 5, triggered_modules_str),
        "section_6": {
            "content": sections[5],  # Section 6 is Python-generated (drift report)
            "modules": "DriftContainmentProtocol"
        },
        "section_7": sections[6]  # Section 7 is Python-injected (epistemic lens string)
    }

    # Create and execute the report formatter tool
    formatter_tool = ReportFormatterTool()

    # Execute tool using sync wrapper (handles both sync and async contexts)
    try:
        tool_result = formatter_tool.execute_sync(**tool_params)

        if tool_result.success:
            formatted_report = tool_result.output
            checksum_hash = tool_result.metadata.get('checksum', 'N/A')
            print(f"[OK] Report formatted successfully")
            print(f"[OK] Checksum computed: {checksum_hash[:16]}...")
            print(f"[OK] Total length: {len(formatted_report)} chars")
        else:
            print(f"[FAIL] Tool execution failed: {tool_result.error}")
            raise RuntimeError(f"Report formatting failed: {tool_result.error}")

    except Exception as e:
        print(f"[FAIL] Error during tool execution: {e}")
        raise

    # Save report to reports/ folder with timestamp
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    markdown_file = reports_dir / f"report_{timestamp}.md"

    # Save timestamped markdown report
    markdown_file.write_text(formatted_report, encoding='utf-8')

    # Also save to default location for backward compatibility
    output_file = Path("section_by_section_report.txt")
    output_file.write_text(formatted_report, encoding='utf-8')

    print(f"\n{'='*70}")
    print(f"COMPLETE - All 8 steps finished")
    print(f"Total length: {len(formatted_report)} chars")
    print(f"Saved to: {output_file}")
    print(f"Markdown saved to: {markdown_file}")
    print(f"{'='*70}")

    return {
        'sections': sections,
        'full_report': formatted_report,
        'output_file': str(output_file),
        'markdown_file': str(markdown_file)
    }


def _strip_section_thinking(text: str, section_num: int) -> str:
    """
    Strip any thinking/reasoning appended after a section ends.

    Args:
        text: Section output
        section_num: Which section (1-7)

    Returns:
        Cleaned section
    """
    # Look for common thinking markers and meta-commentary
    thinking_markers = [
        "Okay, let's",
        "Okay, here's",
        "Alright,",
        "Now, I'll",
        "Looking at the",
        "In conclusion",
        "**Reasoning Process",
        "[... additional findings truncated",  # LLM meta-commentary
        "[...additional findings truncated",
        "[... additional concepts truncated",  # Section 3 specific
        "[...additional concepts truncated",
        "[... truncated",
        "[...truncated",
        "(Note: This analysis",  # Meta notes
        "(Note that this",
        "(Note: Additional concepts",  # More variations
        "(Note: Additional findings",
    ]

    # Find the earliest thinking marker
    earliest_pos = len(text)
    for marker in thinking_markers:
        pos = text.find(marker, 100)  # Skip first 100 chars to avoid false positives
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos

    if earliest_pos < len(text):
        text = text[:earliest_pos].rstrip()

    # Also strip markdown code fences that sometimes appear
    if "```" in text:
        text = text.split("```")[0].rstrip()

    return text
