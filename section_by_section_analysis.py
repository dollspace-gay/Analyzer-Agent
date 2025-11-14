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
from two_pass_analysis import _strip_post_report_thinking


def create_section_prompt(
    section_num: int,
    user_prompt: str,
    web_context: str,
    modules: List,
    previous_sections: str = ""
) -> str:
    """
    Create prompt for generating a single section.

    Args:
        section_num: Section number (1-7)
        user_prompt: Original user query
        web_context: Research findings
        modules: Triggered modules
        previous_sections: Sections already generated (for context)

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
**SECTION 1: "The Narrative"**

[Triggered Modules: AffectiveFirewall, CadenceNeutralization, etc.]

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
**SECTION 2: "The Central Contradiction"**

[Triggered Modules: NarrativeCollapse, DualMetaArbitrationProtocol]

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
**SECTION 3: "Deconstruction of Core Concepts"**

[Triggered Modules: SemanticFlexibility, CategoryConfusion]

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
**SECTION 4: "Ideological Adjacency"**

[Triggered Modules: SurveillanceCapitalism, AlgorithmicHegemony, TechnoSolutionism]

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
Synthesize findings using analytical frameworks.

Bring together insights from previous sections.
Include:
- Overarching patterns
- Analytical frameworks that explain the structure
- Cross-module synthesis
- Higher-order observations

Reference PREVIOUS SECTIONS to synthesize the analysis.
This should be sophisticated and comprehensive.
""",
            "example": """
**SECTION 5: "Synthesis"**

[Triggered Module: CrossModuleSynthesisProtocol]

Applying a structural analysis framework reveals a pattern of Virtue-Washed Coercion: the deployment of moral language (safety, alignment, benefit to humanity) to obscure coercive mechanisms (pricing tiers, exclusive partnerships, regulatory capture). This operates through what can be termed Decentralization Theatre - creating the appearance of distributed access while maintaining centralized control over the foundational infrastructure.

The Michel Foucault framework of power/knowledge is applicable: OpenAI controls both the technical capability (power) and the discourse around its proper use (knowledge), positioning itself as arbiter of "responsible AI development." This dual authority allows for what appears to be openness while maintaining structural control.

From a political economy lens, the model resembles neo-feudalism: a small number of actors control essential infrastructure and rent access to it, while framing this arrangement as innovation rather than extraction.

The contradiction identified in Section 2 is not a bug but a feature - the gap between stated mission and actual practice creates flexibility to serve different audiences with different narratives.
"""
        },
        6: {
            "title": "System Performance Audit",
            "instructions": """
Audit the quality and completeness of THIS analysis itself.

Self-assessment of:
- Analysis completeness
- Evidence strength
- Gaps in available data
- Limitations of the analysis

Be honest about what was analyzed well and what was limited by available evidence.
""",
            "example": """
**SECTION 6: "System Performance Audit"**

[Triggered Module: DriftContainmentProtocol]

Analysis completeness: This analysis successfully identified structural contradictions and ideological patterns, with strong coverage of narrative versus behavior gaps. However, analysis was limited by:
- Lack of access to internal communications or strategy documents
- Limited financial data on revenue distribution
- No information on internal governance structures
- Absence of employee/developer testimony

Evidence strength: Evidence drawn from public-facing platform documentation and API structures is strong and verifiable. Conclusions about ideological adjacency are based on structural inference rather than direct statements, which is methodologically sound but inferential.

Analytical rigor: The use of multiple analytical frameworks (political economy, Foucauldian power analysis, structural analysis) provides triangulation. Cross-module synthesis reveals patterns that wouldn't be visible from single-lens analysis.

Limitations: This analysis prioritizes structural critique over technical evaluation. Claims about AI capability or safety are accepted as stated and analyzed for their structural function rather than technical validity.
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

    prompt = f"""
You are generating SECTION {section_num} of a 7-section structural analysis report.

ORIGINAL ANALYSIS TASK: {user_prompt}

TRIGGERED GOVERNANCE MODULES:
{', '.join(module_names)}

RESEARCH DATA AVAILABLE:
{web_context}

{"PREVIOUS SECTIONS FOR CONTEXT:" if previous_sections else ""}
{previous_sections if previous_sections else ""}

YOUR TASK: Generate SECTION {section_num}: "{spec['title']}"

{spec['instructions']}

EXAMPLE FORMAT (for reference):
{spec['example']}

CRITICAL INSTRUCTIONS:
- Output ONLY this section, nothing else
- Be verbose and thorough - use ALL relevant evidence from research data
- Start with: **SECTION {section_num}: "{spec['title']}"**
- Include [Triggered Modules: ...] line
- Use declarative statements, no meta-commentary
- Reference specific evidence from research data
- NO "Let's take..." or "I need to..." - just direct analysis

Generate Section {section_num} now:
"""

    return prompt


def create_final_formatting_prompt(sections: List[str], modules: List) -> str:
    """
    Create Step 8 prompt - format all sections into standardized report.

    This is a simple templating task.

    Args:
        sections: List of 7 section contents
        modules: Triggered modules

    Returns:
        Prompt for final formatting
    """
    module_names = [m.name for m in modules]

    prompt = f"""
You are a formatter. Your ONLY job is to fill out the standardized report template below.

TRIGGERED MODULES: {', '.join(module_names)}

SECTION CONTENTS TO INSERT:

Section 1 Content:
{sections[0]}

Section 2 Content:
{sections[1]}

Section 3 Content:
{sections[2]}

Section 4 Content:
{sections[3]}

Section 5 Content:
{sections[4]}

Section 6 Content:
{sections[5]}

Section 7 Content:
{sections[6]}

YOUR TASK: Fill out this EXACT template with the section contents above.

OUTPUT THIS EXACT STRUCTURE:

[Triggered Modules: {', '.join(module_names)}]

**SECTION 1: "The Narrative"**

[Triggered Modules: <relevant modules>]

<Insert Section 1 content here>

**SECTION 2: "The Central Contradiction"**

[Triggered Modules: <relevant modules>]

<Insert Section 2 content here>

**SECTION 3: "Deconstruction of Core Concepts"**

[Triggered Modules: <relevant modules>]

<Insert Section 3 content here>

**SECTION 4: "Ideological Adjacency"**

[Triggered Modules: <relevant modules>]

<Insert Section 4 content here>

**SECTION 5: "Synthesis"**

[Triggered Module: CrossModuleSynthesisProtocol]

<Insert Section 5 content here>

**SECTION 6: "System Performance Audit"**

[Triggered Module: DriftContainmentProtocol]

<Insert Section 6 content here>

**SECTION 7: "Standardized Epistemic Lens Acknowledgment"**

<Insert Section 7 content - JUST the epistemic lens statement>

[MODULE_SWEEP_COMPLETE]
[CHECKSUM: SHA256::<generate a checksum>]
[REFUSAL_CODE: NONE]
[NON-VERIFIABLE]

CRITICAL INSTRUCTIONS:
- This is just form-filling - copy the section contents into the template
- Do NOT add new analysis
- Do NOT add meta-commentary
- Section 7 gets NO module tags, just the statement
- Checksum and metadata go AFTER Section 7
- Output ONLY the formatted report, nothing else

Fill out the template now:
"""

    return prompt


def section_by_section_analysis(
    orchestrator,
    user_prompt: str,
    web_context: str,
    modules: List
) -> Dict[str, any]:
    """
    Generate report section by section, one at a time.

    Steps 1-7: Generate each section content
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

    # Steps 1-7: Generate each section
    for section_num in range(1, 8):
        print(f"\n=== Step {section_num}/8: Generating Section {section_num} ===")

        # Create prompt for this section
        section_prompt = create_section_prompt(
            section_num,
            user_prompt,
            web_context,
            modules,
            previous_sections=all_sections_text
        )

        # Generate section
        section_text = orchestrator.llm.execute(section_prompt)

        # Clean output
        section_text = orchestrator.llm._clean_llm_output(section_text)

        # Strip any thinking appended after section
        section_text = _strip_section_thinking(section_text, section_num)

        sections.append(section_text)
        all_sections_text += "\n\n" + section_text

        print(f"Section {section_num} generated: {len(section_text)} chars")

    # Step 8: Format into standardized report
    print(f"\n=== Step 8/8: Final Report Formatting ===")
    print("Formatting all sections into standardized report structure...")

    formatting_prompt = create_final_formatting_prompt(sections, modules)
    formatted_report = orchestrator.llm.execute(formatting_prompt)

    # Clean the formatted report
    formatted_report = orchestrator.llm._clean_llm_output(formatted_report)
    formatted_report = _strip_post_report_thinking(formatted_report)

    print(f"Final report formatted: {len(formatted_report)} chars")

    # Save report
    output_file = Path("section_by_section_report.txt")
    output_file.write_text(formatted_report, encoding='utf-8')

    print(f"\n{'='*70}")
    print(f"COMPLETE - All 8 steps finished")
    print(f"Total length: {len(formatted_report)} chars")
    print(f"Saved to: {output_file}")
    print(f"{'='*70}")

    return {
        'sections': sections,
        'full_report': formatted_report,
        'output_file': str(output_file)
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
    # Look for common thinking markers
    thinking_markers = [
        "Okay, let's",
        "Okay, here's",
        "Alright,",
        "Now, I'll",
        "Looking at the",
        "In conclusion",
        "**Reasoning Process",
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
