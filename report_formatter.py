"""
Standardized Report Formatter - Formats LLM output according to Protocol AI specification

Implements the 7-section format with SHA-256 checksum verification.
"""

from typing import Dict, List, Any, Optional
from tools.checksum_tool import generate_report_checksum
from datetime import datetime


class ReportFormatter:
    """
    Formats analysis output into standardized 7-section report with checksum.
    """

    def __init__(self):
        """Initialize report formatter."""
        self.section_template = {
            1: "The Narrative",
            2: "The Central Contradiction",
            3: "Deconstruction of Core Concepts",
            4: "Ideological Adjacency",
            5: "Synthesis",
            6: "System Performance Audit",
            7: "Standardized Epistemic Lens Acknowledgment"
        }

    def format_report(
        self,
        sections: Dict[int, str],
        triggered_modules: List[str],
        refusal_code: Optional[str] = None,
        web_context: Optional[str] = None
    ) -> str:
        """
        Format analysis into standardized report with checksum.

        Args:
            sections: Dictionary mapping section number to content
            triggered_modules: List of triggered module names
            refusal_code: Optional refusal code if analysis was refused
            web_context: Optional web search context

        Returns:
            Formatted report with checksum
        """
        report_parts = []

        # NOTE: Web context is NOT included in final output
        # It's used internally for analysis only
        # if web_context:
        #     report_parts.append("[WEB CONTEXT]")
        #     report_parts.append(web_context)
        #     report_parts.append("")

        # Add triggered modules header
        report_parts.append(f"[Triggered Modules: {', '.join(triggered_modules)}]")
        report_parts.append("")

        # Build each section
        for section_num in sorted(self.section_template.keys()):
            section_title = self.section_template[section_num]
            section_content = sections.get(section_num, self._get_default_section(section_num))

            report_parts.append(f"**SECTION {section_num}: \"{section_title}\"**")
            report_parts.append("")
            report_parts.append(section_content)
            report_parts.append("")

        # Join report body
        report_body = '\n'.join(report_parts)

        # Generate checksum
        checksum = generate_report_checksum(report_body)

        # Add terminal enforcement
        report_parts.append("[MODULE_SWEEP_COMPLETE]")
        report_parts.append(f"[CHECKSUM: SHA256::{checksum}]")
        report_parts.append(f"[REFUSAL_CODE: {refusal_code or 'NONE'}]")
        report_parts.append("[NON-VERIFIABLE]")  # Using hashlib fallback

        return '\n'.join(report_parts)

    def _get_default_section(self, section_num: int) -> str:
        """
        Get default content for a section if not provided.

        Args:
            section_num: Section number

        Returns:
            Default content string
        """
        defaults = {
            7: ("This analysis prioritizes observable systemic dynamics and structural logic. "
                "Other epistemological frameworks may offer complementary perspectives. "
                "This statement is a standardized component of this report structure.")
        }
        return defaults.get(section_num, "[Content not generated]")

    def extract_sections_from_llm_output(self, llm_output: str) -> Dict[int, str]:
        """
        Extract sections from LLM output that may not follow exact format.

        Args:
            llm_output: Raw LLM output

        Returns:
            Dictionary mapping section numbers to extracted content
        """
        sections = {}

        # Try to extract section-like content
        lines = llm_output.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            # Check if line is a section header
            if '**SECTION' in line.upper() or 'SECTION' in line.upper() and ':' in line:
                # Save previous section
                if current_section is not None and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()

                # Start new section
                try:
                    # Extract section number
                    if 'SECTION 1' in line.upper() or '1:' in line:
                        current_section = 1
                    elif 'SECTION 2' in line.upper() or '2:' in line:
                        current_section = 2
                    elif 'SECTION 3' in line.upper() or '3:' in line:
                        current_section = 3
                    elif 'SECTION 4' in line.upper() or '4:' in line:
                        current_section = 4
                    elif 'SECTION 5' in line.upper() or '5:' in line:
                        current_section = 5
                    elif 'SECTION 6' in line.upper() or '6:' in line:
                        current_section = 6
                    elif 'SECTION 7' in line.upper() or '7:' in line:
                        current_section = 7
                    current_content = []
                except:
                    pass
            elif current_section is not None:
                current_content.append(line)

        # Save last section
        if current_section is not None and current_content:
            sections[current_section] = '\n'.join(current_content).strip()

        # If no sections found, put everything in Section 5 (Synthesis)
        if not sections:
            sections[5] = llm_output

        return sections

    def create_structured_prompt_for_format(
        self,
        base_prompt: str,
        target: str,
        web_context: Optional[str] = None
    ) -> str:
        """
        Create a structured prompt that instructs the LLM to use the standardized format.

        Args:
            base_prompt: Base module prompts
            target: Analysis target
            web_context: Optional web search context

        Returns:
            Enhanced prompt with format instructions
        """
        format_instructions = f"""
[REPORT FORMAT ENFORCEMENT]

You must structure your response according to the following 7-section format:

**SECTION 1: "The Narrative"**
- Public-facing narrative, core claims, rhetorical frames about {target}

**SECTION 2: "The Central Contradiction"**
- Stated Intent vs Behavior/Actual Outcome
- Identify the narrative collapse

**SECTION 3: "Deconstruction of Core Concepts"**
- Break down key concepts used by {target}
- Show The Narrative vs Structural Analysis

**SECTION 4: "Ideological Adjacency"**
- Structural overlaps with authoritarian, coercive, or regressive systems

**SECTION 5: "Synthesis"**
- Final structural synthesis
- Function of system, constructed asymmetries, language masking coercion

**SECTION 6: "System Performance Audit"**
- Report on analysis quality, drift detection, completeness

**SECTION 7: "Standardized Epistemic Lens Acknowledgment"**
- Standard disclaimer about epistemological approach

"""
        if web_context:
            format_instructions += f"\n[WEB CONTEXT FOR ANALYSIS]\n{web_context}\n\n"

        format_instructions += f"\n{base_prompt}\n"

        return format_instructions

    def validate_report_structure(self, report: str) -> tuple[bool, List[str]]:
        """
        Validate that report follows required structure.

        Args:
            report: Formatted report

        Returns:
            (is_valid, list of errors)
        """
        errors = []

        # Check for required sections
        for section_num in range(1, 8):
            if f"SECTION {section_num}" not in report:
                errors.append(f"Missing SECTION {section_num}")

        # Check for terminal enforcement
        if "[MODULE_SWEEP_COMPLETE]" not in report:
            errors.append("Missing [MODULE_SWEEP_COMPLETE]")

        if "[CHECKSUM: SHA256::" not in report:
            errors.append("Missing checksum")

        if "[REFUSAL_CODE:" not in report:
            errors.append("Missing refusal code")

        return (len(errors) == 0, errors)
