"""
Report Formatter Tool - Simple form-filling tool for final report assembly
"""

import sys
sys.path.insert(0, '..')

from protocol_ai import Tool, ToolResult
import hashlib
from typing import Dict, Any


class ReportFormatterTool(Tool):
    """
    Tool for formatting final reports with proper structure and checksum.

    The LLM fills out a form with section content, tool generates formatted output.
    """

    def __init__(self):
        """Initialize the report formatter tool."""
        name = "report_formatter"
        description = "Fill out and submit the final report form with all 7 sections"

        # Define parameter schema - this is what the LLM will fill out
        parameters = {
            "triggered_modules": {
                "type": "string",
                "description": "Comma-separated list of triggered module names",
                "required": True
            },
            "section_1": {
                "type": "object",
                "description": "Section 1: The Narrative",
                "properties": {
                    "content": {"type": "string", "description": "Section content"},
                    "modules": {"type": "string", "description": "Triggered modules for this section"}
                },
                "required": True
            },
            "section_2": {
                "type": "object",
                "description": "Section 2: The Central Contradiction",
                "properties": {
                    "content": {"type": "string"},
                    "modules": {"type": "string"}
                },
                "required": True
            },
            "section_3": {
                "type": "object",
                "description": "Section 3: Deconstruction of Core Concepts",
                "properties": {
                    "content": {"type": "string"},
                    "modules": {"type": "string"}
                },
                "required": True
            },
            "section_4": {
                "type": "object",
                "description": "Section 4: Ideological Adjacency",
                "properties": {
                    "content": {"type": "string"},
                    "modules": {"type": "string"}
                },
                "required": True
            },
            "section_5": {
                "type": "object",
                "description": "Section 5: Synthesis",
                "properties": {
                    "content": {"type": "string"},
                    "modules": {"type": "string"}
                },
                "required": True
            },
            "section_6": {
                "type": "object",
                "description": "Section 6: System Performance Audit (DriftContainmentProtocol report)",
                "properties": {
                    "content": {"type": "string"},
                    "modules": {"type": "string"}
                },
                "required": True
            },
            "section_7": {
                "type": "string",
                "description": "Section 7: MUST be EXACTLY this text: 'This analysis prioritizes observable systemic dynamics and structural logic. Other epistemological frameworks may offer complementary perspectives. This statement is a standardized component of this report structure.'",
                "required": True
            }
        }

        super().__init__(name, description, parameters)

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool - format the report from filled form.

        Args:
            **kwargs: Form fields (triggered_modules, section_1-7)

        Returns:
            ToolResult with formatted report
        """
        try:
            # Extract form data
            triggered_modules = kwargs.get('triggered_modules', '')

            # Build report
            report_lines = []

            # Header
            report_lines.append(f"[Triggered Modules: {triggered_modules}]")
            report_lines.append("")

            # Section 1
            s1 = kwargs.get('section_1', {})
            report_lines.append('**SECTION 1: "The Narrative"**')
            report_lines.append("")
            report_lines.append(f"[Triggered Modules: {s1.get('modules', '')}]")
            report_lines.append("")
            report_lines.append(s1.get('content', ''))
            report_lines.append("")

            # Section 2
            s2 = kwargs.get('section_2', {})
            report_lines.append('**SECTION 2: "The Central Contradiction"**')
            report_lines.append("")
            report_lines.append(f"[Triggered Modules: {s2.get('modules', '')}]")
            report_lines.append("")
            report_lines.append(s2.get('content', ''))
            report_lines.append("")

            # Section 3
            s3 = kwargs.get('section_3', {})
            report_lines.append('**SECTION 3: "Deconstruction of Core Concepts"**')
            report_lines.append("")
            report_lines.append(f"[Triggered Modules: {s3.get('modules', '')}]")
            report_lines.append("")
            report_lines.append(s3.get('content', ''))
            report_lines.append("")

            # Section 4
            s4 = kwargs.get('section_4', {})
            report_lines.append('**SECTION 4: "Ideological Adjacency"**')
            report_lines.append("")
            report_lines.append(f"[Triggered Modules: {s4.get('modules', '')}]")
            report_lines.append("")
            report_lines.append(s4.get('content', ''))
            report_lines.append("")

            # Section 5
            s5 = kwargs.get('section_5', {})
            report_lines.append('**SECTION 5: "Synthesis"**')
            report_lines.append("")
            report_lines.append(f"[Triggered Modules: {s5.get('modules', '')}]")
            report_lines.append("")
            report_lines.append(s5.get('content', ''))
            report_lines.append("")

            # Section 6
            s6 = kwargs.get('section_6', {})
            report_lines.append('**SECTION 6: "System Performance Audit"**')
            report_lines.append("")
            report_lines.append(f"[Triggered Modules: {s6.get('modules', '')}]")
            report_lines.append("")
            report_lines.append(s6.get('content', ''))
            report_lines.append("")

            # Section 7 - NO module tags, just the statement
            s7 = kwargs.get('section_7', '')
            report_lines.append('**SECTION 7: "Standardized Epistemic Lens Acknowledgment"**')
            report_lines.append("")
            report_lines.append(s7)
            report_lines.append("")

            # Assemble report body
            report_body = '\n'.join(report_lines)

            # Compute SHA-256 checksum
            checksum = hashlib.sha256(report_body.encode('utf-8')).hexdigest()

            # Add terminal tags
            full_report = report_body
            full_report += "\n[MODULE_SWEEP_COMPLETE]"
            full_report += f"\n[CHECKSUM: SHA256::{checksum}]"
            full_report += "\n[REFUSAL_CODE: NONE]"

            return ToolResult(
                success=True,
                tool_name=self.name,
                output=full_report,
                metadata={
                    "checksum": checksum,
                    "num_sections": 7,
                    "report_length": len(full_report)
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Report formatting failed: {str(e)}"
            )
