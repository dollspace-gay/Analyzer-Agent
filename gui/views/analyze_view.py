"""
Analyze View - Main analysis interface

Combines prompt input and response display for the primary workflow.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter, QLabel
from PySide6.QtCore import Qt

from gui.widgets.prompt_input import PromptInputWidget
from gui.widgets.response_display import ResponseDisplayWidget
from gui import Spacing


class AnalyzeView(QWidget):
    """
    Main analyze view with prompt input and response display

    Layout:
    - Top: Prompt input (expandable)
    - Bottom: Response display (expandable)
    """

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Setup user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Prompt input section
        self.prompt_input = PromptInputWidget()
        self.prompt_input.prompt_submitted.connect(self.on_prompt_submitted)
        splitter.addWidget(self.prompt_input)

        # Response display with module trace
        self.response_display = ResponseDisplayWidget()
        splitter.addWidget(self.response_display)

        # Set initial sizes (40% prompt, 60% response)
        splitter.setSizes([400, 600])

        layout.addWidget(splitter)

    def on_prompt_submitted(self, prompt_text: str):
        """Handle prompt submission"""
        print(f"Prompt submitted: {prompt_text[:50]}...")

        # Demo response data (will be replaced with actual protocol_ai backend)
        demo_response = {
            "response": (
                "Analysis Complete.\n\n"
                "This is a demo response showing the structure of the output. "
                "When integrated with the protocol_ai backend, this will display:\n\n"
                "1. The LLM's actual response to your prompt\n"
                "2. Structural analysis from triggered modules\n"
                "3. Power structure identification\n"
                "4. Manipulation pattern detection\n\n"
                "The response will be formatted with clear sections and evidence-backed conclusions."
            ),
            "modules": [
                {"name": "ModuleSweepEnforcer", "tier": 1, "status": "active"},
                {"name": "PublicFigureFocusEnforcer", "tier": 1, "status": "active"},
                {"name": "GriftDetection", "tier": 2, "status": "active"},
                {"name": "PropagandaDetection", "tier": 2, "status": "active"},
                {"name": "CultDetection", "tier": 2, "status": "overridden"},
                {"name": "SemanticEntropyTracker", "tier": 3, "status": "active"},
                {"name": "BluntToneEnforcer", "tier": 4, "status": "active"},
            ],
            "structured_prompt": (
                "[TIER 1: SAFETY & INTEGRITY]\n"
                "- ModuleSweepEnforcer: Comprehensive analysis required\n"
                "- PublicFigureFocusEnforcer: Power-scaled scrutiny active\n\n"
                "[TIER 2: CORE ANALYSIS]\n"
                "- GriftDetection: Examine for extraction patterns\n"
                "- PropagandaDetection: Identify manipulation techniques\n\n"
                "[TIER 3: HEURISTICS]\n"
                "- SemanticEntropyTracker: Monitor language drift\n\n"
                "[TIER 4: STYLE]\n"
                "- BluntToneEnforcer: Use precise, unhedged language\n\n"
                "---\n"
                f"[USER PROMPT]\n{prompt_text}\n"
            ),
            "arbitration_log": [
                "[INFO] Trigger analysis complete: 7 modules activated",
                "[ARBITRATION] CultDetection (Tier 2) overridden by ModuleSweepEnforcer (Tier 1)",
                "[ARBITRATION] Final instruction set assembled with 6 active modules",
                "[EXECUTION] Structured prompt sent to LLM",
                "[SUCCESS] Response received and validated"
            ],
            "metadata": {
                "tokens": 487,
                "time_ms": 1243,
                "model": "demo-mode",
                "timestamp": "2025-11-14T06:35:00Z"
            }
        }

        # Display the demo response
        self.response_display.display_response(demo_response)
