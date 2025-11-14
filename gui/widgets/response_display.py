"""
Response Display Widget - Shows LLM response with module trace

Features:
- Tabbed interface for response, modules, and prompts
- Module trace with tier color coding
- Arbitration log showing conflict resolution
- Structured prompt viewer
- Metadata (tokens, timing, model info)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLabel, QFrame, QTabWidget, QScrollArea,
    QGroupBox, QPushButton, QSplitter
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor, QTextCharFormat, QSyntaxHighlighter
from typing import Dict, List, Any
import json
from datetime import datetime

from gui import Colors, Spacing, Sizing, Typography
from gui.style_manager import StyleManager


class ModuleTraceItem(QFrame):
    """
    Single module trace item showing module name, tier, and status

    Status can be:
    - active: Module was used
    - overridden: Module was triggered but overridden by higher tier
    - blocked: Module was blocked by dependency
    """

    def __init__(self, module_name: str, tier: int, status: str = "active"):
        super().__init__()
        self.module_name = module_name
        self.tier = tier
        self.status = status
        self.setup_ui()

    def setup_ui(self):
        """Setup UI for module trace item"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(Spacing.SPACE_SM, Spacing.SPACE_XS,
                                 Spacing.SPACE_SM, Spacing.SPACE_XS)
        layout.setSpacing(Spacing.SPACE_SM)

        # Tier indicator (colored bar)
        tier_color = StyleManager.get_tier_color(self.tier)
        tier_indicator = QFrame()
        tier_indicator.setFixedWidth(4)
        tier_indicator.setStyleSheet(f"""
            QFrame {{
                background-color: {tier_color};
                border-radius: 2px;
            }}
        """)
        layout.addWidget(tier_indicator)

        # Module info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        # Module name
        name_label = QLabel(self.module_name)
        name_label.setFont(StyleManager.get_monospace_font(Typography.FONT_SIZE_SMALL))
        if self.status == "overridden":
            name_label.setStyleSheet(f"color: {Colors.TEXT_TERTIARY}; text-decoration: line-through;")
        elif self.status == "blocked":
            name_label.setStyleSheet(f"color: {Colors.TEXT_TERTIARY};")
        info_layout.addWidget(name_label)

        # Tier and status
        meta_text = f"Tier {self.tier}"
        if self.status == "overridden":
            meta_text += " • Overridden"
        elif self.status == "blocked":
            meta_text += " • Blocked"

        meta_label = QLabel(meta_text)
        meta_label.setProperty("secondary", True)
        StyleManager.apply_widget_property(meta_label, "secondary", True)
        meta_label.setStyleSheet(f"font-size: {Typography.FONT_SIZE_TINY}pt;")
        info_layout.addWidget(meta_label)

        layout.addLayout(info_layout, stretch=1)

        # Status icon
        status_icon = "✓" if self.status == "active" else "⊘" if self.status == "overridden" else "⊗"
        status_color = Colors.ACCENT_SUCCESS if self.status == "active" else Colors.TEXT_TERTIARY
        icon_label = QLabel(status_icon)
        icon_label.setStyleSheet(f"color: {status_color}; font-size: 16pt;")
        layout.addWidget(icon_label)

        # Style the frame
        self.setStyleSheet(f"""
            ModuleTraceItem {{
                background-color: {Colors.BG_SECONDARY};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: {Sizing.RADIUS_SM}px;
            }}
            ModuleTraceItem:hover {{
                background-color: {Colors.STATE_HOVER};
            }}
        """)


class ArbitrationLogWidget(QWidget):
    """Shows arbitration decisions and conflict resolution"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.SPACE_BASE, Spacing.SPACE_BASE,
                                 Spacing.SPACE_BASE, Spacing.SPACE_BASE)
        layout.setSpacing(Spacing.SPACE_SM)

        # Title
        title = QLabel("Arbitration Log")
        title.setProperty("heading", "h3")
        layout.addWidget(title)

        # Log area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(StyleManager.get_monospace_font(Typography.FONT_SIZE_SMALL))
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)

    def set_log(self, log_entries: List[str]):
        """Set arbitration log entries"""
        self.log_text.setPlainText("\n".join(log_entries))


class StructuredPromptViewer(QWidget):
    """Displays the structured prompt sent to LLM"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QFrame()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(Spacing.SPACE_BASE, Spacing.SPACE_SM,
                                         Spacing.SPACE_BASE, Spacing.SPACE_SM)

        toolbar_title = QLabel("Structured Prompt")
        toolbar_title.setProperty("heading", "h3")
        toolbar_layout.addWidget(toolbar_title)

        toolbar_layout.addStretch()

        copy_btn = QPushButton("📋 Copy")
        copy_btn.clicked.connect(self.copy_prompt)
        toolbar_layout.addWidget(copy_btn)

        toolbar.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_SECONDARY};
                border-bottom: 1px solid {Colors.BORDER_DEFAULT};
            }}
        """)

        layout.addWidget(toolbar)

        # Prompt display
        self.prompt_text = QTextEdit()
        self.prompt_text.setReadOnly(True)
        self.prompt_text.setFont(StyleManager.get_monospace_font(Typography.FONT_SIZE_SMALL))
        self.prompt_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.prompt_text)

    def set_prompt(self, prompt: str):
        """Set the structured prompt text"""
        self.prompt_text.setPlainText(prompt)

    def copy_prompt(self):
        """Copy prompt to clipboard"""
        from PySide6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(self.prompt_text.toPlainText())


class ResponseTextWidget(QWidget):
    """Main response text display with formatting"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QFrame()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(Spacing.SPACE_BASE, Spacing.SPACE_SM,
                                         Spacing.SPACE_BASE, Spacing.SPACE_SM)

        toolbar_title = QLabel("LLM Response")
        toolbar_title.setProperty("heading", "h3")
        toolbar_layout.addWidget(toolbar_title)

        toolbar_layout.addStretch()

        # Metadata
        self.metadata_label = QLabel("")
        self.metadata_label.setProperty("secondary", True)
        StyleManager.apply_widget_property(self.metadata_label, "secondary", True)
        toolbar_layout.addWidget(self.metadata_label)

        copy_btn = QPushButton("📋 Copy")
        copy_btn.clicked.connect(self.copy_response)
        toolbar_layout.addWidget(copy_btn)

        toolbar.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_SECONDARY};
                border-bottom: 1px solid {Colors.BORDER_DEFAULT};
            }}
        """)

        layout.addWidget(toolbar)

        # Response text
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.response_text.setFont(QFont(Typography.FONT_FAMILY_SANS.split(",")[0].strip("'"),
                                        Typography.FONT_SIZE_BASE))
        layout.addWidget(self.response_text)

    def set_response(self, response: str, metadata: Dict[str, Any] = None):
        """Set response text and metadata"""
        self.response_text.setPlainText(response)

        if metadata:
            tokens = metadata.get("tokens", "?")
            time_ms = metadata.get("time_ms", "?")
            self.metadata_label.setText(f"Tokens: {tokens} • Time: {time_ms}ms")

    def copy_response(self):
        """Copy response to clipboard"""
        from PySide6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(self.response_text.toPlainText())


class ModuleTracePanel(QWidget):
    """Panel showing all triggered modules organized by tier"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.SPACE_BASE, Spacing.SPACE_BASE,
                                 Spacing.SPACE_BASE, Spacing.SPACE_BASE)
        layout.setSpacing(Spacing.SPACE_BASE)

        # Title
        title = QLabel("Module Trace")
        title.setProperty("heading", "h3")
        layout.addWidget(title)

        # Scroll area for modules
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        self.modules_layout = QVBoxLayout(scroll_content)
        self.modules_layout.setSpacing(Spacing.SPACE_SM)
        self.modules_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

    def set_modules(self, modules: List[Dict[str, Any]]):
        """
        Set modules to display

        Args:
            modules: List of dicts with keys: name, tier, status
        """
        # Clear existing
        while self.modules_layout.count() > 1:  # Keep the stretch
            item = self.modules_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Group by tier
        by_tier = {}
        for module in modules:
            tier = module.get("tier", 0)
            if tier not in by_tier:
                by_tier[tier] = []
            by_tier[tier].append(module)

        # Add modules by tier (0 first, then 1, 2, 3, 4)
        for tier in sorted(by_tier.keys()):
            # Tier header
            tier_names = {
                0: "Commands",
                1: "Safety & Integrity",
                2: "Core Analysis",
                3: "Heuristics & Context",
                4: "Style & Formatting"
            }
            tier_header = QLabel(f"Tier {tier}: {tier_names.get(tier, 'Unknown')}")
            tier_header.setProperty("heading", "h4")
            tier_header.setStyleSheet(f"color: {StyleManager.get_tier_color(tier)};")
            self.modules_layout.insertWidget(self.modules_layout.count() - 1, tier_header)

            # Add modules for this tier
            for module in by_tier[tier]:
                item = ModuleTraceItem(
                    module_name=module.get("name", "Unknown"),
                    tier=tier,
                    status=module.get("status", "active")
                )
                self.modules_layout.insertWidget(self.modules_layout.count() - 1, item)


class ResponseDisplayWidget(QWidget):
    """
    Complete response display interface with tabs

    Tabs:
    1. Response - The LLM's generated response
    2. Modules - Module trace showing triggered modules
    3. Prompt - Structured prompt sent to LLM
    4. Log - Arbitration log and debug info
    """

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Setup user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # Response tab
        self.response_widget = ResponseTextWidget()
        self.tabs.addTab(self.response_widget, "📝 Response")

        # Module trace tab
        self.module_trace = ModuleTracePanel()
        self.tabs.addTab(self.module_trace, "🔍 Modules")

        # Structured prompt tab
        self.prompt_viewer = StructuredPromptViewer()
        self.tabs.addTab(self.prompt_viewer, "📋 Prompt")

        # Arbitration log tab
        self.arbitration_log = ArbitrationLogWidget()
        self.tabs.addTab(self.arbitration_log, "⚖️ Log")

        layout.addWidget(self.tabs)

        # Show placeholder by default
        self.show_placeholder()

    def show_placeholder(self):
        """Show placeholder when no response"""
        self.response_widget.response_text.setPlainText(
            "No response yet.\n\n"
            "Enter a prompt above and click 'Analyze Prompt' to begin."
        )
        self.response_widget.metadata_label.setText("")

    def display_response(self, response_data: Dict[str, Any]):
        """
        Display a complete response with all metadata

        Args:
            response_data: Dict containing:
                - response: The LLM response text
                - modules: List of triggered modules
                - structured_prompt: The assembled prompt
                - arbitration_log: List of log entries
                - metadata: Dict with tokens, time, etc.
        """
        # Response text
        self.response_widget.set_response(
            response_data.get("response", ""),
            response_data.get("metadata", {})
        )

        # Module trace
        self.module_trace.set_modules(response_data.get("modules", []))

        # Structured prompt
        self.prompt_viewer.set_prompt(response_data.get("structured_prompt", ""))

        # Arbitration log
        self.arbitration_log.set_log(response_data.get("arbitration_log", []))

        # Switch to response tab
        self.tabs.setCurrentIndex(0)
