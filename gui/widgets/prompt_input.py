"""
Prompt Input Widget - Advanced text editor for user prompts

Features:
- Multi-line text editing
- Syntax highlighting for keywords
- Module trigger auto-completion
- Line numbers
- Character/word count
- Clear and example prompt functionality
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit,
    QPushButton, QLabel, QFrame, QCompleter, QToolBar,
    QTextEdit
)
from PySide6.QtCore import Qt, Signal, QRect, QSize, QStringListModel
from PySide6.QtGui import (
    QSyntaxHighlighter, QTextCharFormat, QColor, QFont,
    QPainter, QTextFormat, QPalette
)
import re

from gui import Colors, Spacing, Sizing, Typography
from gui.style_manager import StyleManager


class PromptSyntaxHighlighter(QSyntaxHighlighter):
    """
    Syntax highlighter for prompt text

    Highlights:
    - Module trigger keywords
    - Quoted strings
    - Questions
    - Special characters
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_formats()

    def setup_formats(self):
        """Setup text formats for different syntax elements"""

        # Keyword format (module triggers)
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(QColor(Colors.SYNTAX_KEYWORD))
        self.keyword_format.setFontWeight(QFont.Weight.Bold)

        # String format (quoted text)
        self.string_format = QTextCharFormat()
        self.string_format.setForeground(QColor(Colors.SYNTAX_STRING))

        # Question format
        self.question_format = QTextCharFormat()
        self.question_format.setForeground(QColor(Colors.ACCENT_PRIMARY))
        self.question_format.setFontWeight(QFont.Weight.Bold)

        # Number format
        self.number_format = QTextCharFormat()
        self.number_format.setForeground(QColor(Colors.SYNTAX_NUMBER))

        # Special character format
        self.special_format = QTextCharFormat()
        self.special_format.setForeground(QColor(Colors.ACCENT_SECONDARY))

        # Module trigger keywords (common analytical terms)
        self.keywords = [
            'analyze', 'audit', 'detect', 'examine', 'evaluate', 'assess',
            'grift', 'cult', 'propaganda', 'coercion', 'manipulation',
            'ethics', 'consent', 'power', 'structure', 'bias', 'symmetry',
            'decentralization', 'surveillance', 'exploitation', 'trajectory',
            'displacement', 'laundering', 'shield', 'tyranny', 'capital'
        ]

    def highlightBlock(self, text):
        """Apply syntax highlighting to a block of text"""

        # Highlight keywords
        for keyword in self.keywords:
            pattern = r'\b' + keyword + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                self.setFormat(match.start(), match.end() - match.start(),
                             self.keyword_format)

        # Highlight quoted strings
        string_pattern = r'"[^"]*"|\'[^\']*\''
        for match in re.finditer(string_pattern, text):
            self.setFormat(match.start(), match.end() - match.start(),
                         self.string_format)

        # Highlight question words
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
        for qword in question_words:
            pattern = r'\b' + qword + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                self.setFormat(match.start(), match.end() - match.start(),
                             self.question_format)

        # Highlight numbers
        number_pattern = r'\b\d+\b'
        for match in re.finditer(number_pattern, text):
            self.setFormat(match.start(), match.end() - match.start(),
                         self.number_format)

        # Highlight question marks and exclamation marks
        special_pattern = r'[?!]'
        for match in re.finditer(special_pattern, text):
            self.setFormat(match.start(), match.end() - match.start(),
                         self.special_format)


class LineNumberArea(QWidget):
    """Line number display area for text editor"""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.editor.line_number_area_paint_event(event)


class PromptTextEdit(QPlainTextEdit):
    """Enhanced QPlainTextEdit with line numbers"""

    def __init__(self):
        super().__init__()

        # Line number area
        self.line_number_area = LineNumberArea(self)

        # Connect signals
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.cursorPositionChanged.connect(self.highlight_current_line)

        # Initial setup
        self.update_line_number_area_width(0)
        self.highlight_current_line()

        # Set monospace font
        self.setFont(StyleManager.get_monospace_font(Typography.FONT_SIZE_BASE))

        # Tab behavior
        self.setTabStopDistance(40)  # 4 spaces equivalent

    def line_number_area_width(self):
        """Calculate width needed for line numbers"""
        digits = len(str(max(1, self.blockCount())))
        space = 10 + self.fontMetrics().horizontalAdvance('9') * digits
        return space

    def update_line_number_area_width(self, _):
        """Update viewport margins for line numbers"""
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect, dy):
        """Update line number area on scroll"""
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(),
                                        self.line_number_area.width(),
                                        rect.height())

        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event):
        """Handle resize event"""
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(
            QRect(cr.left(), cr.top(),
                  self.line_number_area_width(), cr.height())
        )

    def highlight_current_line(self):
        """Highlight the line containing the cursor"""
        extra_selections = []

        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()

            line_color = QColor(Colors.STATE_HOVER)
            selection.format.setBackground(line_color)
            selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()

            extra_selections.append(selection)

        self.setExtraSelections(extra_selections)

    def line_number_area_paint_event(self, event):
        """Paint line numbers"""
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor(Colors.BG_SECONDARY))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(
            self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QColor(Colors.TEXT_TERTIARY))
                painter.drawText(0, top,
                               self.line_number_area.width() - 5,
                               self.fontMetrics().height(),
                               Qt.AlignmentFlag.AlignRight, number)

            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1


class PromptInputWidget(QWidget):
    """
    Complete prompt input interface

    Signals:
        prompt_submitted: Emitted when user submits prompt (text: str)
        text_changed: Emitted when text changes
    """

    prompt_submitted = Signal(str)
    text_changed = Signal()

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.load_module_triggers()

    def setup_ui(self):
        """Setup user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = self.create_toolbar()
        layout.addWidget(toolbar)

        # Text editor
        self.text_edit = PromptTextEdit()
        self.text_edit.setPlaceholderText(
            "Enter your prompt here...\n\n"
            "Examples:\n"
            "- Analyze this grift scheme...\n"
            "- What are the power structures in...\n"
            "- Audit the consent architecture of..."
        )

        # Apply syntax highlighting
        self.highlighter = PromptSyntaxHighlighter(self.text_edit.document())

        # Connect signals
        self.text_edit.textChanged.connect(self.on_text_changed)

        layout.addWidget(self.text_edit, stretch=1)

        # Stats bar
        stats_bar = self.create_stats_bar()
        layout.addWidget(stats_bar)

        # Action buttons
        button_bar = self.create_button_bar()
        layout.addWidget(button_bar)

    def create_toolbar(self) -> QToolBar:
        """Create toolbar with quick actions"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(16, 16))

        # Example prompts dropdown
        self.example_btn = QPushButton("📝 Examples")
        self.example_btn.clicked.connect(self.show_example_prompts)
        toolbar.addWidget(self.example_btn)

        toolbar.addSeparator()

        # Load prompt
        self.load_btn = QPushButton("📂 Load")
        self.load_btn.clicked.connect(self.load_prompt)
        toolbar.addWidget(self.load_btn)

        # Save prompt
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.clicked.connect(self.save_prompt)
        toolbar.addWidget(self.save_btn)

        toolbar.addSeparator()

        # Clear
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.clicked.connect(self.clear_text)
        toolbar.addWidget(self.clear_btn)

        return toolbar

    def create_stats_bar(self) -> QFrame:
        """Create statistics display bar"""
        stats_frame = QFrame()
        stats_frame.setFrameShape(QFrame.Shape.NoFrame)
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(Spacing.SPACE_SM, Spacing.SPACE_XS,
                                       Spacing.SPACE_SM, Spacing.SPACE_XS)

        # Character count
        self.char_count_label = QLabel("Characters: 0")
        self.char_count_label.setProperty("secondary", True)
        StyleManager.apply_widget_property(self.char_count_label, "secondary", True)
        stats_layout.addWidget(self.char_count_label)

        stats_layout.addSpacing(Spacing.SPACE_BASE)

        # Word count
        self.word_count_label = QLabel("Words: 0")
        self.word_count_label.setProperty("secondary", True)
        StyleManager.apply_widget_property(self.word_count_label, "secondary", True)
        stats_layout.addWidget(self.word_count_label)

        stats_layout.addSpacing(Spacing.SPACE_BASE)

        # Line count
        self.line_count_label = QLabel("Lines: 1")
        self.line_count_label.setProperty("secondary", True)
        StyleManager.apply_widget_property(self.line_count_label, "secondary", True)
        stats_layout.addWidget(self.line_count_label)

        stats_layout.addStretch()

        # Cursor position
        self.cursor_pos_label = QLabel("Ln 1, Col 1")
        self.cursor_pos_label.setProperty("secondary", True)
        StyleManager.apply_widget_property(self.cursor_pos_label, "secondary", True)
        stats_layout.addWidget(self.cursor_pos_label)

        # Connect cursor position updates
        self.text_edit.cursorPositionChanged.connect(self.update_cursor_position)

        stats_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_SECONDARY};
                border-top: 1px solid {Colors.BORDER_DEFAULT};
            }}
        """)

        return stats_frame

    def create_button_bar(self) -> QFrame:
        """Create action button bar"""
        button_frame = QFrame()
        button_frame.setFrameShape(QFrame.Shape.NoFrame)
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(Spacing.SPACE_BASE, Spacing.SPACE_SM,
                                        Spacing.SPACE_BASE, Spacing.SPACE_SM)

        button_layout.addStretch()

        # Submit button
        self.submit_btn = QPushButton("🚀 Analyze Prompt")
        self.submit_btn.setProperty("primary", True)
        StyleManager.apply_widget_property(self.submit_btn, "primary", True)
        self.submit_btn.setMinimumWidth(150)
        self.submit_btn.clicked.connect(self.submit_prompt)
        self.submit_btn.setShortcut("Ctrl+Return")
        button_layout.addWidget(self.submit_btn)

        button_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_SECONDARY};
                border-top: 1px solid {Colors.BORDER_DEFAULT};
            }}
        """)

        return button_frame

    def load_module_triggers(self):
        """Load module trigger keywords for auto-completion"""
        # This will be populated from actual modules later
        self.trigger_keywords = [
            'analyze', 'audit', 'detect', 'examine', 'evaluate',
            'grift', 'cult', 'propaganda', 'coercion', 'consent',
            'power', 'structure', 'bias', 'symmetry', 'ethics'
        ]

        # Note: QPlainTextEdit doesn't support QCompleter natively
        # Auto-completion is handled through syntax highlighting instead
        # Keywords are visually highlighted to guide user

    def on_text_changed(self):
        """Handle text change event"""
        text = self.text_edit.toPlainText()

        # Update character count
        char_count = len(text)
        self.char_count_label.setText(f"Characters: {char_count}")

        # Update word count
        word_count = len(text.split()) if text.strip() else 0
        self.word_count_label.setText(f"Words: {word_count}")

        # Update line count
        line_count = text.count('\n') + 1 if text else 1
        self.line_count_label.setText(f"Lines: {line_count}")

        # Enable/disable submit button
        self.submit_btn.setEnabled(bool(text.strip()))

        # Emit signal
        self.text_changed.emit()

    def update_cursor_position(self):
        """Update cursor position display"""
        cursor = self.text_edit.textCursor()
        line = cursor.blockNumber() + 1
        col = cursor.columnNumber() + 1
        self.cursor_pos_label.setText(f"Ln {line}, Col {col}")

    def get_text(self) -> str:
        """Get current prompt text"""
        return self.text_edit.toPlainText()

    def set_text(self, text: str):
        """Set prompt text"""
        self.text_edit.setPlainText(text)

    def clear_text(self):
        """Clear all text"""
        self.text_edit.clear()

    def submit_prompt(self):
        """Submit prompt for analysis"""
        text = self.get_text().strip()
        if text:
            self.prompt_submitted.emit(text)

    def show_example_prompts(self):
        """Show example prompts menu"""
        # Placeholder - will be implemented with actual menu
        examples = [
            "Analyze this organization's fundraising tactics for grift patterns.",
            "What power structures exist in this decentralized platform?",
            "Audit the consent architecture of this terms of service agreement.",
            "Detect propaganda techniques in this political messaging.",
            "Examine the cult dynamics in this online community."
        ]
        # For now, just set the first example
        self.set_text(examples[0])

    def load_prompt(self):
        """Load prompt from file"""
        # Placeholder for file dialog
        print("Load prompt dialog (to be implemented)")

    def save_prompt(self):
        """Save prompt to file"""
        # Placeholder for file dialog
        print("Save prompt dialog (to be implemented)")
