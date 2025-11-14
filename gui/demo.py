"""
Protocol AI - Design System Demo

Showcases the dark mode design system with all components and styling options.
Run this to preview the GUI design system.
"""

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox,
    QCheckBox, QRadioButton, QSlider, QProgressBar, QGroupBox,
    QTabWidget, QListWidget, QTreeWidget, QTreeWidgetItem,
    QTableWidget, QTableWidgetItem, QSpinBox, QFrame, QSplitter,
    QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from gui import create_style_manager, Colors, Typography, Spacing, Sizing
from gui.style_manager import StyleManager


class DesignSystemDemo(QMainWindow):
    """Demo window showcasing all design system components"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Protocol AI - Design System Demo")
        self.setGeometry(100, 100, 1400, 900)

        # Create main widget with scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(Spacing.SPACE_LG)
        main_layout.setContentsMargins(Spacing.SPACE_LG, Spacing.SPACE_LG,
                                      Spacing.SPACE_LG, Spacing.SPACE_LG)

        # Add sections
        main_layout.addWidget(self.create_header())
        main_layout.addWidget(self.create_buttons_section())
        main_layout.addWidget(self.create_inputs_section())
        main_layout.addWidget(self.create_selection_section())
        main_layout.addWidget(self.create_data_display_section())
        main_layout.addWidget(self.create_tier_visualization())
        main_layout.addWidget(self.create_typography_section())
        main_layout.addStretch()

        scroll.setWidget(main_widget)
        self.setCentralWidget(scroll)

    def create_header(self) -> QWidget:
        """Create header section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        title = QLabel("Protocol AI Design System")
        title.setProperty("heading", "h1")
        layout.addWidget(title)

        subtitle = QLabel("Dark Mode UI Components & Styling")
        subtitle.setProperty("heading", "h2")
        subtitle.setProperty("secondary", True)
        layout.addWidget(subtitle)

        description = QLabel(
            "A comprehensive dark mode design system built with PySide6. "
            "All components follow the design tokens defined in design_tokens.py."
        )
        layout.addWidget(description)

        return widget

    def create_buttons_section(self) -> QGroupBox:
        """Create buttons showcase"""
        group = QGroupBox("Buttons")
        layout = QHBoxLayout(group)

        # Default button
        btn_default = QPushButton("Default Button")
        layout.addWidget(btn_default)

        # Primary button
        btn_primary = QPushButton("Primary Button")
        btn_primary.setProperty("primary", True)
        StyleManager.apply_widget_property(btn_primary, "primary", True)
        layout.addWidget(btn_primary)

        # Danger button
        btn_danger = QPushButton("Danger Button")
        btn_danger.setProperty("danger", True)
        StyleManager.apply_widget_property(btn_danger, "danger", True)
        layout.addWidget(btn_danger)

        # Disabled button
        btn_disabled = QPushButton("Disabled Button")
        btn_disabled.setEnabled(False)
        layout.addWidget(btn_disabled)

        layout.addStretch()
        return group

    def create_inputs_section(self) -> QGroupBox:
        """Create input fields showcase"""
        group = QGroupBox("Input Fields")
        layout = QVBoxLayout(group)

        # Line edit
        line_edit = QLineEdit()
        line_edit.setPlaceholderText("Enter text here...")
        layout.addWidget(QLabel("Line Edit:"))
        layout.addWidget(line_edit)

        # Text edit
        text_edit = QTextEdit()
        text_edit.setPlaceholderText("Enter multiple lines here...")
        text_edit.setMaximumHeight(100)
        layout.addWidget(QLabel("Text Edit:"))
        layout.addWidget(text_edit)

        # Combo box
        combo = QComboBox()
        combo.addItems(["Option 1", "Option 2", "Option 3"])
        layout.addWidget(QLabel("Combo Box:"))
        layout.addWidget(combo)

        # Spin box
        spin = QSpinBox()
        spin.setRange(0, 100)
        spin.setValue(50)
        layout.addWidget(QLabel("Spin Box:"))
        layout.addWidget(spin)

        return group

    def create_selection_section(self) -> QGroupBox:
        """Create selection controls showcase"""
        group = QGroupBox("Selection Controls")
        layout = QVBoxLayout(group)

        # Checkboxes
        cb_layout = QHBoxLayout()
        cb1 = QCheckBox("Checkbox 1")
        cb2 = QCheckBox("Checkbox 2 (Checked)")
        cb2.setChecked(True)
        cb3 = QCheckBox("Checkbox 3 (Disabled)")
        cb3.setEnabled(False)
        cb_layout.addWidget(cb1)
        cb_layout.addWidget(cb2)
        cb_layout.addWidget(cb3)
        cb_layout.addStretch()
        layout.addLayout(cb_layout)

        # Radio buttons
        rb_layout = QHBoxLayout()
        rb1 = QRadioButton("Radio 1")
        rb2 = QRadioButton("Radio 2 (Selected)")
        rb2.setChecked(True)
        rb3 = QRadioButton("Radio 3")
        rb_layout.addWidget(rb1)
        rb_layout.addWidget(rb2)
        rb_layout.addWidget(rb3)
        rb_layout.addStretch()
        layout.addLayout(rb_layout)

        # Slider
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(QLabel("Slider:"))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        slider_layout.addWidget(slider)
        layout.addLayout(slider_layout)

        # Progress bar
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(QLabel("Progress Bar:"))
        progress = QProgressBar()
        progress.setValue(65)
        progress_layout.addWidget(progress)
        layout.addLayout(progress_layout)

        return group

    def create_data_display_section(self) -> QGroupBox:
        """Create data display widgets showcase"""
        group = QGroupBox("Data Display")
        layout = QHBoxLayout(group)

        # List widget
        list_widget = QListWidget()
        list_widget.addItems([f"List Item {i}" for i in range(1, 6)])
        list_widget.setMaximumWidth(200)
        layout.addWidget(list_widget)

        # Tree widget
        tree_widget = QTreeWidget()
        tree_widget.setHeaderLabel("Tree View")
        tree_widget.setMaximumWidth(250)

        # Add tree items
        for i in range(3):
            parent = QTreeWidgetItem(tree_widget, [f"Parent {i+1}"])
            for j in range(3):
                QTreeWidgetItem(parent, [f"Child {i+1}.{j+1}"])
        tree_widget.expandAll()
        layout.addWidget(tree_widget)

        # Table widget
        table = QTableWidget(4, 3)
        table.setHorizontalHeaderLabels(["Column 1", "Column 2", "Column 3"])
        for row in range(4):
            for col in range(3):
                table.setItem(row, col, QTableWidgetItem(f"Cell {row},{col}"))
        layout.addWidget(table)

        return group

    def create_tier_visualization(self) -> QGroupBox:
        """Create tier color visualization"""
        group = QGroupBox("Module Tier Colors")
        layout = QVBoxLayout(group)

        tiers = [
            (0, "Tier 0: Commands", Colors.TIER_0),
            (1, "Tier 1: Safety & Integrity", Colors.TIER_1),
            (2, "Tier 2: Core Analysis", Colors.TIER_2),
            (3, "Tier 3: Heuristics & Context", Colors.TIER_3),
            (4, "Tier 4: Style & Formatting", Colors.TIER_4),
        ]

        for tier_num, tier_name, tier_color in tiers:
            tier_widget = QFrame()
            tier_widget.setProperty("tier", tier_num)
            tier_widget.setFrameShape(QFrame.Shape.StyledPanel)
            StyleManager.apply_widget_property(tier_widget, "tier", tier_num)

            tier_layout = QHBoxLayout(tier_widget)
            label = QLabel(tier_name)
            label.setStyleSheet(f"border: none; color: {tier_color}; font-weight: 600;")
            tier_layout.addWidget(label)

            layout.addWidget(tier_widget)

        return group

    def create_typography_section(self) -> QGroupBox:
        """Create typography showcase"""
        group = QGroupBox("Typography")
        layout = QVBoxLayout(group)

        # Headings
        h1 = QLabel("Heading 1")
        h1.setProperty("heading", "h1")
        layout.addWidget(h1)

        h2 = QLabel("Heading 2")
        h2.setProperty("heading", "h2")
        layout.addWidget(h2)

        h3 = QLabel("Heading 3")
        h3.setProperty("heading", "h3")
        layout.addWidget(h3)

        # Text variants
        primary = QLabel("Primary text - regular body content")
        layout.addWidget(primary)

        secondary = QLabel("Secondary text - less emphasis")
        secondary.setProperty("secondary", True)
        StyleManager.apply_widget_property(secondary, "secondary", True)
        layout.addWidget(secondary)

        tertiary = QLabel("Tertiary text - minimal emphasis or disabled")
        tertiary.setProperty("tertiary", True)
        StyleManager.apply_widget_property(tertiary, "tertiary", True)
        layout.addWidget(tertiary)

        # Monospace/code
        code = QTextEdit()
        code.setProperty("monospace", True)
        code.setPlainText("def hello_world():\n    print('Hello from Protocol AI!')\n    return True")
        code.setMaximumHeight(80)
        code.setFont(StyleManager.get_monospace_font())
        layout.addWidget(QLabel("Monospace (Code):"))
        layout.addWidget(code)

        return group


def main():
    """Run design system demo"""
    app = QApplication(sys.argv)

    # Initialize style manager
    style_manager = create_style_manager(app)

    # Create and show demo window
    demo = DesignSystemDemo()
    demo.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
