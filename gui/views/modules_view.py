"""
Modules View - Module Library Browser and Manager

Features:
- Browse all available modules organized by tier
- Search and filter modules
- View module details (triggers, prompts, dependencies)
- Enable/disable modules
- Module statistics
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QLineEdit, QComboBox, QPushButton,
    QScrollArea, QSplitter, QTextEdit, QGroupBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from typing import List, Dict, Any
import yaml
from pathlib import Path

from gui import Colors, Spacing, Sizing, Typography
from gui.style_manager import StyleManager


class ModuleCard(QFrame):
    """
    Single module card showing module info

    Signals:
        clicked: Emitted when card is clicked (module_name: str)
    """

    clicked = Signal(str)

    def __init__(self, module_data: Dict[str, Any]):
        super().__init__()
        self.module_data = module_data
        self.module_name = module_data.get("name", "Unknown")
        self.tier = module_data.get("tier", 0)
        self.is_selected = False
        self.setup_ui()

    def setup_ui(self):
        """Setup UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(Spacing.SPACE_SM, Spacing.SPACE_SM,
                                 Spacing.SPACE_SM, Spacing.SPACE_SM)
        layout.setSpacing(Spacing.SPACE_SM)

        # Tier indicator
        tier_color = StyleManager.get_tier_color(self.tier)
        tier_bar = QFrame()
        tier_bar.setFixedWidth(4)
        tier_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {tier_color};
                border-radius: 2px;
            }}
        """)
        layout.addWidget(tier_bar)

        # Module info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        # Module name
        name_label = QLabel(self.module_name)
        name_label.setFont(QFont(Typography.FONT_FAMILY_SANS.split(",")[0].strip("'"),
                                Typography.FONT_SIZE_MEDIUM, QFont.Weight.Bold))
        info_layout.addWidget(name_label)

        # Tier and trigger count
        tier_names = {
            0: "Commands",
            1: "Safety & Integrity",
            2: "Core Analysis",
            3: "Heuristics & Context",
            4: "Style & Formatting"
        }
        trigger_count = len(self.module_data.get("triggers", []))
        meta_text = f"Tier {self.tier}: {tier_names.get(self.tier, 'Unknown')} • {trigger_count} triggers"

        meta_label = QLabel(meta_text)
        meta_label.setProperty("secondary", True)
        StyleManager.apply_widget_property(meta_label, "secondary", True)
        meta_label.setStyleSheet(f"font-size: {Typography.FONT_SIZE_SMALL}pt;")
        info_layout.addWidget(meta_label)

        # Description (first line only)
        description = self.module_data.get("description", "No description available")
        if isinstance(description, list):
            description = " ".join(description)
        # Truncate to first sentence or 100 chars
        if "." in description:
            description = description.split(".")[0] + "."
        elif len(description) > 100:
            description = description[:97] + "..."

        desc_label = QLabel(description)
        desc_label.setProperty("secondary", True)
        StyleManager.apply_widget_property(desc_label, "secondary", True)
        desc_label.setWordWrap(True)
        info_layout.addWidget(desc_label)

        layout.addLayout(info_layout, stretch=1)

        # Status indicator
        status = self.module_data.get("enabled", True)
        status_label = QLabel("✓ Enabled" if status else "○ Disabled")
        status_label.setStyleSheet(f"""
            color: {Colors.ACCENT_SUCCESS if status else Colors.TEXT_TERTIARY};
            font-size: {Typography.FONT_SIZE_SMALL}pt;
        """)
        layout.addWidget(status_label)

        # Style the card
        self.update_style()

        # Make clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def update_style(self):
        """Update card styling based on selection state"""
        if self.is_selected:
            border_color = StyleManager.get_tier_color(self.tier)
            bg_color = Colors.STATE_SELECTED
        else:
            border_color = Colors.BORDER_DEFAULT
            bg_color = Colors.BG_SECONDARY

        self.setStyleSheet(f"""
            ModuleCard {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: {Sizing.RADIUS_SM}px;
            }}
            ModuleCard:hover {{
                background-color: {Colors.STATE_HOVER};
                border-color: {StyleManager.get_tier_color(self.tier)};
            }}
        """)

    def mousePressEvent(self, event):
        """Handle mouse click"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.module_name)

    def set_selected(self, selected: bool):
        """Set selection state"""
        self.is_selected = selected
        self.update_style()


class ModuleDetailPanel(QWidget):
    """Panel showing detailed module information"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.SPACE_BASE, Spacing.SPACE_BASE,
                                 Spacing.SPACE_BASE, Spacing.SPACE_BASE)
        layout.setSpacing(Spacing.SPACE_BASE)

        # Header
        header_layout = QHBoxLayout()

        self.title_label = QLabel("Select a module")
        self.title_label.setProperty("heading", "h2")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        # Action buttons
        self.enable_btn = QPushButton("Enable")
        self.enable_btn.setVisible(False)
        header_layout.addWidget(self.enable_btn)

        self.disable_btn = QPushButton("Disable")
        self.disable_btn.setVisible(False)
        header_layout.addWidget(self.disable_btn)

        layout.addLayout(header_layout)

        # Metadata
        self.meta_label = QLabel("")
        self.meta_label.setProperty("secondary", True)
        StyleManager.apply_widget_property(self.meta_label, "secondary", True)
        layout.addWidget(self.meta_label)

        # Description
        desc_group = QGroupBox("Description")
        desc_layout = QVBoxLayout(desc_group)
        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_text.setMaximumHeight(150)
        desc_layout.addWidget(self.description_text)
        layout.addWidget(desc_group)

        # Triggers
        triggers_group = QGroupBox("Triggers")
        triggers_layout = QVBoxLayout(triggers_group)
        self.triggers_text = QTextEdit()
        self.triggers_text.setReadOnly(True)
        self.triggers_text.setFont(StyleManager.get_monospace_font(Typography.FONT_SIZE_SMALL))
        self.triggers_text.setMaximumHeight(150)
        triggers_layout.addWidget(self.triggers_text)
        layout.addWidget(triggers_group)

        # Structured prompts
        prompts_group = QGroupBox("Structured Prompts")
        prompts_layout = QVBoxLayout(prompts_group)
        self.prompts_text = QTextEdit()
        self.prompts_text.setReadOnly(True)
        self.prompts_text.setFont(StyleManager.get_monospace_font(Typography.FONT_SIZE_SMALL))
        prompts_layout.addWidget(self.prompts_text)
        layout.addWidget(prompts_group)

        layout.addStretch()

    def show_module(self, module_data: Dict[str, Any]):
        """Display module details"""
        # Title
        name = module_data.get("name", "Unknown")
        self.title_label.setText(name)

        # Metadata
        tier = module_data.get("tier", 0)
        version = module_data.get("version", "unknown")
        tier_names = {
            0: "Commands",
            1: "Safety & Integrity",
            2: "Core Analysis",
            3: "Heuristics & Context",
            4: "Style & Formatting"
        }
        tier_name = tier_names.get(tier, "Unknown")
        tier_color = StyleManager.get_tier_color(tier)

        self.meta_label.setText(f"Tier {tier}: {tier_name} • Version {version}")
        self.meta_label.setStyleSheet(f"color: {tier_color}; font-weight: bold;")

        # Description
        description = module_data.get("description", "No description available")
        if isinstance(description, list):
            description = "\n".join(description)
        self.description_text.setPlainText(description)

        # Triggers
        triggers = module_data.get("triggers", [])
        if triggers:
            triggers_text = "\n".join([f"• {trigger}" for trigger in triggers])
        else:
            triggers_text = "No triggers defined"
        self.triggers_text.setPlainText(triggers_text)

        # Structured prompts
        prompts = module_data.get("structured_prompts", [])
        if prompts:
            prompts_text = "\n\n".join([f"[Prompt {i+1}]\n{prompt}" for i, prompt in enumerate(prompts)])
        else:
            prompts_text = "No structured prompts defined"
        self.prompts_text.setPlainText(prompts_text)

        # Buttons
        enabled = module_data.get("enabled", True)
        self.enable_btn.setVisible(not enabled)
        self.disable_btn.setVisible(enabled)

    def clear(self):
        """Clear the detail panel"""
        self.title_label.setText("Select a module")
        self.meta_label.setText("")
        self.description_text.clear()
        self.triggers_text.clear()
        self.prompts_text.clear()
        self.enable_btn.setVisible(False)
        self.disable_btn.setVisible(False)


class ModulesView(QWidget):
    """
    Complete module library browser interface

    Features:
    - Browse modules by tier
    - Search and filter
    - View module details
    - Enable/disable modules
    """

    def __init__(self):
        super().__init__()
        self.modules = []
        self.current_filter_tier = -1  # -1 = all tiers
        self.selected_module = None
        self.module_cards = {}
        self.setup_ui()
        self.load_modules()

    def setup_ui(self):
        """Setup user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = self.create_toolbar()
        layout.addWidget(toolbar)

        # Main content with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Module list
        list_panel = self.create_list_panel()
        splitter.addWidget(list_panel)

        # Right: Module details
        self.detail_panel = ModuleDetailPanel()
        splitter.addWidget(self.detail_panel)

        # Set initial sizes (40% list, 60% detail)
        splitter.setSizes([400, 600])

        layout.addWidget(splitter)

    def create_toolbar(self) -> QFrame:
        """Create toolbar with search and filters"""
        toolbar = QFrame()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(Spacing.SPACE_BASE, Spacing.SPACE_SM,
                                         Spacing.SPACE_BASE, Spacing.SPACE_SM)
        toolbar_layout.setSpacing(Spacing.SPACE_BASE)

        # Title
        title = QLabel("📚 Module Library")
        title.setProperty("heading", "h2")
        toolbar_layout.addWidget(title)

        toolbar_layout.addStretch()

        # Search box
        search_label = QLabel("Search:")
        toolbar_layout.addWidget(search_label)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Filter modules...")
        self.search_box.setMinimumWidth(200)
        self.search_box.textChanged.connect(self.on_search_changed)
        toolbar_layout.addWidget(self.search_box)

        # Tier filter
        tier_label = QLabel("Tier:")
        toolbar_layout.addWidget(tier_label)

        self.tier_filter = QComboBox()
        self.tier_filter.addItems([
            "All Tiers",
            "Tier 0: Commands",
            "Tier 1: Safety",
            "Tier 2: Analysis",
            "Tier 3: Heuristics",
            "Tier 4: Style"
        ])
        self.tier_filter.currentIndexChanged.connect(self.on_tier_filter_changed)
        toolbar_layout.addWidget(self.tier_filter)

        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.load_modules)
        toolbar_layout.addWidget(refresh_btn)

        toolbar.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_SECONDARY};
                border-bottom: 1px solid {Colors.BORDER_DEFAULT};
            }}
        """)

        return toolbar

    def create_list_panel(self) -> QWidget:
        """Create module list panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(Spacing.SPACE_BASE, Spacing.SPACE_BASE,
                                 Spacing.SPACE_BASE, Spacing.SPACE_BASE)
        layout.setSpacing(Spacing.SPACE_SM)

        # Stats
        self.stats_label = QLabel("Loading modules...")
        self.stats_label.setProperty("secondary", True)
        StyleManager.apply_widget_property(self.stats_label, "secondary", True)
        layout.addWidget(self.stats_label)

        # Scroll area for module cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        scroll_content = QWidget()
        self.modules_layout = QVBoxLayout(scroll_content)
        self.modules_layout.setSpacing(Spacing.SPACE_SM)
        self.modules_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        return panel

    def load_modules(self):
        """Load all modules from the modules directory"""
        modules_dir = Path("modules")
        self.modules = []
        self.module_cards = {}

        if not modules_dir.exists():
            self.stats_label.setText("⚠️ No modules directory found")
            return

        # Load all YAML files from modules directory
        for yaml_file in modules_dir.rglob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    module_data = yaml.safe_load(f)
                    if module_data:
                        # Add enabled status (default True)
                        module_data["enabled"] = module_data.get("enabled", True)
                        module_data["file_path"] = str(yaml_file)
                        self.modules.append(module_data)
            except Exception as e:
                print(f"Error loading {yaml_file}: {e}")

        # Sort by tier, then by name
        self.modules.sort(key=lambda m: (m.get("tier", 0), m.get("name", "")))

        # Update display
        self.update_stats()
        self.display_modules()

    def update_stats(self):
        """Update module statistics"""
        total = len(self.modules)
        by_tier = {}
        for module in self.modules:
            tier = module.get("tier", 0)
            by_tier[tier] = by_tier.get(tier, 0) + 1

        stats_text = f"Total: {total} modules"
        if by_tier:
            tier_counts = ", ".join([f"T{tier}: {count}" for tier, count in sorted(by_tier.items())])
            stats_text += f" ({tier_counts})"

        self.stats_label.setText(stats_text)

    def display_modules(self):
        """Display filtered modules"""
        # Clear existing cards
        while self.modules_layout.count() > 1:  # Keep the stretch
            item = self.modules_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.module_cards = {}

        # Filter modules
        filtered_modules = self.get_filtered_modules()

        if not filtered_modules:
            no_results = QLabel("No modules found")
            no_results.setProperty("secondary", True)
            StyleManager.apply_widget_property(no_results, "secondary", True)
            no_results.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.modules_layout.insertWidget(0, no_results)
            return

        # Create cards for filtered modules
        for module in filtered_modules:
            card = ModuleCard(module)
            card.clicked.connect(self.on_module_clicked)
            self.module_cards[module.get("name", "")] = card
            self.modules_layout.insertWidget(self.modules_layout.count() - 1, card)

    def get_filtered_modules(self) -> List[Dict[str, Any]]:
        """Get modules matching current filters"""
        filtered = self.modules

        # Filter by tier
        if self.current_filter_tier >= 0:
            filtered = [m for m in filtered if m.get("tier", 0) == self.current_filter_tier]

        # Filter by search text
        search_text = self.search_box.text().lower()
        if search_text:
            filtered = [m for m in filtered
                       if search_text in m.get("name", "").lower()
                       or search_text in str(m.get("description", "")).lower()]

        return filtered

    def on_tier_filter_changed(self, index: int):
        """Handle tier filter change"""
        self.current_filter_tier = index - 1  # -1 for "All Tiers"
        self.display_modules()

    def on_search_changed(self, text: str):
        """Handle search text change"""
        self.display_modules()

    def on_module_clicked(self, module_name: str):
        """Handle module card click"""
        # Deselect all cards
        for card in self.module_cards.values():
            card.set_selected(False)

        # Select clicked card
        if module_name in self.module_cards:
            self.module_cards[module_name].set_selected(True)

        # Find module data
        module_data = next((m for m in self.modules if m.get("name") == module_name), None)
        if module_data:
            self.selected_module = module_data
            self.detail_panel.show_module(module_data)
