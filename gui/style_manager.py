"""
Style Manager - Central theme and styling system for Protocol AI GUI

Manages loading and applying QSS styles, theme switching, and dynamic
style updates.
"""

from pathlib import Path
from typing import Optional
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QPalette, QColor
from PySide6.QtCore import QObject, Signal

from gui.design_tokens import Colors, Typography


class StyleManager(QObject):
    """
    Central manager for application styling and theming

    Responsibilities:
    - Load and apply QSS stylesheet
    - Manage theme switching (future: light mode)
    - Provide style utilities
    - Handle dynamic style updates
    """

    style_changed = Signal()

    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app
        self.current_theme = "dark"
        self._stylesheet_cache: Optional[str] = None

    def load_stylesheet(self, theme: str = "dark") -> str:
        """
        Load QSS stylesheet from file

        Args:
            theme: Theme name ("dark" for now, "light" in future)

        Returns:
            Stylesheet content as string
        """
        # For now, only dark theme exists
        stylesheet_path = Path(__file__).parent / "styles.qss"

        try:
            with open(stylesheet_path, 'r', encoding='utf-8') as f:
                stylesheet = f.read()
            self._stylesheet_cache = stylesheet
            return stylesheet
        except FileNotFoundError:
            print(f"Warning: Stylesheet not found at {stylesheet_path}")
            return ""

    def apply_stylesheet(self, theme: str = "dark"):
        """
        Load and apply stylesheet to application

        Args:
            theme: Theme name to apply
        """
        stylesheet = self.load_stylesheet(theme)
        if stylesheet:
            self.app.setStyleSheet(stylesheet)
            self.current_theme = theme
            self.style_changed.emit()

    def set_global_font(self):
        """Set application-wide default font"""
        font = QFont(
            "Inter, Segoe UI, system-ui",
            Typography.FONT_SIZE_BASE,
            Typography.FONT_WEIGHT_NORMAL
        )
        font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
        self.app.setFont(font)

    def set_dark_palette(self):
        """
        Set QPalette for dark theme

        This provides fallback colors for widgets that don't use stylesheets
        """
        palette = QPalette()

        # Window (background)
        palette.setColor(QPalette.ColorRole.Window, QColor(Colors.BG_PRIMARY))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(Colors.TEXT_PRIMARY))

        # Base (input backgrounds)
        palette.setColor(QPalette.ColorRole.Base, QColor(Colors.BG_PRIMARY))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(Colors.BG_SECONDARY))

        # Text
        palette.setColor(QPalette.ColorRole.Text, QColor(Colors.TEXT_PRIMARY))
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(Colors.TEXT_TERTIARY))

        # Buttons
        palette.setColor(QPalette.ColorRole.Button, QColor(Colors.BG_TERTIARY))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(Colors.TEXT_PRIMARY))

        # Highlights
        palette.setColor(QPalette.ColorRole.Highlight, QColor(Colors.STATE_SELECTED))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))

        # Links
        palette.setColor(QPalette.ColorRole.Link, QColor(Colors.TEXT_LINK))
        palette.setColor(QPalette.ColorRole.LinkVisited, QColor(Colors.ACCENT_SECONDARY))

        # Tooltips
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(Colors.BG_OVERLAY))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(Colors.TEXT_PRIMARY))

        # Disabled state
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText,
                        QColor(Colors.TEXT_TERTIARY))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,
                        QColor(Colors.TEXT_TERTIARY))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText,
                        QColor(Colors.TEXT_TERTIARY))

        self.app.setPalette(palette)

    def initialize(self):
        """
        Initialize complete styling system

        Call this once at application startup
        """
        self.set_global_font()
        self.set_dark_palette()
        self.apply_stylesheet("dark")

    @staticmethod
    def get_tier_color(tier: int) -> str:
        """
        Get color for module tier

        Args:
            tier: Tier number (0-4)

        Returns:
            Hex color string
        """
        tier_colors = {
            0: Colors.TIER_0,
            1: Colors.TIER_1,
            2: Colors.TIER_2,
            3: Colors.TIER_3,
            4: Colors.TIER_4,
        }
        return tier_colors.get(tier, Colors.TEXT_SECONDARY)

    @staticmethod
    def get_monospace_font(size: Optional[int] = None) -> QFont:
        """
        Get monospace font for code display

        Args:
            size: Font size in points (default: base size)

        Returns:
            Configured QFont object
        """
        if size is None:
            size = Typography.FONT_SIZE_BASE - 1  # Slightly smaller for code

        font = QFont("JetBrains Mono, Cascadia Code, Consolas", size)
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setFixedPitch(True)
        return font

    @staticmethod
    def apply_widget_property(widget, property_name: str, value):
        """
        Apply dynamic property to widget and update style

        Args:
            widget: QWidget to modify
            property_name: Property name (e.g., "tier", "primary")
            value: Property value
        """
        widget.setProperty(property_name, value)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()


def create_style_manager(app: QApplication) -> StyleManager:
    """
    Factory function to create and initialize StyleManager

    Args:
        app: QApplication instance

    Returns:
        Initialized StyleManager
    """
    manager = StyleManager(app)
    manager.initialize()
    return manager
