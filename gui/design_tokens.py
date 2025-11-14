"""
Design Tokens - Protocol AI Dark Mode Design System

Centralized design tokens for colors, typography, spacing, and sizing.
Based on modern dark mode best practices with emphasis on readability and
analytical precision.
"""

from typing import Dict, Any


class Colors:
    """Dark mode color palette - High contrast, low eye strain"""

    # Base Background Colors
    BG_PRIMARY = "#0d1117"      # Main background (GitHub dark)
    BG_SECONDARY = "#161b22"    # Secondary panels
    BG_TERTIARY = "#1c2128"     # Elevated elements
    BG_OVERLAY = "#21262d"      # Overlays, modals

    # Surface Colors
    SURFACE_RAISED = "#1c2128"  # Raised surfaces
    SURFACE_SUNKEN = "#0d1117"  # Sunken/inset areas

    # Text Colors
    TEXT_PRIMARY = "#e6edf3"    # Primary text (high contrast)
    TEXT_SECONDARY = "#8b949e"  # Secondary text
    TEXT_TERTIARY = "#6e7681"   # Tertiary/disabled text
    TEXT_LINK = "#58a6ff"       # Links
    TEXT_INVERSE = "#0d1117"    # Text on light backgrounds

    # Accent Colors
    ACCENT_PRIMARY = "#58a6ff"   # Primary accent (blue)
    ACCENT_SECONDARY = "#bc8cff"  # Secondary accent (purple)
    ACCENT_SUCCESS = "#3fb950"    # Success/positive
    ACCENT_WARNING = "#d29922"    # Warning/caution
    ACCENT_ERROR = "#f85149"      # Error/danger
    ACCENT_INFO = "#58a6ff"       # Info

    # Semantic Colors
    SUCCESS = "#238636"          # Success background
    SUCCESS_EMPHASIS = "#2ea043" # Success emphasis
    WARNING = "#9e6a03"          # Warning background
    WARNING_EMPHASIS = "#bb8009" # Warning emphasis
    ERROR = "#da3633"            # Error background
    ERROR_EMPHASIS = "#f85149"   # Error emphasis

    # Border Colors
    BORDER_DEFAULT = "#30363d"   # Default borders
    BORDER_MUTED = "#21262d"     # Subtle borders
    BORDER_EMPHASIS = "#6e7681"  # Emphasized borders
    BORDER_ACCENT = "#58a6ff"    # Accent borders

    # Tier Colors (for module hierarchy)
    TIER_0 = "#f0883e"  # Commands (orange)
    TIER_1 = "#f85149"  # Safety (red)
    TIER_2 = "#58a6ff"  # Analysis (blue)
    TIER_3 = "#bc8cff"  # Heuristics (purple)
    TIER_4 = "#3fb950"  # Style (green)

    # Syntax Highlighting Colors
    SYNTAX_KEYWORD = "#ff7b72"    # Keywords
    SYNTAX_STRING = "#a5d6ff"     # Strings
    SYNTAX_NUMBER = "#79c0ff"     # Numbers
    SYNTAX_COMMENT = "#8b949e"    # Comments
    SYNTAX_FUNCTION = "#d2a8ff"   # Functions
    SYNTAX_VARIABLE = "#ffa657"   # Variables
    SYNTAX_CLASS = "#7ee787"      # Classes
    SYNTAX_OPERATOR = "#ff7b72"   # Operators

    # State Colors
    STATE_HOVER = "#30363d"       # Hover state
    STATE_ACTIVE = "#21262d"      # Active/pressed state
    STATE_FOCUS = "#58a6ff40"     # Focus outline (with alpha)
    STATE_SELECTED = "#1f6feb"    # Selected items
    STATE_DISABLED = "#484f58"    # Disabled elements

    # Chart/Visualization Colors
    CHART_1 = "#58a6ff"
    CHART_2 = "#bc8cff"
    CHART_3 = "#3fb950"
    CHART_4 = "#d29922"
    CHART_5 = "#f85149"
    CHART_6 = "#79c0ff"


class Typography:
    """Typography system - Optimized for code and analytical content"""

    # Font Families
    FONT_FAMILY_SANS = "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"
    FONT_FAMILY_MONO = "'JetBrains Mono', 'Cascadia Code', 'Fira Code', 'Consolas', monospace"
    FONT_FAMILY_DISPLAY = "'Inter', 'Segoe UI', system-ui, sans-serif"

    # Font Sizes (pt for Qt)
    FONT_SIZE_TINY = 10
    FONT_SIZE_SMALL = 11
    FONT_SIZE_BASE = 13
    FONT_SIZE_MEDIUM = 14
    FONT_SIZE_LARGE = 16
    FONT_SIZE_XL = 18
    FONT_SIZE_XXL = 24
    FONT_SIZE_DISPLAY = 32

    # Font Weights
    FONT_WEIGHT_LIGHT = 300
    FONT_WEIGHT_NORMAL = 400
    FONT_WEIGHT_MEDIUM = 500
    FONT_WEIGHT_SEMIBOLD = 600
    FONT_WEIGHT_BOLD = 700

    # Line Heights (multipliers)
    LINE_HEIGHT_TIGHT = 1.25
    LINE_HEIGHT_NORMAL = 1.5
    LINE_HEIGHT_RELAXED = 1.75
    LINE_HEIGHT_LOOSE = 2.0


class Spacing:
    """Spacing and sizing system - 4px base unit"""

    # Base unit: 4px
    UNIT = 4

    # Spacing scale
    SPACE_NONE = 0
    SPACE_XXS = 2    # 2px
    SPACE_XS = 4     # 4px
    SPACE_SM = 8     # 8px
    SPACE_MD = 12    # 12px
    SPACE_BASE = 16  # 16px
    SPACE_LG = 24    # 24px
    SPACE_XL = 32    # 32px
    SPACE_XXL = 48   # 48px
    SPACE_XXXL = 64  # 64px

    # Component-specific spacing
    PADDING_BUTTON = (8, 16)      # (vertical, horizontal)
    PADDING_INPUT = (8, 12)
    PADDING_PANEL = 16
    PADDING_CARD = 20
    PADDING_CONTAINER = 24

    # Gaps
    GAP_TIGHT = 4
    GAP_NORMAL = 8
    GAP_RELAXED = 12
    GAP_LOOSE = 16


class Sizing:
    """Size constants for UI elements"""

    # Border radius
    RADIUS_NONE = 0
    RADIUS_SM = 3
    RADIUS_BASE = 6
    RADIUS_MD = 8
    RADIUS_LG = 12
    RADIUS_FULL = 9999  # Fully rounded

    # Border widths
    BORDER_THIN = 1
    BORDER_BASE = 1
    BORDER_THICK = 2
    BORDER_HEAVY = 3

    # Component heights
    HEIGHT_INPUT = 32
    HEIGHT_BUTTON_SM = 28
    HEIGHT_BUTTON_BASE = 32
    HEIGHT_BUTTON_LG = 40
    HEIGHT_TOOLBAR = 48
    HEIGHT_HEADER = 56
    HEIGHT_FOOTER = 40

    # Widths
    WIDTH_SIDEBAR = 280
    WIDTH_PANEL_SM = 320
    WIDTH_PANEL_MD = 480
    WIDTH_PANEL_LG = 640

    # Icon sizes
    ICON_XS = 12
    ICON_SM = 16
    ICON_BASE = 20
    ICON_LG = 24
    ICON_XL = 32


class Shadows:
    """Shadow definitions for depth"""

    SHADOW_NONE = "none"
    SHADOW_SM = "0 1px 3px rgba(0, 0, 0, 0.5)"
    SHADOW_BASE = "0 2px 8px rgba(0, 0, 0, 0.5)"
    SHADOW_MD = "0 4px 16px rgba(0, 0, 0, 0.5)"
    SHADOW_LG = "0 8px 32px rgba(0, 0, 0, 0.6)"
    SHADOW_XL = "0 16px 48px rgba(0, 0, 0, 0.7)"


class Transitions:
    """Animation timing constants"""

    DURATION_INSTANT = 0
    DURATION_FAST = 100      # ms
    DURATION_BASE = 200      # ms
    DURATION_MODERATE = 300  # ms
    DURATION_SLOW = 500      # ms

    EASING_LINEAR = "linear"
    EASING_EASE = "ease"
    EASING_EASE_IN = "ease-in"
    EASING_EASE_OUT = "ease-out"
    EASING_EASE_IN_OUT = "ease-in-out"


class ZIndex:
    """Z-index layering system"""

    BASE = 0
    DROPDOWN = 1000
    STICKY = 1020
    FIXED = 1030
    MODAL_BACKDROP = 1040
    MODAL = 1050
    POPOVER = 1060
    TOOLTIP = 1070
    NOTIFICATION = 1080


# Design token export for easy access
DESIGN_TOKENS: Dict[str, Any] = {
    "colors": Colors,
    "typography": Typography,
    "spacing": Spacing,
    "sizing": Sizing,
    "shadows": Shadows,
    "transitions": Transitions,
    "zindex": ZIndex,
}


def get_token(category: str, token: str) -> Any:
    """
    Get a design token value

    Usage:
        color = get_token("colors", "ACCENT_PRIMARY")
        size = get_token("typography", "FONT_SIZE_BASE")
    """
    if category in DESIGN_TOKENS:
        token_class = DESIGN_TOKENS[category]
        return getattr(token_class, token, None)
    return None
