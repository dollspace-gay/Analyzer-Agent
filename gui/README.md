# Protocol AI - Dark Mode Design System

Professional dark mode GUI design system built with **PySide6** (Qt for Python).

## 🎨 Design System Structure

```
gui/
├── design_tokens.py    # Color palette, typography, spacing constants
├── styles.qss          # QSS stylesheet (Qt Style Sheets - like CSS)
├── style_manager.py    # Central styling and theme management
├── demo.py             # Interactive design system showcase
└── __init__.py         # Package exports
```

## 🚀 Quick Start

### Installation

```bash
pip install PySide6
```

### Basic Usage

```python
from PySide6.QtWidgets import QApplication, QMainWindow
from gui import create_style_manager

# Create application
app = QApplication([])

# Initialize dark mode design system
style_manager = create_style_manager(app)

# Create your window
window = QMainWindow()
window.setWindowTitle("Protocol AI")
window.show()

app.exec()
```

### Run Demo

```bash
python -m gui.demo
```

## 🎨 Design Tokens

All design values are centralized in `design_tokens.py`:

### Colors

```python
from gui import Colors

# Background colors
Colors.BG_PRIMARY      # #0d1117 - Main background
Colors.BG_SECONDARY    # #161b22 - Secondary panels
Colors.BG_TERTIARY     # #1c2128 - Elevated elements

# Text colors
Colors.TEXT_PRIMARY    # #e6edf3 - Primary text
Colors.TEXT_SECONDARY  # #8b949e - Secondary text
Colors.TEXT_LINK       # #58a6ff - Links

# Accent colors
Colors.ACCENT_PRIMARY  # #58a6ff - Blue
Colors.ACCENT_SUCCESS  # #3fb950 - Green
Colors.ACCENT_ERROR    # #f85149 - Red

# Tier colors (for module hierarchy)
Colors.TIER_0  # #f0883e - Commands (orange)
Colors.TIER_1  # #f85149 - Safety (red)
Colors.TIER_2  # #58a6ff - Analysis (blue)
Colors.TIER_3  # #bc8cff - Heuristics (purple)
Colors.TIER_4  # #3fb950 - Style (green)
```

### Typography

```python
from gui import Typography

# Font families
Typography.FONT_FAMILY_SANS  # Inter, Segoe UI, system-ui
Typography.FONT_FAMILY_MONO  # JetBrains Mono, Cascadia Code

# Font sizes (pt)
Typography.FONT_SIZE_BASE    # 13pt
Typography.FONT_SIZE_LARGE   # 16pt
Typography.FONT_SIZE_DISPLAY # 32pt

# Font weights
Typography.FONT_WEIGHT_NORMAL   # 400
Typography.FONT_WEIGHT_SEMIBOLD # 600
Typography.FONT_WEIGHT_BOLD     # 700
```

### Spacing

```python
from gui import Spacing

# Spacing scale (4px base unit)
Spacing.SPACE_XS    # 4px
Spacing.SPACE_SM    # 8px
Spacing.SPACE_BASE  # 16px
Spacing.SPACE_LG    # 24px
Spacing.SPACE_XL    # 32px
```

### Sizing

```python
from gui import Sizing

# Border radius
Sizing.RADIUS_SM    # 3px
Sizing.RADIUS_BASE  # 6px
Sizing.RADIUS_LG    # 12px

# Component heights
Sizing.HEIGHT_BUTTON_BASE  # 32px
Sizing.HEIGHT_TOOLBAR      # 48px

# Widths
Sizing.WIDTH_SIDEBAR    # 280px
Sizing.WIDTH_PANEL_MD   # 480px
```

## 🧩 Component Styles

### Buttons

```python
from PySide6.QtWidgets import QPushButton

# Default button (automatically styled)
button = QPushButton("Click Me")

# Primary button (green)
primary_btn = QPushButton("Submit")
primary_btn.setProperty("primary", True)

# Danger button (red)
danger_btn = QPushButton("Delete")
danger_btn.setProperty("danger", True)
```

### Inputs

```python
from PySide6.QtWidgets import QLineEdit, QTextEdit

# Line edit
line_edit = QLineEdit()
line_edit.setPlaceholderText("Enter text...")

# Text edit (multiline)
text_edit = QTextEdit()
text_edit.setPlaceholderText("Enter multiple lines...")
```

### Typography Labels

```python
from PySide6.QtWidgets import QLabel

# Heading 1
h1 = QLabel("Large Heading")
h1.setProperty("heading", "h1")

# Heading 2
h2 = QLabel("Medium Heading")
h2.setProperty("heading", "h2")

# Secondary text
secondary = QLabel("Less important text")
secondary.setProperty("secondary", True)
```

### Module Tier Colors

```python
from PySide6.QtWidgets import QFrame

# Create widget with tier color border
tier1_widget = QFrame()
tier1_widget.setProperty("tier", 1)  # Safety tier (red)

# Apply the style
from gui.style_manager import StyleManager
StyleManager.apply_widget_property(tier1_widget, "tier", 1)
```

### Monospace/Code Display

```python
from PySide6.QtWidgets import QTextEdit
from gui.style_manager import StyleManager

# Code editor
code_editor = QTextEdit()
code_editor.setFont(StyleManager.get_monospace_font())
code_editor.setProperty("monospace", True)
```

## 🎯 Style Manager API

```python
from gui import StyleManager

# Get tier color
color = style_manager.get_tier_color(2)  # Returns blue for Tier 2

# Get monospace font
font = StyleManager.get_monospace_font(size=12)

# Apply dynamic property to widget
StyleManager.apply_widget_property(widget, "primary", True)
```

## 🎨 Color Palette

### Background Colors
- **Primary** (#0d1117): Main application background
- **Secondary** (#161b22): Panels, sidebars
- **Tertiary** (#1c2128): Elevated surfaces, cards
- **Overlay** (#21262d): Modals, tooltips

### Text Colors
- **Primary** (#e6edf3): Main text (high contrast)
- **Secondary** (#8b949e): Less emphasis
- **Tertiary** (#6e7681): Disabled, minimal emphasis

### Accent Colors
- **Blue** (#58a6ff): Primary accent, links
- **Green** (#3fb950): Success, positive actions
- **Red** (#f85149): Errors, danger actions
- **Yellow** (#d29922): Warnings, caution
- **Purple** (#bc8cff): Secondary accent

### Tier Colors (Protocol AI Specific)
- **Tier 0** (#f0883e): Commands - Orange
- **Tier 1** (#f85149): Safety & Integrity - Red
- **Tier 2** (#58a6ff): Core Analysis - Blue
- **Tier 3** (#bc8cff): Heuristics & Context - Purple
- **Tier 4** (#3fb950): Style & Formatting - Green

## 📝 QSS Stylesheet

The `styles.qss` file contains comprehensive styling for all Qt widgets:

- Buttons (default, primary, danger, disabled)
- Input fields (line edit, text edit, combo boxes)
- Selection controls (checkboxes, radio buttons, sliders)
- Data display (lists, trees, tables)
- Containers (frames, group boxes, tabs)
- Scrollbars (custom styled, minimal)
- Menus and toolbars
- Dock widgets and splitters

## 🔧 Customization

### Adding Custom Colors

Edit `gui/design_tokens.py`:

```python
class Colors:
    # Add your custom color
    CUSTOM_ACCENT = "#ff6b6b"
```

### Modifying Styles

Edit `gui/styles.qss` to change widget appearances:

```qss
QPushButton[custom="true"] {
    background-color: #ff6b6b;
    color: #ffffff;
}
```

### Creating Custom Widgets

```python
from PySide6.QtWidgets import QWidget
from gui import Colors, Spacing

class CustomPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"""
            CustomPanel {{
                background-color: {Colors.BG_SECONDARY};
                border-radius: {Spacing.RADIUS_BASE}px;
                padding: {Spacing.SPACE_BASE}px;
            }}
        """)
```

## 🎓 Best Practices

1. **Use Design Tokens**: Always reference `design_tokens.py` instead of hardcoding values
2. **Consistent Spacing**: Use spacing scale (4px base unit)
3. **Semantic Colors**: Use `TEXT_PRIMARY`, `ACCENT_SUCCESS` instead of hex codes
4. **Dynamic Properties**: Use `setProperty()` for state-based styling
5. **Monospace for Code**: Always use monospace fonts for code display

## 📚 Examples

### Creating a Module Display Panel

```python
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
from gui import Colors, Spacing
from gui.style_manager import StyleManager

def create_module_panel(module_name: str, tier: int):
    panel = QFrame()
    panel.setProperty("tier", tier)
    StyleManager.apply_widget_property(panel, "tier", tier)

    layout = QVBoxLayout(panel)
    layout.setSpacing(Spacing.SPACE_SM)

    # Module name
    name_label = QLabel(module_name)
    name_label.setProperty("heading", "h3")
    layout.addWidget(name_label)

    # Tier indicator
    tier_label = QLabel(f"Tier {tier}")
    tier_label.setProperty("secondary", True)
    layout.addWidget(tier_label)

    return panel
```

### Creating a Code Editor

```python
from PySide6.QtWidgets import QTextEdit
from gui.style_manager import StyleManager

editor = QTextEdit()
editor.setFont(StyleManager.get_monospace_font(12))
editor.setProperty("monospace", True)
editor.setPlainText("def analyze():\n    return True")
```

## 🐛 Troubleshooting

**Issue**: Styles not applying
- **Solution**: Ensure `create_style_manager(app)` is called before creating widgets

**Issue**: Dynamic properties not updating
- **Solution**: Use `StyleManager.apply_widget_property()` to force style refresh

**Issue**: Fonts look wrong
- **Solution**: Install Inter and JetBrains Mono fonts for best results

## 📦 Dependencies

- **PySide6** >= 6.10.0 (Qt 6.10)
- Python >= 3.9

## 📄 License

MIT License - Part of the Protocol AI project
