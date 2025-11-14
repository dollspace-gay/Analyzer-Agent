"""
Protocol AI - Main Dashboard Window

The primary application window with sidebar navigation, header bar, and
dynamic content area for different views.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QStackedWidget, QSplitter, QScrollArea,
    QToolBar, QStatusBar
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QAction, QIcon

from gui import Colors, Spacing, Sizing, Typography
from gui.style_manager import StyleManager
from gui.views import AnalyzeView, ModulesView, SettingsView
import psutil
import platform


class HeaderBar(QWidget):
    """Top header bar with title and system status"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(Spacing.SPACE_BASE, Spacing.SPACE_SM,
                                 Spacing.SPACE_BASE, Spacing.SPACE_SM)
        layout.setSpacing(Spacing.SPACE_BASE)

        # Application title
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(0)

        self.title = QLabel("Protocol AI")
        self.title.setProperty("heading", "h2")
        title_layout.addWidget(self.title)

        self.subtitle = QLabel("Governance Layer Interface")
        self.subtitle.setProperty("secondary", True)
        title_layout.addWidget(self.subtitle)

        layout.addWidget(title_container)
        layout.addStretch()

        # System status indicator
        self.status_indicator = QLabel("● System Ready")
        self.status_indicator.setStyleSheet(f"color: {Colors.ACCENT_SUCCESS};")
        layout.addWidget(self.status_indicator)

        # Style the header
        self.setStyleSheet(f"""
            HeaderBar {{
                background-color: {Colors.BG_SECONDARY};
                border-bottom: 1px solid {Colors.BORDER_DEFAULT};
            }}
        """)

    def set_status(self, status: str, color: str = Colors.ACCENT_SUCCESS):
        """Update status indicator"""
        self.status_indicator.setText(f"● {status}")
        self.status_indicator.setStyleSheet(f"color: {color};")


class SidebarButton(QPushButton):
    """Custom sidebar navigation button"""

    def __init__(self, text: str, icon_text: str = ""):
        super().__init__(text)
        self.icon_text = icon_text
        self.setCheckable(True)
        self.setMinimumHeight(Sizing.HEIGHT_BUTTON_LG)
        self.setStyleSheet(f"""
            SidebarButton {{
                text-align: left;
                padding-left: {Spacing.SPACE_BASE}px;
                font-size: {Typography.FONT_SIZE_BASE}pt;
                font-weight: {Typography.FONT_WEIGHT_MEDIUM};
            }}
            SidebarButton:checked {{
                background-color: {Colors.STATE_SELECTED};
                border-left: 3px solid {Colors.ACCENT_PRIMARY};
            }}
        """)


class Sidebar(QWidget):
    """Left sidebar with navigation buttons"""

    view_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.buttons = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, Spacing.SPACE_SM, 0, 0)
        layout.setSpacing(Spacing.SPACE_XXS)

        # Navigation buttons
        nav_items = [
            ("analyze", "Analyze", "🔍"),
            ("modules", "Module Library", "📚"),
            ("history", "History", "📜"),
            ("settings", "Settings", "⚙️"),
        ]

        for view_id, text, icon in nav_items:
            btn = SidebarButton(f"{icon}  {text}")
            btn.clicked.connect(lambda checked, v=view_id: self.on_nav_clicked(v))
            self.buttons[view_id] = btn
            layout.addWidget(btn)

        layout.addStretch()

        # System info at bottom
        self.system_info = QWidget()
        self.setup_system_info()
        layout.addWidget(self.system_info)

        # Style sidebar
        self.setFixedWidth(Sizing.WIDTH_SIDEBAR)
        self.setStyleSheet(f"""
            Sidebar {{
                background-color: {Colors.BG_SECONDARY};
                border-right: 1px solid {Colors.BORDER_DEFAULT};
            }}
        """)

        # Select first button by default
        if "analyze" in self.buttons:
            self.buttons["analyze"].setChecked(True)

    def setup_system_info(self):
        """Create system info panel"""
        layout = QVBoxLayout(self.system_info)
        layout.setContentsMargins(Spacing.SPACE_SM, Spacing.SPACE_SM,
                                 Spacing.SPACE_SM, Spacing.SPACE_SM)
        layout.setSpacing(Spacing.SPACE_XS)

        # System stats
        self.cpu_label = QLabel("CPU: --")
        self.cpu_label.setProperty("secondary", True)
        layout.addWidget(self.cpu_label)

        self.memory_label = QLabel("Memory: --")
        self.memory_label.setProperty("secondary", True)
        layout.addWidget(self.memory_label)

        self.gpu_label = QLabel("GPU: N/A")
        self.gpu_label.setProperty("secondary", True)
        layout.addWidget(self.gpu_label)

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_system_stats)
        self.update_timer.start(2000)  # Update every 2 seconds
        self.update_system_stats()

        self.system_info.setStyleSheet(f"""
            QWidget {{
                background-color: {Colors.BG_PRIMARY};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: {Sizing.RADIUS_BASE}px;
            }}
        """)

    def update_system_stats(self):
        """Update system statistics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_label.setText(f"CPU: {cpu_percent:.1f}%")

            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_gb = memory.used / (1024 ** 3)
            self.memory_label.setText(f"Memory: {memory_gb:.1f}GB ({memory_percent:.0f}%)")

            # GPU info (basic - would need GPU-specific library for detailed stats)
            try:
                import torch
                if torch.cuda.is_available():
                    self.gpu_label.setText(f"GPU: {torch.cuda.get_device_name(0)}")
                else:
                    self.gpu_label.setText("GPU: CPU Mode")
            except ImportError:
                self.gpu_label.setText("GPU: N/A")

        except Exception as e:
            print(f"Error updating system stats: {e}")

    def on_nav_clicked(self, view_id: str):
        """Handle navigation button click"""
        # Uncheck all other buttons
        for btn_id, btn in self.buttons.items():
            if btn_id != view_id:
                btn.setChecked(False)

        # Ensure clicked button stays checked
        self.buttons[view_id].setChecked(True)

        # Emit signal
        self.view_changed.emit(view_id)


class QuickActionsPanel(QWidget):
    """Quick actions panel with common operations"""

    analyze_clicked = Signal()
    clear_clicked = Signal()
    export_clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.SPACE_BASE, Spacing.SPACE_BASE,
                                 Spacing.SPACE_BASE, Spacing.SPACE_BASE)
        layout.setSpacing(Spacing.SPACE_SM)

        # Title
        title = QLabel("Quick Actions")
        title.setProperty("heading", "h3")
        layout.addWidget(title)

        # Action buttons
        self.analyze_btn = QPushButton("🔍 Analyze Prompt")
        self.analyze_btn.setProperty("primary", True)
        StyleManager.apply_widget_property(self.analyze_btn, "primary", True)
        self.analyze_btn.clicked.connect(self.analyze_clicked.emit)
        layout.addWidget(self.analyze_btn)

        self.clear_btn = QPushButton("🗑️ Clear Session")
        self.clear_btn.clicked.connect(self.clear_clicked.emit)
        layout.addWidget(self.clear_btn)

        self.export_btn = QPushButton("💾 Export Results")
        self.export_btn.clicked.connect(self.export_clicked.emit)
        layout.addWidget(self.export_btn)

        layout.addStretch()

        # Module stats
        stats_container = QFrame()
        stats_layout = QVBoxLayout(stats_container)
        stats_layout.setSpacing(Spacing.SPACE_XS)

        stats_title = QLabel("Module Stats")
        stats_title.setProperty("secondary", True)
        stats_layout.addWidget(stats_title)

        self.modules_loaded = QLabel("Loaded: --")
        self.modules_triggered = QLabel("Triggered: --")
        self.modules_selected = QLabel("Selected: --")

        stats_layout.addWidget(self.modules_loaded)
        stats_layout.addWidget(self.modules_triggered)
        stats_layout.addWidget(self.modules_selected)

        layout.addWidget(stats_container)

        self.setFixedWidth(Sizing.WIDTH_PANEL_SM)
        self.setStyleSheet(f"""
            QuickActionsPanel {{
                background-color: {Colors.BG_SECONDARY};
                border-left: 1px solid {Colors.BORDER_DEFAULT};
            }}
        """)

    def update_module_stats(self, loaded: int, triggered: int, selected: int):
        """Update module statistics"""
        self.modules_loaded.setText(f"Loaded: {loaded}")
        self.modules_triggered.setText(f"Triggered: {triggered}")
        self.modules_selected.setText(f"Selected: {selected}")


class ContentArea(QWidget):
    """Main content area with stacked views"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Stacked widget for different views
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        # Create views
        self.views = {}
        self.add_view("analyze", AnalyzeView())  # Real analyze view with prompt input
        self.add_view("modules", ModulesView())  # Real modules view with browser
        self.add_view("history", self.create_placeholder("History", "Conversation history will go here"))
        self.add_view("settings", SettingsView())  # Real settings view with configuration

    def add_view(self, view_id: str, widget: QWidget):
        """Add a view to the stack"""
        self.views[view_id] = widget
        self.stack.addWidget(widget)

    def show_view(self, view_id: str):
        """Switch to specified view"""
        if view_id in self.views:
            self.stack.setCurrentWidget(self.views[view_id])

    def replace_view(self, view_id: str, widget: QWidget):
        """Replace an existing view"""
        if view_id in self.views:
            old_widget = self.views[view_id]
            index = self.stack.indexOf(old_widget)
            self.stack.removeWidget(old_widget)
            old_widget.deleteLater()

        self.views[view_id] = widget
        self.stack.addWidget(widget)

    def create_placeholder(self, title: str, description: str) -> QWidget:
        """Create placeholder view"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel(title)
        title_label.setProperty("heading", "h1")
        layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)

        desc_label = QLabel(description)
        desc_label.setProperty("secondary", True)
        layout.addWidget(desc_label, alignment=Qt.AlignmentFlag.AlignCenter)

        return widget


class ProtocolAIMainWindow(QMainWindow):
    """Main application window with complete dashboard layout"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Protocol AI - Governance Layer")
        self.setGeometry(100, 100, 1600, 1000)

        self.setup_ui()
        self.create_menu_bar()
        self.create_status_bar()

    def setup_ui(self):
        """Setup main UI layout"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header bar
        self.header = HeaderBar()
        main_layout.addWidget(self.header)

        # Main content area with sidebar and content
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Sidebar
        self.sidebar = Sidebar()
        self.sidebar.view_changed.connect(self.on_view_changed)
        content_layout.addWidget(self.sidebar)

        # Content area
        self.content_area = ContentArea()
        content_layout.addWidget(self.content_area, stretch=1)

        # Quick actions panel (right side)
        self.quick_actions = QuickActionsPanel()
        self.quick_actions.analyze_clicked.connect(self.on_analyze_clicked)
        self.quick_actions.clear_clicked.connect(self.on_clear_clicked)
        self.quick_actions.export_clicked.connect(self.on_export_clicked)
        content_layout.addWidget(self.quick_actions)

        main_layout.addLayout(content_layout, stretch=1)

    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Session", self)
        new_action.setShortcut("Ctrl+N")
        file_menu.addAction(new_action)

        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)

        save_action = QAction("&Save", self)
        save_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        copy_action = QAction("&Copy", self)
        copy_action.setShortcut("Ctrl+C")
        edit_menu.addAction(copy_action)

        paste_action = QAction("&Paste", self)
        paste_action.setShortcut("Ctrl+V")
        edit_menu.addAction(paste_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        toggle_sidebar_action = QAction("Toggle &Sidebar", self)
        toggle_sidebar_action.setShortcut("Ctrl+B")
        toggle_sidebar_action.triggered.connect(self.toggle_sidebar)
        view_menu.addAction(toggle_sidebar_action)

        toggle_quick_actions_action = QAction("Toggle &Quick Actions", self)
        toggle_quick_actions_action.setShortcut("Ctrl+K")
        toggle_quick_actions_action.triggered.connect(self.toggle_quick_actions)
        view_menu.addAction(toggle_quick_actions_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        docs_action = QAction("&Documentation", self)
        docs_action.setShortcut("F1")
        help_menu.addAction(docs_action)

        about_action = QAction("&About", self)
        help_menu.addAction(about_action)

    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def on_view_changed(self, view_id: str):
        """Handle view change from sidebar"""
        self.content_area.show_view(view_id)
        self.status_bar.showMessage(f"Switched to {view_id.title()} view")

    def on_analyze_clicked(self):
        """Handle analyze button click"""
        self.status_bar.showMessage("Analysis started...")
        self.header.set_status("Analyzing...", Colors.ACCENT_WARNING)

    def on_clear_clicked(self):
        """Handle clear button click"""
        self.status_bar.showMessage("Session cleared")
        self.header.set_status("System Ready", Colors.ACCENT_SUCCESS)

    def on_export_clicked(self):
        """Handle export button click"""
        self.status_bar.showMessage("Exporting results...")

    def toggle_sidebar(self):
        """Toggle sidebar visibility"""
        self.sidebar.setVisible(not self.sidebar.isVisible())

    def toggle_quick_actions(self):
        """Toggle quick actions panel visibility"""
        self.quick_actions.setVisible(not self.quick_actions.isVisible())
