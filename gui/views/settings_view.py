"""
Settings View - Application and LLM Configuration

Features:
- LLM model configuration (path, parameters)
- Generation settings (temperature, tokens, etc.)
- Application preferences
- Module management
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QLineEdit, QPushButton, QSpinBox,
    QDoubleSpinBox, QSlider, QComboBox, QCheckBox,
    QGroupBox, QFileDialog, QScrollArea, QTabWidget
)
from PySide6.QtCore import Qt, Signal
from pathlib import Path

from gui import Colors, Spacing, Sizing, Typography
from gui.style_manager import StyleManager


class LLMConfigPanel(QWidget):
    """Panel for LLM model configuration"""

    model_path_changed = Signal(str)
    backend_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.SPACE_BASE, Spacing.SPACE_BASE,
                                 Spacing.SPACE_BASE, Spacing.SPACE_BASE)
        layout.setSpacing(Spacing.SPACE_BASE)

        # Model Path
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(Spacing.SPACE_SM)

        path_layout = QHBoxLayout()
        path_label = QLabel("Model Path:")
        path_label.setMinimumWidth(120)
        path_layout.addWidget(path_label)

        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Path to GGUF model file...")
        self.model_path_input.textChanged.connect(self.model_path_changed.emit)
        path_layout.addWidget(self.model_path_input, stretch=1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_model)
        path_layout.addWidget(browse_btn)

        model_layout.addLayout(path_layout)

        # Backend selection
        backend_layout = QHBoxLayout()
        backend_label = QLabel("Backend:")
        backend_label.setMinimumWidth(120)
        backend_layout.addWidget(backend_label)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems([
            "ctransformers",
            "llama.cpp",
            "transformers",
            "vllm"
        ])
        self.backend_combo.currentTextChanged.connect(self.backend_changed.emit)
        backend_layout.addWidget(self.backend_combo, stretch=1)

        model_layout.addLayout(backend_layout)

        # GPU selection
        gpu_layout = QHBoxLayout()
        gpu_label = QLabel("Device:")
        gpu_label.setMinimumWidth(120)
        gpu_layout.addWidget(gpu_label)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["CUDA (GPU)", "CPU", "Metal (Mac)"])
        gpu_layout.addWidget(self.device_combo, stretch=1)

        model_layout.addLayout(gpu_layout)

        # Context size
        context_layout = QHBoxLayout()
        context_label = QLabel("Context Size:")
        context_label.setMinimumWidth(120)
        context_layout.addWidget(context_label)

        self.context_size_spin = QSpinBox()
        self.context_size_spin.setRange(512, 32768)
        self.context_size_spin.setValue(4096)
        self.context_size_spin.setSingleStep(512)
        context_layout.addWidget(self.context_size_spin)

        context_help = QLabel("(tokens)")
        context_help.setProperty("secondary", True)
        StyleManager.apply_widget_property(context_help, "secondary", True)
        context_layout.addWidget(context_help)
        context_layout.addStretch()

        model_layout.addLayout(context_layout)

        layout.addWidget(model_group)
        layout.addStretch()

    def browse_model(self):
        """Open file dialog to select model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GGUF Model File",
            "",
            "GGUF Files (*.gguf);;All Files (*.*)"
        )
        if file_path:
            self.model_path_input.setText(file_path)

    def get_config(self) -> dict:
        """Get current LLM configuration"""
        return {
            "model_path": self.model_path_input.text(),
            "backend": self.backend_combo.currentText(),
            "device": self.device_combo.currentText(),
            "context_size": self.context_size_spin.value()
        }

    def set_config(self, config: dict):
        """Set configuration values"""
        self.model_path_input.setText(config.get("model_path", ""))
        backend = config.get("backend", "ctransformers")
        index = self.backend_combo.findText(backend)
        if index >= 0:
            self.backend_combo.setCurrentIndex(index)
        device = config.get("device", "CUDA (GPU)")
        index = self.device_combo.findText(device)
        if index >= 0:
            self.device_combo.setCurrentIndex(index)
        self.context_size_spin.setValue(config.get("context_size", 4096))


class GenerationParamsPanel(QWidget):
    """Panel for generation parameters"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.SPACE_BASE, Spacing.SPACE_BASE,
                                 Spacing.SPACE_BASE, Spacing.SPACE_BASE)
        layout.setSpacing(Spacing.SPACE_BASE)

        # Temperature
        temp_group = QGroupBox("Temperature")
        temp_layout = QVBoxLayout(temp_group)

        temp_row = QHBoxLayout()
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 200)  # 0.0 to 2.0, scaled by 100
        self.temp_slider.setValue(70)  # 0.7 default
        self.temp_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.temp_slider.setTickInterval(20)
        temp_row.addWidget(self.temp_slider)

        self.temp_value_label = QLabel("0.70")
        self.temp_value_label.setMinimumWidth(50)
        self.temp_value_label.setFont(StyleManager.get_monospace_font(Typography.FONT_SIZE_BASE))
        temp_row.addWidget(self.temp_value_label)

        self.temp_slider.valueChanged.connect(
            lambda v: self.temp_value_label.setText(f"{v/100:.2f}")
        )

        temp_layout.addLayout(temp_row)

        temp_desc = QLabel("Higher = more creative, lower = more deterministic")
        temp_desc.setProperty("secondary", True)
        StyleManager.apply_widget_property(temp_desc, "secondary", True)
        temp_desc.setStyleSheet(f"font-size: {Typography.FONT_SIZE_SMALL}pt;")
        temp_layout.addWidget(temp_desc)

        layout.addWidget(temp_group)

        # Max tokens
        tokens_group = QGroupBox("Max Tokens")
        tokens_layout = QVBoxLayout(tokens_group)

        tokens_row = QHBoxLayout()
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 8192)
        self.max_tokens_spin.setValue(2048)
        self.max_tokens_spin.setSingleStep(128)
        tokens_row.addWidget(self.max_tokens_spin)
        tokens_row.addStretch()

        tokens_layout.addLayout(tokens_row)

        tokens_desc = QLabel("Maximum number of tokens to generate")
        tokens_desc.setProperty("secondary", True)
        StyleManager.apply_widget_property(tokens_desc, "secondary", True)
        tokens_desc.setStyleSheet(f"font-size: {Typography.FONT_SIZE_SMALL}pt;")
        tokens_layout.addWidget(tokens_desc)

        layout.addWidget(tokens_group)

        # Top P
        top_p_group = QGroupBox("Top P (Nucleus Sampling)")
        top_p_layout = QVBoxLayout(top_p_group)

        top_p_row = QHBoxLayout()
        self.top_p_slider = QSlider(Qt.Orientation.Horizontal)
        self.top_p_slider.setRange(0, 100)  # 0.0 to 1.0, scaled by 100
        self.top_p_slider.setValue(95)  # 0.95 default
        self.top_p_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.top_p_slider.setTickInterval(10)
        top_p_row.addWidget(self.top_p_slider)

        self.top_p_value_label = QLabel("0.95")
        self.top_p_value_label.setMinimumWidth(50)
        self.top_p_value_label.setFont(StyleManager.get_monospace_font(Typography.FONT_SIZE_BASE))
        top_p_row.addWidget(self.top_p_value_label)

        self.top_p_slider.valueChanged.connect(
            lambda v: self.top_p_value_label.setText(f"{v/100:.2f}")
        )

        top_p_layout.addLayout(top_p_row)

        top_p_desc = QLabel("Cumulative probability cutoff for token selection")
        top_p_desc.setProperty("secondary", True)
        StyleManager.apply_widget_property(top_p_desc, "secondary", True)
        top_p_desc.setStyleSheet(f"font-size: {Typography.FONT_SIZE_SMALL}pt;")
        top_p_layout.addWidget(top_p_desc)

        layout.addWidget(top_p_group)

        # Repeat penalty
        repeat_group = QGroupBox("Repeat Penalty")
        repeat_layout = QVBoxLayout(repeat_group)

        repeat_row = QHBoxLayout()
        self.repeat_penalty_spin = QDoubleSpinBox()
        self.repeat_penalty_spin.setRange(1.0, 2.0)
        self.repeat_penalty_spin.setValue(1.1)
        self.repeat_penalty_spin.setSingleStep(0.1)
        self.repeat_penalty_spin.setDecimals(2)
        repeat_row.addWidget(self.repeat_penalty_spin)
        repeat_row.addStretch()

        repeat_layout.addLayout(repeat_row)

        repeat_desc = QLabel("Penalty for repeating tokens (1.0 = no penalty)")
        repeat_desc.setProperty("secondary", True)
        StyleManager.apply_widget_property(repeat_desc, "secondary", True)
        repeat_desc.setStyleSheet(f"font-size: {Typography.FONT_SIZE_SMALL}pt;")
        repeat_layout.addWidget(repeat_desc)

        layout.addWidget(repeat_group)

        layout.addStretch()

    def get_config(self) -> dict:
        """Get current generation parameters"""
        return {
            "temperature": self.temp_slider.value() / 100.0,
            "max_tokens": self.max_tokens_spin.value(),
            "top_p": self.top_p_slider.value() / 100.0,
            "repeat_penalty": self.repeat_penalty_spin.value()
        }

    def set_config(self, config: dict):
        """Set parameter values"""
        self.temp_slider.setValue(int(config.get("temperature", 0.7) * 100))
        self.max_tokens_spin.setValue(config.get("max_tokens", 2048))
        self.top_p_slider.setValue(int(config.get("top_p", 0.95) * 100))
        self.repeat_penalty_spin.setValue(config.get("repeat_penalty", 1.1))


class AppPreferencesPanel(QWidget):
    """Panel for application preferences"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.SPACE_BASE, Spacing.SPACE_BASE,
                                 Spacing.SPACE_BASE, Spacing.SPACE_BASE)
        layout.setSpacing(Spacing.SPACE_BASE)

        # Paths
        paths_group = QGroupBox("Paths")
        paths_layout = QVBoxLayout(paths_group)
        paths_layout.setSpacing(Spacing.SPACE_SM)

        # Modules directory
        modules_row = QHBoxLayout()
        modules_label = QLabel("Modules Directory:")
        modules_label.setMinimumWidth(150)
        modules_row.addWidget(modules_label)

        self.modules_path_input = QLineEdit()
        self.modules_path_input.setText("./modules")
        modules_row.addWidget(self.modules_path_input, stretch=1)

        modules_browse_btn = QPushButton("Browse...")
        modules_browse_btn.clicked.connect(self.browse_modules_dir)
        modules_row.addWidget(modules_browse_btn)

        paths_layout.addLayout(modules_row)

        layout.addWidget(paths_group)

        # UI Preferences
        ui_group = QGroupBox("User Interface")
        ui_layout = QVBoxLayout(ui_group)
        ui_layout.setSpacing(Spacing.SPACE_SM)

        # Theme selection
        theme_row = QHBoxLayout()
        theme_label = QLabel("Theme:")
        theme_label.setMinimumWidth(150)
        theme_row.addWidget(theme_label)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark Mode", "Light Mode (Coming Soon)"])
        theme_row.addWidget(self.theme_combo)
        theme_row.addStretch()

        ui_layout.addLayout(theme_row)

        # Auto-save
        self.autosave_checkbox = QCheckBox("Auto-save settings on change")
        self.autosave_checkbox.setChecked(True)
        ui_layout.addWidget(self.autosave_checkbox)

        # Confirm exit
        self.confirm_exit_checkbox = QCheckBox("Confirm before exit")
        self.confirm_exit_checkbox.setChecked(True)
        ui_layout.addWidget(self.confirm_exit_checkbox)

        layout.addWidget(ui_group)

        # Analysis Preferences
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)

        # Auto-analyze on enter
        self.auto_analyze_checkbox = QCheckBox("Auto-analyze on Enter (Ctrl+Enter to manually trigger)")
        analysis_layout.addWidget(self.auto_analyze_checkbox)

        # Show module trace by default
        self.show_trace_checkbox = QCheckBox("Show module trace by default")
        self.show_trace_checkbox.setChecked(True)
        analysis_layout.addWidget(self.show_trace_checkbox)

        layout.addWidget(analysis_group)

        layout.addStretch()

    def browse_modules_dir(self):
        """Browse for modules directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Modules Directory",
            self.modules_path_input.text()
        )
        if dir_path:
            self.modules_path_input.setText(dir_path)

    def get_config(self) -> dict:
        """Get current preferences"""
        return {
            "modules_path": self.modules_path_input.text(),
            "theme": self.theme_combo.currentText(),
            "autosave": self.autosave_checkbox.isChecked(),
            "confirm_exit": self.confirm_exit_checkbox.isChecked(),
            "auto_analyze": self.auto_analyze_checkbox.isChecked(),
            "show_trace": self.show_trace_checkbox.isChecked()
        }

    def set_config(self, config: dict):
        """Set preference values"""
        self.modules_path_input.setText(config.get("modules_path", "./modules"))
        theme = config.get("theme", "Dark Mode")
        index = self.theme_combo.findText(theme)
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)
        self.autosave_checkbox.setChecked(config.get("autosave", True))
        self.confirm_exit_checkbox.setChecked(config.get("confirm_exit", True))
        self.auto_analyze_checkbox.setChecked(config.get("auto_analyze", False))
        self.show_trace_checkbox.setChecked(config.get("show_trace", True))


class SettingsView(QWidget):
    """
    Complete settings interface with tabs

    Tabs:
    1. LLM Config - Model selection and backend configuration
    2. Generation - Temperature, tokens, sampling parameters
    3. Preferences - Application preferences and UI settings
    """

    settings_changed = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Setup user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = self.create_header()
        layout.addWidget(header)

        # Tab widget
        self.tabs = QTabWidget()

        # Create panels
        self.llm_config_panel = LLMConfigPanel()
        self.tabs.addTab(self.llm_config_panel, "🤖 LLM Configuration")

        self.generation_panel = GenerationParamsPanel()
        self.tabs.addTab(self.generation_panel, "⚙️ Generation Parameters")

        self.preferences_panel = AppPreferencesPanel()
        self.tabs.addTab(self.preferences_panel, "🎨 Preferences")

        layout.addWidget(self.tabs)

        # Action buttons
        button_bar = self.create_button_bar()
        layout.addWidget(button_bar)

    def create_header(self) -> QFrame:
        """Create header bar"""
        header = QFrame()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(Spacing.SPACE_BASE, Spacing.SPACE_SM,
                                        Spacing.SPACE_BASE, Spacing.SPACE_SM)

        title = QLabel("⚙️ Settings")
        title.setProperty("heading", "h2")
        header_layout.addWidget(title)

        header_layout.addStretch()

        header.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_SECONDARY};
                border-bottom: 1px solid {Colors.BORDER_DEFAULT};
            }}
        """)

        return header

    def create_button_bar(self) -> QFrame:
        """Create action button bar"""
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(Spacing.SPACE_BASE, Spacing.SPACE_SM,
                                        Spacing.SPACE_BASE, Spacing.SPACE_SM)

        button_layout.addStretch()

        # Reset to defaults
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(reset_btn)

        # Save settings
        save_btn = QPushButton("💾 Save Settings")
        save_btn.setProperty("primary", True)
        StyleManager.apply_widget_property(save_btn, "primary", True)
        save_btn.clicked.connect(self.save_settings)
        button_layout.addWidget(save_btn)

        button_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_SECONDARY};
                border-top: 1px solid {Colors.BORDER_DEFAULT};
            }}
        """)

        return button_frame

    def get_all_config(self) -> dict:
        """Get all configuration from all panels"""
        return {
            "llm": self.llm_config_panel.get_config(),
            "generation": self.generation_panel.get_config(),
            "preferences": self.preferences_panel.get_config()
        }

    def set_all_config(self, config: dict):
        """Set configuration for all panels"""
        if "llm" in config:
            self.llm_config_panel.set_config(config["llm"])
        if "generation" in config:
            self.generation_panel.set_config(config["generation"])
        if "preferences" in config:
            self.preferences_panel.set_config(config["preferences"])

    def save_settings(self):
        """Save current settings"""
        config = self.get_all_config()
        self.settings_changed.emit(config)
        print("Settings saved:", config)
        # TODO: Persist to file

    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        defaults = {
            "llm": {
                "model_path": "",
                "backend": "ctransformers",
                "device": "CUDA (GPU)",
                "context_size": 4096
            },
            "generation": {
                "temperature": 0.7,
                "max_tokens": 2048,
                "top_p": 0.95,
                "repeat_penalty": 1.1
            },
            "preferences": {
                "modules_path": "./modules",
                "theme": "Dark Mode",
                "autosave": True,
                "confirm_exit": True,
                "auto_analyze": False,
                "show_trace": True
            }
        }
        self.set_all_config(defaults)
