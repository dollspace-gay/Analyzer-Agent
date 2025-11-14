"""
Protocol AI - Application Entry Point

Main application launcher with proper initialization.
"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from gui import create_style_manager
from gui.main_window import ProtocolAIMainWindow


def main():
    """Launch Protocol AI GUI application"""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Protocol AI")
    app.setOrganizationName("Protocol AI")

    # Initialize style manager (dark mode design system)
    style_manager = create_style_manager(app)

    # Create and show main window
    window = ProtocolAIMainWindow()
    window.show()

    # Run application event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
