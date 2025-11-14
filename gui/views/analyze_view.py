"""
Analyze View - Main analysis interface

Combines prompt input and response display for the primary workflow.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter, QLabel
from PySide6.QtCore import Qt, QThread, Signal

from gui.widgets.prompt_input import PromptInputWidget
from gui.widgets.response_display import ResponseDisplayWidget
from gui.backend_service import get_backend_service
from gui import Spacing


class PromptWorker(QThread):
    """Worker thread for async prompt processing"""

    finished = Signal(dict)  # Emits response data when complete
    error = Signal(str)  # Emits error message on failure

    def __init__(self, prompt: str):
        super().__init__()
        self.prompt = prompt
        self.backend = get_backend_service()

    def run(self):
        """Execute prompt processing in background thread"""
        try:
            response = self.backend.process_prompt_sync(self.prompt)
            self.finished.emit(response)
        except Exception as e:
            self.error.emit(str(e))


class AnalyzeView(QWidget):
    """
    Main analyze view with prompt input and response display

    Layout:
    - Top: Prompt input (expandable)
    - Bottom: Response display (expandable)
    """

    def __init__(self):
        super().__init__()
        self.worker = None  # Current worker thread
        self.backend = get_backend_service()
        self.setup_ui()

    def setup_ui(self):
        """Setup user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Prompt input section
        self.prompt_input = PromptInputWidget()
        self.prompt_input.prompt_submitted.connect(self.on_prompt_submitted)
        splitter.addWidget(self.prompt_input)

        # Response display with module trace
        self.response_display = ResponseDisplayWidget()
        splitter.addWidget(self.response_display)

        # Set initial sizes (40% prompt, 60% response)
        splitter.setSizes([400, 600])

        layout.addWidget(splitter)

    def on_prompt_submitted(self, prompt_text: str):
        """Handle prompt submission"""
        print(f"Prompt submitted: {prompt_text[:50]}...")

        # Check if backend is initialized
        if not self.backend.is_initialized:
            # Show demo response if backend not ready
            demo_response = {
                "response": (
                    "⚠️ Backend Not Initialized\n\n"
                    "The Protocol AI backend is not configured. Please:\n\n"
                    "1. Go to Settings (⚙️)\n"
                    "2. Configure LLM path and parameters\n"
                    "3. Click 'Save Settings'\n"
                    "4. Return here to analyze prompts\n\n"
                    "Once configured, this will run real analysis through the "
                    "Governance Layer with module triggering, arbitration, and LLM execution."
                ),
                "modules": [],
                "structured_prompt": "Backend not initialized",
                "arbitration_log": [
                    "[WARNING] Backend not initialized",
                    "[INFO] Configure LLM in Settings to enable analysis"
                ],
                "metadata": {
                    "tokens": 0,
                    "time_ms": 0,
                    "model": "not-configured",
                    "timestamp": ""
                }
            }
            self.response_display.display_response(demo_response)
            return

        # Disable submit button while processing
        self.prompt_input.submit_btn.setEnabled(False)
        self.prompt_input.submit_btn.setText("🔄 Processing...")

        # Create and start worker thread
        self.worker = PromptWorker(prompt_text)
        self.worker.finished.connect(self.on_response_ready)
        self.worker.error.connect(self.on_response_error)
        self.worker.start()

    def on_response_ready(self, response_data: dict):
        """Handle successful response from backend"""
        # Re-enable submit button
        self.prompt_input.submit_btn.setEnabled(True)
        self.prompt_input.submit_btn.setText("🚀 Analyze Prompt")

        # Display the response
        self.response_display.display_response(response_data)

        # Clean up worker
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def on_response_error(self, error_message: str):
        """Handle error from backend"""
        print(f"Error processing prompt: {error_message}")

        # Re-enable submit button
        self.prompt_input.submit_btn.setEnabled(True)
        self.prompt_input.submit_btn.setText("🚀 Analyze Prompt")

        # Show error response
        error_response = {
            "response": f"❌ Error: {error_message}",
            "modules": [],
            "structured_prompt": "Error occurred",
            "arbitration_log": [
                f"[ERROR] {error_message}"
            ],
            "metadata": {
                "tokens": 0,
                "time_ms": 0,
                "model": "error",
                "timestamp": ""
            }
        }
        self.response_display.display_response(error_response)

        # Clean up worker
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
