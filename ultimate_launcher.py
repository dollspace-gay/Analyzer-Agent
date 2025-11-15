
import sys
import os
import urllib.request
import zipfile
import subprocess
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import threading

class ProtocolAILauncher:
    """
    Ultimate one-click launcher for Protocol AI
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Protocol AI - First Run Setup")
        self.root.geometry("600x400")
        self.root.resizable(False, False)

        # Get installation directory (where exe is located)
        if getattr(sys, 'frozen', False):
            self.install_dir = Path(sys.executable).parent
        else:
            self.install_dir = Path(__file__).parent

        self.python_dir = self.install_dir / "runtime" / "python"
        self.models_dir = self.install_dir / "models"
        self.config_file = self.install_dir / "config.json"

        self.setup_ui()
        self.check_installation()

    def setup_ui(self):
        """Create the UI"""
        # Header
        header = tk.Label(
            self.root,
            text="Protocol AI - Governance Layer",
            font=("Arial", 18, "bold"),
            pady=20
        )
        header.pack()

        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Checking installation...",
            font=("Arial", 11),
            pady=10
        )
        self.status_label.pack()

        # Progress bar
        self.progress = ttk.Progressbar(
            self.root,
            length=500,
            mode='determinate'
        )
        self.progress.pack(pady=20)

        # Detail label
        self.detail_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 9),
            fg="gray"
        )
        self.detail_label.pack()

        # Launch button (initially disabled)
        self.launch_button = tk.Button(
            self.root,
            text="Launch Protocol AI",
            command=self.launch_app,
            state=tk.DISABLED,
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10
        )
        self.launch_button.pack(pady=30)

    def update_status(self, message, detail="", progress=0):
        """Update UI with current status"""
        self.status_label.config(text=message)
        self.detail_label.config(text=detail)
        self.progress['value'] = progress
        self.root.update()

    def check_installation(self):
        """Check if installation is complete"""
        python_exe = self.python_dir / "python.exe"

        if python_exe.exists() and self.models_dir.exists():
            # Already installed
            self.update_status("✓ Installation complete!", "Ready to launch", 100)
            self.launch_button.config(state=tk.NORMAL)
        else:
            # Need to install
            self.update_status("First-run setup required", "This will take 5-10 minutes", 0)
            self.root.after(2000, self.start_installation)

    def start_installation(self):
        """Start installation in background thread"""
        thread = threading.Thread(target=self.install_all, daemon=True)
        thread.start()

    def install_all(self):
        """Install everything needed"""
        try:
            # Step 1: Download Python
            self.update_status("Downloading Python runtime...", "~25 MB", 10)
            self.download_python()

            # Step 2: Install pip
            self.update_status("Setting up package manager...", "", 30)
            self.install_pip()

            # Step 3: Install dependencies
            self.update_status("Installing dependencies...", "This takes a few minutes", 40)
            self.install_dependencies()

            # Step 4: Download model
            self.update_status("Checking for model file...", "", 70)
            if not self.check_model():
                self.update_status("Downloading model file...", "~4-8 GB - please wait", 75)
                self.download_model()

            # Step 5: Copy project files
            self.update_status("Setting up Protocol AI...", "", 90)
            self.setup_project_files()

            # Done!
            self.update_status("✓ Installation complete!", "Ready to launch", 100)
            self.launch_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Installation Error", f"Setup failed:\n{str(e)}")
            self.root.quit()

    def download_python(self):
        """Download embedded Python"""
        python_url = "https://www.python.org/ftp/python/3.11.8/python-3.11.8-embed-amd64.zip"
        zip_path = self.install_dir / "python.zip"

        # Download with progress
        urllib.request.urlretrieve(python_url, zip_path)

        # Extract
        self.python_dir.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.python_dir)

        zip_path.unlink()

        # Enable site-packages
        pth_file = list(self.python_dir.glob("*._pth"))[0]
        content = pth_file.read_text()
        content = content.replace('#import site', 'import site')
        pth_file.write_text(content)

    def install_pip(self):
        """Install pip"""
        get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
        get_pip = self.install_dir / "get-pip.py"
        urllib.request.urlretrieve(get_pip_url, get_pip)

        python_exe = self.python_dir / "python.exe"
        subprocess.run([str(python_exe), str(get_pip)], check=True, capture_output=True)
        get_pip.unlink()

    def install_dependencies(self):
        """Install all Python dependencies"""
        python_exe = self.python_dir / "python.exe"

        # Install llama-cpp-python first with pre-built wheels (CPU version)
        # This avoids compilation which requires Visual Studio Build Tools
        subprocess.run(
            [str(python_exe), "-m", "pip", "install", "llama-cpp-python",
             "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu",
             "--quiet"],
            check=True,
            capture_output=True
        )

        # Install other dependencies
        deps = [
            "pyyaml",
            "sentence-transformers",
            "numpy",
            "scikit-learn",
            "torch",
            "requests",
            "duckduckgo-search",
            "PyQt6",
        ]

        for dep in deps:
            subprocess.run(
                [str(python_exe), "-m", "pip", "install", dep, "--quiet"],
                check=True,
                capture_output=True
            )

    def check_model(self):
        """Check if model file exists"""
        self.models_dir.mkdir(exist_ok=True)
        models = list(self.models_dir.glob("*.gguf"))
        return len(models) > 0

    def download_model(self):
        """Download a model file (placeholder - user must provide)"""
        # For now, just create a README
        readme = self.models_dir / "DOWNLOAD_MODEL.txt"
        readme.write_text(
            "Please download a GGUF model file and place it here.\\n\\n"
            "Recommended:\\n"
            "- DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf\\n\\n"
            "Download from HuggingFace or TheBloke's collection."
        )

        # Show message to user
        self.root.after(0, lambda: messagebox.showinfo(
            "Model Required",
            "Please download a GGUF model file and place it in:\\n" +
            str(self.models_dir) + "\\n\\nThen relaunch this program."
        ))

    def setup_project_files(self):
        """Copy embedded project files"""
        # Project files are embedded in the exe by PyInstaller
        # This would copy them to the installation directory
        pass

    def launch_app(self):
        """Launch the main Protocol AI GUI"""
        python_exe = self.python_dir / "python.exe"
        app_script = self.install_dir / "gui" / "app.py"

        if app_script.exists():
            subprocess.Popen([str(python_exe), str(app_script)])
            self.root.quit()
        else:
            messagebox.showerror("Error", "Application files not found!")

if __name__ == "__main__":
    app = ProtocolAILauncher()
    app.root.mainloop()
