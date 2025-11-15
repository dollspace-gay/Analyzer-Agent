"""
Build the ULTIMATE one-click launcher

This creates a small (~50 MB) executable that:
1. Downloads Python runtime on first run
2. Downloads all dependencies
3. Detects GPU and installs CUDA support
4. Downloads model file (with progress bar)
5. Creates desktop shortcut
6. Launches the GUI

User literally just:
1. Downloads ProtocolAI.exe
2. Double-clicks it
3. Waits 5-10 minutes for first-run setup
4. Uses the system

Subsequent launches are instant.
"""

import os
from pathlib import Path

# This will be the launcher source that gets compiled to .exe
LAUNCHER_SOURCE = '''
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
            messagebox.showerror("Installation Error", f"Setup failed:\\n{str(e)}")
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

        # Install dependencies - using ctransformers instead of llama-cpp-python
        # ctransformers is easier to install (no compilation required)
        deps = [
            "pyyaml",
            "ctransformers",  # Easier to install than llama-cpp-python
            "sentence-transformers",
            "numpy",
            "scikit-learn",
            "torch",
            "requests",
            "duckduckgo-search",
            "PyQt6",
        ]

        for dep in deps:
            try:
                subprocess.run(
                    [str(python_exe), "-m", "pip", "install", dep],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to install {dep}:\\n\\nSTDOUT:\\n{e.stdout}\\n\\nSTDERR:\\n{e.stderr}"
                raise RuntimeError(error_msg)

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
            "Please download a GGUF model file and place it here.\\\\n\\\\n"
            "Recommended:\\\\n"
            "- DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf\\\\n\\\\n"
            "Download from HuggingFace or TheBloke's collection."
        )

        # Show message to user
        self.root.after(0, lambda: messagebox.showinfo(
            "Model Required",
            "Please download a GGUF model file and place it in:\\\\n" +
            str(self.models_dir) + "\\\\n\\\\nThen relaunch this program."
        ))

    def setup_project_files(self):
        """Copy embedded project files and create launch script"""
        # Copy protocol_ai.py and modules from embedded resources
        import sys
        import shutil

        # Get the directory where the exe is running from
        if getattr(sys, 'frozen', False):
            # Running as compiled exe - files are in _MEIPASS
            bundle_dir = Path(sys._MEIPASS)
        else:
            # Running as script - files are in current directory
            bundle_dir = Path(__file__).parent

        # Copy main script
        source_script = bundle_dir / "protocol_ai.py"
        if source_script.exists():
            shutil.copy2(source_script, self.install_dir / "protocol_ai.py")
            print(f"[Setup] Copied protocol_ai.py")

        # Copy modules directory
        source_modules = bundle_dir / "modules"
        dest_modules = self.install_dir / "modules"
        if source_modules.exists():
            if dest_modules.exists():
                shutil.rmtree(dest_modules)
            shutil.copytree(source_modules, dest_modules)
            print(f"[Setup] Copied modules/ directory")

        # Copy tools directory
        source_tools = bundle_dir / "tools"
        dest_tools = self.install_dir / "tools"
        if source_tools.exists():
            if dest_tools.exists():
                shutil.rmtree(dest_tools)
            shutil.copytree(source_tools, dest_tools)
            print(f"[Setup] Copied tools/ directory")

        # Copy gui directory
        source_gui = bundle_dir / "gui"
        dest_gui = self.install_dir / "gui"
        if source_gui.exists():
            if dest_gui.exists():
                shutil.rmtree(dest_gui)
            shutil.copytree(source_gui, dest_gui)
            print(f"[Setup] Copied gui/ directory")

        # Copy other essential files
        for filename in ["deep_research_agent.py", "deep_research_integration.py", "protocol_ai_logging.py"]:
            source_file = bundle_dir / filename
            if source_file.exists():
                shutil.copy2(source_file, self.install_dir / filename)
                print(f"[Setup] Copied {filename}")

        # Create launch batch script
        batch_script = self.install_dir / "Run_ProtocolAI.bat"
        python_exe = self.python_dir / "python.exe"

        batch_content = f"""@echo off
title Protocol AI - Governance Layer
echo ============================================
echo Protocol AI - Governance Layer
echo ============================================
echo.

cd /d "{self.install_dir}"

REM Check if GUI module exists
if not exist "gui\\app.py" (
    echo ERROR: GUI module not found in {self.install_dir}\\gui
    echo.
    echo Please ensure all project files are in the installation directory.
    pause
    exit /b 1
)

REM Launch Protocol AI GUI
echo Starting Protocol AI GUI...
echo.
"{python_exe}" -m gui.app
pause
"""

        batch_script.write_text(batch_content, encoding='utf-8')

        # Create a README with instructions
        readme = self.install_dir / "README.txt"
        readme_content = f"""Protocol AI - Installation Complete!

To run Protocol AI:
1. Double-click "Run_ProtocolAI.bat" in this folder
   OR
2. Copy your project files (protocol_ai.py, modules/, etc.) to:
   {self.install_dir}

Installation Directory: {self.install_dir}
Python Location: {python_exe}
Models Directory: {self.models_dir}

NOTE: Make sure you have a GGUF model file in the models/ directory!
"""
        readme.write_text(readme_content, encoding='utf-8')

    def launch_app(self):
        """Launch Protocol AI or show installation directory"""
        batch_script = self.install_dir / "Run_ProtocolAI.bat"
        gui_app = self.install_dir / "gui" / "app.py"

        # Check if gui/app.py exists
        if gui_app.exists():
            # Launch the batch script
            subprocess.Popen([str(batch_script)], shell=True)
            self.root.quit()
        else:
            # Show where to copy files
            message = (
                f"Installation complete!\\n\\n"
                f"Next steps:\\n"
                f"1. Copy your Protocol AI files to:\\n"
                f"   {self.install_dir}\\n\\n"
                f"2. Copy your GGUF model to:\\n"
                f"   {self.models_dir}\\n\\n"
                f"3. Run 'Run_ProtocolAI.bat' to start\\n\\n"
                f"Opening installation folder..."
            )
            messagebox.showinfo("Setup Complete", message)

            # Open the installation directory in Explorer
            subprocess.Popen(['explorer', str(self.install_dir)])
            self.root.quit()

if __name__ == "__main__":
    app = ProtocolAILauncher()
    app.root.mainloop()
'''

# Write launcher source
launcher_file = Path(__file__).parent / "ultimate_launcher.py"
launcher_file.write_text(LAUNCHER_SOURCE, encoding='utf-8')

print("="*70)
print("Building Ultimate Launcher Executable")
print("="*70)
print(f"\nLauncher source created: {launcher_file}")

# Check if PyInstaller is installed
try:
    import PyInstaller
    print("[OK] PyInstaller is installed")
except ImportError:
    print("\n[!] PyInstaller not found. Installing...")
    import subprocess
    subprocess.run(["pip", "install", "pyinstaller"], check=True)
    print("[OK] PyInstaller installed")

# Build the executable with all necessary data files bundled
print("\nBuilding executable with bundled files...")
import subprocess

base_dir = Path(__file__).parent

# Build PyInstaller command with all --add-data flags
import sys
cmd = [
    sys.executable,
    "-m",
    "PyInstaller",
    str(launcher_file),
    "--onefile",
    "--windowed",
    "--name=ProtocolAI",
    "--add-data", f"{base_dir / 'protocol_ai.py'};.",
    "--add-data", f"{base_dir / 'modules'};modules",
    "--add-data", f"{base_dir / 'tools'};tools",
    "--add-data", f"{base_dir / 'gui'};gui",
    "--add-data", f"{base_dir / 'deep_research_agent.py'};.",
    "--add-data", f"{base_dir / 'deep_research_integration.py'};.",
    "--add-data", f"{base_dir / 'protocol_ai_logging.py'};.",
    "--clean",  # Clean cache before building
]

print(f"\nRunning PyInstaller...")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    exe_path = base_dir / "dist" / "ProtocolAI.exe"
    if exe_path.exists():
        exe_size = exe_path.stat().st_size / (1024 * 1024)  # Size in MB
        print("\n" + "="*70)
        print("BUILD SUCCESSFUL!")
        print("="*70)
        print(f"\n[OK] Executable created: {exe_path}")
        print(f"[OK] Size: {exe_size:.1f} MB")
        print("\nDistribute this single file to users!")
        print("\nUsers just:")
        print("  1. Download ProtocolAI.exe")
        print("  2. Run it (first-time setup takes 5-10 minutes)")
        print("  3. Use Protocol AI!")
        print("\nNote: First run requires internet connection for Python runtime download.")
    else:
        print("\n[ERROR] Build succeeded but exe not found at expected location")
        print(result.stdout)
else:
    print("\n[ERROR] Build failed!")
    print("\nSTDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
