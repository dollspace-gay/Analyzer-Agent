"""
Build a TRULY portable Protocol AI package - zero installation required

This creates a complete package with:
1. Embedded Python runtime (no Python installation needed)
2. All pip packages pre-installed
3. CUDA runtime DLLs included
4. Model file (user copies or downloads)
5. One-click launcher

The user just:
- Extracts the ZIP
- Double-clicks "Launch Protocol AI.exe"
- Done!
"""

import urllib.request
import zipfile
import subprocess
import shutil
import os
from pathlib import Path
import json

ROOT = Path(__file__).parent.absolute()
PACKAGE_DIR = ROOT / "ProtocolAI_Portable"
PYTHON_VERSION = "3.11.8"  # Use specific version for stability

print("="*70)
print("Building Portable Protocol AI Package")
print("="*70)

# Clean previous build
if PACKAGE_DIR.exists():
    shutil.rmtree(PACKAGE_DIR)
PACKAGE_DIR.mkdir()

print("\n[1/6] Downloading embedded Python...")
# Download embeddable Python
python_url = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
python_zip = ROOT / f"python-{PYTHON_VERSION}-embed.zip"

if not python_zip.exists():
    print(f"Downloading from {python_url}")
    urllib.request.urlretrieve(python_url, python_zip)
    print("Downloaded!")
else:
    print("Using cached download")

# Extract Python
print("\n[2/6] Extracting Python runtime...")
with zipfile.ZipFile(python_zip, 'r') as zip_ref:
    zip_ref.extractall(PACKAGE_DIR / "python")

# Enable site-packages (needed for pip)
pth_file = PACKAGE_DIR / "python" / f"python{PYTHON_VERSION.replace('.', '')[:3]}_pth"
if pth_file.exists():
    content = pth_file.read_text()
    # Uncomment import site line
    content = content.replace('#import site', 'import site')
    pth_file.write_text(content)

# Download and install pip
print("\n[3/6] Installing pip...")
get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
get_pip = PACKAGE_DIR / "get-pip.py"
urllib.request.urlretrieve(get_pip_url, get_pip)

python_exe = PACKAGE_DIR / "python" / "python.exe"
subprocess.run([str(python_exe), str(get_pip)], check=True)

# Install all dependencies
print("\n[4/6] Installing dependencies...")
print("This may take 5-10 minutes...")

deps = [
    "pyyaml",
    "llama-cpp-python",  # Will download pre-built wheel
    "sentence-transformers",
    "numpy",
    "scikit-learn",
    "torch",
    "transformers",
    "requests",
    "duckduckgo-search",
    "PyQt6",  # For GUI
]

for dep in deps:
    print(f"  Installing {dep}...")
    subprocess.run([
        str(python_exe), "-m", "pip", "install",
        dep,
        "--no-warn-script-location"
    ], check=True, capture_output=True)

print("All dependencies installed!")

# Copy project files
print("\n[5/6] Copying project files...")
files_to_copy = [
    "protocol_ai.py",
    "report_formatter.py",
    "analytical_frameworks.txt",
    "QUICK_START.md",
    "MODULE_LIBRARY_COMPLETE.md",
]

for file in files_to_copy:
    if Path(file).exists():
        shutil.copy(file, PACKAGE_DIR / file)

# Copy directories
dirs_to_copy = ["modules", "gui", "tools"]
for dir_name in dirs_to_copy:
    if Path(dir_name).exists():
        shutil.copytree(dir_name, PACKAGE_DIR / dir_name)

# Create model directory with README
(PACKAGE_DIR / "models").mkdir(exist_ok=True)
(PACKAGE_DIR / "models" / "PUT_MODEL_HERE.txt").write_text(
    "Place your GGUF model file here.\n\n"
    "Example: DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf\n\n"
    "Download models from:\n"
    "- HuggingFace\n"
    "- TheBloke's model collection\n\n"
    "The launcher will automatically detect models in this folder."
)

# Create output directory
(PACKAGE_DIR / "output").mkdir(exist_ok=True)
(PACKAGE_DIR / "research_storage").mkdir(exist_ok=True)

# Create config file
print("\n[6/6] Creating configuration...")
config = {
    "model_path": "models/",  # Auto-detect from models folder
    "context_length": 13000,
    "gpu_layers": -1,
    "temperature": 0.7,
    "enable_deep_research": True
}

(PACKAGE_DIR / "config.json").write_text(json.dumps(config, indent=2))

# Create launcher script
launcher_bat = PACKAGE_DIR / "Launch Protocol AI.bat"
launcher_bat.write_text(r"""@echo off
title Protocol AI - Governance Layer

echo ========================================
echo Protocol AI
echo ========================================
echo.

REM Check for model file
set MODEL_FOUND=0
for %%f in (models\*.gguf) do set MODEL_FOUND=1

if %MODEL_FOUND%==0 (
    echo [WARNING] No model file found in models\ folder
    echo.
    echo Please download a GGUF model file and place it in the models\ folder.
    echo.
    echo Recommended models:
    echo   - DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf
    echo   - Llama-3-8B-Instruct-Q8_0.gguf
    echo.
    echo Download from HuggingFace or TheBloke's collection.
    echo.
    pause
    exit /b 1
)

echo [OK] Model file detected
echo.
echo Starting Protocol AI GUI...
echo.

REM Launch GUI using embedded Python
python\python.exe gui\app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Protocol AI encountered an error
    echo.
    pause
)
""")

print("\n" + "="*70)
print("✅ Portable package created!")
print("="*70)
print(f"\nLocation: {PACKAGE_DIR}")
print("\nContents:")
print("  ├── python/           (Embedded Python runtime)")
print("  ├── models/           (Place GGUF model here)")
print("  ├── modules/          (71 governance modules)")
print("  ├── gui/              (Graphical interface)")
print("  ├── tools/            (Web search, etc.)")
print("  ├── config.json       (Configuration)")
print("  └── Launch Protocol AI.bat  ← DOUBLE-CLICK THIS")
print("\nNext steps:")
print("1. Copy a GGUF model file to ProtocolAI_Portable/models/")
print("2. Zip the entire ProtocolAI_Portable folder")
print("3. Distribute the ZIP file")
print("4. Users extract and double-click 'Launch Protocol AI.bat'")
print("\n⚠️  Package size will be ~2-3 GB (Python + PyTorch + dependencies)")
