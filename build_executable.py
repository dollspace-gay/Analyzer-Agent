"""
Build standalone Protocol AI executable with PyInstaller

This creates a single .exe file with ALL dependencies bundled:
- Python runtime
- All pip packages
- CUDA libraries
- GUI assets

The user just double-clicks the .exe - no installation needed.
"""

import PyInstaller.__main__
import shutil
import os
from pathlib import Path

# Get absolute paths
ROOT = Path(__file__).parent.absolute()
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"

print("="*70)
print("Building Protocol AI Standalone Executable")
print("="*70)

# Clean previous builds
if DIST_DIR.exists():
    shutil.rmtree(DIST_DIR)
if BUILD_DIR.exists():
    shutil.rmtree(BUILD_DIR)

# PyInstaller configuration
PyInstaller.__main__.run([
    'gui/app.py',  # Main entry point
    '--name=ProtocolAI',
    '--onefile',  # Single executable
    '--windowed',  # No console window (GUI only)
    '--icon=gui/icon.ico' if os.path.exists('gui/icon.ico') else '',

    # Add all data files
    '--add-data=modules;modules',  # All YAML modules
    '--add-data=analytical_frameworks.txt;.',
    '--add-data=gui;gui',  # GUI assets
    '--add-data=tools;tools',  # Tools directory

    # Hidden imports (packages not auto-detected)
    '--hidden-import=yaml',
    '--hidden-import=llama_cpp',
    '--hidden-import=sentence_transformers',
    '--hidden-import=sklearn',
    '--hidden-import=numpy',
    '--hidden-import=torch',
    '--hidden-import=transformers',
    '--hidden-import=requests',
    '--hidden-import=duckduckgo_search',

    # Collect CUDA libraries
    '--collect-all=torch',
    '--collect-all=llama_cpp',

    # Optimization
    '--clean',
    '--noconfirm',

    # Output directory
    f'--distpath={DIST_DIR}',
    f'--workpath={BUILD_DIR}',
])

print("\n" + "="*70)
print("Build Complete!")
print("="*70)
print(f"\nExecutable location: {DIST_DIR / 'ProtocolAI.exe'}")
print("\nNext steps:")
print("1. Copy the executable to desired location")
print("2. Ensure model file is in the same directory or configure path")
print("3. Double-click ProtocolAI.exe to run")
