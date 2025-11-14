# Protocol AI - Quick Start Guide

**Get up and running in 2 minutes!**

## Option 1: Double-Click Launcher (Easiest)

### First Time Setup:

1. **Install Python 3.9+** (if not already installed)
   - Download from: https://www.python.org/downloads/
   - ✅ **IMPORTANT:** Check "Add Python to PATH" during installation

2. **Download Model** (if not already downloaded)
   - Place your GGUF model file in `F:\Agent\DeepSeek-R1\` folder
   - Or update the model path in GUI settings

3. **Run Protocol AI:**
   - Double-click `run_protocol_ai.bat`
   - First run will install dependencies (takes 2-3 minutes)
   - GUI will launch automatically

### Create Desktop Shortcut:

**Option A - PowerShell (Recommended):**
1. Right-click `create_desktop_shortcut.ps1`
2. Select "Run with PowerShell"
3. Double-click "Protocol AI" icon on desktop

**Option B - Manual:**
1. Right-click `run_protocol_ai.bat`
2. Select "Send to" → "Desktop (create shortcut)"
3. Rename to "Protocol AI"

---

## Option 2: Command Line

```bash
# Install dependencies
pip install -r requirements.txt

# Run GUI
cd gui
python app.py
```

---

## Option 3: Python Script

```bash
python run_analysis.py "Analyze OpenAI"
```

---

## System Requirements

- **Python:** 3.9 or higher
- **RAM:** 16GB minimum (32GB recommended for 8B models)
- **GPU:** NVIDIA GPU with CUDA support (optional but recommended)
- **Storage:** 10GB for model files + dependencies

---

## Default Configuration

- **Model Path:** `F:/Agent/DeepSeek-R1/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf`
- **Modules:** 71 modules loaded from `./modules/`
- **Deep Research:** Enabled by default (50+ sources per query)
- **Context Size:** 13,000 tokens
- **Output Format:** Standardized 7-section reports with checksums

---

## First Run

When you first launch Protocol AI:

1. **Dependencies Install** - Takes 2-3 minutes
2. **Model Loading** - Takes 30-60 seconds
3. **GUI Opens** - Ready to use!

The system will:
- ✅ Load all 71 governance modules
- ✅ Initialize deep research system (RAG + multi-source gathering)
- ✅ Enable cadence neutralization and affective firewall
- ✅ Configure output cleaning (no web context in reports)

---

## Usage Examples

### In GUI:
1. Type your query: `Analyze OpenAI`
2. Click "Analyze" or press Enter
3. Wait for deep research (10 queries across 50+ sources)
4. Receive clean 7-section report

### Command Line:
```bash
python run_analysis.py "Analyze effective altruism"
```

---

## Troubleshooting

### "Python is not installed or not in PATH"
- Reinstall Python with "Add to PATH" checked
- OR manually add Python to system PATH

### "Failed to load model"
- Check model path in settings
- Ensure GGUF file exists at specified location
- Try a different quantization (Q4_K_M, Q8_0, etc.)

### "Module import errors"
- Run: `pip install -r requirements.txt --upgrade`
- Restart the launcher

### GUI won't open
- Check console output for errors
- Ensure no other instance is running
- Try: `cd gui && python app.py` directly

---

## Files Overview

- **run_protocol_ai.bat** - Main launcher (double-click this)
- **create_desktop_shortcut.ps1** - Creates desktop shortcut
- **protocol_ai.py** - Core orchestration engine
- **modules/** - 71 YAML governance modules
- **gui/** - Graphical interface
- **requirements.txt** - Python dependencies

---


**You're ready to go! Double-click `run_protocol_ai.bat` to start analyzing.**
