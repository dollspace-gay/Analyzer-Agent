# Installing llama-cpp-python with CUDA Support

## Windows Installation (CUDA)

The protocol_ai.py system has been updated to use llama-cpp-python instead of ctransformers for better CUDA support on Windows.

### Prerequisites
- NVIDIA GPU with CUDA support (detected: RTX 3060 12GB)
- CUDA Toolkit 11.8 or later
- Visual Studio 2019/2022 with C++ build tools

### Installation Steps

#### Option 1: Pre-built Wheels (Recommended)
```bash
# For CUDA 12.x
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# For CUDA 11.x
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118
```

#### Option 2: Build from Source (Advanced)
```bash
# Set environment variables for CUDA support
set CMAKE_ARGS=-DGGML_CUDA=on
set FORCE_CMAKE=1

# Install
pip install llama-cpp-python --no-cache-dir
```

### Verification

Run the GPU detection test:
```bash
python test_gpu_detection.py
```

Expected output:
```
GPU detected with 12.0GB VRAM
Result: -1 layers
[OK] GPU detected - all layers will be offloaded
[OK] llama-cpp-python is installed
```

### Features

The updated LLMInterface now includes:

1. **Automatic GPU Detection**
   - Detects NVIDIA GPU using nvidia-smi
   - Automatically calculates optimal layer offloading based on VRAM
   - Returns -1 (all layers) for GPUs with 12GB+ VRAM

2. **Manual Override**
   - Set `gpu_layers=0` for CPU-only mode
   - Set `gpu_layers=20` for manual layer control
   - Set `gpu_layers=-1` to force all layers to GPU

3. **Auto-threading**
   - Automatically detects CPU cores and sets optimal thread count
   - Can be manually overridden with `n_threads` parameter

### Usage Example

```python
from protocol_ai import LLMInterface

# Auto-detect GPU (recommended)
llm = LLMInterface(
    model_path="./models/model.gguf",
    gpu_layers=None  # Auto-detect
)

# Manual GPU configuration
llm = LLMInterface(
    model_path="./models/model.gguf",
    gpu_layers=35  # Offload 35 layers to GPU
)

# CPU-only mode
llm = LLMInterface(
    model_path="./models/model.gguf",
    gpu_layers=0  # Use CPU only
)
```

## Troubleshooting

### CUDA not found
If you get CUDA errors, ensure:
1. NVIDIA drivers are up to date (560.94+)
2. CUDA toolkit is installed
3. Environment variables are set correctly

### Import errors
```python
# Check if installation succeeded
python -c "from llama_cpp import Llama; print('OK')"
```

### Performance issues
- Ensure GPU drivers are updated
- Monitor GPU usage with `nvidia-smi`
- Adjust `gpu_layers` if running out of VRAM

## Migration from ctransformers

The new implementation is API-compatible. Key changes:
- `model_type` parameter removed (auto-detected)
- `gpu_layers=None` enables auto-detection
- Response format changed from string to dict (handled internally)

No code changes needed for existing scripts!
