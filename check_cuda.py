#!/usr/bin/env python3
"""Check if llama-cpp-python has CUDA support"""

import sys

print("Checking llama-cpp-python CUDA support...")
print("=" * 50)

try:
    import llama_cpp
    print(f"[OK] llama-cpp-python version: {llama_cpp.__version__}")
except ImportError as e:
    print(f"[FAIL] Failed to import llama-cpp-python: {e}")
    sys.exit(1)

# Check if CUDA functions are available
try:
    from llama_cpp.llama_cpp import llama_supports_gpu_offload
    supports_gpu = llama_supports_gpu_offload()
    print(f"[OK] llama_supports_gpu_offload(): {supports_gpu}")

    if not supports_gpu:
        print("\n[WARNING] GPU offload is NOT supported!")
        print("llama-cpp-python was not compiled with CUDA support.")
        print("\nTo fix this, reinstall with CUDA:")
        print('  $env:CMAKE_ARGS="-DGGML_CUDA=on"')
        print('  pip uninstall llama-cpp-python -y')
        print('  pip install llama-cpp-python --no-cache-dir --force-reinstall')
    else:
        print("\n[OK] CUDA support is enabled!")

except Exception as e:
    print(f"[FAIL] Error checking GPU support: {e}")
    print("\nThis likely means CUDA support was not compiled in.")

# Try to check for CUDA libraries
print("\n" + "=" * 50)
print("Checking for CUDA runtime...")
try:
    import torch
    print(f"[OK] PyTorch installed: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("[INFO] PyTorch not installed (not required for llama-cpp-python)")

print("\n" + "=" * 50)
print("Testing actual model loading with GPU...")
try:
    from llama_cpp import Llama
    print("Creating Llama instance with n_gpu_layers=1 (test)...")
    # This should fail fast if CUDA isn't supported
    print("(This is just a test - will fail if model doesn't exist, that's OK)")
    test_model = Llama(
        model_path="nonexistent.gguf",
        n_gpu_layers=1,
        n_ctx=512,
        verbose=True
    )
except FileNotFoundError:
    print("[OK] File not found (expected), but CUDA initialization didn't error")
    print("[OK] This suggests CUDA support is working")
except Exception as e:
    error_str = str(e).lower()
    if 'cuda' in error_str or 'gpu' in error_str:
        print(f"[FAIL] CUDA-related error: {e}")
        print("This suggests CUDA support is NOT working properly")
    else:
        print(f"  Other error (might be OK): {e}")

print("\n" + "=" * 50)
print("Summary:")
print("If you see 'GPU offload is NOT supported', you need to reinstall llama-cpp-python with CUDA.")
