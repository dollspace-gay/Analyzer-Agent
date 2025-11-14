"""
Test GPU detection and llama-cpp-python integration.
"""

import sys
import os

# Test GPU detection function
print("="*60)
print("GPU DETECTION TEST")
print("="*60)

# Import the detect_gpu_layers function
from protocol_ai import detect_gpu_layers

print("\n1. Testing GPU detection...")
gpu_layers = detect_gpu_layers()
print(f"   Result: {gpu_layers} layers")

if gpu_layers > 0:
    print("   [OK] GPU detected and will be used")
elif gpu_layers == -1:
    print("   [OK] GPU detected - all layers will be offloaded")
else:
    print("   [INFO] CPU mode (no GPU detected or low VRAM)")

# Test LLMInterface initialization
print("\n2. Testing LLMInterface initialization...")
from protocol_ai import LLMInterface

try:
    # Test with auto-detection
    llm_auto = LLMInterface(
        model_path="./test_model.gguf",  # Dummy path for testing init
        gpu_layers=None  # Should auto-detect
    )
    print(f"   Auto-detected layers: {llm_auto.gpu_layers}")
    print(f"   CPU threads: {llm_auto.n_threads}")
    print("   [OK] Auto-detection works")

    # Test with manual override
    llm_manual = LLMInterface(
        model_path="./test_model.gguf",
        gpu_layers=20
    )
    print(f"   Manual override layers: {llm_manual.gpu_layers}")
    print("   [OK] Manual override works")

except Exception as e:
    print(f"   [ERROR] Error: {e}")
    import traceback
    traceback.print_exc()

# Test import check
print("\n3. Testing llama-cpp-python import...")
try:
    from llama_cpp import Llama
    print("   [OK] llama-cpp-python is installed")
except ImportError:
    print("   [ERROR] llama-cpp-python is NOT installed")
    print("   Install with: pip install llama-cpp-python")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
