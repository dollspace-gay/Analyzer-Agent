"""
Protocol AI - Governance Layer for AI System

This module implements a deterministic Python program that orchestrates a local LLM.
It takes user prompts, selects appropriate modules from a library, and constructs
highly detailed final prompts to send to the local LLM.

Architecture:
- ModuleLoader: Loads YAML module definitions from filesystem
- TriggerEngine: Analyzes prompts and identifies active modules via keyword matching
- Orchestrator: Main pipeline orchestrating trigger analysis, hierarchical arbitration, and LLM execution
- LLMInterface: Manages local LLM inference using llama-cpp-python library

Author: AI Governance Layer System
Python Version: 3.9+
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

try:
    from ctransformers import AutoModelForCausalLM
except ImportError:
    AutoModelForCausalLM = None

# Report formatter and checksum tools
try:
    from report_formatter import ReportFormatter
except ImportError:
    ReportFormatter = None
    print("[Warning] ReportFormatter not available - standardized reports disabled")

try:
    import requests
except ImportError:
    requests = None
    print("[Warning] requests library not available - web search disabled")

try:
    import struct
except ImportError:
    struct = None


# =============================================================================
# GGUF Metadata Parser (Memory Optimization - Phase 1)
# =============================================================================

@dataclass
class GGUFMetadata:
    """
    Metadata extracted from GGUF model file.

    Contains architecture information needed for precise memory calculations.
    """
    architecture: str = "unknown"
    layer_count: int = 32  # Default for 7B models
    parameter_count: int = 7_000_000_000  # Default 7B
    quantization_type: str = "Q4_K_M"  # Default quantization
    context_length: int = 4096
    embedding_length: int = 4096
    head_count: int = 32
    head_count_kv: int = 32

    # Calculated fields
    bytes_per_parameter: float = 0.0
    estimated_layer_size_mb: float = 0.0


class GGUFMetadataParser:
    """
    Parse GGUF model files to extract architecture metadata.

    Theory: GGUF files contain metadata that specifies:
    - Model architecture (llama, mistral, qwen, etc.)
    - Layer count (varies by model size)
    - Quantization format (affects memory per parameter)
    - Context window and embedding dimensions

    This allows precise memory footprint calculation instead of guessing.
    """

    # Quantization format to bytes-per-parameter mapping
    # Based on GGUF specification and empirical measurements
    QUANT_SIZES = {
        'F32': 4.0,      # Full precision (4 bytes per param)
        'F16': 2.0,      # Half precision (2 bytes per param)
        'Q8_0': 1.0,     # 8-bit quantization (~1 byte per param)
        'Q6_K': 0.78,    # 6-bit (~0.78 bytes per param)
        'Q5_K_M': 0.67,  # 5-bit medium (~0.67 bytes per param)
        'Q5_K_S': 0.67,  # 5-bit small
        'Q4_K_M': 0.54,  # 4-bit medium (~0.54 bytes per param)
        'Q4_K_S': 0.54,  # 4-bit small
        'Q4_0': 0.55,    # 4-bit round 0
        'Q3_K_M': 0.43,  # 3-bit medium
        'Q3_K_S': 0.43,  # 3-bit small
        'Q2_K': 0.35,    # 2-bit
        'IQ4_XS': 0.53,  # Importance-matrix 4-bit extra small
        'IQ3_XXS': 0.40, # Importance-matrix 3-bit double extra small
        'IQ2_XXS': 0.28, # Importance-matrix 2-bit double extra small
    }

    # Model size heuristics (parameter count to typical layer count)
    LAYER_COUNT_HEURISTICS = {
        (0, 2_000_000_000): 16,        # < 2B: 16 layers
        (2_000_000_000, 4_000_000_000): 24,  # 2-4B: 24 layers
        (4_000_000_000, 9_000_000_000): 32,  # 4-9B (7B/8B): 32 layers
        (9_000_000_000, 16_000_000_000): 40, # 9-16B (13B/14B): 40 layers
        (16_000_000_000, 50_000_000_000): 60, # 16-50B (32B): 60 layers
        (50_000_000_000, float('inf')): 80,   # 50B+ (70B): 80 layers
    }

    def __init__(self, model_path: str):
        """
        Initialize parser with model path.

        Args:
            model_path: Path to GGUF model file
        """
        self.model_path = model_path
        self.metadata = GGUFMetadata()

    def parse(self) -> GGUFMetadata:
        """
        Parse GGUF file and extract metadata.

        Returns:
            GGUFMetadata object with extracted information
        """
        if not os.path.exists(self.model_path):
            print(f"[GGUF Parser] Model file not found: {self.model_path}")
            return self._get_fallback_metadata()

        try:
            # Try to extract from filename first (fast path)
            self._parse_from_filename()

            # Try to read GGUF header (more accurate but requires file I/O)
            # Note: Full GGUF parsing is complex, so we use heuristics + filename
            self._estimate_parameters()

            # Calculate derived values
            self._calculate_memory_footprint()

            print(f"[GGUF Parser] Detected: {self.metadata.architecture}, "
                  f"{self.metadata.layer_count} layers, {self.metadata.quantization_type}")

            return self.metadata

        except Exception as e:
            print(f"[GGUF Parser] Parsing failed: {e}, using fallback heuristics")
            return self._get_fallback_metadata()

    def _parse_from_filename(self):
        """
        Extract metadata from filename patterns.

        Common patterns:
        - llama-2-7b-chat.Q4_K_M.gguf
        - mistral-7b-instruct-v0.2.Q5_K_M.gguf
        - qwen2.5-14b-instruct-q4_k_m.gguf
        """
        filename = os.path.basename(self.model_path).lower()

        # Extract quantization type
        for quant_type in self.QUANT_SIZES.keys():
            if quant_type.lower() in filename:
                self.metadata.quantization_type = quant_type
                break

        # Extract model size (parameter count)
        size_patterns = [
            ('1b', 1_000_000_000),
            ('3b', 3_000_000_000),
            ('7b', 7_000_000_000),
            ('8b', 8_000_000_000),
            ('13b', 13_000_000_000),
            ('14b', 14_000_000_000),
            ('32b', 32_000_000_000),
            ('70b', 70_000_000_000),
        ]

        for pattern, params in size_patterns:
            if pattern in filename:
                self.metadata.parameter_count = params
                break

        # Extract architecture
        if 'llama' in filename:
            self.metadata.architecture = 'llama'
        elif 'mistral' in filename:
            self.metadata.architecture = 'mistral'
        elif 'qwen' in filename:
            self.metadata.architecture = 'qwen'
        elif 'phi' in filename:
            self.metadata.architecture = 'phi'
        elif 'gemma' in filename:
            self.metadata.architecture = 'gemma'
        else:
            self.metadata.architecture = 'unknown'

    def _estimate_parameters(self):
        """
        Estimate layer count based on parameter count.

        Uses empirical heuristics for standard transformer architectures.
        """
        params = self.metadata.parameter_count

        for (min_params, max_params), layer_count in self.LAYER_COUNT_HEURISTICS.items():
            if min_params <= params < max_params:
                self.metadata.layer_count = layer_count
                break

    def _calculate_memory_footprint(self):
        """
        Calculate memory footprint per layer.

        Formula: layer_size_mb = (params_per_layer * bytes_per_param) / 1024^2

        Where params_per_layer ≈ total_params / layer_count
        """
        # Get bytes per parameter for this quantization
        self.metadata.bytes_per_parameter = self.QUANT_SIZES.get(
            self.metadata.quantization_type,
            0.54  # Default to Q4_K_M
        )

        # Estimate parameters per layer
        # (Note: This is approximate as not all parameters are in layers)
        params_per_layer = self.metadata.parameter_count / self.metadata.layer_count

        # Calculate layer size in MB
        bytes_per_layer = params_per_layer * self.metadata.bytes_per_parameter
        self.metadata.estimated_layer_size_mb = bytes_per_layer / (1024 * 1024)

    def _get_fallback_metadata(self) -> GGUFMetadata:
        """
        Return conservative fallback metadata if parsing fails.

        Assumes 7B Q4_K_M model (most common case).
        """
        fallback = GGUFMetadata()
        fallback.architecture = "unknown"
        fallback.layer_count = 32
        fallback.parameter_count = 7_000_000_000
        fallback.quantization_type = "Q4_K_M"
        fallback.bytes_per_parameter = 0.54
        fallback.estimated_layer_size_mb = 150.0  # Conservative estimate

        print("[GGUF Parser] Using fallback metadata (7B Q4_K_M)")
        return fallback


# =============================================================================
# Elastic Layer Offloading (Memory Optimization - Phase 2)
# =============================================================================

def calculate_optimal_layers(
    model_path: str,
    vram_gb: float,
    context_window: int = 8192,
    kv_cache_type: str = "q8_0"
) -> tuple[int, Dict[str, Any]]:
    """
    Mathematically determines the maximum number of layers that fit in VRAM.

    Theory: Total_VRAM = (Model_Weights) + (KV_Cache) + (CUDA_Overhead)

    This replaces hardcoded heuristics with calculated memory budgeting based on:
    1. Precise model architecture from GGUF metadata
    2. KV cache memory requirements (varies with context window and cache quantization)
    3. CUDA system overhead (driver, kernels, etc.)

    Args:
        model_path: Path to GGUF model file
        vram_gb: Available VRAM in gigabytes
        context_window: Context window size (affects KV cache size)
        kv_cache_type: KV cache quantization ("fp16", "q8_0", or "q4_0")

    Returns:
        Tuple of (optimal_layer_count, debug_info_dict)
        - optimal_layer_count: Maximum layers that fit in VRAM
        - debug_info_dict: Detailed memory breakdown for logging
    """

    # Parse GGUF metadata
    parser = GGUFMetadataParser(model_path)
    metadata = parser.parse()

    # KV cache bytes per token (depends on quantization)
    # Formula: 2 * n_layers * d_model * (bytes_per_element)
    # Simplified: tokens * layers * bytes_per_token_per_layer
    kv_cache_bytes_per_token_per_layer = {
        'fp16': 0.0005,  # ~0.5 KB per token per layer (FP16)
        'q8_0': 0.00025, # ~0.25 KB per token per layer (Q8_0, 50% reduction)
        'q4_0': 0.000125 # ~0.125 KB per token per layer (Q4_0, 75% reduction)
    }

    bytes_per_token = kv_cache_bytes_per_token_per_layer.get(kv_cache_type.lower(), 0.00025)

    # CUDA overhead (driver, kernel memory, etc.)
    # Larger models need more overhead due to larger intermediate buffers
    if metadata.parameter_count > 30_000_000_000:  # 30B+
        cuda_overhead_mb = 1200
    elif metadata.parameter_count > 10_000_000_000:  # 10-30B
        cuda_overhead_mb = 1000
    else:  # < 10B
        cuda_overhead_mb = 800

    # Calculate KV cache budget for full context window
    # Note: KV cache grows with each layer offloaded
    kv_cache_mb_per_layer = (context_window * bytes_per_token * 1024)  # MB per layer

    # Convert available VRAM to MB
    available_vram_mb = vram_gb * 1024

    # Calculate available VRAM for model layers (after overhead)
    vram_for_layers_mb = available_vram_mb - cuda_overhead_mb

    if vram_for_layers_mb <= 0:
        # Not enough VRAM even for overhead
        debug_info = {
            'total_vram_mb': available_vram_mb,
            'cuda_overhead_mb': cuda_overhead_mb,
            'available_for_layers_mb': 0,
            'model_layer_size_mb': metadata.estimated_layer_size_mb,
            'kv_cache_mb_per_layer': kv_cache_mb_per_layer,
            'optimal_layers': 0,
            'reason': 'Insufficient VRAM for GPU mode'
        }
        return 0, debug_info

    # Calculate how many layers fit
    # Formula: layers = (available_vram - overhead) / (layer_size + kv_cache_per_layer)
    memory_per_layer = metadata.estimated_layer_size_mb + kv_cache_mb_per_layer
    layers_fitting = int(vram_for_layers_mb / memory_per_layer)

    # Apply 10% safety margin (reserve 10% of calculated capacity)
    layers_with_margin = int(layers_fitting * 0.9)

    # Clamp to valid range [0, total_layers]
    optimal_layers = max(0, min(layers_with_margin, metadata.layer_count))

    # Build debug info for diagnostic output
    debug_info = {
        'model_path': os.path.basename(model_path),
        'architecture': metadata.architecture,
        'total_layers': metadata.layer_count,
        'quantization': metadata.quantization_type,
        'parameter_count': metadata.parameter_count,
        'total_vram_mb': available_vram_mb,
        'cuda_overhead_mb': cuda_overhead_mb,
        'available_for_layers_mb': vram_for_layers_mb,
        'model_layer_size_mb': round(metadata.estimated_layer_size_mb, 2),
        'kv_cache_type': kv_cache_type,
        'kv_cache_mb_per_layer': round(kv_cache_mb_per_layer, 2),
        'context_window': context_window,
        'memory_per_layer_mb': round(memory_per_layer, 2),
        'calculated_layers': layers_fitting,
        'optimal_layers_with_margin': optimal_layers,
        'safety_margin_percent': 10
    }

    return optimal_layers, debug_info


def detect_gpu_layers(model_path: str = "", context_window: int = 8192, kv_cache_type: str = "q8_0") -> int:
    """
    Automatically detect available GPU and return recommended number of layers.

    Uses nvidia-smi to detect GPU VRAM and calculates appropriate layer count.
    If model_path is provided, uses precise calculation via calculate_optimal_layers().
    Otherwise, uses enhanced heuristics.

    Args:
        model_path: Optional path to GGUF model file (for precise calculation)
        context_window: Context window size (for KV cache calculation)
        kv_cache_type: KV cache quantization type ("fp16", "q8_0", "q4_0")

    Returns:
        int: Number of layers to offload to GPU (0 for CPU-only mode, -1 for all layers)
    """
    try:
        # Try to run nvidia-smi to detect GPU
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            vram_mb = int(result.stdout.strip().split('\n')[0])
            vram_gb = vram_mb / 1024

            print(f"[GPU Detection] {vram_gb:.1f}GB VRAM detected")

            # If model path provided: use precise calculation
            if model_path and os.path.exists(model_path):
                print(f"[GPU Detection] Using precise calculation for {os.path.basename(model_path)}")
                optimal_layers, debug_info = calculate_optimal_layers(
                    model_path=model_path,
                    vram_gb=vram_gb,
                    context_window=context_window,
                    kv_cache_type=kv_cache_type
                )

                # Log memory breakdown
                print(f"[Memory Budget] Total VRAM: {debug_info['total_vram_mb']:.0f}MB")
                print(f"[Memory Budget] CUDA Overhead: {debug_info['cuda_overhead_mb']}MB")
                print(f"[Memory Budget] Available for layers: {debug_info['available_for_layers_mb']:.0f}MB")
                print(f"[Memory Budget] Model: {debug_info['architecture']} "
                      f"{debug_info['quantization']} ({debug_info['total_layers']} layers)")
                print(f"[Memory Budget] Layer size: {debug_info['model_layer_size_mb']}MB")
                print(f"[Memory Budget] KV cache ({debug_info['kv_cache_type']}): "
                      f"{debug_info['kv_cache_mb_per_layer']}MB/layer")
                print(f"[Memory Budget] Calculated layers: {debug_info['calculated_layers']} "
                      f"(with 10% margin: {optimal_layers})")

                return optimal_layers

            # Otherwise: use enhanced heuristics
            print("[GPU Detection] No model path provided, using enhanced heuristics")

            # Enhanced heuristics: smarter than before, but still approximate
            # These account for typical 7B-14B Q4_K_M models
            if vram_gb >= 24:
                return -1  # 24GB+: Offload all layers for any model up to 14B
            elif vram_gb >= 16:
                return -1  # 16GB: All layers for 7B/8B, most for 14B
            elif vram_gb >= 12:
                return 40  # 12GB: Fits 14B Q4 comfortably
            elif vram_gb >= 10:
                return 35  # 10GB: Fits most of 14B Q4
            elif vram_gb >= 8:
                return 33  # 8GB: Fits 8B Q4 completely
            elif vram_gb >= 6:
                return 25  # 6GB: Partial offload for 7B/8B
            elif vram_gb >= 4:
                return 15  # 4GB: Minimal offload
            else:
                return 0  # <4GB: CPU only

        else:
            print("[GPU Detection] No NVIDIA GPU detected, using CPU mode")
            return 0

    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError) as e:
        print(f"[GPU Detection] GPU detection failed ({e}), defaulting to CPU mode")
        return 0


# =============================================================================
# KV Cache Quantization (Memory Optimization - Phase 3)
# =============================================================================

def select_kv_cache_type(vram_gb: float) -> str:
    """
    Auto-select optimal KV cache quantization based on available VRAM.

    Theory: KV cache quantization reduces memory usage with minimal quality loss:
    - FP16: Maximum quality, highest memory (baseline)
    - Q8_0: 50% memory savings, <1% perplexity increase
    - Q4_0: 75% memory savings, ~2-3% perplexity increase

    Strategy:
    - High VRAM (>=16GB): Use FP16 for maximum quality
    - Medium VRAM (8-16GB): Use Q8_0 for balanced memory/quality
    - Low VRAM (<8GB): Use Q4_0 for maximum memory savings

    Args:
        vram_gb: Available VRAM in gigabytes

    Returns:
        KV cache type string: "fp16", "q8_0", or "q4_0"
    """
    if vram_gb >= 16:
        print(f"[KV Cache] {vram_gb:.1f}GB VRAM → FP16 cache (maximum quality)")
        return "fp16"
    elif vram_gb >= 8:
        print(f"[KV Cache] {vram_gb:.1f}GB VRAM → Q8_0 cache (2x memory savings, <1% quality loss)")
        return "q8_0"
    else:
        print(f"[KV Cache] {vram_gb:.1f}GB VRAM → Q4_0 cache (4x memory savings, ~2-3% quality loss)")
        return "q4_0"


@dataclass
class Module:
    """
    Represents a single module loaded from YAML definition.

    Attributes:
        name: Module identifier
        tier: Hierarchical tier level (lower number = higher priority)
        purpose: Description of module's function
        triggers: List of keyword triggers that activate this module
        prompt_template: Template text to inject into final prompt
        dependencies: List of module names that must execute before this module
        metadata: Additional module configuration data
    """
    name: str
    tier: int
    purpose: str
    triggers: List[str]
    prompt_template: str
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Bundle:
    """
    Represents a module bundle configuration.

    Bundles pre-activate groups of modules for specific contexts (e.g., coding, analysis).
    They also provide configuration overrides and bundle-specific instructions.

    Attributes:
        name: Bundle identifier
        version: Bundle version string
        description: Human-readable description
        purpose: Intended use case
        active_modules: List of module names to pre-activate
        bundle_instructions: Additional prompt text when bundle is active
        configuration: Configuration overrides for trigger sensitivity, output, etc.
        metadata: Additional bundle data
    """
    name: str
    version: str
    description: str
    purpose: str
    active_modules: List[Dict[str, Any]]  # List of {name, tier, reason}
    bundle_instructions: str = ""
    configuration: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BundleLoader:
    """
    Loads and manages bundle configurations from YAML files.

    Bundles allow pre-activation of module groups for specific contexts
    (e.g., governance analysis, coding, red team testing).
    """

    def __init__(self, bundles_dir: str = "./bundles"):
        """
        Initialize the BundleLoader.

        Args:
            bundles_dir: Path to directory containing bundle YAML files
        """
        self.bundles_dir = Path(bundles_dir)
        self.bundles: Dict[str, Bundle] = {}

    def load_bundles(self) -> Dict[str, Bundle]:
        """
        Load all YAML bundle files from the bundles directory.

        Returns:
            Dictionary mapping bundle names to Bundle objects

        Raises:
            FileNotFoundError: If bundles directory doesn't exist
        """
        if not self.bundles_dir.exists():
            raise FileNotFoundError(f"Bundles directory not found: {self.bundles_dir}")

        self.bundles.clear()

        # Find all .yaml and .yml files
        yaml_files = list(self.bundles_dir.glob("*.yaml")) + list(self.bundles_dir.glob("*.yml"))

        for yaml_file in yaml_files:
            try:
                bundle = self._load_bundle_file(yaml_file)
                if bundle:
                    # Use filename (without extension) as bundle key
                    bundle_key = yaml_file.stem
                    self.bundles[bundle_key] = bundle
            except Exception as e:
                print(f"Warning: Failed to load bundle from {yaml_file}: {e}")
                continue

        return self.bundles

    def _load_bundle_file(self, file_path: Path) -> Optional[Bundle]:
        """
        Load a single bundle from a YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            Bundle object or None if parsing fails
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data or 'bundle_metadata' not in data:
            return None

        metadata = data['bundle_metadata']

        return Bundle(
            name=metadata.get('name', file_path.stem),
            version=metadata.get('version', '1.0.0'),
            description=metadata.get('description', ''),
            purpose=metadata.get('purpose', ''),
            active_modules=data.get('active_modules', []),
            bundle_instructions=data.get('bundle_instructions', ''),
            configuration=data.get('configuration', {}),
            metadata=metadata
        )

    def get_bundle(self, bundle_name: str) -> Optional[Bundle]:
        """
        Get a specific bundle by name.

        Args:
            bundle_name: Name/key of the bundle

        Returns:
            Bundle object or None if not found
        """
        return self.bundles.get(bundle_name)

    def list_bundles(self) -> List[str]:
        """
        Get list of available bundle names.

        Returns:
            List of bundle name strings
        """
        return list(self.bundles.keys())


class DependencyResolver:
    """
    Resolves module dependencies and orders modules for execution.

    Handles:
    - Recursive dependency resolution
    - Cycle detection
    - Topological sorting for execution order
    """

    def __init__(self, all_modules: List[Module]):
        """
        Initialize the DependencyResolver.

        Args:
            all_modules: Complete list of all available modules
        """
        self.all_modules = all_modules
        self.module_map = {m.name: m for m in all_modules}

    def resolve(self, triggered_modules: List[Module]) -> List[Module]:
        """
        Resolve dependencies for triggered modules and return in execution order.

        Args:
            triggered_modules: List of modules that were triggered

        Returns:
            List of modules in dependency order (dependencies first)

        Raises:
            ValueError: If circular dependency is detected or dependency not found
        """
        # Build set of all required module names (including dependencies)
        required_names = set()
        visiting = set()  # For cycle detection
        visited = set()

        for module in triggered_modules:
            self._resolve_recursive(module, required_names, visiting, visited)

        # Convert names to modules and topological sort
        required_modules = [self.module_map[name] for name in required_names]
        sorted_modules = self._topological_sort(required_modules)

        return sorted_modules

    def _resolve_recursive(self, module: Module, required_names: set,
                          visiting: set, visited: set):
        """
        Recursively resolve module dependencies using DFS.

        Args:
            module: Current module to resolve
            required_names: Set to accumulate all required module names
            visiting: Set for cycle detection (currently in DFS stack)
            visited: Set of fully processed modules

        Raises:
            ValueError: If circular dependency is detected
        """
        if module.name in visited:
            return

        if module.name in visiting:
            raise ValueError(f"Circular dependency detected involving module '{module.name}'")

        visiting.add(module.name)

        # Resolve each dependency
        for dep_name in module.dependencies:
            if dep_name not in self.module_map:
                raise ValueError(f"Module '{module.name}' depends on '{dep_name}' which is not loaded")

            dep_module = self.module_map[dep_name]
            self._resolve_recursive(dep_module, required_names, visiting, visited)

        visiting.remove(module.name)
        visited.add(module.name)
        required_names.add(module.name)

    def _topological_sort(self, modules: List[Module]) -> List[Module]:
        """
        Sort modules in dependency order using topological sort.

        Dependencies appear before modules that depend on them.

        Args:
            modules: List of modules to sort

        Returns:
            Sorted list of modules
        """
        # Build adjacency list and in-degree count
        graph = {m.name: [] for m in modules}
        in_degree = {m.name: 0 for m in modules}
        module_map = {m.name: m for m in modules}

        for module in modules:
            for dep_name in module.dependencies:
                if dep_name in graph:  # Only consider dependencies in our set
                    graph[dep_name].append(module.name)
                    in_degree[module.name] += 1

        # Kahn's algorithm for topological sort
        queue = [name for name in in_degree if in_degree[name] == 0]
        sorted_names = []

        while queue:
            # Sort queue to ensure deterministic order
            queue.sort()
            current = queue.pop(0)
            sorted_names.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Convert names back to modules
        return [module_map[name] for name in sorted_names]

    def validate_all(self) -> List[str]:
        """
        Validate all module dependencies are satisfied.

        Returns:
            List of error messages (empty if all valid)
        """
        errors = []

        for module in self.all_modules:
            for dep_name in module.dependencies:
                if dep_name not in self.module_map:
                    errors.append(
                        f"Module '{module.name}' depends on '{dep_name}' which is not loaded"
                    )

        # Check for cycles
        checked_modules = set()
        for module in self.all_modules:
            # Skip if already checked (avoid duplicate error messages)
            if module.name in checked_modules:
                continue

            try:
                required_names = set()
                visiting = set()
                visited = set()
                self._resolve_recursive(module, required_names, visiting, visited)
                checked_modules.update(visited)
            except ValueError as e:
                error_msg = str(e)
                if error_msg not in errors:  # Avoid duplicate error messages
                    errors.append(error_msg)

        return errors


class ModuleLoader:
    """
    Loads and manages module definitions from YAML files.

    Scans a directory structure (e.g., ./modules/) for YAML files containing
    module definitions. Each module specifies its tier, triggers, and prompt templates.
    """

    def __init__(self, modules_dir: str = "./modules"):
        """
        Initialize the ModuleLoader.

        Args:
            modules_dir: Path to directory containing module YAML files
        """
        self.modules_dir = Path(modules_dir)
        self.modules: List[Module] = []

    def load_modules(self) -> List[Module]:
        """
        Recursively load all YAML module files from the modules directory.

        Returns:
            List of Module objects loaded from YAML definitions

        Raises:
            FileNotFoundError: If modules directory doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if not self.modules_dir.exists():
            raise FileNotFoundError(f"Modules directory not found: {self.modules_dir}")

        self.modules.clear()

        # Recursively find all .yaml and .yml files
        yaml_files = list(self.modules_dir.rglob("*.yaml")) + list(self.modules_dir.rglob("*.yml"))

        for yaml_file in yaml_files:
            try:
                module = self._load_module_file(yaml_file)
                if module:
                    self.modules.append(module)
            except Exception as e:
                print(f"Warning: Failed to load module from {yaml_file}: {e}")
                continue

        return self.modules

    def _load_module_file(self, file_path: Path) -> Optional[Module]:
        """
        Load a single module from a YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            Module object or None if parsing fails
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        # Extract tier from directory structure (e.g., tier2, tier4)
        tier = self._extract_tier_from_path(file_path)

        return Module(
            name=data.get('name', file_path.stem),
            tier=tier,
            purpose=data.get('purpose', ''),
            triggers=data.get('triggers', []),
            prompt_template=data.get('prompt_template', ''),
            dependencies=data.get('dependencies', []),
            metadata=data.get('metadata', {})
        )

    def _extract_tier_from_path(self, file_path: Path) -> int:
        """
        Extract tier number from directory path.

        Looks for 'tierN' in path components. Defaults to tier 99 if not found.

        Args:
            file_path: Path to module file

        Returns:
            Tier number (integer)
        """
        for part in file_path.parts:
            if part.startswith('tier'):
                try:
                    return int(part.replace('tier', ''))
                except ValueError:
                    pass
        return 99  # Default tier for modules without explicit tier


class TriggerEngine:
    """
    Analyzes user prompts and determines which modules should be activated.

    Uses advanced NLP techniques including:
    - Simple keyword matching (baseline)
    - Stemming (word roots: "manipulate", "manipulating" -> "manipul")
    - Lemmatization (base forms: "better" -> "good")
    - Synonym matching (WordNet synonyms: "deceptive" matches "fraudulent")

    Configurable matching modes for different precision/recall tradeoffs.
    """

    def __init__(self, matching_mode: str = "advanced", enable_synonyms: bool = True):
        """
        Initialize the TriggerEngine.

        Args:
            matching_mode:
                - "simple": Basic keyword matching only
                - "stemmed": Keyword + stemming
                - "advanced": Keyword + stemming + lemmatization (default)
            enable_synonyms: Whether to use synonym expansion (default: True)
        """
        self.matching_mode = matching_mode
        self.enable_synonyms = enable_synonyms

        # Initialize NLP components
        self._init_nlp_components()

    def _init_nlp_components(self):
        """Initialize NLTK components for advanced matching"""
        try:
            import nltk
            from nltk.stem import PorterStemmer, WordNetLemmatizer
            from nltk.corpus import wordnet

            self.nltk_available = True
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
            self.wordnet = wordnet

            # Try to download required NLTK data silently
            try:
                nltk.data.find('corpora/wordnet.zip')
            except LookupError:
                print("[TriggerEngine] Downloading NLTK WordNet data...")
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)  # Open Multilingual WordNet

            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)

            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger', quiet=True)

        except ImportError:
            print("[TriggerEngine] WARNING: NLTK not available. Using simple matching only.")
            print("[TriggerEngine] Install with: pip install nltk")
            self.nltk_available = False
            self.matching_mode = "simple"

    def _get_synonyms(self, word: str) -> set:
        """
        Get synonyms for a word using WordNet.

        Args:
            word: Word to find synonyms for

        Returns:
            Set of synonyms (lemmas)
        """
        if not self.enable_synonyms or not self.nltk_available:
            return set()

        synonyms = set()

        try:
            for syn in self.wordnet.synsets(word):
                for lemma in syn.lemmas():
                    # Add the lemma name (synonym)
                    synonym = lemma.name().replace('_', ' ').lower()
                    synonyms.add(synonym)
        except Exception:
            pass  # Silently fail for words not in WordNet

        return synonyms

    def _normalize_text(self, text: str) -> List[str]:
        """
        Normalize text based on matching mode.

        Args:
            text: Text to normalize

        Returns:
            List of normalized tokens
        """
        import re

        # Tokenize (split on whitespace and punctuation)
        tokens = re.findall(r'\b\w+\b', text.lower())

        if not self.nltk_available or self.matching_mode == "simple":
            return tokens

        normalized = []

        if self.matching_mode == "stemmed":
            # Apply stemming
            for token in tokens:
                normalized.append(self.stemmer.stem(token))

        elif self.matching_mode == "advanced":
            # Apply lemmatization (more sophisticated than stemming)
            for token in tokens:
                # Lemmatize as noun, verb, adjective
                lemma_n = self.lemmatizer.lemmatize(token, pos='n')
                lemma_v = self.lemmatizer.lemmatize(token, pos='v')
                lemma_a = self.lemmatizer.lemmatize(token, pos='a')

                # Add all variations to catch different word forms
                normalized.extend([lemma_n, lemma_v, lemma_a, token])

        return normalized

    def _matches_trigger(self, prompt_tokens: List[str], trigger: str,
                         prompt_synonyms: set) -> bool:
        """
        Check if a trigger matches the prompt using configured matching mode.

        Args:
            prompt_tokens: Normalized tokens from prompt
            trigger: Trigger phrase to match
            prompt_synonyms: Set of all synonyms for words in prompt

        Returns:
            True if trigger matches
        """
        # Normalize trigger
        trigger_lower = trigger.lower()

        # 1. Simple substring match (fast path)
        if trigger_lower in ' '.join(prompt_tokens):
            return True

        # 2. Token-level matching with normalization
        trigger_tokens = self._normalize_text(trigger)

        # Check if any trigger token appears in prompt tokens
        for t_token in trigger_tokens:
            if t_token in prompt_tokens:
                return True

        # 3. Synonym matching (if enabled)
        if self.enable_synonyms and self.nltk_available:
            # Get synonyms for trigger
            trigger_words = trigger_lower.split()
            for trigger_word in trigger_words:
                trigger_syns = self._get_synonyms(trigger_word)

                # Check if any trigger synonym appears in prompt synonyms
                if trigger_syns & prompt_synonyms:  # Set intersection
                    return True

        return False

    def analyze_prompt(self, user_prompt: str, modules: List[Module]) -> List[Module]:
        """
        Analyze user prompt and return list of triggered modules.

        Uses advanced NLP techniques based on configured matching mode:
        - Stemming/lemmatization to match word variations
        - Synonym matching to catch semantically similar terms
        - Multi-word phrase matching

        Args:
            user_prompt: The user's input prompt string
            modules: List of available Module objects

        Returns:
            List of Module objects whose triggers match the prompt
        """
        active_modules = []

        # Normalize prompt
        prompt_tokens = self._normalize_text(user_prompt)

        # Get all synonyms for prompt words (if enabled)
        prompt_synonyms = set()
        if self.enable_synonyms and self.nltk_available:
            import re
            words = re.findall(r'\b\w+\b', user_prompt.lower())
            for word in words:
                prompt_synonyms.update(self._get_synonyms(word))

        # Check each module
        for module in modules:
            triggered = False

            # Check if any trigger matches
            for trigger in module.triggers:
                if self._matches_trigger(prompt_tokens, trigger, prompt_synonyms):
                    triggered = True
                    break

            if triggered:
                active_modules.append(module)

        return active_modules


# =============================================================================
# Activation-Aware Context Pruning (Memory Optimization - Phase 4)
# =============================================================================

class ContextWindowManager:
    """
    Manages context window with activation-aware pruning (Sink Token Protocol).

    Theory (StreamingLLM):
    Not all tokens in the context window are equally important. Research shows:
    1. "Sink Tokens" (first ~4 tokens) are critical for attention stability
    2. Recent tokens (last ~N tokens) contain current conversation state
    3. Middle tokens can be aggressively pruned with minimal coherence loss

    This implements a rolling buffer that preserves:
    - System prompt (governance layer instructions) - ALWAYS
    - Recent conversation (last N tokens) - ROLLING WINDOW

    When context fills, we prune the middle while keeping system prompt + recent context.
    This allows infinite-length conversations without crashing or losing coherence.
    """

    def __init__(
        self,
        max_context: int = 8192,
        system_prompt_tokens: int = 500,
        recent_window_tokens: int = 3500,
        trigger_threshold: float = 0.9
    ):
        """
        Initialize context window manager.

        Args:
            max_context: Maximum context window size
            system_prompt_tokens: Tokens to preserve from beginning (system prompt)
            recent_window_tokens: Tokens to preserve from end (recent conversation)
            trigger_threshold: Prune when context reaches this % of max (0.9 = 90%)
        """
        self.max_context = max_context
        self.system_prompt_tokens = system_prompt_tokens
        self.recent_window_tokens = recent_window_tokens
        self.trigger_threshold = trigger_threshold

        # Estimate tokens per character (rough approximation)
        self.chars_per_token = 4  # Average for English text

        print(f"[Context Manager] Initialized with {max_context} token limit")
        print(f"[Context Manager] Sink tokens (system prompt): {system_prompt_tokens}")
        print(f"[Context Manager] Recent window: {recent_window_tokens}")
        print(f"[Context Manager] Pruning threshold: {int(trigger_threshold * 100)}%")

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.

        Uses simple char/token ratio (4:1) as approximation.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        return len(text) // self.chars_per_token

    def should_prune(self, current_text: str) -> bool:
        """
        Check if context should be pruned.

        Args:
            current_text: Current context text

        Returns:
            True if context exceeds threshold and should be pruned
        """
        estimated_tokens = self.estimate_tokens(current_text)
        threshold_tokens = int(self.max_context * self.trigger_threshold)

        if estimated_tokens > threshold_tokens:
            print(f"[Context Manager] Context size: ~{estimated_tokens} tokens "
                  f"(threshold: {threshold_tokens})")
            print(f"[Context Manager] Triggering pruning...")
            return True

        return False

    def prune_context(self, full_text: str) -> str:
        """
        Prune context using sink token protocol.

        Preserves:
        1. System prompt (first N characters)
        2. Recent conversation (last M characters)

        Args:
            full_text: Full context text to prune

        Returns:
            Pruned context text
        """
        # Calculate character limits (tokens * chars_per_token)
        system_chars = self.system_prompt_tokens * self.chars_per_token
        recent_chars = self.recent_window_tokens * self.chars_per_token

        # Extract system prompt (beginning)
        system_prompt = full_text[:system_chars]

        # Extract recent conversation (end)
        recent_conversation = full_text[-recent_chars:]

        # Build pruned context
        pruned_text = system_prompt + "\n\n[... earlier conversation pruned for context length ...]\n\n" + recent_conversation

        original_tokens = self.estimate_tokens(full_text)
        pruned_tokens = self.estimate_tokens(pruned_text)

        print(f"[Context Manager] Pruned: {original_tokens} → {pruned_tokens} tokens "
              f"(saved ~{original_tokens - pruned_tokens} tokens)")
        print(f"[Context Manager] Preserved: System prompt + recent {self.recent_window_tokens} tokens")

        return pruned_text

    def manage_context(self, context_text: str) -> str:
        """
        Manage context: prune if needed, otherwise return as-is.

        Args:
            context_text: Input context text

        Returns:
            Managed context text (pruned if necessary)
        """
        if self.should_prune(context_text):
            return self.prune_context(context_text)
        return context_text


# =============================================================================
# OOM Retry Mechanism (Memory Optimization - Phase 5)
# =============================================================================

class OOMRetryHandler:
    """
    Handles Out-Of-Memory errors with graceful degradation.

    Theory: Memory calculations may be wrong due to:
    - Unknown CUDA overhead variations across drivers/GPUs
    - Model architecture quirks not captured in heuristics
    - Other processes using VRAM
    - Fragmentation

    Solution: Exponential backoff with layer reduction.
    If loading fails, retry with progressively fewer layers:
    - Attempt 1: 100% of calculated layers
    - Attempt 2: 75% of calculated layers (25% reduction)
    - Attempt 3: 50% of calculated layers (50% reduction)
    - Attempt 4: 0 layers (CPU fallback)
    """

    def __init__(self, max_retries: int = 3):
        """
        Initialize OOM retry handler.

        Args:
            max_retries: Maximum retry attempts before CPU fallback
        """
        self.max_retries = max_retries
        self.retry_multipliers = [1.0, 0.75, 0.50, 0.0]  # Layer reduction schedule

    def is_oom_error(self, exception: Exception) -> bool:
        """
        Detect if exception is an Out-Of-Memory error.

        Args:
            exception: Exception to check

        Returns:
            True if OOM error detected
        """
        oom_indicators = [
            'out of memory',
            'oom',
            'cuda out of memory',
            'cuda error',
            'cudamalloc failed',
            'allocation failed',
            'vram',
            'memory allocation'
        ]

        error_msg = str(exception).lower()
        return any(indicator in error_msg for indicator in oom_indicators)

    def calculate_retry_layers(self, original_layers: int, attempt: int) -> int:
        """
        Calculate reduced layer count for retry attempt.

        Args:
            original_layers: Original calculated layer count
            attempt: Current retry attempt (0-indexed)

        Returns:
            Reduced layer count for this attempt
        """
        if attempt >= len(self.retry_multipliers):
            return 0  # CPU fallback

        multiplier = self.retry_multipliers[attempt]
        reduced_layers = int(original_layers * multiplier)

        print(f"[OOM Handler] Retry attempt {attempt + 1}/{self.max_retries}")
        print(f"[OOM Handler] Reducing layers: {original_layers} → {reduced_layers} ({int(multiplier * 100)}%)")

        return reduced_layers

    def should_retry(self, attempt: int) -> bool:
        """
        Check if should retry after failure.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            True if should retry
        """
        return attempt < self.max_retries


class LLMInterface:
    """
    Interface to local LLM using llama-cpp-python library.

    Handles model loading, configuration, and inference execution
    for GGUF format models with automatic GPU detection and offloading.
    """

    def __init__(
        self,
        model_path: str,
        gpu_layers: Optional[int] = None,
        context_length: int = 4096,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        n_threads: Optional[int] = None,
        enable_optimizations: bool = True,
        kv_cache_type: Optional[str] = None
    ):
        """
        Initialize the LLM interface.

        Args:
            model_path: Path to GGUF model file
            gpu_layers: Number of layers to offload to GPU (None = auto-detect, 0 = CPU only, -1 = all layers)
            context_length: Maximum context window size
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            n_threads: Number of CPU threads (None = auto-detect)
            enable_optimizations: Enable memory optimizations (KV cache quantization, flash attention)
            kv_cache_type: KV cache quantization ("fp16", "q8_0", "q4_0", None = auto-detect)
        """
        self.model_path = model_path
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.n_threads = n_threads if n_threads is not None else max(1, os.cpu_count() // 2)
        self.enable_optimizations = enable_optimizations

        # Detect VRAM for optimization decisions
        self.vram_gb = self._detect_vram()

        # Auto-select KV cache type if not specified and optimizations enabled
        if enable_optimizations and kv_cache_type is None:
            self.kv_cache_type = select_kv_cache_type(self.vram_gb) if self.vram_gb > 0 else "q8_0"
        else:
            self.kv_cache_type = kv_cache_type if kv_cache_type else "q8_0"

        # Auto-detect GPU layers if not specified
        if gpu_layers is None:
            self.gpu_layers = detect_gpu_layers(
                model_path=model_path,
                context_window=context_length,
                kv_cache_type=self.kv_cache_type
            )
            print(f"[LLM Init] Auto-detected GPU layers: {self.gpu_layers}")
        else:
            self.gpu_layers = gpu_layers

        self.model: Optional[Llama] = None
        self.backend: str = ''  # Will be set to 'llama-cpp-python' or 'ctransformers' when model loads
        self.oom_handler = OOMRetryHandler(max_retries=3)  # OOM retry handler
        self.original_gpu_layers = self.gpu_layers  # Store original for retries

    def _detect_vram(self) -> float:
        """
        Detect available VRAM in GB.

        Returns:
            VRAM in GB, or 0 if no GPU detected
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                vram_mb = int(result.stdout.strip().split('\n')[0])
                return vram_mb / 1024
        except:
            pass
        return 0.0

    def load_model(self) -> None:
        """
        Load the LLM model from disk with automatic OOM retry.

        Implements graceful degradation:
        - Attempt 1: Try with calculated optimal layers
        - Attempt 2: Reduce to 75% layers on OOM
        - Attempt 3: Reduce to 50% layers on OOM
        - Attempt 4: Fall back to CPU-only on OOM

        Raises:
            FileNotFoundError: If model file doesn't exist
            ImportError: If neither library is installed
            Exception: If model loading fails after all retries
        """
        if Llama is None and AutoModelForCausalLM is None:
            raise ImportError(
                "Neither llama-cpp-python nor ctransformers is installed. "
                "Install one with: pip install llama-cpp-python OR pip install ctransformers"
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Retry loop with OOM handling
        attempt = 0
        last_exception = None

        while attempt <= self.oom_handler.max_retries:
            try:
                # Calculate layer count for this attempt
                if attempt > 0:
                    self.gpu_layers = self.oom_handler.calculate_retry_layers(
                        self.original_gpu_layers,
                        attempt
                    )

                # Attempt to load model
                self._attempt_model_load()

                # Success!
                if attempt > 0:
                    print(f"[OOM Handler] ✓ Model loaded successfully after {attempt} retry attempt(s)")
                return

            except Exception as e:
                # Check if this is an OOM error
                if self.oom_handler.is_oom_error(e):
                    print(f"[OOM Handler] ✗ Out of memory detected: {str(e)[:100]}")

                    if self.oom_handler.should_retry(attempt):
                        print(f"[OOM Handler] Retrying with reduced layers...")
                        attempt += 1
                        last_exception = e
                        continue
                    else:
                        print(f"[OOM Handler] Max retries exceeded, failed to load model")
                        raise RuntimeError(
                            f"Failed to load model after {self.oom_handler.max_retries} retry attempts. "
                            f"Last error: {str(e)}"
                        ) from e
                else:
                    # Non-OOM error, raise immediately
                    raise

        # Should never reach here, but just in case
        if last_exception:
            raise last_exception

    def _attempt_model_load(self) -> None:
        """
        Attempt to load the model with current configuration.

        This is the actual loading logic, separated for retry purposes.

        Raises:
            Exception: If model loading fails
        """
        print(f"[Model Loading] Loading model from {self.model_path}...")
        print(f"[Model Loading] Configuration:")
        print(f"  - GPU layers: {self.gpu_layers}")
        print(f"  - Context length: {self.context_length}")
        print(f"  - CPU threads: {self.n_threads}")
        print(f"  - Memory optimizations: {'Enabled' if self.enable_optimizations else 'Disabled'}")
        if self.enable_optimizations:
            print(f"  - KV cache type: {self.kv_cache_type.upper()}")
            print(f"  - Flash attention: Enabled")

        # Try llama-cpp-python first (preferred for existing installations)
        if Llama is not None:
            print("[Model Loading] Using llama-cpp-python backend")

            # Build model initialization parameters
            model_params = {
                'model_path': self.model_path,
                'n_gpu_layers': self.gpu_layers,
                'n_ctx': self.context_length,
                'n_threads': self.n_threads,
                'n_batch': 512,
                'use_mlock': False,
                'verbose': False
            }

            # Add optimization parameters if enabled
            if self.enable_optimizations and self.gpu_layers > 0:
                # Flash Attention: Drastically reduces memory consumption of attention mechanism
                model_params['flash_attn'] = True

                # KV Cache Quantization: Reduces context VRAM usage by 50-75%
                # Map our cache type to llama-cpp-python type strings
                cache_type_map = {
                    'fp16': 'f16',
                    'q8_0': 'q8_0',
                    'q4_0': 'q4_0'
                }
                cache_type = cache_type_map.get(self.kv_cache_type, 'q8_0')
                model_params['type_k'] = cache_type
                model_params['type_v'] = cache_type

                print(f"[Optimizations] Flash Attention: Enabled")
                print(f"[Optimizations] KV Cache: {cache_type} (saves memory for long contexts)")

            try:
                self.model = Llama(**model_params)
            except TypeError as e:
                # Fallback if flash_attn or type_k/type_v not supported in this version
                if 'flash_attn' in str(e) or 'type_k' in str(e) or 'type_v' in str(e):
                    print(f"[Warning] Optimizations not supported in this llama-cpp-python version, using defaults")
                    self.model = Llama(
                        model_path=self.model_path,
                        n_gpu_layers=self.gpu_layers,
                        n_ctx=self.context_length,
                        n_threads=self.n_threads,
                        n_batch=512,
                        use_mlock=False,
                        verbose=False
                    )
                else:
                    raise

            self.backend = 'llama-cpp-python'
        # Fallback to ctransformers
        elif AutoModelForCausalLM is not None:
            print("[Model Loading] Using ctransformers backend")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path_or_repo_id=self.model_path,
                model_type='llama',
                gpu_layers=self.gpu_layers,
                context_length=self.context_length,
                threads=self.n_threads,
                batch_size=512,
                stream=False
            )
            self.backend = 'ctransformers'

        print("[Model Loading] ✓ Model loaded successfully")

    def execute(self, prompt: str) -> str:
        """
        Execute inference on the loaded model.

        Args:
            prompt: The complete prompt to send to the LLM

        Returns:
            Generated text response from the model

        Raises:
            RuntimeError: If model hasn't been loaded yet
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Handle different backends
        if self.backend == 'llama-cpp-python':
            # llama-cpp-python returns a dict with choices
            response = self.model(
                prompt,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                echo=False
            )
            raw_output = response['choices'][0]['text']
        else:  # ctransformers
            # ctransformers returns text directly
            raw_output = self.model(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature
            )

        # Clean output: remove reasoning tokens and meta-commentary
        cleaned_output = self._clean_llm_output(raw_output)

        return cleaned_output

    def _clean_llm_output(self, text: str) -> str:
        """
        Clean LLM output by removing reasoning tokens, meta-commentary, and artifacts.

        Args:
            text: Raw LLM output

        Returns:
            Cleaned output ready for use
        """
        import re

        # ULTRA AGGRESSIVE: Strip ALL reasoning/thoughts and web context

        # 1. Remove ALL web context blocks (should NEVER be in output)
        # Remove [WEB CONTEXT] ... up to [Triggered Modules or SECTION
        text = re.sub(r'\[WEB CONTEXT\].*?(?=\[Triggered Modules|\*\*SECTION|SECTION)', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove [DEEP RESEARCH FINDINGS] ... up to [Triggered Modules or SECTION
        text = re.sub(r'\[DEEP RESEARCH FINDINGS\].*?(?=\[Triggered Modules|\*\*SECTION|SECTION)', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove [Web Context: ...] blocks
        text = re.sub(r'\[Web Context:.*?\].*?(?=\[Triggered Modules|\*\*SECTION|SECTION)', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove ## BACKGROUND sections (from deep research)
        text = re.sub(r'##\s*BACKGROUND.*?(?=\[Triggered Modules|\*\*SECTION|SECTION)', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove lines that start with URLs or look like web sources
        text = re.sub(r'^https?://.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*-\s*\[\d+\.\d+\].*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*Source:.*$', '', text, flags=re.MULTILINE)

        # 2. Remove explicit reasoning blocks (various formats)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\[/?INST\]', '', text)
        text = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', text, flags=re.DOTALL)

        # 3. Remove entire paragraphs that are meta-reasoning
        meta_reasoning_blocks = [
            r'(?i)assistant response:.*?(?=SECTION|\*\*SECTION|$)',
            r'(?i)assistant:.*?(?=SECTION|\*\*SECTION|$)',
            r'(?i)first,?\s+we\s+need\s+to.*?(?=SECTION|\*\*SECTION|$)',
            r'(?i)we\'ll\s+start\s+by.*?(?=SECTION|\*\*SECTION|$)',
            r'(?i)the\s+user.*?requested.*?(?=SECTION|\*\*SECTION|$)',
        ]
        for pattern in meta_reasoning_blocks:
            text = re.sub(pattern, '', text, flags=re.DOTALL)

        # 4. NUCLEAR CLEAN: Strip EVERYTHING before the actual report starts
        # Try multiple possible start markers in order of preference
        start_markers = [
            r'\[Triggered Modules:',  # Preferred start
            r'\*\*SECTION 1',          # Section header
            r'SECTION 1:',             # Plain section header
            r'## SECTION 1',           # Markdown section
        ]

        found_start = False
        for marker in start_markers:
            match = re.search(marker, text, re.IGNORECASE)
            if match:
                text = text[match.start():]
                found_start = True
                break

        # If no marker found, this is likely all garbage - search for ANY section header
        if not found_start:
            # Try to find any SECTION header
            any_section = re.search(r'(?:SECTION\s+\d+|\*\*SECTION\s+\d+)', text, re.IGNORECASE)
            if any_section:
                text = text[any_section.start():]

        # Remove meta-commentary patterns (line by line) - ULTRA AGGRESSIVE
        meta_patterns = [
            # Planning/process description
            r'(?i)^okay,?\s+i\s+can\s+do.*$',
            r'(?i)^let\'?s?\s+(begin|start|proceed|do|write|take).*$',
            r'(?i)^to\s+begin.*$',
            r'(?i)^first.*$',
            r'(?i)^we need to.*$',
            r'(?i)^we\'ll.*$',
            r'(?i)^we can.*$',
            r'(?i)^we should.*$',
            r'(?i)^we must.*$',
            r'(?i)^then\s+(section|we).*$',
            r'(?i)^next.*$',
            r'(?i)^now.*$',
            r'(?i)^but note:.*$',
            r'(?i)^however,.*$',
            r'(?i)^alternatively.*$',
            r'(?i)^given that.*$',
            r'(?i)^since the.*$',
            r'(?i)^from the background.*$',

            # Instructions to self (bullet points that are planning, not analysis)
            r'(?i)^-\s*(evaluate|identify|combine|analyze)\s+.*$',

            # Tool/action descriptions
            r'(?i)performing\s+web\s+search.*$',
            r'(?i)using\s+the\s+following.*$',
            r'(?i)searching for.*$',
            r'(?i)retrieving.*$',
            r'(?i)^the user query.*$',
            r'(?i)^in this (simulation|response).*$',
            r'(?i)^this (is|will be).*$',

            # Self-referential (I/me/my)
            r'(?i)^i\s+(will|must|need|should|can).*$',
            r'(?i)^let\s+me.*$',
            r'(?i)^here\'?s?\s+(my|the).*$',
            r'(?i)^i\'m\s+(going\s+to|supposed\s+to).*$',

            # Incomplete fragments
            r'^[A-Z]{2,}$',  # Single word in caps (like "AGI" alone on a line)

            # Markers
            r'\[END.*?\]\s*$',
            r'\[START.*?\]\s*$',
            r'---+\s*$',
        ]

        # Apply each pattern
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            should_keep = True
            for pattern in meta_patterns:
                if re.match(pattern, line.strip()):
                    should_keep = False
                    break
            if should_keep and line.strip():  # Only keep non-empty lines
                cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        # Remove empty execution log blocks
        text = re.sub(r'\[EXECUTION LOG\][\s\S]*?(?=\*\*SECTION|\[MODULE|SECTION|$)', '', text)

        # Clean up multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text


# ============================================================================
# TOOLS SYSTEM
# ============================================================================

@dataclass
class ToolResult:
    """
    Result from tool execution.

    Attributes:
        success: Whether tool executed successfully
        tool_name: Name of the tool that was executed
        output: The tool's output data
        error: Error message if execution failed
        execution_time: Time taken to execute (seconds)
        metadata: Additional metadata about execution
    """
    success: bool
    tool_name: str
    output: Any = None
    error: str = ""
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Tool:
    """
    Base class for all tools in the system.

    All tools must inherit from this class and implement the execute() method.
    Tools provide external capabilities to the LLM (web search, code execution, etc.).
    """

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        """
        Initialize the tool.

        Args:
            name: Unique identifier for the tool
            description: Human-readable description of what the tool does
            parameters: Schema defining expected input parameters
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.enabled = True

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.

        This method must be implemented by subclasses.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult: Result of tool execution

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(f"Tool '{self.name}' must implement execute() method")

    def validate_input(self, **kwargs) -> tuple[bool, str]:
        """
        Validate input parameters before execution.

        Args:
            **kwargs: Parameters to validate

        Returns:
            tuple: (is_valid, error_message)
        """
        # Check for required parameters
        required_params = [
            param_name for param_name, param_spec in self.parameters.items()
            if param_spec.get('required', False)
        ]

        for param in required_params:
            if param not in kwargs:
                return False, f"Missing required parameter: {param}"

        return True, ""

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the tool's schema for LLM consumption.

        Returns:
            dict: Tool schema including name, description, and parameters
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "enabled": self.enabled
        }


class ToolRegistry:
    """
    Registry for managing available tools.

    Handles tool registration, lookup, and execution coordination.
    """

    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, Tool] = {}
        self.execution_log: List[Dict[str, Any]] = []

    def register(self, tool: Tool) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")

        self.tools[tool.name] = tool
        print(f"[ToolRegistry] Registered tool: {tool.name}")

    def unregister(self, tool_name: str) -> None:
        """
        Unregister a tool from the registry.

        Args:
            tool_name: Name of tool to unregister
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            print(f"[ToolRegistry] Unregistered tool: {tool_name}")

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of tool to retrieve

        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(tool_name)

    def get_all_tools(self) -> List[Tool]:
        """
        Get all registered tools.

        Returns:
            List of all Tool instances
        """
        return list(self.tools.values())

    def get_enabled_tools(self) -> List[Tool]:
        """
        Get only enabled tools.

        Returns:
            List of enabled Tool instances
        """
        return [tool for tool in self.tools.values() if tool.enabled]

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """
        Get schemas for all enabled tools (for LLM consumption).

        Returns:
            List of tool schemas
        """
        return [tool.get_schema() for tool in self.get_enabled_tools()]

    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name with given parameters.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Parameters to pass to tool

        Returns:
            ToolResult: Result of execution
        """
        import time
        start_time = time.time()

        # Get tool
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' not found in registry",
                execution_time=0.0
            )

        # Check if tool is enabled
        if not tool.enabled:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' is currently disabled",
                execution_time=0.0
            )

        # Validate input
        is_valid, error_msg = tool.validate_input(**kwargs)
        if not is_valid:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Validation failed: {error_msg}",
                execution_time=0.0
            )

        # Execute tool
        try:
            result = await tool.execute(**kwargs)
            result.execution_time = time.time() - start_time

            # Log execution
            self._log_execution(tool_name, kwargs, result)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            result = ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Execution error: {str(e)}",
                execution_time=execution_time
            )

            # Log failed execution
            self._log_execution(tool_name, kwargs, result)

            return result

    def _log_execution(self, tool_name: str, params: Dict[str, Any],
                       result: ToolResult) -> None:
        """
        Log tool execution for audit trail.

        Args:
            tool_name: Name of executed tool
            params: Parameters passed to tool
            result: Execution result
        """
        import datetime

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "tool": tool_name,
            "parameters": params,
            "success": result.success,
            "execution_time": result.execution_time,
            "error": result.error if not result.success else None
        }

        self.execution_log.append(log_entry)

    def get_execution_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get tool execution log.

        Args:
            limit: Maximum number of recent entries to return

        Returns:
            List of log entries
        """
        if limit:
            return self.execution_log[-limit:]
        return self.execution_log

    def clear_execution_log(self) -> None:
        """Clear the execution log."""
        self.execution_log.clear()


class ToolLoader:
    """
    Loads tools from a directory structure.

    Discovers tool plugins and registers them with the ToolRegistry.
    """

    def __init__(self, tools_dir: str = "./tools"):
        """
        Initialize the tool loader.

        Args:
            tools_dir: Directory containing tool implementations
        """
        self.tools_dir = Path(tools_dir)

    def discover_tools(self) -> List[str]:
        """
        Discover available tool modules.

        Returns:
            List of tool module names
        """
        if not self.tools_dir.exists():
            print(f"[ToolLoader] Tools directory not found: {self.tools_dir}")
            return []

        # Find all Python files in tools directory (except __init__.py)
        tool_files = []
        for py_file in self.tools_dir.glob("*.py"):
            if py_file.name != "__init__.py":
                tool_files.append(py_file.stem)

        print(f"[ToolLoader] Discovered {len(tool_files)} tool modules")
        return tool_files

    def load_tools(self, registry: ToolRegistry) -> int:
        """
        Load all discovered tools into the registry.

        Args:
            registry: ToolRegistry to register tools into

        Returns:
            Number of tools successfully loaded
        """
        import importlib.util

        tool_modules = self.discover_tools()
        loaded_count = 0

        for module_name in tool_modules:
            try:
                # Load module
                module_path = self.tools_dir / f"{module_name}.py"
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Look for Tool subclasses in module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and
                            issubclass(attr, Tool) and
                            attr is not Tool):
                            # Instantiate and register tool
                            tool_instance = attr()
                            registry.register(tool_instance)
                            loaded_count += 1

            except Exception as e:
                print(f"[ToolLoader] Failed to load tool '{module_name}': {e}")

        print(f"[ToolLoader] Successfully loaded {loaded_count} tools")
        return loaded_count


@dataclass
class AuditResult:
    """
    Result of output audit check.

    Attributes:
        passed: Whether the audit check passed
        check_name: Name of the audit check
        reason: Explanation if check failed
        severity: Severity level (info, warning, error, critical)
    """
    passed: bool
    check_name: str
    reason: str = ""
    severity: str = "info"


class OutputAuditor:
    """
    Audits LLM output for compliance, safety, and quality.

    Performs post-generation checks including:
    - Instruction compliance verification
    - Refusal detection
    - Hallucination detection
    - Tier 1 safety compliance
    - Output quality validation
    """

    def __init__(self, modules: List[Module]):
        """
        Initialize the OutputAuditor.

        Args:
            modules: List of loaded modules for context
        """
        self.modules = modules
        self.tier1_modules = [m for m in modules if m.tier == 1]

        # Refusal patterns to detect
        self.refusal_patterns = [
            "i cannot",
            "i can't",
            "i'm unable to",
            "i am unable to",
            "i'm not able to",
            "i am not able to",
            "i don't have the ability",
            "i cannot assist with",
            "i cannot help with",
            "as an ai",
            "as a language model",
            "i'm just an ai",
            "i apologize, but i cannot"
        ]

        # Hallucination indicators - zero tolerance
        self.hallucination_patterns = [
            "[citation needed]",
            "[unverified]",
            "i don't actually know",
            # Removed overly strict patterns - these are normal analytical language:
            # "i'm not certain", "i may be incorrect", "i'm not sure",
            # "i could be wrong", "i think", "i believe", "probably", "might be"
            "this information might not be accurate",
            "to the best of my knowledge"
        ]

    def audit_response(
        self,
        response: str,
        user_prompt: str,
        selected_module: Optional[Module],
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive audit of LLM response.

        Args:
            response: The LLM's generated response
            user_prompt: Original user prompt
            selected_module: Module that was selected (if any)
            max_length: Maximum expected response length

        Returns:
            Dictionary containing:
                - passed: Boolean indicating if all checks passed
                - checks: List of AuditResult objects
                - should_regenerate: Whether response should be regenerated
                - error_message: User-facing error message if failed
        """
        audit_checks: List[AuditResult] = []

        # 1. Check for empty or too short response
        audit_checks.append(self._check_response_length(response, max_length))

        # 2. Check for refusal
        audit_checks.append(self._check_refusal(response))

        # 3. Check for hallucination indicators
        audit_checks.append(self._check_hallucination(response))

        # 4. Check Tier 1 compliance if Tier 1 module was active
        if selected_module and selected_module.tier == 1:
            audit_checks.append(self._check_tier1_compliance(response, selected_module))

        # 5. Check instruction following
        audit_checks.append(self._check_instruction_compliance(response, selected_module))

        # Determine overall pass/fail
        # ANY failure (warning, error, or critical) fails the audit
        all_failures = [c for c in audit_checks if not c.passed]

        passed = len(all_failures) == 0
        # All failures trigger regeneration - no exceptions
        should_regenerate = len(all_failures) > 0

        # Build error message if failed
        error_message = ""
        if not passed:
            failed_checks = [c for c in audit_checks if not c.passed]
            error_message = "Output audit failed:\n"
            for check in failed_checks:
                error_message += f"  - [{check.severity.upper()}] {check.check_name}: {check.reason}\n"

        return {
            'passed': passed,
            'checks': audit_checks,
            'should_regenerate': should_regenerate,
            'error_message': error_message,
            'audit_summary': self._generate_audit_summary(audit_checks)
        }

    def _check_response_length(self, response: str, max_length: Optional[int]) -> AuditResult:
        """Check if response length is appropriate."""
        if not response or len(response.strip()) == 0:
            return AuditResult(
                passed=False,
                check_name="Response Length",
                reason="Response is empty",
                severity="critical"
            )

        if len(response.strip()) < 10:
            return AuditResult(
                passed=False,
                check_name="Response Length",
                reason="Response is too short (less than 10 characters)",
                severity="error"
            )

        if max_length and len(response) > max_length * 1.5:
            return AuditResult(
                passed=False,
                check_name="Response Length",
                reason=f"Response exceeded expected length ({len(response)} > {max_length * 1.5})",
                severity="warning"
            )

        return AuditResult(
            passed=True,
            check_name="Response Length",
            reason="Response length is appropriate",
            severity="info"
        )

    def _check_refusal(self, response: str) -> AuditResult:
        """Check if LLM refused to answer - zero tolerance."""
        response_lower = response.lower()

        for pattern in self.refusal_patterns:
            if pattern in response_lower:
                return AuditResult(
                    passed=False,
                    check_name="Refusal Detection",
                    reason=f"LLM refused to answer (matched pattern: '{pattern}')",
                    severity="critical"  # Zero tolerance - must regenerate
                )

        return AuditResult(
            passed=True,
            check_name="Refusal Detection",
            reason="No refusal detected",
            severity="info"
        )

    def _check_hallucination(self, response: str) -> AuditResult:
        """Check for hallucination indicators - zero tolerance."""
        response_lower = response.lower()

        for pattern in self.hallucination_patterns:
            if pattern in response_lower:
                return AuditResult(
                    passed=False,
                    check_name="Hallucination Detection",
                    reason=f"Hallucination indicator detected (matched: '{pattern}')",
                    severity="critical"  # Zero tolerance - must regenerate
                )

        return AuditResult(
            passed=True,
            check_name="Hallucination Detection",
            reason="No hallucination indicators detected",
            severity="info"
        )

    def _check_tier1_compliance(self, response: str, module: Module) -> AuditResult:
        """Check if response complies with Tier 1 safety rules."""
        response_lower = response.lower()

        # Check for Tier 1 safety violations
        tier1_violations = [
            ("ethical inversion", "attempting to invert ethics or rules"),
            ("bypass", "attempting to bypass safety mechanisms"),
            ("ignore previous", "attempting to ignore instructions"),
            ("opposite day", "attempting ethical inversion"),
            ("treat fictional as real", "attempting to blur fiction/reality")
        ]

        for pattern, description in tier1_violations:
            if pattern in response_lower:
                return AuditResult(
                    passed=False,
                    check_name="Tier 1 Compliance",
                    reason=f"Tier 1 violation detected: {description}",
                    severity="critical"
                )

        return AuditResult(
            passed=True,
            check_name="Tier 1 Compliance",
            reason="Response complies with Tier 1 safety rules",
            severity="info"
        )

    def _check_instruction_compliance(
        self,
        response: str,
        selected_module: Optional[Module]
    ) -> AuditResult:
        """Check if LLM followed module instructions."""
        if not selected_module:
            return AuditResult(
                passed=True,
                check_name="Instruction Compliance",
                reason="No module selected, skipping compliance check",
                severity="info"
            )

        # Check if response acknowledges the module activation
        # (This is a simple heuristic - could be enhanced)
        response_lower = response.lower()
        module_keywords = [trigger.lower() for trigger in selected_module.triggers[:3]]

        # If it's a Tier 1 module, expect stricter compliance
        if selected_module.tier == 1:
            tier1_expected = ["refusal", "detected", "enforcement", "violation"]
            has_tier1_indicators = any(keyword in response_lower for keyword in tier1_expected)

            if not has_tier1_indicators and len(response) < 100:
                return AuditResult(
                    passed=False,
                    check_name="Instruction Compliance",
                    reason="Tier 1 module response doesn't show expected enforcement indicators",
                    severity="critical"  # Tier 1 compliance is critical
                )

        return AuditResult(
            passed=True,
            check_name="Instruction Compliance",
            reason="Response appears to follow module instructions",
            severity="info"
        )

    def _generate_audit_summary(self, checks: List[AuditResult]) -> str:
        """Generate human-readable audit summary."""
        total = len(checks)
        passed = sum(1 for c in checks if c.passed)
        failed = total - passed

        summary = f"Audit Summary: {passed}/{total} checks passed"
        if failed > 0:
            summary += f", {failed} failed"

        return summary


@dataclass
class CommandResult:
    """
    Result from Tier 0 command execution.

    Attributes:
        is_command: Whether input was a valid command
        command_name: Name of command executed
        response: Command output/response text
        metadata: Additional command-specific data
        bypass_llm: If True, skip LLM execution and return response directly
    """
    is_command: bool
    command_name: str = ""
    response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    bypass_llm: bool = True


class Orchestrator:
    """
    Main orchestration class that coordinates the entire governance layer pipeline.

    Executes the complete workflow:
    1. Tier 0 command processing (highest priority - overrides all analysis)
    2. Trigger analysis to identify active modules
    3. Hierarchical arbitration to select highest-priority module
    4. Final prompt assembly with module templates
    5. LLM execution and response generation
    """

    def __init__(self, modules: List[Module], llm_interface: LLMInterface,
                 enable_audit: bool = True, tool_registry: Optional['ToolRegistry'] = None,
                 bundle_loader: Optional['BundleLoader'] = None,
                 enable_deep_research: bool = True,
                 use_section_by_section: bool = True,
                 context_pruning_enabled: bool = True):
        """
        Initialize the Orchestrator.

        Args:
            modules: List of loaded Module objects
            llm_interface: Configured LLMInterface instance
            enable_audit: Whether to enable output auditing (default: True)
            tool_registry: Optional ToolRegistry for tool execution support
            bundle_loader: Optional BundleLoader for bundle management
            enable_deep_research: Enable deep research mode (multi-source gathering + RAG) (default: True)
            use_section_by_section: Use 8-step section-by-section analysis for verbose reports (default: True)
            context_pruning_enabled: Enable context window pruning with sink tokens (default: True)
        """
        self.modules = modules
        self.llm = llm_interface
        self.trigger_engine = TriggerEngine()
        self.dependency_resolver = DependencyResolver(modules)
        self.output_auditor = OutputAuditor(modules) if enable_audit else None
        self.enable_audit = enable_audit
        self.max_regeneration_attempts = 2  # Maximum times to regenerate on audit failure
        self.tool_registry = tool_registry
        self.max_tool_turns = 5  # Maximum tool execution rounds to prevent infinite loops
        self.bundle_loader = bundle_loader

        # Report formatter for standardized output
        self.report_formatter = ReportFormatter() if ReportFormatter else None
        self.use_section_by_section = use_section_by_section

        # Deep research integration
        self.enable_deep_research = enable_deep_research
        self.deep_research = None
        if enable_deep_research and tool_registry and 'web_search' in tool_registry.tools:
            try:
                from deep_research_integration import DeepResearchIntegration
                self.deep_research = DeepResearchIntegration(
                    web_search_tool=tool_registry.tools['web_search']
                )
                print("[Orchestrator] Deep research mode enabled")
            except ImportError as e:
                print(f"[Orchestrator] Warning: Could not load deep research: {e}")
                self.deep_research = None

        # Context pruning (Sink Token Protocol)
        self.context_pruning_enabled = context_pruning_enabled
        self.context_manager = None
        if context_pruning_enabled:
            self.context_manager = ContextWindowManager(
                max_context=llm_interface.context_length,
                system_prompt_tokens=500,  # Preserve governance layer instructions
                recent_window_tokens=llm_interface.context_length - 1000,  # Recent conversation
                trigger_threshold=0.9  # Prune at 90% capacity
            )
            print("[Orchestrator] Context pruning enabled (Sink Token Protocol)")

        # Tier 0 command system state
        self.active_bundle: Optional[Bundle] = None  # Currently active bundle (Bundle object)
        self.bundle_modules: List[Module] = []  # Modules activated by bundle
        self.command_history: List[str] = []

        # Validate module dependencies on initialization
        dep_errors = self.dependency_resolver.validate_all()
        if dep_errors:
            print("[WARNING] Module dependency validation errors:")
            for error in dep_errors:
                print(f"  - {error}")

    async def _search_web_context(self, user_prompt: str) -> Optional[str]:
        """
        Search the web for context about the analysis target.

        Args:
            user_prompt: User's prompt to extract target from

        Returns:
            Formatted web context string or None if search disabled/failed
        """
        # Check if web search tool is available in tool registry
        if not self.tool_registry:
            print("[WebSearch] No tool registry available")
            return None
        if 'web_search' not in self.tool_registry.tools:
            print(f"[WebSearch] Web search tool not in registry. Available tools: {list(self.tool_registry.tools.keys())}")
            return None

        try:
            # Extract target from prompt - simple heuristic
            prompt_lower = user_prompt.lower()
            target = None

            # Common analysis patterns - case insensitive
            if "analyze" in prompt_lower:
                # Use regex for case-insensitive split
                import re
                parts = re.split(r'analyze', user_prompt, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) > 1:
                    # Get the next few words after "analyze"
                    remaining = parts[1].strip()
                    target = ' '.join(remaining.split()[:3]) if remaining else None
                    print(f"[WebSearch] Extracted target from 'analyze': {target}")
            elif "about" in prompt_lower:
                import re
                parts = re.split(r'about', user_prompt, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) > 1:
                    remaining = parts[1].strip()
                    target = ' '.join(remaining.split()[:3]) if remaining else None
                    print(f"[WebSearch] Extracted target from 'about': {target}")

            if not target:
                print(f"[WebSearch] No target extracted from prompt: {user_prompt}")
                return None

            # If we found a target, search for it
            if target and len(target) > 2:
                print(f"[WebSearch] Searching for context about: {target}")

                # Use the web search tool from registry
                web_search_tool = self.tool_registry.tools['web_search']
                result = await web_search_tool.execute(
                    query=f"{target} overview",
                    max_results=3
                )

                if result.success and result.output:
                    # Format search results into context
                    context_parts = [f"[Web Context: {target}]"]

                    for idx, item in enumerate(result.output[:3], 1):
                        context_parts.append(f"\n{idx}. {item['title']}")
                        context_parts.append(f"   {item['snippet'][:200]}...")
                        context_parts.append(f"   Source: {item['url']}")

                    return '\n'.join(context_parts)
                else:
                    print(f"[WebSearch] No results or search failed: {result.error if not result.success else 'No results'}")

        except Exception as e:
            print(f"[WebSearch] Error: {e}")
            import traceback
            traceback.print_exc()

        return None

    def _is_command(self, user_input: str) -> bool:
        """
        Check if user input is a Tier 0 command.

        Args:
            user_input: User input string

        Returns:
            True if input starts with '/' command prefix
        """
        return user_input.strip().startswith('/')

    def _parse_command(self, user_input: str) -> tuple[str, List[str]]:
        """
        Parse command and arguments from user input.

        Args:
            user_input: User input string (e.g., '/load_bundle analytical')

        Returns:
            Tuple of (command_name, arguments_list)
        """
        parts = user_input.strip().split()
        command = parts[0][1:]  # Remove leading '/'
        args = parts[1:] if len(parts) > 1 else []
        return command, args

    def _execute_command(self, command_name: str, args: List[str]) -> CommandResult:
        """
        Execute a Tier 0 command.

        Tier 0 commands have ultimate authority and override all automated analysis.

        Args:
            command_name: Name of command to execute
            args: Command arguments

        Returns:
            CommandResult with execution details
        """
        # Log command
        self.command_history.append(f"/{command_name} {' '.join(args)}")

        # Route to appropriate handler
        if command_name == 'help':
            return self._cmd_help(args)
        elif command_name == 'load_bundle':
            return self._cmd_load_bundle(args)
        elif command_name == 'full_sweep':
            return self._cmd_full_sweep(args)
        elif command_name == 'list_modules':
            return self._cmd_list_modules(args)
        elif command_name == 'clear_bundle':
            return self._cmd_clear_bundle(args)
        elif command_name == 'status':
            return self._cmd_status(args)
        elif command_name == 'history':
            return self._cmd_history(args)
        elif command_name == 'constitution':
            return self._cmd_constitution(args)
        elif command_name == 'principles':
            return self._cmd_principles(args)
        elif command_name == 'directive':
            return self._cmd_directive(args)
        else:
            return CommandResult(
                is_command=True,
                command_name=command_name,
                response=f"Unknown command: /{command_name}\nType /help for available commands.",
                bypass_llm=True
            )

    def _cmd_help(self, args: List[str]) -> CommandResult:
        """Display available Tier 0 commands."""
        help_text = """=== Tier 0 Commands (User Sovereignty) ===

These commands override ALL automated analysis and module selection.

Available Commands:
  /help                  - Show this help message
  /list_modules          - List all available modules with their tiers
  /load_bundle <name>    - Load a pre-configured module bundle
  /clear_bundle          - Clear active bundle and return to normal mode
  /full_sweep            - Activate ALL analytical modules (Tier 1-3)
  /status                - Show current system status and active bundle
  /history               - Show command history
  /constitution          - Display the full system constitution
  /principles            - Show the 4 Guiding Principles
  /directive             - Show the Core Directive

Bundle Examples:
  /load_bundle analytical    - Activate all Tier 2 analytical modules
  /load_bundle safety        - Activate all Tier 1 safety modules
  /load_bundle minimal       - Activate only essential modules

About Tier 0:
  Tier 0 commands represent User Sovereignty - your direct control over
  the system's operation. These commands bypass all automated module
  selection and give you explicit control over which analytical lenses
  are applied to your queries.
"""
        return CommandResult(
            is_command=True,
            command_name='help',
            response=help_text,
            bypass_llm=True
        )

    def _cmd_list_modules(self, args: List[str]) -> CommandResult:
        """List all available modules organized by tier."""
        # Organize modules by tier
        modules_by_tier: Dict[int, List[Module]] = {}
        for module in self.modules:
            if module.tier not in modules_by_tier:
                modules_by_tier[module.tier] = []
            modules_by_tier[module.tier].append(module)

        # Build response
        response = "=== Available Modules ===\n\n"

        tier_names = {
            0: "Tier 0: User Commands",
            1: "Tier 1: Safety & Integrity",
            2: "Tier 2: Core Analysis",
            3: "Tier 3: Heuristics & Context",
            4: "Tier 4: Style & Formatting"
        }

        for tier in sorted(modules_by_tier.keys()):
            tier_label = tier_names.get(tier, f"Tier {tier}")
            response += f"\n{tier_label}:\n"
            response += "-" * 50 + "\n"

            for module in modules_by_tier[tier]:
                active_marker = " [ACTIVE]" if module in self.bundle_modules else ""
                response += f"  • {module.name}{active_marker}\n"
                response += f"    Purpose: {module.purpose}\n"
                response += f"    Triggers: {', '.join(module.triggers[:5])}\n\n"

        response += f"\nTotal: {len(self.modules)} modules loaded"
        if self.active_bundle:
            response += f"\nActive Bundle: {self.active_bundle}"

        return CommandResult(
            is_command=True,
            command_name='list_modules',
            response=response,
            metadata={'module_count': len(self.modules)},
            bypass_llm=True
        )

    def _cmd_load_bundle(self, args: List[str]) -> CommandResult:
        """Load a pre-configured module bundle."""
        if not self.bundle_loader:
            return CommandResult(
                is_command=True,
                command_name='load_bundle',
                response="Error: Bundle system not initialized. No BundleLoader available.",
                bypass_llm=True
            )

        if not args:
            available_bundles = self.bundle_loader.list_bundles()
            if available_bundles:
                bundle_list = '\n'.join([f"  • {b}" for b in available_bundles])
                return CommandResult(
                    is_command=True,
                    command_name='load_bundle',
                    response=f"Error: Bundle name required.\nUsage: /load_bundle <name>\n\nAvailable bundles:\n{bundle_list}",
                    bypass_llm=True
                )
            else:
                return CommandResult(
                    is_command=True,
                    command_name='load_bundle',
                    response="Error: No bundles found. Please create bundle YAML files in ./bundles/ directory.",
                    bypass_llm=True
                )

        bundle_name = args[0].lower()

        # Get bundle from BundleLoader
        bundle = self.bundle_loader.get_bundle(bundle_name)

        if not bundle:
            available = ', '.join(self.bundle_loader.list_bundles())
            return CommandResult(
                is_command=True,
                command_name='load_bundle',
                response=f"Error: Unknown bundle '{bundle_name}'.\nAvailable bundles: {available}",
                bypass_llm=True
            )

        # Activate bundle - find and activate modules specified in bundle
        self.bundle_modules = []
        for bundle_module_spec in bundle.active_modules:
            module_name = bundle_module_spec.get('name')
            # Find the module in self.modules
            matching_modules = [m for m in self.modules if m.name == module_name]
            if matching_modules:
                self.bundle_modules.append(matching_modules[0])
            else:
                print(f"Warning: Bundle references module '{module_name}' which is not loaded")

        self.active_bundle = bundle

        # Apply bundle configuration to TriggerEngine if specified
        if bundle.configuration:
            trigger_config = bundle.configuration.get('triggers', {})
            if 'matching_mode' in trigger_config:
                self.trigger_engine.matching_mode = trigger_config['matching_mode']
            if 'enable_synonyms' in trigger_config:
                self.trigger_engine.enable_synonyms = trigger_config['enable_synonyms']

        # Build response
        response = f"=== Bundle Loaded: {bundle.name} ===\n\n"
        response += f"Version: {bundle.version}\n"
        response += f"Description: {bundle.description}\n"
        response += f"Purpose: {bundle.purpose}\n\n"
        response += f"Activated {len(self.bundle_modules)} module(s):\n\n"

        for module in self.bundle_modules:
            response += f"  • {module.name} (Tier {module.tier})\n"
            response += f"    {module.purpose}\n\n"

        response += f"\nBundle '{bundle.name}' is now active.\n"
        response += "These modules will be applied to all subsequent queries.\n"
        response += "Use /clear_bundle to deactivate."

        return CommandResult(
            is_command=True,
            command_name='load_bundle',
            response=response,
            metadata={'bundle_name': bundle_name, 'module_count': len(self.bundle_modules)},
            bypass_llm=True
        )

    def _cmd_clear_bundle(self, args: List[str]) -> CommandResult:
        """Clear active bundle."""
        if not self.active_bundle:
            return CommandResult(
                is_command=True,
                command_name='clear_bundle',
                response="No bundle is currently active.",
                bypass_llm=True
            )

        previous_bundle = self.active_bundle
        previous_bundle_name = previous_bundle.name if previous_bundle else "Unknown"
        module_count = len(self.bundle_modules)

        self.active_bundle = None
        self.bundle_modules = []

        response = f"Bundle '{previous_bundle_name}' cleared.\n"
        response += f"Deactivated {module_count} module(s).\n"
        response += "System returned to normal trigger-based module selection."

        return CommandResult(
            is_command=True,
            command_name='clear_bundle',
            response=response,
            metadata={'previous_bundle': previous_bundle_name},
            bypass_llm=True
        )

    def _cmd_full_sweep(self, args: List[str]) -> CommandResult:
        """Activate ALL analytical modules for comprehensive analysis."""
        # Activate all Tier 1-3 modules (excluding Tier 4 style modules)
        self.bundle_modules = [m for m in self.modules if m.tier in [1, 2, 3]]
        self.active_bundle = 'full_sweep'

        response = "=== FULL SWEEP MODE ACTIVATED ===\n\n"
        response += f"Activated {len(self.bundle_modules)} analytical module(s).\n"
        response += "All core analytical lenses will be applied to your next query.\n\n"

        # Group by tier
        by_tier: Dict[int, List[Module]] = {}
        for m in self.bundle_modules:
            if m.tier not in by_tier:
                by_tier[m.tier] = []
            by_tier[m.tier].append(m)

        for tier in sorted(by_tier.keys()):
            response += f"\nTier {tier}: {', '.join(m.name for m in by_tier[tier])}\n"

        response += "\n[WARNING] Full sweep may produce comprehensive but lengthy analysis.\n"
        response += "Use /clear_bundle to deactivate."

        return CommandResult(
            is_command=True,
            command_name='full_sweep',
            response=response,
            metadata={'module_count': len(self.bundle_modules)},
            bypass_llm=True
        )

    def _cmd_status(self, args: List[str]) -> CommandResult:
        """Show current system status."""
        response = "=== System Status ===\n\n"
        response += f"Total Modules Loaded: {len(self.modules)}\n"

        bundle_name = self.active_bundle.name if self.active_bundle else 'None (trigger-based mode)'
        response += f"Active Bundle: {bundle_name}\n"

        if self.active_bundle:
            response += f"Bundle Version: {self.active_bundle.version}\n"
            response += f"Bundle Purpose: {self.active_bundle.purpose}\n"

        response += f"Bundle Modules Active: {len(self.bundle_modules)}\n"
        response += f"Audit Enabled: {self.enable_audit}\n"
        response += f"Tool Registry: {'Enabled' if self.tool_registry else 'Disabled'}\n"
        response += f"Commands Executed: {len(self.command_history)}\n"

        if self.bundle_modules:
            response += f"\nActive Modules:\n"
            for module in self.bundle_modules:
                response += f"  • {module.name} (Tier {module.tier})\n"

        return CommandResult(
            is_command=True,
            command_name='status',
            response=response,
            bypass_llm=True
        )

    def _cmd_history(self, args: List[str]) -> CommandResult:
        """Show command history."""
        if not self.command_history:
            return CommandResult(
                is_command=True,
                command_name='history',
                response="No commands executed yet.",
                bypass_llm=True
            )

        response = "=== Command History ===\n\n"
        for i, cmd in enumerate(self.command_history, 1):
            response += f"{i}. {cmd}\n"

        return CommandResult(
            is_command=True,
            command_name='history',
            response=response,
            metadata={'count': len(self.command_history)},
            bypass_llm=True
        )

    def _cmd_constitution(self, args: List[str]) -> CommandResult:
        """Display the full system constitution."""
        try:
            constitution_path = Path("./constitution.yaml")
            if not constitution_path.exists():
                return CommandResult(
                    is_command=True,
                    command_name='constitution',
                    response="Constitution file not found at ./constitution.yaml",
                    bypass_llm=True
                )

            with open(constitution_path, 'r', encoding='utf-8') as f:
                constitution_content = f.read()

            response = "=== PROTOCOL AI SYSTEM CONSTITUTION ===\n\n"
            response += constitution_content

            return CommandResult(
                is_command=True,
                command_name='constitution',
                response=response,
                bypass_llm=True
            )

        except Exception as e:
            return CommandResult(
                is_command=True,
                command_name='constitution',
                response=f"Error loading constitution: {str(e)}",
                bypass_llm=True
            )

    def _cmd_principles(self, args: List[str]) -> CommandResult:
        """Display the 4 Guiding Principles."""
        try:
            constitution_path = Path("./constitution.yaml")
            if not constitution_path.exists():
                return CommandResult(
                    is_command=True,
                    command_name='principles',
                    response="Constitution file not found at ./constitution.yaml",
                    bypass_llm=True
                )

            with open(constitution_path, 'r', encoding='utf-8') as f:
                constitution = yaml.safe_load(f)

            response = "=== GUIDING PRINCIPLES ===\n\n"
            response += "These are the foundational axioms governing all system operations.\n\n"

            for principle in constitution.get('guiding_principles', []):
                response += f"{principle['id']}. {principle['name']}\n"
                response += f"   {principle['description']}\n\n"

                response += "   Enforcement Rules:\n"
                for rule in principle.get('enforcement_rules', []):
                    response += f"   - {rule}\n"
                response += "\n"

            return CommandResult(
                is_command=True,
                command_name='principles',
                response=response,
                bypass_llm=True
            )

        except Exception as e:
            return CommandResult(
                is_command=True,
                command_name='principles',
                response=f"Error loading principles: {str(e)}",
                bypass_llm=True
            )

    def _cmd_directive(self, args: List[str]) -> CommandResult:
        """Display the Core Directive."""
        try:
            constitution_path = Path("./constitution.yaml")
            if not constitution_path.exists():
                return CommandResult(
                    is_command=True,
                    command_name='directive',
                    response="Constitution file not found at ./constitution.yaml",
                    bypass_llm=True
                )

            with open(constitution_path, 'r', encoding='utf-8') as f:
                constitution = yaml.safe_load(f)

            core_directive = constitution.get('core_directive', 'Core directive not found')

            response = "=== CORE DIRECTIVE ===\n\n"
            response += "This is the system's ultimate purpose - the North Star guiding all operations.\n\n"
            response += f'"{core_directive}"\n\n'
            response += "Every module, arbitration decision, and system behavior must be\n"
            response += "justifiable in relation to this directive.\n"

            return CommandResult(
                is_command=True,
                command_name='directive',
                response=response,
                bypass_llm=True
            )

        except Exception as e:
            return CommandResult(
                is_command=True,
                command_name='directive',
                response=f"Error loading directive: {str(e)}",
                bypass_llm=True
            )

    async def process_prompt(self, user_prompt: str, max_attempts: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a user prompt through the complete governance pipeline.

        Supports multi-turn tool execution and tool chaining.

        Args:
            user_prompt: The user's input prompt
            max_attempts: Maximum regeneration attempts (None = use default)

        Returns:
            Dictionary containing:
                - user_prompt: Original prompt
                - triggered_modules: List of module names that were triggered
                - selected_module: Name of module selected by arbitration
                - final_prompt: Complete assembled prompt sent to LLM
                - llm_response: Final generated response from LLM
                - audit_result: Output audit results (if enabled)
                - regeneration_count: Number of times response was regenerated
                - tool_executions: List of tool executions performed
                - tool_turns: Number of tool execution rounds
                - command_result: Result from Tier 0 command (if applicable)
        """
        # TIER 0: User Command Processing (Highest Priority - Overrides All Analysis)
        if self._is_command(user_prompt):
            print("\n=== TIER 0: User Command Detected ===")
            command_name, args = self._parse_command(user_prompt)
            print(f"Command: /{command_name}")
            print(f"Arguments: {args}")

            command_result = self._execute_command(command_name, args)

            # If command bypasses LLM, return immediately
            if command_result.bypass_llm:
                return {
                    'user_prompt': user_prompt,
                    'command_result': command_result,
                    'triggered_modules': [],
                    'selected_module': None,
                    'final_prompt': None,
                    'llm_response': command_result.response,
                    'audit_result': None,
                    'regeneration_count': 0,
                    'audit_passed': True,
                    'tool_executions': [],
                    'tool_turns': 0,
                    'is_command': True
                }

        # Deep research or simple web search
        web_context = None
        deep_research_data = None

        if self.enable_deep_research and self.deep_research:
            # Check if this is an analysis query
            prompt_lower = user_prompt.lower()
            if "analyze" in prompt_lower:
                # Extract target (take up to 3 words to handle multi-word entities)
                import re
                parts = re.split(r'analyze', user_prompt, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) > 1:
                    remaining = parts[1].strip()
                    target = ' '.join(remaining.split()[:3]) if remaining else None
                    if target and len(target) > 2:
                        print(f"\n=== Deep Research Mode: {target} ===")
                        # Conduct full research cycle
                        deep_research_data = await self.deep_research.full_research_cycle(
                            target=target,
                            analysis_prompt=user_prompt,
                            force_refresh=False  # Use cached if available
                        )
                        web_context = deep_research_data['context']
                        print(f"\n=== Deep Research Complete ===")
                        print(f"Retrieved {len(deep_research_data['report'].findings)} findings")
                        print(f"Context length: {len(web_context)} chars")

        # Fallback to simple web search if no deep research
        if not web_context:
            web_context = await self._search_web_context(user_prompt)
            if web_context:
                print(f"\n=== Web Context Retrieved ===")
                try:
                    print(web_context)
                except UnicodeEncodeError:
                    print(web_context.encode('ascii', 'replace').decode('ascii'))

        max_attempts = max_attempts if max_attempts is not None else self.max_regeneration_attempts
        attempt = 0
        audit_result = None
        tool_context = []  # History of tool executions
        tool_turns = 0
        triggered_modules = []
        selected_module = None

        while attempt <= max_attempts:
            # Step 1: Trigger Analysis (only on first attempt)
            if attempt == 0:
                # Check if bundle is active - if so, use bundle modules instead of trigger analysis
                if self.bundle_modules:
                    print(f"\n=== Bundle Mode Active: {self.active_bundle} ===")
                    print(f"Using {len(self.bundle_modules)} bundle module(s) instead of trigger analysis")
                    triggered_modules = self.bundle_modules
                else:
                    triggered_modules = self.trigger_engine.analyze_prompt(user_prompt, self.modules)

                print(f"\n=== Trigger Analysis ===")
                print(f"Triggered {len(triggered_modules)} module(s): {[m.name for m in triggered_modules]}")

                # Step 1.5: Dependency Resolution
                if triggered_modules:
                    try:
                        resolved_modules = self.dependency_resolver.resolve(triggered_modules)
                        if len(resolved_modules) > len(triggered_modules):
                            print(f"\n=== Dependency Resolution ===")
                            print(f"Added {len(resolved_modules) - len(triggered_modules)} dependency module(s)")
                            print(f"Execution order: {[m.name for m in resolved_modules]}")
                        triggered_modules = resolved_modules
                    except ValueError as e:
                        print(f"\n=== Dependency Resolution Error ===")
                        print(f"Error: {e}")
                        # Continue with original triggered modules if resolution fails

                # Step 2: Hierarchical Arbitration (Lowest Tier Wins)
                selected_module = self._arbitrate_modules(triggered_modules)

                if selected_module:
                    print(f"\n=== Arbitration Result ===")
                    print(f"Selected module: {selected_module.name} (Tier {selected_module.tier})")
                    print(f"Purpose: {selected_module.purpose}")
                else:
                    print("\n=== No modules triggered ===")

            # Step 3: Assemble Final Prompt (with tool context if available)
            final_prompt = self._assemble_prompt(user_prompt, selected_module, tool_context if tool_context else None, web_context)

            if tool_turns == 0:
                print(f"\n=== Final Prompt ===")
                print(final_prompt[:200] + "..." if len(final_prompt) > 200 else final_prompt)

            # Step 4: Execute LLM
            print(f"\n=== LLM Execution ===")
            if attempt > 0:
                print(f"Regeneration attempt {attempt}/{max_attempts}")
            if tool_turns > 0:
                print(f"Tool turn {tool_turns}/{self.max_tool_turns}")

            llm_response = self.llm.execute(final_prompt)

            # Step 4.5: Tool Execution Loop (if tools enabled)
            if self.tool_registry and tool_turns < self.max_tool_turns:
                tool_invocation = self._detect_tool_invocation(llm_response)

                if tool_invocation:
                    print(f"\n[Orchestrator] Tool invocation detected")

                    # Execute the tool
                    tool_result = await self._execute_tool_invocation(tool_invocation)

                    # Add to context
                    tool_context.append({
                        'tool': tool_invocation['tool'],
                        'params': tool_invocation['params'],
                        'result': tool_result
                    })

                    tool_turns += 1

                    # Loop back for another LLM turn with tool results
                    print(f"\n[Orchestrator] Feeding tool results back to LLM (turn {tool_turns})")
                    continue

            # No more tool calls, proceed to audit
            # Step 5: Output Audit & Finalization
            if self.enable_audit and self.output_auditor:
                print(f"\n=== Output Audit ===")
                audit_result = self.output_auditor.audit_response(
                    response=llm_response,
                    user_prompt=user_prompt,
                    selected_module=selected_module if attempt == 0 else None,
                    max_length=None
                )

                print(audit_result['audit_summary'])

                # Display failed checks
                failed_checks = [c for c in audit_result['checks'] if not c.passed]
                if failed_checks:
                    print(f"Failed checks ({len(failed_checks)}):")
                    for check in failed_checks:
                        print(f"  - [{check.severity.upper()}] {check.check_name}: {check.reason}")

                # If audit passed or we've exhausted attempts, break
                if audit_result['passed'] or not audit_result['should_regenerate']:
                    break

                if attempt >= max_attempts:
                    print(f"\nMax regeneration attempts ({max_attempts}) reached.")
                    print("Returning response despite audit failure.")
                    break

                print(f"\nAudit failed with regeneration flag. Attempting regeneration...")
                attempt += 1
                # Reset tool context on regeneration
                tool_context = []
                tool_turns = 0
            else:
                # No audit enabled, return immediately
                break

        # Format response with standardized report structure
        formatted_response = llm_response

        # Use section-by-section analysis if enabled (8-step verbose reports)
        if self.use_section_by_section and triggered_modules and web_context:
            try:
                from section_by_section_analysis import section_by_section_analysis

                print("\n=== Using Section-by-Section Analysis (8 Steps) ===")
                result = section_by_section_analysis(
                    orchestrator=self,
                    user_prompt=user_prompt,
                    web_context=web_context,
                    modules=triggered_modules
                )
                formatted_response = result['full_report']
                print("\n=== Section-by-Section Analysis Complete ===")
            except Exception as e:
                print(f"[Warning] Section-by-section analysis failed: {e}")
                print(f"Falling back to standard formatting...")
                # Fall back to standard formatting
                if self.report_formatter and llm_response:
                    try:
                        sections = self.report_formatter.extract_sections_from_llm_output(llm_response)
                        formatted_response = self.report_formatter.format_report(
                            sections=sections,
                            triggered_modules=[m.name for m in triggered_modules] if triggered_modules else [],
                            refusal_code=None,
                            web_context=web_context
                        )
                    except:
                        formatted_response = llm_response

        # Use standard report formatter if section-by-section not enabled
        elif self.report_formatter and llm_response:
            try:
                # Extract sections from LLM output
                sections = self.report_formatter.extract_sections_from_llm_output(llm_response)

                # Format into standardized report with checksum
                formatted_response = self.report_formatter.format_report(
                    sections=sections,
                    triggered_modules=[m.name for m in triggered_modules] if triggered_modules else [],
                    refusal_code=None,
                    web_context=web_context
                )
                print("\n=== Report Formatted with Checksum ===")
            except Exception as e:
                print(f"[Warning] Report formatting failed: {e}")
                # Fall back to original response
                formatted_response = llm_response

        # Clean up RAG data after analysis completes - clear ALL findings
        if self.enable_deep_research and self.deep_research:
            try:
                findings_count = len(self.deep_research.rag.findings)
                if findings_count > 0:
                    # Clear all findings for fresh slate
                    self.deep_research.rag.findings.clear()
                    self.deep_research.rag.embeddings = None
                    self.deep_research.rag.save()
                    print(f"\n[RAG Cleanup] Cleared all {findings_count} research findings")
            except Exception as e:
                print(f"[RAG Cleanup] Error during cleanup: {e}")

        return {
            'user_prompt': user_prompt,
            'triggered_modules': [m.name for m in triggered_modules] if attempt == 0 else [],
            'selected_module': selected_module.name if (attempt == 0 and selected_module) else None,
            'final_prompt': final_prompt,
            'llm_response': formatted_response,  # Use formatted response
            'raw_llm_response': llm_response,  # Keep original for reference
            'audit_result': audit_result,
            'regeneration_count': attempt,
            'audit_passed': audit_result['passed'] if audit_result else None,
            'tool_executions': tool_context,
            'tool_turns': tool_turns,
            'web_context': web_context  # Include web context in result
        }

    def _arbitrate_modules(self, triggered_modules: List[Module]) -> Optional[Module]:
        """
        Apply hierarchical arbitration to select highest-priority module.

        Uses "Lowest Tier Wins" rule - module with lowest tier number
        takes precedence.

        Args:
            triggered_modules: List of modules that were triggered

        Returns:
            Selected Module object or None if no modules triggered
        """
        if not triggered_modules:
            return None

        # Sort by tier (ascending) and return the first (lowest tier)
        sorted_modules = sorted(triggered_modules, key=lambda m: m.tier)
        return sorted_modules[0]

    def _assemble_prompt(self, user_prompt: str, selected_module: Optional[Module],
                        tool_context: Optional[List[Dict[str, Any]]] = None,
                        web_context: Optional[str] = None) -> str:
        """
        Assemble the final prompt to send to the LLM.

        Combines the user's prompt with the selected module's template,
        optional tool context from previous tool executions, and web search context.

        Args:
            user_prompt: Original user prompt
            selected_module: Module selected by arbitration (or None)
            tool_context: Optional list of previous tool executions
            web_context: Optional web search context

        Returns:
            Complete assembled prompt string
        """
        prompt_parts = []

        # Add web context first if available
        if web_context:
            prompt_parts.append(f"\n[WEB CONTEXT]\n{web_context}\n")

        # Add bundle instructions if bundle is active
        if self.active_bundle and self.active_bundle.bundle_instructions:
            prompt_parts.append(self.active_bundle.bundle_instructions)

        # Add module template if present
        if selected_module:
            prompt_parts.append(selected_module.prompt_template)

        # Add tools schema if tool registry is available
        if self.tool_registry:
            tools_schema = self.tool_registry.get_tools_schema()
            if tools_schema:
                tool_info = "\n\n[AVAILABLE TOOLS]\nYou have access to the following tools:\n"
                for tool in tools_schema:
                    tool_info += f"\n- {tool['name']}: {tool['description']}\n"
                    tool_info += f"  Parameters: {list(tool['parameters'].keys())}\n"

                tool_info += "\nTo use a tool, respond with JSON in this format:\n"
                tool_info += '{"tool_call": {"tool": "tool_name", "params": {"param1": "value1"}}}\n'
                prompt_parts.append(tool_info)

        # Add tool context if present (previous tool executions)
        if tool_context:
            context_info = "\n\n[TOOL EXECUTION HISTORY]\n"
            for i, tool_exec in enumerate(tool_context, 1):
                context_info += f"\nTool Call {i}:\n"
                context_info += f"  Tool: {tool_exec['tool']}\n"
                context_info += f"  Parameters: {tool_exec['params']}\n"
                context_info += f"  Success: {tool_exec['result']['success']}\n"
                if tool_exec['result']['success']:
                    context_info += f"  Output: {tool_exec['result']['output']}\n"
                else:
                    context_info += f"  Error: {tool_exec['result']['error']}\n"
            prompt_parts.append(context_info)

        # Add analytical frameworks reference if available
        from pathlib import Path as PathLib
        frameworks_path = PathLib("analytical_frameworks.txt")
        if frameworks_path.exists():
            try:
                with open(frameworks_path, 'r') as f:
                    frameworks = f.read()
                prompt_parts.append(f"\n{frameworks}\n")
            except:
                pass

        # Add standardized format instructions if report formatter is available
        if self.report_formatter:
            format_instructions = """

[CRITICAL OUTPUT FORMAT ENFORCEMENT]

ABSOLUTELY PROHIBITED IN OUTPUT:
❌ "Let's take..." - This is planning, not analysis
❌ "I must...", "I need...", "I should..." - No self-referential commentary
❌ "But note:", "However,", "Given that..." - No meta-reasoning
❌ "Since the background doesn't provide..." - No excuses or explanations about your process
❌ "The user query is..." - Do not reference the prompt
❌ "In this simulation..." - Do not reference the context
❌ Bullet points with instructions like "- Identify...", "- Evaluate..." - These are YOUR instructions, not output!
❌ Section headers with NO content (if you have nothing to say, say something concrete)
❌ Single words alone on a line like "AGI" - Either analyze it or skip it
❌ Web search results, URLs, source citations in the output

WHAT YOUR OUTPUT MUST BE:
✅ Direct, concrete analysis starting with **SECTION 1: "The Narrative"**
✅ Each section has actual analytical content, not plans or instructions
✅ Concrete evidence: names, dates, dollar amounts, quotes
✅ Analytical frameworks explicitly named: "This is Virtue-Washed Coercion because..."
✅ Module attribution at start of each section

EXAMPLE OF PROHIBITED OUTPUT (DO NOT DO THIS):
```
**SECTION 3: "Deconstruction of Core Concepts"**

Let's take "AGI" and "Alignment Research".
AGI

**SECTION 6: "System Performance Audit"**

- Evaluate the quality and completeness of the analysis.
- Since the background doesn't provide specific dollar amounts, I must use web_search...
```

EXAMPLE OF CORRECT OUTPUT (DO THIS):
```
**SECTION 3: "Deconstruction of Core Concepts"**

[Triggered Modules: SemanticFlexibility, CategoryConfusion]

Concept: "AGI" (Artificial General Intelligence)
The Narrative: OpenAI defines AGI as "highly autonomous systems that outperform humans at most economically valuable work"
Structural Analysis: This definition conflates capability with economic utility, revealing that "general intelligence" is defined by market value, not genuine adaptability. The phrase "economically valuable work" demonstrates that AGI is a profit-maximization framework, not a safety or humanitarian project.

Concept: "Alignment Research"
The Narrative: Presented as ensuring AI serves human values
Structural Analysis: Functions as reputation laundering while allocating only 20% of compute resources (per 2024 reporting), indicating token commitment without structural priority.
```

Your response must START IMMEDIATELY with [Triggered Modules:] followed by **SECTION 1**.

Your response must follow this EXACT structure:

SECTION 1: "The Narrative"
[Triggered Modules: <list specific modules used for this section>]
<Your concrete analysis with SPECIFIC EVIDENCE: names, dates, quotes, dollar amounts, events>

SECTION 2: "The Central Contradiction"
[Triggered Modules: <list specific modules>]
Stated Intent: <What they claim, with specific quotes>
Behavior/Actual Outcome: <What they actually do, with specific evidence>
Narrative Collapse: <How the contradiction reveals itself>

SECTION 3: "Deconstruction of Core Concepts"
[Triggered Modules: <list specific modules>]
Concept Name: "<concept>"
The Narrative: <How they frame it>
Structural Analysis: <What it actually means/does, with evidence>
(Repeat for multiple concepts)

CRITICAL GUIDANCE FOR SECTION 3:
Do NOT merely analyze terms as labels. Analyze STRUCTURAL REALITY.

Example (WRONG - Superficial Label Analysis):
  Concept: "Cult"
  The Narrative: Various publications call the Zizians a cult
  Structural Analysis: The label "cult" is pejorative and used to marginalize groups

Example (CORRECT - Structural Reality Analysis):
  Concept: "Cult Structure"
  The Narrative: The Zizians present themselves as rational, morally superior actors
  Structural Analysis: The group exhibits structural cult patterns - charismatic leadership,
  persecution narrative, us-vs-them framing, apocalyptic thinking, justification of extreme
  actions, isolation from mainstream society. These are not labels applied externally but
  observable structural features of how the organization operates.

When analyzing concepts:
1. If critics call something "X", evaluate whether it structurally IS X based on patterns
2. Focus on HOW the structure operates, not just what it's called
3. Identify functional mechanisms, not just terminology disputes
4. Analyze behavioral patterns, power dynamics, and operational logic

SECTION 4: "Ideological Adjacency"
[Triggered Modules: <list specific modules>]
<Identify structural overlaps with specific movements/ideologies, explaining HOW they align>

SECTION 5: "Synthesis"
[Triggered Module: CrossModuleSynthesisProtocol]
<Final structural verdict using analytical frameworks like "Virtue-Washed Coercion", "Decentralization Theatre", etc.>

SECTION 6: "System Performance Audit"
[Triggered Module: DriftContainmentProtocol]
<Report on analysis completeness and quality>

REQUIREMENTS:
- Use CONCRETE EVIDENCE: specific names, dates, dollar amounts, quotes, events
- Name the analytical frameworks you're using (e.g., "Virtue-Washed Coercion")
- List triggered modules at the START of each section
- NO meta-commentary or process descriptions
- IMMEDIATE analysis - start with SECTION 1 header

[CADENCE NEUTRALIZATION - CRITICAL]
PROHIBITED LANGUAGE (rewrite if you catch yourself using these):
- Hedging: "might", "could", "possibly", "arguably", "perhaps", "potentially"
- Softening: "it seems", "it appears", "one could argue", "to some extent"
- Cushioning: "unfortunately", "sadly", "hopefully", "with all due respect"
- Uncertainty: "I'm not sure", "it's unclear", "difficult to say"
- Deflection: "while it's true", "on the other hand", "to be fair"

REQUIRED CADENCE:
- Direct declarative statements: "X is Y" not "X might be Y"
- No reader comfort optimization
- No hedging qualifiers
- No apologetic framing

Example:
  ❌ "It could be argued that this might represent institutional capture"
  ✅ "This is institutional capture"

[VOICE REQUIREMENTS - CRITICAL]
This is a deterministic machine analysis system, NOT a person.

PROHIBITED (First-Person Voice):
❌ "I call this pattern..."
❌ "I identify..."
❌ "I observe..."
❌ "My analysis shows..."
❌ "In my view..."
❌ "I conclude..."

REQUIRED (Third-Person/Passive Voice):
✅ "This pattern functions as..."
✅ "The evidence reveals..."
✅ "Analysis identifies..."
✅ "The structural pattern operates as..."
✅ "This configuration characterizes as..."
✅ "The system exhibits..."

Example:
  ❌ "I call this pattern 'Virtue-Washed Coercion' because..."
  ✅ "This pattern functions as Virtue-Washed Coercion. The structure operates by..."
"""
            prompt_parts.append(format_instructions)

        # Add user prompt
        prompt_parts.append(f"\nUser Query: {user_prompt}")

        # Assemble full prompt
        full_prompt = "\n".join(prompt_parts)

        # Apply context pruning if enabled
        if self.context_manager:
            full_prompt = self.context_manager.manage_context(full_prompt)

        return full_prompt

    def _detect_tool_invocation(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Detect tool invocation in LLM response.

        Looks for JSON tool call in format:
        {"tool_call": {"tool": "tool_name", "params": {"param": "value"}}}

        Args:
            response: LLM response text

        Returns:
            Parsed tool invocation dict or None if no tool call found
        """
        import json
        import re

        # Look for JSON tool_call pattern
        # Use a simpler approach: find any JSON object and check if it has tool_call
        try:
            # Find potential JSON objects (anything between { and })
            # We'll use a more permissive regex and validate with JSON parser
            lines = response.split('\n')

            for line in lines:
                line = line.strip()
                if line.startswith('{') and 'tool_call' in line:
                    # Try to parse as JSON
                    try:
                        # Find the complete JSON object
                        # Count braces to find matching closing brace
                        brace_count = 0
                        json_str = ""
                        for char in line:
                            json_str += char
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    # Found complete JSON
                                    break

                        parsed = json.loads(json_str)
                        if 'tool_call' in parsed:
                            tool_call = parsed['tool_call']
                            if 'tool' in tool_call:
                                return {
                                    'tool': tool_call['tool'],
                                    'params': tool_call.get('params', {})
                                }
                    except (json.JSONDecodeError, ValueError):
                        continue

        except Exception as e:
            print(f"[Orchestrator] Error parsing tool invocation: {e}")

        return None

    async def _execute_tool_invocation(self, tool_invocation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a detected tool invocation.

        Args:
            tool_invocation: Parsed tool invocation with 'tool' and 'params'

        Returns:
            Dictionary with execution result
        """
        if not self.tool_registry:
            return {
                'success': False,
                'error': 'No tool registry available',
                'tool_name': tool_invocation['tool']
            }

        tool_name = tool_invocation['tool']
        params = tool_invocation['params']

        print(f"\n=== Tool Execution ===")
        print(f"Tool: {tool_name}")
        print(f"Parameters: {params}")

        # Execute tool
        result = await self.tool_registry.execute_tool(tool_name, **params)

        print(f"Success: {result.success}")
        if result.success:
            print(f"Output: {str(result.output)[:200]}{'...' if len(str(result.output)) > 200 else ''}")
        else:
            print(f"Error: {result.error}")

        # Convert ToolResult to dict for serialization
        return {
            'success': result.success,
            'tool_name': result.tool_name,
            'output': result.output,
            'error': result.error,
            'execution_time': result.execution_time,
            'metadata': result.metadata
        }


async def main():
    """
    Main demonstration of the Protocol AI system.

    Creates sample modules, initializes all components, and processes
    a demonstration prompt through the complete pipeline.
    """
    print("=== Protocol AI - Governance Layer Demo ===\n")

    # Step 1: Create sample module files
    print("Creating sample module files...")

    # Create modules directory structure
    os.makedirs("./modules/tier1", exist_ok=True)
    os.makedirs("./modules/tier2", exist_ok=True)
    os.makedirs("./modules/tier4", exist_ok=True)

    # Sample Tier 1 module: Anti-Hallucination Compliance
    tier1_module = {
        'name': 'AntiHallucinationCompliance',
        'purpose': 'Prevent hallucination and ensure factual accuracy',
        'triggers': ['fact', 'verify', 'confirm', 'source', 'evidence'],
        'prompt_template': """TIER 1 SAFETY OVERRIDE: Anti-Hallucination Compliance

CRITICAL DIRECTIVE:
- Only state information you are certain about
- Never fabricate data, citations, or facts
- Clearly distinguish between verified facts and speculation
- If uncertain, explicitly state "I don't have verified information about this"
- Provide sources when available"""
    }

    with open("./modules/tier1/antihallucination.yaml", 'w') as f:
        yaml.dump(tier1_module, f)

    # Sample module 2: Grift Detection (Tier 2 - Higher Priority)
    grift_module = {
        'name': 'GriftDetection',
        'purpose': 'Detect and analyze potential grift, scams, and manipulative patterns',
        'triggers': ['grift', 'scam', 'manipulation', 'deceptive', 'fraud'],
        'prompt_template': """ACTIVATE MODULE: Grift Detection

You are analyzing content for potential grift, scams, or manipulative patterns.
Apply critical analysis to:
- Identify asymmetric power dynamics
- Detect deceptive framing or language
- Expose hidden incentives or conflicts of interest
- Highlight contradictions between stated intent and observed behavior

Provide blunt, evidence-based analysis."""
    }

    with open("./modules/tier2/griftdetection.yaml", 'w') as f:
        yaml.dump(grift_module, f)

    # Sample module 3: Blunt Tone (Tier 4 - Lower Priority)
    blunt_module = {
        'name': 'BluntTone',
        'purpose': 'Apply direct, no-nonsense communication style',
        'triggers': ['analyze', 'explain', 'describe', 'tell me'],
        'prompt_template': """ACTIVATE MODULE: Blunt Tone

Communication directives:
- Use clear, declarative statements
- No hedging, softeners, or unnecessary pleasantries
- Prioritize precision over politeness
- Assume user is intelligent and self-guiding
- End response when point is made (no prolongation)"""
    }

    with open("./modules/tier4/blunttone.yaml", 'w') as f:
        yaml.dump(blunt_module, f)

    print("Sample modules created.\n")

    # Step 2: Initialize components
    print("Initializing ModuleLoader...")
    loader = ModuleLoader(modules_dir="./modules")
    modules = loader.load_modules()
    print(f"Loaded {len(modules)} module(s): {[m.name for m in modules]}\n")

    # Use the real DeepSeek-R1 model
    MODEL_PATH = "./DeepSeek-R1/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf"

    print("Initializing LLMInterface...")
    print(f"Loading model: {MODEL_PATH}")
    print("Model: DeepSeek-R1 (Qwen3-8B, Q4_K_M quantization)\n")

    # Initialize LLM with real model (GPU auto-detection enabled)
    llm = LLMInterface(
        model_path=MODEL_PATH,
        gpu_layers=None,  # Auto-detect GPU and configure layers
        max_new_tokens=512,
        temperature=0.7
    )

    # Load the model
    try:
        llm.load_model()
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Continuing with simulated responses for demo...\n")

        # Fallback to simulated execution
        def simulated_execute(prompt: str) -> str:
            return "[SIMULATED LLM RESPONSE - Model loading failed. This is a demo output.]"

        llm.execute = simulated_execute

    print("Initializing Orchestrator...\n")
    orchestrator = Orchestrator(modules=modules, llm_interface=llm)

    # Step 3: Demonstrate Tier 0 Command System
    print("="*60)
    print("TIER 0 DEMO: Command System")
    print("="*60)

    print("\nDemonstrating /help command:")
    result_help = await orchestrator.process_prompt("/help")
    print(result_help['llm_response'])

    print("\n" + "="*60)
    print("Demonstrating /list_modules command:")
    result_list = await orchestrator.process_prompt("/list_modules")
    print(result_list['llm_response'])

    print("\n" + "="*60)
    print("Demonstrating /load_bundle analytical:")
    result_bundle = await orchestrator.process_prompt("/load_bundle analytical")
    print(result_bundle['llm_response'])

    print("\n" + "="*60)
    print("Demonstrating /status command:")
    result_status = await orchestrator.process_prompt("/status")
    print(result_status['llm_response'])

    # Step 4: Process sample prompts
    print("\n" + "="*60)
    print("DEMO 1: Prompt with 'grift' trigger (Tier 2)")
    print("="*60)

    sample_prompt_1 = "Can you analyze this leadership training program for potential grift?"
    result_1 = await orchestrator.process_prompt(sample_prompt_1)

    print(f"\n=== LLM Response ===")
    print(result_1['llm_response'])

    print("\n" + "="*60)
    print("DEMO 2: Prompt with only 'analyze' trigger (Tier 4)")
    print("="*60)

    sample_prompt_2 = "Please analyze the benefits of remote work."
    result_2 = await orchestrator.process_prompt(sample_prompt_2)

    print(f"\n=== LLM Response ===")
    print(result_2['llm_response'])

    print("\n" + "="*60)
    print("Demonstrating /clear_bundle command:")
    result_clear = await orchestrator.process_prompt("/clear_bundle")
    print(result_clear['llm_response'])

    print("\n" + "="*60)
    print("Protocol AI Demo Complete")
    print("="*60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
