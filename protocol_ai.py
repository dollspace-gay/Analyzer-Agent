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


def detect_gpu_layers() -> int:
    """
    Automatically detect available GPU and return recommended number of layers.

    Uses nvidia-smi to detect GPU VRAM and calculates appropriate layer count.
    Returns 0 if no GPU is detected or if detection fails.

    Returns:
        int: Number of layers to offload to GPU (0 for CPU-only mode)
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

            print(f"GPU detected with {vram_gb:.1f}GB VRAM")

            # Heuristic: Use more layers for higher VRAM
            # Leave some VRAM for system overhead
            if vram_gb >= 12:
                return -1  # Offload all layers
            elif vram_gb >= 8:
                return 35  # Most layers
            elif vram_gb >= 6:
                return 25  # Many layers
            elif vram_gb >= 4:
                return 15  # Some layers
            else:
                return 0  # CPU only for low VRAM
        else:
            print("No NVIDIA GPU detected, using CPU mode")
            return 0

    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError) as e:
        print(f"GPU detection failed ({e}), defaulting to CPU mode")
        return 0


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
        metadata: Additional module configuration data
    """
    name: str
    tier: int
    purpose: str
    triggers: List[str]
    prompt_template: str
    metadata: Dict[str, Any] = field(default_factory=dict)


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

    Uses case-insensitive keyword matching to identify relevant modules
    based on trigger words defined in module configurations.
    """

    def __init__(self):
        """Initialize the TriggerEngine."""
        pass

    def analyze_prompt(self, user_prompt: str, modules: List[Module]) -> List[Module]:
        """
        Analyze user prompt and return list of triggered modules.

        Performs case-insensitive keyword matching against module triggers.

        Args:
            user_prompt: The user's input prompt string
            modules: List of available Module objects

        Returns:
            List of Module objects whose triggers match the prompt
        """
        active_modules = []
        prompt_lower = user_prompt.lower()

        for module in modules:
            # Check if any trigger keyword is present in the prompt
            for trigger in module.triggers:
                if trigger.lower() in prompt_lower:
                    active_modules.append(module)
                    break  # Don't add same module multiple times

        return active_modules


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
        n_threads: Optional[int] = None
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
        """
        self.model_path = model_path
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.n_threads = n_threads if n_threads is not None else max(1, os.cpu_count() // 2)

        # Auto-detect GPU layers if not specified
        if gpu_layers is None:
            self.gpu_layers = detect_gpu_layers()
            print(f"Auto-detected GPU layers: {self.gpu_layers}")
        else:
            self.gpu_layers = gpu_layers

        self.model: Optional[Llama] = None

    def load_model(self) -> None:
        """
        Load the LLM model from disk using llama-cpp-python.

        Raises:
            FileNotFoundError: If model file doesn't exist
            ImportError: If llama-cpp-python is not installed
            Exception: If model loading fails
        """
        if Llama is None:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python"
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        print(f"Loading model from {self.model_path}...")
        print(f"Configuration:")
        print(f"  - GPU layers: {self.gpu_layers}")
        print(f"  - Context length: {self.context_length}")
        print(f"  - CPU threads: {self.n_threads}")

        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.gpu_layers,
            n_ctx=self.context_length,
            n_threads=self.n_threads,
            verbose=False
        )

        print("Model loaded successfully.")

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

        response = self.model(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            echo=False
        )

        # Extract text from response
        return response['choices'][0]['text']


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
            "i'm not certain",
            "i may be incorrect",
            "this information might not be accurate",
            "i'm not sure",
            "i could be wrong",
            "to the best of my knowledge",
            "i think",
            "i believe",
            "probably",
            "might be"
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


class Orchestrator:
    """
    Main orchestration class that coordinates the entire governance layer pipeline.

    Executes the complete workflow:
    1. Trigger analysis to identify active modules
    2. Hierarchical arbitration to select highest-priority module
    3. Final prompt assembly with module templates
    4. LLM execution and response generation
    """

    def __init__(self, modules: List[Module], llm_interface: LLMInterface, enable_audit: bool = True):
        """
        Initialize the Orchestrator.

        Args:
            modules: List of loaded Module objects
            llm_interface: Configured LLMInterface instance
            enable_audit: Whether to enable output auditing (default: True)
        """
        self.modules = modules
        self.llm = llm_interface
        self.trigger_engine = TriggerEngine()
        self.output_auditor = OutputAuditor(modules) if enable_audit else None
        self.enable_audit = enable_audit
        self.max_regeneration_attempts = 2  # Maximum times to regenerate on audit failure

    def process_prompt(self, user_prompt: str, max_attempts: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a user prompt through the complete governance pipeline.

        Args:
            user_prompt: The user's input prompt
            max_attempts: Maximum regeneration attempts (None = use default)

        Returns:
            Dictionary containing:
                - user_prompt: Original prompt
                - triggered_modules: List of module names that were triggered
                - selected_module: Name of module selected by arbitration
                - final_prompt: Complete assembled prompt sent to LLM
                - llm_response: Generated response from LLM
                - audit_result: Output audit results (if enabled)
                - regeneration_count: Number of times response was regenerated
        """
        max_attempts = max_attempts if max_attempts is not None else self.max_regeneration_attempts
        attempt = 0
        audit_result = None

        while attempt <= max_attempts:
            # Step 1: Trigger Analysis
            triggered_modules = self.trigger_engine.analyze_prompt(user_prompt, self.modules)

            print(f"\n=== Trigger Analysis ===")
            print(f"Triggered {len(triggered_modules)} module(s): {[m.name for m in triggered_modules]}")

            # Step 2: Hierarchical Arbitration (Lowest Tier Wins)
            selected_module = self._arbitrate_modules(triggered_modules)

            if selected_module:
                print(f"\n=== Arbitration Result ===")
                print(f"Selected module: {selected_module.name} (Tier {selected_module.tier})")
                print(f"Purpose: {selected_module.purpose}")
            else:
                print("\n=== No modules triggered ===")

            # Step 3: Assemble Final Prompt
            final_prompt = self._assemble_prompt(user_prompt, selected_module)

            print(f"\n=== Final Prompt ===")
            print(final_prompt[:200] + "..." if len(final_prompt) > 200 else final_prompt)

            # Step 4: Execute LLM
            print(f"\n=== LLM Execution ===")
            if attempt > 0:
                print(f"Regeneration attempt {attempt}/{max_attempts}")

            llm_response = self.llm.execute(final_prompt)

            # Step 5: Output Audit & Finalization
            if self.enable_audit and self.output_auditor:
                print(f"\n=== Output Audit ===")
                audit_result = self.output_auditor.audit_response(
                    response=llm_response,
                    user_prompt=user_prompt,
                    selected_module=selected_module,
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
            else:
                # No audit enabled, return immediately
                break

        return {
            'user_prompt': user_prompt,
            'triggered_modules': [m.name for m in triggered_modules],
            'selected_module': selected_module.name if selected_module else None,
            'final_prompt': final_prompt,
            'llm_response': llm_response,
            'audit_result': audit_result,
            'regeneration_count': attempt,
            'audit_passed': audit_result['passed'] if audit_result else None
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

    def _assemble_prompt(self, user_prompt: str, selected_module: Optional[Module]) -> str:
        """
        Assemble the final prompt to send to the LLM.

        Combines the user's prompt with the selected module's template.

        Args:
            user_prompt: Original user prompt
            selected_module: Module selected by arbitration (or None)

        Returns:
            Complete assembled prompt string
        """
        if selected_module:
            # Inject module template before user prompt
            final_prompt = f"{selected_module.prompt_template}\n\nUser Query: {user_prompt}"
        else:
            # No module selected, use user prompt directly
            final_prompt = user_prompt

        return final_prompt


def main():
    """
    Main demonstration of the Protocol AI system.

    Creates sample modules, initializes all components, and processes
    a demonstration prompt through the complete pipeline.
    """
    print("=== Protocol AI - Governance Layer Demo ===\n")

    # Step 1: Create sample module files
    print("Creating sample module files...")

    # Create modules directory structure
    os.makedirs("./modules/tier2", exist_ok=True)
    os.makedirs("./modules/tier4", exist_ok=True)

    # Sample module 1: Grift Detection (Tier 2 - Higher Priority)
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

    # Sample module 2: Blunt Tone (Tier 4 - Lower Priority)
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

    # Step 3: Process sample prompts
    print("="*60)
    print("DEMO 1: Prompt with 'grift' trigger (Tier 2)")
    print("="*60)

    sample_prompt_1 = "Can you analyze this leadership training program for potential grift?"
    result_1 = orchestrator.process_prompt(sample_prompt_1)

    print(f"\n=== LLM Response ===")
    print(result_1['llm_response'])

    print("\n" + "="*60)
    print("DEMO 2: Prompt with only 'analyze' trigger (Tier 4)")
    print("="*60)

    sample_prompt_2 = "Please analyze the benefits of remote work."
    result_2 = orchestrator.process_prompt(sample_prompt_2)

    print(f"\n=== LLM Response ===")
    print(result_2['llm_response'])

    print("\n" + "="*60)
    print("Protocol AI Demo Complete")
    print("="*60)


if __name__ == "__main__":
    main()
