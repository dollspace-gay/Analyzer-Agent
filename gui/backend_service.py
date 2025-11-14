"""
Backend Service - Bridge between GUI and protocol_ai

Manages the Orchestrator, ModuleLoader, and LLMInterface lifecycle.
Provides thread-safe async execution for GUI integration.
"""

import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import time

# Import protocol_ai components
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from protocol_ai import (
        ModuleLoader, Orchestrator, LLMInterface,
        BundleLoader, Module, ToolRegistry
    )
    PROTOCOL_AI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: protocol_ai not available: {e}")
    PROTOCOL_AI_AVAILABLE = False
    ModuleLoader = None
    Orchestrator = None
    LLMInterface = None
    BundleLoader = None
    ToolRegistry = None

# Import web search tool
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
    from web_search_tool import WebSearchTool
    WEB_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: web_search_tool not available: {e}")
    WEB_SEARCH_AVAILABLE = False
    WebSearchTool = None


class BackendService:
    """
    Service layer managing protocol_ai backend for GUI.

    Handles:
    - Module loading and reloading
    - LLM initialization
    - Async prompt processing
    - Response formatting for GUI
    """

    def __init__(self):
        """Initialize backend service"""
        self.orchestrator: Optional[Orchestrator] = None
        self.module_loader: Optional[ModuleLoader] = None
        self.bundle_loader: Optional[BundleLoader] = None
        self.llm_interface: Optional[LLMInterface] = None
        self.tool_registry: Optional['ToolRegistry'] = None
        self.modules: List[Module] = []
        self.is_initialized = False
        self.model_path: Optional[str] = None

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the backend with configuration.

        Args:
            config: Configuration dictionary with keys:
                - model_path: Path to GGUF model file
                - modules_path: Path to modules directory
                - context_size: LLM context window size
                - temperature: Generation temperature
                - max_tokens: Max tokens to generate
                - top_p: Top P sampling parameter
                - device: Device type (GPU/CPU)

        Returns:
            True if initialization succeeded, False otherwise
        """
        if not PROTOCOL_AI_AVAILABLE:
            print("Protocol AI backend not available")
            return False

        try:
            # Extract config
            model_path = config.get("model_path", "")
            modules_path = config.get("modules_path", "./modules")
            context_size = config.get("context_size", 4096)
            temperature = config.get("temperature", 0.7)
            max_tokens = config.get("max_tokens", 2048)
            top_p = config.get("top_p", 0.95)
            device = config.get("device", "CUDA (GPU)")

            # Load modules
            print(f"Loading modules from {modules_path}...")
            self.module_loader = ModuleLoader(modules_dir=modules_path)
            self.modules = self.module_loader.load_modules()
            print(f"Loaded {len(self.modules)} modules")

            # Load bundles (optional)
            try:
                self.bundle_loader = BundleLoader(bundles_dir="./bundles")
                bundles = self.bundle_loader.load_bundles()
                print(f"Loaded {len(bundles)} bundles")
            except Exception as e:
                print(f"Bundle loading failed (non-critical): {e}")
                self.bundle_loader = None

            # Initialize LLM (only if model path provided)
            if model_path and Path(model_path).exists():
                print(f"Initializing LLM: {model_path}")
                print(f"Device setting: {device}")

                # Determine GPU layers based on device
                if "GPU" in device or "CUDA" in device:
                    gpu_layers = -1  # All layers to GPU
                    print(f"GPU mode enabled: gpu_layers = {gpu_layers} (all layers)")
                else:
                    gpu_layers = 0  # CPU only
                    print(f"CPU mode: gpu_layers = {gpu_layers}")

                self.llm_interface = LLMInterface(
                    model_path=model_path,
                    gpu_layers=gpu_layers,
                    context_length=context_size,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
                self.model_path = model_path

                print(f"LLMInterface created with gpu_layers={gpu_layers}")

                # Load the model
                print("Loading model into memory...")
                self.llm_interface.load_model()
                print("LLM initialized and loaded successfully")
            else:
                print("No valid model path provided, LLM not initialized")
                self.llm_interface = None

            # Initialize tool registry and register web search
            if WEB_SEARCH_AVAILABLE and ToolRegistry:
                self.tool_registry = ToolRegistry()
                web_search_tool = WebSearchTool()
                self.tool_registry.register(web_search_tool)
                print("[Backend] Web search tool registered")
            else:
                print("[Backend] Web search tool not available")
                self.tool_registry = None

            # Create orchestrator
            if self.llm_interface:
                # Check if user wants deep research mode (can be configured)
                enable_deep_research = config.get("enable_deep_research", False)

                self.orchestrator = Orchestrator(
                    modules=self.modules,
                    llm_interface=self.llm_interface,
                    enable_audit=True,
                    tool_registry=self.tool_registry,
                    bundle_loader=self.bundle_loader,
                    enable_deep_research=enable_deep_research
                )
                print(f"Orchestrator initialized (deep research: {enable_deep_research})")
                self.is_initialized = True
            else:
                print("Cannot create orchestrator without LLM")
                self.is_initialized = False

            return self.is_initialized

        except Exception as e:
            print(f"Backend initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def process_prompt_async(self, prompt: str) -> Dict[str, Any]:
        """
        Process a prompt through the governance layer (async).

        Args:
            prompt: User prompt text

        Returns:
            Dictionary formatted for GUI display with keys:
                - response: LLM response text
                - modules: List of module dicts with name, tier, status
                - structured_prompt: Final assembled prompt
                - arbitration_log: List of log entries
                - metadata: Dict with tokens, time_ms, etc.
        """
        if not self.is_initialized or not self.orchestrator:
            return self._get_error_response("Backend not initialized. Please configure LLM in Settings.")

        start_time = time.time()

        try:
            # Call protocol_ai orchestrator
            result = await self.orchestrator.process_prompt(prompt)

            # Convert to GUI format
            gui_response = self._convert_to_gui_format(result, start_time)
            return gui_response

        except Exception as e:
            print(f"Prompt processing error: {e}")
            import traceback
            traceback.print_exc()
            return self._get_error_response(f"Processing error: {str(e)}")

    def process_prompt_sync(self, prompt: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for process_prompt_async.

        Args:
            prompt: User prompt text

        Returns:
            GUI-formatted response dictionary
        """
        # Create new event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run async function
        if loop.is_running():
            # If loop is already running (e.g., in Qt), use run_until_complete with new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.process_prompt_async(prompt))
                return future.result()
        else:
            return loop.run_until_complete(self.process_prompt_async(prompt))

    def _convert_to_gui_format(self, backend_result: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """
        Convert protocol_ai result to GUI format.

        Args:
            backend_result: Result from Orchestrator.process_prompt()
            start_time: Start time for duration calculation

        Returns:
            GUI-formatted response dictionary
        """
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Extract response
        response_text = backend_result.get('llm_response', 'No response generated')

        # Convert triggered modules to GUI format
        triggered_modules = backend_result.get('triggered_modules', [])
        selected_module_name = backend_result.get('selected_module', None)

        modules_list = []
        for module in triggered_modules:
            # Handle both string names and Module objects
            if isinstance(module, str):
                module_name = module
                status = "active" if (not selected_module_name or module_name == selected_module_name) else "overridden"
                modules_list.append({
                    "name": module_name,
                    "tier": 0,  # Default tier since we don't have the object
                    "status": status
                })
            else:
                # Module object
                status = "active" if (not selected_module_name or module.name == selected_module_name) else "overridden"
                modules_list.append({
                    "name": module.name,
                    "tier": module.tier,
                    "status": status
                })

        # Build arbitration log
        log_entries = []
        log_entries.append(f"[INFO] Trigger analysis complete: {len(triggered_modules)} modules activated")

        if selected_module_name:
            log_entries.append(f"[ARBITRATION] Selected module: {selected_module_name}")

        # Check if regeneration occurred
        regen_count = backend_result.get('regeneration_count', 0)
        if regen_count > 0:
            log_entries.append(f"[AUDIT] Response regenerated {regen_count} time(s)")

        # Add audit result if available
        audit_result = backend_result.get('audit_result')
        if audit_result:
            passed = backend_result.get('audit_passed', True)
            status = "PASSED" if passed else "FAILED"
            log_entries.append(f"[AUDIT] Output audit: {status}")

        log_entries.append("[EXECUTION] Structured prompt sent to LLM")
        log_entries.append("[SUCCESS] Response received and validated")

        # Build structured prompt view
        final_prompt = backend_result.get('final_prompt', 'N/A')

        # Estimate token count (rough approximation)
        token_estimate = len(response_text.split())

        return {
            "response": response_text,
            "modules": modules_list,
            "structured_prompt": final_prompt,
            "arbitration_log": log_entries,
            "metadata": {
                "tokens": token_estimate,
                "time_ms": elapsed_ms,
                "model": Path(self.model_path).name if self.model_path else "unknown",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "regeneration_count": regen_count
            }
        }

    def _get_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create error response in GUI format.

        Args:
            error_message: Error description

        Returns:
            GUI-formatted error response
        """
        return {
            "response": f"❌ Error: {error_message}",
            "modules": [],
            "structured_prompt": "Error occurred before prompt assembly",
            "arbitration_log": [
                "[ERROR] " + error_message
            ],
            "metadata": {
                "tokens": 0,
                "time_ms": 0,
                "model": "error",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        }

    def get_modules(self) -> List[Module]:
        """Get list of loaded modules"""
        return self.modules

    def reload_modules(self, modules_path: str = "./modules") -> bool:
        """
        Reload modules from filesystem.

        Args:
            modules_path: Path to modules directory

        Returns:
            True if reload succeeded
        """
        try:
            if not self.module_loader:
                self.module_loader = ModuleLoader(modules_dir=modules_path)

            self.modules = self.module_loader.load_modules()

            # Recreate orchestrator if LLM is available
            if self.llm_interface:
                # Preserve deep research setting if orchestrator exists
                enable_deep_research = (
                    self.orchestrator.enable_deep_research
                    if self.orchestrator else False
                )

                self.orchestrator = Orchestrator(
                    modules=self.modules,
                    llm_interface=self.llm_interface,
                    enable_audit=True,
                    tool_registry=self.tool_registry,
                    bundle_loader=self.bundle_loader,
                    enable_deep_research=enable_deep_research
                )

            print(f"Reloaded {len(self.modules)} modules")
            return True
        except Exception as e:
            print(f"Module reload failed: {e}")
            return False


# Global backend service instance
_backend_service: Optional[BackendService] = None


def get_backend_service() -> BackendService:
    """Get singleton backend service instance"""
    global _backend_service
    if _backend_service is None:
        _backend_service = BackendService()
    return _backend_service
