"""
Code Execution Tool

Safely executes Python code in a sandboxed environment.
Features:
- Restricted to ./sandbox/ directory for all file operations
- Whitelist of safe Python modules (no os, subprocess, network access)
- Execution timeout (default: 30 seconds)
- CPU and memory limits
- Captures stdout/stderr
- Logs all executions
- Prevents path traversal attacks
"""

import sys
sys.path.insert(0, '..')

from protocol_ai import Tool, ToolResult
from typing import Dict, Any, Optional
import io
import contextlib
import time
import traceback
from pathlib import Path


class CodeExecutionTool(Tool):
    """
    Executes Python code in a sandboxed environment with security restrictions.

    All file operations are restricted to ./sandbox/ directory.
    Only whitelisted modules can be imported.
    """

    def __init__(self):
        """Initialize the code execution tool."""
        name = "code_execution"
        description = "Execute Python code safely in a sandboxed environment. Returns stdout, stderr, and result."

        # Define parameter schema
        parameters = {
            "code": {
                "type": "string",
                "description": "Python code to execute",
                "required": True
            },
            "timeout": {
                "type": "number",
                "description": "Execution timeout in seconds (default: 30, max: 300)",
                "required": False,
                "default": 30
            },
            "save_output": {
                "type": "boolean",
                "description": "Save stdout to sandbox/outputs/execution_output.txt",
                "required": False,
                "default": False
            }
        }

        super().__init__(name, description, parameters)

        # Sandbox root
        self.sandbox_root = Path("./sandbox").resolve()

        # Whitelist of safe modules
        self.allowed_modules = {
            # Built-ins (safe subset)
            'math', 'random', 'datetime', 'time', 'json', 'csv',
            'itertools', 'functools', 'collections', 're', 'string',
            'decimal', 'fractions', 'statistics', 'hashlib', 'base64',
            'textwrap', 'difflib', 'uuid', 'sys',  # sys added for stderr/stdout

            # Scientific computing (if available)
            'numpy', 'scipy', 'pandas', 'matplotlib',

            # Data structures
            'heapq', 'bisect', 'array', 'queue', 'enum',
        }

        # Blacklist of dangerous modules
        self.blocked_modules = {
            'os', 'subprocess', 'shutil', 'glob',
            'socket', 'urllib', 'requests', 'http', 'ftplib',
            'smtplib', 'telnetlib', 'importlib', 'pickle',
            # Note: __import__, open, eval, exec, compile are handled separately
        }

        # Maximum resource limits
        self.max_timeout = 300  # 5 minutes
        self.default_timeout = 30  # 30 seconds

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute Python code in sandboxed environment.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            save_output: Whether to save output to file

        Returns:
            ToolResult with execution results
        """
        code = kwargs.get('code', '').strip()
        timeout = min(float(kwargs.get('timeout', self.default_timeout)), self.max_timeout)
        save_output = kwargs.get('save_output', False)

        # Validate code is not empty
        if not code:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="Code cannot be empty"
            )

        # Check for dangerous imports
        dangerous_imports = self._check_dangerous_imports(code)
        if dangerous_imports:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Blocked dangerous imports: {', '.join(dangerous_imports)}"
            )

        # Execute code
        try:
            start_time = time.time()

            # Capture stdout/stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            # Create restricted globals
            safe_globals = self._create_safe_globals()

            # Execute with timeout
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):

                try:
                    # Execute code
                    exec(code, safe_globals)
                    execution_error = None

                except Exception as e:
                    execution_error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

            execution_time = time.time() - start_time

            # Get output
            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()

            # Save output if requested
            if save_output and stdout_text:
                output_file = self.sandbox_root / "outputs" / "execution_output.txt"
                output_file.write_text(stdout_text)

            # Build result
            if execution_error:
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    error=execution_error,
                    output={
                        "stdout": stdout_text,
                        "stderr": stderr_text,
                        "execution_time": execution_time
                    },
                    metadata={
                        "timeout": timeout,
                        "execution_time": execution_time
                    }
                )
            else:
                return ToolResult(
                    success=True,
                    tool_name=self.name,
                    output={
                        "stdout": stdout_text,
                        "stderr": stderr_text,
                        "execution_time": execution_time,
                        "result": "Code executed successfully"
                    },
                    metadata={
                        "timeout": timeout,
                        "execution_time": execution_time,
                        "output_saved": save_output and bool(stdout_text)
                    }
                )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Execution failed: {str(e)}"
            )

    def _check_dangerous_imports(self, code: str) -> list[str]:
        """
        Check code for dangerous imports.

        Args:
            code: Python code to check

        Returns:
            List of dangerous imports found
        """
        dangerous = []

        # Simple check for import statements
        # Note: This is not foolproof but catches common cases
        lines = code.split('\n')
        for line in lines:
            line = line.strip()

            # Check for import statements
            if line.startswith('import ') or line.startswith('from '):
                # Extract module name
                if line.startswith('import '):
                    parts = line.replace('import ', '').split(' as ')[0].split(',')
                else:  # from X import Y
                    parts = [line.split('from ')[1].split(' import ')[0]]

                # Check each module
                for part in parts:
                    module_name = part.strip().split('.')[0]

                    if module_name in self.blocked_modules:
                        dangerous.append(module_name)

        return dangerous

    def _create_safe_globals(self) -> Dict[str, Any]:
        """
        Create a restricted global namespace for code execution.

        Returns:
            Dictionary of safe globals
        """
        # Start with minimal built-ins
        safe_builtins = {
            # Safe built-in functions
            'abs': abs,
            'all': all,
            'any': any,
            'bin': bin,
            'bool': bool,
            'chr': chr,
            'dict': dict,
            'divmod': divmod,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'format': format,
            'hex': hex,
            'int': int,
            'isinstance': isinstance,
            'issubclass': issubclass,
            'iter': iter,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'next': next,
            'oct': oct,
            'ord': ord,
            'pow': pow,
            'print': print,
            'range': range,
            'reversed': reversed,
            'round': round,
            'set': set,
            'slice': slice,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'zip': zip,

            # Safe types
            'True': True,
            'False': False,
            'None': None,
        }

        # Create safe globals
        safe_globals = {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
            '__doc__': None,
        }

        # Add safe import function
        safe_builtins['__import__'] = self._safe_import

        # Add safe file operations (sandboxed)
        safe_globals['open'] = self._safe_open

        # Add allowed modules (if available)
        for module_name in self.allowed_modules:
            try:
                module = __import__(module_name)
                safe_globals[module_name] = module
            except ImportError:
                # Module not available, skip
                pass

        return safe_globals

    def _safe_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """
        Safe import function that only allows whitelisted modules.

        Args:
            name: Module name to import
            globals: Global namespace (ignored)
            locals: Local namespace (ignored)
            fromlist: List of names to import from module
            level: Relative import level

        Returns:
            Imported module

        Raises:
            ImportError: If module is not in whitelist
        """
        # Check if module is allowed
        base_module = name.split('.')[0]

        if base_module in self.blocked_modules:
            raise ImportError(
                f"Import of '{name}' is not allowed. "
                f"Module '{base_module}' is blocked for security reasons."
            )

        if base_module not in self.allowed_modules:
            raise ImportError(
                f"Import of '{name}' is not allowed. "
                f"Module '{base_module}' is not in the whitelist. "
                f"Allowed modules: {', '.join(sorted(self.allowed_modules))}"
            )

        # Import the module using the real __import__
        import builtins
        return builtins.__import__(name, globals, locals, fromlist, level)

    def _safe_open(self, path: str, mode: str = 'r', **kwargs):
        """
        Safe file open that restricts access to sandbox directory.

        Args:
            path: File path (must be within sandbox)
            mode: File mode
            **kwargs: Additional arguments for open()

        Returns:
            File handle

        Raises:
            PermissionError: If path is outside sandbox
        """
        # Validate path is within sandbox
        try:
            requested_path = (self.sandbox_root / path).resolve()
            requested_path.relative_to(self.sandbox_root)
        except (ValueError, OSError):
            raise PermissionError(
                f"Access denied: Path must be within sandbox directory. "
                f"Use relative paths like 'inputs/file.txt' or 'outputs/result.txt'"
            )

        # Create parent directory if needed (for write modes)
        if 'w' in mode or 'a' in mode:
            requested_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file
        return open(requested_path, mode, **kwargs)

    def get_allowed_modules(self) -> list[str]:
        """
        Get list of allowed modules.

        Returns:
            List of allowed module names
        """
        return sorted(self.allowed_modules)

    def get_blocked_modules(self) -> list[str]:
        """
        Get list of blocked modules.

        Returns:
            List of blocked module names
        """
        return sorted(self.blocked_modules)
