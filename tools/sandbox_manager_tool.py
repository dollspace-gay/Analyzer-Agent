"""
Sandbox Manager Tool

Manages the sandbox environment for code execution and file operations.
Features:
- Auto-creation of sandbox directory structure
- Path validation to prevent directory traversal
- Cleanup and maintenance operations
- File/directory browsing within sandbox
- Usage statistics tracking
"""

import sys
sys.path.insert(0, '..')

from protocol_ai import Tool, ToolResult
from typing import Dict, Any, List, Optional
from pathlib import Path
import os
import shutil
import datetime


class SandboxManagerTool(Tool):
    """
    Manages the sandbox environment for safe code execution and file operations.

    Ensures all file operations are restricted to the ./sandbox/ directory.
    """

    def __init__(self):
        """Initialize the sandbox manager tool."""
        name = "sandbox_manager"
        description = "Manage sandbox environment: create, cleanup, browse files, validate paths"

        # Define parameter schema
        parameters = {
            "operation": {
                "type": "string",
                "description": "Operation to perform: 'init', 'cleanup', 'list', 'stats', 'validate_path', 'clear_temp'",
                "required": True,
                "enum": ["init", "cleanup", "list", "stats", "validate_path", "clear_temp", "remove_file"]
            },
            "path": {
                "type": "string",
                "description": "Path within sandbox (for validate_path, list, remove_file operations)",
                "required": False
            },
            "recursive": {
                "type": "boolean",
                "description": "List directories recursively (for list operation)",
                "required": False,
                "default": False
            }
        }

        super().__init__(name, description, parameters)

        # Sandbox root directory
        self.sandbox_root = Path("./sandbox").resolve()

        # Subdirectories
        self.subdirs = {
            "inputs": "User-provided input files",
            "outputs": "Generated output files",
            "temp": "Temporary files (auto-cleanup)",
            "logs": "Execution logs"
        }

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute sandbox management operation.

        Args:
            operation: Operation to perform
            path: Optional path for certain operations
            recursive: Optional recursive flag for list operation

        Returns:
            ToolResult with operation result
        """
        operation = kwargs.get('operation')
        path = kwargs.get('path', '')
        recursive = kwargs.get('recursive', False)

        try:
            if operation == "init":
                return await self._init_sandbox()

            elif operation == "cleanup":
                return await self._cleanup_sandbox()

            elif operation == "list":
                return await self._list_sandbox(path, recursive)

            elif operation == "stats":
                return await self._get_stats()

            elif operation == "validate_path":
                return await self._validate_path(path)

            elif operation == "clear_temp":
                return await self._clear_temp()

            elif operation == "remove_file":
                return await self._remove_file(path)

            else:
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    error=f"Unknown operation: {operation}"
                )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Sandbox operation failed: {str(e)}"
            )

    async def _init_sandbox(self) -> ToolResult:
        """
        Initialize sandbox directory structure.

        Returns:
            ToolResult indicating success/failure
        """
        created_dirs = []

        try:
            # Create root sandbox directory
            if not self.sandbox_root.exists():
                self.sandbox_root.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(self.sandbox_root))

            # Create subdirectories
            for subdir_name, description in self.subdirs.items():
                subdir_path = self.sandbox_root / subdir_name
                if not subdir_path.exists():
                    subdir_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(str(subdir_path))

            # Create .gitkeep files to preserve directory structure
            for subdir_name in self.subdirs.keys():
                gitkeep = self.sandbox_root / subdir_name / ".gitkeep"
                if not gitkeep.exists():
                    gitkeep.touch()

            message = f"Sandbox initialized at {self.sandbox_root}"
            if created_dirs:
                message += f"\nCreated {len(created_dirs)} directories"

            return ToolResult(
                success=True,
                tool_name=self.name,
                output=message,
                metadata={
                    "sandbox_root": str(self.sandbox_root),
                    "created_dirs": created_dirs,
                    "subdirectories": list(self.subdirs.keys())
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Failed to initialize sandbox: {str(e)}"
            )

    async def _cleanup_sandbox(self) -> ToolResult:
        """
        Clean up sandbox (remove all files but preserve structure).

        Returns:
            ToolResult with cleanup statistics
        """
        try:
            files_removed = 0
            dirs_cleaned = []

            # Clean each subdirectory
            for subdir_name in self.subdirs.keys():
                subdir_path = self.sandbox_root / subdir_name

                if subdir_path.exists():
                    # Remove all files in subdirectory
                    for item in subdir_path.iterdir():
                        if item.is_file() and item.name != ".gitkeep":
                            item.unlink()
                            files_removed += 1
                        elif item.is_dir():
                            shutil.rmtree(item)
                            files_removed += 1  # Count as single item

                    dirs_cleaned.append(subdir_name)

            return ToolResult(
                success=True,
                tool_name=self.name,
                output=f"Sandbox cleaned: {files_removed} items removed",
                metadata={
                    "files_removed": files_removed,
                    "directories_cleaned": dirs_cleaned
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Failed to cleanup sandbox: {str(e)}"
            )

    async def _clear_temp(self) -> ToolResult:
        """
        Clear only the temp directory.

        Returns:
            ToolResult with cleanup statistics
        """
        try:
            temp_dir = self.sandbox_root / "temp"
            files_removed = 0

            if temp_dir.exists():
                for item in temp_dir.iterdir():
                    if item.name != ".gitkeep":
                        if item.is_file():
                            item.unlink()
                        else:
                            shutil.rmtree(item)
                        files_removed += 1

            return ToolResult(
                success=True,
                tool_name=self.name,
                output=f"Temp directory cleared: {files_removed} items removed",
                metadata={"files_removed": files_removed}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Failed to clear temp directory: {str(e)}"
            )

    async def _list_sandbox(self, subpath: str = "", recursive: bool = False) -> ToolResult:
        """
        List contents of sandbox or subdirectory.

        Args:
            subpath: Subdirectory within sandbox to list
            recursive: Whether to list recursively

        Returns:
            ToolResult with directory listing
        """
        try:
            # Validate and resolve path
            if subpath:
                is_valid, resolved_path = self.validate_sandbox_path(subpath)
                if not is_valid:
                    return ToolResult(
                        success=False,
                        tool_name=self.name,
                        error=resolved_path  # Error message
                    )
                list_path = Path(resolved_path)
            else:
                list_path = self.sandbox_root

            if not list_path.exists():
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    error=f"Path does not exist: {subpath or 'sandbox root'}"
                )

            # List contents
            contents = []

            if recursive:
                # Recursive listing
                for item in list_path.rglob("*"):
                    relative_path = item.relative_to(self.sandbox_root)
                    contents.append({
                        "path": str(relative_path),
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else 0,
                        "modified": datetime.datetime.fromtimestamp(
                            item.stat().st_mtime
                        ).isoformat()
                    })
            else:
                # Non-recursive listing
                for item in list_path.iterdir():
                    relative_path = item.relative_to(self.sandbox_root)
                    contents.append({
                        "path": str(relative_path),
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else 0,
                        "modified": datetime.datetime.fromtimestamp(
                            item.stat().st_mtime
                        ).isoformat()
                    })

            # Sort by type then name
            contents.sort(key=lambda x: (x["type"], x["path"]))

            return ToolResult(
                success=True,
                tool_name=self.name,
                output=contents,
                metadata={
                    "listed_path": str(list_path.relative_to(self.sandbox_root)) if list_path != self.sandbox_root else ".",
                    "item_count": len(contents),
                    "recursive": recursive
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Failed to list directory: {str(e)}"
            )

    async def _get_stats(self) -> ToolResult:
        """
        Get sandbox usage statistics.

        Returns:
            ToolResult with statistics
        """
        try:
            stats = {
                "sandbox_root": str(self.sandbox_root),
                "exists": self.sandbox_root.exists(),
                "subdirectories": {}
            }

            if self.sandbox_root.exists():
                # Get stats for each subdirectory
                for subdir_name, description in self.subdirs.items():
                    subdir_path = self.sandbox_root / subdir_name

                    if subdir_path.exists():
                        file_count = sum(1 for _ in subdir_path.rglob("*") if _.is_file())
                        total_size = sum(
                            f.stat().st_size
                            for f in subdir_path.rglob("*")
                            if f.is_file()
                        )

                        stats["subdirectories"][subdir_name] = {
                            "description": description,
                            "file_count": file_count,
                            "total_size_bytes": total_size,
                            "total_size_mb": round(total_size / (1024 * 1024), 2)
                        }

                # Calculate totals
                stats["total_files"] = sum(
                    s["file_count"] for s in stats["subdirectories"].values()
                )
                stats["total_size_bytes"] = sum(
                    s["total_size_bytes"] for s in stats["subdirectories"].values()
                )
                stats["total_size_mb"] = round(
                    stats["total_size_bytes"] / (1024 * 1024), 2
                )

            return ToolResult(
                success=True,
                tool_name=self.name,
                output=stats,
                metadata={"sandbox_initialized": stats["exists"]}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Failed to get statistics: {str(e)}"
            )

    async def _validate_path(self, path: str) -> ToolResult:
        """
        Validate that a path is within the sandbox.

        Args:
            path: Path to validate

        Returns:
            ToolResult with validation result
        """
        is_valid, result = self.validate_sandbox_path(path)

        if is_valid:
            return ToolResult(
                success=True,
                tool_name=self.name,
                output=f"Path is valid: {result}",
                metadata={
                    "requested_path": path,
                    "resolved_path": result,
                    "is_within_sandbox": True
                }
            )
        else:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=result,  # Error message
                metadata={
                    "requested_path": path,
                    "is_within_sandbox": False
                }
            )

    async def _remove_file(self, path: str) -> ToolResult:
        """
        Remove a file or directory from sandbox.

        Args:
            path: Path to remove (must be within sandbox)

        Returns:
            ToolResult indicating success/failure
        """
        if not path:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="Path is required for remove_file operation"
            )

        # Validate path
        is_valid, resolved_path = self.validate_sandbox_path(path)
        if not is_valid:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=resolved_path  # Error message
            )

        try:
            target = Path(resolved_path)

            if not target.exists():
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    error=f"Path does not exist: {path}"
                )

            # Remove file or directory
            if target.is_file():
                target.unlink()
                item_type = "file"
            else:
                shutil.rmtree(target)
                item_type = "directory"

            return ToolResult(
                success=True,
                tool_name=self.name,
                output=f"Removed {item_type}: {path}",
                metadata={
                    "removed_path": path,
                    "item_type": item_type
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Failed to remove {path}: {str(e)}"
            )

    def validate_sandbox_path(self, path: str) -> tuple[bool, str]:
        """
        Validate that a path is within the sandbox.

        This is a critical security function that prevents directory traversal attacks.

        Args:
            path: Path to validate (relative or absolute)

        Returns:
            tuple: (is_valid, absolute_path_or_error_message)
        """
        try:
            # Handle empty path
            if not path:
                return True, str(self.sandbox_root)

            # Resolve the requested path
            # First, try as relative to sandbox root
            requested_path = (self.sandbox_root / path).resolve()

            # Check if resolved path is within sandbox
            try:
                requested_path.relative_to(self.sandbox_root)
                # Path is within sandbox
                return True, str(requested_path)
            except ValueError:
                # Path is outside sandbox
                return False, f"Path outside sandbox: {path}"

        except Exception as e:
            return False, f"Invalid path: {str(e)}"


# Auto-initialize sandbox on module load
async def auto_init_sandbox():
    """Auto-initialize sandbox when tool is loaded."""
    manager = SandboxManagerTool()
    result = await manager._init_sandbox()
    if result.success:
        print(f"[SandboxManager] {result.output}")
    else:
        print(f"[SandboxManager] Initialization failed: {result.error}")


# Run auto-init (commented out for now - will be called by main system)
# import asyncio
# asyncio.run(auto_init_sandbox())
