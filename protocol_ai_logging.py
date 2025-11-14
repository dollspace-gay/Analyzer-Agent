"""
Protocol AI Logging and Error Handling Framework

Provides comprehensive logging with verbosity levels, structured error handling,
and debugging utilities for troubleshooting the governance layer system.

Features:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Structured logging with context (module names, timestamps, categories)
- Log file rotation and archiving
- Console and file output with independent levels
- Module-specific loggers
- Performance metrics logging
- Error context capture
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from logging.handlers import RotatingFileHandler
import json


class ProtocolAILogger:
    """
    Centralized logging system for Protocol AI.

    Manages console and file logging with independent verbosity levels,
    structured output, and context tracking.
    """

    # Log levels
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    # Category constants
    CATEGORY_SYSTEM = "system"
    CATEGORY_MODULE = "module"
    CATEGORY_TRIGGER = "trigger"
    CATEGORY_ARBITRATION = "arbitration"
    CATEGORY_LLM = "llm"
    CATEGORY_DEPENDENCY = "dependency"
    CATEGORY_BUNDLE = "bundle"
    CATEGORY_TEST = "test"
    CATEGORY_ERROR = "error"

    def __init__(self,
                 name: str = "protocol_ai",
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG,
                 log_dir: str = "./logs",
                 enable_file_logging: bool = True,
                 max_bytes: int = 10_000_000,  # 10MB
                 backup_count: int = 5):
        """
        Initialize the Protocol AI logger.

        Args:
            name: Logger name
            console_level: Minimum level for console output
            file_level: Minimum level for file output
            log_dir: Directory for log files
            enable_file_logging: Whether to enable file logging
            max_bytes: Maximum size of log file before rotation
            backup_count: Number of backup log files to keep
        """
        self.name = name
        self.console_level = console_level
        self.file_level = file_level
        self.log_dir = Path(log_dir)
        self.enable_file_logging = enable_file_logging

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Capture everything, filter at handlers
        self.logger.handlers.clear()  # Remove any existing handlers

        # Console handler
        self._setup_console_handler()

        # File handler
        if enable_file_logging:
            self._setup_file_handler(max_bytes, backup_count)

        # Context tracking
        self.context: Dict[str, Any] = {}

    def _setup_console_handler(self):
        """Setup console (stdout) logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)

        # Console format: compact and readable
        console_format = logging.Formatter(
            fmt='[%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)

        self.logger.addHandler(console_handler)

    def _setup_file_handler(self, max_bytes: int, backup_count: int):
        """Setup file logging handler with rotation."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Log file name with timestamp
        log_file = self.log_dir / f"{self.name}.log"

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self.file_level)

        # File format: detailed with timestamp and context
        file_format = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)

        self.logger.addHandler(file_handler)

    def set_console_level(self, level: int):
        """Change console logging level."""
        self.console_level = level
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(level)

    def set_file_level(self, level: int):
        """Change file logging level."""
        self.file_level = level
        for handler in self.logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                handler.setLevel(level)

    def set_context(self, **kwargs):
        """
        Set context for subsequent log messages.

        Example:
            logger.set_context(module="GriftDetection", tier=2)
        """
        self.context.update(kwargs)

    def clear_context(self):
        """Clear all context."""
        self.context.clear()

    def _format_message(self, message: str, category: Optional[str] = None,
                       extra_context: Optional[Dict[str, Any]] = None) -> str:
        """Format message with context."""
        parts = []

        # Add category
        if category:
            parts.append(f"[{category.upper()}]")

        # Add context
        context = {**self.context, **(extra_context or {})}
        if context:
            context_str = " | ".join(f"{k}={v}" for k, v in context.items())
            parts.append(f"({context_str})")

        # Add message
        parts.append(message)

        return " ".join(parts)

    def debug(self, message: str, category: Optional[str] = None, **context):
        """Log debug message."""
        formatted = self._format_message(message, category, context)
        self.logger.debug(formatted)

    def info(self, message: str, category: Optional[str] = None, **context):
        """Log info message."""
        formatted = self._format_message(message, category, context)
        self.logger.info(formatted)

    def warning(self, message: str, category: Optional[str] = None, **context):
        """Log warning message."""
        formatted = self._format_message(message, category, context)
        self.logger.warning(formatted)

    def error(self, message: str, category: Optional[str] = None,
              exc_info: bool = False, **context):
        """Log error message."""
        formatted = self._format_message(message, category, context)
        self.logger.error(formatted, exc_info=exc_info)

    def critical(self, message: str, category: Optional[str] = None,
                 exc_info: bool = False, **context):
        """Log critical message."""
        formatted = self._format_message(message, category, context)
        self.logger.critical(formatted, exc_info=exc_info)

    def exception(self, message: str, category: Optional[str] = None, **context):
        """Log exception with traceback."""
        formatted = self._format_message(message, category, context)
        self.logger.exception(formatted)

    def log_module_trigger(self, module_name: str, prompt_excerpt: str, matched_trigger: str):
        """Log module trigger event."""
        self.debug(
            f"Module '{module_name}' triggered by '{matched_trigger}'",
            category=self.CATEGORY_TRIGGER,
            module=module_name,
            trigger=matched_trigger
        )

    def log_arbitration(self, triggered_count: int, selected_module: str, tier: int):
        """Log arbitration decision."""
        self.info(
            f"Arbitration: Selected '{selected_module}' (Tier {tier}) from {triggered_count} triggered modules",
            category=self.CATEGORY_ARBITRATION,
            selected=selected_module,
            tier=tier,
            triggered_count=triggered_count
        )

    def log_dependency_resolution(self, module_name: str, dependencies: list, resolved_count: int):
        """Log dependency resolution."""
        self.debug(
            f"Resolved {resolved_count} dependencies for '{module_name}'",
            category=self.CATEGORY_DEPENDENCY,
            module=module_name,
            dependency_count=len(dependencies)
        )

    def log_bundle_load(self, bundle_name: str, module_count: int):
        """Log bundle loading."""
        self.info(
            f"Loaded bundle '{bundle_name}' with {module_count} modules",
            category=self.CATEGORY_BUNDLE,
            bundle=bundle_name,
            modules=module_count
        )

    def log_llm_execution(self, prompt_length: int, response_length: int, execution_time: float):
        """Log LLM execution metrics."""
        self.debug(
            f"LLM execution: {prompt_length}→{response_length} chars in {execution_time:.2f}s",
            category=self.CATEGORY_LLM,
            prompt_len=prompt_length,
            response_len=response_length,
            time=execution_time
        )

    def log_performance_metric(self, operation: str, duration: float, **metrics):
        """Log performance metrics."""
        self.debug(
            f"Performance: {operation} completed in {duration:.3f}s",
            category="performance",
            operation=operation,
            duration=duration,
            **metrics
        )


class ProtocolAIError(Exception):
    """Base exception for Protocol AI errors."""

    def __init__(self, message: str, category: str = "general",
                 context: Optional[Dict[str, Any]] = None):
        """
        Initialize Protocol AI error.

        Args:
            message: Error message
            category: Error category
            context: Additional context
        """
        self.message = message
        self.category = category
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()

        super().__init__(self.format_error())

    def format_error(self) -> str:
        """Format error with context."""
        parts = [f"[{self.category.upper()}] {self.message}"]

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'category': self.category,
            'context': self.context,
            'timestamp': self.timestamp
        }


class ModuleError(ProtocolAIError):
    """Error related to module operations."""

    def __init__(self, message: str, module_name: Optional[str] = None, **context):
        context['module'] = module_name
        super().__init__(message, category="module", context=context)


class TriggerError(ProtocolAIError):
    """Error related to trigger analysis."""

    def __init__(self, message: str, **context):
        super().__init__(message, category="trigger", context=context)


class DependencyError(ProtocolAIError):
    """Error related to dependency resolution."""

    def __init__(self, message: str, **context):
        super().__init__(message, category="dependency", context=context)


class ArbitrationError(ProtocolAIError):
    """Error related to module arbitration."""

    def __init__(self, message: str, **context):
        super().__init__(message, category="arbitration", context=context)


class LLMError(ProtocolAIError):
    """Error related to LLM execution."""

    def __init__(self, message: str, **context):
        super().__init__(message, category="llm", context=context)


class BundleError(ProtocolAIError):
    """Error related to bundle operations."""

    def __init__(self, message: str, bundle_name: Optional[str] = None, **context):
        context['bundle'] = bundle_name
        super().__init__(message, category="bundle", context=context)


# Global logger instance
_global_logger: Optional[ProtocolAILogger] = None


def get_logger(name: str = "protocol_ai",
               console_level: int = logging.INFO,
               file_level: int = logging.DEBUG,
               **kwargs) -> ProtocolAILogger:
    """
    Get or create the global Protocol AI logger.

    Args:
        name: Logger name
        console_level: Console logging level
        file_level: File logging level
        **kwargs: Additional arguments for ProtocolAILogger

    Returns:
        ProtocolAILogger instance
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = ProtocolAILogger(
            name=name,
            console_level=console_level,
            file_level=file_level,
            **kwargs
        )

    return _global_logger


def set_verbosity(level: str):
    """
    Set global verbosity level.

    Args:
        level: One of 'debug', 'info', 'warning', 'error', 'critical'
    """
    logger = get_logger()

    level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    log_level = level_map.get(level.lower(), logging.INFO)
    logger.set_console_level(log_level)
