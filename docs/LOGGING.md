# Protocol AI Logging and Error Handling

Comprehensive logging and error handling framework for Protocol AI Governance Layer.

## Features

- **Multi-level logging**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Dual output**: Independent console and file logging levels
- **Structured logging**: Context tracking and categorization
- **File rotation**: Automatic log file management
- **Custom errors**: Typed exceptions with context capture
- **Performance metrics**: Operation timing and profiling
- **Global logger**: Singleton pattern for system-wide logging

## Quick Start

### Basic Usage

```python
from protocol_ai_logging import get_logger

# Get the global logger
logger = get_logger()

# Log messages at different levels
logger.debug("Detailed debugging information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical failure")
```

### Set Verbosity

```python
from protocol_ai_logging import set_verbosity

# Set verbosity level
set_verbosity('debug')   # Most verbose
set_verbosity('info')    # Standard (default)
set_verbosity('warning') # Only warnings and errors
set_verbosity('error')   # Only errors
set_verbosity('critical') # Only critical
```

### Using Command-Line Tool

```bash
# Set verbosity from command line
python set_verbosity.py debug
python set_verbosity.py info
python set_verbosity.py error
```

## Advanced Features

### Context Tracking

Track context across multiple log messages:

```python
logger = get_logger()

# Set persistent context
logger.set_context(module="GriftDetection", tier=2)

logger.info("Starting analysis")  # Includes module and tier
logger.debug("Checking patterns")  # Includes module and tier

# Clear context
logger.clear_context()
```

### Categorized Logging

Use category-specific methods:

```python
# Module trigger logging
logger.log_module_trigger("GriftDetection", "test prompt", "grift")

# Arbitration logging
logger.log_arbitration(5, "GriftDetection", 2)

# Dependency resolution
logger.log_dependency_resolution("Module", ["Dep1"], 1)

# Bundle operations
logger.log_bundle_load("governance", 5)

# LLM execution
logger.log_llm_execution(1000, 500, 1.5)

# Performance metrics
logger.log_performance_metric("operation_name", 0.123)
```

### Structured Error Handling

Use typed exceptions with context:

```python
from protocol_ai_logging import ModuleError, DependencyError, BundleError

# Module errors
raise ModuleError(
    "Failed to load module",
    module_name="TestModule",
    file_path="./modules/test.yaml"
)

# Dependency errors
raise DependencyError(
    "Circular dependency detected",
    module="ModuleA",
    depends_on="ModuleB"
)

# Bundle errors
raise BundleError(
    "Bundle not found",
    bundle_name="missing"
)
```

### File Logging Configuration

```python
from protocol_ai_logging import ProtocolAILogger
import logging

logger = ProtocolAILogger(
    name="custom_logger",
    console_level=logging.INFO,  # Console shows INFO+
    file_level=logging.DEBUG,    # File captures DEBUG+
    log_dir="./logs",
    enable_file_logging=True,
    max_bytes=10_000_000,  # 10MB before rotation
    backup_count=5          # Keep 5 backup files
)
```

## Error Classes

### Base Error

```python
class ProtocolAIError(Exception):
    """Base exception for all Protocol AI errors"""
```

### Specialized Errors

- `ModuleError`: Module loading and execution errors
- `TriggerError`: Trigger analysis errors
- `DependencyError`: Dependency resolution errors
- `ArbitrationError`: Module arbitration errors
- `LLMError`: LLM execution errors
- `BundleError`: Bundle loading errors

### Error Context

All errors capture context automatically:

```python
try:
    raise ModuleError("Error message", module="Test", tier=2)
except ModuleError as e:
    print(e.message)    # "Error message"
    print(e.category)   # "module"
    print(e.context)    # {'module': 'Test', 'tier': 2}
    print(e.timestamp)  # ISO timestamp
    print(e.to_dict())  # Full error as dictionary
```

## Log Format

### Console Output

Compact and readable:
```
[INFO] Message text
[WARNING] Warning message
[ERROR] Error occurred
```

### File Output

Detailed with timestamps:
```
2025-11-13 19:30:45 | INFO     | protocol_ai | Message text
2025-11-13 19:30:46 | WARNING  | protocol_ai | Warning message
2025-11-13 19:30:47 | ERROR    | protocol_ai | Error occurred
```

### With Context

```
[INFO] (module=GriftDetection | tier=2) Module triggered
[DEBUG] [TRIGGER] (module=GriftDetection | trigger=grift) Matched pattern
[ERROR] [ERROR] (file_path=./test.yaml | module=Test) Failed to load
```

## Log Categories

- `system`: System-level events
- `module`: Module operations
- `trigger`: Trigger analysis
- `arbitration`: Module arbitration
- `llm`: LLM execution
- `dependency`: Dependency resolution
- `bundle`: Bundle operations
- `test`: Test execution
- `error`: Error conditions
- `performance`: Performance metrics

## Integration Examples

### With Module Loader

```python
from protocol_ai import ModuleLoader
from protocol_ai_logging import get_logger

logger = get_logger()

try:
    loader = ModuleLoader()
    modules = loader.load_modules()
    logger.info(f"Loaded {len(modules)} modules", category="system")
except Exception as e:
    logger.exception("Module loading failed", category="error")
```

### With Orchestrator

```python
from protocol_ai import Orchestrator
from protocol_ai_logging import get_logger

logger = get_logger()
logger.set_context(operation="orchestration")

try:
    result = await orchestrator.process_prompt(prompt)
    logger.info("Processing complete", category="system")
except Exception as e:
    logger.error(f"Processing failed: {e}", category="error", exc_info=True)
finally:
    logger.clear_context()
```

## Debugging Tips

### Enable Debug Logging

```python
set_verbosity('debug')
```

### Focus on Specific Module

```python
logger = get_logger()
logger.set_context(module="GriftDetection")
# All subsequent logs will include module context
```

### Track Performance

```python
import time

start = time.time()
# ... operation ...
duration = time.time() - start

logger.log_performance_metric("operation_name", duration)
```

### Capture Full Tracebacks

```python
try:
    # risky operation
    pass
except Exception as e:
    logger.exception("Operation failed", category="error")
    # Full traceback logged automatically
```

## Best Practices

1. **Use appropriate log levels**
   - DEBUG: Detailed diagnostic information
   - INFO: General informational messages
   - WARNING: Warning messages for potential issues
   - ERROR: Error messages for failures
   - CRITICAL: Critical failures requiring immediate attention

2. **Add context**
   ```python
   logger.set_context(module="ModuleName", operation="operation")
   ```

3. **Use categories**
   ```python
   logger.info("Message", category="system")
   ```

4. **Use typed exceptions**
   ```python
   raise ModuleError("Error", module_name="Test")
   ```

5. **Log performance metrics**
   ```python
   logger.log_performance_metric("operation", duration)
   ```

6. **Clean up context**
   ```python
   try:
       logger.set_context(...)
       # operations
   finally:
       logger.clear_context()
   ```

## File Locations

- **Log files**: `./logs/protocol_ai.log`
- **Backup logs**: `./logs/protocol_ai.log.1`, `.2`, etc.
- **Configuration**: Set in logger initialization

## Testing

Run logging framework tests:

```bash
# Unit tests
python tests/test_logging_framework.py

# Integration tests
python tests/test_logging_integration.py
```

## Configuration

### Environment-Based Configuration

```python
import os
import logging

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
set_verbosity(log_level.lower())
```

### Production vs Development

```python
# Development
set_verbosity('debug')

# Production
set_verbosity('warning')
```
