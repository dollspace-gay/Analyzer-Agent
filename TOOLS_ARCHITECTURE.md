# Tools System Architecture

The Protocol AI system includes a modular tools framework that allows the LLM to interact with external systems and execute code safely.

## Overview

The tools system provides:
- **Modular plugin architecture** for easy tool addition
- **Sandboxed code execution** restricted to `./sandbox/` folder
- **Web search capabilities** for real-time information retrieval
- **File system operations** within the sandbox environment
- **GUI integration** for tool management and result visualization

## Implementation Status

✅ **COMPLETED (Agent-1ug)**: Core tools architecture
- Base Tool class with async execution
- ToolRegistry for registration and execution
- ToolLoader for plugin discovery
- Input validation and error handling
- Execution logging and audit trail
- Comprehensive test suite (100% passing)

## Architecture

### Base Tool Interface

The base `Tool` class is implemented in `protocol_ai.py`:

```python
class Tool:
    """Base class for all tools."""

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.enabled = True

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        raise NotImplementedError(f"Tool '{self.name}' must implement execute() method")

    def validate_input(self, **kwargs) -> tuple[bool, str]:
        """Validate input parameters before execution."""
        # Validates required parameters against schema
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's schema for LLM consumption."""
        pass
```

### ToolResult Dataclass

```python
@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    tool_name: str
    output: Any = None
    error: str = ""
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### ToolRegistry

Manages tool registration, lookup, and execution:

```python
class ToolRegistry:
    """Registry for managing available tools."""

    def register(self, tool: Tool) -> None
    def unregister(self, tool_name: str) -> None
    def get_tool(self, tool_name: str) -> Optional[Tool]
    def get_enabled_tools(self) -> List[Tool]
    def get_tools_schema(self) -> List[Dict[str, Any]]
    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult
    def get_execution_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]
```

Features:
- Prevents duplicate registration
- Validates inputs before execution
- Logs all executions with timestamps
- Handles errors gracefully
- Supports enable/disable per tool

### ToolLoader

Discovers and loads tool plugins from filesystem:

```python
class ToolLoader:
    """Loads tools from a directory structure."""

    def __init__(self, tools_dir: str = "./tools")
    def discover_tools(self) -> List[str]
    def load_tools(self, registry: ToolRegistry) -> int
```

Plugin discovery:
- Scans `./tools/` directory for Python files
- Dynamically imports modules
- Finds Tool subclasses
- Auto-registers discovered tools

### Tool Categories

#### 1. Web Search Tool
- **Purpose**: Search the web for current information
- **Engines**: DuckDuckGo, Google, Bing (configurable)
- **Features**:
  - Rate limiting to prevent abuse
  - Result caching for efficiency
  - Structured output (title, snippet, URL)
  - Error handling and retry logic

#### 2. Sandboxed Code Execution
- **Purpose**: Execute Python code safely
- **Restrictions**:
  - File operations limited to `./sandbox/` folder only
  - Restricted import list (no os, subprocess, network access)
  - CPU and memory limits enforced
  - Execution timeout (default: 30 seconds)
- **Security**:
  - Path traversal prevention
  - No network access
  - All file operations logged
  - Automatic cleanup of temp files

#### 3. File System Tool
- **Purpose**: Read/write files within sandbox
- **Operations**:
  - Read text/binary files
  - Write files with size limits
  - List directory contents
  - Search files by pattern
  - Delete files (sandbox only)
- **Validation**:
  - Extension whitelist/blacklist
  - File size limits (default: 10MB)
  - Path sanitization

## Sandbox Environment

### Folder Structure
```
./sandbox/
├── inputs/          # User-provided input files
├── outputs/         # Generated output files
├── temp/            # Temporary files (auto-cleanup)
└── logs/            # Execution logs
```

### Security Rules
1. **No path traversal**: Cannot access `../` or absolute paths outside sandbox
2. **Read-only system**: Cannot modify files outside sandbox
3. **Resource limits**:
   - Max file size: 10MB
   - Max execution time: 30s
   - Max memory: 512MB
4. **Restricted imports**: Whitelist of safe Python modules only

## Integration with LLM

### Tool Invocation Flow

1. **LLM generates tool request**:
   ```json
   {
     "tool": "web_search",
     "params": {
       "query": "latest AI governance frameworks",
       "max_results": 5
     }
   }
   ```

2. **Orchestrator detects tool invocation**
3. **Tool framework executes the request**
4. **Results fed back into LLM prompt**:
   ```
   Tool: web_search
   Query: latest AI governance frameworks
   Results:
   1. [Title] - [Snippet] - [URL]
   2. [Title] - [Snippet] - [URL]
   ...
   ```

5. **LLM generates response using tool results**

### Multi-Turn Tool Usage

The system supports chaining multiple tools:

```
User: "Search for Python sorting algorithms and implement quicksort"

Turn 1: LLM → web_search("Python quicksort algorithm")
        Tool → Returns search results

Turn 2: LLM → code_execute("implement quicksort based on results")
        Tool → Executes code in sandbox, returns output

Turn 3: LLM → Explains implementation to user
```

## GUI Integration

### Tools Panel Features
- Enable/disable individual tools
- Configure tool settings (timeouts, rate limits, etc.)
- View tool execution history
- Monitor resource usage
- Clear sandbox contents

### Result Visualization
- Search results displayed as cards with expand/collapse
- Code execution output with syntax highlighting
- File operations log with tree view
- Error messages with helpful debugging info

## Future Tools (Extensible)

The modular architecture supports adding:
- Database query tool
- API request tool
- Image generation/manipulation
- Data visualization
- Mathematical computation (SymPy, NumPy)
- External API integrations

## Security Considerations

1. **Input Validation**: All tool inputs validated before execution
2. **Output Sanitization**: Tool outputs sanitized before display
3. **Audit Logging**: All tool executions logged with timestamps
4. **Resource Monitoring**: CPU/memory/disk usage tracked
5. **Automatic Cleanup**: Sandbox cleaned on errors or timeouts

## Development Roadmap

See BD issues:
- [Agent-1ug](Agent-1ug): Tools architecture design (P0)
- [Agent-7k1](Agent-7k1): Web search tool (P1)
- [Agent-bc4](Agent-bc4): Sandboxed code execution (P1)
- [Agent-wjm](Agent-wjm): Sandbox management (P1)
- [Agent-dh2](Agent-dh2): LLM integration (P1)
- [Agent-3qc](Agent-3qc): File system tool (P2)
- [Agent-a2u](Agent-a2u): Tools GUI panel (P2)
- [Agent-k0k](Agent-k0k): Result visualization (P2)
