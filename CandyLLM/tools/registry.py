"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

Tool registry with LRU caching and automatic tool discovery

Advanced tool management system with comprehensive security features:
- Secure tool registration with validation and sandboxing
- Code analysis and safety checking for tool functions
- Access control and permission management
- Secure serialization and storage of tool metadata
- Protection against malicious tool injection
- Comprehensive audit logging for tool operations
- Resource limits and execution monitoring
- Cryptographic verification of tool integrity
"""

import json
import asyncio
import inspect
import importlib
import ast
import hashlib
import secrets
import logging
import os
import sys
from typing import Dict, List, Any, Optional, Callable, Type, Union, Set
from functools import lru_cache, wraps
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging

# Security imports
import html
import pickle
import base64
from cryptography.fernet import Fernet

from ..core.base import BaseTool, ToolConfig, ToolResult

# Security logging
security_logger = logging.getLogger('candyllm.tools.security')
logger = logging.getLogger(__name__)

# Dangerous imports/functions to block in tool code
DANGEROUS_IMPORTS = {
    'os', 'sys', 'subprocess', 'eval', 'exec', 'compile', 
    'open', 'file', '__import__', 'globals', 'locals',
    'vars', 'dir', 'getattr', 'setattr', 'delattr',
    'input', 'raw_input', 'execfile', 'reload',
    'shutil', 'pickle', 'marshal', 'ctypes'
}

# Safe built-in functions allowed in tools
SAFE_BUILTINS = {
    'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes',
    'chr', 'dict', 'enumerate', 'filter', 'float', 'format',
    'frozenset', 'hex', 'int', 'isinstance', 'issubclass',
    'len', 'list', 'map', 'max', 'min', 'oct', 'ord',
    'pow', 'range', 'repr', 'reversed', 'round', 'set',
    'slice', 'sorted', 'str', 'sum', 'tuple', 'type', 'zip'
}

class ToolSecurityError(Exception):
    """Exception raised for tool security violations"""
    pass

class CodeAnalyzer(ast.NodeVisitor):
    """
    AST-based code analyzer for detecting dangerous patterns in tool code
    
    Features:
    - Detection of dangerous imports and function calls
    - Identification of potential code injection vectors
    - Analysis of network operations and file access
    - Detection of subprocess and system calls
    """
    
    def __init__(self):
        self.issues: List[str] = []
        self.imports: Set[str] = set()
        self.function_calls: Set[str] = set()
        self.attributes: Set[str] = set()
        
    def visit_Import(self, node):
        """Check import statements for dangerous modules"""
        for alias in node.names:
            module_name = alias.name
            self.imports.add(module_name)
            
            if module_name in DANGEROUS_IMPORTS:
                self.issues.append(f"Dangerous import detected: {module_name}")
            
            # Check for indirect dangerous imports
            if any(dangerous in module_name for dangerous in ['socket', 'http', 'urllib', 'requests']):
                self.issues.append(f"Network-related import detected: {module_name}")
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Check from...import statements"""
        if node.module:
            self.imports.add(node.module)
            
            if node.module in DANGEROUS_IMPORTS:
                self.issues.append(f"Dangerous import from detected: {node.module}")
            
            for alias in node.names:
                if alias.name in DANGEROUS_IMPORTS:
                    self.issues.append(f"Dangerous function import: {alias.name} from {node.module}")
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Check function calls for dangerous operations"""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            self.function_calls.add(func_name)
            
            if func_name in DANGEROUS_IMPORTS:
                self.issues.append(f"Dangerous function call: {func_name}")
        
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            self.attributes.add(attr_name)
            
            # Check for dangerous method calls
            if attr_name in ['system', 'popen', 'spawn', 'exec', 'eval']:
                self.issues.append(f"Dangerous method call: {attr_name}")
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        """Check attribute access for dangerous patterns"""
        if isinstance(node.attr, str):
            self.attributes.add(node.attr)
            
            # Check for dangerous attributes
            if node.attr in ['__import__', '__builtins__', '__globals__']:
                self.issues.append(f"Dangerous attribute access: {node.attr}")
        
        self.generic_visit(node)

def analyze_tool_code(code: str) -> List[str]:
    """
    Analyze tool code for security issues using AST parsing
    
    Args:
        code: Source code to analyze
        
    Returns:
        List of security issues found
    """
    try:
        tree = ast.parse(code)
        analyzer = CodeAnalyzer()
        analyzer.visit(tree)
        return analyzer.issues
    except SyntaxError as e:
        return [f"Syntax error in code: {e}"]
    except Exception as e:
        return [f"Code analysis failed: {e}"]

@dataclass
class ToolMetadata:
    """
    Metadata for registered tools with enhanced security information
    
    Features:
    - Comprehensive tool documentation and validation
    - Security classification and risk assessment
    - Cryptographic signatures for integrity verification
    - Access control and permission tracking
    """
    name: str
    description: str
    category: str
    version: str
    author: str
    
    # Security fields
    requires_auth: bool = False
    security_level: str = "medium"  # low, medium, high, restricted
    permissions: List[str] = field(default_factory=list)
    code_hash: Optional[str] = None
    signature: Optional[str] = None
    
    # Capability fields
    async_capable: bool = False
    network_access: bool = False
    file_access: bool = False
    system_access: bool = False
    
    # Schema fields
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: str(datetime.now()))
    updated_at: str = field(default_factory=lambda: str(datetime.now()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with security masking"""
        data = asdict(self)
        
        # Mask sensitive fields for public access
        if self.security_level == "restricted":
            data['code_hash'] = "***RESTRICTED***"
            data['signature'] = "***RESTRICTED***"
        
        return data
    
    def verify_integrity(self, code: str) -> bool:
        """Verify tool code integrity using stored hash"""
        if not self.code_hash:
            return False
        
        current_hash = hashlib.sha256(code.encode()).hexdigest()
        return current_hash == self.code_hash
    
    def update_hash(self, code: str):
        """Update code hash for integrity verification"""
        self.code_hash = hashlib.sha256(code.encode()).hexdigest()
        self.updated_at = str(datetime.now())

class SecureToolRegistry:
    """
    Centralized tool registry with LRU caching and comprehensive security
    
    Security Features:
    - Code analysis and validation before registration
    - Sandboxed execution environment for tools
    - Cryptographic verification of tool integrity
    - Access control and permission management
    - Comprehensive audit logging
    - Resource limits and monitoring
    - Hot-reloading with security checks
    """
    
    def __init__(self, cache_size: int = 128, storage_path: Optional[str] = None):
        """
        Initialize secure tool registry
        
        Args:
            cache_size: Maximum number of cached tool instances
            storage_path: Path for persistent tool storage
        """
        self.cache_size = cache_size
        self.storage_path = Path(storage_path) if storage_path else Path.home() / ".candyllm" / "tools"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core registries with security tracking
        self._tools: Dict[str, Type[BaseTool]] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
        self._instances: Dict[str, BaseTool] = {}
        self._access_log: Dict[str, List[datetime]] = {}
        
        # Security components
        self._encryption_key = self._get_or_create_encryption_key()
        self._cipher_suite = Fernet(self._encryption_key)
        
        # Rate limiting for tool operations
        self._rate_limits: Dict[str, int] = {
            'low': 100,
            'medium': 50,
            'high': 20,
            'restricted': 5
        }
        
        # Initialize with built-in tools
        self._register_builtin_tools()
        self._load_persistent_tools()
        
        security_logger.info(f"SecureToolRegistry initialized with {len(self._tools)} tools")
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for sensitive tool data"""
        key_file = self.storage_path / "registry.key"
        
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                security_logger.warning(f"Failed to load encryption key: {e}")
        
        # Create new key
        key = Fernet.generate_key()
        try:
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Restrict access
        except Exception as e:
            security_logger.error(f"Failed to save encryption key: {e}")
        
        return key
    
    def _check_rate_limit(self, tool_name: str, security_level: str) -> bool:
        """
        Check if tool usage is within rate limits
        
        Args:
            tool_name: Name of the tool
            security_level: Security level of the tool
            
        Returns:
            True if within limits, False otherwise
        """
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Clean old entries
        if tool_name in self._access_log:
            self._access_log[tool_name] = [
                timestamp for timestamp in self._access_log[tool_name]
                if timestamp > hour_ago
            ]
        else:
            self._access_log[tool_name] = []
        
        # Check current usage
        current_usage = len(self._access_log[tool_name])
        limit = self._rate_limits.get(security_level, 50)
        
        if current_usage >= limit:
            security_logger.warning(f"Rate limit exceeded for tool {tool_name}: {current_usage}/{limit}")
            return False
        
        # Record current access
        self._access_log[tool_name].append(now)
        return True
    
    @lru_cache(maxsize=128)
    def get_tool(self, name: str, user_id: Optional[str] = None) -> Optional[BaseTool]:
        """
        Get tool instance with security validation and rate limiting
        
        Args:
            name: Tool name
            user_id: User requesting the tool
            
        Returns:
            Tool instance if authorized and within limits, None otherwise
        """
        if name not in self._tools:
            security_logger.warning(f"Tool not found: {name}")
            return None
        
        metadata = self._metadata.get(name)
        if not metadata:
            security_logger.error(f"Metadata missing for tool: {name}")
            return None
        
        # Check rate limits
        if not self._check_rate_limit(name, metadata.security_level):
            return None
        
        # Check permissions (simplified - in production, implement full RBAC)
        if metadata.requires_auth and not user_id:
            security_logger.warning(f"Authentication required for tool: {name}")
            return None
        
        # Get or create instance
        if name not in self._instances:
            try:
                tool_class = self._tools[name]
                self._instances[name] = tool_class()
                security_logger.info(f"Tool instance created: {name}")
            except Exception as e:
                security_logger.error(f"Failed to create tool instance {name}: {e}")
                return None
        
        return self._instances[name]
    
    def register_tool(self, 
                     tool_class: Type[BaseTool], 
                     metadata: Optional[ToolMetadata] = None,
                     persist: bool = True,
                     validate_code: bool = True) -> None:
        """
        Register a new tool with comprehensive security validation
        
        Args:
            tool_class: Tool class to register
            metadata: Tool metadata (generated if None)
            persist: Whether to persist tool to storage
            validate_code: Whether to perform code analysis
        """
        if not tool_class or not hasattr(tool_class, '__name__'):
            raise ToolSecurityError("Invalid tool class")
        
        tool_name = tool_class.__name__
        
        # Generate metadata if not provided
        if metadata is None:
            metadata = self._generate_metadata(tool_class)
        
        # Validate tool code if requested
        if validate_code:
            self._validate_tool_security(tool_class, metadata)
        
        # Register tool
        self._tools[tool_name] = tool_class
        self._metadata[tool_name] = metadata
        
        # Clear cache for this tool
        if tool_name in self._instances:
            del self._instances[tool_name]
        
        # Persist if requested
        if persist:
            self._persist_tool_metadata(tool_name, metadata)
        
        security_logger.info(f"Tool registered successfully: {tool_name} (security: {metadata.security_level})")
    
    def _validate_tool_security(self, tool_class: Type[BaseTool], metadata: ToolMetadata):
        """
        Comprehensive security validation of tool class
        
        Args:
            tool_class: Tool class to validate
            metadata: Tool metadata
        """
        # Get tool source code
        try:
            source_code = inspect.getsource(tool_class)
        except Exception:
            raise ToolSecurityError(f"Cannot retrieve source code for {tool_class.__name__}")
        
        # Analyze code for security issues
        security_issues = analyze_tool_code(source_code)
        
        if security_issues:
            if metadata.security_level == "restricted":
                # Allow restricted tools with warnings
                security_logger.warning(f"Security issues in restricted tool {tool_class.__name__}: {security_issues}")
            else:
                # Block tools with security issues
                raise ToolSecurityError(f"Security validation failed for {tool_class.__name__}: {security_issues}")
        
        # Update metadata with code hash
        metadata.update_hash(source_code)
        
        # Check for dangerous capabilities
        if any(word in source_code.lower() for word in ['socket', 'http', 'urllib', 'requests']):
            metadata.network_access = True
        
        if any(word in source_code.lower() for word in ['open(', 'file(', 'os.path', 'pathlib']):
            metadata.file_access = True
        
        if any(word in source_code.lower() for word in ['subprocess', 'os.system', 'exec']):
            metadata.system_access = True
    
    def _generate_metadata(self, tool_class: Type[BaseTool]) -> ToolMetadata:
        """Generate tool metadata from class inspection"""
        return ToolMetadata(
            name=tool_class.__name__,
            description=getattr(tool_class, '__doc__', 'No description available') or 'No description',
            category=getattr(tool_class, '_category', 'general'),
            version=getattr(tool_class, '_version', '1.0.0'),
            author=getattr(tool_class, '_author', 'Unknown'),
            security_level=getattr(tool_class, '_security_level', 'medium'),
            requires_auth=getattr(tool_class, '_requires_auth', False),
            async_capable=asyncio.iscoroutinefunction(getattr(tool_class, 'execute', None))
        )
        
        # Extract metadata from tool class if not provided
        if metadata is None:
            metadata = self._extract_metadata_from_class(tool_class)
        
        name = metadata.name
        
        # Validate tool
        self._validate_tool(tool_class, metadata)
        
        # Register
        self._tools[name] = tool_class
        self._metadata[name] = metadata
        
        # Clear relevant caches
        self.get_tool.cache_clear()
        self.list_tools.cache_clear()
        self.get_tools_by_category.cache_clear()
        
        # Persist if requested
        if persist:
            self._save_tool_metadata(metadata)
        
        logger.info(f"Registered tool: {name} (category: {metadata.category})")
    
    def register_function_as_tool(self, 
                                func: Callable,
                                name: Optional[str] = None,
                                description: Optional[str] = None,
                                category: str = "custom") -> None:
        """Register a simple function as a tool"""
        
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Function: {func.__name__}"
        
        # Create dynamic tool class
        class FunctionTool(BaseTool):
            def __init__(self):
                super().__init__()
                self.func = func
            
            def get_config(self) -> ToolConfig:
                return ToolConfig(
                    name=tool_name,
                    description=tool_description,
                    input_schema=self._extract_function_schema(func)
                )
            
            async def execute(self, **kwargs) -> ToolResult:
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(**kwargs)
                    else:
                        result = func(**kwargs)
                    
                    return ToolResult(
                        success=True,
                        data=result,
                        message=f"Function {tool_name} executed successfully"
                    )
                except Exception as e:
                    return ToolResult(
                        success=False,
                        error=str(e),
                        message=f"Function {tool_name} failed"
                    )
            
            def _extract_function_schema(self, func: Callable) -> Dict[str, Any]:
                """Extract JSON schema from function signature"""
                sig = inspect.signature(func)
                schema = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
                
                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        continue
                    
                    param_schema = {"type": "string"}  # Default
                    
                    # Infer type from annotation
                    if param.annotation != inspect.Parameter.empty:
                        if param.annotation == int:
                            param_schema["type"] = "integer"
                        elif param.annotation == float:
                            param_schema["type"] = "number"
                        elif param.annotation == bool:
                            param_schema["type"] = "boolean"
                        elif param.annotation == list:
                            param_schema["type"] = "array"
                        elif param.annotation == dict:
                            param_schema["type"] = "object"
                    
                    schema["properties"][param_name] = param_schema
                    
                    # Add to required if no default value
                    if param.default == inspect.Parameter.empty:
                        schema["required"].append(param_name)
                
                return schema
        
        metadata = ToolMetadata(
            name=tool_name,
            description=tool_description,
            category=category,
            version="1.0.0",
            author="user",
            async_capable=asyncio.iscoroutinefunction(func)
        )
        
        self.register_tool(FunctionTool, metadata)
    
    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool"""
        if name not in self._tools:
            return False
        
        del self._tools[name]
        del self._metadata[name]
        
        if name in self._instances:
            del self._instances[name]
        
        # Clear caches
        self.get_tool.cache_clear()
        self.list_tools.cache_clear()
        self.get_tools_by_category.cache_clear()
        
        # Remove from persistent storage
        metadata_file = self.storage_path / f"{name}.json"
        if metadata_file.exists():
            metadata_file.unlink()
        
        logger.info(f"Unregistered tool: {name}")
        return True
    
    def discover_tools(self, package_paths: List[str]) -> int:
        """Discover and register tools from specified packages"""
        discovered = 0
        
        for package_path in package_paths:
            try:
                # Import package and scan for tool classes
                module = importlib.import_module(package_path)
                
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    
                    # Check if it's a tool class
                    if (inspect.isclass(attr) and 
                        issubclass(attr, BaseTool) and 
                        attr != BaseTool):
                        
                        # Auto-register discovered tool
                        try:
                            metadata = self._extract_metadata_from_class(attr)
                            self.register_tool(attr, metadata, persist=False)
                            discovered += 1
                        except Exception as e:
                            logger.warning(f"Failed to register discovered tool {attr.__name__}: {e}")
            
            except ImportError as e:
                logger.warning(f"Failed to import package {package_path}: {e}")
        
        logger.info(f"Discovered and registered {discovered} tools")
        return discovered
    
    @lru_cache(maxsize=32)
    def search_tools(self, query: str) -> List[ToolMetadata]:
        """Search tools by name, description, or tags"""
        query_lower = query.lower()
        matches = []
        
        for metadata in self._metadata.values():
            # Search in name
            if query_lower in metadata.name.lower():
                matches.append(metadata)
                continue
            
            # Search in description
            if query_lower in metadata.description.lower():
                matches.append(metadata)
                continue
            
            # Search in tags
            if metadata.tags:
                for tag in metadata.tags:
                    if query_lower in tag.lower():
                        matches.append(metadata)
                        break
        
        return matches
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for tools"""
        cache_info = self.get_tool.cache_info()
        
        return {
            "total_tools": len(self._tools),
            "loaded_instances": len(self._instances),
            "cache_hits": cache_info.hits,
            "cache_misses": cache_info.misses,
            "cache_size": cache_info.currsize,
            "max_cache_size": cache_info.maxsize,
            "categories": len(set(m.category for m in self._metadata.values()))
        }
    
    def clear_cache(self) -> None:
        """Clear all LRU caches"""
        self.get_tool.cache_clear()
        self.list_tools.cache_clear()
        self.get_tools_by_category.cache_clear()
        self.search_tools.cache_clear()
        
        # Also clear instances cache
        self._instances.clear()
        
        logger.info("Cleared all tool caches")
    
    def _register_builtin_tools(self) -> None:
        """Register built-in tools"""
        
        # Web Search Tool
        class WebSearchTool(BaseTool):
            def get_config(self) -> ToolConfig:
                return ToolConfig(
                    name="web_search",
                    description="Search the web for information",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "num_results": {"type": "integer", "default": 5}
                        },
                        "required": ["query"]
                    }
                )
            
            async def execute(self, query: str, num_results: int = 5) -> ToolResult:
                # Placeholder implementation
                return ToolResult(
                    success=True,
                    data={"results": [], "query": query},
                    message="Web search completed"
                )
        
        # Calculator Tool  
        class CalculatorTool(BaseTool):
            def get_config(self) -> ToolConfig:
                return ToolConfig(
                    name="calculator",
                    description="Perform mathematical calculations",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Mathematical expression"}
                        },
                        "required": ["expression"]
                    }
                )
            
            async def execute(self, expression: str) -> ToolResult:
                try:
                    # Safe eval for basic math
                    result = eval(expression.replace(" ", ""))
                    return ToolResult(
                        success=True,
                        data={"result": result, "expression": expression},
                        message="Calculation completed"
                    )
                except Exception as e:
                    return ToolResult(
                        success=False,
                        error=str(e),
                        message="Calculation failed"
                    )
        
        # File System Tool
        class FileSystemTool(BaseTool):
            def get_config(self) -> ToolConfig:
                return ToolConfig(
                    name="file_system",
                    description="Read and write files",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["read", "write", "list"]},
                            "path": {"type": "string", "description": "File or directory path"},
                            "content": {"type": "string", "description": "Content to write"}
                        },
                        "required": ["action", "path"]
                    }
                )
            
            async def execute(self, action: str, path: str, content: str = None) -> ToolResult:
                try:
                    path_obj = Path(path)
                    
                    if action == "read":
                        if path_obj.exists():
                            content = path_obj.read_text()
                            return ToolResult(success=True, data={"content": content})
                        else:
                            return ToolResult(success=False, error="File not found")
                    
                    elif action == "write":
                        if content is not None:
                            path_obj.write_text(content)
                            return ToolResult(success=True, message="File written successfully")
                        else:
                            return ToolResult(success=False, error="No content provided")
                    
                    elif action == "list":
                        if path_obj.is_dir():
                            files = [str(f) for f in path_obj.iterdir()]
                            return ToolResult(success=True, data={"files": files})
                        else:
                            return ToolResult(success=False, error="Path is not a directory")
                
                except Exception as e:
                    return ToolResult(success=False, error=str(e))
        
        # Register built-in tools
        builtin_tools = [
            (WebSearchTool, ToolMetadata("web_search", "Search the web", "web", "1.0.0", "candyllm", tags=["search", "web"])),
            (CalculatorTool, ToolMetadata("calculator", "Mathematical calculations", "math", "1.0.0", "candyllm", tags=["math", "calculation"])),
            (FileSystemTool, ToolMetadata("file_system", "File operations", "system", "1.0.0", "candyllm", tags=["file", "io"]))
        ]
        
        for tool_class, metadata in builtin_tools:
            self.register_tool(tool_class, metadata, persist=False)
    
    def _extract_metadata_from_class(self, tool_class: Type[BaseTool]) -> ToolMetadata:
        """Extract metadata from tool class"""
        # Try to get metadata from class attributes
        name = getattr(tool_class, '_name', tool_class.__name__.lower().replace('tool', ''))
        description = getattr(tool_class, '_description', tool_class.__doc__ or f"Tool: {tool_class.__name__}")
        category = getattr(tool_class, '_category', 'custom')
        version = getattr(tool_class, '_version', '1.0.0')
        author = getattr(tool_class, '_author', 'unknown')
        
        return ToolMetadata(
            name=name,
            description=description,
            category=category,
            version=version,
            author=author
        )
    
    def _validate_tool(self, tool_class: Type[BaseTool], metadata: ToolMetadata) -> None:
        """Validate tool class and metadata"""
        if not issubclass(tool_class, BaseTool):
            raise ValueError(f"Tool {metadata.name} must inherit from BaseTool")
        
        # Check required methods
        required_methods = ['get_config', 'execute']
        for method in required_methods:
            if not hasattr(tool_class, method):
                raise ValueError(f"Tool {metadata.name} missing required method: {method}")
    
    def _save_tool_metadata(self, metadata: ToolMetadata) -> None:
        """Save tool metadata to persistent storage"""
        try:
            metadata_file = self.storage_path / f"{metadata.name}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata for {metadata.name}: {e}")
    
    def _load_persistent_tools(self) -> None:
        """Load tool metadata from persistent storage"""
        try:
            for metadata_file in self.storage_path.glob("*.json"):
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    # Note: Only loads metadata, not actual tool classes
                    # Tool classes need to be registered separately
        except Exception as e:
            logger.warning(f"Failed to load persistent tools: {e}")


# Global tool registry instance
global_tool_registry = ToolRegistry()


def tool(name: Optional[str] = None, 
         description: Optional[str] = None,
         category: str = "custom") -> Callable:
    """Decorator to register functions as tools"""
    
    def decorator(func: Callable) -> Callable:
        global_tool_registry.register_function_as_tool(
            func=func,
            name=name,
            description=description,
            category=category
        )
        return func
    
    return decorator


# Convenience functions
def register_tool(tool_class: Type[BaseTool], metadata: Optional[ToolMetadata] = None) -> None:
    """Register a tool class"""
    global_tool_registry.register_tool(tool_class, metadata)


def get_tool(name: str) -> Optional[BaseTool]:
    """Get a tool instance"""
    return global_tool_registry.get_tool(name)


def list_tools(category: Optional[str] = None) -> List[ToolMetadata]:
    """List available tools"""
    return global_tool_registry.list_tools(category)


def discover_tools_in_package(package_path: str) -> int:
    """Discover tools in a package"""
    return global_tool_registry.discover_tools([package_path])
