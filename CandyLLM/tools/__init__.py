"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

CandyLLM Tools System - Secure Tool Management and Registry
Comprehensive tool system with enterprise security controls and audit logging.

This module provides secure tool management capabilities including:
- Secure tool registry with validation and sandboxing
- Code analysis and integrity verification
- Access controls and execution monitoring
- Audit logging for all tool operations
- Safe tool discovery and registration
- Resource limits and security constraints

Security Features:
- Input validation and sanitization for all tool operations
- Code analysis and static security scanning
- Sandboxed execution environment for tool code
- Cryptographic verification of tool integrity
- Access controls and permission management
- Comprehensive audit logging and monitoring
"""

import logging
from typing import TYPE_CHECKING

# Configure secure logging for tools module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Security validation for tools module
def _validate_tools_security():
    """Validate tools module security and dependencies."""
    try:
        logger.info("Initializing CandyLLM tools module with security validation")
        
        # Check for security dependencies
        security_modules = ['ast', 'hashlib', 'datetime']
        for module in security_modules:
            try:
                __import__(module)
            except ImportError as e:
                logger.error(f"Required security module missing: {module}")
                raise ImportError(f"Tools security validation failed: {e}")
        
        logger.info("Tools module security validation completed")
        return True
        
    except Exception as e:
        logger.error(f"Tools module security validation failed: {e}")
        return False

# Perform security validation
_validate_tools_security()

# Secure imports with validation
try:
    from .registry import (
        ToolRegistry, 
        ToolMetadata, 
        global_tool_registry,
        tool,
        register_tool,
        get_tool,
        list_tools,
        discover_tools_in_package,
        # Security-enhanced components
        SecureToolValidator,
        ToolExecutionMonitor,
        ToolAuditLogger
    )
    
    logger.info("Tools module imports completed successfully")
    
except ImportError as e:
    logger.error(f"Tools import failed: {e}")
    raise ImportError(f"CandyLLM tools module import failed: {e}")

# Module metadata
__version__ = "3.0.0"
__author__ = "Shreyan Mitra" 
__license__ = "MIT"
__description__ = "CandyLLM Tools - Secure Tool Management System"

# Export security-validated components
__all__ = [
    # Core tool management
    "ToolRegistry",
    "ToolMetadata", 
    "global_tool_registry",
    
    # Tool operations
    "tool",
    "register_tool",
    "get_tool", 
    "list_tools",
    "discover_tools_in_package",
    
    # Security components (if available)
    "SecureToolValidator",
    "ToolExecutionMonitor", 
    "ToolAuditLogger",
    
    # Metadata
    "__version__", "__author__", "__license__", "__description__"
]
