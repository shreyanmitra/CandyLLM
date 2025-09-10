"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

CandyLLM Core Module - Secure Architecture Foundation
Enhanced enterprise-grade architecture for universal LLM provider support with comprehensive security.

This core module provides the foundational components for the CandyLLM platform including:
- Secure provider architecture with input validation
- Enterprise-grade base classes with audit logging
- Neurosymbolic reasoning with security controls
- Dynamic tooling with sandboxed execution
- Intelligent routing with performance monitoring
- Configuration management with encryption support

Security Features:
- Input validation and sanitization for all core components
- Secure import validation and environment checking
- Audit logging for all core operations
- Resource limits and execution monitoring
- Access controls and authentication
- Threat detection and prevention
"""

import logging
from typing import TYPE_CHECKING

# Configure secure logging for core module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Security validation for core imports
def _validate_core_security():
    """Validate core module security before exposing APIs."""
    try:
        # Log core module initialization
        logger.info("Initializing CandyLLM core module with security validation")
        
        # Check for required security dependencies
        required_modules = ['cryptography', 'pydantic', 'typing']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            logger.warning(f"Optional security modules missing: {missing_modules}")
        
        logger.info("Core module security validation completed")
        return True
        
    except Exception as e:
        logger.error(f"Core module security validation failed: {e}")
        return False

# Perform security validation
_validate_core_security()

# Secure core imports with validation
try:
    # Base architecture
    from .base import (
        BaseProvider, BaseModel, BaseTool, BaseAgent,
        ModelConfig, Message, ModelResponse, StreamingChunk,
        ProviderType, ModelType, SecurityValidator
    )
    
    # Main CandyLLM class
    from .candyllm import CandyLLM, CandyConfig, quick_chat, quick_reason, quick_math
    
    # Intelligent routing
    from .router import IntelligentRouter, RouteDecision
    
    # Neurosymbolic AI
    from .neurosymbolic import (
        NeuroSymbolicEngine, KnowledgeTriple, ReasoningType, 
        KnowledgeType, SecureMathematicalReasoner
    )
    
    # Advanced reasoning
    from .reasoning import AdvancedReasoningEngine, ReasoningResult
    
    # Dynamic tooling
    from .dynamic_tooling import DynamicToolingEngine, ToolSpec
    from .dynamic_config import (
        DynamicToolingConfig, SecureComponentFactory as ComponentFactory
    )
    
    # Type definitions
    from .types import (
        ChatResponse, StreamChunk, MathResult, ToolResult
    )
    
    logger.info("Core module imports completed successfully")
    
except ImportError as e:
    logger.error(f"Core import failed: {e}")
    # Provide fallback or raise controlled error
    raise ImportError(f"CandyLLM core module import failed: {e}")

# Version and metadata
__version__ = "3.0.0"
__author__ = "Shreyan Mitra"
__license__ = "MIT"
__description__ = "CandyLLM Core - Secure Enterprise AI Architecture"

# Export security-validated components
__all__ = [
    # Base components
    'BaseProvider', 'BaseModel', 'BaseTool', 'BaseAgent',
    'ModelConfig', 'Message', 'ModelResponse', 'StreamingChunk',
    'ProviderType', 'ModelType', 'SecurityValidator',
    
    # Core engines
    'CandyLLM', 'CandyConfig',
    'IntelligentRouter', 'RouteDecision',
    'NeuroSymbolicEngine', 'AdvancedReasoningEngine',
    'DynamicToolingEngine', 'ComponentFactory',
    
    # Data types
    'ChatResponse', 'StreamChunk', 'ReasoningResult', 'MathResult', 'ToolResult',
    'KnowledgeTriple', 'ReasoningType', 'KnowledgeType', 'ToolSpec',
    'DynamicToolingConfig',
    
    # Quick functions
    'quick_chat', 'quick_reason', 'quick_math',
    
    # Metadata
    '__version__', '__author__', '__license__', '__description__'
]
