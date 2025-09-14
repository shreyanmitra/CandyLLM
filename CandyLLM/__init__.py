"""
CandyLLM - Enterprise AI Framework with Multi-Provider Support

A comprehensive AI framework supporting 100+ models across multiple providers
with enterprise-grade security, dynamic tool synthesis, and agentic capabilities.

Features:
- Multi-LLM provider support (OpenAI, Anthropic, LiteLLM, Universal)
- Unified agentic framework supporting LangChain, CrewAI, OpenAI Assistants, and more
- Enterprise security with audit trails and rate limiting
- Dynamic tool synthesis and cross-framework compatibility
- Intelligent agent routing and multi-agent orchestration

(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.
"""

# Import core providers
from .providers import (
    UniversalModelProvider, ModelFactory,
    OpenAIProvider, AnthropicProvider, LiteLLMProvider, UselessProvider,
    PROVIDER_SECURITY_LEVELS, SECURITY_FEATURES
)

# Import agentic provider system
from .agents import (
    BaseAgentProvider, AgentManager, AgentConfig, AgentResponse,
    AgentCapability, AgentSecurityLevel, ToolSpec,
    CandyLLMAgentProvider, AgentProviderManager, AgentSecurityManager,
    UniversalToolSpec, FrameworkToolAdapter, ToolFormat,
    register_provider, get_provider, list_providers,
    global_registry
)

# Version information
__version__ = "1.0.0"
__author__ = "Shreyan Mitra"
__email__ = "shreyan@candyllm.com"

# Export main components
__all__ = [
    # LLM Providers
    "UniversalModelProvider", "ModelFactory",
    "OpenAI", "AnthropicProvider", "LiteLLMProvider", "UselessProvider",
    
    # Agentic System - Core Interfaces
    "BaseAgentProvider", "AgentManager", "AgentConfig", "AgentResponse",
    "AgentCapability", "AgentSecurityLevel", "ToolSpec",
    
    # Agentic System - Implementations  
    "CandyLLMAgentProvider", "AgentProviderManager", "AgentSecurityManager",
    
    # Tool System
    "UniversalToolSpec", "FrameworkToolAdapter", "ToolFormat",
    
    # Registry Functions
    "register_provider", "get_provider", "list_providers", "global_registry",
    
    # Security
    "PROVIDER_SECURITY_LEVELS", "SECURITY_FEATURES",
    
    # Metadata
    "__version__", "__author__", "__email__"
]

# Initialize default providers
def initialize_candyllm():
    """Initialize CandyLLM with default agentic providers"""
    from .agents import global_registry
    from .agents.candyllm_provider import CandyLLMAgentProvider
    
    # Register core CandyLLM provider
    candyllm_provider = CandyLLMAgentProvider()
    global_registry.register_provider(candyllm_provider)
    
    # Try to register external providers
    try:
        from .agents.langchain_provider import LangChainAgentProvider
        langchain_provider = LangChainAgentProvider()
        global_registry.register_provider(langchain_provider)
    except ImportError:
        pass
    
    try:
        from .agents.crewai_provider import CrewAIAgentProvider
        crewai_provider = CrewAIAgentProvider()
        global_registry.register_provider(crewai_provider)
    except ImportError:
        pass
    
    try:
        from .agents.openai_assistants_provider import OpenAIAssistantsProvider
        openai_provider = OpenAIAssistantsProvider()
        global_registry.register_provider(openai_provider)
    except ImportError:
        pass

# Auto-initialize on import
initialize_candyllm()