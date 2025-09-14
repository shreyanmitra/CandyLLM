"""
CandyLLM Agentic Provider System

A unified framework for integrating multiple agentic AI frameworks through 
a consistent provider interface, extending CandyLLM's proven LLM provider pattern.

This module treats CandyLLM's existing agentic capabilities as just another provider
alongside external frameworks like LangChain, CrewAI, AutoGen, and others.
"""

from .base import BaseAgentProvider, AgentManager, AgentConfig
from .candyllm_provider import CandyLLMAgentProvider
from .manager import AgentProviderManager
from .security import AgentSecurityManager
from .tools import UniversalToolSpec, FrameworkToolAdapter

# Core interfaces
__all__ = [
    'BaseAgentProvider',
    'AgentManager', 
    'AgentConfig',
    'AgentProviderManager',
    'AgentSecurityManager',
    'UniversalToolSpec',
    'FrameworkToolAdapter',
    'CandyLLMAgentProvider'
]

# Provider registry - dynamically populated based on available frameworks
AVAILABLE_PROVIDERS = {}

def register_provider(name: str, provider_class: type):
    """Register an agentic provider"""
    AVAILABLE_PROVIDERS[name] = provider_class

def get_provider(name: str) -> type:
    """Get a registered provider class"""
    return AVAILABLE_PROVIDERS.get(name)

def list_providers() -> list:
    """List all available agentic providers"""
    return list(AVAILABLE_PROVIDERS.keys())

# Auto-register core provider
register_provider('candyllm', CandyLLMAgentProvider)

# Auto-discover and register external providers
try:
    from .langchain_provider import LangChainAgentProvider
    register_provider('langchain', LangChainAgentProvider)
except ImportError:
    pass

try:
    from .crewai_provider import CrewAIAgentProvider
    register_provider('crewai', CrewAIAgentProvider)
except ImportError:
    pass

try:
    from .openai_assistants_provider import OpenAIAssistantsProvider
    register_provider('openai_assistants', OpenAIAssistantsProvider)
except ImportError:
    pass

try:
    from .autogen_provider import AutoGenAgentProvider
    register_provider('autogen', AutoGenAgentProvider)
except ImportError:
    pass

try:
    from .swarm_provider import SwarmAgentProvider
    register_provider('swarm', SwarmAgentProvider)
except ImportError:
    pass

try:
    from .strands_provider import StrandsAgentProvider
    register_provider('strands', StrandsAgentProvider)
except ImportError:
    pass