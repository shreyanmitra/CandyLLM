"""
Base interfaces for the CandyLLM Agentic Provider System

Defines abstract base classes and core interfaces that all agentic providers must implement,
following the same pattern as CandyLLM's LLM provider architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from datetime import datetime


class AgentCapability(Enum):
    """Capabilities that an agentic provider can support"""
    TOOL_SYNTHESIS = "tool_synthesis"
    MULTI_AGENT = "multi_agent"
    MEMORY_PERSISTENCE = "memory_persistence"
    HUMAN_IN_LOOP = "human_in_loop"
    CODE_EXECUTION = "code_execution"
    WEB_BROWSING = "web_browsing"
    FILE_OPERATIONS = "file_operations"
    REASONING_CHAINS = "reasoning_chains"
    ROLE_PLAYING = "role_playing"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"


class AgentSecurityLevel(Enum):
    """Security levels for agent operations"""
    UNRESTRICTED = "unrestricted"
    SANDBOXED = "sandboxed" 
    MONITORED = "monitored"
    ENTERPRISE = "enterprise"


@dataclass
class AgentConfig:
    """Configuration for agent creation and behavior"""
    name: str
    description: str = ""
    system_prompt: str = ""
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    tools: List[str] = field(default_factory=list)
    capabilities: List[AgentCapability] = field(default_factory=list)
    security_level: AgentSecurityLevel = AgentSecurityLevel.SANDBOXED
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Framework-specific configurations
    framework_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from an agent execution"""
    content: str
    agent_id: str
    provider: str
    timestamp: datetime = field(default_factory=datetime.now)
    tools_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    security_violations: List[str] = field(default_factory=list)


@dataclass
class ToolSpec:
    """Universal tool specification for cross-framework compatibility"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Optional[Callable] = None
    security_policy: Dict[str, Any] = field(default_factory=dict)
    framework_adapters: Dict[str, Any] = field(default_factory=dict)


class BaseAgentProvider(ABC):
    """
    Abstract base class for all agentic providers.
    
    Follows the same pattern as CandyLLM's BaseProvider for LLMs,
    ensuring consistent interface across all frameworks.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._initialized = False
        self._capabilities = self._get_capabilities()
        
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the agentic provider"""
        pass
    
    @property
    @abstractmethod
    def supported_capabilities(self) -> List[AgentCapability]:
        """List of capabilities this provider supports"""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider and its dependencies"""
        pass
    
    @abstractmethod
    async def create_agent(self, config: AgentConfig) -> str:
        """
        Create a new agent instance.
        
        Args:
            config: Agent configuration
            
        Returns:
            Agent ID for future interactions
        """
        pass
    
    @abstractmethod
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Execute an agent with the given prompt.
        
        Args:
            agent_id: ID of the agent to execute
            prompt: Input prompt for the agent
            context: Additional context for execution
            
        Returns:
            Agent response with results and metadata
        """
        pass
    
    @abstractmethod
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """
        Register a tool with an agent.
        
        Args:
            agent_id: ID of the agent
            tool_spec: Universal tool specification
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """
        Dynamically synthesize a tool based on description.
        
        Args:
            agent_id: ID of the agent
            tool_description: Natural language description of desired tool
            examples: Optional examples of tool usage
            
        Returns:
            Synthesized tool specification
        """
        pass
    
    @abstractmethod
    async def destroy_agent(self, agent_id: str) -> bool:
        """
        Clean up and destroy an agent instance.
        
        Args:
            agent_id: ID of the agent to destroy
            
        Returns:
            Success status
        """
        pass
    
    # Optional methods with default implementations
    
    async def list_agents(self) -> List[str]:
        """List all active agent IDs for this provider"""
        return []
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a specific agent"""
        return {}
    
    async def update_agent_config(self, agent_id: str, config: AgentConfig) -> bool:
        """Update agent configuration"""
        return False
    
    async def pause_agent(self, agent_id: str) -> bool:
        """Pause agent execution"""
        return False
    
    async def resume_agent(self, agent_id: str) -> bool:
        """Resume agent execution"""
        return False
    
    def supports_capability(self, capability: AgentCapability) -> bool:
        """Check if provider supports a specific capability"""
        return capability in self.supported_capabilities
    
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get capabilities from provider implementation"""
        return self.supported_capabilities
    
    async def health_check(self) -> Dict[str, Any]:
        """Provider health check"""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "provider": self.provider_name,
            "capabilities": [cap.value for cap in self.supported_capabilities],
            "timestamp": datetime.now().isoformat()
        }


class AgentManager(ABC):
    """
    Abstract agent manager for coordinating multiple agents and providers.
    
    Provides intelligent routing, load balancing, and multi-agent orchestration.
    """
    
    @abstractmethod
    async def route_request(self, prompt: str, requirements: List[AgentCapability] = None) -> Tuple[str, str]:
        """
        Route a request to the most appropriate provider and agent.
        
        Args:
            prompt: Input prompt
            requirements: Required capabilities
            
        Returns:
            Tuple of (provider_name, agent_id)
        """
        pass
    
    @abstractmethod
    async def orchestrate_multi_agent(self, 
                                    agents: List[Tuple[str, str]], 
                                    workflow: Dict[str, Any],
                                    context: Dict[str, Any] = None) -> List[AgentResponse]:
        """
        Orchestrate a multi-agent workflow.
        
        Args:
            agents: List of (provider_name, agent_id) tuples
            workflow: Workflow definition
            context: Shared context
            
        Returns:
            List of responses from all agents
        """
        pass


class ProviderRegistry:
    """Registry for managing agentic providers"""
    
    def __init__(self):
        self._providers: Dict[str, BaseAgentProvider] = {}
        self._capabilities_map: Dict[AgentCapability, List[str]] = {}
        
    def register_provider(self, provider: BaseAgentProvider):
        """Register a new provider"""
        self._providers[provider.provider_name] = provider
        
        # Update capabilities map
        for capability in provider.supported_capabilities:
            if capability not in self._capabilities_map:
                self._capabilities_map[capability] = []
            if provider.provider_name not in self._capabilities_map[capability]:
                self._capabilities_map[capability].append(provider.provider_name)
    
    def get_provider(self, name: str) -> Optional[BaseAgentProvider]:
        """Get a provider by name"""
        return self._providers.get(name)
    
    def get_providers_by_capability(self, capability: AgentCapability) -> List[BaseAgentProvider]:
        """Get all providers that support a capability"""
        provider_names = self._capabilities_map.get(capability, [])
        return [self._providers[name] for name in provider_names if name in self._providers]
    
    def list_providers(self) -> List[str]:
        """List all registered provider names"""
        return list(self._providers.keys())
    
    async def initialize_all(self) -> Dict[str, bool]:
        """Initialize all registered providers"""
        results = {}
        for name, provider in self._providers.items():
            try:
                results[name] = await provider.initialize()
            except Exception as e:
                logging.error(f"Failed to initialize provider {name}: {e}")
                results[name] = False
        return results


# Global provider registry instance
global_registry = ProviderRegistry()