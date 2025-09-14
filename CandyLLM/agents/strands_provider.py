"""
Strands Agent Provider for CandyLLM

Integration file that registers the Strands provider with CandyLLM's agent system.
This makes Strands available as the 28th agentic provider in the ecosystem.
"""

from typing import Dict, Any, Optional
import logging

from .base import BaseAgentProvider, AgentCapability
from .strands import StrandsProvider, create_strands_provider


logger = logging.getLogger(__name__)


class StrandsAgentProvider(BaseAgentProvider):
    """
    Main Strands Agent Provider for CandyLLM integration.
    
    This class serves as the primary interface between CandyLLM and the Strands SDK,
    providing seamless integration with all Strands features including:
    
    - Agent loop with recursive tool execution
    - Complete tools ecosystem (Python, MCP, community)
    - Session management and persistence
    - Multi-agent systems (A2A, Swarm, Graph, Workflow)
    - All 10+ model providers with advanced features
    - Streaming, hooks, multi-modal prompting, and more
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._strands_provider: Optional[StrandsProvider] = None
        
    @property
    def provider_name(self) -> str:
        return "strands"
    
    @property
    def supported_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability.TOOL_SYNTHESIS,
            AgentCapability.MULTI_AGENT,
            AgentCapability.MEMORY_PERSISTENCE,
            AgentCapability.HUMAN_IN_LOOP,
            AgentCapability.CODE_EXECUTION,
            AgentCapability.WEB_BROWSING,
            AgentCapability.FILE_OPERATIONS,
            AgentCapability.REASONING_CHAINS,
            AgentCapability.ROLE_PLAYING,
            AgentCapability.WORKFLOW_ORCHESTRATION
        ]
    
    async def initialize(self) -> bool:
        """Initialize the Strands provider with full feature support."""
        try:
            self._strands_provider = create_strands_provider(self.config)
            success = await self._strands_provider.initialize()
            
            if success:
                self._initialized = True
                self.logger.info("Strands agent provider initialized successfully")
                
                # Log available features
                health = await self._strands_provider.health_check()
                features = health.get('features', {})
                self.logger.info(f"Strands features: {features}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Strands provider: {e}")
            return False
    
    async def create_agent(self, config) -> str:
        """Create a new Strands agent."""
        if not self._initialized or not self._strands_provider:
            await self.initialize()
        
        return await self._strands_provider.create_agent(config)
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None):
        """Execute a Strands agent."""
        if not self._strands_provider:
            raise RuntimeError("Strands provider not initialized")
        
        return await self._strands_provider.execute_agent(agent_id, prompt, context)
    
    async def register_tool(self, agent_id: str, tool_spec) -> bool:
        """Register a tool with a Strands agent."""
        if not self._strands_provider:
            return False
        
        return await self._strands_provider.register_tool(agent_id, tool_spec)
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: list[str] = None):
        """Synthesize a tool using Strands capabilities."""
        if not self._strands_provider:
            raise RuntimeError("Strands provider not initialized")
        
        return await self._strands_provider.synthesize_tool(agent_id, tool_description, examples)
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Destroy a Strands agent."""
        if not self._strands_provider:
            return False
        
        return await self._strands_provider.destroy_agent(agent_id)
    
    # Extended Strands-specific methods
    
    async def create_swarm(self, agents: list[str], config: Dict[str, Any] = None) -> str:
        """Create a Strands Swarm for collaborative multi-agent tasks."""
        if not self._strands_provider:
            raise RuntimeError("Strands provider not initialized")
        
        return await self._strands_provider.create_swarm(agents, config)
    
    async def execute_swarm(self, swarm_id: str, task: str) -> Dict[str, Any]:
        """Execute a Strands Swarm."""
        if not self._strands_provider:
            raise RuntimeError("Strands provider not initialized")
        
        return await self._strands_provider.execute_swarm(swarm_id, task)
    
    async def create_graph(self, agents: list[str], config: Dict[str, Any] = None) -> str:
        """Create a Strands Graph for complex workflows."""
        if not self._strands_provider:
            raise RuntimeError("Strands provider not initialized")
        
        return await self._strands_provider.create_graph(agents, config)
    
    async def execute_graph(self, graph_id: str, task: str, start_node: str = None) -> Dict[str, Any]:
        """Execute a Strands Graph."""
        if not self._strands_provider:
            raise RuntimeError("Strands provider not initialized")
        
        return await self._strands_provider.execute_graph(graph_id, task, start_node)
    
    async def create_workflow(self, steps: list[Dict[str, Any]], config: Dict[str, Any] = None) -> str:
        """Create a Strands Workflow."""
        if not self._strands_provider:
            raise RuntimeError("Strands provider not initialized")
        
        return await self._strands_provider.create_workflow(steps, config)
    
    async def execute_workflow(self, workflow_id: str, initial_input: Any = None) -> Dict[str, Any]:
        """Execute a Strands Workflow."""
        if not self._strands_provider:
            raise RuntimeError("Strands provider not initialized")
        
        return await self._strands_provider.execute_workflow(workflow_id, initial_input)
    
    async def create_a2a_server(self, agent_id: str, config: Dict[str, Any] = None) -> str:
        """Create an A2A server for agent-to-agent communication."""
        if not self._strands_provider:
            raise RuntimeError("Strands provider not initialized")
        
        return await self._strands_provider.create_a2a_server(agent_id, config)
    
    async def list_agents(self) -> list[str]:
        """List all active Strands agents."""
        if not self._strands_provider:
            return []
        
        return await self._strands_provider.list_agents()
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed information about a Strands agent."""
        if not self._strands_provider:
            return {}
        
        return await self._strands_provider.get_agent_info(agent_id)
    
    async def update_agent_config(self, agent_id: str, config) -> bool:
        """Update Strands agent configuration."""
        if not self._strands_provider:
            return False
        
        return await self._strands_provider.update_agent_config(agent_id, config)
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for Strands integration."""
        base_health = await super().health_check()
        
        if self._strands_provider:
            strands_health = await self._strands_provider.health_check()
            base_health.update({
                'strands_details': strands_health,
                'integration_status': 'healthy'
            })
        else:
            base_health.update({
                'integration_status': 'not_initialized',
                'strands_details': None
            })
        
        return base_health


def create_strands_agent_provider(config: Dict[str, Any] = None) -> StrandsAgentProvider:
    """
    Factory function to create a Strands agent provider with optimal defaults.
    
    Args:
        config: Configuration for the Strands provider
        
    Returns:
        Configured StrandsAgentProvider instance
    """
    default_config = {
        # Model provider configuration
        'model_provider': 'bedrock',
        'model': {
            'provider': 'bedrock',
            'model_id': 'anthropic.claude-sonnet-4-20250514-v1:0',
            'temperature': 0.7,
            'streaming': True,
            'region': 'us-west-2'
        },
        
        # Session and storage configuration
        'session_storage': 'file',
        'session_dir': None,  # Will use temp directory
        
        # Tools configuration
        'auto_load_tools': True,
        'mcp_servers': [],  # Can be configured with MCP servers
        
        # Feature toggles
        'enable_streaming': True,
        'enable_multi_agent': True,
        
        # Security and enterprise features
        'security_level': 'sandboxed',
        'enable_observability': True,
        'enable_telemetry': False
    }
    
    if config:
        # Deep merge configuration
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        default_config = deep_merge(default_config, config)
    
    return StrandsAgentProvider(default_config)


# Export for CandyLLM integration
__all__ = [
    'StrandsAgentProvider',
    'create_strands_agent_provider'
]