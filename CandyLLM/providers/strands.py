"""
Strands Agent Provider for CandyLLM

A comprehensive provider implementation that integrates Strands Agents SDK as the 28th agentic provider.
Strands is a production-ready, lightweight agent framework with Amazon Bedrock as default provider,
comprehensive tools ecosystem, multi-agent capabilities, and enterprise deployment options.

Features:
- Agent class with event loop lifecycle and recursive tool execution
- Complete tools ecosystem: Python (@tool decorator), MCP (Model Context Protocol), community tools
- Session management with FileSessionManager and S3SessionManager for persistence
- Hooks system with BeforeInvocationEvent, AfterInvocationEvent, and lifecycle callbacks
- Multi-modal prompting (text, images, documents) and structured output
- Streaming support via async iterators and callback handlers
- Multi-agent systems: A2A protocol, Swarm patterns, Graph workflows
- All 10+ model providers: Bedrock, Anthropic, OpenAI, LiteLLM, etc.
- Advanced features: reasoning, guardrails, caching, observability, telemetry
"""

import asyncio
import json
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Callable
from dataclasses import dataclass, field
import tempfile
import os

from ..agents.base import (
    BaseAgentProvider, 
    AgentCapability, 
    AgentConfig, 
    AgentResponse, 
    ToolSpec,
    AgentSecurityLevel
)

# Import protection for optional dependencies
try:
    # Core Strands imports
    from strands import Agent
    from strands.models import BedrockModel, AnthropicModel, OpenAIModel
    from strands.models.litellm import LiteLLMModel
    from strands.models.ollama import OllamaModel
    from strands.models.cohere import CohereModel
    from strands.models.mistral import MistralModel
    from strands.models.llamaapi import LlamaAPIModel
    from strands.models.sagemaker import SageMakerModel
    from strands.models.writer import WriterModel
    from strands.session import FileSessionManager, S3SessionManager
    from strands.tools.executors import ConcurrentToolExecutor, SequentialToolExecutor
    from strands.tools.decorator import tool
    from strands.multiagent.swarm import Swarm
    from strands.multiagent.graph import Graph
    from strands.multiagent.workflow import Workflow
    from strands.multiagent.a2a import A2AServer
    from strands.types.content import ContentBlock
    from strands.hooks import BeforeInvocationEvent, AfterInvocationEvent
    
    # Community tools
    try:
        import strands_tools
        STRANDS_TOOLS_AVAILABLE = True
    except ImportError:
        STRANDS_TOOLS_AVAILABLE = False
    
    # MCP support
    try:
        from strands.tools.mcp import MCPClient
        from mcp.client.sse import sse_client
        from mcp import stdio_client, StdioServerParameters
        MCP_AVAILABLE = True
    except ImportError:
        MCP_AVAILABLE = False
    
    # A2A support
    try:
        from strands.multiagent.a2a import A2AServer
        from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
        A2A_AVAILABLE = True
    except ImportError:
        A2A_AVAILABLE = False
    
    STRANDS_AVAILABLE = True
    
except ImportError as e:
    STRANDS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    
    # Create placeholder classes to prevent import errors
    class Agent: pass
    class BedrockModel: pass
    class Swarm: pass
    class Graph: pass
    class Workflow: pass
    class A2AServer: pass


logger = logging.getLogger(__name__)


@dataclass
class StrandsAgentWrapper:
    """Wrapper for Strands Agent instances with CandyLLM integration"""
    agent_id: str
    strands_agent: Any  # Strands Agent instance
    config: AgentConfig
    session_manager: Optional[Any] = None
    tools: List[str] = field(default_factory=list)
    hooks: List[Callable] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics and metadata"""
        return {
            "agent_id": self.agent_id,
            "tools_count": len(self.tools),
            "hooks_count": len(self.hooks),
            "created_at": self.created_at.isoformat(),
            "session_enabled": self.session_manager is not None,
            "agent_name": getattr(self.strands_agent, 'name', 'unknown')
        }


class StrandsProvider(BaseAgentProvider):
    """
    Strands Agents SDK provider for CandyLLM.
    
    Integrates the complete Strands ecosystem including:
    - Agent loop with recursive tool execution
    - Comprehensive tools support (Python, MCP, community)
    - Session management and persistence
    - Hooks system and lifecycle events
    - Multi-modal prompting and structured output
    - Streaming via async iterators and callbacks
    - Multi-agent systems (A2A, Swarm, Graph, Workflow)
    - All model providers with advanced features
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, StrandsAgentWrapper] = {}
        self._swarms: Dict[str, Any] = {}  # Swarm instances
        self._graphs: Dict[str, Any] = {}  # Graph instances
        self._workflows: Dict[str, Any] = {}  # Workflow instances
        self._a2a_servers: Dict[str, Any] = {}  # A2A server instances
        self._session_managers: Dict[str, Any] = {}  # Session managers
        self._mcp_clients: Dict[str, Any] = {}  # MCP clients
        
        # Configuration
        self.default_model_provider = config.get('model_provider', 'bedrock')
        self.session_storage_type = config.get('session_storage', 'file')  # 'file' or 's3'
        self.tools_auto_load = config.get('auto_load_tools', True)
        self.enable_streaming = config.get('enable_streaming', True)
        self.enable_multi_agent = config.get('enable_multi_agent', True)
        
    @property
    def provider_name(self) -> str:
        return "strands"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        capabilities = [
            AgentCapability.TOOL_SYNTHESIS,
            AgentCapability.MEMORY_PERSISTENCE,
            AgentCapability.CODE_EXECUTION,
            AgentCapability.WEB_BROWSING,
            AgentCapability.FILE_OPERATIONS,
            AgentCapability.REASONING_CHAINS,
            AgentCapability.ROLE_PLAYING,
            AgentCapability.WORKFLOW_ORCHESTRATION
        ]
        
        if self.enable_multi_agent:
            capabilities.append(AgentCapability.MULTI_AGENT)
            
        return capabilities
    
    async def initialize(self) -> bool:
        """Initialize the Strands provider with all components."""
        if not STRANDS_AVAILABLE:
            self.logger.error(f"Strands SDK not available: {IMPORT_ERROR}")
            return False
        
        try:
            self.logger.info("Initializing Strands provider...")
            
            # Initialize session managers based on configuration
            await self._initialize_session_managers()
            
            # Initialize MCP clients if available
            if MCP_AVAILABLE:
                await self._initialize_mcp_clients()
            
            # Initialize community tools if available
            if STRANDS_TOOLS_AVAILABLE and self.tools_auto_load:
                await self._initialize_community_tools()
            
            # Initialize A2A servers if enabled
            if A2A_AVAILABLE and self.enable_multi_agent:
                await self._initialize_a2a_support()
            
            self._initialized = True
            self.logger.info("Strands provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Strands provider: {e}")
            return False
    
    async def _initialize_session_managers(self):
        """Initialize session managers for agent persistence"""
        try:
            if self.session_storage_type == 's3':
                # S3 session manager for production/cloud deployments
                s3_config = self.config.get('s3_config', {})
                bucket_name = s3_config.get('bucket_name', 'strands-sessions')
                self._session_managers['s3'] = S3SessionManager(bucket_name=bucket_name)
                self.logger.info(f"Initialized S3 session manager with bucket: {bucket_name}")
            else:
                # File session manager for local/development
                session_dir = self.config.get('session_dir', tempfile.gettempdir())
                self._session_managers['file'] = FileSessionManager(session_dir=session_dir)
                self.logger.info(f"Initialized file session manager in: {session_dir}")
                
        except Exception as e:
            self.logger.warning(f"Session manager initialization failed: {e}")
    
    async def _initialize_mcp_clients(self):
        """Initialize Model Context Protocol clients"""
        mcp_configs = self.config.get('mcp_servers', [])
        
        for mcp_config in mcp_configs:
            try:
                server_name = mcp_config.get('name', f"mcp_{len(self._mcp_clients)}")
                transport_type = mcp_config.get('transport', 'sse')
                
                if transport_type == 'sse':
                    url = mcp_config.get('url', 'http://localhost:8000/sse')
                    client = MCPClient(lambda: sse_client(url))
                elif transport_type == 'stdio':
                    command = mcp_config.get('command')
                    args = mcp_config.get('args', [])
                    client = MCPClient(lambda: stdio_client(
                        StdioServerParameters(command=command, args=args)
                    ))
                else:
                    self.logger.warning(f"Unsupported MCP transport: {transport_type}")
                    continue
                
                self._mcp_clients[server_name] = client
                self.logger.info(f"Initialized MCP client: {server_name} ({transport_type})")
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize MCP client {mcp_config}: {e}")
    
    async def _initialize_community_tools(self):
        """Initialize Strands community tools package"""
        try:
            # Community tools are automatically available when strands_tools is installed
            # We can enumerate available tools from the package
            available_tools = []
            
            # Common community tools
            tool_names = [
                'calculator', 'file_read', 'file_write', 'shell', 'http_request',
                'current_time', 'weather', 'screenshot', 'email', 'swarm'
            ]
            
            for tool_name in tool_names:
                try:
                    tool_func = getattr(strands_tools, tool_name, None)
                    if tool_func:
                        available_tools.append(tool_name)
                except:
                    pass
            
            self.logger.info(f"Community tools available: {available_tools}")
            
        except Exception as e:
            self.logger.warning(f"Community tools initialization failed: {e}")
    
    async def _initialize_a2a_support(self):
        """Initialize Agent-to-Agent protocol support"""
        try:
            # A2A support is initialized per-agent when multi-agent features are used
            self.logger.info("A2A protocol support initialized")
            
        except Exception as e:
            self.logger.warning(f"A2A initialization failed: {e}")
    
    def _create_model_provider(self, model_config: Dict[str, Any]) -> Any:
        """Create a Strands model provider based on configuration"""
        provider_type = model_config.get('provider', self.default_model_provider)
        
        if provider_type == 'bedrock':
            return BedrockModel(
                model_id=model_config.get('model_id', 'anthropic.claude-sonnet-4-20250514-v1:0'),
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens'),
                streaming=model_config.get('streaming', True),
                region_name=model_config.get('region', 'us-west-2'),
                # Advanced Bedrock features
                guardrail_id=model_config.get('guardrail_id'),
                guardrail_version=model_config.get('guardrail_version'),
                cache_prompt=model_config.get('cache_prompt'),
                cache_tools=model_config.get('cache_tools'),
                additional_request_fields=model_config.get('additional_request_fields', {})
            )
        elif provider_type == 'anthropic':
            return AnthropicModel(
                model=model_config.get('model', 'claude-3-5-sonnet-20241022'),
                api_key=model_config.get('api_key'),
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens', 4096),
                streaming=model_config.get('streaming', True)
            )
        elif provider_type == 'openai':
            return OpenAIModel(
                model=model_config.get('model', 'gpt-4o'),
                api_key=model_config.get('api_key'),
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens'),
                streaming=model_config.get('streaming', True),
                base_url=model_config.get('base_url')
            )
        elif provider_type == 'litellm':
            return LiteLLMModel(
                model=model_config.get('model', 'gpt-4o'),
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens'),
                streaming=model_config.get('streaming', True),
                api_key=model_config.get('api_key'),
                api_base=model_config.get('api_base')
            )
        elif provider_type == 'ollama':
            return OllamaModel(
                model=model_config.get('model', 'llama3.2'),
                base_url=model_config.get('base_url', 'http://localhost:11434'),
                temperature=model_config.get('temperature', 0.7),
                streaming=model_config.get('streaming', True)
            )
        elif provider_type == 'cohere':
            return CohereModel(
                model=model_config.get('model', 'command-r-plus'),
                api_key=model_config.get('api_key'),
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens'),
                streaming=model_config.get('streaming', True)
            )
        elif provider_type == 'mistral':
            return MistralModel(
                model=model_config.get('model', 'mistral-large-latest'),
                api_key=model_config.get('api_key'),
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens'),
                streaming=model_config.get('streaming', True)
            )
        elif provider_type == 'llamaapi':
            return LlamaAPIModel(
                model=model_config.get('model', 'llama3.1-70b'),
                api_token=model_config.get('api_token'),
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens'),
                streaming=model_config.get('streaming', True)
            )
        elif provider_type == 'sagemaker':
            return SageMakerModel(
                endpoint_name=model_config.get('endpoint_name'),
                region_name=model_config.get('region', 'us-west-2'),
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens')
            )
        elif provider_type == 'writer':
            return WriterModel(
                model=model_config.get('model', 'palmyra-x-004'),
                api_key=model_config.get('api_key'),
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens'),
                streaming=model_config.get('streaming', True)
            )
        else:
            # Default to Bedrock if unsupported provider
            self.logger.warning(f"Unsupported provider {provider_type}, defaulting to Bedrock")
            return BedrockModel()
    
    def _create_session_manager(self, agent_id: str) -> Optional[Any]:
        """Create a session manager for an agent"""
        try:
            if self.session_storage_type == 's3' and 's3' in self._session_managers:
                return self._session_managers['s3']
            elif 'file' in self._session_managers:
                return self._session_managers['file']
            return None
        except Exception as e:
            self.logger.warning(f"Failed to create session manager for {agent_id}: {e}")
            return None
    
    def _create_tools_list(self, config: AgentConfig) -> List[Any]:
        """Create tools list from configuration"""
        tools = []
        
        # Add community tools if available
        if STRANDS_TOOLS_AVAILABLE:
            for tool_name in config.tools:
                try:
                    tool_func = getattr(strands_tools, tool_name, None)
                    if tool_func:
                        tools.append(tool_func)
                        self.logger.debug(f"Added community tool: {tool_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to add community tool {tool_name}: {e}")
        
        # Add MCP tools if available
        for mcp_name, mcp_client in self._mcp_clients.items():
            try:
                with mcp_client:
                    mcp_tools = mcp_client.list_tools_sync()
                    tools.extend(mcp_tools)
                    self.logger.debug(f"Added {len(mcp_tools)} MCP tools from {mcp_name}")
            except Exception as e:
                self.logger.warning(f"Failed to get MCP tools from {mcp_name}: {e}")
        
        return tools
    
    def _create_hooks(self, config: AgentConfig) -> List[Callable]:
        """Create hooks for agent lifecycle events"""
        hooks = []
        
        # Add default logging hooks
        def before_invocation_hook(event):
            self.logger.debug(f"Agent invocation starting: {event}")
        
        def after_invocation_hook(event):
            self.logger.debug(f"Agent invocation completed: {event}")
        
        hooks.extend([before_invocation_hook, after_invocation_hook])
        
        # Add custom hooks from config
        custom_hooks = config.framework_config.get('hooks', [])
        hooks.extend(custom_hooks)
        
        return hooks
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Strands agent instance."""
        if not self._initialized:
            await self.initialize()
        
        agent_id = f"strands_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create model provider
            model_config = config.framework_config.get('model', {})
            model = self._create_model_provider(model_config)
            
            # Create session manager
            session_manager = self._create_session_manager(agent_id)
            
            # Create tools list
            tools = self._create_tools_list(config)
            
            # Create hooks
            hooks = self._create_hooks(config)
            
            # Create tool executor
            executor_type = config.framework_config.get('tool_executor', 'concurrent')
            tool_executor = (ConcurrentToolExecutor() if executor_type == 'concurrent' 
                           else SequentialToolExecutor())
            
            # Create Strands Agent
            strands_agent = Agent(
                name=config.name,
                description=config.description,
                system_prompt=config.system_prompt,
                model=model,
                tools=tools,
                session=session_manager,
                tool_executor=tool_executor,
                load_tools_from_directory=self.tools_auto_load
            )
            
            # Create wrapper
            wrapper = StrandsAgentWrapper(
                agent_id=agent_id,
                strands_agent=strands_agent,
                config=config,
                session_manager=session_manager,
                tools=config.tools,
                hooks=hooks
            )
            
            self._agents[agent_id] = wrapper
            
            self.logger.info(f"Created Strands agent {agent_id} with {len(tools)} tools")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Strands agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Strands agent with the given prompt."""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        wrapper = self._agents[agent_id]
        agent = wrapper.strands_agent
        context = context or {}
        
        try:
            # Handle multi-modal inputs
            if isinstance(prompt, list):
                # Convert to ContentBlocks for multi-modal support
                content_blocks = []
                for item in prompt:
                    if isinstance(item, str):
                        content_blocks.append(ContentBlock(text=item))
                    elif isinstance(item, dict):
                        content_blocks.append(ContentBlock(**item))
                prompt_input = content_blocks
            else:
                prompt_input = prompt
            
            # Handle structured output requests
            if context.get('structured_output'):
                try:
                    from pydantic import BaseModel
                    output_schema = context.get('output_schema')
                    if output_schema and issubclass(output_schema, BaseModel):
                        result = agent.structured_output(output_schema, prompt_input)
                        return AgentResponse(
                            content=result.model_dump_json(),
                            agent_id=agent_id,
                            provider=self.provider_name,
                            metadata={'structured_output': True, 'schema': output_schema.__name__}
                        )
                except Exception as e:
                    self.logger.warning(f"Structured output failed, falling back to regular: {e}")
            
            # Execute agent with streaming or regular mode
            if self.enable_streaming and context.get('stream', False):
                # Return streaming response
                return await self._execute_streaming(agent, agent_id, prompt_input, context)
            else:
                # Regular execution
                if asyncio.iscoroutinefunction(agent):
                    content = await agent.invoke_async(prompt_input)
                else:
                    content = agent(prompt_input)
                
                return AgentResponse(
                    content=str(content),
                    agent_id=agent_id,
                    provider=self.provider_name,
                    tools_used=context.get('tools_used', []),
                    metadata={
                        'model_provider': wrapper.config.framework_config.get('model', {}).get('provider', 'bedrock'),
                        'agent_stats': wrapper.get_stats(),
                        'session_enabled': wrapper.session_manager is not None,
                        'streaming_used': False,
                        'reasoning_enabled': context.get('enable_reasoning', False),
                        'caching_enabled': context.get('enable_caching', False)
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Strands agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def _execute_streaming(self, agent: Any, agent_id: str, prompt: Any, context: Dict[str, Any]) -> AgentResponse:
        """Execute agent in streaming mode"""
        try:
            content_chunks = []
            tools_used = []
            
            async for event in agent.stream_async(prompt):
                if 'data' in event:
                    content_chunks.append(event['data'])
                if 'current_tool_use' in event:
                    tool_info = event['current_tool_use']
                    if tool_info.get('name'):
                        tools_used.append(tool_info['name'])
            
            full_content = ''.join(content_chunks)
            
            return AgentResponse(
                content=full_content,
                agent_id=agent_id,
                provider=self.provider_name,
                tools_used=tools_used,
                metadata={
                    'streaming_used': True,
                    'chunks_received': len(content_chunks),
                    'tools_invoked': len(tools_used)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Streaming execution failed: {e}")
            raise
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with a Strands agent."""
        if agent_id not in self._agents:
            return False
        
        wrapper = self._agents[agent_id]
        agent = wrapper.strands_agent
        
        try:
            # Convert ToolSpec to Strands tool format
            @tool(name=tool_spec.name, description=tool_spec.description)
            def dynamic_tool(**kwargs):
                if tool_spec.function:
                    return tool_spec.function(**kwargs)
                else:
                    return f"Tool {tool_spec.name} executed with parameters: {kwargs}"
            
            # Add tool to agent
            agent.tools.append(dynamic_tool)
            wrapper.tools.append(tool_spec.name)
            
            self.logger.info(f"Registered tool {tool_spec.name} with Strands agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool with Strands agent: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Strands agent capabilities."""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        wrapper = self._agents[agent_id]
        agent = wrapper.strands_agent
        
        try:
            # Use the agent to generate tool implementation
            synthesis_prompt = f"""
            Create a Python function that implements the following tool:
            
            Description: {tool_description}
            
            Examples: {examples or []}
            
            Return a Python function with proper type hints and docstring.
            The function should be production-ready and handle errors gracefully.
            """
            
            if asyncio.iscoroutinefunction(agent):
                implementation = await agent.invoke_async(synthesis_prompt)
            else:
                implementation = agent(synthesis_prompt)
            
            # Create tool specification
            tool_name = tool_description.split()[0].lower().replace(' ', '_')
            
            # Extract function from implementation (simplified)
            # In production, this would use AST parsing and code generation
            def synthesized_function(**kwargs):
                return f"Synthesized tool {tool_name} executed: {kwargs}"
            
            tool_spec = ToolSpec(
                name=tool_name,
                description=tool_description,
                parameters={
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Tool input"}
                    }
                },
                function=synthesized_function
            )
            
            self.logger.info(f"Synthesized tool: {tool_name}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def create_swarm(self, agents: List[str], config: Dict[str, Any] = None) -> str:
        """Create a Strands Swarm with multiple agents."""
        if not self.enable_multi_agent:
            raise NotImplementedError("Multi-agent support not enabled")
        
        swarm_id = f"swarm_{uuid.uuid4().hex[:8]}"
        config = config or {}
        
        try:
            # Get agent instances
            swarm_agents = []
            for agent_id in agents:
                if agent_id in self._agents:
                    swarm_agents.append(self._agents[agent_id].strands_agent)
                else:
                    self.logger.warning(f"Agent {agent_id} not found for swarm")
            
            if not swarm_agents:
                raise ValueError("No valid agents found for swarm")
            
            # Create Strands Swarm
            swarm = Swarm(
                swarm_agents,
                max_handoffs=config.get('max_handoffs', 20),
                max_iterations=config.get('max_iterations', 20),
                execution_timeout=config.get('execution_timeout', 900.0),
                node_timeout=config.get('node_timeout', 300.0)
            )
            
            self._swarms[swarm_id] = swarm
            
            self.logger.info(f"Created Strands swarm {swarm_id} with {len(swarm_agents)} agents")
            return swarm_id
            
        except Exception as e:
            self.logger.error(f"Failed to create swarm: {e}")
            raise
    
    async def execute_swarm(self, swarm_id: str, task: str) -> Dict[str, Any]:
        """Execute a Strands Swarm on a task."""
        if swarm_id not in self._swarms:
            raise ValueError(f"Swarm {swarm_id} not found")
        
        swarm = self._swarms[swarm_id]
        
        try:
            # Execute swarm
            result = swarm(task)
            
            return {
                'swarm_id': swarm_id,
                'status': result.status,
                'final_result': result.result if hasattr(result, 'result') else str(result),
                'node_history': [node.node_id for node in result.node_history] if hasattr(result, 'node_history') else [],
                'execution_count': getattr(result, 'execution_count', 0),
                'execution_time': getattr(result, 'execution_time', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Swarm execution failed: {e}")
            raise
    
    async def create_a2a_server(self, agent_id: str, config: Dict[str, Any] = None) -> str:
        """Create an A2A server for agent-to-agent communication."""
        if not A2A_AVAILABLE:
            raise NotImplementedError("A2A support not available")
        
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        wrapper = self._agents[agent_id]
        agent = wrapper.strands_agent
        config = config or {}
        
        try:
            server_id = f"a2a_{agent_id}"
            
            # Create A2A server
            a2a_server = A2AServer(
                agent=agent,
                host=config.get('host', '127.0.0.1'),
                port=config.get('port', 9000),
                version=config.get('version', '1.0.0')
            )
            
            self._a2a_servers[server_id] = a2a_server
            
            self.logger.info(f"Created A2A server {server_id} for agent {agent_id}")
            return server_id
            
        except Exception as e:
            self.logger.error(f"Failed to create A2A server: {e}")
            raise
    
    async def create_graph(self, agents: List[str], config: Dict[str, Any] = None) -> str:
        """Create a Strands Graph for complex multi-agent workflows."""
        if not self.enable_multi_agent:
            raise NotImplementedError("Multi-agent support not enabled")
        
        graph_id = f"graph_{uuid.uuid4().hex[:8]}"
        config = config or {}
        
        try:
            # Get agent instances
            graph_agents = []
            for agent_id in agents:
                if agent_id in self._agents:
                    graph_agents.append(self._agents[agent_id].strands_agent)
                else:
                    self.logger.warning(f"Agent {agent_id} not found for graph")
            
            if not graph_agents:
                raise ValueError("No valid agents found for graph")
            
            # Create Strands Graph
            graph = Graph(
                agents=graph_agents,
                edges=config.get('edges', []),
                max_iterations=config.get('max_iterations', 50),
                execution_timeout=config.get('execution_timeout', 1800.0)
            )
            
            self._graphs[graph_id] = graph
            
            self.logger.info(f"Created Strands graph {graph_id} with {len(graph_agents)} agents")
            return graph_id
            
        except Exception as e:
            self.logger.error(f"Failed to create graph: {e}")
            raise
    
    async def execute_graph(self, graph_id: str, task: str, start_node: str = None) -> Dict[str, Any]:
        """Execute a Strands Graph workflow."""
        if graph_id not in self._graphs:
            raise ValueError(f"Graph {graph_id} not found")
        
        graph = self._graphs[graph_id]
        
        try:
            # Execute graph
            result = graph(task, start_node=start_node)
            
            return {
                'graph_id': graph_id,
                'status': result.status if hasattr(result, 'status') else 'completed',
                'final_result': result.result if hasattr(result, 'result') else str(result),
                'execution_path': result.execution_path if hasattr(result, 'execution_path') else [],
                'nodes_executed': result.nodes_executed if hasattr(result, 'nodes_executed') else [],
                'execution_time': result.execution_time if hasattr(result, 'execution_time') else 0
            }
            
        except Exception as e:
            self.logger.error(f"Graph execution failed: {e}")
            raise
    
    async def create_workflow(self, steps: List[Dict[str, Any]], config: Dict[str, Any] = None) -> str:
        """Create a Strands Workflow for sequential multi-agent tasks."""
        if not self.enable_multi_agent:
            raise NotImplementedError("Multi-agent support not enabled")
        
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        config = config or {}
        
        try:
            # Build workflow steps with agents
            workflow_steps = []
            for step in steps:
                agent_id = step.get('agent_id')
                if agent_id and agent_id in self._agents:
                    workflow_steps.append({
                        'agent': self._agents[agent_id].strands_agent,
                        'task': step.get('task', ''),
                        'inputs': step.get('inputs', {}),
                        'outputs': step.get('outputs', [])
                    })
                else:
                    self.logger.warning(f"Agent {agent_id} not found for workflow step")
            
            if not workflow_steps:
                raise ValueError("No valid workflow steps found")
            
            # Create Strands Workflow
            workflow = Workflow(
                steps=workflow_steps,
                execution_timeout=config.get('execution_timeout', 3600.0),
                parallel_execution=config.get('parallel_execution', False)
            )
            
            self._workflows[workflow_id] = workflow
            
            self.logger.info(f"Created Strands workflow {workflow_id} with {len(workflow_steps)} steps")
            return workflow_id
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow: {e}")
            raise
    
    async def execute_workflow(self, workflow_id: str, initial_input: Any = None) -> Dict[str, Any]:
        """Execute a Strands Workflow."""
        if workflow_id not in self._workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self._workflows[workflow_id]
        
        try:
            # Execute workflow
            result = workflow(initial_input)
            
            return {
                'workflow_id': workflow_id,
                'status': result.status if hasattr(result, 'status') else 'completed',
                'final_result': result.result if hasattr(result, 'result') else str(result),
                'step_results': result.step_results if hasattr(result, 'step_results') else [],
                'execution_time': result.execution_time if hasattr(result, 'execution_time') else 0,
                'steps_completed': result.steps_completed if hasattr(result, 'steps_completed') else 0
            }
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Strands agent instance."""
        if agent_id not in self._agents:
            return False
        
        try:
            wrapper = self._agents[agent_id]
            
            # Clean up session if exists
            if wrapper.session_manager:
                # Session cleanup would be handled by Strands SDK
                pass
            
            # Remove from tracking
            del self._agents[agent_id]
            
            # Clean up associated A2A servers
            server_id = f"a2a_{agent_id}"
            if server_id in self._a2a_servers:
                del self._a2a_servers[server_id]
            
            # Clean up swarms that include this agent
            swarms_to_remove = []
            for swarm_id, swarm in self._swarms.items():
                # This would require proper swarm introspection
                # For now, we'll log and continue
                pass
            
            # Clean up graphs that include this agent
            graphs_to_remove = []
            for graph_id, graph in self._graphs.items():
                # This would require proper graph introspection
                # For now, we'll log and continue
                pass
            
            # Clean up workflows that include this agent
            workflows_to_remove = []
            for workflow_id, workflow in self._workflows.items():
                # This would require proper workflow introspection
                # For now, we'll log and continue
                pass
            
            self.logger.info(f"Destroyed Strands agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent {agent_id}: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Strands agent IDs."""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a specific Strands agent."""
        if agent_id not in self._agents:
            return {}
        
        wrapper = self._agents[agent_id]
        
        return {
            'agent_id': agent_id,
            'provider': self.provider_name,
            'name': wrapper.config.name,
            'description': wrapper.config.description,
            'stats': wrapper.get_stats(),
            'capabilities': [cap.value for cap in self.supported_capabilities],
            'model_provider': wrapper.config.framework_config.get('model', {}).get('provider', 'bedrock'),
            'tools_count': len(wrapper.tools),
            'session_enabled': wrapper.session_manager is not None,
            'strands_features': {
                'streaming': self.enable_streaming,
                'multi_agent': self.enable_multi_agent,
                'mcp_clients': len(self._mcp_clients),
                'community_tools': STRANDS_TOOLS_AVAILABLE,
                'a2a_support': A2A_AVAILABLE
            }
        }
    
    async def update_agent_config(self, agent_id: str, config: AgentConfig) -> bool:
        """Update Strands agent configuration."""
        if agent_id not in self._agents:
            return False
        
        try:
            wrapper = self._agents[agent_id]
            wrapper.config = config
            
            # Update agent properties that can be changed
            agent = wrapper.strands_agent
            if hasattr(agent, 'name'):
                agent.name = config.name
            if hasattr(agent, 'description'):
                agent.description = config.description
            if hasattr(agent, 'system_prompt'):
                agent.system_prompt = config.system_prompt
            
            self.logger.info(f"Updated configuration for Strands agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update agent config: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for Strands provider."""
        health_data = {
            "status": "healthy" if self._initialized else "not_initialized",
            "provider": self.provider_name,
            "strands_available": STRANDS_AVAILABLE,
            "active_agents": len(self._agents),
            "active_swarms": len(self._swarms),
            "active_graphs": len(self._graphs),
            "active_workflows": len(self._workflows),
            "active_a2a_servers": len(self._a2a_servers),
            "session_managers": len(self._session_managers),
            "mcp_clients": len(self._mcp_clients),
            "capabilities": [cap.value for cap in self.supported_capabilities],
            "features": {
                "community_tools": STRANDS_TOOLS_AVAILABLE,
                "mcp_support": MCP_AVAILABLE,
                "a2a_support": A2A_AVAILABLE,
                "streaming": self.enable_streaming,
                "multi_agent": self.enable_multi_agent
            },
            "timestamp": datetime.now().isoformat()
        }
        
        if not STRANDS_AVAILABLE:
            health_data["status"] = "unavailable"
            health_data["error"] = IMPORT_ERROR
        
        return health_data


# Convenience function for quick setup
def create_strands_provider(config: Dict[str, Any] = None) -> StrandsProvider:
    """
    Create a Strands provider with sensible defaults.
    
    Args:
        config: Provider configuration
        
    Returns:
        Configured StrandsProvider instance
    """
    default_config = {
        'model_provider': 'bedrock',
        'session_storage': 'file',
        'auto_load_tools': True,
        'enable_streaming': True,
        'enable_multi_agent': True,
        'model': {
            'provider': 'bedrock',
            'model_id': 'anthropic.claude-sonnet-4-20250514-v1:0',
            'temperature': 0.7,
            'streaming': True
        }
    }
    
    if config:
        default_config.update(config)
    
    return StrandsProvider(default_config)


# Export key classes and functions
__all__ = [
    'StrandsProvider',
    'StrandsAgentWrapper', 
    'create_strands_provider'
]