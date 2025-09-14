"""
OpenAI Swarm Agent Provider

Integrates OpenAI's Swarm framework for lightweight multi-agent coordination
into the CandyLLM ecosystem, enabling agent handoffs and collaborative workflows.
"""

import uuid
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass

from .base import (
    BaseAgentProvider, 
    AgentConfig, 
    AgentResponse, 
    ToolSpec, 
    AgentCapability,
    AgentSecurityLevel
)
from .security import AgentSecurityManager

try:
    from swarm import Swarm, Agent
    from swarm.types import Response, Result
    SWARM_AVAILABLE = True
except ImportError:
    SWARM_AVAILABLE = False
    # Mock classes for when Swarm is not available
    class Swarm:
        pass
    class Agent:
        pass
    class Response:
        pass
    class Result:
        pass


@dataclass
class SwarmAgentConfig:
    """Configuration for Swarm agents"""
    name: str
    instructions: str
    functions: List[Callable] = None
    tool_choice: str = None
    parallel_tool_calls: bool = True


@dataclass
class HandoffRule:
    """Rules for agent handoffs in Swarm"""
    from_agent: str
    to_agent: str
    condition: str
    trigger_keywords: List[str] = None


class SwarmOrchestrator:
    """Orchestrator for Swarm multi-agent workflows"""
    
    def __init__(self, orchestrator_id: str, swarm_client: Swarm,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.orchestrator_id = orchestrator_id
        self.swarm_client = swarm_client
        self.security_manager = security_manager
        self._agents: Dict[str, Agent] = {}
        self._handoff_rules: List[HandoffRule] = []
        self._conversation_history = []
        self._created_at = datetime.now()
        
    def add_agent(self, agent_name: str, agent_config: SwarmAgentConfig):
        """Add an agent to the swarm"""
        # Create Swarm agent
        functions = agent_config.functions or []
        
        # Add handoff functions
        handoff_functions = self._create_handoff_functions()
        functions.extend(handoff_functions)
        
        agent = Agent(
            name=agent_config.name,
            instructions=agent_config.instructions,
            functions=functions,
            tool_choice=agent_config.tool_choice,
            parallel_tool_calls=agent_config.parallel_tool_calls
        )
        
        self._agents[agent_name] = agent
        
    def add_handoff_rule(self, rule: HandoffRule):
        """Add a handoff rule between agents"""
        self._handoff_rules.append(rule)
        
    def _create_handoff_functions(self) -> List[Callable]:
        """Create handoff functions for agent transfers"""
        handoff_functions = []
        
        for agent_name in self._agents.keys():
            def create_handoff_func(target_agent_name):
                def handoff_function():
                    """Transfer conversation to another agent"""
                    return self._agents[target_agent_name]
                
                handoff_function.__name__ = f"transfer_to_{target_agent_name}"
                handoff_function.__doc__ = f"Transfer the conversation to {target_agent_name}"
                return handoff_function
            
            handoff_functions.append(create_handoff_func(agent_name))
        
        return handoff_functions
    
    async def execute_conversation(self, message: str, initial_agent: str = None) -> Dict[str, Any]:
        """Execute a conversation in the swarm"""
        if not self._agents:
            raise ValueError("No agents available in swarm")
        
        # Select initial agent
        if initial_agent and initial_agent in self._agents:
            agent = self._agents[initial_agent]
        else:
            agent = list(self._agents.values())[0]
        
        try:
            # Execute conversation
            if asyncio.iscoroutinefunction(self.swarm_client.run):
                response = await self.swarm_client.run(
                    agent=agent,
                    messages=[{"role": "user", "content": message}]
                )
            else:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.swarm_client.run(
                        agent=agent,
                        messages=[{"role": "user", "content": message}]
                    )
                )
            
            # Extract conversation results
            conversation_messages = []
            if hasattr(response, 'messages'):
                for msg in response.messages:
                    conversation_messages.append({
                        'role': msg.get('role', 'unknown'),
                        'content': msg.get('content', ''),
                        'agent': getattr(response, 'agent', {}).get('name', 'unknown') if hasattr(response, 'agent') else 'unknown',
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Get final response content
            final_content = ""
            if hasattr(response, 'messages') and response.messages:
                final_content = response.messages[-1].get('content', '')
            
            return {
                'content': final_content,
                'conversation_messages': conversation_messages,
                'final_agent': getattr(response, 'agent', {}).get('name', 'unknown') if hasattr(response, 'agent') else initial_agent,
                'total_messages': len(conversation_messages),
                'handoffs_count': len([msg for msg in conversation_messages if 'transfer_to_' in msg.get('content', '')])
            }
            
        except Exception as e:
            return {
                'content': '',
                'error': str(e),
                'conversation_messages': [],
                'final_agent': initial_agent or 'unknown'
            }


class SwarmAgentProvider(BaseAgentProvider):
    """
    Provider implementation for OpenAI Swarm framework.
    
    Enables lightweight multi-agent coordination with automatic handoffs
    and collaborative problem-solving workflows.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._orchestrators: Dict[str, SwarmOrchestrator] = {}
        self._individual_agents: Dict[str, Agent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._swarm_client = None
        self._security_manager = None
        
        if not SWARM_AVAILABLE:
            self.logger.warning("Swarm not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "swarm"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.MULTI_AGENT,
            AgentCapability.WORKFLOW_ORCHESTRATION,
            AgentCapability.TOOL_SYNTHESIS,
            AgentCapability.REASONING_CHAINS
        ]
    
    async def initialize(self) -> bool:
        """Initialize Swarm provider"""
        if not SWARM_AVAILABLE:
            self.logger.error("Swarm not available")
            return False
        
        try:
            # Initialize Swarm client
            self._swarm_client = Swarm()
            
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Swarm agent provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Swarm provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Swarm agent or orchestrator"""
        if not self._initialized:
            await self.initialize()
        
        if not SWARM_AVAILABLE:
            raise RuntimeError("Swarm not available")
        
        agent_id = f"swarm_{uuid.uuid4().hex[:8]}"
        
        try:
            # Check if this should be a multi-agent orchestrator
            if AgentCapability.MULTI_AGENT in config.capabilities:
                # Create orchestrator with multiple agents
                orchestrator = SwarmOrchestrator(
                    orchestrator_id=agent_id,
                    swarm_client=self._swarm_client,
                    security_manager=self._security_manager
                )
                
                # Create default agents for orchestrator
                specialist_config = SwarmAgentConfig(
                    name="Specialist",
                    instructions=f"You are a specialist agent. {config.description}",
                    functions=[]
                )
                
                coordinator_config = SwarmAgentConfig(
                    name="Coordinator", 
                    instructions="You are a coordinator agent that manages workflow and delegates tasks.",
                    functions=[]
                )
                
                reviewer_config = SwarmAgentConfig(
                    name="Reviewer",
                    instructions="You are a reviewer agent that validates and improves outputs.",
                    functions=[]
                )
                
                orchestrator.add_agent("specialist", specialist_config)
                orchestrator.add_agent("coordinator", coordinator_config) 
                orchestrator.add_agent("reviewer", reviewer_config)
                
                # Add default handoff rules
                orchestrator.add_handoff_rule(HandoffRule(
                    from_agent="coordinator",
                    to_agent="specialist",
                    condition="when specialized work is needed",
                    trigger_keywords=["analyze", "compute", "calculate", "research"]
                ))
                
                orchestrator.add_handoff_rule(HandoffRule(
                    from_agent="specialist", 
                    to_agent="reviewer",
                    condition="when work needs review",
                    trigger_keywords=["review", "check", "validate", "improve"]
                ))
                
                self._orchestrators[agent_id] = orchestrator
                
            else:
                # Create individual Swarm agent
                functions = []
                
                # Add tools as functions
                for tool_name in config.tools:
                    tool_func = await self._create_tool_function(tool_name)
                    if tool_func:
                        functions.append(tool_func)
                
                agent = Agent(
                    name=config.name or f"SwarmAgent_{agent_id}",
                    instructions=config.system_prompt or config.description,
                    functions=functions,
                    parallel_tool_calls=True
                )
                
                self._individual_agents[agent_id] = agent
            
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Swarm agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Swarm agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Swarm agent or orchestrator"""
        context = context or {}
        
        try:
            start_time = datetime.now()
            
            # Check if it's an orchestrator
            if agent_id in self._orchestrators:
                orchestrator = self._orchestrators[agent_id]
                result = await orchestrator.execute_conversation(
                    prompt,
                    context.get('initial_agent')
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return AgentResponse(
                    content=result.get('content', ''),
                    agent_id=agent_id,
                    provider=self.provider_name,
                    metadata={
                        'execution_time_seconds': execution_time,
                        'conversation_messages': result.get('conversation_messages', []),
                        'final_agent': result.get('final_agent'),
                        'total_messages': result.get('total_messages', 0),
                        'handoffs_count': result.get('handoffs_count', 0),
                        'agent_type': 'orchestrator'
                    },
                    error=result.get('error')
                )
            
            # Individual agent execution
            elif agent_id in self._individual_agents:
                agent = self._individual_agents[agent_id]
                
                # Execute with Swarm client
                if asyncio.iscoroutinefunction(self._swarm_client.run):
                    response = await self._swarm_client.run(
                        agent=agent,
                        messages=[{"role": "user", "content": prompt}]
                    )
                else:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._swarm_client.run(
                            agent=agent,
                            messages=[{"role": "user", "content": prompt}]
                        )
                    )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Extract response content
                content = ""
                if hasattr(response, 'messages') and response.messages:
                    content = response.messages[-1].get('content', '')
                
                return AgentResponse(
                    content=content,
                    agent_id=agent_id,
                    provider=self.provider_name,
                    metadata={
                        'execution_time_seconds': execution_time,
                        'agent_type': 'individual',
                        'agent_name': agent.name
                    }
                )
            
            else:
                raise ValueError(f"Agent {agent_id} not found")
                
        except Exception as e:
            self.logger.error(f"Swarm agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Swarm agent"""
        try:
            # Create function from tool spec
            def tool_function(**kwargs):
                if tool_spec.function:
                    return tool_spec.function(**kwargs)
                return f"Tool {tool_spec.name} executed with {kwargs}"
            
            tool_function.__name__ = tool_spec.name
            tool_function.__doc__ = tool_spec.description
            
            # Add to individual agent
            if agent_id in self._individual_agents:
                agent = self._individual_agents[agent_id]
                if hasattr(agent, 'functions'):
                    agent.functions.append(tool_function)
                else:
                    agent.functions = [tool_function]
                
                self.logger.info(f"Registered tool {tool_spec.name} with Swarm agent {agent_id}")
                return True
            
            # Add to orchestrator agents
            elif agent_id in self._orchestrators:
                orchestrator = self._orchestrators[agent_id]
                for agent in orchestrator._agents.values():
                    if hasattr(agent, 'functions'):
                        agent.functions.append(tool_function)
                    else:
                        agent.functions = [tool_function]
                
                self.logger.info(f"Registered tool {tool_spec.name} with Swarm orchestrator {agent_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Swarm agent capabilities"""
        if agent_id not in self._individual_agents and agent_id not in self._orchestrators:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Use Swarm to generate tool implementation
            synthesis_prompt = f"""
            Create a Python function that implements the following tool:
            
            Description: {tool_description}
            {f'Examples: {examples}' if examples else ''}
            
            Requirements:
            1. Function should be self-contained and importable
            2. Include proper error handling and validation
            3. Return structured results
            4. Add comprehensive docstring
            
            Provide only the function definition.
            """
            
            # Execute synthesis
            response = await self.execute_agent(agent_id, synthesis_prompt)
            
            # Create tool spec
            tool_spec = ToolSpec(
                name=f"swarm_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            self.logger.info(f"Synthesized tool for Swarm agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Swarm agent"""
        try:
            removed = False
            
            if agent_id in self._orchestrators:
                del self._orchestrators[agent_id]
                removed = True
            
            if agent_id in self._individual_agents:
                del self._individual_agents[agent_id]
                removed = True
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            if removed:
                self.logger.info(f"Destroyed Swarm agent {agent_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Swarm agent IDs"""
        return list(set(self._orchestrators.keys()) | set(self._individual_agents.keys()))
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Swarm agent"""
        if agent_id not in self._agent_configs:
            return {}
        
        config = self._agent_configs[agent_id]
        info = {
            'agent_id': agent_id,
            'provider': self.provider_name,
            'config': config.__dict__
        }
        
        if agent_id in self._orchestrators:
            orchestrator = self._orchestrators[agent_id]
            info.update({
                'type': 'orchestrator',
                'agents': list(orchestrator._agents.keys()),
                'handoff_rules': len(orchestrator._handoff_rules),
                'created_at': orchestrator._created_at.isoformat()
            })
        elif agent_id in self._individual_agents:
            agent = self._individual_agents[agent_id]
            info.update({
                'type': 'individual_agent',
                'name': agent.name,
                'instructions': agent.instructions,
                'functions_count': len(agent.functions) if hasattr(agent, 'functions') else 0
            })
        
        return info
    
    async def _create_tool_function(self, tool_name: str) -> Optional[Callable]:
        """Create a function from a tool name"""
        # This would integrate with CandyLLM's tool registry
        # For now, create basic tool functions
        
        if tool_name == "search":
            def search_tool(query: str) -> str:
                """Search for information"""
                return f"Search results for: {query}"
            return search_tool
        
        elif tool_name == "calculator":
            def calculator_tool(expression: str) -> str:
                """Calculate mathematical expressions"""
                try:
                    # Simple safe evaluation
                    result = eval(expression.replace("^", "**"))
                    return f"Result: {result}"
                except:
                    return "Invalid expression"
            return calculator_tool
        
        elif tool_name == "file_read":
            def file_read_tool(file_path: str) -> str:
                """Read file contents"""
                return f"Contents of {file_path}"
            return file_read_tool
        
        return None