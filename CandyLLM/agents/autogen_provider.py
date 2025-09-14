"""
AutoGen Agent Provider

Integrates Microsoft AutoGen's multi-agent conversation framework into CandyLLM,
enabling sophisticated group chat scenarios and role-based agent interactions.
"""

import uuid
import asyncio
from typing import Dict, List, Optional, Any, Union
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
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    from autogen.agentchat.conversable_agent import ConversableAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    # Mock classes for when AutoGen is not available
    class AssistantAgent:
        pass
    class UserProxyAgent:
        pass
    class GroupChat:
        pass
    class GroupChatManager:
        pass


@dataclass
class AutoGenAgentConfig:
    """Configuration for AutoGen agent types"""
    name: str
    role: str
    system_message: str
    human_input_mode: str = "NEVER"  # NEVER, TERMINATE, ALWAYS
    max_consecutive_auto_reply: int = 10
    code_execution_config: Dict[str, Any] = None
    function_map: Dict[str, Any] = None


@dataclass
class GroupChatConfig:
    """Configuration for AutoGen group chat"""
    agents: List[str]
    max_round: int = 10
    admin_name: str = "Admin"
    speaker_selection_method: str = "auto"  # auto, manual, random, round_robin
    allow_repeat_speaker: bool = True


class AutoGenGroupChat:
    """Wrapper for AutoGen group chat with CandyLLM integration"""
    
    def __init__(self, group_id: str, config: GroupChatConfig,
                 llm_config: Dict[str, Any], security_manager: Optional[AgentSecurityManager] = None):
        self.group_id = group_id
        self.config = config
        self.llm_config = llm_config
        self.security_manager = security_manager
        self._agents: Dict[str, ConversableAgent] = {}
        self._group_chat = None
        self._manager = None
        self._created_at = datetime.now()
        
    async def initialize(self):
        """Initialize the AutoGen group chat"""
        if not AUTOGEN_AVAILABLE:
            raise RuntimeError("AutoGen not available")
        
        # Create agents
        autogen_agents = []
        for agent_name in self.config.agents:
            agent = self._create_autogen_agent(agent_name)
            if agent:
                autogen_agents.append(agent)
                self._agents[agent_name] = agent
        
        if not autogen_agents:
            raise ValueError("No valid agents created for group chat")
        
        # Create group chat
        self._group_chat = GroupChat(
            agents=autogen_agents,
            messages=[],
            max_round=self.config.max_round,
            speaker_selection_method=self.config.speaker_selection_method,
            allow_repeat_speaker=self.config.allow_repeat_speaker
        )
        
        # Create group chat manager
        self._manager = GroupChatManager(
            groupchat=self._group_chat,
            llm_config=self.llm_config,
            name=self.config.admin_name
        )
    
    def _create_autogen_agent(self, agent_name: str) -> Optional[ConversableAgent]:
        """Create an AutoGen agent based on configuration"""
        # This would be configured based on agent requirements
        if "assistant" in agent_name.lower():
            return AssistantAgent(
                name=agent_name,
                system_message=f"You are {agent_name}, a helpful AI assistant.",
                llm_config=self.llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=10
            )
        elif "proxy" in agent_name.lower() or "user" in agent_name.lower():
            return UserProxyAgent(
                name=agent_name,
                system_message=f"You are {agent_name}, representing user interests.",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=5,
                code_execution_config={"work_dir": "coding", "use_docker": False}
            )
        else:
            # Generic assistant
            return AssistantAgent(
                name=agent_name,
                system_message=f"You are {agent_name}.",
                llm_config=self.llm_config,
                human_input_mode="NEVER"
            )
    
    async def execute_conversation(self, message: str, initiator: str = None) -> Dict[str, Any]:
        """Execute a group conversation"""
        if not self._manager:
            await self.initialize()
        
        try:
            # Find initiator agent
            initiator_agent = None
            if initiator and initiator in self._agents:
                initiator_agent = self._agents[initiator]
            else:
                # Use first agent as initiator
                initiator_agent = list(self._agents.values())[0]
            
            # Start conversation
            if asyncio.iscoroutinefunction(initiator_agent.initiate_chat):
                result = await initiator_agent.initiate_chat(
                    self._manager,
                    message=message
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: initiator_agent.initiate_chat(self._manager, message=message)
                )
            
            # Extract conversation results
            conversation_history = []
            if hasattr(self._group_chat, 'messages'):
                for msg in self._group_chat.messages:
                    conversation_history.append({
                        'sender': getattr(msg, 'name', 'unknown'),
                        'content': getattr(msg, 'content', str(msg)),
                        'timestamp': datetime.now().isoformat()
                    })
            
            return {
                'result': str(result) if result else "Conversation completed",
                'conversation_history': conversation_history,
                'participants': list(self._agents.keys()),
                'total_messages': len(conversation_history)
            }
            
        except Exception as e:
            return {
                'result': '',
                'error': str(e),
                'conversation_history': [],
                'participants': list(self._agents.keys())
            }


class AutoGenAgentProvider(BaseAgentProvider):
    """
    Provider implementation for Microsoft AutoGen framework.
    
    Enables sophisticated multi-agent conversations, role-based interactions,
    and group chat scenarios within the CandyLLM ecosystem.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._group_chats: Dict[str, AutoGenGroupChat] = {}
        self._individual_agents: Dict[str, ConversableAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not AUTOGEN_AVAILABLE:
            self.logger.warning("AutoGen not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "autogen"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.MULTI_AGENT,
            AgentCapability.ROLE_PLAYING,
            AgentCapability.CODE_EXECUTION,
            AgentCapability.WORKFLOW_ORCHESTRATION,
            AgentCapability.HUMAN_IN_LOOP,
            AgentCapability.REASONING_CHAINS
        ]
    
    async def initialize(self) -> bool:
        """Initialize AutoGen provider"""
        if not AUTOGEN_AVAILABLE:
            self.logger.error("AutoGen not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("AutoGen agent provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AutoGen provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new AutoGen agent or group chat"""
        if not self._initialized:
            await self.initialize()
        
        if not AUTOGEN_AVAILABLE:
            raise RuntimeError("AutoGen not available")
        
        agent_id = f"autogen_{uuid.uuid4().hex[:8]}"
        
        try:
            # Prepare LLM config
            llm_config = {
                "model": config.model or self.config.get('default_model', 'gpt-3.5-turbo'),
                "api_key": self.config.get('api_key'),
                "temperature": config.temperature,
                "timeout": 60
            }
            
            # Check if this should be a group chat (multi-agent capability)
            if AgentCapability.MULTI_AGENT in config.capabilities:
                # Create group chat
                group_config = GroupChatConfig(
                    agents=["assistant_1", "assistant_2", "user_proxy"],
                    max_round=self.config.get('max_rounds', 10),
                    admin_name=config.name or "GroupManager"
                )
                
                group_chat = AutoGenGroupChat(
                    group_id=agent_id,
                    config=group_config,
                    llm_config=llm_config,
                    security_manager=self._security_manager
                )
                
                self._group_chats[agent_id] = group_chat
                
            else:
                # Create individual agent
                if AgentCapability.CODE_EXECUTION in config.capabilities:
                    agent = UserProxyAgent(
                        name=config.name or f"Agent_{agent_id}",
                        system_message=config.system_prompt or config.description,
                        human_input_mode="NEVER",
                        max_consecutive_auto_reply=10,
                        code_execution_config={
                            "work_dir": "autogen_coding",
                            "use_docker": False,
                            "timeout": 60
                        },
                        llm_config=llm_config
                    )
                else:
                    agent = AssistantAgent(
                        name=config.name or f"Agent_{agent_id}",
                        system_message=config.system_prompt or config.description,
                        llm_config=llm_config,
                        human_input_mode="NEVER",
                        max_consecutive_auto_reply=15
                    )
                
                self._individual_agents[agent_id] = agent
            
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created AutoGen agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create AutoGen agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute an AutoGen agent or group chat"""
        context = context or {}
        
        try:
            start_time = datetime.now()
            
            # Check if it's a group chat
            if agent_id in self._group_chats:
                group_chat = self._group_chats[agent_id]
                result = await group_chat.execute_conversation(
                    prompt, 
                    context.get('initiator')
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return AgentResponse(
                    content=result.get('result', ''),
                    agent_id=agent_id,
                    provider=self.provider_name,
                    metadata={
                        'execution_time_seconds': execution_time,
                        'conversation_history': result.get('conversation_history', []),
                        'participants': result.get('participants', []),
                        'total_messages': result.get('total_messages', 0),
                        'agent_type': 'group_chat'
                    },
                    error=result.get('error')
                )
            
            # Individual agent execution
            elif agent_id in self._individual_agents:
                agent = self._individual_agents[agent_id]
                
                # Create a simple user proxy for conversation
                user_proxy = UserProxyAgent(
                    name="TempUser",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=1,
                    code_execution_config=False
                )
                
                # Execute conversation
                if asyncio.iscoroutinefunction(user_proxy.initiate_chat):
                    result = await user_proxy.initiate_chat(agent, message=prompt)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: user_proxy.initiate_chat(agent, message=prompt)
                    )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return AgentResponse(
                    content=str(result) if result else "Conversation completed",
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
            self.logger.error(f"AutoGen agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with AutoGen agent"""
        # AutoGen handles tools through function maps
        try:
            if agent_id in self._individual_agents:
                agent = self._individual_agents[agent_id]
                
                # Create function for AutoGen
                def tool_function(**kwargs):
                    if tool_spec.function:
                        return tool_spec.function(**kwargs)
                    return f"Tool {tool_spec.name} executed with {kwargs}"
                
                # Register function with agent
                if hasattr(agent, 'register_function'):
                    agent.register_function({
                        tool_spec.name: tool_function
                    })
                    
                self.logger.info(f"Registered tool {tool_spec.name} with AutoGen agent {agent_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using AutoGen's code execution capabilities"""
        if agent_id not in self._individual_agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Use AutoGen agent to generate tool code
            synthesis_prompt = f"""
            Create a Python function that implements the following tool:
            
            Description: {tool_description}
            {f'Examples: {examples}' if examples else ''}
            
            Requirements:
            1. Function should be self-contained
            2. Include proper error handling
            3. Return meaningful results
            4. Add type hints and documentation
            
            Provide only the function definition.
            """
            
            # Execute synthesis using the agent
            response = await self.execute_agent(agent_id, synthesis_prompt)
            
            # Create tool spec from response
            tool_spec = ToolSpec(
                name=f"autogen_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            self.logger.info(f"Synthesized tool for AutoGen agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy an AutoGen agent"""
        try:
            removed = False
            
            if agent_id in self._group_chats:
                del self._group_chats[agent_id]
                removed = True
            
            if agent_id in self._individual_agents:
                del self._individual_agents[agent_id]
                removed = True
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            if removed:
                self.logger.info(f"Destroyed AutoGen agent {agent_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active AutoGen agent IDs"""
        return list(set(self._group_chats.keys()) | set(self._individual_agents.keys()))
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about an AutoGen agent"""
        if agent_id not in self._agent_configs:
            return {}
        
        config = self._agent_configs[agent_id]
        info = {
            'agent_id': agent_id,
            'provider': self.provider_name,
            'config': config.__dict__
        }
        
        if agent_id in self._group_chats:
            group_chat = self._group_chats[agent_id]
            info.update({
                'type': 'group_chat',
                'participants': list(group_chat._agents.keys()),
                'created_at': group_chat._created_at.isoformat()
            })
        elif agent_id in self._individual_agents:
            agent = self._individual_agents[agent_id]
            info.update({
                'type': 'individual_agent',
                'name': agent.name,
                'system_message': getattr(agent, 'system_message', '')
            })
        
        return info