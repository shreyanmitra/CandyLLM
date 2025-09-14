"""
CrewAI Agent Provider

Integrates CrewAI's role-based multi-agent capabilities into the CandyLLM provider ecosystem,
enabling coordinated team-based AI workflows with security controls.
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
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool as CrewAIBaseTool
    from crewai.agent import Agent as CrewAIAgent
    from crewai.task import Task as CrewAITask
    from crewai.crew import Crew as CrewAICrew
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    # Mock classes for when CrewAI is not available
    class Agent:
        pass
    class Task:
        pass
    class Crew:
        pass
    class CrewAIBaseTool:
        pass


@dataclass
class CrewMember:
    """Configuration for a crew member agent"""
    role: str
    goal: str
    backstory: str
    tools: List[str] = None
    allow_delegation: bool = False
    verbose: bool = True
    max_iter: int = 10


@dataclass
class CrewTask:
    """Configuration for a crew task"""
    description: str
    expected_output: str
    agent_role: str
    tools: List[str] = None
    dependencies: List[str] = None
    async_execution: bool = False


@dataclass
class CrewConfig:
    """Configuration for a complete crew"""
    name: str
    description: str
    members: List[CrewMember]
    tasks: List[CrewTask]
    process: str = "sequential"  # sequential, hierarchical
    verbose: bool = True
    memory: bool = False


class CrewAIToolAdapter:
    """Adapter to convert between CandyLLM ToolSpec and CrewAI BaseTool"""
    
    @staticmethod
    def to_crewai_tool(tool_spec: ToolSpec) -> CrewAIBaseTool:
        """Convert CandyLLM ToolSpec to CrewAI BaseTool"""
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI not available")
        
        class CandyLLMCrewTool(CrewAIBaseTool):
            name: str = tool_spec.name
            description: str = tool_spec.description
            
            def _run(self, *args, **kwargs) -> str:
                """Run the tool"""
                if tool_spec.function:
                    try:
                        # Handle both positional and keyword arguments
                        if args and not kwargs:
                            result = tool_spec.function(*args)
                        elif kwargs and not args:
                            result = tool_spec.function(**kwargs)
                        else:
                            result = tool_spec.function(*args, **kwargs)
                        return str(result)
                    except Exception as e:
                        return f"Tool execution failed: {e}"
                return f"Tool {self.name} executed"
        
        return CandyLLMCrewTool()
    
    @staticmethod
    def from_crewai_tool(crewai_tool: CrewAIBaseTool) -> ToolSpec:
        """Convert CrewAI BaseTool to CandyLLM ToolSpec"""
        return ToolSpec(
            name=crewai_tool.name,
            description=crewai_tool.description,
            parameters={
                'type': 'object',
                'properties': {}  # CrewAI tools don't have structured schemas
            },
            function=crewai_tool._run if hasattr(crewai_tool, '_run') else None
        )


class CandyLLMCrewAIAgent:
    """Wrapper for CrewAI agents with CandyLLM integration"""
    
    def __init__(self, crew_id: str, crew_config: CrewConfig, 
                 llm_config: Dict[str, Any], security_manager: Optional[AgentSecurityManager] = None):
        self.crew_id = crew_id
        self.config = crew_config
        self.llm_config = llm_config
        self.security_manager = security_manager
        self._crew = None
        self._agents = {}
        self._tasks = {}
        self._tools = {}
        self._created_at = datetime.now()
        
    async def initialize(self):
        """Initialize the CrewAI crew with agents and tasks"""
        if not CREWAI_AVAILABLE:
            raise RuntimeError("CrewAI not available")
        
        # Create LLM instance for agents
        llm = await self._create_llm()
        
        # Create individual agents
        crew_agents = []
        for member in self.config.members:
            agent_tools = []
            if member.tools:
                for tool_name in member.tools:
                    tool = await self._get_tool(tool_name)
                    if tool:
                        agent_tools.append(tool)
            
            agent = Agent(
                role=member.role,
                goal=member.goal,
                backstory=member.backstory,
                tools=agent_tools,
                allow_delegation=member.allow_delegation,
                verbose=member.verbose,
                llm=llm,
                max_iter=member.max_iter
            )
            
            crew_agents.append(agent)
            self._agents[member.role] = agent
        
        # Create tasks
        crew_tasks = []
        for task_config in self.config.tasks:
            # Find the agent for this task
            agent = self._agents.get(task_config.agent_role)
            if not agent:
                raise ValueError(f"Agent with role {task_config.agent_role} not found")
            
            task_tools = []
            if task_config.tools:
                for tool_name in task_config.tools:
                    tool = await self._get_tool(tool_name)
                    if tool:
                        task_tools.append(tool)
            
            task = Task(
                description=task_config.description,
                expected_output=task_config.expected_output,
                agent=agent,
                tools=task_tools,
                async_execution=task_config.async_execution
            )
            
            crew_tasks.append(task)
            self._tasks[task_config.description[:50]] = task
        
        # Create the crew
        process = Process.sequential if self.config.process == "sequential" else Process.hierarchical
        
        self._crew = Crew(
            agents=crew_agents,
            tasks=crew_tasks,
            process=process,
            verbose=self.config.verbose,
            memory=self.config.memory
        )
    
    async def execute(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the crew workflow"""
        if not self._crew:
            await self.initialize()
        
        try:
            # Execute crew
            if asyncio.iscoroutinefunction(self._crew.kickoff):
                result = await self._crew.kickoff(inputs=inputs or {})
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._crew.kickoff, inputs or {}
                )
            
            return {
                'result': str(result),
                'crew_id': self.crew_id,
                'agents_used': list(self._agents.keys()),
                'tasks_completed': len(self._tasks)
            }
            
        except Exception as e:
            raise RuntimeError(f"Crew execution failed: {e}")
    
    async def _create_llm(self):
        """Create LLM instance for CrewAI agents"""
        llm_type = self.llm_config.get('type', 'openai')
        
        if llm_type == 'openai':
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.llm_config.get('model', 'gpt-3.5-turbo'),
                temperature=self.llm_config.get('temperature', 0.7),
                openai_api_key=self.llm_config.get('api_key')
            )
        elif llm_type == 'anthropic':
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.llm_config.get('model', 'claude-3-sonnet-20240229'),
                anthropic_api_key=self.llm_config.get('api_key')
            )
        else:
            raise ValueError(f"Unsupported LLM type for CrewAI: {llm_type}")
    
    async def _get_tool(self, tool_name: str) -> Optional[CrewAIBaseTool]:
        """Get a tool by name"""
        if tool_name in self._tools:
            return self._tools[tool_name]
        
        # Try to get built-in tools
        builtin_tool = await self._get_builtin_tool(tool_name)
        if builtin_tool:
            self._tools[tool_name] = builtin_tool
            return builtin_tool
        
        return None
    
    async def _get_builtin_tool(self, tool_name: str) -> Optional[CrewAIBaseTool]:
        """Get built-in CrewAI tools"""
        try:
            if tool_name == 'web_search':
                from crewai_tools import SerperDevTool
                return SerperDevTool()
            elif tool_name == 'file_read':
                from crewai_tools import FileReadTool
                return FileReadTool()
            elif tool_name == 'directory_read':
                from crewai_tools import DirectoryReadTool
                return DirectoryReadTool()
            elif tool_name == 'code_interpreter':
                from crewai_tools import CodeInterpreterTool
                return CodeInterpreterTool()
            elif tool_name == 'scrape_website':
                from crewai_tools import ScrapeWebsiteTool
                return ScrapeWebsiteTool()
        except ImportError:
            pass
        
        return None


class CrewAIAgentProvider(BaseAgentProvider):
    """
    Provider implementation for CrewAI multi-agent teams.
    
    Enables role-based agent coordination and team workflows within
    the CandyLLM unified provider ecosystem.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._crews: Dict[str, CandyLLMCrewAIAgent] = {}
        self._crew_configs: Dict[str, CrewConfig] = {}
        self._security_manager = None
        
        if not CREWAI_AVAILABLE:
            self.logger.warning("CrewAI not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "crewai"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.MULTI_AGENT,
            AgentCapability.ROLE_PLAYING,
            AgentCapability.WORKFLOW_ORCHESTRATION,
            AgentCapability.TOOL_SYNTHESIS,
            AgentCapability.WEB_BROWSING,
            AgentCapability.FILE_OPERATIONS,
            AgentCapability.CODE_EXECUTION
        ]
    
    async def initialize(self) -> bool:
        """Initialize CrewAI provider"""
        if not CREWAI_AVAILABLE:
            self.logger.error("CrewAI not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("CrewAI agent provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CrewAI provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new CrewAI agent (crew) instance"""
        if not self._initialized:
            await self.initialize()
        
        if not CREWAI_AVAILABLE:
            raise RuntimeError("CrewAI not available")
        
        crew_id = f"crewai_{uuid.uuid4().hex[:8]}"
        
        try:
            # Convert AgentConfig to CrewConfig
            crew_config = await self._convert_agent_config_to_crew(config)
            
            # Create crew wrapper
            llm_config = {
                'type': self.config.get('llm_type', 'openai'),
                'model': config.model or self.config.get('default_model', 'gpt-3.5-turbo'),
                'temperature': config.temperature,
                'api_key': self.config.get('api_key')
            }
            
            crew_agent = CandyLLMCrewAIAgent(
                crew_id=crew_id,
                crew_config=crew_config,
                llm_config=llm_config,
                security_manager=self._security_manager
            )
            
            self._crews[crew_id] = crew_agent
            self._crew_configs[crew_id] = crew_config
            
            self.logger.info(f"Created CrewAI crew {crew_id} with {len(crew_config.members)} agents")
            return crew_id
            
        except Exception as e:
            self.logger.error(f"Failed to create CrewAI crew: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a CrewAI crew with the given prompt"""
        if agent_id not in self._crews:
            raise ValueError(f"Crew {agent_id} not found")
        
        crew = self._crews[agent_id]
        context = context or {}
        
        try:
            # Prepare inputs for crew
            inputs = {
                'main_task': prompt,
                **context
            }
            
            # Execute crew
            start_time = datetime.now()
            result = await crew.execute(inputs)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                content=result.get('result', ''),
                agent_id=agent_id,
                provider=self.provider_name,
                tools_used=context.get('tools_used', []),
                metadata={
                    'execution_time_seconds': execution_time,
                    'agents_used': result.get('agents_used', []),
                    'tasks_completed': result.get('tasks_completed', 0),
                    'crew_process': self._crew_configs[agent_id].process
                }
            )
            
        except Exception as e:
            self.logger.error(f"CrewAI execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with a CrewAI crew"""
        if agent_id not in self._crews:
            return False
        
        try:
            # Convert to CrewAI tool
            crewai_tool = CrewAIToolAdapter.to_crewai_tool(tool_spec)
            
            # Register with crew
            crew = self._crews[agent_id]
            crew._tools[tool_spec.name] = crewai_tool
            
            # Security validation
            if self._security_manager:
                self._security_manager.register_tool(agent_id, tool_spec)
            
            self.logger.info(f"Registered tool {tool_spec.name} with CrewAI crew {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using CrewAI's capabilities"""
        if agent_id not in self._crews:
            raise ValueError(f"Crew {agent_id} not found")
        
        try:
            # Create a specialized crew for tool synthesis
            synthesis_config = CrewConfig(
                name="tool_synthesis",
                description="Synthesize new tools",
                members=[
                    CrewMember(
                        role="Tool Developer",
                        goal="Create effective and secure tools based on requirements",
                        backstory="You are an expert tool developer with deep knowledge of Python and security best practices.",
                        tools=["code_interpreter"]
                    )
                ],
                tasks=[
                    CrewTask(
                        description=f"Create a Python function that implements: {tool_description}",
                        expected_output="A complete Python function with documentation and error handling",
                        agent_role="Tool Developer"
                    )
                ]
            )
            
            # Execute synthesis (simplified)
            tool_spec = ToolSpec(
                name=f"synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            self.logger.info(f"Synthesized tool for CrewAI crew {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a CrewAI crew"""
        if agent_id not in self._crews:
            return False
        
        try:
            # Cleanup crew resources
            del self._crews[agent_id]
            del self._crew_configs[agent_id]
            
            # Cleanup security manager
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed CrewAI crew {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy crew: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active CrewAI crew IDs"""
        return list(self._crews.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a CrewAI crew"""
        if agent_id not in self._crews:
            return {}
        
        crew = self._crews[agent_id]
        config = self._crew_configs[agent_id]
        
        return {
            'agent_id': agent_id,
            'provider': self.provider_name,
            'crew_name': config.name,
            'description': config.description,
            'members': [member.role for member in config.members],
            'tasks': len(config.tasks),
            'process': config.process,
            'created_at': crew._created_at.isoformat()
        }
    
    # Helper methods
    
    async def _convert_agent_config_to_crew(self, config: AgentConfig) -> CrewConfig:
        """Convert AgentConfig to CrewConfig"""
        
        # Determine crew structure based on config
        if AgentCapability.MULTI_AGENT in config.capabilities:
            # Create specialized multi-agent crew
            members = [
                CrewMember(
                    role="Lead Agent",
                    goal=f"Accomplish the main objective: {config.description}",
                    backstory="You are the lead agent responsible for coordinating the team and ensuring objectives are met.",
                    tools=config.tools,
                    allow_delegation=True
                ),
                CrewMember(
                    role="Specialist Agent", 
                    goal="Provide specialized expertise and support",
                    backstory="You are a specialist with deep domain knowledge to support the lead agent.",
                    tools=config.tools[:3] if len(config.tools) > 3 else config.tools  # Limit tools
                ),
                CrewMember(
                    role="Quality Assurance",
                    goal="Review and validate the work quality",
                    backstory="You are responsible for ensuring high quality outputs and catching any issues.",
                    tools=[]
                )
            ]
            
            tasks = [
                CrewTask(
                    description="Analyze the main task and create a detailed plan",
                    expected_output="A comprehensive plan with clear steps and responsibilities",
                    agent_role="Lead Agent"
                ),
                CrewTask(
                    description="Execute the specialized aspects of the plan",
                    expected_output="Completed specialized work according to the plan",
                    agent_role="Specialist Agent"
                ),
                CrewTask(
                    description="Review the completed work and provide final validation",
                    expected_output="A quality-assured final result with recommendations",
                    agent_role="Quality Assurance"
                )
            ]
        else:
            # Simple single-agent crew
            members = [
                CrewMember(
                    role=config.name or "Main Agent",
                    goal=config.description or "Complete the assigned task effectively",
                    backstory="You are a capable AI agent designed to handle various tasks efficiently.",
                    tools=config.tools
                )
            ]
            
            tasks = [
                CrewTask(
                    description="Complete the assigned task",
                    expected_output="A comprehensive and accurate result",
                    agent_role=config.name or "Main Agent"
                )
            ]
        
        return CrewConfig(
            name=config.name,
            description=config.description,
            members=members,
            tasks=tasks,
            process="sequential",
            verbose=True,
            memory=AgentCapability.MEMORY_PERSISTENCE in config.capabilities
        )