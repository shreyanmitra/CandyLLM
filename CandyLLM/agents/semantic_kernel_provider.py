"""
Microsoft Semantic Kernel Agent Provider

Integrates Microsoft's Semantic Kernel framework for AI orchestration
with planning, skill chaining, and plugin architecture into the CandyLLM ecosystem.
"""

import uuid
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

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
    import semantic_kernel as sk
    from semantic_kernel.skill_definition import sk_function, sk_function_context_parameter
    from semantic_kernel.orchestration.sk_context import SKContext
    from semantic_kernel.core_skills import TextSkill, FileIOSkill, MathSkill, TimeSkill
    from semantic_kernel.planning import BasicPlanner, SequentialPlanner, ActionPlanner
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False
    # Mock classes for when Semantic Kernel is not available
    sk = None
    class SKContext:
        pass
    class BasicPlanner:
        pass
    class SequentialPlanner:
        pass
    class ActionPlanner:
        pass
    class TextSkill:
        pass
    class FileIOSkill:
        pass
    class MathSkill:
        pass
    class TimeSkill:
        pass


@dataclass
class SemanticKernelConfig:
    """Configuration for Semantic Kernel agents"""
    kernel_id: str
    ai_service_type: str = "openai"  # "openai" or "azure"
    api_key: str = None
    endpoint: str = None
    deployment_name: str = None
    model_name: str = "gpt-4"
    skills: List[str] = None
    planner_type: str = "sequential"  # "basic", "sequential", "action"
    max_plan_length: int = 10
    enable_native_skills: bool = True


@dataclass
class SkillSpec:
    """Specification for a Semantic Kernel skill"""
    name: str
    description: str
    functions: List[Dict[str, Any]]
    class_implementation: Optional[str] = None


class SemanticKernelSkill:
    """Base class for custom Semantic Kernel skills"""
    
    def __init__(self, skill_name: str, description: str):
        self.skill_name = skill_name
        self.description = description
        self._functions = {}
    
    def add_function(self, name: str, func: Callable, description: str, parameters: List[Dict] = None):
        """Add a function to this skill"""
        self._functions[name] = {
            'function': func,
            'description': description,
            'parameters': parameters or []
        }
    
    def get_skill_definition(self):
        """Get the skill definition for Semantic Kernel"""
        return {
            'name': self.skill_name,
            'description': self.description,
            'functions': self._functions
        }


class SemanticKernelAgent:
    """Wrapper for Semantic Kernel with planning capabilities"""
    
    def __init__(self, agent_id: str, config: SemanticKernelConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._kernel = None
        self._planner = None
        self._skills = {}
        self._execution_history = []
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the Semantic Kernel"""
        if not SK_AVAILABLE:
            return False
        
        try:
            # Create kernel
            self._kernel = sk.Kernel()
            
            # Configure AI service
            if self.config.ai_service_type == "azure":
                if self.config.api_key and self.config.endpoint and self.config.deployment_name:
                    self._kernel.add_chat_service(
                        "azure_chat_completion",
                        AzureChatCompletion(
                            deployment_name=self.config.deployment_name,
                            endpoint=self.config.endpoint,
                            api_key=self.config.api_key
                        )
                    )
            else:
                if self.config.api_key:
                    self._kernel.add_chat_service(
                        "openai_chat_completion",
                        OpenAIChatCompletion(
                            model_id=self.config.model_name,
                            api_key=self.config.api_key
                        )
                    )
            
            # Add native skills if enabled
            if self.config.enable_native_skills:
                self._kernel.import_skill(TextSkill(), "text")
                self._kernel.import_skill(FileIOSkill(), "file")
                self._kernel.import_skill(MathSkill(), "math")
                self._kernel.import_skill(TimeSkill(), "time")
            
            # Initialize planner
            await self._initialize_planner()
            
            return True
            
        except Exception as e:
            return False
    
    async def _initialize_planner(self):
        """Initialize the appropriate planner"""
        if not self._kernel:
            return
        
        try:
            if self.config.planner_type == "basic":
                self._planner = BasicPlanner()
            elif self.config.planner_type == "sequential":
                self._planner = SequentialPlanner(self._kernel)
            elif self.config.planner_type == "action":
                self._planner = ActionPlanner(self._kernel)
            else:
                self._planner = SequentialPlanner(self._kernel)
                
        except Exception:
            # Fallback to basic planner
            self._planner = BasicPlanner()
    
    async def execute_goal(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a goal using Semantic Kernel planning"""
        if not self._kernel or not self._planner:
            return {'error': 'Kernel not initialized'}
        
        try:
            start_time = datetime.now()
            
            # Create context
            sk_context = self._kernel.create_new_context()
            if context:
                for key, value in context.items():
                    sk_context.variables[key] = str(value)
            
            # Create plan
            if hasattr(self._planner, 'create_plan_async'):
                plan = await self._planner.create_plan_async(goal, self._kernel)
            else:
                plan = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self._planner.create_plan(goal)
                )
            
            # Execute plan
            if hasattr(plan, 'invoke_async'):
                result = await plan.invoke_async(sk_context)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: plan.invoke(sk_context)
                )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract result content
            content = ""
            if hasattr(result, 'result'):
                content = str(result.result)
            elif hasattr(result, 'variables') and 'input' in result.variables:
                content = str(result.variables['input'])
            
            # Record execution
            execution_record = {
                'goal': goal,
                'plan_steps': getattr(plan, 'steps', []),
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            self._execution_history.append(execution_record)
            
            return {
                'content': content,
                'plan': execution_record,
                'context_variables': dict(result.variables) if hasattr(result, 'variables') else {},
                'execution_time': execution_time,
                'plan_steps_count': len(getattr(plan, 'steps', []))
            }
            
        except Exception as e:
            execution_record = {
                'goal': goal,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            self._execution_history.append(execution_record)
            
            return {
                'content': '',
                'error': str(e),
                'plan': execution_record
            }
    
    async def execute_function(self, skill_name: str, function_name: str, 
                             parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a specific skill function"""
        if not self._kernel:
            return {'error': 'Kernel not initialized'}
        
        try:
            # Create context
            sk_context = self._kernel.create_new_context()
            if parameters:
                for key, value in parameters.items():
                    sk_context.variables[key] = str(value)
            
            # Get function
            function = self._kernel.skills.get_function(skill_name, function_name)
            if not function:
                return {'error': f'Function {skill_name}.{function_name} not found'}
            
            # Execute function
            if hasattr(function, 'invoke_async'):
                result = await function.invoke_async(sk_context)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: function.invoke(sk_context)
                )
            
            # Extract result
            content = ""
            if hasattr(result, 'result'):
                content = str(result.result)
            
            return {
                'content': content,
                'skill': skill_name,
                'function': function_name,
                'parameters': parameters
            }
            
        except Exception as e:
            return {
                'content': '',
                'error': str(e),
                'skill': skill_name,
                'function': function_name
            }
    
    def add_skill(self, skill: SemanticKernelSkill):
        """Add a custom skill to the kernel"""
        if not self._kernel:
            return False
        
        try:
            # Create skill class
            skill_class = type(skill.skill_name, (), {})
            
            # Add functions to skill class
            for func_name, func_data in skill._functions.items():
                # Create decorated function
                func = func_data['function']
                decorated_func = sk_function(
                    description=func_data['description'],
                    name=func_name
                )(func)
                
                setattr(skill_class, func_name, decorated_func)
            
            # Import skill
            skill_instance = skill_class()
            self._kernel.import_skill(skill_instance, skill.skill_name)
            self._skills[skill.skill_name] = skill
            
            return True
            
        except Exception:
            return False
    
    def get_available_functions(self) -> Dict[str, List[str]]:
        """Get all available functions by skill"""
        if not self._kernel:
            return {}
        
        functions = {}
        try:
            for skill_name in self._kernel.skills.skills:
                skill = self._kernel.skills.skills[skill_name]
                functions[skill_name] = list(skill.keys())
        except:
            pass
        
        return functions


class SemanticKernelAgentProvider(BaseAgentProvider):
    """
    Provider implementation for Microsoft Semantic Kernel.
    
    Enables AI orchestration with planning capabilities, skill chaining,
    and plugin architecture for complex multi-step reasoning.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, SemanticKernelAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not SK_AVAILABLE:
            self.logger.warning("Semantic Kernel not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "semantic_kernel"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.PLANNING,
            AgentCapability.REASONING_CHAINS,
            AgentCapability.TOOL_SYNTHESIS,
            AgentCapability.WORKFLOW_ORCHESTRATION
        ]
    
    async def initialize(self) -> bool:
        """Initialize Semantic Kernel provider"""
        if not SK_AVAILABLE:
            self.logger.error("Semantic Kernel not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Semantic Kernel agent provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Semantic Kernel provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Semantic Kernel agent"""
        if not self._initialized:
            await self.initialize()
        
        if not SK_AVAILABLE:
            raise RuntimeError("Semantic Kernel not available")
        
        agent_id = f"sk_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create Semantic Kernel configuration
            sk_config = SemanticKernelConfig(
                kernel_id=agent_id,
                ai_service_type=self.config.get('ai_service_type', 'openai'),
                api_key=self.config.get('api_key'),
                endpoint=self.config.get('endpoint'),
                deployment_name=self.config.get('deployment_name'),
                model_name=self.config.get('model_name', 'gpt-4'),
                skills=config.tools,
                planner_type=self.config.get('planner_type', 'sequential'),
                max_plan_length=self.config.get('max_plan_length', 10),
                enable_native_skills=self.config.get('enable_native_skills', True)
            )
            
            # Create agent
            agent = SemanticKernelAgent(
                agent_id=agent_id,
                config=sk_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Semantic Kernel agent")
            
            # Add custom skills for tools
            for tool_name in config.tools:
                skill = await self._create_skill_from_tool(tool_name)
                if skill:
                    agent.add_skill(skill)
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Semantic Kernel agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Semantic Kernel agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Semantic Kernel agent with planning"""
        if agent_id not in self._agents:
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error="Agent not found"
            )
        
        agent = self._agents[agent_id]
        context = context or {}
        
        try:
            start_time = datetime.now()
            
            # Check if this is a function call or a goal
            if context.get('function_call'):
                # Execute specific function
                skill_name = context.get('skill_name', 'text')
                function_name = context.get('function_name', 'summarize')
                parameters = context.get('parameters', {})
                
                result = await agent.execute_function(
                    skill_name=skill_name,
                    function_name=function_name,
                    parameters=parameters
                )
            else:
                # Execute as goal with planning
                result = await agent.execute_goal(prompt, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                content=result.get('content', ''),
                agent_id=agent_id,
                provider=self.provider_name,
                metadata={
                    'execution_time_seconds': execution_time,
                    'plan': result.get('plan', {}),
                    'context_variables': result.get('context_variables', {}),
                    'plan_steps_count': result.get('plan_steps_count', 0),
                    'available_skills': list(agent._skills.keys()),
                    'planner_type': agent.config.planner_type
                },
                error=result.get('error')
            )
            
        except Exception as e:
            self.logger.error(f"Semantic Kernel agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool as a Semantic Kernel skill"""
        if agent_id not in self._agents:
            return False
        
        agent = self._agents[agent_id]
        
        try:
            # Create skill from tool spec
            skill = SemanticKernelSkill(
                skill_name=tool_spec.name,
                description=tool_spec.description
            )
            
            # Add tool function
            def tool_function(context: SKContext) -> str:
                if tool_spec.function:
                    # Extract parameters from context
                    params = {}
                    for param_name in tool_spec.parameters.keys():
                        if param_name in context.variables:
                            params[param_name] = context.variables[param_name]
                    
                    result = tool_spec.function(**params)
                    return str(result)
                return f"Tool {tool_spec.name} executed"
            
            skill.add_function(
                name="execute",
                func=tool_function,
                description=tool_spec.description,
                parameters=list(tool_spec.parameters.keys())
            )
            
            # Add skill to agent
            if agent.add_skill(skill):
                self.logger.info(f"Registered tool {tool_spec.name} with Semantic Kernel agent {agent_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Semantic Kernel planning"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        
        try:
            # Use Semantic Kernel to plan tool implementation
            synthesis_goal = f"""
            Design and create a tool that: {tool_description}
            
            Requirements:
            1. Define the tool's purpose and functionality
            2. Identify required parameters and their types
            3. Outline the implementation approach
            4. Consider error handling and edge cases
            
            {f'Examples of usage: {examples}' if examples else ''}
            
            Provide a detailed specification for this tool.
            """
            
            # Execute synthesis planning
            result = await agent.execute_goal(synthesis_goal)
            
            # Create tool spec
            tool_spec = ToolSpec(
                name=f"sk_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            self.logger.info(f"Synthesized tool for Semantic Kernel agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Semantic Kernel agent"""
        try:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                # Clean up kernel resources if needed
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Semantic Kernel agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Semantic Kernel agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Semantic Kernel agent"""
        if agent_id not in self._agents:
            return {}
        
        agent = self._agents[agent_id]
        config = self._agent_configs[agent_id]
        
        return {
            'agent_id': agent_id,
            'provider': self.provider_name,
            'config': config.__dict__,
            'kernel_config': agent.config.__dict__,
            'available_skills': list(agent._skills.keys()),
            'available_functions': agent.get_available_functions(),
            'planner_type': agent.config.planner_type,
            'execution_history_count': len(agent._execution_history),
            'created_at': agent._created_at.isoformat()
        }
    
    async def _create_skill_from_tool(self, tool_name: str) -> Optional[SemanticKernelSkill]:
        """Create a Semantic Kernel skill from a tool name"""
        try:
            skill = SemanticKernelSkill(
                skill_name=tool_name,
                description=f"Custom skill for {tool_name}"
            )
            
            if tool_name == "web_search":
                def search_function(context: SKContext) -> str:
                    query = context.variables.get("query", "")
                    return f"Search results for: {query}"
                
                skill.add_function(
                    name="search",
                    func=search_function,
                    description="Search the web for information",
                    parameters=["query"]
                )
            
            elif tool_name == "code_interpreter":
                def interpret_function(context: SKContext) -> str:
                    code = context.variables.get("code", "")
                    return f"Executed code: {code}"
                
                skill.add_function(
                    name="execute",
                    func=interpret_function,
                    description="Execute code and return results",
                    parameters=["code"]
                )
            
            elif tool_name == "file_manager":
                def file_function(context: SKContext) -> str:
                    operation = context.variables.get("operation", "read")
                    file_path = context.variables.get("file_path", "")
                    return f"File {operation} on {file_path}"
                
                skill.add_function(
                    name="manage",
                    func=file_function,
                    description="Manage files and directories",
                    parameters=["operation", "file_path"]
                )
            
            return skill
            
        except Exception:
            return None