"""
LangChain Agent Provider

Integrates LangChain's agentic capabilities into the CandyLLM provider ecosystem,
enabling seamless use of LangChain agents with CandyLLM's security and tooling.
"""

import uuid
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

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
    from langchain.agents import initialize_agent, AgentType
    from langchain.agents.agent import AgentExecutor
    from langchain.schema import BaseLanguageModel
    from langchain.tools import BaseTool
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import AgentAction, AgentFinish
    from langchain.memory import ConversationBufferMemory
    from langchain.tools.base import BaseTool as LangChainBaseTool
    from langchain.prompts import ChatPromptTemplate
    from langchain.agents.format_scratchpad import format_to_openai_function_messages
    from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Mock classes for when LangChain is not available
    class AgentExecutor:
        pass
    class BaseTool:
        pass
    class BaseCallbackHandler:
        pass


class CandyLLMSecurityCallback(BaseCallbackHandler):
    """LangChain callback for integrating CandyLLM security monitoring"""
    
    def __init__(self, security_manager: 'AgentSecurityManager', agent_id: str):
        super().__init__()
        self.security_manager = security_manager
        self.agent_id = agent_id
        self.tools_used = []
        self.security_violations = []
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> Any:
        """Monitor agent actions for security compliance"""
        if not self.security_manager:
            return
        
        # Log tool usage
        self.tools_used.append(action.tool)
        
        # Validate tool access
        if not self.security_manager.validate_tool_access(self.agent_id, action.tool):
            violation = f"Unauthorized tool access: {action.tool}"
            self.security_violations.append(violation)
            raise PermissionError(violation)
        
        # Validate tool inputs
        if not self.security_manager.validate_tool_inputs(action.tool, action.tool_input):
            violation = f"Invalid tool inputs for {action.tool}: {action.tool_input}"
            self.security_violations.append(violation)
            raise ValueError(violation)
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> Any:
        """Validate final agent output"""
        if not self.security_manager:
            return
        
        # Sanitize output
        if hasattr(finish, 'return_values'):
            for key, value in finish.return_values.items():
                if isinstance(value, str):
                    finish.return_values[key] = self.security_manager.sanitize_output(value)


class LangChainToolAdapter:
    """Adapter to convert between CandyLLM ToolSpec and LangChain BaseTool"""
    
    @staticmethod
    def to_langchain_tool(tool_spec: ToolSpec) -> BaseTool:
        """Convert CandyLLM ToolSpec to LangChain BaseTool"""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available")
        
        class CandyLLMTool(BaseTool):
            name = tool_spec.name
            description = tool_spec.description
            
            def _run(self, **kwargs) -> str:
                """Run the tool synchronously"""
                if tool_spec.function:
                    try:
                        result = tool_spec.function(**kwargs)
                        return str(result)
                    except Exception as e:
                        return f"Tool execution failed: {e}"
                return f"Tool {self.name} executed with {kwargs}"
            
            async def _arun(self, **kwargs) -> str:
                """Run the tool asynchronously"""
                if tool_spec.function:
                    try:
                        if asyncio.iscoroutinefunction(tool_spec.function):
                            result = await tool_spec.function(**kwargs)
                        else:
                            result = tool_spec.function(**kwargs)
                        return str(result)
                    except Exception as e:
                        return f"Tool execution failed: {e}"
                return f"Tool {self.name} executed with {kwargs}"
        
        return CandyLLMTool()
    
    @staticmethod
    def from_langchain_tool(langchain_tool: BaseTool) -> ToolSpec:
        """Convert LangChain BaseTool to CandyLLM ToolSpec"""
        return ToolSpec(
            name=langchain_tool.name,
            description=langchain_tool.description,
            parameters={
                'type': 'object',
                'properties': getattr(langchain_tool, 'args_schema', {})
            },
            function=langchain_tool._run if hasattr(langchain_tool, '_run') else None
        )


class LangChainAgentProvider(BaseAgentProvider):
    """
    Provider implementation for LangChain agents.
    
    Integrates LangChain's powerful agent ecosystem with CandyLLM's
    security controls and unified provider interface.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, AgentExecutor] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._agent_memories: Dict[str, Any] = {}
        self._llm = None
        self._security_manager = None
        
        if not LANGCHAIN_AVAILABLE:
            self.logger.warning("LangChain not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "langchain"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.TOOL_SYNTHESIS,
            AgentCapability.MEMORY_PERSISTENCE,
            AgentCapability.REASONING_CHAINS,
            AgentCapability.WORKFLOW_ORCHESTRATION,
            AgentCapability.WEB_BROWSING,
            AgentCapability.FILE_OPERATIONS
        ]
    
    async def initialize(self) -> bool:
        """Initialize LangChain provider with LLM and security"""
        if not LANGCHAIN_AVAILABLE:
            self.logger.error("LangChain not available")
            return False
        
        try:
            # Initialize LLM based on config
            await self._initialize_llm()
            
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("LangChain agent provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain provider: {e}")
            return False
    
    async def _initialize_llm(self):
        """Initialize the LangChain LLM based on config"""
        llm_type = self.config.get('llm_type', 'openai')
        
        if llm_type == 'openai':
            from langchain.llms import OpenAI
            from langchain.chat_models import ChatOpenAI
            
            model_name = self.config.get('model', 'gpt-3.5-turbo')
            if 'gpt-3.5' in model_name or 'gpt-4' in model_name:
                self._llm = ChatOpenAI(
                    model_name=model_name,
                    temperature=self.config.get('temperature', 0.7),
                    openai_api_key=self.config.get('openai_api_key')
                )
            else:
                self._llm = OpenAI(
                    model_name=model_name,
                    temperature=self.config.get('temperature', 0.7),
                    openai_api_key=self.config.get('openai_api_key')
                )
        
        elif llm_type == 'anthropic':
            from langchain.llms import Anthropic
            self._llm = Anthropic(
                model=self.config.get('model', 'claude-3-sonnet-20240229'),
                anthropic_api_key=self.config.get('anthropic_api_key')
            )
        
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new LangChain agent instance"""
        if not self._initialized:
            await self.initialize()
        
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain not available")
        
        agent_id = f"langchain_{uuid.uuid4().hex[:8]}"
        
        try:
            # Set up memory based on config
            memory = None
            if AgentCapability.MEMORY_PERSISTENCE in config.capabilities:
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
                self._agent_memories[agent_id] = memory
            
            # Convert tools
            langchain_tools = []
            for tool_name in config.tools:
                # This would convert from tool registry or create built-in tools
                tool = await self._get_builtin_tool(tool_name)
                if tool:
                    langchain_tools.append(tool)
            
            # Set up security callback
            security_callback = CandyLLMSecurityCallback(self._security_manager, agent_id)
            
            # Create agent based on type preference
            agent_type = self._determine_agent_type(config)
            
            if agent_type == AgentType.OPENAI_FUNCTIONS:
                # Use OpenAI Functions agent for better tool use
                agent = initialize_agent(
                    tools=langchain_tools,
                    llm=self._llm,
                    agent=AgentType.OPENAI_FUNCTIONS,
                    memory=memory,
                    callbacks=[security_callback],
                    verbose=self.config.get('verbose', False),
                    handle_parsing_errors=True,
                    max_iterations=self.config.get('max_iterations', 10)
                )
            else:
                # Use standard agent
                agent = initialize_agent(
                    tools=langchain_tools,
                    llm=self._llm,
                    agent=agent_type,
                    memory=memory,
                    callbacks=[security_callback],
                    verbose=self.config.get('verbose', False),
                    handle_parsing_errors=True
                )
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created LangChain agent {agent_id} with {len(langchain_tools)} tools")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create LangChain agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a LangChain agent with the given prompt"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        context = context or {}
        
        try:
            # Prepare input
            agent_input = prompt
            if context:
                # Inject context into prompt
                agent_input = f"Context: {context}\n\nTask: {prompt}"
            
            # Execute agent
            start_time = datetime.now()
            
            if asyncio.iscoroutinefunction(agent.run):
                result = await agent.run(agent_input)
            else:
                # Run in thread pool for sync agents
                result = await asyncio.get_event_loop().run_in_executor(
                    None, agent.run, agent_input
                )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract callback data
            security_callback = None
            for callback in agent.callbacks:
                if isinstance(callback, CandyLLMSecurityCallback):
                    security_callback = callback
                    break
            
            return AgentResponse(
                content=str(result),
                agent_id=agent_id,
                provider=self.provider_name,
                tools_used=security_callback.tools_used if security_callback else [],
                security_violations=security_callback.security_violations if security_callback else [],
                metadata={
                    'execution_time_seconds': execution_time,
                    'agent_type': str(agent.agent.llm_chain.prompt),
                    'memory_messages': len(agent.memory.chat_memory.messages) if agent.memory else 0
                }
            )
            
        except Exception as e:
            self.logger.error(f"LangChain agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with a LangChain agent"""
        if agent_id not in self._agents:
            return False
        
        try:
            # Convert to LangChain tool
            langchain_tool = LangChainToolAdapter.to_langchain_tool(tool_spec)
            
            # Add to agent tools
            agent = self._agents[agent_id]
            agent.tools.append(langchain_tool)
            
            # Security validation
            if self._security_manager:
                self._security_manager.register_tool(agent_id, tool_spec)
            
            self.logger.info(f"Registered tool {tool_spec.name} with LangChain agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using LangChain's capabilities"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Use LangChain agent to synthesize tool code
            synthesis_prompt = f"""
            Create a Python function that implements the following tool:
            
            Description: {tool_description}
            
            {f'Examples: {examples}' if examples else ''}
            
            The function should:
            1. Have clear parameter types and documentation
            2. Handle errors gracefully
            3. Return a meaningful result
            4. Be secure and not execute arbitrary code
            
            Provide only the function definition.
            """
            
            agent = self._agents[agent_id]
            code_response = await self.execute_agent(agent_id, synthesis_prompt)
            
            # Extract function code (simplified - would need better parsing)
            function_code = code_response.content
            
            # Create tool spec
            tool_spec = ToolSpec(
                name=f"synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            # TODO: Actually compile and validate the synthesized function
            self.logger.info(f"Synthesized tool for agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a LangChain agent"""
        if agent_id not in self._agents:
            return False
        
        try:
            # Clear memory if exists
            if agent_id in self._agent_memories:
                del self._agent_memories[agent_id]
            
            # Remove agent and config
            del self._agents[agent_id]
            del self._agent_configs[agent_id]
            
            # Cleanup security manager
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed LangChain agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active LangChain agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a LangChain agent"""
        if agent_id not in self._agents:
            return {}
        
        agent = self._agents[agent_id]
        config = self._agent_configs[agent_id]
        
        return {
            'agent_id': agent_id,
            'provider': self.provider_name,
            'config': config.__dict__,
            'tools': [tool.name for tool in agent.tools],
            'memory_messages': len(agent.memory.chat_memory.messages) if agent.memory else 0,
            'agent_type': str(type(agent.agent).__name__)
        }
    
    # Helper methods
    
    def _determine_agent_type(self, config: AgentConfig) -> AgentType:
        """Determine best LangChain agent type based on config"""
        if AgentCapability.TOOL_SYNTHESIS in config.capabilities:
            return AgentType.OPENAI_FUNCTIONS
        elif AgentCapability.REASONING_CHAINS in config.capabilities:
            return AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
        else:
            return AgentType.ZERO_SHOT_REACT_DESCRIPTION
    
    async def _get_builtin_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a built-in LangChain tool by name"""
        builtin_tools = {
            'serpapi': self._get_serpapi_tool,
            'python_repl': self._get_python_repl_tool,
            'shell': self._get_shell_tool,
            'file_management': self._get_file_management_tool
        }
        
        if tool_name in builtin_tools:
            return await builtin_tools[tool_name]()
        
        return None
    
    async def _get_serpapi_tool(self) -> Optional[BaseTool]:
        """Get SerpAPI search tool"""
        try:
            from langchain.utilities import SerpAPIWrapper
            from langchain.tools import Tool
            
            search = SerpAPIWrapper(serpapi_api_key=self.config.get('serpapi_key'))
            return Tool(
                name="Search",
                description="Search the internet for current information",
                func=search.run
            )
        except ImportError:
            return None
    
    async def _get_python_repl_tool(self) -> Optional[BaseTool]:
        """Get Python REPL tool"""
        try:
            from langchain.tools import PythonREPLTool
            return PythonREPLTool()
        except ImportError:
            return None
    
    async def _get_shell_tool(self) -> Optional[BaseTool]:
        """Get shell command tool"""
        try:
            from langchain.tools import ShellTool
            return ShellTool()
        except ImportError:
            return None
    
    async def _get_file_management_tool(self) -> Optional[BaseTool]:
        """Get file management tools"""
        try:
            from langchain.tools.file_management import (
                ReadFileTool,
                WriteFileTool,
                ListDirectoryTool
            )
            
            # Return the read file tool as primary - would normally return a toolkit
            return ReadFileTool()
        except ImportError:
            return None