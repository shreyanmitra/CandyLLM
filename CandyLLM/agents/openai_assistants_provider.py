"""
OpenAI Assistants Agent Provider

Integrates OpenAI's Assistants API into the CandyLLM provider ecosystem,
providing access to OpenAI's hosted agent capabilities with advanced tools.
"""

import uuid
import asyncio
import json
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
    import openai
    from openai import OpenAI, AsyncOpenAI
    from openai.types.beta import Assistant, Thread
    from openai.types.beta.threads import Run, Message
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Mock classes for when OpenAI is not available
    class Assistant:
        pass
    class Thread:
        pass
    class Run:
        pass
    class Message:
        pass


class OpenAIAssistantWrapper:
    """Wrapper for OpenAI Assistant with CandyLLM integration"""
    
    def __init__(self, assistant_id: str, thread_id: str, 
                 client: AsyncOpenAI, config: AgentConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.client = client
        self.config = config
        self.security_manager = security_manager
        self._created_at = datetime.now()
        self._message_count = 0
        
    async def execute(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the assistant with a prompt"""
        try:
            # Add message to thread
            message = await self.client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role="user",
                content=prompt
            )
            
            # Create run
            run = await self.client.beta.threads.runs.create(
                thread_id=self.thread_id,
                assistant_id=self.assistant_id,
                instructions=context.get('additional_instructions') if context else None
            )
            
            # Wait for completion
            while run.status in ['queued', 'in_progress', 'cancelling']:
                await asyncio.sleep(1)
                run = await self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread_id,
                    run_id=run.id
                )
            
            if run.status == 'completed':
                # Get messages
                messages = await self.client.beta.threads.messages.list(
                    thread_id=self.thread_id,
                    order="desc",
                    limit=1
                )
                
                if messages.data:
                    content = messages.data[0].content[0].text.value
                    self._message_count += 1
                    
                    # Security validation
                    if self.security_manager:
                        content = self.security_manager.sanitize_output(content)
                    
                    return {
                        'content': content,
                        'status': 'completed',
                        'run_id': run.id,
                        'tools_used': self._extract_tools_used(run),
                        'message_count': self._message_count
                    }
            
            elif run.status == 'requires_action':
                # Handle tool calls
                return await self._handle_tool_calls(run)
            
            else:
                # Handle errors
                error_msg = f"Run failed with status: {run.status}"
                if run.last_error:
                    error_msg += f" - {run.last_error.message}"
                
                return {
                    'content': '',
                    'status': 'error',
                    'error': error_msg
                }
                
        except Exception as e:
            return {
                'content': '',
                'status': 'error', 
                'error': str(e)
            }
    
    async def _handle_tool_calls(self, run: Run) -> Dict[str, Any]:
        """Handle tool calls from the assistant"""
        if not run.required_action or not run.required_action.submit_tool_outputs:
            return {'content': '', 'status': 'error', 'error': 'No tool calls found'}
        
        tool_outputs = []
        tools_used = []
        
        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            tools_used.append(function_name)
            
            # Security validation
            if self.security_manager:
                if not self.security_manager.validate_tool_access(self.assistant_id, function_name):
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": "Tool access denied by security policy"
                    })
                    continue
                
                if not self.security_manager.validate_tool_inputs(function_name, function_args):
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": "Tool inputs rejected by security policy"
                    })
                    continue
            
            # Execute tool (simplified - would integrate with actual tool registry)
            try:
                output = await self._execute_tool(function_name, function_args)
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": str(output)
                })
            except Exception as e:
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": f"Tool execution failed: {str(e)}"
                })
        
        # Submit tool outputs
        run = await self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread_id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )
        
        # Wait for completion again
        while run.status in ['queued', 'in_progress']:
            await asyncio.sleep(1)
            run = await self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run.id
            )
        
        if run.status == 'completed':
            messages = await self.client.beta.threads.messages.list(
                thread_id=self.thread_id,
                order="desc",
                limit=1
            )
            
            content = messages.data[0].content[0].text.value if messages.data else ""
            
            return {
                'content': content,
                'status': 'completed',
                'run_id': run.id,
                'tools_used': tools_used,
                'message_count': self._message_count
            }
        
        return {
            'content': '',
            'status': 'error',
            'error': f"Tool execution run failed with status: {run.status}"
        }
    
    async def _execute_tool(self, function_name: str, function_args: Dict[str, Any]) -> str:
        """Execute a tool function (placeholder)"""
        # This would integrate with CandyLLM's tool registry
        return f"Executed {function_name} with args {function_args}"
    
    def _extract_tools_used(self, run: Run) -> List[str]:
        """Extract list of tools used during run"""
        # This would parse the run details to extract tool usage
        return []


class OpenAIAssistantsProvider(BaseAgentProvider):
    """
    Provider implementation for OpenAI Assistants API.
    
    Integrates OpenAI's hosted agent capabilities with CandyLLM's
    security controls and unified provider interface.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._assistants: Dict[str, OpenAIAssistantWrapper] = {}
        self._assistant_configs: Dict[str, AgentConfig] = {}
        self._client = None
        self._security_manager = None
        
        if not OPENAI_AVAILABLE:
            self.logger.warning("OpenAI not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "openai_assistants"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.CODE_EXECUTION,
            AgentCapability.FILE_OPERATIONS, 
            AgentCapability.WEB_BROWSING,
            AgentCapability.MEMORY_PERSISTENCE,
            AgentCapability.TOOL_SYNTHESIS,
            AgentCapability.REASONING_CHAINS
        ]
    
    async def initialize(self) -> bool:
        """Initialize OpenAI Assistants provider"""
        if not OPENAI_AVAILABLE:
            self.logger.error("OpenAI library not available")
            return False
        
        try:
            # Initialize OpenAI client
            api_key = self.config.get('openai_api_key')
            if not api_key:
                raise ValueError("OpenAI API key required")
            
            self._client = AsyncOpenAI(api_key=api_key)
            
            # Test connection
            await self._client.models.list()
            
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("OpenAI Assistants provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI Assistants provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new OpenAI Assistant"""
        if not self._initialized:
            await self.initialize()
        
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI not available")
        
        agent_id = f"openai_assistant_{uuid.uuid4().hex[:8]}"
        
        try:
            # Determine tools based on config
            tools = await self._prepare_tools(config)
            
            # Create assistant
            assistant = await self._client.beta.assistants.create(
                name=config.name,
                instructions=config.system_prompt or f"You are {config.name}. {config.description}",
                model=config.model or self.config.get('default_model', 'gpt-4-1106-preview'),
                tools=tools,
                temperature=config.temperature,
                file_ids=[]  # Would handle file uploads here
            )
            
            # Create thread for conversations
            thread = await self._client.beta.threads.create()
            
            # Create wrapper
            assistant_wrapper = OpenAIAssistantWrapper(
                assistant_id=assistant.id,
                thread_id=thread.id,
                client=self._client,
                config=config,
                security_manager=self._security_manager
            )
            
            self._assistants[agent_id] = assistant_wrapper
            self._assistant_configs[agent_id] = config
            
            self.logger.info(f"Created OpenAI Assistant {agent_id} with {len(tools)} tools")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create OpenAI Assistant: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute an OpenAI Assistant with the given prompt"""
        if agent_id not in self._assistants:
            raise ValueError(f"Assistant {agent_id} not found")
        
        assistant = self._assistants[agent_id]
        context = context or {}
        
        try:
            start_time = datetime.now()
            result = await assistant.execute(prompt, context)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result['status'] == 'completed':
                return AgentResponse(
                    content=result['content'],
                    agent_id=agent_id,
                    provider=self.provider_name,
                    tools_used=result.get('tools_used', []),
                    metadata={
                        'execution_time_seconds': execution_time,
                        'run_id': result.get('run_id'),
                        'message_count': result.get('message_count', 0),
                        'openai_model': assistant.config.model
                    }
                )
            else:
                return AgentResponse(
                    content=result.get('content', ''),
                    agent_id=agent_id,
                    provider=self.provider_name,
                    error=result.get('error', 'Unknown error'),
                    metadata={'execution_time_seconds': execution_time}
                )
                
        except Exception as e:
            self.logger.error(f"OpenAI Assistant execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with an OpenAI Assistant"""
        if agent_id not in self._assistants:
            return False
        
        try:
            assistant_wrapper = self._assistants[agent_id]
            
            # Create OpenAI function definition
            function_def = {
                "type": "function",
                "function": {
                    "name": tool_spec.name,
                    "description": tool_spec.description,
                    "parameters": tool_spec.parameters
                }
            }
            
            # Update assistant with new tool
            assistant = await self._client.beta.assistants.retrieve(assistant_wrapper.assistant_id)
            
            # Add new tool to existing tools
            updated_tools = list(assistant.tools) + [function_def]
            
            await self._client.beta.assistants.update(
                assistant_id=assistant_wrapper.assistant_id,
                tools=updated_tools
            )
            
            # Security validation
            if self._security_manager:
                self._security_manager.register_tool(agent_id, tool_spec)
            
            self.logger.info(f"Registered tool {tool_spec.name} with OpenAI Assistant {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using OpenAI Assistant's code interpreter"""
        if agent_id not in self._assistants:
            raise ValueError(f"Assistant {agent_id} not found")
        
        try:
            # Create a synthesis prompt for the assistant
            synthesis_prompt = f"""
            Create a Python function that implements the following tool:
            
            Description: {tool_description}
            
            {f'Examples: {examples}' if examples else ''}
            
            Requirements:
            1. Function should have clear parameter types and documentation
            2. Include proper error handling
            3. Return meaningful results
            4. Be secure and validate inputs
            5. Follow Python best practices
            
            Please provide:
            1. The function definition
            2. Parameter schema in JSON format
            3. Usage examples
            """
            
            # Execute synthesis
            assistant = self._assistants[agent_id]
            result = await assistant.execute(synthesis_prompt)
            
            if result['status'] == 'completed':
                # Parse the response to extract function details
                # This is simplified - real implementation would parse the code
                
                tool_spec = ToolSpec(
                    name=f"synthesized_{uuid.uuid4().hex[:8]}",
                    description=tool_description,
                    parameters={
                        'type': 'object',
                        'properties': {
                            'input': {'type': 'string', 'description': 'Tool input'}
                        },
                        'required': ['input']
                    },
                    security_policy={'risk_level': 'medium', 'requires_approval': True}
                )
                
                self.logger.info(f"Synthesized tool for OpenAI Assistant {agent_id}")
                return tool_spec
            else:
                raise RuntimeError(f"Tool synthesis failed: {result.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy an OpenAI Assistant"""
        if agent_id not in self._assistants:
            return False
        
        try:
            assistant_wrapper = self._assistants[agent_id]
            
            # Delete assistant and thread
            await self._client.beta.assistants.delete(assistant_wrapper.assistant_id)
            await self._client.beta.threads.delete(assistant_wrapper.thread_id)
            
            # Remove from local storage
            del self._assistants[agent_id]
            del self._assistant_configs[agent_id]
            
            # Cleanup security manager
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed OpenAI Assistant {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy assistant: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active OpenAI Assistant IDs"""
        return list(self._assistants.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about an OpenAI Assistant"""
        if agent_id not in self._assistants:
            return {}
        
        assistant_wrapper = self._assistants[agent_id]
        config = self._assistant_configs[agent_id]
        
        try:
            # Get latest assistant info from OpenAI
            assistant = await self._client.beta.assistants.retrieve(assistant_wrapper.assistant_id)
            
            return {
                'agent_id': agent_id,
                'provider': self.provider_name,
                'config': config.__dict__,
                'openai_assistant_id': assistant.id,
                'openai_thread_id': assistant_wrapper.thread_id,
                'model': assistant.model,
                'tools': [tool.get('function', {}).get('name', str(tool)) for tool in assistant.tools],
                'created_at': assistant_wrapper._created_at.isoformat(),
                'message_count': assistant_wrapper._message_count
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get assistant info: {e}")
            return {
                'agent_id': agent_id,
                'provider': self.provider_name,
                'error': str(e)
            }
    
    # Helper methods
    
    async def _prepare_tools(self, config: AgentConfig) -> List[Dict[str, Any]]:
        """Prepare tools for OpenAI Assistant creation"""
        tools = []
        
        # Add capability-based tools
        if AgentCapability.CODE_EXECUTION in config.capabilities:
            tools.append({"type": "code_interpreter"})
        
        if AgentCapability.FILE_OPERATIONS in config.capabilities:
            tools.append({"type": "retrieval"})
        
        # Add custom tools from config
        for tool_name in config.tools:
            custom_tool = await self._get_custom_tool_definition(tool_name)
            if custom_tool:
                tools.append(custom_tool)
        
        return tools
    
    async def _get_custom_tool_definition(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get custom tool definition for OpenAI Assistant"""
        # This would integrate with CandyLLM's tool registry
        # For now, return some common tool examples
        
        common_tools = {
            'web_search': {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "description": "Maximum results", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            },
            'file_read': {
                "type": "function", 
                "function": {
                    "name": "file_read",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to file"}
                        },
                        "required": ["file_path"]
                    }
                }
            }
        }
        
        return common_tools.get(tool_name)