"""
Groq Agent Provider

Integrates Groq's high-speed inference platform with agent capabilities,
providing ultra-fast language model execution with specialized hardware acceleration.
"""

import uuid
import asyncio
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass, field

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
    import groq
    from groq import Groq, AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    # Mock classes for when Groq is not available
    groq = None
    Groq = None
    AsyncGroq = None


@dataclass
class GroqConfig:
    """Configuration for Groq agent"""
    api_key: str = ""
    model: str = "mixtral-8x7b-32768"  # High-speed models available on Groq
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False
    seed: Optional[int] = None
    timeout: float = 30.0
    enable_system_prompts: bool = True
    enable_function_calling: bool = True
    enable_parallel_execution: bool = True
    max_parallel_requests: int = 10


@dataclass
class GroqMessage:
    """Message structure for Groq chat"""
    role: str  # "system", "user", "assistant", "function"
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GroqFunction:
    """Function definition for Groq function calling"""
    name: str
    description: str
    parameters: Dict[str, Any]
    implementation: Optional[callable] = None


class GroqAgent:
    """Groq agent with high-speed inference capabilities"""
    
    def __init__(self, agent_id: str, config: GroqConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._client = None
        self._async_client = None
        self._conversations = {}
        self._functions = {}
        self._usage_stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_execution_time': 0.0,
            'average_tokens_per_second': 0.0
        }
        self._interaction_history = []
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the Groq agent"""
        if not GROQ_AVAILABLE:
            return False
        
        try:
            if not self.config.api_key:
                return False
            
            # Initialize clients
            self._client = Groq(api_key=self.config.api_key)
            self._async_client = AsyncGroq(api_key=self.config.api_key)
            
            # Test connection
            await self._test_connection()
            
            return True
            
        except Exception as e:
            return False
    
    async def _test_connection(self):
        """Test API connection with a simple request"""
        try:
            # Test with minimal request
            response = await self._async_client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model=self.config.model,
                max_tokens=1
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Groq API test failed: {e}")
    
    async def chat_completion(self, messages: List[GroqMessage], 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create chat completion using Groq"""
        try:
            context = context or {}
            start_time = datetime.now()
            
            # Convert messages to Groq format
            groq_messages = []
            for msg in messages:
                groq_msg = {
                    "role": msg.role,
                    "content": msg.content
                }
                if msg.name:
                    groq_msg["name"] = msg.name
                if msg.function_call:
                    groq_msg["function_call"] = msg.function_call
                groq_messages.append(groq_msg)
            
            # Prepare completion parameters
            completion_params = {
                'messages': groq_messages,
                'model': context.get('model', self.config.model),
                'max_tokens': context.get('max_tokens', self.config.max_tokens),
                'temperature': context.get('temperature', self.config.temperature),
                'top_p': context.get('top_p', self.config.top_p),
                'frequency_penalty': context.get('frequency_penalty', self.config.frequency_penalty),
                'presence_penalty': context.get('presence_penalty', self.config.presence_penalty),
                'stream': context.get('stream', self.config.stream),
                'stop': context.get('stop_sequences', self.config.stop_sequences) or None
            }
            
            # Add function calling if enabled and functions available
            if self.config.enable_function_calling and self._functions:
                completion_params['functions'] = [
                    {
                        'name': func.name,
                        'description': func.description,
                        'parameters': func.parameters
                    }
                    for func in self._functions.values()
                ]
                completion_params['function_call'] = context.get('function_call', 'auto')
            
            # Add seed if provided
            if self.config.seed is not None:
                completion_params['seed'] = self.config.seed
            
            # Remove None values
            completion_params = {k: v for k, v in completion_params.items() if v is not None}
            
            # Create completion
            if completion_params.get('stream'):
                return await self._stream_completion(**completion_params)
            else:
                response = await self._async_client.chat.completions.create(**completion_params)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract response data
            choice = response.choices[0]
            message = choice.message
            
            # Update usage statistics
            if hasattr(response, 'usage'):
                usage = response.usage
                self._usage_stats['total_requests'] += 1
                self._usage_stats['total_tokens'] += usage.total_tokens
                self._usage_stats['total_execution_time'] += execution_time
                
                # Calculate tokens per second
                if execution_time > 0:
                    tokens_per_second = usage.total_tokens / execution_time
                    self._usage_stats['average_tokens_per_second'] = (
                        (self._usage_stats['average_tokens_per_second'] * (self._usage_stats['total_requests'] - 1) + 
                         tokens_per_second) / self._usage_stats['total_requests']
                    )
            
            result = {
                'response': message.content,
                'role': message.role,
                'finish_reason': choice.finish_reason,
                'execution_time': execution_time,
                'model_used': response.model,
                'usage': response.usage.__dict__ if hasattr(response, 'usage') else {},
                'function_call': message.function_call.__dict__ if hasattr(message, 'function_call') and message.function_call else None,
                'tokens_per_second': (response.usage.total_tokens / execution_time) if hasattr(response, 'usage') and execution_time > 0 else 0
            }
            
            # Handle function calls
            if result['function_call'] and self.config.enable_function_calling:
                function_result = await self._execute_function(result['function_call'])
                result['function_result'] = function_result
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _stream_completion(self, **completion_params) -> Dict[str, Any]:
        """Handle streaming completions"""
        try:
            full_content = ""
            function_call = None
            model_used = None
            
            stream = await self._async_client.chat.completions.create(**completion_params)
            
            async for chunk in stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    if hasattr(delta, 'content') and delta.content:
                        full_content += delta.content
                    
                    if hasattr(delta, 'function_call') and delta.function_call:
                        if function_call is None:
                            function_call = {}
                        
                        if hasattr(delta.function_call, 'name') and delta.function_call.name:
                            function_call['name'] = delta.function_call.name
                        
                        if hasattr(delta.function_call, 'arguments') and delta.function_call.arguments:
                            if 'arguments' not in function_call:
                                function_call['arguments'] = ""
                            function_call['arguments'] += delta.function_call.arguments
                
                if hasattr(chunk, 'model'):
                    model_used = chunk.model
            
            result = {
                'response': full_content,
                'stream': True,
                'model_used': model_used,
                'function_call': function_call
            }
            
            # Handle function calls
            if function_call and self.config.enable_function_calling:
                function_result = await self._execute_function(function_call)
                result['function_result'] = function_result
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _execute_function(self, function_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function call"""
        try:
            function_name = function_call.get('name')
            if function_name not in self._functions:
                return {'error': f'Function {function_name} not found'}
            
            func = self._functions[function_name]
            if not func.implementation:
                return {'error': f'Function {function_name} has no implementation'}
            
            # Parse arguments
            arguments = function_call.get('arguments', '{}')
            if isinstance(arguments, str):
                args = json.loads(arguments)
            else:
                args = arguments
            
            # Execute function
            if asyncio.iscoroutinefunction(func.implementation):
                result = await func.implementation(**args)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: func.implementation(**args)
                )
            
            return {
                'function_name': function_name,
                'arguments': args,
                'result': result
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def generate_text(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate text using Groq (wrapper around chat completion)"""
        try:
            messages = [GroqMessage(role="user", content=prompt)]
            return await self.chat_completion(messages, context)
            
        except Exception as e:
            return {'error': str(e)}
    
    async def continue_conversation(self, conversation_id: str, message: str,
                                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Continue an existing conversation"""
        try:
            # Get or create conversation
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = []
            
            conversation = self._conversations[conversation_id]
            
            # Add user message
            user_msg = GroqMessage(role="user", content=message)
            conversation.append(user_msg)
            
            # Generate response
            result = await self.chat_completion(conversation, context)
            
            # Add assistant response to conversation
            if 'response' in result and not result.get('error'):
                assistant_msg = GroqMessage(role="assistant", content=result['response'])
                conversation.append(assistant_msg)
                
                # Handle function calls
                if result.get('function_call'):
                    function_msg = GroqMessage(
                        role="function",
                        content=json.dumps(result.get('function_result', {})),
                        name=result['function_call'].get('name')
                    )
                    conversation.append(function_msg)
            
            result['conversation_id'] = conversation_id
            result['conversation_length'] = len(conversation)
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    async def add_function(self, function: GroqFunction) -> bool:
        """Add a function for function calling"""
        try:
            self._functions[function.name] = function
            return True
        except Exception as e:
            return False
    
    async def parallel_execute(self, prompts: List[str], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute multiple prompts in parallel for high throughput"""
        if not self.config.enable_parallel_execution:
            # Execute sequentially
            results = []
            for prompt in prompts:
                result = await self.generate_text(prompt, context)
                results.append(result)
            return results
        
        try:
            # Limit concurrent requests
            semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
            
            async def execute_with_semaphore(prompt):
                async with semaphore:
                    return await self.generate_text(prompt, context)
            
            # Execute all prompts in parallel
            tasks = [execute_with_semaphore(prompt) for prompt in prompts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to error results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        'error': str(result),
                        'prompt_index': i
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            return [{'error': str(e)} for _ in prompts]
    
    def create_system_prompt(self, system_content: str) -> GroqMessage:
        """Create a system prompt message"""
        return GroqMessage(role="system", content=system_content)
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history"""
        if conversation_id not in self._conversations:
            return []
        
        return [
            {
                'role': msg.role,
                'content': msg.content,
                'name': msg.name,
                'function_call': msg.function_call,
                'timestamp': msg.timestamp.isoformat()
            }
            for msg in self._conversations[conversation_id]
        ]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self._usage_stats.copy()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': self.config.__dict__,
            'active_conversations': len(self._conversations),
            'registered_functions': len(self._functions),
            'usage_stats': self._usage_stats,
            'total_interactions': len(self._interaction_history),
            'groq_available': GROQ_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class GroqAgentProvider(BaseAgentProvider):
    """
    Provider implementation for Groq.
    
    Enables ultra-fast language model inference with specialized hardware acceleration,
    supporting high-throughput parallel execution and function calling.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, GroqAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not GROQ_AVAILABLE:
            self.logger.warning("Groq not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "groq"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.HIGH_SPEED_INFERENCE,
            AgentCapability.PARALLEL_EXECUTION,
            AgentCapability.FUNCTION_CALLING,
            AgentCapability.STREAMING,
            AgentCapability.SYSTEM_PROMPTS
        ]
    
    async def initialize(self) -> bool:
        """Initialize Groq provider"""
        if not GROQ_AVAILABLE:
            self.logger.error("Groq not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Groq provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Groq provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Groq agent"""
        if not self._initialized:
            await self.initialize()
        
        if not GROQ_AVAILABLE:
            raise RuntimeError("Groq not available")
        
        agent_id = f"groq_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create Groq configuration
            groq_config = GroqConfig(
                api_key=self.config.get('api_key', ''),
                model=self.config.get('model', 'mixtral-8x7b-32768'),
                max_tokens=self.config.get('max_tokens', 1024),
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                frequency_penalty=self.config.get('frequency_penalty', 0.0),
                presence_penalty=self.config.get('presence_penalty', 0.0),
                stream=self.config.get('stream', False),
                seed=self.config.get('seed'),
                timeout=self.config.get('timeout', 30.0),
                enable_system_prompts=self.config.get('enable_system_prompts', True),
                enable_function_calling=self.config.get('enable_function_calling', True),
                enable_parallel_execution=self.config.get('enable_parallel_execution', True),
                max_parallel_requests=self.config.get('max_parallel_requests', 10)
            )
            
            # Create agent
            agent = GroqAgent(
                agent_id=agent_id,
                config=groq_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Groq agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Groq agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Groq agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Groq agent"""
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
            
            # Check for conversation mode
            conversation_id = context.get('conversation_id')
            if conversation_id:
                result = await agent.continue_conversation(conversation_id, prompt, context)
            else:
                result = await agent.generate_text(prompt, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                content=result.get('response', ''),
                agent_id=agent_id,
                provider=self.provider_name,
                metadata={
                    'execution_time_seconds': execution_time,
                    'model_used': result.get('model_used'),
                    'finish_reason': result.get('finish_reason'),
                    'usage': result.get('usage', {}),
                    'tokens_per_second': result.get('tokens_per_second', 0),
                    'function_call': result.get('function_call'),
                    'function_result': result.get('function_result'),
                    'conversation_id': result.get('conversation_id'),
                    'conversation_length': result.get('conversation_length'),
                    'stream': result.get('stream', False),
                    'usage_stats': agent.get_usage_stats(),
                    'agent_info': agent.get_agent_info()
                },
                error=result.get('error')
            )
            
        except Exception as e:
            self.logger.error(f"Groq agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Groq agent as a function"""
        if agent_id not in self._agents:
            return False
        
        agent = self._agents[agent_id]
        
        try:
            # Create Groq function from tool spec
            groq_function = GroqFunction(
                name=tool_spec.name,
                description=tool_spec.description,
                parameters=tool_spec.parameters,
                implementation=tool_spec.function
            )
            
            return await agent.add_function(groq_function)
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Groq agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Use Groq agent to generate tool specification
            synthesis_prompt = f"""
            Create a detailed tool specification for: {tool_description}
            
            Requirements:
            1. Define the tool's purpose and functionality
            2. Specify input parameters with types and validation
            3. Describe expected outputs and return formats
            4. Include error handling and edge cases
            5. Provide implementation guidelines for Python
            6. Consider integration with high-speed inference requirements
            7. Optimize for parallel execution where applicable
            
            {f'Examples: {examples}' if examples else ''}
            
            Generate a comprehensive tool specification optimized for Groq's high-speed inference.
            """
            
            agent = self._agents[agent_id]
            result = await agent.generate_text(synthesis_prompt)
            
            # Create tool spec
            tool_spec = ToolSpec(
                name=f"groq_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            self.logger.info(f"Synthesized tool for Groq agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def create_conversation(self, agent_id: str) -> str:
        """Create a conversation session with Groq agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        agent = self._agents[agent_id]
        agent._conversations[conversation_id] = []
        return conversation_id
    
    async def batch_execute(self, agent_id: str, prompts: List[str], context: Dict[str, Any] = None) -> List[AgentResponse]:
        """Execute multiple prompts in parallel for high throughput"""
        if agent_id not in self._agents:
            return [AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error="Agent not found"
            ) for _ in prompts]
        
        agent = self._agents[agent_id]
        
        try:
            start_time = datetime.now()
            results = await agent.parallel_execute(prompts, context)
            total_execution_time = (datetime.now() - start_time).total_seconds()
            
            responses = []
            for i, result in enumerate(results):
                responses.append(AgentResponse(
                    content=result.get('response', ''),
                    agent_id=agent_id,
                    provider=self.provider_name,
                    metadata={
                        'batch_index': i,
                        'batch_size': len(prompts),
                        'total_batch_time': total_execution_time,
                        'individual_execution_time': result.get('execution_time'),
                        'tokens_per_second': result.get('tokens_per_second', 0),
                        'model_used': result.get('model_used'),
                        'usage': result.get('usage', {})
                    },
                    error=result.get('error')
                ))
            
            return responses
            
        except Exception as e:
            return [AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            ) for _ in prompts]
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Groq agent"""
        try:
            if agent_id in self._agents:
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Groq agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Groq agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Groq agent"""
        if agent_id not in self._agents:
            return {}
        
        agent = self._agents[agent_id]
        config = self._agent_configs[agent_id]
        
        return {
            'agent_id': agent_id,
            'provider': self.provider_name,
            'config': config.__dict__,
            'agent_info': agent.get_agent_info()
        }