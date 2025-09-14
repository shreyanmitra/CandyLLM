"""
Together AI Agent Provider

Integrates Together AI's distributed inference platform with agent capabilities,
providing access to multiple open-source models with fine-tuning and scaling.
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
    import together
    from together import Together, AsyncTogether
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    # Mock classes for when Together is not available
    together = None
    Together = None
    AsyncTogether = None


@dataclass
class TogetherConfig:
    """Configuration for Together AI agent"""
    api_key: str = ""
    model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Default open source model
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False
    safety_model: str = ""
    enable_function_calling: bool = True
    enable_fine_tuning: bool = True
    enable_embeddings: bool = True
    base_url: str = "https://api.together.xyz"
    timeout: float = 60.0
    max_retries: int = 3
    request_timeout: float = 300.0


@dataclass
class TogetherMessage:
    """Message structure for Together AI chat"""
    role: str  # "system", "user", "assistant", "function"
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TogetherFunction:
    """Function definition for Together AI function calling"""
    name: str
    description: str
    parameters: Dict[str, Any]
    implementation: Optional[callable] = None


class TogetherAgent:
    """Together AI agent with distributed inference capabilities"""
    
    def __init__(self, agent_id: str, config: TogetherConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._client = None
        self._async_client = None
        self._conversations = {}
        self._functions = {}
        self._fine_tuned_models = {}
        self._usage_stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'fine_tuning_jobs': 0,
            'function_calls': 0,
            'models_used': set()
        }
        self._interaction_history = []
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the Together AI agent"""
        if not TOGETHER_AVAILABLE:
            return False
        
        try:
            if not self.config.api_key:
                return False
            
            # Initialize clients
            self._client = Together(api_key=self.config.api_key)
            self._async_client = AsyncTogether(api_key=self.config.api_key)
            
            # Test connection
            await self._test_connection()
            
            return True
            
        except Exception as e:
            return False
    
    async def _test_connection(self):
        """Test API connection"""
        try:
            # Test with a simple completion
            response = await self._async_client.completions.create(
                prompt="Hello",
                model=self.config.model,
                max_tokens=1
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Together AI API test failed: {e}")
    
    async def chat_completion(self, messages: List[TogetherMessage], 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create chat completion using Together AI"""
        try:
            context = context or {}
            start_time = datetime.now()
            
            # Convert messages to Together format
            together_messages = []
            for msg in messages:
                together_msg = {
                    "role": msg.role,
                    "content": msg.content
                }
                if msg.name:
                    together_msg["name"] = msg.name
                if msg.function_call:
                    together_msg["function_call"] = msg.function_call
                together_messages.append(together_msg)
            
            # Prepare completion parameters
            completion_params = {
                'model': context.get('model', self.config.model),
                'messages': together_messages,
                'max_tokens': context.get('max_tokens', self.config.max_tokens),
                'temperature': context.get('temperature', self.config.temperature),
                'top_p': context.get('top_p', self.config.top_p),
                'top_k': context.get('top_k', self.config.top_k),
                'repetition_penalty': context.get('repetition_penalty', self.config.repetition_penalty),
                'stop': context.get('stop_sequences', self.config.stop_sequences) or None,
                'stream': context.get('stream', self.config.stream)
            }
            
            # Add safety model if specified
            if self.config.safety_model:
                completion_params['safety_model'] = self.config.safety_model
            
            # Add functions if available and function calling is enabled
            if self.config.enable_function_calling and self._functions:
                functions = []
                for func in self._functions.values():
                    functions.append({
                        'name': func.name,
                        'description': func.description,
                        'parameters': func.parameters
                    })
                completion_params['functions'] = functions
                completion_params['function_call'] = context.get('function_call', 'auto')
            
            # Remove None values
            completion_params = {k: v for k, v in completion_params.items() if v is not None}
            
            # Create completion
            if completion_params.get('stream'):
                return await self._stream_chat(**completion_params)
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
                self._usage_stats['total_prompt_tokens'] += usage.prompt_tokens
                self._usage_stats['total_completion_tokens'] += usage.completion_tokens
                self._usage_stats['models_used'].add(completion_params['model'])
            
            result = {
                'response': message.content or "",
                'role': message.role,
                'finish_reason': choice.finish_reason,
                'execution_time': execution_time,
                'model_used': response.model,
                'usage': response.usage.__dict__ if hasattr(response, 'usage') else {},
                'function_call': message.function_call if hasattr(message, 'function_call') and message.function_call else None
            }
            
            # Handle function calls
            if result['function_call'] and self.config.enable_function_calling:
                function_result = await self._execute_function(result['function_call'])
                result['function_result'] = function_result
                self._usage_stats['function_calls'] += 1
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _stream_chat(self, **completion_params) -> Dict[str, Any]:
        """Handle streaming chat completions"""
        try:
            full_content = ""
            function_call = None
            model_used = None
            finish_reason = None
            
            stream = await self._async_client.chat.completions.create(**completion_params)
            
            async for chunk in stream:
                if hasattr(chunk, 'choices') and chunk.choices:
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
                    
                    if hasattr(choice, 'finish_reason') and choice.finish_reason:
                        finish_reason = choice.finish_reason
                
                if hasattr(chunk, 'model'):
                    model_used = chunk.model
            
            result = {
                'response': full_content,
                'stream': True,
                'model_used': model_used,
                'finish_reason': finish_reason,
                'function_call': function_call
            }
            
            # Handle function calls
            if function_call and self.config.enable_function_calling:
                function_result = await self._execute_function(function_call)
                result['function_result'] = function_result
                self._usage_stats['function_calls'] += 1
            
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
    
    async def text_completion(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate text completion using Together AI"""
        try:
            context = context or {}
            start_time = datetime.now()
            
            # Prepare completion parameters
            completion_params = {
                'prompt': prompt,
                'model': context.get('model', self.config.model),
                'max_tokens': context.get('max_tokens', self.config.max_tokens),
                'temperature': context.get('temperature', self.config.temperature),
                'top_p': context.get('top_p', self.config.top_p),
                'top_k': context.get('top_k', self.config.top_k),
                'repetition_penalty': context.get('repetition_penalty', self.config.repetition_penalty),
                'stop': context.get('stop_sequences', self.config.stop_sequences) or None,
                'stream': context.get('stream', self.config.stream)
            }
            
            # Add safety model if specified
            if self.config.safety_model:
                completion_params['safety_model'] = self.config.safety_model
            
            # Remove None values
            completion_params = {k: v for k, v in completion_params.items() if v is not None}
            
            # Create completion
            if completion_params.get('stream'):
                return await self._stream_completion(**completion_params)
            else:
                response = await self._async_client.completions.create(**completion_params)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract response data
            choice = response.choices[0]
            
            # Update usage statistics
            if hasattr(response, 'usage'):
                usage = response.usage
                self._usage_stats['total_requests'] += 1
                self._usage_stats['total_tokens'] += usage.total_tokens
                self._usage_stats['total_prompt_tokens'] += usage.prompt_tokens
                self._usage_stats['total_completion_tokens'] += usage.completion_tokens
                self._usage_stats['models_used'].add(completion_params['model'])
            
            return {
                'response': choice.text,
                'finish_reason': choice.finish_reason,
                'execution_time': execution_time,
                'model_used': response.model,
                'usage': response.usage.__dict__ if hasattr(response, 'usage') else {}
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _stream_completion(self, **completion_params) -> Dict[str, Any]:
        """Handle streaming text completions"""
        try:
            full_text = ""
            model_used = None
            finish_reason = None
            
            stream = await self._async_client.completions.create(**completion_params)
            
            async for chunk in stream:
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    
                    if hasattr(choice, 'text') and choice.text:
                        full_text += choice.text
                    
                    if hasattr(choice, 'finish_reason') and choice.finish_reason:
                        finish_reason = choice.finish_reason
                
                if hasattr(chunk, 'model'):
                    model_used = chunk.model
            
            return {
                'response': full_text,
                'stream': True,
                'model_used': model_used,
                'finish_reason': finish_reason
            }
            
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
            user_msg = TogetherMessage(role="user", content=message)
            conversation.append(user_msg)
            
            # Generate response
            result = await self.chat_completion(conversation, context)
            
            # Add assistant response to conversation
            if 'response' in result and not result.get('error'):
                assistant_msg = TogetherMessage(
                    role="assistant", 
                    content=result['response'],
                    function_call=result.get('function_call')
                )
                conversation.append(assistant_msg)
                
                # Handle function calls
                if result.get('function_call'):
                    function_msg = TogetherMessage(
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
    
    async def create_fine_tuning_job(self, training_file: str, model: str, 
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a fine-tuning job"""
        if not self.config.enable_fine_tuning:
            return {'error': 'Fine-tuning not enabled'}
        
        try:
            context = context or {}
            
            # Create fine-tuning job
            job = await self._async_client.fine_tuning.jobs.create(
                training_file=training_file,
                model=model,
                hyperparameters=context.get('hyperparameters', {}),
                suffix=context.get('suffix')
            )
            
            job_id = job.id
            self._fine_tuned_models[job_id] = {
                'job_id': job_id,
                'base_model': model,
                'training_file': training_file,
                'status': job.status,
                'created_at': datetime.now().isoformat()
            }
            
            self._usage_stats['fine_tuning_jobs'] += 1
            
            return {
                'job_id': job_id,
                'status': job.status,
                'model': job.fine_tuned_model if hasattr(job, 'fine_tuned_model') else None,
                'created_at': job.created_at if hasattr(job, 'created_at') else None
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def get_fine_tuning_job(self, job_id: str) -> Dict[str, Any]:
        """Get fine-tuning job status"""
        try:
            job = await self._async_client.fine_tuning.jobs.retrieve(job_id)
            
            # Update local tracking
            if job_id in self._fine_tuned_models:
                self._fine_tuned_models[job_id]['status'] = job.status
                if hasattr(job, 'fine_tuned_model') and job.fine_tuned_model:
                    self._fine_tuned_models[job_id]['fine_tuned_model'] = job.fine_tuned_model
            
            return {
                'job_id': job_id,
                'status': job.status,
                'model': job.fine_tuned_model if hasattr(job, 'fine_tuned_model') else None,
                'training_file': job.training_file if hasattr(job, 'training_file') else None,
                'created_at': job.created_at if hasattr(job, 'created_at') else None,
                'finished_at': job.finished_at if hasattr(job, 'finished_at') else None
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def list_models(self) -> Dict[str, Any]:
        """List available models"""
        try:
            models = await self._async_client.models.list()
            
            model_list = []
            for model in models.data:
                model_list.append({
                    'id': model.id,
                    'owned_by': model.owned_by if hasattr(model, 'owned_by') else None,
                    'created': model.created if hasattr(model, 'created') else None
                })
            
            return {
                'models': model_list,
                'total_count': len(model_list)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def add_function(self, function: TogetherFunction) -> bool:
        """Add a function for function calling"""
        try:
            self._functions[function.name] = function
            return True
        except Exception as e:
            return False
    
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
        stats = self._usage_stats.copy()
        stats['models_used'] = list(stats['models_used'])  # Convert set to list
        return stats
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': self.config.__dict__,
            'active_conversations': len(self._conversations),
            'registered_functions': len(self._functions),
            'fine_tuned_models': len(self._fine_tuned_models),
            'usage_stats': self.get_usage_stats(),
            'total_interactions': len(self._interaction_history),
            'together_available': TOGETHER_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class TogetherAgentProvider(BaseAgentProvider):
    """
    Provider implementation for Together AI.
    
    Enables distributed inference with multiple open-source models,
    fine-tuning capabilities, and scalable AI deployment.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, TogetherAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not TOGETHER_AVAILABLE:
            self.logger.warning("Together AI not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "together"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.DISTRIBUTED_INFERENCE,
            AgentCapability.FINE_TUNING,
            AgentCapability.OPEN_SOURCE_MODELS,
            AgentCapability.FUNCTION_CALLING,
            AgentCapability.STREAMING,
            AgentCapability.MODEL_SCALING
        ]
    
    async def initialize(self) -> bool:
        """Initialize Together AI provider"""
        if not TOGETHER_AVAILABLE:
            self.logger.error("Together AI not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Together AI provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Together AI provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Together AI agent"""
        if not self._initialized:
            await self.initialize()
        
        if not TOGETHER_AVAILABLE:
            raise RuntimeError("Together AI not available")
        
        agent_id = f"together_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create Together configuration
            together_config = TogetherConfig(
                api_key=self.config.get('api_key', ''),
                model=self.config.get('model', 'mistralai/Mixtral-8x7B-Instruct-v0.1'),
                max_tokens=self.config.get('max_tokens', 1024),
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                top_k=self.config.get('top_k', 50),
                repetition_penalty=self.config.get('repetition_penalty', 1.0),
                stream=self.config.get('stream', False),
                safety_model=self.config.get('safety_model', ''),
                enable_function_calling=self.config.get('enable_function_calling', True),
                enable_fine_tuning=self.config.get('enable_fine_tuning', True),
                enable_embeddings=self.config.get('enable_embeddings', True),
                base_url=self.config.get('base_url', 'https://api.together.xyz'),
                timeout=self.config.get('timeout', 60.0),
                max_retries=self.config.get('max_retries', 3),
                request_timeout=self.config.get('request_timeout', 300.0)
            )
            
            # Create agent
            agent = TogetherAgent(
                agent_id=agent_id,
                config=together_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Together AI agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Together AI agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Together AI agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Together AI agent"""
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
            
            # Determine execution mode
            mode = context.get('mode', 'chat')
            conversation_id = context.get('conversation_id')
            
            if mode == 'chat' or conversation_id:
                if conversation_id:
                    result = await agent.continue_conversation(conversation_id, prompt, context)
                else:
                    messages = [TogetherMessage(role="user", content=prompt)]
                    result = await agent.chat_completion(messages, context)
            else:
                # Text completion mode
                result = await agent.text_completion(prompt, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                content=result.get('response', ''),
                agent_id=agent_id,
                provider=self.provider_name,
                metadata={
                    'execution_time_seconds': execution_time,
                    'mode': mode,
                    'model_used': result.get('model_used'),
                    'finish_reason': result.get('finish_reason'),
                    'usage': result.get('usage', {}),
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
            self.logger.error(f"Together AI agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Together AI agent"""
        if agent_id not in self._agents:
            return False
        
        agent = self._agents[agent_id]
        
        try:
            # Create Together function from tool spec
            together_function = TogetherFunction(
                name=tool_spec.name,
                description=tool_spec.description,
                parameters=tool_spec.parameters,
                implementation=tool_spec.function
            )
            
            return await agent.add_function(together_function)
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Together AI agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Use Together AI agent to generate tool specification
            synthesis_prompt = f"""
            Create a detailed tool specification for: {tool_description}
            
            Requirements:
            1. Define the tool's purpose and functionality
            2. Specify input parameters with types and validation
            3. Describe expected outputs and return formats
            4. Include error handling and edge cases
            5. Provide implementation guidelines for Python
            6. Consider distributed inference optimization
            7. Optimize for open-source model capabilities
            
            {f'Examples: {examples}' if examples else ''}
            
            Generate a comprehensive tool specification for Together AI integration.
            """
            
            agent = self._agents[agent_id]
            result = await agent.text_completion(synthesis_prompt)
            
            # Create tool spec
            tool_spec = ToolSpec(
                name=f"together_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            self.logger.info(f"Synthesized tool for Together AI agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def create_conversation(self, agent_id: str) -> str:
        """Create a conversation session with Together AI agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        agent = self._agents[agent_id]
        agent._conversations[conversation_id] = []
        return conversation_id
    
    async def create_fine_tuning_job(self, agent_id: str, training_file: str, 
                                   model: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a fine-tuning job using Together AI agent"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.create_fine_tuning_job(training_file, model, context)
    
    async def get_fine_tuning_job(self, agent_id: str, job_id: str) -> Dict[str, Any]:
        """Get fine-tuning job status"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.get_fine_tuning_job(job_id)
    
    async def list_models(self, agent_id: str) -> Dict[str, Any]:
        """List available models"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.list_models()
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Together AI agent"""
        try:
            if agent_id in self._agents:
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Together AI agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Together AI agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Together AI agent"""
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