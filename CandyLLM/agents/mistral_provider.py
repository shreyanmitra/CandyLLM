"""
Mistral AI Agent Provider

Integrates Mistral's language models with agent capabilities including
function calling, embeddings, and chat completion with European AI excellence.
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
    from mistralai.client import MistralClient
    from mistralai.async_client import MistralAsyncClient
    from mistralai.models.chat_completion import ChatMessage, ToolCall, FunctionCall
    from mistralai.models.embeddings import EmbeddingRequest
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    # Mock classes for when Mistral is not available
    MistralClient = None
    MistralAsyncClient = None
    ChatMessage = None
    ToolCall = None
    FunctionCall = None
    EmbeddingRequest = None


@dataclass
class MistralConfig:
    """Configuration for Mistral AI agent"""
    api_key: str = ""
    model: str = "mistral-large-latest"  # mistral-small, mistral-medium, mistral-large-latest
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    random_seed: Optional[int] = None
    safe_prompt: bool = False
    enable_function_calling: bool = True
    enable_embeddings: bool = True
    stream: bool = False
    endpoint: str = "https://api.mistral.ai"
    timeout: float = 60.0
    max_retries: int = 3
    embedding_model: str = "mistral-embed"
    chat_history_max: int = 50


@dataclass
class MistralMessage:
    """Message structure for Mistral chat"""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MistralTool:
    """Tool definition for Mistral function calling"""
    type: str = "function"
    function: Dict[str, Any] = field(default_factory=dict)
    implementation: Optional[callable] = None


class MistralAgent:
    """Mistral AI agent with European AI capabilities"""
    
    def __init__(self, agent_id: str, config: MistralConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._client = None
        self._async_client = None
        self._conversations = {}
        self._tools = {}
        self._embeddings_cache = {}
        self._usage_stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_completion_tokens': 0,
            'total_prompt_tokens': 0,
            'function_calls': 0,
            'embeddings_created': 0
        }
        self._interaction_history = []
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the Mistral agent"""
        if not MISTRAL_AVAILABLE:
            return False
        
        try:
            if not self.config.api_key:
                return False
            
            # Initialize clients
            self._client = MistralClient(
                api_key=self.config.api_key,
                endpoint=self.config.endpoint,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
            
            self._async_client = MistralAsyncClient(
                api_key=self.config.api_key,
                endpoint=self.config.endpoint,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
            
            # Test connection
            await self._test_connection()
            
            return True
            
        except Exception as e:
            return False
    
    async def _test_connection(self):
        """Test API connection"""
        try:
            # Test with a simple chat completion
            test_message = ChatMessage(role="user", content="Hello")
            response = await self._async_client.chat(
                model=self.config.model,
                messages=[test_message],
                max_tokens=1
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Mistral API test failed: {e}")
    
    async def chat_completion(self, messages: List[MistralMessage], 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create chat completion using Mistral"""
        try:
            context = context or {}
            start_time = datetime.now()
            
            # Convert messages to Mistral format
            mistral_messages = []
            for msg in messages:
                mistral_msg = ChatMessage(
                    role=msg.role,
                    content=msg.content
                )
                if msg.name:
                    mistral_msg.name = msg.name
                if msg.tool_calls:
                    mistral_msg.tool_calls = msg.tool_calls
                if msg.tool_call_id:
                    mistral_msg.tool_call_id = msg.tool_call_id
                mistral_messages.append(mistral_msg)
            
            # Prepare completion parameters
            completion_params = {
                'model': context.get('model', self.config.model),
                'messages': mistral_messages,
                'max_tokens': context.get('max_tokens', self.config.max_tokens),
                'temperature': context.get('temperature', self.config.temperature),
                'top_p': context.get('top_p', self.config.top_p),
                'random_seed': context.get('random_seed', self.config.random_seed),
                'safe_prompt': context.get('safe_prompt', self.config.safe_prompt),
                'stream': context.get('stream', self.config.stream)
            }
            
            # Add tools if available and function calling is enabled
            if self.config.enable_function_calling and self._tools:
                tools = []
                for tool in self._tools.values():
                    tools.append({
                        'type': tool.type,
                        'function': tool.function
                    })
                completion_params['tools'] = tools
                completion_params['tool_choice'] = context.get('tool_choice', 'auto')
            
            # Remove None values
            completion_params = {k: v for k, v in completion_params.items() if v is not None}
            
            # Create completion
            if completion_params.get('stream'):
                return await self._stream_completion(**completion_params)
            else:
                response = await self._async_client.chat(**completion_params)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract response data
            choice = response.choices[0]
            message = choice.message
            
            # Update usage statistics
            if hasattr(response, 'usage'):
                usage = response.usage
                self._usage_stats['total_requests'] += 1
                self._usage_stats['total_tokens'] += usage.total_tokens
                self._usage_stats['total_completion_tokens'] += usage.completion_tokens
                self._usage_stats['total_prompt_tokens'] += usage.prompt_tokens
            
            result = {
                'response': message.content or "",
                'role': message.role,
                'finish_reason': choice.finish_reason,
                'execution_time': execution_time,
                'model_used': response.model,
                'usage': response.usage.__dict__ if hasattr(response, 'usage') else {},
                'tool_calls': message.tool_calls if hasattr(message, 'tool_calls') and message.tool_calls else None
            }
            
            # Handle tool calls
            if result['tool_calls'] and self.config.enable_function_calling:
                tool_results = await self._execute_tools(result['tool_calls'])
                result['tool_results'] = tool_results
                self._usage_stats['function_calls'] += len(result['tool_calls'])
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _stream_completion(self, **completion_params) -> Dict[str, Any]:
        """Handle streaming completions"""
        try:
            full_content = ""
            tool_calls = []
            model_used = None
            finish_reason = None
            
            async for chunk in await self._async_client.chat_stream(**completion_params):
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    if hasattr(delta, 'content') and delta.content:
                        full_content += delta.content
                    
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        tool_calls.extend(delta.tool_calls)
                    
                    if hasattr(choice, 'finish_reason') and choice.finish_reason:
                        finish_reason = choice.finish_reason
                
                if hasattr(chunk, 'model'):
                    model_used = chunk.model
            
            result = {
                'response': full_content,
                'stream': True,
                'model_used': model_used,
                'finish_reason': finish_reason,
                'tool_calls': tool_calls if tool_calls else None
            }
            
            # Handle tool calls
            if tool_calls and self.config.enable_function_calling:
                tool_results = await self._execute_tools(tool_calls)
                result['tool_results'] = tool_results
                self._usage_stats['function_calls'] += len(tool_calls)
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _execute_tools(self, tool_calls: List[ToolCall]) -> List[Dict[str, Any]]:
        """Execute tool calls"""
        results = []
        
        for tool_call in tool_calls:
            try:
                function_call = tool_call.function
                function_name = function_call.name
                
                if function_name not in self._tools:
                    results.append({
                        'tool_call_id': tool_call.id,
                        'error': f'Tool {function_name} not found'
                    })
                    continue
                
                tool = self._tools[function_name]
                if not tool.implementation:
                    results.append({
                        'tool_call_id': tool_call.id,
                        'error': f'Tool {function_name} has no implementation'
                    })
                    continue
                
                # Parse arguments
                arguments = function_call.arguments
                if isinstance(arguments, str):
                    args = json.loads(arguments)
                else:
                    args = arguments
                
                # Execute function
                if asyncio.iscoroutinefunction(tool.implementation):
                    result = await tool.implementation(**args)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: tool.implementation(**args)
                    )
                
                results.append({
                    'tool_call_id': tool_call.id,
                    'function_name': function_name,
                    'arguments': args,
                    'result': result
                })
                
            except Exception as e:
                results.append({
                    'tool_call_id': tool_call.id,
                    'error': str(e)
                })
        
        return results
    
    async def generate_text(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate text using Mistral (wrapper around chat completion)"""
        try:
            messages = [MistralMessage(role="user", content=prompt)]
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
            user_msg = MistralMessage(role="user", content=message)
            conversation.append(user_msg)
            
            # Limit conversation history
            if len(conversation) > self.config.chat_history_max:
                conversation = conversation[-self.config.chat_history_max:]
                self._conversations[conversation_id] = conversation
            
            # Generate response
            result = await self.chat_completion(conversation, context)
            
            # Add assistant response to conversation
            if 'response' in result and not result.get('error'):
                assistant_msg = MistralMessage(
                    role="assistant", 
                    content=result['response'],
                    tool_calls=result.get('tool_calls')
                )
                conversation.append(assistant_msg)
                
                # Add tool results as tool messages
                if result.get('tool_results'):
                    for tool_result in result['tool_results']:
                        tool_msg = MistralMessage(
                            role="tool",
                            content=json.dumps(tool_result['result']),
                            tool_call_id=tool_result['tool_call_id']
                        )
                        conversation.append(tool_msg)
            
            result['conversation_id'] = conversation_id
            result['conversation_length'] = len(conversation)
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    async def create_embeddings(self, texts: List[str], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create embeddings using Mistral"""
        if not self.config.enable_embeddings:
            return {'error': 'Embeddings not enabled'}
        
        try:
            context = context or {}
            start_time = datetime.now()
            
            # Check cache first
            cache_key = hash(tuple(texts))
            if cache_key in self._embeddings_cache:
                return self._embeddings_cache[cache_key]
            
            # Create embeddings
            response = await self._async_client.embeddings(
                model=context.get('embedding_model', self.config.embedding_model),
                input=texts
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            embeddings = [embedding.embedding for embedding in response.data]
            
            result = {
                'embeddings': embeddings,
                'model_used': response.model,
                'execution_time': execution_time,
                'embedding_dimension': len(embeddings[0]) if embeddings else 0,
                'text_count': len(texts),
                'usage': response.usage.__dict__ if hasattr(response, 'usage') else {}
            }
            
            # Cache result
            self._embeddings_cache[cache_key] = result
            self._usage_stats['embeddings_created'] += len(texts)
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    async def add_tool(self, tool_name: str, tool_function: Dict[str, Any], 
                      implementation: callable = None) -> bool:
        """Add a tool for function calling"""
        try:
            tool = MistralTool(
                type="function",
                function=tool_function,
                implementation=implementation
            )
            
            self._tools[tool_name] = tool
            return True
            
        except Exception as e:
            return False
    
    async def similarity_search(self, query: str, documents: List[str], 
                               top_k: int = 5) -> Dict[str, Any]:
        """Perform similarity search using embeddings"""
        try:
            # Get query embedding
            query_result = await self.create_embeddings([query])
            if query_result.get('error'):
                return query_result
            
            query_embedding = query_result['embeddings'][0]
            
            # Get document embeddings
            doc_result = await self.create_embeddings(documents)
            if doc_result.get('error'):
                return doc_result
            
            doc_embeddings = doc_result['embeddings']
            
            # Calculate similarities (cosine similarity)
            similarities = []
            for i, doc_embedding in enumerate(doc_embeddings):
                # Simple dot product (normalized embeddings)
                similarity = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                similarities.append((i, similarity, documents[i]))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:top_k]
            
            return {
                'query': query,
                'results': [
                    {
                        'document': doc,
                        'similarity_score': score,
                        'index': idx
                    }
                    for idx, score, doc in top_results
                ],
                'total_documents': len(documents)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history"""
        if conversation_id not in self._conversations:
            return []
        
        return [
            {
                'role': msg.role,
                'content': msg.content,
                'name': msg.name,
                'tool_calls': msg.tool_calls,
                'tool_call_id': msg.tool_call_id,
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
            'registered_tools': len(self._tools),
            'cached_embeddings': len(self._embeddings_cache),
            'usage_stats': self._usage_stats,
            'total_interactions': len(self._interaction_history),
            'mistral_available': MISTRAL_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class MistralAgentProvider(BaseAgentProvider):
    """
    Provider implementation for Mistral AI.
    
    Enables European AI excellence with function calling, embeddings,
    chat completion, and advanced reasoning capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, MistralAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not MISTRAL_AVAILABLE:
            self.logger.warning("Mistral AI not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "mistral"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.FUNCTION_CALLING,
            AgentCapability.EMBEDDINGS,
            AgentCapability.CHAT,
            AgentCapability.STREAMING,
            AgentCapability.SIMILARITY_SEARCH,
            AgentCapability.EUROPEAN_AI_COMPLIANCE
        ]
    
    async def initialize(self) -> bool:
        """Initialize Mistral provider"""
        if not MISTRAL_AVAILABLE:
            self.logger.error("Mistral AI not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Mistral AI provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Mistral provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Mistral agent"""
        if not self._initialized:
            await self.initialize()
        
        if not MISTRAL_AVAILABLE:
            raise RuntimeError("Mistral AI not available")
        
        agent_id = f"mistral_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create Mistral configuration
            mistral_config = MistralConfig(
                api_key=self.config.get('api_key', ''),
                model=self.config.get('model', 'mistral-large-latest'),
                max_tokens=self.config.get('max_tokens', 1000),
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                random_seed=self.config.get('random_seed'),
                safe_prompt=self.config.get('safe_prompt', False),
                enable_function_calling=self.config.get('enable_function_calling', True),
                enable_embeddings=self.config.get('enable_embeddings', True),
                stream=self.config.get('stream', False),
                endpoint=self.config.get('endpoint', 'https://api.mistral.ai'),
                timeout=self.config.get('timeout', 60.0),
                max_retries=self.config.get('max_retries', 3),
                embedding_model=self.config.get('embedding_model', 'mistral-embed'),
                chat_history_max=self.config.get('chat_history_max', 50)
            )
            
            # Create agent
            agent = MistralAgent(
                agent_id=agent_id,
                config=mistral_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Mistral agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Mistral agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Mistral agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Mistral agent"""
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
                    'tool_calls': result.get('tool_calls'),
                    'tool_results': result.get('tool_results'),
                    'conversation_id': result.get('conversation_id'),
                    'conversation_length': result.get('conversation_length'),
                    'stream': result.get('stream', False),
                    'usage_stats': agent.get_usage_stats(),
                    'agent_info': agent.get_agent_info()
                },
                error=result.get('error')
            )
            
        except Exception as e:
            self.logger.error(f"Mistral agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Mistral agent"""
        if agent_id not in self._agents:
            return False
        
        agent = self._agents[agent_id]
        
        try:
            # Create Mistral function definition
            function_def = {
                'name': tool_spec.name,
                'description': tool_spec.description,
                'parameters': tool_spec.parameters
            }
            
            return await agent.add_tool(
                tool_spec.name,
                function_def,
                tool_spec.function
            )
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Mistral agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Use Mistral agent to generate tool specification
            synthesis_prompt = f"""
            Create a detailed tool specification for: {tool_description}
            
            Requirements:
            1. Define the tool's purpose and functionality
            2. Specify input parameters with types and validation
            3. Describe expected outputs and return formats
            4. Include error handling and edge cases
            5. Provide implementation guidelines for Python
            6. Consider European AI compliance and data protection
            7. Optimize for Mistral's function calling capabilities
            
            {f'Examples: {examples}' if examples else ''}
            
            Generate a comprehensive tool specification for Mistral AI integration.
            """
            
            agent = self._agents[agent_id]
            result = await agent.generate_text(synthesis_prompt)
            
            # Create tool spec
            tool_spec = ToolSpec(
                name=f"mistral_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            self.logger.info(f"Synthesized tool for Mistral agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def create_conversation(self, agent_id: str) -> str:
        """Create a conversation session with Mistral agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        agent = self._agents[agent_id]
        agent._conversations[conversation_id] = []
        return conversation_id
    
    async def create_embeddings(self, agent_id: str, texts: List[str]) -> Dict[str, Any]:
        """Create embeddings using Mistral agent"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.create_embeddings(texts)
    
    async def similarity_search(self, agent_id: str, query: str, documents: List[str], top_k: int = 5) -> Dict[str, Any]:
        """Perform similarity search using Mistral embeddings"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.similarity_search(query, documents, top_k)
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Mistral agent"""
        try:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                
                # Close client connections
                if agent._client:
                    await agent._client.close()
                if agent._async_client:
                    await agent._async_client.close()
                
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Mistral agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Mistral agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Mistral agent"""
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