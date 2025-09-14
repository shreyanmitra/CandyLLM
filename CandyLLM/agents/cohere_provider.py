"""
Cohere Agent Provider

Integrates Cohere's language models and agent capabilities including
command models, chat functionality, and retrieval-augmented generation.
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
    import cohere
    from cohere import Client, AsyncClient
    from cohere.responses.chat import StreamedChatResponse
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    # Mock classes for when Cohere is not available
    cohere = None
    Client = None
    AsyncClient = None
    StreamedChatResponse = None


@dataclass
class CohereConfig:
    """Configuration for Cohere agent"""
    api_key: str = ""
    model: str = "command"  # "command", "command-light", "command-nightly"
    max_tokens: int = 500
    temperature: float = 0.7
    k: int = 0  # Top-k sampling
    p: float = 0.75  # Top-p (nucleus) sampling
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    end_sequences: List[str] = field(default_factory=list)
    stop_sequences: List[str] = field(default_factory=list)
    return_likelihoods: str = "NONE"  # "GENERATION", "ALL", "NONE"
    truncate: str = "END"  # "START", "END"
    enable_connectors: bool = True
    enable_search: bool = True
    enable_rag: bool = True
    chat_history_max: int = 20
    stream_responses: bool = False


@dataclass
class ChatMessage:
    """Chat message structure for Cohere"""
    user_name: str
    text: str
    timestamp: datetime = field(default_factory=datetime.now)


class CohereAgent:
    """Cohere agent with chat and RAG capabilities"""
    
    def __init__(self, agent_id: str, config: CohereConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._client = None
        self._async_client = None
        self._chat_history = {}
        self._documents = {}
        self._connectors = []
        self._interaction_history = []
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the Cohere agent"""
        if not COHERE_AVAILABLE:
            return False
        
        try:
            if not self.config.api_key:
                return False
            
            # Initialize clients
            self._client = Client(api_key=self.config.api_key)
            self._async_client = AsyncClient(api_key=self.config.api_key)
            
            # Test connection
            await self._test_connection()
            
            return True
            
        except Exception as e:
            return False
    
    async def _test_connection(self):
        """Test API connection"""
        try:
            # Simple generation test
            response = await self._async_client.generate(
                prompt="Test",
                model=self.config.model,
                max_tokens=1
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Cohere API test failed: {e}")
    
    async def generate_text(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate text using Cohere"""
        try:
            context = context or {}
            start_time = datetime.now()
            
            # Prepare generation parameters
            gen_params = {
                'prompt': prompt,
                'model': context.get('model', self.config.model),
                'max_tokens': context.get('max_tokens', self.config.max_tokens),
                'temperature': context.get('temperature', self.config.temperature),
                'k': context.get('k', self.config.k),
                'p': context.get('p', self.config.p),
                'frequency_penalty': context.get('frequency_penalty', self.config.frequency_penalty),
                'presence_penalty': context.get('presence_penalty', self.config.presence_penalty),
                'end_sequences': context.get('end_sequences', self.config.end_sequences),
                'stop_sequences': context.get('stop_sequences', self.config.stop_sequences),
                'return_likelihoods': context.get('return_likelihoods', self.config.return_likelihoods),
                'truncate': context.get('truncate', self.config.truncate)
            }
            
            # Remove empty parameters
            gen_params = {k: v for k, v in gen_params.items() if v or k == 'k'}
            
            # Generate response
            response = await self._async_client.generate(**gen_params)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'response': response.generations[0].text,
                'token_count': len(response.generations[0].text.split()),
                'likelihood': response.generations[0].likelihood if hasattr(response.generations[0], 'likelihood') else None,
                'execution_time': execution_time,
                'finish_reason': response.generations[0].finish_reason if hasattr(response.generations[0], 'finish_reason') else None,
                'model_used': gen_params['model']
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def chat(self, message: str, conversation_id: str = None, 
                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Chat using Cohere's chat endpoint"""
        try:
            context = context or {}
            start_time = datetime.now()
            conversation_id = conversation_id or f"chat_{uuid.uuid4().hex[:8]}"
            
            # Get or create chat history
            if conversation_id not in self._chat_history:
                self._chat_history[conversation_id] = []
            
            chat_history = self._chat_history[conversation_id]
            
            # Prepare chat parameters
            chat_params = {
                'message': message,
                'model': context.get('model', self.config.model),
                'chat_history': [
                    {'user_name': msg.user_name, 'text': msg.text}
                    for msg in chat_history[-self.config.chat_history_max:]
                ],
                'temperature': context.get('temperature', self.config.temperature),
                'max_tokens': context.get('max_tokens', self.config.max_tokens),
                'k': context.get('k', self.config.k),
                'p': context.get('p', self.config.p),
                'frequency_penalty': context.get('frequency_penalty', self.config.frequency_penalty),
                'presence_penalty': context.get('presence_penalty', self.config.presence_penalty),
                'end_sequences': context.get('end_sequences', self.config.end_sequences),
                'stop_sequences': context.get('stop_sequences', self.config.stop_sequences)
            }
            
            # Add search and RAG if enabled
            if self.config.enable_search and context.get('search_queries'):
                chat_params['search_queries'] = context['search_queries']
            
            if self.config.enable_connectors and self._connectors:
                chat_params['connectors'] = [{'id': conn} for conn in self._connectors]
            
            if context.get('documents'):
                chat_params['documents'] = context['documents']
            
            # Remove empty parameters
            chat_params = {k: v for k, v in chat_params.items() if v or k == 'k'}
            
            # Generate chat response
            if self.config.stream_responses:
                return await self._stream_chat(**chat_params)
            else:
                response = await self._async_client.chat(**chat_params)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update chat history
            chat_history.append(ChatMessage(user_name="User", text=message))
            chat_history.append(ChatMessage(user_name="Chatbot", text=response.text))
            
            result = {
                'response': response.text,
                'conversation_id': conversation_id,
                'execution_time': execution_time,
                'model_used': chat_params['model'],
                'generation_id': response.generation_id if hasattr(response, 'generation_id') else None,
                'citations': response.citations if hasattr(response, 'citations') else [],
                'documents': response.documents if hasattr(response, 'documents') else [],
                'search_results': response.search_results if hasattr(response, 'search_results') else []
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _stream_chat(self, **chat_params) -> Dict[str, Any]:
        """Handle streaming chat responses"""
        try:
            full_response = ""
            citations = []
            documents = []
            search_results = []
            
            stream = await self._async_client.chat_stream(**chat_params)
            
            async for token in stream:
                if hasattr(token, 'text'):
                    full_response += token.text
                if hasattr(token, 'citations') and token.citations:
                    citations.extend(token.citations)
                if hasattr(token, 'documents') and token.documents:
                    documents.extend(token.documents)
                if hasattr(token, 'search_results') and token.search_results:
                    search_results.extend(token.search_results)
            
            return {
                'response': full_response,
                'stream': True,
                'citations': citations,
                'documents': documents,
                'search_results': search_results
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def embed_documents(self, texts: List[str], doc_type: str = "search_document") -> Dict[str, Any]:
        """Embed documents using Cohere embeddings"""
        try:
            response = await self._async_client.embed(
                texts=texts,
                model="embed-english-v2.0",
                input_type=doc_type
            )
            
            embeddings = response.embeddings
            
            # Store documents with embeddings
            doc_id = f"docs_{uuid.uuid4().hex[:8]}"
            self._documents[doc_id] = {
                'texts': texts,
                'embeddings': embeddings,
                'doc_type': doc_type,
                'created_at': datetime.now().isoformat()
            }
            
            return {
                'document_id': doc_id,
                'embedding_count': len(embeddings),
                'embedding_dimension': len(embeddings[0]) if embeddings else 0
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def rerank_documents(self, query: str, documents: List[str], top_n: int = 5) -> Dict[str, Any]:
        """Rerank documents using Cohere reranker"""
        try:
            response = await self._async_client.rerank(
                model="rerank-english-v2.0",
                query=query,
                documents=documents,
                top_n=top_n
            )
            
            ranked_docs = []
            for result in response.results:
                ranked_docs.append({
                    'document': documents[result.index],
                    'relevance_score': result.relevance_score,
                    'index': result.index
                })
            
            return {
                'ranked_documents': ranked_docs,
                'query': query,
                'total_documents': len(documents)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def classify_text(self, inputs: List[str], examples: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Classify text using Cohere classify endpoint"""
        try:
            if not examples:
                # Default sentiment analysis examples
                examples = [
                    {"text": "This is great!", "label": "positive"},
                    {"text": "This is terrible!", "label": "negative"},
                    {"text": "This is okay.", "label": "neutral"}
                ]
            
            response = await self._async_client.classify(
                inputs=inputs,
                examples=examples
            )
            
            results = []
            for classification in response.classifications:
                results.append({
                    'input': classification.input,
                    'prediction': classification.prediction,
                    'confidence': classification.confidence
                })
            
            return {
                'classifications': results,
                'example_count': len(examples)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def summarize_text(self, text: str, length: str = "medium", format: str = "paragraph") -> Dict[str, Any]:
        """Summarize text using Cohere summarization"""
        try:
            response = await self._async_client.summarize(
                text=text,
                length=length,  # "short", "medium", "long"
                format=format,  # "paragraph", "bullets"
                model="summarize-xlarge",
                additional_command="",
                temperature=0.3
            )
            
            return {
                'summary': response.summary,
                'original_length': len(text),
                'summary_length': len(response.summary),
                'compression_ratio': len(response.summary) / len(text) if text else 0
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def detect_language(self, texts: List[str]) -> Dict[str, Any]:
        """Detect language using Cohere"""
        try:
            response = await self._async_client.detect_language(texts=texts)
            
            results = []
            for detection in response.results:
                results.append({
                    'text': texts[detection.text_index] if detection.text_index < len(texts) else "",
                    'language_code': detection.language_code,
                    'language_name': detection.language_name
                })
            
            return {'detections': results}
            
        except Exception as e:
            return {'error': str(e)}
    
    async def add_connector(self, connector_id: str) -> bool:
        """Add a connector for search"""
        try:
            if connector_id not in self._connectors:
                self._connectors.append(connector_id)
            return True
        except Exception as e:
            return False
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a conversation"""
        if conversation_id not in self._chat_history:
            return []
        
        return [
            {
                'user_name': msg.user_name,
                'text': msg.text,
                'timestamp': msg.timestamp.isoformat()
            }
            for msg in self._chat_history[conversation_id]
        ]
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': self.config.__dict__,
            'active_conversations': len(self._chat_history),
            'stored_documents': len(self._documents),
            'connectors': self._connectors,
            'total_interactions': len(self._interaction_history),
            'cohere_available': COHERE_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class CohereAgentProvider(BaseAgentProvider):
    """
    Provider implementation for Cohere.
    
    Enables chat, generation, summarization, classification, embeddings,
    reranking, and retrieval-augmented generation capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, CohereAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not COHERE_AVAILABLE:
            self.logger.warning("Cohere not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "cohere"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.CHAT,
            AgentCapability.RAG,
            AgentCapability.SEARCH,
            AgentCapability.SUMMARIZATION,
            AgentCapability.CLASSIFICATION,
            AgentCapability.LANGUAGE_DETECTION
        ]
    
    async def initialize(self) -> bool:
        """Initialize Cohere provider"""
        if not COHERE_AVAILABLE:
            self.logger.error("Cohere not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Cohere provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Cohere provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Cohere agent"""
        if not self._initialized:
            await self.initialize()
        
        if not COHERE_AVAILABLE:
            raise RuntimeError("Cohere not available")
        
        agent_id = f"cohere_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create Cohere configuration
            cohere_config = CohereConfig(
                api_key=self.config.get('api_key', ''),
                model=self.config.get('model', 'command'),
                max_tokens=self.config.get('max_tokens', 500),
                temperature=self.config.get('temperature', 0.7),
                k=self.config.get('k', 0),
                p=self.config.get('p', 0.75),
                frequency_penalty=self.config.get('frequency_penalty', 0.0),
                presence_penalty=self.config.get('presence_penalty', 0.0),
                enable_connectors=self.config.get('enable_connectors', True),
                enable_search=self.config.get('enable_search', True),
                enable_rag=self.config.get('enable_rag', True),
                chat_history_max=self.config.get('chat_history_max', 20),
                stream_responses=self.config.get('stream_responses', False)
            )
            
            # Create agent
            agent = CohereAgent(
                agent_id=agent_id,
                config=cohere_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Cohere agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Cohere agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Cohere agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Cohere agent"""
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
            
            if mode == 'chat':
                conversation_id = context.get('conversation_id')
                result = await agent.chat(prompt, conversation_id, context)
            elif mode == 'generate':
                result = await agent.generate_text(prompt, context)
            elif mode == 'summarize':
                result = await agent.summarize_text(prompt, 
                                                  context.get('length', 'medium'),
                                                  context.get('format', 'paragraph'))
            elif mode == 'classify':
                result = await agent.classify_text([prompt], context.get('examples'))
            else:
                # Default to chat
                result = await agent.chat(prompt, context.get('conversation_id'), context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                content=result.get('response', result.get('summary', '')),
                agent_id=agent_id,
                provider=self.provider_name,
                metadata={
                    'execution_time_seconds': execution_time,
                    'mode': mode,
                    'model_used': result.get('model_used'),
                    'conversation_id': result.get('conversation_id'),
                    'citations': result.get('citations', []),
                    'documents': result.get('documents', []),
                    'search_results': result.get('search_results', []),
                    'classifications': result.get('classifications', []),
                    'token_count': result.get('token_count'),
                    'agent_info': agent.get_agent_info()
                },
                error=result.get('error')
            )
            
        except Exception as e:
            self.logger.error(f"Cohere agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Cohere agent (basic implementation)"""
        if agent_id not in self._agents:
            return False
        
        try:
            # Cohere doesn't have native tool support like OpenAI
            # This would need to be implemented as a custom wrapper
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Cohere agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Use Cohere to generate tool specification
            synthesis_prompt = f"""
            Create a detailed tool specification for: {tool_description}
            
            Requirements:
            1. Define the tool's purpose and functionality
            2. Specify input parameters with types and validation
            3. Describe expected outputs and return formats
            4. Include error handling and edge cases
            5. Provide implementation guidelines
            6. Consider integration with Cohere's capabilities (chat, classification, summarization, etc.)
            
            {f'Examples: {examples}' if examples else ''}
            
            Generate a comprehensive tool specification for Cohere integration.
            """
            
            agent = self._agents[agent_id]
            result = await agent.generate_text(synthesis_prompt)
            
            # Create tool spec
            tool_spec = ToolSpec(
                name=f"cohere_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            self.logger.info(f"Synthesized tool for Cohere agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def create_conversation(self, agent_id: str) -> str:
        """Create a conversation session with Cohere agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        agent = self._agents[agent_id]
        agent._chat_history[conversation_id] = []
        return conversation_id
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Cohere agent"""
        try:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                
                # Clean up client connections
                if agent._client:
                    agent._client.close()
                if agent._async_client:
                    await agent._async_client.close()
                
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Cohere agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Cohere agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Cohere agent"""
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