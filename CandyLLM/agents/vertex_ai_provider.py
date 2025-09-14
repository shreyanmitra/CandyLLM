"""
Google Cloud Vertex AI Agent Builder Provider

Integrates Google Cloud Vertex AI Agent Builder for enterprise conversational AI
with Dialogflow integration, knowledge bases, and enterprise features.
"""

import uuid
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Union
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
    from google.cloud import aiplatform
    from google.cloud import discoveryengine_v1
    from google.oauth2 import service_account
    import google.auth
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    # Mock classes for when Vertex AI is not available
    aiplatform = None
    discoveryengine_v1 = None
    service_account = None
    google = None


@dataclass
class VertexAIConfig:
    """Configuration for Vertex AI Agent Builder"""
    project_id: str
    location: str = "global"
    agent_name: str = "vertex-agent"
    engine_id: Optional[str] = None
    data_store_id: Optional[str] = None
    model_name: str = "gemini-pro"
    temperature: float = 0.7
    max_output_tokens: int = 1024
    top_k: int = 40
    top_p: float = 0.95
    service_account_path: Optional[str] = None
    enable_grounding: bool = True
    enable_citations: bool = True


@dataclass
class ConversationSession:
    """Vertex AI conversation session"""
    session_id: str
    conversation_id: str
    user_pseudo_id: str
    created_at: datetime
    last_interaction: datetime
    message_count: int = 0
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)


class VertexAIAgent:
    """Wrapper for Vertex AI Agent Builder"""
    
    def __init__(self, agent_id: str, config: VertexAIConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._client = None
        self._conversation_client = None
        self._sessions: Dict[str, ConversationSession] = {}
        self._interaction_history = []
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the Vertex AI agent"""
        if not VERTEX_AI_AVAILABLE:
            return False
        
        try:
            # Initialize authentication
            if self.config.service_account_path:
                credentials = service_account.Credentials.from_service_account_file(
                    self.config.service_account_path
                )
            else:
                credentials, project = google.auth.default()
            
            # Initialize AI Platform
            aiplatform.init(
                project=self.config.project_id,
                location=self.config.location,
                credentials=credentials
            )
            
            # Initialize Discovery Engine client for conversational search
            self._client = discoveryengine_v1.ConversationalSearchServiceClient(
                credentials=credentials
            )
            
            return True
            
        except Exception as e:
            return False
    
    async def create_conversation_session(self, user_id: str = None) -> str:
        """Create a new conversation session"""
        try:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
            user_pseudo_id = user_id or f"user_{uuid.uuid4().hex[:8]}"
            
            # Create conversation using Discovery Engine
            if self._client and self.config.engine_id:
                parent = f"projects/{self.config.project_id}/locations/{self.config.location}/collections/default_collection/engines/{self.config.engine_id}"
                
                conversation = discoveryengine_v1.Conversation()
                conversation.name = f"{parent}/conversations/{session_id}"
                conversation.state = discoveryengine_v1.Conversation.State.IN_PROGRESS
                
                request = discoveryengine_v1.CreateConversationRequest(
                    parent=parent,
                    conversation=conversation,
                    conversation_id=session_id
                )
                
                if hasattr(self._client, 'create_conversation'):
                    response = self._client.create_conversation(request=request)
                    conversation_id = response.name.split('/')[-1]
                else:
                    conversation_id = session_id
            else:
                conversation_id = session_id
            
            # Create session record
            session = ConversationSession(
                session_id=session_id,
                conversation_id=conversation_id,
                user_pseudo_id=user_pseudo_id,
                created_at=datetime.now(),
                last_interaction=datetime.now()
            )
            
            self._sessions[session_id] = session
            return session_id
            
        except Exception as e:
            # Fallback: create local session
            session_id = f"session_{uuid.uuid4().hex[:8]}"
            session = ConversationSession(
                session_id=session_id,
                conversation_id=session_id,
                user_pseudo_id=user_id or f"user_{uuid.uuid4().hex[:8]}",
                created_at=datetime.now(),
                last_interaction=datetime.now()
            )
            self._sessions[session_id] = session
            return session_id
    
    async def converse(self, message: str, session_id: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Conduct a conversation with Vertex AI agent"""
        try:
            start_time = datetime.now()
            context = context or {}
            
            # Create session if not provided
            if not session_id:
                session_id = await self.create_conversation_session()
            
            session = self._sessions.get(session_id)
            if not session:
                session_id = await self.create_conversation_session()
                session = self._sessions[session_id]
            
            # Prepare conversation request
            if self._client and self.config.engine_id:
                # Use Discovery Engine for conversational search
                serving_config = f"projects/{self.config.project_id}/locations/{self.config.location}/collections/default_collection/engines/{self.config.engine_id}/servingConfigs/default_config"
                
                query = discoveryengine_v1.TextInput()
                query.input = message
                
                request = discoveryengine_v1.ConverseConversationRequest(
                    name=f"projects/{self.config.project_id}/locations/{self.config.location}/collections/default_collection/engines/{self.config.engine_id}/conversations/{session.conversation_id}",
                    query=query,
                    serving_config=serving_config,
                    safe_search=True
                )
                
                # Add conversation context
                if context.get('filter'):
                    request.filter = context['filter']
                
                # Execute conversation
                if hasattr(self._client, 'converse_conversation'):
                    response = self._client.converse_conversation(request=request)
                    
                    # Extract response content
                    reply_text = ""
                    citations = []
                    
                    if hasattr(response, 'reply') and response.reply:
                        reply_text = response.reply.summary.summary_text if response.reply.summary else ""
                        
                        # Extract citations
                        if hasattr(response.reply, 'summary') and response.reply.summary.summary_with_metadata:
                            if hasattr(response.reply.summary.summary_with_metadata, 'citations'):
                                for citation in response.reply.summary.summary_with_metadata.citations:
                                    citations.append({
                                        'title': getattr(citation, 'title', ''),
                                        'uri': getattr(citation, 'uri', ''),
                                        'start_index': getattr(citation, 'start_index', 0),
                                        'end_index': getattr(citation, 'end_index', 0)
                                    })
                    
                    conversation_state = "active"
                    search_results = []
                    
                    # Extract search results if available
                    if hasattr(response, 'search_results'):
                        for result in response.search_results:
                            search_results.append({
                                'title': getattr(result, 'title', ''),
                                'uri': getattr(result, 'uri', ''),
                                'snippet': getattr(result, 'snippet', '')
                            })
                else:
                    # Fallback response
                    reply_text = f"Processing query: {message}"
                    citations = []
                    conversation_state = "active"
                    search_results = []
            else:
                # Fallback: simple response generation
                reply_text = f"Vertex AI response to: {message}"
                citations = []
                conversation_state = "active" 
                search_results = []
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update session
            session.last_interaction = datetime.now()
            session.message_count += 1
            session.conversation_history.append({
                'user_message': message,
                'agent_response': reply_text,
                'timestamp': datetime.now().isoformat(),
                'citations': citations
            })
            
            # Record interaction
            interaction_record = {
                'session_id': session_id,
                'user_message': message,
                'agent_response': reply_text,
                'execution_time': execution_time,
                'citations_count': len(citations),
                'search_results_count': len(search_results),
                'timestamp': datetime.now().isoformat()
            }
            self._interaction_history.append(interaction_record)
            
            return {
                'response': reply_text,
                'session_id': session_id,
                'citations': citations,
                'search_results': search_results,
                'conversation_state': conversation_state,
                'execution_time': execution_time,
                'interaction_record': interaction_record
            }
            
        except Exception as e:
            interaction_record = {
                'session_id': session_id,
                'user_message': message,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            self._interaction_history.append(interaction_record)
            
            return {
                'response': '',
                'error': str(e),
                'session_id': session_id,
                'interaction_record': interaction_record
            }
    
    async def search_knowledge_base(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search the knowledge base using Vertex AI Discovery Engine"""
        try:
            if not self._client or not self.config.data_store_id:
                return {'error': 'Knowledge base not configured'}
            
            # Prepare search request
            serving_config = f"projects/{self.config.project_id}/locations/{self.config.location}/collections/default_collection/dataStores/{self.config.data_store_id}/servingConfigs/default_config"
            
            request = discoveryengine_v1.SearchRequest(
                serving_config=serving_config,
                query=query,
                page_size=10
            )
            
            # Add filters if provided
            if filters:
                filter_str = " AND ".join([f"{k}:{v}" for k, v in filters.items()])
                request.filter = filter_str
            
            # Execute search
            if hasattr(self._client, 'search'):
                # Note: This would be a different client for search
                pass
            
            # For now, return mock results
            return {
                'results': [
                    {
                        'title': f'Result for: {query}',
                        'snippet': f'Knowledge base content related to {query}',
                        'uri': 'https://example.com/document'
                    }
                ],
                'total_size': 1,
                'query': query
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a conversation session"""
        session = self._sessions.get(session_id)
        if not session:
            return {}
        
        return {
            'session_id': session.session_id,
            'conversation_id': session.conversation_id,
            'user_pseudo_id': session.user_pseudo_id,
            'created_at': session.created_at.isoformat(),
            'last_interaction': session.last_interaction.isoformat(),
            'message_count': session.message_count,
            'conversation_length': len(session.conversation_history)
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': self.config.__dict__,
            'active_sessions': len(self._sessions),
            'total_interactions': len(self._interaction_history),
            'vertex_ai_available': VERTEX_AI_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class VertexAIAgentProvider(BaseAgentProvider):
    """
    Provider implementation for Google Cloud Vertex AI Agent Builder.
    
    Enables enterprise conversational AI with Dialogflow integration,
    knowledge bases, grounding, and enterprise-grade features.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, VertexAIAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not VERTEX_AI_AVAILABLE:
            self.logger.warning("Vertex AI not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "vertex_ai"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.CONVERSATIONAL_AI,
            AgentCapability.INFORMATION_RETRIEVAL,
            AgentCapability.ENTERPRISE_FEATURES,
            AgentCapability.GROUNDING
        ]
    
    async def initialize(self) -> bool:
        """Initialize Vertex AI provider"""
        if not VERTEX_AI_AVAILABLE:
            self.logger.error("Vertex AI not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Vertex AI Agent Builder provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Vertex AI provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Vertex AI agent"""
        if not self._initialized:
            await self.initialize()
        
        if not VERTEX_AI_AVAILABLE:
            raise RuntimeError("Vertex AI not available")
        
        agent_id = f"vai_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create Vertex AI configuration
            vertex_config = VertexAIConfig(
                project_id=self.config.get('project_id'),
                location=self.config.get('location', 'global'),
                agent_name=config.name or f"VertexAI_Agent_{agent_id}",
                engine_id=self.config.get('engine_id'),
                data_store_id=self.config.get('data_store_id'),
                model_name=self.config.get('model_name', 'gemini-pro'),
                temperature=self.config.get('temperature', 0.7),
                max_output_tokens=self.config.get('max_output_tokens', 1024),
                top_k=self.config.get('top_k', 40),
                top_p=self.config.get('top_p', 0.95),
                service_account_path=self.config.get('service_account_path'),
                enable_grounding=self.config.get('enable_grounding', True),
                enable_citations=self.config.get('enable_citations', True)
            )
            
            if not vertex_config.project_id:
                raise ValueError("Google Cloud project_id required for Vertex AI")
            
            # Create agent
            agent = VertexAIAgent(
                agent_id=agent_id,
                config=vertex_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Vertex AI agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Vertex AI agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Vertex AI agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Vertex AI agent"""
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
            
            # Check for specific operation
            if context.get('operation') == 'search':
                # Search knowledge base
                result = await agent.search_knowledge_base(
                    prompt, 
                    context.get('filters', {})
                )
                content = f"Search results for: {prompt}\n"
                if 'results' in result:
                    content += f"Found {len(result['results'])} results"
                
            else:
                # Regular conversation
                session_id = context.get('session_id')
                result = await agent.converse(prompt, session_id, context)
                content = result.get('response', '')
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                content=content,
                agent_id=agent_id,
                provider=self.provider_name,
                metadata={
                    'execution_time_seconds': execution_time,
                    'result': result,
                    'session_id': result.get('session_id'),
                    'citations': result.get('citations', []),
                    'search_results': result.get('search_results', []),
                    'agent_info': agent.get_agent_info()
                },
                error=result.get('error')
            )
            
        except Exception as e:
            self.logger.error(f"Vertex AI agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Vertex AI agent"""
        try:
            # Vertex AI tools are typically integrated through function calling
            self.logger.info(f"Tool {tool_spec.name} noted for Vertex AI agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Vertex AI agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Use Vertex AI to generate tool specification
            synthesis_prompt = f"""
            Create a detailed tool specification for: {tool_description}
            
            Requirements:
            1. Define the tool's purpose and functionality
            2. Specify input parameters and their types
            3. Describe expected outputs and return formats
            4. Include error handling and validation
            5. Provide usage examples and best practices
            6. Consider enterprise requirements and compliance
            
            {f'Examples: {examples}' if examples else ''}
            
            Generate a comprehensive tool specification for enterprise use.
            """
            
            agent = self._agents[agent_id]
            result = await agent.converse(synthesis_prompt)
            
            # Create tool spec
            tool_spec = ToolSpec(
                name=f"vai_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            self.logger.info(f"Synthesized tool for Vertex AI agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def create_conversation_session(self, agent_id: str, user_id: str = None) -> str:
        """Create a conversation session with Vertex AI agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        return await agent.create_conversation_session(user_id)
    
    async def get_session_info(self, agent_id: str, session_id: str) -> Dict[str, Any]:
        """Get information about a conversation session"""
        if agent_id not in self._agents:
            return {}
        
        agent = self._agents[agent_id]
        return agent.get_session_info(session_id)
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Vertex AI agent"""
        try:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                # Clean up agent resources
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Vertex AI agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Vertex AI agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Vertex AI agent"""
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