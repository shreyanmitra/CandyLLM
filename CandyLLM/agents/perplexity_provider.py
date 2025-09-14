"""
Perplexity AI Agent Provider

Integrates Perplexity AI's search-augmented generation with agent capabilities,
providing real-time information retrieval with citations and research automation.
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
    import httpx
    from openai import OpenAI, AsyncOpenAI  # Perplexity uses OpenAI-compatible API
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False
    # Mock classes for when dependencies are not available
    httpx = None
    OpenAI = None
    AsyncOpenAI = None


@dataclass
class PerplexityConfig:
    """Configuration for Perplexity AI agent"""
    api_key: str = ""
    model: str = "llama-3.1-sonar-large-128k-online"  # Default online model
    max_tokens: int = 1000
    temperature: float = 0.2  # Lower for factual accuracy
    top_p: float = 0.9
    frequency_penalty: float = 1.0
    presence_penalty: float = 0.0
    stream: bool = False
    return_citations: bool = True
    return_images: bool = True
    return_related_questions: bool = True
    search_domain_filter: List[str] = field(default_factory=list)
    search_recency_filter: str = "month"  # "hour", "day", "week", "month", "year"
    base_url: str = "https://api.perplexity.ai"
    timeout: float = 60.0
    max_retries: int = 3


@dataclass
class PerplexityMessage:
    """Message structure for Perplexity chat"""
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerplexityCitation:
    """Citation information from Perplexity response"""
    index: int
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    domain: Optional[str] = None


@dataclass
class PerplexityResponse:
    """Enhanced response from Perplexity with research metadata"""
    content: str
    citations: List[PerplexityCitation] = field(default_factory=list)
    related_questions: List[str] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    search_focus: Optional[str] = None
    model_used: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class PerplexityAgent:
    """Perplexity AI agent with search-augmented generation capabilities"""
    
    def __init__(self, agent_id: str, config: PerplexityConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._client = None
        self._async_client = None
        self._conversations = {}
        self._research_sessions = {}
        self._usage_stats = {
            'total_queries': 0,
            'total_tokens': 0,
            'total_citations': 0,
            'total_searches': 0,
            'research_sessions': 0,
            'models_used': set()
        }
        self._interaction_history = []
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the Perplexity agent"""
        if not PERPLEXITY_AVAILABLE:
            return False
        
        try:
            if not self.config.api_key:
                return False
            
            # Initialize clients with Perplexity API base URL
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
            
            self._async_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
            
            # Test connection
            await self._test_connection()
            
            return True
            
        except Exception as e:
            return False
    
    async def _test_connection(self):
        """Test API connection"""
        try:
            # Test with a simple query
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Perplexity API test failed: {e}")
    
    async def search_and_answer(self, query: str, context: Dict[str, Any] = None) -> PerplexityResponse:
        """Search and generate answer with citations"""
        try:
            context = context or {}
            start_time = datetime.now()
            
            # Prepare chat completion parameters
            messages = [{"role": "user", "content": query}]
            
            # Add system message if provided
            if context.get('system_prompt'):
                messages.insert(0, {"role": "system", "content": context['system_prompt']})
            
            completion_params = {
                'model': context.get('model', self.config.model),
                'messages': messages,
                'max_tokens': context.get('max_tokens', self.config.max_tokens),
                'temperature': context.get('temperature', self.config.temperature),
                'top_p': context.get('top_p', self.config.top_p),
                'frequency_penalty': context.get('frequency_penalty', self.config.frequency_penalty),
                'presence_penalty': context.get('presence_penalty', self.config.presence_penalty),
                'stream': context.get('stream', self.config.stream)
            }
            
            # Add Perplexity-specific parameters
            if self.config.return_citations:
                completion_params['return_citations'] = True
            
            if self.config.return_images:
                completion_params['return_images'] = True
                
            if self.config.return_related_questions:
                completion_params['return_related_questions'] = True
            
            if self.config.search_domain_filter:
                completion_params['search_domain_filter'] = self.config.search_domain_filter
            
            if self.config.search_recency_filter:
                completion_params['search_recency_filter'] = self.config.search_recency_filter
            
            # Remove None values
            completion_params = {k: v for k, v in completion_params.items() if v is not None}
            
            # Create completion
            if completion_params.get('stream'):
                return await self._stream_completion(**completion_params)
            else:
                response = await self._async_client.chat.completions.create(**completion_params)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract response components
            choice = response.choices[0]
            message = choice.message
            
            # Parse citations
            citations = []
            if hasattr(response, 'citations') and response.citations:
                for i, citation in enumerate(response.citations):
                    citations.append(PerplexityCitation(
                        index=i + 1,
                        url=citation.get('url', ''),
                        title=citation.get('title'),
                        snippet=citation.get('snippet'),
                        domain=citation.get('domain')
                    ))
            
            # Update usage statistics
            if hasattr(response, 'usage'):
                usage = response.usage
                self._usage_stats['total_queries'] += 1
                self._usage_stats['total_tokens'] += usage.total_tokens
                self._usage_stats['total_citations'] += len(citations)
                self._usage_stats['total_searches'] += 1
                self._usage_stats['models_used'].add(completion_params['model'])
            
            # Create enhanced response
            perplexity_response = PerplexityResponse(
                content=message.content,
                citations=citations,
                related_questions=getattr(response, 'related_questions', []),
                images=getattr(response, 'images', []),
                search_focus=context.get('search_focus'),
                model_used=completion_params['model'],
                usage=response.usage.__dict__ if hasattr(response, 'usage') else {}
            )
            
            return perplexity_response
            
        except Exception as e:
            return PerplexityResponse(
                content="",
                citations=[],
                related_questions=[],
                images=[]
            )
    
    async def _stream_completion(self, **completion_params) -> PerplexityResponse:
        """Handle streaming completions"""
        try:
            full_content = ""
            citations = []
            related_questions = []
            images = []
            model_used = None
            
            stream = await self._async_client.chat.completions.create(**completion_params)
            
            async for chunk in stream:
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    if hasattr(delta, 'content') and delta.content:
                        full_content += delta.content
                
                if hasattr(chunk, 'model'):
                    model_used = chunk.model
                
                # Extract metadata from final chunk
                if hasattr(chunk, 'citations') and chunk.citations:
                    for i, citation in enumerate(chunk.citations):
                        citations.append(PerplexityCitation(
                            index=i + 1,
                            url=citation.get('url', ''),
                            title=citation.get('title'),
                            snippet=citation.get('snippet'),
                            domain=citation.get('domain')
                        ))
                
                if hasattr(chunk, 'related_questions') and chunk.related_questions:
                    related_questions = chunk.related_questions
                
                if hasattr(chunk, 'images') and chunk.images:
                    images = chunk.images
            
            return PerplexityResponse(
                content=full_content,
                citations=citations,
                related_questions=related_questions,
                images=images,
                model_used=model_used
            )
            
        except Exception as e:
            return PerplexityResponse(
                content="",
                citations=[],
                related_questions=[],
                images=[]
            )
    
    async def continue_conversation(self, conversation_id: str, message: str,
                                  context: Dict[str, Any] = None) -> PerplexityResponse:
        """Continue an existing conversation with search context"""
        try:
            # Get or create conversation
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = []
            
            conversation = self._conversations[conversation_id]
            
            # Add user message
            user_msg = PerplexityMessage(role="user", content=message)
            conversation.append(user_msg)
            
            # Convert to API format
            messages = []
            for msg in conversation:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Modify context to include conversation
            context = context or {}
            context['messages'] = messages
            
            # Generate response
            result = await self.search_and_answer(message, context)
            
            # Add assistant response to conversation
            if result.content:
                assistant_msg = PerplexityMessage(role="assistant", content=result.content)
                conversation.append(assistant_msg)
            
            return result
            
        except Exception as e:
            return PerplexityResponse(
                content="",
                citations=[],
                related_questions=[],
                images=[]
            )
    
    async def research_topic(self, topic: str, depth: int = 3, 
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Conduct comprehensive research on a topic"""
        try:
            context = context or {}
            session_id = f"research_{uuid.uuid4().hex[:8]}"
            
            research_session = {
                'session_id': session_id,
                'topic': topic,
                'depth': depth,
                'queries': [],
                'findings': [],
                'citations': [],
                'created_at': datetime.now().isoformat()
            }
            
            # Initial research query
            initial_query = f"Provide a comprehensive overview of {topic}. Include key concepts, recent developments, and important considerations."
            
            initial_response = await self.search_and_answer(initial_query, context)
            
            research_session['queries'].append({
                'query': initial_query,
                'response': initial_response.content,
                'citations': [c.__dict__ for c in initial_response.citations],
                'related_questions': initial_response.related_questions
            })
            
            research_session['findings'].append(initial_response.content)
            research_session['citations'].extend([c.__dict__ for c in initial_response.citations])
            
            # Follow-up research based on related questions
            for i in range(min(depth - 1, len(initial_response.related_questions))):
                follow_up_query = initial_response.related_questions[i]
                
                follow_up_response = await self.search_and_answer(follow_up_query, context)
                
                research_session['queries'].append({
                    'query': follow_up_query,
                    'response': follow_up_response.content,
                    'citations': [c.__dict__ for c in follow_up_response.citations],
                    'related_questions': follow_up_response.related_questions
                })
                
                research_session['findings'].append(follow_up_response.content)
                research_session['citations'].extend([c.__dict__ for c in follow_up_response.citations])
            
            # Generate research summary
            summary_query = f"Based on the research findings, provide a comprehensive summary of {topic}, highlighting the most important insights and conclusions."
            summary_context = context.copy()
            summary_context['system_prompt'] = f"You are conducting a research summary. Use the following findings to create a comprehensive overview:\n\n" + "\n\n".join(research_session['findings'])
            
            summary_response = await self.search_and_answer(summary_query, summary_context)
            
            research_session['summary'] = {
                'content': summary_response.content,
                'citations': [c.__dict__ for c in summary_response.citations]
            }
            
            # Store research session
            self._research_sessions[session_id] = research_session
            self._usage_stats['research_sessions'] += 1
            
            return research_session
            
        except Exception as e:
            return {'error': str(e)}
    
    async def fact_check(self, claim: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fact-check a claim with citations"""
        try:
            fact_check_query = f"""
            Please fact-check the following claim: "{claim}"
            
            Provide:
            1. Whether the claim is true, false, or partially true
            2. Evidence supporting or refuting the claim
            3. Reliable sources and citations
            4. Any important context or nuances
            """
            
            fact_check_context = context or {}
            fact_check_context['search_recency_filter'] = 'month'  # Use recent sources
            fact_check_context['temperature'] = 0.1  # Low temperature for accuracy
            
            response = await self.search_and_answer(fact_check_query, fact_check_context)
            
            return {
                'claim': claim,
                'verification': response.content,
                'citations': [c.__dict__ for c in response.citations],
                'related_questions': response.related_questions,
                'confidence_indicators': {
                    'citation_count': len(response.citations),
                    'model_used': response.model_used,
                    'search_recency': fact_check_context.get('search_recency_filter')
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def get_recent_news(self, topic: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get recent news about a topic"""
        try:
            news_query = f"What are the latest news and developments about {topic}? Provide recent updates with dates and sources."
            
            news_context = context or {}
            news_context['search_recency_filter'] = 'day'  # Recent news
            news_context['return_images'] = True
            
            response = await self.search_and_answer(news_query, news_context)
            
            return {
                'topic': topic,
                'news_summary': response.content,
                'citations': [c.__dict__ for c in response.citations],
                'images': response.images,
                'related_questions': response.related_questions,
                'search_metadata': {
                    'recency_filter': news_context.get('search_recency_filter'),
                    'model_used': response.model_used
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_research_session(self, session_id: str) -> Dict[str, Any]:
        """Get research session by ID"""
        return self._research_sessions.get(session_id, {})
    
    def list_research_sessions(self) -> List[Dict[str, Any]]:
        """List all research sessions"""
        return [
            {
                'session_id': session_id,
                'topic': session['topic'],
                'depth': session['depth'],
                'query_count': len(session['queries']),
                'citation_count': len(session['citations']),
                'created_at': session['created_at']
            }
            for session_id, session in self._research_sessions.items()
        ]
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history"""
        if conversation_id not in self._conversations:
            return []
        
        return [
            {
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat()
            }
            for msg in self._conversations[conversation_id]
        ]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = self._usage_stats.copy()
        stats['models_used'] = list(stats['models_used'])  # Convert set to list
        
        # Calculate averages
        if stats['total_queries'] > 0:
            stats['average_citations_per_query'] = stats['total_citations'] / stats['total_queries']
            stats['average_tokens_per_query'] = stats['total_tokens'] / stats['total_queries']
        else:
            stats['average_citations_per_query'] = 0.0
            stats['average_tokens_per_query'] = 0.0
        
        return stats
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': self.config.__dict__,
            'active_conversations': len(self._conversations),
            'research_sessions': len(self._research_sessions),
            'usage_stats': self.get_usage_stats(),
            'total_interactions': len(self._interaction_history),
            'perplexity_available': PERPLEXITY_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class PerplexityAgentProvider(BaseAgentProvider):
    """
    Provider implementation for Perplexity AI.
    
    Enables search-augmented generation with real-time information retrieval,
    citations, fact-checking, and comprehensive research capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, PerplexityAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not PERPLEXITY_AVAILABLE:
            self.logger.warning("Perplexity AI dependencies not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "perplexity"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.SEARCH_AUGMENTED_GENERATION,
            AgentCapability.REAL_TIME_INFORMATION,
            AgentCapability.CITATION_GENERATION,
            AgentCapability.FACT_CHECKING,
            AgentCapability.RESEARCH_AUTOMATION,
            AgentCapability.NEWS_MONITORING
        ]
    
    async def initialize(self) -> bool:
        """Initialize Perplexity provider"""
        if not PERPLEXITY_AVAILABLE:
            self.logger.error("Perplexity AI dependencies not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Perplexity AI provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Perplexity provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Perplexity agent"""
        if not self._initialized:
            await self.initialize()
        
        if not PERPLEXITY_AVAILABLE:
            raise RuntimeError("Perplexity AI dependencies not available")
        
        agent_id = f"perplexity_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create Perplexity configuration
            perplexity_config = PerplexityConfig(
                api_key=self.config.get('api_key', ''),
                model=self.config.get('model', 'llama-3.1-sonar-large-128k-online'),
                max_tokens=self.config.get('max_tokens', 1000),
                temperature=self.config.get('temperature', 0.2),
                top_p=self.config.get('top_p', 0.9),
                frequency_penalty=self.config.get('frequency_penalty', 1.0),
                presence_penalty=self.config.get('presence_penalty', 0.0),
                stream=self.config.get('stream', False),
                return_citations=self.config.get('return_citations', True),
                return_images=self.config.get('return_images', True),
                return_related_questions=self.config.get('return_related_questions', True),
                search_domain_filter=self.config.get('search_domain_filter', []),
                search_recency_filter=self.config.get('search_recency_filter', 'month'),
                base_url=self.config.get('base_url', 'https://api.perplexity.ai'),
                timeout=self.config.get('timeout', 60.0),
                max_retries=self.config.get('max_retries', 3)
            )
            
            # Create agent
            agent = PerplexityAgent(
                agent_id=agent_id,
                config=perplexity_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Perplexity agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Perplexity agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Perplexity agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Perplexity agent"""
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
            mode = context.get('mode', 'search')
            conversation_id = context.get('conversation_id')
            
            if mode == 'research':
                depth = context.get('depth', 3)
                result = await agent.research_topic(prompt, depth, context)
                response_content = result.get('summary', {}).get('content', 'Research completed')
                
            elif mode == 'fact_check':
                result = await agent.fact_check(prompt, context)
                response_content = result.get('verification', '')
                
            elif mode == 'news':
                result = await agent.get_recent_news(prompt, context)
                response_content = result.get('news_summary', '')
                
            elif conversation_id:
                result = await agent.continue_conversation(conversation_id, prompt, context)
                response_content = result.content
                
            else:
                # Default search and answer
                result = await agent.search_and_answer(prompt, context)
                response_content = result.content
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract metadata based on result type
            metadata = {
                'execution_time_seconds': execution_time,
                'mode': mode,
                'usage_stats': agent.get_usage_stats(),
                'agent_info': agent.get_agent_info()
            }
            
            if hasattr(result, 'citations'):
                metadata['citations'] = [c.__dict__ for c in result.citations]
                metadata['citation_count'] = len(result.citations)
            elif isinstance(result, dict) and 'citations' in result:
                metadata['citations'] = result['citations']
                metadata['citation_count'] = len(result['citations'])
            
            if hasattr(result, 'related_questions'):
                metadata['related_questions'] = result.related_questions
            elif isinstance(result, dict) and 'related_questions' in result:
                metadata['related_questions'] = result['related_questions']
            
            if hasattr(result, 'model_used'):
                metadata['model_used'] = result.model_used
            elif isinstance(result, dict) and 'model_used' in result:
                metadata['model_used'] = result['model_used']
            
            if hasattr(result, 'images'):
                metadata['images'] = result.images
            elif isinstance(result, dict) and 'images' in result:
                metadata['images'] = result['images']
            
            # Add mode-specific metadata
            if mode == 'research' and isinstance(result, dict):
                metadata['research_session_id'] = result.get('session_id')
                metadata['research_depth'] = result.get('depth')
                metadata['query_count'] = len(result.get('queries', []))
            
            return AgentResponse(
                content=response_content,
                agent_id=agent_id,
                provider=self.provider_name,
                metadata=metadata,
                error=result.get('error') if isinstance(result, dict) else None
            )
            
        except Exception as e:
            self.logger.error(f"Perplexity agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Perplexity agent (basic implementation)"""
        if agent_id not in self._agents:
            return False
        
        try:
            # Perplexity doesn't have native tool support
            # This would be implemented as custom search queries
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Perplexity agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Use Perplexity agent to generate tool specification
            synthesis_prompt = f"""
            Create a detailed tool specification for: {tool_description}
            
            Requirements:
            1. Define the tool's purpose and functionality
            2. Specify input parameters with types and validation
            3. Describe expected outputs and return formats
            4. Include error handling and edge cases
            5. Provide implementation guidelines for Python
            6. Consider search-augmented generation capabilities
            7. Optimize for real-time information retrieval
            
            {f'Examples: {examples}' if examples else ''}
            
            Generate a comprehensive tool specification for Perplexity AI integration.
            """
            
            agent = self._agents[agent_id]
            result = await agent.search_and_answer(synthesis_prompt)
            
            # Create tool spec
            tool_spec = ToolSpec(
                name=f"perplexity_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            self.logger.info(f"Synthesized tool for Perplexity agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def create_conversation(self, agent_id: str) -> str:
        """Create a conversation session with Perplexity agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        agent = self._agents[agent_id]
        agent._conversations[conversation_id] = []
        return conversation_id
    
    async def research_topic(self, agent_id: str, topic: str, depth: int = 3) -> Dict[str, Any]:
        """Conduct research on a topic"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.research_topic(topic, depth)
    
    async def fact_check(self, agent_id: str, claim: str) -> Dict[str, Any]:
        """Fact-check a claim"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.fact_check(claim)
    
    async def get_recent_news(self, agent_id: str, topic: str) -> Dict[str, Any]:
        """Get recent news about a topic"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.get_recent_news(topic)
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Perplexity agent"""
        try:
            if agent_id in self._agents:
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Perplexity agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Perplexity agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Perplexity agent"""
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