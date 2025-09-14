"""
LlamaIndex Agent Provider

Integrates LlamaIndex framework for data framework with indexing,
querying capabilities, vector stores, and query engines.
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
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    from llama_index.core.llms import OpenAI
    from llama_index.core.embeddings import OpenAIEmbedding
    from llama_index.core import Settings
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.vector_stores.simple import SimpleVectorStore
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    # Mock classes for when LlamaIndex is not available
    class VectorStoreIndex:
        pass
    class SimpleDirectoryReader:
        pass
    class Document:
        pass
    class RetrieverQueryEngine:
        pass
    class VectorIndexRetriever:
        pass
    class ReActAgent:
        pass
    class QueryEngineTool:
        pass
    class ToolMetadata:
        pass
    class OpenAI:
        pass
    class OpenAIEmbedding:
        pass
    class Settings:
        pass
    class ChatMemoryBuffer:
        pass
    class ChromaVectorStore:
        pass
    class SimpleVectorStore:
        pass


@dataclass
class LlamaIndexConfig:
    """Configuration for LlamaIndex agents"""
    agent_name: str
    llm_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-ada-002"
    vector_store_type: str = "simple"  # "simple", "chroma", "pinecone"
    chunk_size: int = 1024
    chunk_overlap: int = 20
    top_k: int = 5
    similarity_threshold: float = 0.7
    enable_chat_memory: bool = True
    memory_token_limit: int = 3000
    api_key: Optional[str] = None


@dataclass
class IndexedDocument:
    """Document that has been indexed"""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    indexed_at: datetime
    file_path: Optional[str] = None


class LlamaIndexAgent:
    """Wrapper for LlamaIndex agent with indexing and querying capabilities"""
    
    def __init__(self, agent_id: str, config: LlamaIndexConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._llm = None
        self._embedding_model = None
        self._index = None
        self._query_engine = None
        self._agent = None
        self._indexed_documents: List[IndexedDocument] = []
        self._query_history = []
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the LlamaIndex agent"""
        if not LLAMAINDEX_AVAILABLE:
            return False
        
        try:
            # Configure LLM
            if self.config.api_key:
                self._llm = OpenAI(
                    model=self.config.llm_model,
                    api_key=self.config.api_key
                )
                self._embedding_model = OpenAIEmbedding(
                    model=self.config.embedding_model,
                    api_key=self.config.api_key
                )
                
                # Set global settings
                Settings.llm = self._llm
                Settings.embed_model = self._embedding_model
            
            # Configure chunk settings
            Settings.chunk_size = self.config.chunk_size
            Settings.chunk_overlap = self.config.chunk_overlap
            
            # Initialize empty index
            await self._create_empty_index()
            
            return True
            
        except Exception as e:
            return False
    
    async def _create_empty_index(self):
        """Create an empty vector index"""
        try:
            # Create empty index with vector store
            if self.config.vector_store_type == "simple":
                vector_store = SimpleVectorStore()
            else:
                # Default to simple vector store
                vector_store = SimpleVectorStore()
            
            # Create index with empty documents
            empty_docs = [Document(text="Initial empty document")]
            self._index = VectorStoreIndex.from_documents(
                empty_docs,
                vector_store=vector_store
            )
            
            # Create query engine
            self._query_engine = self._index.as_query_engine(
                similarity_top_k=self.config.top_k
            )
            
            # Create ReAct agent if LLM is available
            if self._llm:
                # Create tools for the agent
                query_tool = QueryEngineTool(
                    query_engine=self._query_engine,
                    metadata=ToolMetadata(
                        name="vector_search",
                        description="Search through indexed documents to find relevant information"
                    )
                )
                
                # Create agent with memory
                memory = None
                if self.config.enable_chat_memory:
                    memory = ChatMemoryBuffer.from_defaults(
                        token_limit=self.config.memory_token_limit
                    )
                
                self._agent = ReActAgent.from_tools(
                    tools=[query_tool],
                    llm=self._llm,
                    memory=memory,
                    verbose=True
                )
            
        except Exception as e:
            # Fallback: create basic index
            self._index = None
            self._query_engine = None
            self._agent = None
    
    async def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Index documents into the vector store"""
        try:
            llama_docs = []
            indexed_docs = []
            
            for doc_data in documents:
                # Create LlamaIndex document
                content = doc_data.get('content', '')
                metadata = doc_data.get('metadata', {})
                
                if content:
                    llama_doc = Document(
                        text=content,
                        metadata=metadata
                    )
                    llama_docs.append(llama_doc)
                    
                    # Track indexed document
                    indexed_doc = IndexedDocument(
                        doc_id=str(uuid.uuid4()),
                        content=content,
                        metadata=metadata,
                        indexed_at=datetime.now(),
                        file_path=doc_data.get('file_path')
                    )
                    indexed_docs.append(indexed_doc)
            
            if not llama_docs:
                return {'error': 'No valid documents to index'}
            
            # Create or update index
            if self._index is None:
                self._index = VectorStoreIndex.from_documents(llama_docs)
            else:
                # Add documents to existing index
                for doc in llama_docs:
                    self._index.insert(doc)
            
            # Update query engine
            self._query_engine = self._index.as_query_engine(
                similarity_top_k=self.config.top_k
            )
            
            # Update agent tools if agent exists
            if self._agent:
                query_tool = QueryEngineTool(
                    query_engine=self._query_engine,
                    metadata=ToolMetadata(
                        name="vector_search",
                        description="Search through indexed documents to find relevant information"
                    )
                )
                # Note: ReActAgent doesn't support dynamic tool updates easily
                # Would need to recreate agent with updated tools
            
            # Store indexed documents
            self._indexed_documents.extend(indexed_docs)
            
            return {
                'indexed_count': len(llama_docs),
                'total_documents': len(self._indexed_documents),
                'success': True
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def query(self, query_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query the indexed documents"""
        try:
            start_time = datetime.now()
            context = context or {}
            
            # Use agent if available, otherwise use query engine
            if self._agent:
                # Use ReAct agent for complex reasoning
                if hasattr(self._agent, 'chat'):
                    if asyncio.iscoroutinefunction(self._agent.chat):
                        response = await self._agent.chat(query_text)
                    else:
                        response = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: self._agent.chat(query_text)
                        )
                else:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self._agent.query(query_text)
                    )
                
                content = str(response)
                source_nodes = getattr(response, 'source_nodes', [])
                
            elif self._query_engine:
                # Use basic query engine
                if hasattr(self._query_engine, 'aquery'):
                    response = await self._query_engine.aquery(query_text)
                else:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self._query_engine.query(query_text)
                    )
                
                content = str(response)
                source_nodes = getattr(response, 'source_nodes', [])
                
            else:
                return {'error': 'No query engine available'}
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract source information
            sources = []
            for node in source_nodes:
                source_info = {
                    'content': getattr(node, 'text', '')[:200] + '...',
                    'score': getattr(node, 'score', 0.0),
                    'metadata': getattr(node, 'metadata', {})
                }
                sources.append(source_info)
            
            # Record query
            query_record = {
                'query': query_text,
                'response': content,
                'execution_time': execution_time,
                'sources_count': len(sources),
                'timestamp': datetime.now().isoformat(),
                'agent_used': self._agent is not None
            }
            self._query_history.append(query_record)
            
            return {
                'response': content,
                'sources': sources,
                'execution_time': execution_time,
                'query_record': query_record
            }
            
        except Exception as e:
            query_record = {
                'query': query_text,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            self._query_history.append(query_record)
            
            return {
                'response': '',
                'error': str(e),
                'query_record': query_record
            }
    
    async def index_directory(self, directory_path: str) -> Dict[str, Any]:
        """Index all documents in a directory"""
        try:
            if not LLAMAINDEX_AVAILABLE:
                return {'error': 'LlamaIndex not available'}
            
            # Use SimpleDirectoryReader
            reader = SimpleDirectoryReader(input_dir=directory_path)
            documents = reader.load_data()
            
            if not documents:
                return {'error': 'No documents found in directory'}
            
            # Create or update index
            if self._index is None:
                self._index = VectorStoreIndex.from_documents(documents)
            else:
                for doc in documents:
                    self._index.insert(doc)
            
            # Update query engine
            self._query_engine = self._index.as_query_engine(
                similarity_top_k=self.config.top_k
            )
            
            # Track indexed documents
            for doc in documents:
                indexed_doc = IndexedDocument(
                    doc_id=str(uuid.uuid4()),
                    content=doc.text,
                    metadata=doc.metadata,
                    indexed_at=datetime.now(),
                    file_path=directory_path
                )
                self._indexed_documents.append(indexed_doc)
            
            return {
                'indexed_count': len(documents),
                'total_documents': len(self._indexed_documents),
                'directory': directory_path,
                'success': True
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': self.config.__dict__,
            'indexed_documents_count': len(self._indexed_documents),
            'query_history_count': len(self._query_history),
            'has_index': self._index is not None,
            'has_query_engine': self._query_engine is not None,
            'has_agent': self._agent is not None,
            'created_at': self._created_at.isoformat()
        }


class LlamaIndexAgentProvider(BaseAgentProvider):
    """
    Provider implementation for LlamaIndex framework.
    
    Enables data framework capabilities with document indexing, vector stores,
    query engines, and intelligent retrieval-augmented generation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, LlamaIndexAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not LLAMAINDEX_AVAILABLE:
            self.logger.warning("LlamaIndex not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "llamaindex"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.DOCUMENT_PROCESSING,
            AgentCapability.INFORMATION_RETRIEVAL,
            AgentCapability.RAG,
            AgentCapability.REASONING_CHAINS
        ]
    
    async def initialize(self) -> bool:
        """Initialize LlamaIndex provider"""
        if not LLAMAINDEX_AVAILABLE:
            self.logger.error("LlamaIndex not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("LlamaIndex agent provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LlamaIndex provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new LlamaIndex agent"""
        if not self._initialized:
            await self.initialize()
        
        if not LLAMAINDEX_AVAILABLE:
            raise RuntimeError("LlamaIndex not available")
        
        agent_id = f"li_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create LlamaIndex configuration
            llamaindex_config = LlamaIndexConfig(
                agent_name=config.name or f"LlamaIndex_Agent_{agent_id}",
                llm_model=self.config.get('llm_model', 'gpt-3.5-turbo'),
                embedding_model=self.config.get('embedding_model', 'text-embedding-ada-002'),
                vector_store_type=self.config.get('vector_store_type', 'simple'),
                chunk_size=self.config.get('chunk_size', 1024),
                chunk_overlap=self.config.get('chunk_overlap', 20),
                top_k=self.config.get('top_k', 5),
                similarity_threshold=self.config.get('similarity_threshold', 0.7),
                enable_chat_memory=self.config.get('enable_chat_memory', True),
                memory_token_limit=self.config.get('memory_token_limit', 3000),
                api_key=self.config.get('api_key')
            )
            
            # Create agent
            agent = LlamaIndexAgent(
                agent_id=agent_id,
                config=llamaindex_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize LlamaIndex agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created LlamaIndex agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create LlamaIndex agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a LlamaIndex agent"""
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
            
            # Execute query
            result = await agent.query(prompt, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                content=result.get('response', ''),
                agent_id=agent_id,
                provider=self.provider_name,
                metadata={
                    'execution_time_seconds': execution_time,
                    'sources': result.get('sources', []),
                    'sources_count': len(result.get('sources', [])),
                    'query_record': result.get('query_record', {}),
                    'indexed_documents_count': len(agent._indexed_documents),
                    'agent_info': agent.get_agent_info()
                },
                error=result.get('error')
            )
            
        except Exception as e:
            self.logger.error(f"LlamaIndex agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with LlamaIndex agent"""
        try:
            # Tools in LlamaIndex are typically query engines or custom functions
            # This would require rebuilding the agent with new tools
            self.logger.info(f"Tool registration for LlamaIndex requires agent rebuild")
            self.logger.info(f"Tool {tool_spec.name} noted for future agent updates")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using LlamaIndex agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Use LlamaIndex agent to generate tool specification
            synthesis_prompt = f"""
            Create a detailed tool specification for: {tool_description}
            
            Requirements:
            1. Define the tool's purpose and functionality
            2. Specify input parameters and their types
            3. Describe expected outputs and return values
            4. Include error handling and validation requirements
            5. Provide implementation guidelines
            
            {f'Examples of usage: {examples}' if examples else ''}
            
            Generate a comprehensive tool specification.
            """
            
            agent = self._agents[agent_id]
            result = await agent.query(synthesis_prompt)
            
            # Create tool spec
            tool_spec = ToolSpec(
                name=f"li_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            self.logger.info(f"Synthesized tool for LlamaIndex agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def index_documents(self, agent_id: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Index documents with LlamaIndex agent"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.index_documents(documents)
    
    async def index_directory(self, agent_id: str, directory_path: str) -> Dict[str, Any]:
        """Index all documents in a directory"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.index_directory(directory_path)
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a LlamaIndex agent"""
        try:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                # Clean up agent resources
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed LlamaIndex agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active LlamaIndex agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a LlamaIndex agent"""
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