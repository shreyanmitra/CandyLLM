"""
Haystack Agent Provider

Integrates Haystack framework for building search and QA systems with
pipeline orchestration, document processing, and retrieval capabilities.
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
    from haystack import Pipeline, Document
    from haystack.components.builders import PromptBuilder, AnswerBuilder
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.retrievers import InMemoryBM25Retriever
    from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
    from haystack.components.writers import DocumentWriter
    from haystack.document_stores import InMemoryDocumentStore
    from haystack.utils import ComponentDevice
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False
    # Mock classes for when Haystack is not available
    class Pipeline:
        pass
    class Document:
        pass
    class PromptBuilder:
        pass
    class AnswerBuilder:
        pass
    class OpenAIGenerator:
        pass
    class InMemoryBM25Retriever:
        pass
    class DocumentSplitter:
        pass
    class DocumentCleaner:
        pass
    class DocumentWriter:
        pass
    class InMemoryDocumentStore:
        pass


@dataclass
class HaystackPipelineConfig:
    """Configuration for Haystack pipelines"""
    pipeline_name: str
    pipeline_type: str = "rag"  # "rag", "qa", "generation", "indexing", "search"
    document_store_type: str = "memory"
    generator_model: str = "gpt-3.5-turbo"
    retriever_top_k: int = 5
    enable_preprocessing: bool = True
    chunk_size: int = 500
    chunk_overlap: int = 50
    api_key: Optional[str] = None


@dataclass
class DocumentSource:
    """Source for documents to be indexed"""
    source_type: str  # "text", "file", "url", "database"
    content: Optional[str] = None
    file_path: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HaystackPipelineAgent:
    """Wrapper for Haystack pipeline execution"""
    
    def __init__(self, agent_id: str, config: HaystackPipelineConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._document_store = None
        self._pipeline = None
        self._indexed_documents = []
        self._pipeline_runs = []
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the Haystack pipeline"""
        if not HAYSTACK_AVAILABLE:
            return False
        
        try:
            # Initialize document store
            if self.config.document_store_type == "memory":
                self._document_store = InMemoryDocumentStore()
            
            # Create pipeline based on type
            if self.config.pipeline_type == "rag":
                await self._create_rag_pipeline()
            elif self.config.pipeline_type == "qa":
                await self._create_qa_pipeline()
            elif self.config.pipeline_type == "generation":
                await self._create_generation_pipeline()
            elif self.config.pipeline_type == "indexing":
                await self._create_indexing_pipeline()
            elif self.config.pipeline_type == "search":
                await self._create_search_pipeline()
            else:
                await self._create_rag_pipeline()  # Default to RAG
            
            return True
            
        except Exception as e:
            return False
    
    async def _create_rag_pipeline(self):
        """Create a Retrieval-Augmented Generation pipeline"""
        pipeline = Pipeline()
        
        # Add components
        pipeline.add_component("retriever", InMemoryBM25Retriever(
            document_store=self._document_store,
            top_k=self.config.retriever_top_k
        ))
        
        pipeline.add_component("prompt_builder", PromptBuilder(
            template="""
            Context: {{ documents }}
            
            Question: {{ question }}
            
            Please provide a comprehensive answer based on the given context.
            """
        ))
        
        if self.config.api_key:
            pipeline.add_component("generator", OpenAIGenerator(
                model=self.config.generator_model,
                api_key=self.config.api_key
            ))
        
        pipeline.add_component("answer_builder", AnswerBuilder())
        
        # Connect components
        pipeline.connect("retriever", "prompt_builder.documents")
        pipeline.connect("prompt_builder", "generator")
        pipeline.connect("generator.replies", "answer_builder.replies")
        pipeline.connect("retriever", "answer_builder.documents")
        
        self._pipeline = pipeline
    
    async def _create_qa_pipeline(self):
        """Create a Question Answering pipeline"""
        pipeline = Pipeline()
        
        # Add components for QA
        pipeline.add_component("retriever", InMemoryBM25Retriever(
            document_store=self._document_store,
            top_k=self.config.retriever_top_k
        ))
        
        pipeline.add_component("prompt_builder", PromptBuilder(
            template="""
            Answer the following question based on the provided documents:
            
            Documents: {{ documents }}
            Question: {{ question }}
            
            Answer:
            """
        ))
        
        if self.config.api_key:
            pipeline.add_component("generator", OpenAIGenerator(
                model=self.config.generator_model,
                api_key=self.config.api_key
            ))
        
        # Connect components
        pipeline.connect("retriever", "prompt_builder.documents")
        pipeline.connect("prompt_builder", "generator")
        
        self._pipeline = pipeline
    
    async def _create_generation_pipeline(self):
        """Create a text generation pipeline"""
        pipeline = Pipeline()
        
        pipeline.add_component("prompt_builder", PromptBuilder(
            template="{{ prompt }}"
        ))
        
        if self.config.api_key:
            pipeline.add_component("generator", OpenAIGenerator(
                model=self.config.generator_model,
                api_key=self.config.api_key
            ))
        
        # Connect components
        pipeline.connect("prompt_builder", "generator")
        
        self._pipeline = pipeline
    
    async def _create_indexing_pipeline(self):
        """Create a document indexing pipeline"""
        pipeline = Pipeline()
        
        if self.config.enable_preprocessing:
            pipeline.add_component("cleaner", DocumentCleaner())
            pipeline.add_component("splitter", DocumentSplitter(
                split_by="sentence",
                split_length=self.config.chunk_size,
                split_overlap=self.config.chunk_overlap
            ))
        
        pipeline.add_component("writer", DocumentWriter(
            document_store=self._document_store
        ))
        
        # Connect components
        if self.config.enable_preprocessing:
            pipeline.connect("cleaner", "splitter")
            pipeline.connect("splitter", "writer")
        
        self._pipeline = pipeline
    
    async def _create_search_pipeline(self):
        """Create a document search pipeline"""
        pipeline = Pipeline()
        
        pipeline.add_component("retriever", InMemoryBM25Retriever(
            document_store=self._document_store,
            top_k=self.config.retriever_top_k
        ))
        
        self._pipeline = pipeline
    
    async def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the pipeline with a query"""
        if not self._pipeline:
            return {'error': 'Pipeline not initialized'}
        
        try:
            start_time = datetime.now()
            context = context or {}
            
            # Prepare pipeline inputs based on type
            pipeline_inputs = {}
            
            if self.config.pipeline_type in ["rag", "qa", "search"]:
                pipeline_inputs["retriever"] = {"query": query}
                if "prompt_builder" in [comp.name for comp in self._pipeline.graph.nodes]:
                    pipeline_inputs["prompt_builder"] = {"question": query}
            elif self.config.pipeline_type == "generation":
                pipeline_inputs["prompt_builder"] = {"prompt": query}
            elif self.config.pipeline_type == "indexing":
                # For indexing, the query should contain documents
                documents = context.get('documents', [])
                if isinstance(documents, list) and documents:
                    if self.config.enable_preprocessing:
                        pipeline_inputs["cleaner"] = {"documents": documents}
                    else:
                        pipeline_inputs["writer"] = {"documents": documents}
                else:
                    return {'error': 'No documents provided for indexing'}
            
            # Execute pipeline
            if hasattr(self._pipeline, 'run'):
                if asyncio.iscoroutinefunction(self._pipeline.run):
                    result = await self._pipeline.run(pipeline_inputs)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: self._pipeline.run(pipeline_inputs)
                    )
            else:
                return {'error': 'Pipeline execution not supported'}
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Record pipeline run
            run_record = {
                'query': query,
                'pipeline_type': self.config.pipeline_type,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'inputs': pipeline_inputs
            }
            self._pipeline_runs.append(run_record)
            
            return {
                'result': result,
                'execution_time': execution_time,
                'pipeline_type': self.config.pipeline_type,
                'run_record': run_record
            }
            
        except Exception as e:
            run_record = {
                'query': query,
                'pipeline_type': self.config.pipeline_type,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            self._pipeline_runs.append(run_record)
            
            return {
                'result': {},
                'error': str(e),
                'run_record': run_record
            }
    
    async def index_documents(self, document_sources: List[DocumentSource]) -> Dict[str, Any]:
        """Index documents into the document store"""
        if not self._document_store:
            return {'error': 'Document store not initialized'}
        
        try:
            documents = []
            
            for source in document_sources:
                if source.source_type == "text" and source.content:
                    doc = Document(
                        content=source.content,
                        meta=source.metadata
                    )
                    documents.append(doc)
                elif source.source_type == "file" and source.file_path:
                    # Read file content
                    try:
                        with open(source.file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        doc = Document(
                            content=content,
                            meta={**source.metadata, 'file_path': source.file_path}
                        )
                        documents.append(doc)
                    except Exception as e:
                        continue
            
            if not documents:
                return {'error': 'No valid documents to index'}
            
            # Index using indexing pipeline if available
            if self.config.pipeline_type == "indexing" and self._pipeline:
                result = await self.execute("", {'documents': documents})
                if 'error' not in result:
                    self._indexed_documents.extend(documents)
                return result
            else:
                # Direct indexing
                self._document_store.write_documents(documents)
                self._indexed_documents.extend(documents)
                
                return {
                    'indexed_count': len(documents),
                    'total_documents': len(self._indexed_documents),
                    'success': True
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def get_document_count(self) -> int:
        """Get the number of indexed documents"""
        if self._document_store:
            return self._document_store.count_documents()
        return len(self._indexed_documents)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        pipeline_info = {
            'agent_id': self.agent_id,
            'config': self.config.__dict__,
            'document_count': self.get_document_count(),
            'pipeline_runs_count': len(self._pipeline_runs),
            'created_at': self._created_at.isoformat(),
            'pipeline_type': self.config.pipeline_type
        }
        
        if self._pipeline:
            try:
                # Get pipeline components info
                pipeline_info['components'] = []
                if hasattr(self._pipeline, 'graph') and hasattr(self._pipeline.graph, 'nodes'):
                    for node in self._pipeline.graph.nodes:
                        pipeline_info['components'].append({
                            'name': getattr(node, 'name', str(node)),
                            'type': type(node).__name__
                        })
            except:
                pass
        
        return pipeline_info


class HaystackAgentProvider(BaseAgentProvider):
    """
    Provider implementation for Haystack framework.
    
    Enables building search and QA systems with pipeline orchestration,
    document processing, retrieval, and answer generation capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, HaystackPipelineAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not HAYSTACK_AVAILABLE:
            self.logger.warning("Haystack not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "haystack"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.DOCUMENT_PROCESSING,
            AgentCapability.INFORMATION_RETRIEVAL,
            AgentCapability.WORKFLOW_ORCHESTRATION,
            AgentCapability.RAG
        ]
    
    async def initialize(self) -> bool:
        """Initialize Haystack provider"""
        if not HAYSTACK_AVAILABLE:
            self.logger.error("Haystack not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Haystack agent provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Haystack provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Haystack pipeline agent"""
        if not self._initialized:
            await self.initialize()
        
        if not HAYSTACK_AVAILABLE:
            raise RuntimeError("Haystack not available")
        
        agent_id = f"hs_{uuid.uuid4().hex[:8]}"
        
        try:
            # Determine pipeline type from capabilities
            pipeline_type = "rag"  # default
            if AgentCapability.DOCUMENT_PROCESSING in config.capabilities:
                pipeline_type = "indexing"
            elif AgentCapability.INFORMATION_RETRIEVAL in config.capabilities:
                pipeline_type = "search"
            elif AgentCapability.RAG in config.capabilities:
                pipeline_type = "rag"
            
            # Create Haystack pipeline configuration
            haystack_config = HaystackPipelineConfig(
                pipeline_name=config.name or f"Pipeline_{agent_id}",
                pipeline_type=pipeline_type,
                document_store_type=self.config.get('document_store_type', 'memory'),
                generator_model=self.config.get('generator_model', 'gpt-3.5-turbo'),
                retriever_top_k=self.config.get('retriever_top_k', 5),
                enable_preprocessing=self.config.get('enable_preprocessing', True),
                chunk_size=self.config.get('chunk_size', 500),
                chunk_overlap=self.config.get('chunk_overlap', 50),
                api_key=self.config.get('api_key')
            )
            
            # Create pipeline agent
            agent = HaystackPipelineAgent(
                agent_id=agent_id,
                config=haystack_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Haystack pipeline agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Haystack pipeline agent {agent_id} with type {pipeline_type}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Haystack agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Haystack pipeline agent"""
        if agent_id not in self._agents:
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error="Pipeline agent not found"
            )
        
        agent = self._agents[agent_id]
        context = context or {}
        
        try:
            start_time = datetime.now()
            
            # Execute pipeline
            result = await agent.execute(prompt, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract content from result
            content = ""
            if 'result' in result and result['result']:
                pipeline_result = result['result']
                
                # Extract based on pipeline type
                if agent.config.pipeline_type in ["rag", "qa"]:
                    if 'answer_builder' in pipeline_result:
                        answers = pipeline_result['answer_builder'].get('answers', [])
                        if answers:
                            content = str(answers[0])
                    elif 'generator' in pipeline_result:
                        replies = pipeline_result['generator'].get('replies', [])
                        if replies:
                            content = str(replies[0])
                elif agent.config.pipeline_type == "generation":
                    if 'generator' in pipeline_result:
                        replies = pipeline_result['generator'].get('replies', [])
                        if replies:
                            content = str(replies[0])
                elif agent.config.pipeline_type == "search":
                    if 'retriever' in pipeline_result:
                        documents = pipeline_result['retriever'].get('documents', [])
                        content = f"Found {len(documents)} relevant documents"
                        if documents:
                            content += f"\n\nTop result: {documents[0].content[:200]}..."
                elif agent.config.pipeline_type == "indexing":
                    content = f"Indexing completed: {result.get('result', {})}"
            
            return AgentResponse(
                content=content,
                agent_id=agent_id,
                provider=self.provider_name,
                metadata={
                    'execution_time_seconds': execution_time,
                    'pipeline_type': agent.config.pipeline_type,
                    'document_count': agent.get_document_count(),
                    'pipeline_runs_count': len(agent._pipeline_runs),
                    'run_record': result.get('run_record', {}),
                    'pipeline_result': result.get('result', {})
                },
                error=result.get('error')
            )
            
        except Exception as e:
            self.logger.error(f"Haystack pipeline execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Haystack pipeline"""
        # Tools in Haystack are typically components in pipelines
        # This would require rebuilding the pipeline
        try:
            self.logger.info(f"Tool registration for Haystack requires pipeline component integration")
            self.logger.info(f"Tool {tool_spec.name} noted for future pipeline updates")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Haystack pipeline"""
        if agent_id not in self._agents:
            raise ValueError(f"Pipeline agent {agent_id} not found")
        
        try:
            # Use Haystack pipeline to generate tool specification
            synthesis_prompt = f"""
            Create a tool specification for: {tool_description}
            
            Requirements:
            1. Define tool purpose and functionality
            2. Specify input/output parameters
            3. Include implementation guidelines
            4. Add error handling considerations
            
            {f'Examples: {examples}' if examples else ''}
            
            Provide a detailed tool specification.
            """
            
            agent = self._agents[agent_id]
            result = await agent.execute(synthesis_prompt)
            
            # Create tool spec
            tool_spec = ToolSpec(
                name=f"hs_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            self.logger.info(f"Synthesized tool for Haystack pipeline {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def index_documents(self, agent_id: str, document_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Index documents using Haystack pipeline"""
        if agent_id not in self._agents:
            return {'error': 'Pipeline agent not found'}
        
        agent = self._agents[agent_id]
        
        try:
            # Convert to DocumentSource objects
            sources = []
            for doc_data in document_sources:
                source = DocumentSource(
                    source_type=doc_data.get('type', 'text'),
                    content=doc_data.get('content'),
                    file_path=doc_data.get('file_path'),
                    url=doc_data.get('url'),
                    metadata=doc_data.get('metadata', {})
                )
                sources.append(source)
            
            result = await agent.index_documents(sources)
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Haystack pipeline agent"""
        try:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                # Clean up pipeline resources
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Haystack pipeline agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy pipeline agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Haystack pipeline agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Haystack pipeline agent"""
        if agent_id not in self._agents:
            return {}
        
        agent = self._agents[agent_id]
        config = self._agent_configs[agent_id]
        
        info = {
            'agent_id': agent_id,
            'provider': self.provider_name,
            'config': config.__dict__,
            'pipeline_info': agent.get_pipeline_info()
        }
        
        return info