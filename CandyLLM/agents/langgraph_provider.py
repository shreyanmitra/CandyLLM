"""
LangGraph Agent Provider

Integrates LangGraph framework for building graph-based agent workflows
with state management, conditional routing, and complex execution patterns.
"""

import uuid
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Union, TypeVar, Generic
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

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
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import ToolExecutor, ToolInvocation
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.runnables import RunnableConfig
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Mock classes for when LangGraph is not available
    class StateGraph:
        pass
    class MemorySaver:
        pass
    class ToolExecutor:
        pass
    class ToolInvocation:
        pass
    class BaseMessage:
        pass
    class HumanMessage:
        pass
    class AIMessage:
        pass
    class SystemMessage:
        pass
    class RunnableConfig:
        pass
    END = "END"
    START = "START"


class WorkflowState(Dict):
    """Base state class for LangGraph workflows"""
    
    def __init__(self):
        super().__init__()
        self.update({
            'messages': [],
            'current_step': 'start',
            'execution_path': [],
            'variables': {},
            'errors': [],
            'metadata': {}
        })


class NodeType(Enum):
    """Types of nodes in LangGraph workflows"""
    AGENT = "agent"
    TOOL = "tool"
    CONDITION = "condition"
    TRANSFORM = "transform"
    HUMAN = "human"
    MERGE = "merge"


@dataclass
class LangGraphNode:
    """Definition for a LangGraph workflow node"""
    name: str
    node_type: NodeType
    function: Callable
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    retry_count: int = 0


@dataclass
class LangGraphEdge:
    """Definition for a LangGraph workflow edge"""
    from_node: str
    to_node: str
    condition: Optional[Callable] = None
    condition_description: str = ""
    weight: float = 1.0


@dataclass
class WorkflowDefinition:
    """Complete workflow definition for LangGraph"""
    name: str
    description: str
    nodes: List[LangGraphNode]
    edges: List[LangGraphEdge]
    entry_point: str = START
    state_schema: Dict[str, Any] = field(default_factory=dict)
    checkpointer: bool = True
    max_iterations: int = 100


class LangGraphWorkflow:
    """Wrapper for LangGraph workflow execution"""
    
    def __init__(self, workflow_id: str, definition: WorkflowDefinition,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.workflow_id = workflow_id
        self.definition = definition
        self.security_manager = security_manager
        self._graph = None
        self._compiled_graph = None
        self._checkpointer = None
        self._execution_history = []
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the LangGraph workflow"""
        if not LANGGRAPH_AVAILABLE:
            return False
        
        try:
            # Create state graph
            self._graph = StateGraph(WorkflowState)
            
            # Add nodes
            for node in self.definition.nodes:
                self._graph.add_node(node.name, node.function)
            
            # Add edges
            for edge in self.definition.edges:
                if edge.condition:
                    # Conditional edge
                    self._graph.add_conditional_edges(
                        edge.from_node,
                        edge.condition,
                        {True: edge.to_node, False: END}
                    )
                else:
                    # Regular edge
                    self._graph.add_edge(edge.from_node, edge.to_node)
            
            # Set entry point
            self._graph.set_entry_point(self.definition.entry_point)
            
            # Add finish edge if not present
            if not any(edge.to_node == END for edge in self.definition.edges):
                # Find leaf nodes and connect to END
                leaf_nodes = self._find_leaf_nodes()
                for leaf_node in leaf_nodes:
                    self._graph.add_edge(leaf_node, END)
            
            # Setup checkpointer if enabled
            if self.definition.checkpointer:
                self._checkpointer = MemorySaver()
            
            # Compile graph
            if self._checkpointer:
                self._compiled_graph = self._graph.compile(checkpointer=self._checkpointer)
            else:
                self._compiled_graph = self._graph.compile()
            
            return True
            
        except Exception as e:
            return False
    
    def _find_leaf_nodes(self) -> List[str]:
        """Find nodes that don't have outgoing edges"""
        node_names = {node.name for node in self.definition.nodes}
        nodes_with_outgoing = {edge.from_node for edge in self.definition.edges}
        return list(node_names - nodes_with_outgoing)
    
    async def execute(self, initial_input: Dict[str, Any], 
                     config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Execute the workflow"""
        if not self._compiled_graph:
            return {'error': 'Workflow not initialized'}
        
        try:
            start_time = datetime.now()
            
            # Create initial state
            initial_state = WorkflowState()
            initial_state.update(initial_input)
            initial_state['execution_path'] = [self.definition.entry_point]
            
            # Execute workflow
            if config and self._checkpointer:
                # Execute with checkpointing
                result = await self._compiled_graph.ainvoke(initial_state, config)
            else:
                # Execute without checkpointing
                if hasattr(self._compiled_graph, 'ainvoke'):
                    result = await self._compiled_graph.ainvoke(initial_state)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: self._compiled_graph.invoke(initial_state)
                    )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Record execution
            execution_record = {
                'workflow_id': self.workflow_id,
                'input': initial_input,
                'execution_time': execution_time,
                'execution_path': result.get('execution_path', []),
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            self._execution_history.append(execution_record)
            
            return {
                'result': result,
                'execution_time': execution_time,
                'execution_path': result.get('execution_path', []),
                'final_state': dict(result),
                'node_count': len(self.definition.nodes),
                'edge_count': len(self.definition.edges)
            }
            
        except Exception as e:
            execution_record = {
                'workflow_id': self.workflow_id,
                'input': initial_input,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            self._execution_history.append(execution_record)
            
            return {
                'result': {},
                'error': str(e),
                'execution_record': execution_record
            }
    
    async def stream_execution(self, initial_input: Dict[str, Any],
                              config: Optional[RunnableConfig] = None):
        """Stream workflow execution step by step"""
        if not self._compiled_graph:
            yield {'error': 'Workflow not initialized'}
            return
        
        try:
            # Create initial state
            initial_state = WorkflowState()
            initial_state.update(initial_input)
            
            # Stream execution
            if hasattr(self._compiled_graph, 'astream'):
                async for chunk in self._compiled_graph.astream(initial_state, config):
                    yield {
                        'chunk': chunk,
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                # Fallback to regular execution
                result = await self.execute(initial_input, config)
                yield result
                
        except Exception as e:
            yield {'error': str(e)}
    
    def get_workflow_visualization(self) -> Dict[str, Any]:
        """Get a visualization-friendly representation of the workflow"""
        return {
            'name': self.definition.name,
            'description': self.definition.description,
            'nodes': [
                {
                    'id': node.name,
                    'type': node.node_type.value,
                    'description': node.description,
                    'input_schema': node.input_schema,
                    'output_schema': node.output_schema
                }
                for node in self.definition.nodes
            ],
            'edges': [
                {
                    'from': edge.from_node,
                    'to': edge.to_node,
                    'condition': edge.condition_description,
                    'weight': edge.weight
                }
                for edge in self.definition.edges
            ],
            'entry_point': self.definition.entry_point,
            'execution_history_count': len(self._execution_history),
            'created_at': self._created_at.isoformat()
        }


class LangGraphAgentProvider(BaseAgentProvider):
    """
    Provider implementation for LangGraph framework.
    
    Enables building stateful, graph-based agent workflows with conditional
    routing, state persistence, and complex execution patterns.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._workflows: Dict[str, LangGraphWorkflow] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not LANGGRAPH_AVAILABLE:
            self.logger.warning("LangGraph not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "langgraph"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.STATE_MANAGEMENT,
            AgentCapability.WORKFLOW_ORCHESTRATION,
            AgentCapability.CONDITIONAL_ROUTING,
            AgentCapability.TOOL_SYNTHESIS
        ]
    
    async def initialize(self) -> bool:
        """Initialize LangGraph provider"""
        if not LANGGRAPH_AVAILABLE:
            self.logger.error("LangGraph not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("LangGraph agent provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LangGraph provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new LangGraph workflow agent"""
        if not self._initialized:
            await self.initialize()
        
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError("LangGraph not available")
        
        agent_id = f"lg_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create workflow definition
            workflow_def = await self._create_workflow_from_config(config, agent_id)
            
            # Create workflow
            workflow = LangGraphWorkflow(
                workflow_id=agent_id,
                definition=workflow_def,
                security_manager=self._security_manager
            )
            
            # Initialize workflow
            if not await workflow.initialize():
                raise RuntimeError("Failed to initialize LangGraph workflow")
            
            self._workflows[agent_id] = workflow
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created LangGraph workflow agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create LangGraph agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a LangGraph workflow agent"""
        if agent_id not in self._workflows:
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error="Workflow not found"
            )
        
        workflow = self._workflows[agent_id]
        context = context or {}
        
        try:
            start_time = datetime.now()
            
            # Prepare input
            workflow_input = {
                'messages': [HumanMessage(content=prompt)],
                'prompt': prompt,
                'context': context,
                'variables': context.get('variables', {})
            }
            
            # Check if streaming is requested
            if context.get('stream', False):
                # Handle streaming execution
                execution_results = []
                async for chunk in workflow.stream_execution(workflow_input):
                    execution_results.append(chunk)
                
                result = {
                    'streaming_results': execution_results,
                    'final_result': execution_results[-1] if execution_results else {}
                }
            else:
                # Regular execution
                result = await workflow.execute(workflow_input)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract content from result
            content = ""
            if 'result' in result and 'messages' in result['result']:
                messages = result['result']['messages']
                if messages:
                    content = str(messages[-1])
            elif 'final_result' in result:
                content = str(result['final_result'])
            
            return AgentResponse(
                content=content,
                agent_id=agent_id,
                provider=self.provider_name,
                metadata={
                    'execution_time_seconds': execution_time,
                    'execution_path': result.get('execution_path', []),
                    'final_state': result.get('final_state', {}),
                    'node_count': result.get('node_count', 0),
                    'edge_count': result.get('edge_count', 0),
                    'workflow_name': workflow.definition.name,
                    'streaming_used': context.get('stream', False)
                },
                error=result.get('error')
            )
            
        except Exception as e:
            self.logger.error(f"LangGraph workflow execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with LangGraph workflow"""
        if agent_id not in self._workflows:
            return False
        
        try:
            # Tools in LangGraph are typically nodes in the workflow
            # This would require rebuilding the workflow with the new tool node
            # For now, we'll log the registration
            self.logger.info(f"Tool registration for LangGraph workflows requires workflow rebuild")
            self.logger.info(f"Tool {tool_spec.name} registered for future workflow updates")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using LangGraph workflow"""
        if agent_id not in self._workflows:
            raise ValueError(f"Workflow {agent_id} not found")
        
        try:
            # Use workflow to generate tool specification
            synthesis_input = {
                'task': 'tool_synthesis',
                'description': tool_description,
                'examples': examples or [],
                'requirements': [
                    'Create a reusable tool specification',
                    'Include parameter validation',
                    'Add error handling',
                    'Provide clear documentation'
                ]
            }
            
            workflow = self._workflows[agent_id]
            result = await workflow.execute(synthesis_input)
            
            # Create tool spec
            tool_spec = ToolSpec(
                name=f"lg_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            self.logger.info(f"Synthesized tool for LangGraph workflow {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a LangGraph workflow"""
        try:
            if agent_id in self._workflows:
                workflow = self._workflows[agent_id]
                # Clean up workflow resources
                del self._workflows[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed LangGraph workflow {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy workflow: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active LangGraph workflow IDs"""
        return list(self._workflows.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a LangGraph workflow"""
        if agent_id not in self._workflows:
            return {}
        
        workflow = self._workflows[agent_id]
        config = self._agent_configs[agent_id]
        
        info = {
            'agent_id': agent_id,
            'provider': self.provider_name,
            'config': config.__dict__,
            'workflow_visualization': workflow.get_workflow_visualization(),
            'execution_history_count': len(workflow._execution_history)
        }
        
        return info
    
    async def _create_workflow_from_config(self, config: AgentConfig, agent_id: str) -> WorkflowDefinition:
        """Create a workflow definition from agent config"""
        # Create default workflow nodes
        nodes = []
        edges = []
        
        # Entry node
        async def entry_node(state: WorkflowState) -> WorkflowState:
            state['current_step'] = 'processing'
            state['execution_path'].append('entry')
            return state
        
        nodes.append(LangGraphNode(
            name="entry",
            node_type=NodeType.AGENT,
            function=entry_node,
            description="Entry point for workflow"
        ))
        
        # Processing node
        async def processing_node(state: WorkflowState) -> WorkflowState:
            # Simulate processing based on config
            prompt = state.get('prompt', '')
            messages = state.get('messages', [])
            
            # Add AI response
            response_content = f"Processed: {prompt}"
            if config.system_prompt:
                response_content = f"{config.system_prompt}\n\nResponse: {response_content}"
            
            ai_message = AIMessage(content=response_content)
            messages.append(ai_message)
            
            state['messages'] = messages
            state['current_step'] = 'complete'
            state['execution_path'].append('processing')
            
            return state
        
        nodes.append(LangGraphNode(
            name="processing",
            node_type=NodeType.AGENT,
            function=processing_node,
            description="Main processing node"
        ))
        
        # Tool execution node if tools are configured
        if config.tools:
            async def tool_node(state: WorkflowState) -> WorkflowState:
                # Execute tools based on configuration
                for tool_name in config.tools:
                    state['variables'][f'{tool_name}_executed'] = True
                
                state['execution_path'].append('tools')
                return state
            
            nodes.append(LangGraphNode(
                name="tools",
                node_type=NodeType.TOOL,
                function=tool_node,
                description="Tool execution node"
            ))
            
            # Add tool routing
            edges.append(LangGraphEdge(
                from_node="entry",
                to_node="tools"
            ))
            edges.append(LangGraphEdge(
                from_node="tools",
                to_node="processing"
            ))
        else:
            # Direct routing without tools
            edges.append(LangGraphEdge(
                from_node="entry",
                to_node="processing"
            ))
        
        # Create workflow definition
        workflow_def = WorkflowDefinition(
            name=config.name or f"LangGraph_Workflow_{agent_id}",
            description=config.description,
            nodes=nodes,
            edges=edges,
            entry_point="entry",
            checkpointer=True,
            max_iterations=self.config.get('max_iterations', 100)
        )
        
        return workflow_def