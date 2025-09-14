"""
Agent Management System

Provides intelligent routing, load balancing, and multi-agent orchestration
for the CandyLLM agentic provider ecosystem.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .base import (
    AgentManager, 
    BaseAgentProvider, 
    AgentConfig, 
    AgentResponse, 
    AgentCapability,
    ProviderRegistry,
    global_registry
)


@dataclass
class RoutingRule:
    """Rule for routing requests to appropriate providers"""
    capabilities: List[AgentCapability]
    provider_preference: List[str]
    conditions: Dict[str, Any]
    priority: int = 0


@dataclass
class WorkflowStep:
    """Single step in a multi-agent workflow"""
    agent_id: str
    provider: str
    prompt_template: str
    dependencies: List[int] = None  # Indices of previous steps this depends on
    parallel: bool = False
    timeout: int = 60


@dataclass
class WorkflowDefinition:
    """Complete workflow definition for multi-agent orchestration"""
    name: str
    description: str
    steps: List[WorkflowStep]
    global_context: Dict[str, Any] = None
    max_parallel: int = 3
    total_timeout: int = 300


class LoadBalancer:
    """Load balancer for distributing requests across providers"""
    
    def __init__(self):
        self._provider_loads: Dict[str, int] = {}
        self._provider_response_times: Dict[str, List[float]] = {}
        
    def record_request(self, provider: str):
        """Record a new request to a provider"""
        self._provider_loads[provider] = self._provider_loads.get(provider, 0) + 1
    
    def record_completion(self, provider: str, response_time: float):
        """Record request completion and response time"""
        self._provider_loads[provider] = max(0, self._provider_loads.get(provider, 1) - 1)
        
        if provider not in self._provider_response_times:
            self._provider_response_times[provider] = []
        
        self._provider_response_times[provider].append(response_time)
        
        # Keep only last 100 response times
        if len(self._provider_response_times[provider]) > 100:
            self._provider_response_times[provider] = self._provider_response_times[provider][-100:]
    
    def get_best_provider(self, candidates: List[str]) -> str:
        """Get the best provider based on load and performance"""
        if not candidates:
            raise ValueError("No candidate providers available")
        
        best_provider = candidates[0]
        best_score = float('inf')
        
        for provider in candidates:
            load = self._provider_loads.get(provider, 0)
            avg_response_time = self._get_avg_response_time(provider)
            
            # Score = load * 0.7 + normalized_response_time * 0.3
            score = load * 0.7 + (avg_response_time / 1000) * 0.3
            
            if score < best_score:
                best_score = score
                best_provider = provider
        
        return best_provider
    
    def _get_avg_response_time(self, provider: str) -> float:
        """Get average response time for a provider"""
        times = self._provider_response_times.get(provider, [500])  # Default 500ms
        return sum(times) / len(times) if times else 500.0


class IntelligentRouter:
    """Intelligent routing system for agent requests"""
    
    def __init__(self, registry: ProviderRegistry):
        self.registry = registry
        self.routing_rules: List[RoutingRule] = []
        self.load_balancer = LoadBalancer()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_routing_rule(self, rule: RoutingRule):
        """Add a routing rule"""
        self.routing_rules.append(rule)
        # Sort by priority (higher priority first)
        self.routing_rules.sort(key=lambda r: r.priority, reverse=True)
    
    async def route_request(self, prompt: str, requirements: List[AgentCapability] = None) -> Tuple[str, str]:
        """Route a request to the most appropriate provider and agent"""
        requirements = requirements or []
        
        # Find matching routing rules
        matching_rules = []
        for rule in self.routing_rules:
            if self._rule_matches(rule, requirements, prompt):
                matching_rules.append(rule)
        
        # Get candidate providers
        candidate_providers = []
        
        if matching_rules:
            # Use provider preferences from matching rules
            for rule in matching_rules:
                candidate_providers.extend(rule.provider_preference)
        else:
            # Fall back to capability-based routing
            if requirements:
                for capability in requirements:
                    providers = self.registry.get_providers_by_capability(capability)
                    candidate_providers.extend([p.provider_name for p in providers])
            else:
                # Default to all providers
                candidate_providers = self.registry.list_providers()
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for provider in candidate_providers:
            if provider not in seen and self.registry.get_provider(provider):
                seen.add(provider)
                unique_candidates.append(provider)
        
        if not unique_candidates:
            raise RuntimeError("No suitable providers found for request")
        
        # Use load balancer to select best provider
        selected_provider = self.load_balancer.get_best_provider(unique_candidates)
        
        # For now, create a new agent for each request
        # In a production system, you might maintain agent pools
        provider_instance = self.registry.get_provider(selected_provider)
        
        # Create basic agent config
        agent_config = AgentConfig(
            name=f"auto_agent_{uuid.uuid4().hex[:8]}",
            description="Auto-created agent for request routing",
            capabilities=requirements
        )
        
        agent_id = await provider_instance.create_agent(agent_config)
        
        self.logger.info(f"Routed request to {selected_provider}:{agent_id}")
        return selected_provider, agent_id
    
    def _rule_matches(self, rule: RoutingRule, requirements: List[AgentCapability], prompt: str) -> bool:
        """Check if a routing rule matches the request"""
        # Check capability requirements
        if rule.capabilities:
            if not all(cap in requirements for cap in rule.capabilities):
                return False
        
        # Check conditions (e.g., prompt keywords, complexity)
        for condition, expected in rule.conditions.items():
            if condition == 'prompt_contains':
                if not any(keyword.lower() in prompt.lower() for keyword in expected):
                    return False
            elif condition == 'prompt_length':
                if not (expected.get('min', 0) <= len(prompt) <= expected.get('max', float('inf'))):
                    return False
        
        return True


class WorkflowOrchestrator:
    """Orchestrates multi-agent workflows"""
    
    def __init__(self, registry: ProviderRegistry, router: IntelligentRouter):
        self.registry = registry
        self.router = router
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def execute_workflow(self, workflow: WorkflowDefinition, 
                             initial_context: Dict[str, Any] = None) -> List[AgentResponse]:
        """Execute a multi-agent workflow"""
        context = {**(workflow.global_context or {}), **(initial_context or {})}
        step_results = {}
        all_responses = []
        
        # Create semaphore for parallel execution limits
        semaphore = asyncio.Semaphore(workflow.max_parallel)
        
        try:
            # Execute workflow steps
            for i, step in enumerate(workflow.steps):
                if step.parallel and i > 0:
                    # Execute in parallel with previous steps if possible
                    task = asyncio.create_task(
                        self._execute_step_with_semaphore(
                            semaphore, step, i, step_results, context
                        )
                    )
                    all_responses.append(task)
                else:
                    # Wait for dependencies and execute sequentially
                    if step.dependencies:
                        for dep_idx in step.dependencies:
                            if dep_idx in step_results:
                                context.update(step_results[dep_idx])
                    
                    response = await self._execute_step(step, i, context)
                    step_results[i] = {
                        'response': response,
                        'step_index': i,
                        'step_name': f"step_{i}"
                    }
                    all_responses.append(response)
            
            # Wait for any parallel tasks to complete
            final_responses = []
            for response in all_responses:
                if asyncio.iscoroutine(response) or hasattr(response, '__await__'):
                    final_responses.append(await response)
                else:
                    final_responses.append(response)
            
            return final_responses
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def _execute_step_with_semaphore(self, semaphore: asyncio.Semaphore, 
                                         step: WorkflowStep, index: int,
                                         step_results: Dict[int, Any], 
                                         context: Dict[str, Any]) -> AgentResponse:
        """Execute a workflow step with concurrency control"""
        async with semaphore:
            return await self._execute_step(step, index, context)
    
    async def _execute_step(self, step: WorkflowStep, index: int, 
                          context: Dict[str, Any]) -> AgentResponse:
        """Execute a single workflow step"""
        try:
            # Get provider and execute
            provider = self.registry.get_provider(step.provider)
            if not provider:
                raise ValueError(f"Provider {step.provider} not found")
            
            # Format prompt with context
            formatted_prompt = step.prompt_template.format(**context)
            
            # Record request for load balancing
            self.router.load_balancer.record_request(step.provider)
            start_time = datetime.now()
            
            # Execute agent
            response = await asyncio.wait_for(
                provider.execute_agent(step.agent_id, formatted_prompt, context),
                timeout=step.timeout
            )
            
            # Record completion
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self.router.load_balancer.record_completion(step.provider, response_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Step {index} execution failed: {e}")
            # Return error response
            return AgentResponse(
                content="",
                agent_id=step.agent_id,
                provider=step.provider,
                error=str(e)
            )


class AgentProviderManager(AgentManager):
    """
    Main agent management system implementation.
    
    Provides unified interface for multi-provider agent coordination,
    intelligent routing, load balancing, and workflow orchestration.
    """
    
    def __init__(self, registry: ProviderRegistry = None):
        self.registry = registry or global_registry
        self.router = IntelligentRouter(self.registry)
        self.orchestrator = WorkflowOrchestrator(self.registry, self.router)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set up default routing rules
        self._setup_default_routing_rules()
    
    async def route_request(self, prompt: str, requirements: List[AgentCapability] = None) -> Tuple[str, str]:
        """Route a request to the most appropriate provider and agent"""
        return await self.router.route_request(prompt, requirements)
    
    async def orchestrate_multi_agent(self, 
                                    agents: List[Tuple[str, str]], 
                                    workflow: Dict[str, Any],
                                    context: Dict[str, Any] = None) -> List[AgentResponse]:
        """Orchestrate a multi-agent workflow"""
        
        # Convert simple workflow to WorkflowDefinition if needed
        if not isinstance(workflow, WorkflowDefinition):
            workflow_def = self._convert_workflow_dict(workflow, agents)
        else:
            workflow_def = workflow
        
        return await self.orchestrator.execute_workflow(workflow_def, context)
    
    async def execute_single_agent(self, prompt: str, 
                                 requirements: List[AgentCapability] = None,
                                 context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a single agent request with intelligent routing"""
        provider_name, agent_id = await self.route_request(prompt, requirements)
        
        provider = self.registry.get_provider(provider_name)
        
        # Record load balancing metrics
        self.router.load_balancer.record_request(provider_name)
        start_time = datetime.now()
        
        try:
            response = await provider.execute_agent(agent_id, prompt, context)
            
            # Record completion
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self.router.load_balancer.record_completion(provider_name, response_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Single agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=provider_name,
                error=str(e)
            )
    
    def add_routing_rule(self, capabilities: List[AgentCapability], 
                        provider_preference: List[str],
                        conditions: Dict[str, Any] = None,
                        priority: int = 0):
        """Add a custom routing rule"""
        rule = RoutingRule(
            capabilities=capabilities,
            provider_preference=provider_preference,
            conditions=conditions or {},
            priority=priority
        )
        self.router.add_routing_rule(rule)
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics"""
        stats = {
            'providers': {},
            'load_balancing': {},
            'routing_rules': len(self.router.routing_rules),
            'timestamp': datetime.now().isoformat()
        }
        
        # Provider stats
        for provider_name in self.registry.list_providers():
            provider = self.registry.get_provider(provider_name)
            health = await provider.health_check()
            agents = await provider.list_agents()
            
            stats['providers'][provider_name] = {
                'health': health,
                'agent_count': len(agents),
                'capabilities': [cap.value for cap in provider.supported_capabilities]
            }
        
        # Load balancing stats
        stats['load_balancing'] = {
            'current_loads': dict(self.router.load_balancer._provider_loads),
            'avg_response_times': {
                provider: sum(times) / len(times) if times else 0
                for provider, times in self.router.load_balancer._provider_response_times.items()
            }
        }
        
        return stats
    
    def _setup_default_routing_rules(self):
        """Set up default routing rules for common scenarios"""
        
        # Tool synthesis -> CandyLLM (has best dynamic tooling)
        self.add_routing_rule(
            capabilities=[AgentCapability.TOOL_SYNTHESIS],
            provider_preference=['candyllm', 'langchain'],
            priority=10
        )
        
        # Multi-agent workflows -> CrewAI first, then others
        self.add_routing_rule(
            capabilities=[AgentCapability.MULTI_AGENT],
            provider_preference=['crewai', 'autogen', 'langchain'],
            priority=8
        )
        
        # Code execution -> OpenAI Assistants first (sandboxed), then CandyLLM
        self.add_routing_rule(
            capabilities=[AgentCapability.CODE_EXECUTION],
            provider_preference=['openai_assistants', 'candyllm'],
            priority=7
        )
        
        # Long reasoning chains -> LangChain (good for chains)
        self.add_routing_rule(
            capabilities=[AgentCapability.REASONING_CHAINS],
            provider_preference=['langchain', 'candyllm'],
            conditions={'prompt_length': {'min': 1000}},
            priority=6
        )
    
    def _convert_workflow_dict(self, workflow: Dict[str, Any], 
                              agents: List[Tuple[str, str]]) -> WorkflowDefinition:
        """Convert simple workflow dict to WorkflowDefinition"""
        steps = []
        
        for i, (provider, agent_id) in enumerate(agents):
            step = WorkflowStep(
                agent_id=agent_id,
                provider=provider,
                prompt_template=workflow.get('steps', [{}])[i].get('prompt', '{input}'),
                dependencies=workflow.get('steps', [{}])[i].get('dependencies', []),
                parallel=workflow.get('steps', [{}])[i].get('parallel', False)
            )
            steps.append(step)
        
        return WorkflowDefinition(
            name=workflow.get('name', 'auto_workflow'),
            description=workflow.get('description', 'Auto-generated workflow'),
            steps=steps,
            global_context=workflow.get('context', {}),
            max_parallel=workflow.get('max_parallel', 3)
        )