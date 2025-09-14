"""
AgentGPT Agent Provider

Integrates AgentGPT's autonomous AI agent capabilities for goal-oriented task completion,
enabling intelligent task decomposition, execution planning, and autonomous goal achievement.
"""

import uuid
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
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
    import httpx
    import requests
    from openai import OpenAI, AsyncOpenAI
    AGENTGPT_AVAILABLE = True
except ImportError:
    AGENTGPT_AVAILABLE = False
    # Mock classes for when dependencies are not available
    httpx = None
    requests = None
    OpenAI = None
    AsyncOpenAI = None


class TaskType(Enum):
    """Types of tasks AgentGPT can handle"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    PROBLEM_SOLVING = "problem_solving"
    PLANNING = "planning"
    EXECUTION = "execution"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    AUTOMATION = "automation"
    OPTIMIZATION = "optimization"


class ExecutionStatus(Enum):
    """Execution status for tasks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class AgentGPTConfig:
    """Configuration for AgentGPT agent"""
    api_key: str = ""
    model: str = "gpt-4"
    base_url: str = "https://api.openai.com/v1"
    max_iterations: int = 10
    max_tokens: int = 4000
    temperature: float = 0.7
    top_p: float = 0.9
    thinking_budget: int = 20000  # Tokens for thinking
    execution_budget: int = 50000  # Tokens for execution
    goal_refinement: bool = True
    autonomous_mode: bool = True
    safety_checks: bool = True
    enable_web_search: bool = True
    enable_code_execution: bool = False
    enable_file_operations: bool = False
    memory_persistence: bool = True
    max_subtasks: int = 20
    iteration_delay: float = 1.0
    timeout: float = 600.0  # 10 minutes
    max_retries: int = 3


@dataclass
class Goal:
    """Goal definition for AgentGPT"""
    goal_id: str
    description: str
    priority: int = 1  # 1 = highest, 5 = lowest
    deadline: Optional[datetime] = None
    success_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    parent_goal_id: Optional[str] = None
    subtask_ids: List[str] = field(default_factory=list)


@dataclass
class Task:
    """Task definition for goal execution"""
    task_id: str
    goal_id: str
    description: str
    task_type: TaskType
    status: ExecutionStatus = ExecutionStatus.PENDING
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    progress: float = 0.0  # 0.0 to 1.0
    result: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class ExecutionPlan:
    """Execution plan for goal achievement"""
    plan_id: str
    goal_id: str
    tasks: List[Task] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    estimated_total_duration: float = 0.0
    success_probability: float = 0.0
    risks: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationResult:
    """Result of a single iteration"""
    iteration: int
    thought_process: str
    action_taken: str
    observation: str
    reflection: str
    next_action: str
    progress_assessment: float
    tokens_used: int
    execution_time: float


@dataclass
class GoalExecution:
    """Complete goal execution with results"""
    execution_id: str
    goal: Goal
    plan: ExecutionPlan
    iterations: List[IterationResult] = field(default_factory=list)
    current_iteration: int = 0
    status: ExecutionStatus = ExecutionStatus.PENDING
    progress: float = 0.0
    final_result: Optional[str] = None
    total_tokens_used: int = 0
    total_execution_time: float = 0.0
    success: bool = False
    error_log: List[str] = field(default_factory=list)


class AgentGPTAgent:
    """AgentGPT autonomous agent for goal-oriented task completion"""
    
    def __init__(self, agent_id: str, config: AgentGPTConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._client = None
        self._async_client = None
        self._goals = {}
        self._executions = {}
        self._memory = {}
        self._usage_stats = {
            'total_goals': 0,
            'completed_goals': 0,
            'failed_goals': 0,
            'total_tasks': 0,
            'completed_tasks': 0,
            'total_iterations': 0,
            'total_tokens_used': 0,
            'total_execution_time': 0.0,
            'task_types_executed': {task_type.value: 0 for task_type in TaskType},
            'average_goal_completion_time': 0.0,
            'success_rate': 0.0
        }
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the AgentGPT agent"""
        if not AGENTGPT_AVAILABLE:
            return False
        
        try:
            if not self.config.api_key:
                return False
            
            # Initialize OpenAI clients
            client_kwargs = {
                'api_key': self.config.api_key,
                'base_url': self.config.base_url
            }
            
            self._client = OpenAI(**client_kwargs)
            self._async_client = AsyncOpenAI(**client_kwargs)
            
            # Test connection
            await self._test_connection()
            
            return True
            
        except Exception as e:
            return False
    
    async def _test_connection(self):
        """Test API connection"""
        try:
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            raise RuntimeError(f"AgentGPT API test failed: {e}")
    
    async def create_goal(self, description: str, context: Dict[str, Any] = None) -> Goal:
        """Create a new goal"""
        context = context or {}
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        
        goal = Goal(
            goal_id=goal_id,
            description=description,
            priority=context.get('priority', 1),
            deadline=context.get('deadline'),
            success_criteria=context.get('success_criteria', []),
            constraints=context.get('constraints', []),
            resources=context.get('resources', []),
            context=context,
            parent_goal_id=context.get('parent_goal_id')
        )
        
        self._goals[goal_id] = goal
        self._usage_stats['total_goals'] += 1
        
        return goal
    
    async def refine_goal(self, goal: Goal) -> Goal:
        """Refine and clarify the goal using AI"""
        try:
            refinement_prompt = f"""
            You are an expert goal refinement agent. Analyze and improve the following goal:
            
            Goal: {goal.description}
            Context: {goal.context}
            Current Success Criteria: {goal.success_criteria}
            Current Constraints: {goal.constraints}
            
            Please provide:
            1. A refined, more specific goal description
            2. Clear, measurable success criteria
            3. Important constraints to consider
            4. Required resources
            5. Potential risks and mitigation strategies
            
            Respond in JSON format with keys: refined_description, success_criteria, constraints, resources, risks
            """
            
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": refinement_prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            refinement_text = response.choices[0].message.content
            
            # Parse JSON response (simplified - would use proper JSON parsing)
            try:
                if '{' in refinement_text and '}' in refinement_text:
                    json_start = refinement_text.find('{')
                    json_end = refinement_text.rfind('}') + 1
                    refinement_data = json.loads(refinement_text[json_start:json_end])
                    
                    goal.description = refinement_data.get('refined_description', goal.description)
                    goal.success_criteria = refinement_data.get('success_criteria', goal.success_criteria)
                    goal.constraints = refinement_data.get('constraints', goal.constraints)
                    goal.resources = refinement_data.get('resources', goal.resources)
            except:
                pass  # Keep original goal if parsing fails
            
            return goal
            
        except Exception as e:
            return goal  # Return original goal if refinement fails
    
    async def create_execution_plan(self, goal: Goal) -> ExecutionPlan:
        """Create execution plan for a goal"""
        try:
            planning_prompt = f"""
            You are an expert AI planning agent. Create a detailed execution plan for the following goal:
            
            Goal: {goal.description}
            Success Criteria: {goal.success_criteria}
            Constraints: {goal.constraints}
            Resources: {goal.resources}
            Context: {goal.context}
            
            Break this goal into specific, actionable tasks. For each task, provide:
            1. Task description
            2. Task type (research, analysis, creative, problem_solving, planning, execution, communication, learning, automation, optimization)
            3. Priority (1-5)
            4. Dependencies (other tasks that must complete first)
            5. Estimated duration in minutes
            
            Also provide:
            - Overall execution order
            - Success probability (0.0 to 1.0)
            - Potential risks
            - Alternative approaches
            - Resource requirements
            
            Respond in JSON format with a structured plan.
            """
            
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": planning_prompt}],
                max_tokens=2000,
                temperature=0.4
            )
            
            plan_text = response.choices[0].message.content
            plan_id = f"plan_{uuid.uuid4().hex[:8]}"
            
            # Create execution plan (simplified parsing)
            tasks = []
            execution_order = []
            
            # Would implement proper JSON parsing here
            # For now, create a basic plan structure
            task_descriptions = [
                "Analyze the goal and gather initial information",
                "Research relevant approaches and methodologies", 
                "Develop a detailed strategy",
                "Execute the main components of the plan",
                "Review and optimize results",
                "Document findings and conclusions"
            ]
            
            for i, desc in enumerate(task_descriptions):
                task_id = f"task_{uuid.uuid4().hex[:8]}"
                task = Task(
                    task_id=task_id,
                    goal_id=goal.goal_id,
                    description=desc,
                    task_type=TaskType.ANALYSIS if i < 2 else TaskType.EXECUTION,
                    priority=1 if i < 3 else 2,
                    estimated_duration=10.0 + (i * 5)
                )
                tasks.append(task)
                execution_order.append(task_id)
            
            plan = ExecutionPlan(
                plan_id=plan_id,
                goal_id=goal.goal_id,
                tasks=tasks,
                execution_order=execution_order,
                estimated_total_duration=sum(task.estimated_duration for task in tasks),
                success_probability=0.8,
                risks=["Time constraints", "Resource limitations", "Technical complexity"],
                alternatives=["Alternative approach A", "Fallback strategy B"]
            )
            
            return plan
            
        except Exception as e:
            # Return minimal plan
            return ExecutionPlan(
                plan_id=f"plan_{uuid.uuid4().hex[:8]}",
                goal_id=goal.goal_id,
                tasks=[],
                estimated_total_duration=0.0,
                success_probability=0.5
            )
    
    async def execute_goal(self, goal_id: str, context: Dict[str, Any] = None) -> GoalExecution:
        """Execute a goal autonomously"""
        try:
            if goal_id not in self._goals:
                raise ValueError(f"Goal {goal_id} not found")
            
            goal = self._goals[goal_id]
            context = context or {}
            
            # Refine goal if enabled
            if self.config.goal_refinement:
                goal = await self.refine_goal(goal)
            
            # Create execution plan
            plan = await self.create_execution_plan(goal)
            
            # Initialize execution
            execution_id = f"exec_{uuid.uuid4().hex[:8]}"
            execution = GoalExecution(
                execution_id=execution_id,
                goal=goal,
                plan=plan,
                status=ExecutionStatus.IN_PROGRESS
            )
            
            self._executions[execution_id] = execution
            
            start_time = datetime.now()
            
            # Execute iterations
            for iteration in range(1, self.config.max_iterations + 1):
                iteration_start = datetime.now()
                
                # Think about current situation
                iteration_result = await self._execute_iteration(execution, iteration)
                execution.iterations.append(iteration_result)
                execution.current_iteration = iteration
                
                # Update progress
                execution.progress = min(1.0, iteration / self.config.max_iterations)
                execution.total_tokens_used += iteration_result.tokens_used
                
                # Check for completion
                if iteration_result.progress_assessment >= 0.95:
                    execution.status = ExecutionStatus.COMPLETED
                    execution.success = True
                    execution.final_result = "Goal completed successfully"
                    break
                
                # Check for failure conditions
                if "error" in iteration_result.observation.lower() or "failed" in iteration_result.observation.lower():
                    if iteration >= 3:  # Allow a few retries
                        execution.status = ExecutionStatus.FAILED
                        execution.success = False
                        execution.final_result = "Goal execution failed"
                        break
                
                # Delay between iterations
                await asyncio.sleep(self.config.iteration_delay)
            
            # Finalize execution
            execution.total_execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self._usage_stats['total_iterations'] += len(execution.iterations)
            self._usage_stats['total_tokens_used'] += execution.total_tokens_used
            self._usage_stats['total_execution_time'] += execution.total_execution_time
            
            if execution.success:
                self._usage_stats['completed_goals'] += 1
            else:
                self._usage_stats['failed_goals'] += 1
            
            # Update success rate
            total_goals = self._usage_stats['completed_goals'] + self._usage_stats['failed_goals']
            if total_goals > 0:
                self._usage_stats['success_rate'] = self._usage_stats['completed_goals'] / total_goals
            
            return execution
            
        except Exception as e:
            return GoalExecution(
                execution_id=f"failed_{uuid.uuid4().hex[:8]}",
                goal=self._goals.get(goal_id, Goal(goal_id="", description="")),
                plan=ExecutionPlan(plan_id="", goal_id=goal_id),
                status=ExecutionStatus.FAILED,
                final_result=f"Execution failed: {str(e)}",
                error_log=[str(e)]
            )
    
    async def _execute_iteration(self, execution: GoalExecution, iteration: int) -> IterationResult:
        """Execute a single iteration of the goal"""
        try:
            start_time = datetime.now()
            
            # Build context for this iteration
            context = self._build_iteration_context(execution, iteration)
            
            # Generate thinking and action
            iteration_prompt = f"""
            You are AgentGPT, an autonomous AI agent working on the following goal:
            
            Goal: {execution.goal.description}
            Success Criteria: {execution.goal.success_criteria}
            Current Progress: {execution.progress:.1%}
            Iteration: {iteration}/{self.config.max_iterations}
            
            Previous Context:
            {context}
            
            Think step by step about:
            1. Current situation and progress
            2. What needs to be done next
            3. Best action to take
            4. Expected outcome
            
            Then provide:
            - Thought Process: Your reasoning
            - Action: What you will do
            - Expected Result: What you expect to achieve
            
            Be specific and actionable.
            """
            
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": iteration_prompt}],
                max_tokens=1000,
                temperature=self.config.temperature
            )
            
            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Parse response (simplified)
            thought_process = response_text[:200]
            action_taken = response_text[200:400] if len(response_text) > 200 else "Continue working on goal"
            
            # Simulate observation (would be actual action execution)
            observation = f"Completed iteration {iteration} action. Making progress on goal."
            
            # Generate reflection
            reflection_prompt = f"""
            Reflect on the iteration:
            Action: {action_taken}
            Observation: {observation}
            
            Assess:
            1. Was this iteration successful?
            2. What progress was made (0.0 to 1.0)?
            3. What should be done next?
            
            Provide a brief reflection and progress score.
            """
            
            reflection_response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": reflection_prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            reflection = reflection_response.choices[0].message.content
            tokens_used += reflection_response.usage.total_tokens if reflection_response.usage else 0
            
            # Extract progress assessment (simplified)
            progress_assessment = min(1.0, iteration / self.config.max_iterations)
            if "successful" in reflection.lower():
                progress_assessment += 0.1
            if "completed" in reflection.lower():
                progress_assessment = 1.0
            
            next_action = "Continue with next iteration" if progress_assessment < 0.95 else "Finalize goal completion"
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return IterationResult(
                iteration=iteration,
                thought_process=thought_process,
                action_taken=action_taken,
                observation=observation,
                reflection=reflection,
                next_action=next_action,
                progress_assessment=min(1.0, progress_assessment),
                tokens_used=tokens_used,
                execution_time=execution_time
            )
            
        except Exception as e:
            return IterationResult(
                iteration=iteration,
                thought_process=f"Error in iteration: {str(e)}",
                action_taken="Error handling",
                observation=f"Iteration failed: {str(e)}",
                reflection="Need to retry or adjust approach",
                next_action="Continue with error recovery",
                progress_assessment=0.0,
                tokens_used=0,
                execution_time=0.0
            )
    
    def _build_iteration_context(self, execution: GoalExecution, current_iteration: int) -> str:
        """Build context from previous iterations"""
        if not execution.iterations:
            return "Starting fresh with this goal."
        
        recent_iterations = execution.iterations[-3:]  # Last 3 iterations
        context_parts = []
        
        for iter_result in recent_iterations:
            context_parts.append(f"Iteration {iter_result.iteration}: {iter_result.action_taken} -> {iter_result.observation}")
        
        return "\n".join(context_parts)
    
    async def pause_execution(self, execution_id: str) -> bool:
        """Pause goal execution"""
        if execution_id in self._executions:
            self._executions[execution_id].status = ExecutionStatus.PAUSED
            return True
        return False
    
    async def resume_execution(self, execution_id: str) -> bool:
        """Resume paused execution"""
        if execution_id in self._executions:
            execution = self._executions[execution_id]
            if execution.status == ExecutionStatus.PAUSED:
                execution.status = ExecutionStatus.IN_PROGRESS
                return True
        return False
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel goal execution"""
        if execution_id in self._executions:
            self._executions[execution_id].status = ExecutionStatus.CANCELLED
            return True
        return False
    
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get goal by ID"""
        return self._goals.get(goal_id)
    
    def get_execution(self, execution_id: str) -> Optional[GoalExecution]:
        """Get execution by ID"""
        return self._executions.get(execution_id)
    
    def list_goals(self) -> List[Dict[str, Any]]:
        """List all goals"""
        return [
            {
                'goal_id': goal.goal_id,
                'description': goal.description,
                'priority': goal.priority,
                'success_criteria_count': len(goal.success_criteria),
                'constraints_count': len(goal.constraints)
            }
            for goal in self._goals.values()
        ]
    
    def list_executions(self) -> List[Dict[str, Any]]:
        """List all executions"""
        return [
            {
                'execution_id': execution.execution_id,
                'goal_id': execution.goal.goal_id,
                'status': execution.status.value,
                'progress': execution.progress,
                'iterations': len(execution.iterations),
                'success': execution.success
            }
            for execution in self._executions.values()
        ]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self._usage_stats.copy()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': {
                'model': self.config.model,
                'max_iterations': self.config.max_iterations,
                'autonomous_mode': self.config.autonomous_mode,
                'goal_refinement': self.config.goal_refinement,
                'max_subtasks': self.config.max_subtasks
            },
            'goals_count': len(self._goals),
            'executions_count': len(self._executions),
            'usage_stats': self.get_usage_stats(),
            'agentgpt_available': AGENTGPT_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class AgentGPTProvider(BaseAgentProvider):
    """
    Provider implementation for AgentGPT.
    
    Enables autonomous AI agents for goal-oriented task completion with
    intelligent planning, execution, and iterative improvement.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, AgentGPTAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not AGENTGPT_AVAILABLE:
            self.logger.warning("AgentGPT dependencies not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "agentgpt"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.AUTONOMOUS_PLANNING,
            AgentCapability.GOAL_ORIENTED_EXECUTION,
            AgentCapability.ITERATIVE_IMPROVEMENT,
            AgentCapability.TASK_DECOMPOSITION,
            AgentCapability.PROGRESS_TRACKING,
            AgentCapability.ADAPTIVE_EXECUTION
        ]
    
    async def initialize(self) -> bool:
        """Initialize AgentGPT provider"""
        if not AGENTGPT_AVAILABLE:
            self.logger.error("AgentGPT dependencies not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("AgentGPT provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AgentGPT provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new AgentGPT agent"""
        if not self._initialized:
            await self.initialize()
        
        if not AGENTGPT_AVAILABLE:
            raise RuntimeError("AgentGPT dependencies not available")
        
        agent_id = f"agentgpt_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create AgentGPT configuration
            agentgpt_config = AgentGPTConfig(
                api_key=self.config.get('api_key', ''),
                model=self.config.get('model', 'gpt-4'),
                base_url=self.config.get('base_url', 'https://api.openai.com/v1'),
                max_iterations=self.config.get('max_iterations', 10),
                max_tokens=self.config.get('max_tokens', 4000),
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                thinking_budget=self.config.get('thinking_budget', 20000),
                execution_budget=self.config.get('execution_budget', 50000),
                goal_refinement=self.config.get('goal_refinement', True),
                autonomous_mode=self.config.get('autonomous_mode', True),
                safety_checks=self.config.get('safety_checks', True),
                enable_web_search=self.config.get('enable_web_search', True),
                enable_code_execution=self.config.get('enable_code_execution', False),
                enable_file_operations=self.config.get('enable_file_operations', False),
                memory_persistence=self.config.get('memory_persistence', True),
                max_subtasks=self.config.get('max_subtasks', 20),
                iteration_delay=self.config.get('iteration_delay', 1.0),
                timeout=self.config.get('timeout', 600.0),
                max_retries=self.config.get('max_retries', 3)
            )
            
            # Create agent
            agent = AgentGPTAgent(
                agent_id=agent_id,
                config=agentgpt_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize AgentGPT agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created AgentGPT agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create AgentGPT agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute an AgentGPT agent"""
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
            mode = context.get('mode', 'goal')
            
            if mode == 'goal':
                # Create and execute goal
                goal = await agent.create_goal(prompt, context)
                result = await agent.execute_goal(goal.goal_id, context)
                response_content = result.final_result or f"Goal execution completed with {len(result.iterations)} iterations"
                
            elif mode == 'plan':
                # Create goal and plan only
                goal = await agent.create_goal(prompt, context)
                result = await agent.create_execution_plan(goal)
                response_content = f"Execution plan created with {len(result.tasks)} tasks, estimated duration: {result.estimated_total_duration:.1f} minutes"
                
            elif mode == 'continue':
                # Continue existing execution
                execution_id = context.get('execution_id')
                if execution_id and execution_id in agent._executions:
                    execution = agent._executions[execution_id]
                    if execution.status == ExecutionStatus.PAUSED:
                        await agent.resume_execution(execution_id)
                        response_content = f"Resumed execution {execution_id}"
                    else:
                        response_content = f"Execution {execution_id} status: {execution.status.value}"
                    result = execution
                else:
                    response_content = "Execution not found"
                    result = None
                    
            else:
                # Default to goal execution
                goal = await agent.create_goal(prompt, context)
                result = await agent.execute_goal(goal.goal_id, context)
                response_content = result.final_result or "Goal execution completed"
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare metadata
            metadata = {
                'execution_time_seconds': execution_time,
                'mode': mode,
                'usage_stats': agent.get_usage_stats(),
                'agent_info': agent.get_agent_info()
            }
            
            # Add result-specific metadata
            if result and hasattr(result, 'success'):
                metadata['goal_success'] = result.success
                metadata['iterations_executed'] = len(result.iterations)
                metadata['progress'] = result.progress
                metadata['total_tokens_used'] = result.total_tokens_used
                metadata['total_execution_time'] = result.total_execution_time
                
            elif result and hasattr(result, 'tasks'):
                metadata['plan_tasks'] = len(result.tasks)
                metadata['estimated_duration'] = result.estimated_total_duration
                metadata['success_probability'] = result.success_probability
                metadata['risks'] = result.risks
            
            return AgentResponse(
                content=response_content,
                agent_id=agent_id,
                provider=self.provider_name,
                metadata=metadata,
                error=result.error_log[0] if result and hasattr(result, 'error_log') and result.error_log else None
            )
            
        except Exception as e:
            self.logger.error(f"AgentGPT agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def create_goal(self, agent_id: str, description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a goal"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        goal = await agent.create_goal(description, context)
        
        return {
            'goal_id': goal.goal_id,
            'description': goal.description,
            'priority': goal.priority,
            'success_criteria': goal.success_criteria,
            'constraints': goal.constraints
        }
    
    async def execute_goal(self, agent_id: str, goal_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a goal"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        result = await agent.execute_goal(goal_id, context)
        
        return {
            'execution_id': result.execution_id,
            'success': result.success,
            'progress': result.progress,
            'iterations': len(result.iterations),
            'final_result': result.final_result,
            'total_execution_time': result.total_execution_time
        }
    
    async def get_execution_status(self, agent_id: str, execution_id: str) -> Dict[str, Any]:
        """Get execution status"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        execution = agent.get_execution(execution_id)
        
        if not execution:
            return {'error': 'Execution not found'}
        
        return {
            'execution_id': execution_id,
            'status': execution.status.value,
            'progress': execution.progress,
            'current_iteration': execution.current_iteration,
            'success': execution.success,
            'total_iterations': len(execution.iterations),
            'total_execution_time': execution.total_execution_time
        }
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with AgentGPT agent (basic implementation)"""
        if agent_id not in self._agents:
            return False
        
        try:
            # AgentGPT doesn't have native tool support in this implementation
            # This would be implemented as action capabilities
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using AgentGPT capabilities"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Create tool spec for autonomous execution
            tool_spec = ToolSpec(
                name=f"agentgpt_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={
                    'description': tool_description,
                    'execution_type': 'autonomous_goal',
                    'examples': examples or []
                },
                security_policy={
                    'risk_level': 'high',  # Autonomous execution is high risk
                    'requires_approval': True,
                    'max_iterations': self.config.get('max_iterations', 10),
                    'safety_checks': True
                }
            )
            
            self.logger.info(f"Synthesized tool for AgentGPT agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy an AgentGPT agent"""
        try:
            if agent_id in self._agents:
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed AgentGPT agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active AgentGPT agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about an AgentGPT agent"""
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