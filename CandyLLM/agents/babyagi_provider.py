"""
BabyAGI Agent Provider

Integrates BabyAGI's task-driven autonomous agent capabilities with memory and planning,
enabling intelligent task generation, prioritization, and execution with persistent memory.
"""

import uuid
import asyncio
import json
from typing import Dict, List, Optional, Any, Deque
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import heapq

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
    import chromadb
    from chromadb.config import Settings
    BABYAGI_AVAILABLE = True
except ImportError:
    BABYAGI_AVAILABLE = False
    # Mock classes for when dependencies are not available
    httpx = None
    requests = None
    OpenAI = None
    AsyncOpenAI = None
    chromadb = None
    Settings = None


class TaskStatus(Enum):
    """Status of tasks in BabyAGI"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Priority levels for tasks"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class BabyAGIConfig:
    """Configuration for BabyAGI agent"""
    api_key: str = ""
    model: str = "gpt-4"
    base_url: str = "https://api.openai.com/v1"
    objective: str = ""
    max_iterations: int = 50
    max_tasks_per_iteration: int = 3
    max_tokens: int = 2000
    temperature: float = 0.5
    enable_memory: bool = True
    memory_collection_name: str = "babyagi_memory"
    memory_persist_directory: str = "./babyagi_memory"
    task_creation_temperature: float = 0.7
    task_prioritization_temperature: float = 0.3
    task_execution_temperature: float = 0.5
    max_memory_items: int = 1000
    memory_relevance_threshold: float = 0.7
    enable_reflection: bool = True
    reflection_frequency: int = 5  # Every N iterations
    enable_task_decomposition: bool = True
    max_task_depth: int = 3
    execution_timeout: float = 300.0
    max_retries: int = 3


@dataclass
class Task:
    """Individual task in BabyAGI"""
    task_id: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    parent_task_id: Optional[str] = None
    subtask_ids: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    iteration_created: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority.value < other.priority.value


@dataclass
class Memory:
    """Memory item for BabyAGI"""
    memory_id: str
    content: str
    task_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0
    memory_type: str = "task_result"  # "task_result", "observation", "reflection"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Iteration:
    """Single iteration of BabyAGI execution"""
    iteration_number: int
    tasks_created: List[str] = field(default_factory=list)
    tasks_executed: List[str] = field(default_factory=list)
    tasks_completed: List[str] = field(default_factory=list)
    task_queue_size: int = 0
    memories_created: int = 0
    execution_time: float = 0.0
    reflection: Optional[str] = None
    objective_progress: float = 0.0


@dataclass
class BabyAGIExecution:
    """Complete BabyAGI execution session"""
    execution_id: str
    objective: str
    start_time: datetime
    end_time: Optional[datetime] = None
    iterations: List[Iteration] = field(default_factory=list)
    total_tasks_created: int = 0
    total_tasks_completed: int = 0
    final_result: Optional[str] = None
    success: bool = False
    error_log: List[str] = field(default_factory=list)


class BabyAGIAgent:
    """BabyAGI task-driven autonomous agent with memory and planning"""
    
    def __init__(self, agent_id: str, config: BabyAGIConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._client = None
        self._async_client = None
        self._task_queue = []  # Priority queue
        self._completed_tasks = {}
        self._memory_db = None
        self._memory_collection = None
        self._executions = {}
        self._current_iteration = 0
        self._usage_stats = {
            'total_executions': 0,
            'total_iterations': 0,
            'total_tasks_created': 0,
            'total_tasks_completed': 0,
            'total_memories_stored': 0,
            'total_reflections': 0,
            'total_execution_time': 0.0,
            'average_tasks_per_iteration': 0.0,
            'success_rate': 0.0,
            'task_status_counts': {status.value: 0 for status in TaskStatus},
            'task_priority_counts': {priority.value: 0 for priority in TaskPriority}
        }
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the BabyAGI agent"""
        if not BABYAGI_AVAILABLE:
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
            
            # Initialize memory database if enabled
            if self.config.enable_memory:
                await self._initialize_memory()
            
            # Test connection
            await self._test_connection()
            
            return True
            
        except Exception as e:
            return False
    
    async def _initialize_memory(self):
        """Initialize ChromaDB for memory storage"""
        try:
            # Initialize ChromaDB client
            self._memory_db = chromadb.PersistentClient(
                path=self.config.memory_persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self._memory_collection = self._memory_db.get_collection(
                    name=self.config.memory_collection_name
                )
            except:
                self._memory_collection = self._memory_db.create_collection(
                    name=self.config.memory_collection_name,
                    metadata={"description": "BabyAGI memory storage"}
                )
            
        except Exception as e:
            # Fallback to in-memory storage
            self._memory_db = None
            self._memory_collection = None
    
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
            raise RuntimeError(f"BabyAGI API test failed: {e}")
    
    async def execute_objective(self, objective: str, context: Dict[str, Any] = None) -> BabyAGIExecution:
        """Execute an objective using BabyAGI methodology"""
        try:
            execution_id = f"exec_{uuid.uuid4().hex[:8]}"
            start_time = datetime.now()
            context = context or {}
            
            # Initialize execution
            execution = BabyAGIExecution(
                execution_id=execution_id,
                objective=objective,
                start_time=start_time
            )
            
            self._executions[execution_id] = execution
            self._current_iteration = 0
            
            # Clear task queue and initialize with first task
            self._task_queue = []
            await self._initialize_first_task(objective)
            
            # Main execution loop
            for iteration in range(1, self.config.max_iterations + 1):
                iteration_start = datetime.now()
                self._current_iteration = iteration
                
                iteration_result = await self._execute_iteration(objective, iteration)
                execution.iterations.append(iteration_result)
                
                # Update execution stats
                execution.total_tasks_created += len(iteration_result.tasks_created)
                execution.total_tasks_completed += len(iteration_result.tasks_completed)
                
                # Check completion criteria
                if self._check_objective_completion(objective, execution):
                    execution.success = True
                    execution.final_result = await self._generate_final_summary(objective, execution)
                    break
                
                # Check if no more tasks
                if not self._task_queue:
                    execution.final_result = "No more tasks to execute"
                    break
                
                # Small delay between iterations
                await asyncio.sleep(0.5)
            
            execution.end_time = datetime.now()
            
            # Update usage statistics
            self._usage_stats['total_executions'] += 1
            self._usage_stats['total_iterations'] += len(execution.iterations)
            self._usage_stats['total_tasks_created'] += execution.total_tasks_created
            self._usage_stats['total_tasks_completed'] += execution.total_tasks_completed
            self._usage_stats['total_execution_time'] += (execution.end_time - execution.start_time).total_seconds()
            
            if execution.success:
                self._usage_stats['success_rate'] = (
                    self._usage_stats['success_rate'] * (self._usage_stats['total_executions'] - 1) + 1
                ) / self._usage_stats['total_executions']
            
            return execution
            
        except Exception as e:
            return BabyAGIExecution(
                execution_id=f"failed_{uuid.uuid4().hex[:8]}",
                objective=objective,
                start_time=datetime.now(),
                success=False,
                final_result=f"Execution failed: {str(e)}",
                error_log=[str(e)]
            )
    
    async def _initialize_first_task(self, objective: str):
        """Initialize the first task for the objective"""
        first_task = Task(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            description=f"Develop a task list to achieve the objective: {objective}",
            priority=TaskPriority.CRITICAL,
            iteration_created=0
        )
        
        heapq.heappush(self._task_queue, first_task)
        self._usage_stats['total_tasks_created'] += 1
        self._usage_stats['task_priority_counts'][first_task.priority.value] += 1
    
    async def _execute_iteration(self, objective: str, iteration: int) -> Iteration:
        """Execute a single iteration"""
        try:
            iteration_start = datetime.now()
            
            iteration_result = Iteration(
                iteration_number=iteration,
                task_queue_size=len(self._task_queue)
            )
            
            # Execute tasks (limited per iteration)
            tasks_to_execute = min(self.config.max_tasks_per_iteration, len(self._task_queue))
            
            for _ in range(tasks_to_execute):
                if not self._task_queue:
                    break
                
                # Get highest priority task
                task = heapq.heappop(self._task_queue)
                
                # Execute task
                task_result = await self._execute_task(task, objective)
                iteration_result.tasks_executed.append(task.task_id)
                
                if task.status == TaskStatus.COMPLETED:
                    iteration_result.tasks_completed.append(task.task_id)
                    self._completed_tasks[task.task_id] = task
                    
                    # Store result in memory
                    if self.config.enable_memory and task.result:
                        await self._store_memory(task.result, task.task_id, "task_result")
                        iteration_result.memories_created += 1
            
            # Create new tasks based on completed tasks and objective
            if iteration_result.tasks_completed:
                new_tasks = await self._create_new_tasks(objective, iteration_result.tasks_completed)
                for task in new_tasks:
                    task.iteration_created = iteration
                    heapq.heappush(self._task_queue, task)
                    iteration_result.tasks_created.append(task.task_id)
                    self._usage_stats['total_tasks_created'] += 1
                    self._usage_stats['task_priority_counts'][task.priority.value] += 1
            
            # Prioritize task list
            await self._prioritize_tasks(objective)
            
            # Reflection (periodic)
            if self.config.enable_reflection and iteration % self.config.reflection_frequency == 0:
                reflection = await self._generate_reflection(objective, iteration)
                iteration_result.reflection = reflection
                self._usage_stats['total_reflections'] += 1
                
                if self.config.enable_memory:
                    await self._store_memory(reflection, None, "reflection")
            
            # Assess objective progress
            iteration_result.objective_progress = await self._assess_objective_progress(objective)
            
            iteration_result.execution_time = (datetime.now() - iteration_start).total_seconds()
            
            return iteration_result
            
        except Exception as e:
            return Iteration(
                iteration_number=iteration,
                execution_time=0.0
            )
    
    async def _execute_task(self, task: Task, objective: str) -> str:
        """Execute a single task"""
        try:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now()
            start_time = datetime.now()
            
            # Get relevant context from memory
            context = ""
            if self.config.enable_memory:
                relevant_memories = await self._get_relevant_memories(task.description)
                if relevant_memories:
                    context = "\n".join([memory.content for memory in relevant_memories[:5]])
            
            # Build execution prompt
            execution_prompt = f"""
            You are executing a task as part of achieving this objective: {objective}
            
            Current task: {task.description}
            
            {"Relevant context from previous tasks:" + context if context else ""}
            
            Complete this task and provide a clear, actionable result.
            Be specific and focus on achieving the objective.
            """
            
            # Execute task
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": execution_prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.task_execution_temperature
            )
            
            result = response.choices[0].message.content
            
            # Update task
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.execution_time = (datetime.now() - start_time).total_seconds()
            
            self._usage_stats['total_tasks_completed'] += 1
            self._usage_stats['task_status_counts'][TaskStatus.COMPLETED.value] += 1
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            task.execution_time = (datetime.now() - start_time).total_seconds()
            
            self._usage_stats['task_status_counts'][TaskStatus.FAILED.value] += 1
            
            return f"Task failed: {str(e)}"
    
    async def _create_new_tasks(self, objective: str, completed_task_ids: List[str]) -> List[Task]:
        """Create new tasks based on completed tasks and objective"""
        try:
            # Get results from completed tasks
            completed_results = []
            for task_id in completed_task_ids:
                if task_id in self._completed_tasks:
                    task = self._completed_tasks[task_id]
                    completed_results.append(f"Task: {task.description}\nResult: {task.result}")
            
            if not completed_results:
                return []
            
            # Generate new tasks
            creation_prompt = f"""
            Based on the objective and the results of completed tasks, create new tasks that will help achieve the objective.
            
            Objective: {objective}
            
            Completed tasks and their results:
            {chr(10).join(completed_results)}
            
            Create 1-3 new specific, actionable tasks that build on these results and move closer to the objective.
            Each task should be a single, clear action.
            
            Respond with a JSON list of tasks, each with:
            - description: Clear task description
            - priority: 1 (critical), 2 (high), 3 (medium), 4 (low), 5 (background)
            
            Example:
            [
                {{"description": "Research specific topic X", "priority": 2}},
                {{"description": "Analyze data from previous research", "priority": 3}}
            ]
            """
            
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": creation_prompt}],
                max_tokens=1000,
                temperature=self.config.task_creation_temperature
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            new_tasks = []
            try:
                if '[' in response_text and ']' in response_text:
                    json_start = response_text.find('[')
                    json_end = response_text.rfind(']') + 1
                    tasks_data = json.loads(response_text[json_start:json_end])
                    
                    for task_data in tasks_data[:3]:  # Limit to 3 tasks
                        task = Task(
                            task_id=f"task_{uuid.uuid4().hex[:8]}",
                            description=task_data.get('description', ''),
                            priority=TaskPriority(task_data.get('priority', 3))
                        )
                        new_tasks.append(task)
            except:
                # Fallback: create a generic follow-up task
                task = Task(
                    task_id=f"task_{uuid.uuid4().hex[:8]}",
                    description=f"Continue working towards objective: {objective}",
                    priority=TaskPriority.MEDIUM
                )
                new_tasks.append(task)
            
            return new_tasks
            
        except Exception as e:
            return []
    
    async def _prioritize_tasks(self, objective: str):
        """Reprioritize the task queue"""
        try:
            if len(self._task_queue) <= 1:
                return
            
            # Get current tasks
            current_tasks = []
            while self._task_queue:
                current_tasks.append(heapq.heappop(self._task_queue))
            
            if not current_tasks:
                return
            
            # Build prioritization prompt
            task_list = "\n".join([
                f"{i+1}. {task.description} (Current priority: {task.priority.value})"
                for i, task in enumerate(current_tasks)
            ])
            
            prioritization_prompt = f"""
            Prioritize the following tasks based on their importance for achieving the objective.
            
            Objective: {objective}
            
            Tasks:
            {task_list}
            
            Assign priorities:
            1 = Critical (must do first)
            2 = High (important)
            3 = Medium (normal priority)
            4 = Low (can wait)
            5 = Background (nice to have)
            
            Respond with a JSON list of task numbers and their new priorities:
            [{{"task": 1, "priority": 2}}, {{"task": 2, "priority": 1}}, ...]
            """
            
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prioritization_prompt}],
                max_tokens=500,
                temperature=self.config.task_prioritization_temperature
            )
            
            response_text = response.choices[0].message.content
            
            # Parse and apply priorities
            try:
                if '[' in response_text and ']' in response_text:
                    json_start = response_text.find('[')
                    json_end = response_text.rfind(']') + 1
                    priorities_data = json.loads(response_text[json_start:json_end])
                    
                    for priority_item in priorities_data:
                        task_num = priority_item.get('task', 0) - 1
                        new_priority = priority_item.get('priority', 3)
                        
                        if 0 <= task_num < len(current_tasks):
                            current_tasks[task_num].priority = TaskPriority(new_priority)
            except:
                pass  # Keep existing priorities if parsing fails
            
            # Rebuild priority queue
            for task in current_tasks:
                heapq.heappush(self._task_queue, task)
            
        except Exception as e:
            # Put tasks back in queue if prioritization fails
            for task in current_tasks:
                heapq.heappush(self._task_queue, task)
    
    async def _store_memory(self, content: str, task_id: Optional[str] = None, memory_type: str = "task_result"):
        """Store content in memory"""
        try:
            if not self.config.enable_memory or not self._memory_collection:
                return
            
            memory_id = f"memory_{uuid.uuid4().hex[:8]}"
            
            # Add to ChromaDB
            self._memory_collection.add(
                ids=[memory_id],
                documents=[content],
                metadatas=[{
                    "task_id": task_id or "",
                    "memory_type": memory_type,
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": self.agent_id
                }]
            )
            
            self._usage_stats['total_memories_stored'] += 1
            
        except Exception as e:
            pass  # Fail silently if memory storage fails
    
    async def _get_relevant_memories(self, query: str, n_results: int = 5) -> List[Memory]:
        """Retrieve relevant memories for a query"""
        try:
            if not self.config.enable_memory or not self._memory_collection:
                return []
            
            # Query ChromaDB
            results = self._memory_collection.query(
                query_texts=[query],
                n_results=min(n_results, 10)
            )
            
            memories = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 1.0
                    
                    # Only include memories above relevance threshold
                    relevance_score = 1.0 - distance
                    if relevance_score >= self.config.memory_relevance_threshold:
                        memory = Memory(
                            memory_id=results['ids'][0][i],
                            content=doc,
                            task_id=metadata.get('task_id'),
                            relevance_score=relevance_score,
                            memory_type=metadata.get('memory_type', 'task_result'),
                            metadata=metadata
                        )
                        memories.append(memory)
            
            return memories
            
        except Exception as e:
            return []
    
    async def _generate_reflection(self, objective: str, iteration: int) -> str:
        """Generate reflection on progress"""
        try:
            completed_tasks = list(self._completed_tasks.values())[-10:]  # Last 10 completed tasks
            
            reflection_prompt = f"""
            Reflect on the progress towards the objective after {iteration} iterations.
            
            Objective: {objective}
            Current iteration: {iteration}
            
            Recent completed tasks:
            {chr(10).join([f"- {task.description}: {task.result[:100]}..." for task in completed_tasks])}
            
            Pending tasks: {len(self._task_queue)}
            
            Provide a brief reflection on:
            1. Progress towards the objective
            2. What's working well
            3. What could be improved
            4. Recommended next steps
            
            Keep it concise but insightful.
            """
            
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": reflection_prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Reflection generation failed: {str(e)}"
    
    async def _assess_objective_progress(self, objective: str) -> float:
        """Assess progress towards objective (0.0 to 1.0)"""
        try:
            if not self._completed_tasks:
                return 0.0
            
            completed_results = [task.result for task in self._completed_tasks.values() if task.result]
            
            if not completed_results:
                return 0.0
            
            assessment_prompt = f"""
            Assess the progress towards achieving this objective based on completed tasks.
            
            Objective: {objective}
            
            Completed task results:
            {chr(10).join(completed_results[-5:])}  # Last 5 results
            
            On a scale of 0.0 to 1.0, how much progress has been made towards the objective?
            0.0 = No progress
            0.5 = Halfway there
            1.0 = Objective fully achieved
            
            Respond with just a number between 0.0 and 1.0
            """
            
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": assessment_prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract number
            try:
                progress = float(response_text)
                return max(0.0, min(1.0, progress))
            except:
                # Fallback based on task completion ratio
                return min(1.0, len(self._completed_tasks) / 10.0)
            
        except Exception as e:
            return 0.0
    
    def _check_objective_completion(self, objective: str, execution: BabyAGIExecution) -> bool:
        """Check if objective is completed"""
        if not execution.iterations:
            return False
        
        latest_iteration = execution.iterations[-1]
        
        # Check progress threshold
        if latest_iteration.objective_progress >= 0.95:
            return True
        
        # Check if we have enough completed tasks and recent progress is minimal
        if (execution.total_tasks_completed >= 10 and 
            len(execution.iterations) >= 5 and
            latest_iteration.objective_progress > 0.8):
            return True
        
        return False
    
    async def _generate_final_summary(self, objective: str, execution: BabyAGIExecution) -> str:
        """Generate final summary of execution"""
        try:
            completed_tasks = [task.result for task in self._completed_tasks.values() if task.result]
            
            summary_prompt = f"""
            Generate a final summary of the work completed towards this objective.
            
            Objective: {objective}
            
            Iterations completed: {len(execution.iterations)}
            Tasks completed: {execution.total_tasks_completed}
            
            Key results:
            {chr(10).join(completed_tasks[-5:])}  # Last 5 results
            
            Provide a comprehensive summary of what was accomplished and how it addresses the objective.
            """
            
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Summary generation failed. {execution.total_tasks_completed} tasks completed across {len(execution.iterations)} iterations."
    
    def get_execution(self, execution_id: str) -> Optional[BabyAGIExecution]:
        """Get execution by ID"""
        return self._executions.get(execution_id)
    
    def list_executions(self) -> List[Dict[str, Any]]:
        """List all executions"""
        return [
            {
                'execution_id': execution.execution_id,
                'objective': execution.objective,
                'success': execution.success,
                'tasks_completed': execution.total_tasks_completed,
                'iterations': len(execution.iterations),
                'start_time': execution.start_time.isoformat()
            }
            for execution in self._executions.values()
        ]
    
    def get_current_tasks(self) -> List[Dict[str, Any]]:
        """Get current task queue"""
        return [
            {
                'task_id': task.task_id,
                'description': task.description,
                'priority': task.priority.value,
                'status': task.status.value,
                'created_at': task.created_at.isoformat()
            }
            for task in self._task_queue
        ]
    
    def get_completed_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get completed tasks"""
        completed = list(self._completed_tasks.values())[-limit:]
        return [
            {
                'task_id': task.task_id,
                'description': task.description,
                'result': task.result,
                'execution_time': task.execution_time,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None
            }
            for task in completed
        ]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = self._usage_stats.copy()
        
        # Calculate averages
        if stats['total_iterations'] > 0:
            stats['average_tasks_per_iteration'] = stats['total_tasks_created'] / stats['total_iterations']
        
        return stats
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': {
                'model': self.config.model,
                'max_iterations': self.config.max_iterations,
                'enable_memory': self.config.enable_memory,
                'enable_reflection': self.config.enable_reflection,
                'max_tasks_per_iteration': self.config.max_tasks_per_iteration
            },
            'current_iteration': self._current_iteration,
            'task_queue_size': len(self._task_queue),
            'completed_tasks_count': len(self._completed_tasks),
            'executions_count': len(self._executions),
            'memory_enabled': self._memory_collection is not None,
            'usage_stats': self.get_usage_stats(),
            'babyagi_available': BABYAGI_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class BabyAGIProvider(BaseAgentProvider):
    """
    Provider implementation for BabyAGI.
    
    Enables task-driven autonomous agents with memory and planning capabilities
    for intelligent task generation, prioritization, and execution.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, BabyAGIAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not BABYAGI_AVAILABLE:
            self.logger.warning("BabyAGI dependencies not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "babyagi"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.AUTONOMOUS_PLANNING,
            AgentCapability.TASK_DECOMPOSITION,
            AgentCapability.MEMORY_MANAGEMENT,
            AgentCapability.ITERATIVE_IMPROVEMENT,
            AgentCapability.PRIORITY_MANAGEMENT,
            AgentCapability.REFLECTIVE_REASONING
        ]
    
    async def initialize(self) -> bool:
        """Initialize BabyAGI provider"""
        if not BABYAGI_AVAILABLE:
            self.logger.error("BabyAGI dependencies not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("BabyAGI provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize BabyAGI provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new BabyAGI agent"""
        if not self._initialized:
            await self.initialize()
        
        if not BABYAGI_AVAILABLE:
            raise RuntimeError("BabyAGI dependencies not available")
        
        agent_id = f"babyagi_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create BabyAGI configuration
            babyagi_config = BabyAGIConfig(
                api_key=self.config.get('api_key', ''),
                model=self.config.get('model', 'gpt-4'),
                base_url=self.config.get('base_url', 'https://api.openai.com/v1'),
                objective=self.config.get('objective', ''),
                max_iterations=self.config.get('max_iterations', 50),
                max_tasks_per_iteration=self.config.get('max_tasks_per_iteration', 3),
                max_tokens=self.config.get('max_tokens', 2000),
                temperature=self.config.get('temperature', 0.5),
                enable_memory=self.config.get('enable_memory', True),
                memory_collection_name=self.config.get('memory_collection_name', f'babyagi_memory_{agent_id}'),
                memory_persist_directory=self.config.get('memory_persist_directory', f'./babyagi_memory_{agent_id}'),
                task_creation_temperature=self.config.get('task_creation_temperature', 0.7),
                task_prioritization_temperature=self.config.get('task_prioritization_temperature', 0.3),
                task_execution_temperature=self.config.get('task_execution_temperature', 0.5),
                max_memory_items=self.config.get('max_memory_items', 1000),
                memory_relevance_threshold=self.config.get('memory_relevance_threshold', 0.7),
                enable_reflection=self.config.get('enable_reflection', True),
                reflection_frequency=self.config.get('reflection_frequency', 5),
                enable_task_decomposition=self.config.get('enable_task_decomposition', True),
                max_task_depth=self.config.get('max_task_depth', 3),
                execution_timeout=self.config.get('execution_timeout', 300.0),
                max_retries=self.config.get('max_retries', 3)
            )
            
            # Create agent
            agent = BabyAGIAgent(
                agent_id=agent_id,
                config=babyagi_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize BabyAGI agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created BabyAGI agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create BabyAGI agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a BabyAGI agent"""
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
            
            # Execute objective
            result = await agent.execute_objective(prompt, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare response
            response_content = result.final_result or f"Objective execution completed with {result.total_tasks_completed} tasks across {len(result.iterations)} iterations"
            
            # Prepare metadata
            metadata = {
                'execution_time_seconds': execution_time,
                'execution_id': result.execution_id,
                'success': result.success,
                'total_tasks_created': result.total_tasks_created,
                'total_tasks_completed': result.total_tasks_completed,
                'iterations': len(result.iterations),
                'final_progress': result.iterations[-1].objective_progress if result.iterations else 0.0,
                'usage_stats': agent.get_usage_stats(),
                'agent_info': agent.get_agent_info()
            }
            
            # Add iteration details
            if result.iterations:
                metadata['iteration_summary'] = [
                    {
                        'iteration': iter_result.iteration_number,
                        'tasks_executed': len(iter_result.tasks_executed),
                        'tasks_created': len(iter_result.tasks_created),
                        'progress': iter_result.objective_progress,
                        'reflection': iter_result.reflection is not None
                    }
                    for iter_result in result.iterations[-5:]  # Last 5 iterations
                ]
            
            return AgentResponse(
                content=response_content,
                agent_id=agent_id,
                provider=self.provider_name,
                metadata=metadata,
                error=result.error_log[0] if result.error_log else None
            )
            
        except Exception as e:
            self.logger.error(f"BabyAGI agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def execute_objective(self, agent_id: str, objective: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute an objective"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        result = await agent.execute_objective(objective, context)
        
        return {
            'execution_id': result.execution_id,
            'success': result.success,
            'total_tasks_created': result.total_tasks_created,
            'total_tasks_completed': result.total_tasks_completed,
            'iterations': len(result.iterations),
            'final_result': result.final_result
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
            'objective': execution.objective,
            'success': execution.success,
            'tasks_completed': execution.total_tasks_completed,
            'iterations': len(execution.iterations),
            'start_time': execution.start_time.isoformat(),
            'end_time': execution.end_time.isoformat() if execution.end_time else None
        }
    
    async def get_current_tasks(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get current task queue"""
        if agent_id not in self._agents:
            return []
        
        agent = self._agents[agent_id]
        return agent.get_current_tasks()
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with BabyAGI agent (basic implementation)"""
        if agent_id not in self._agents:
            return False
        
        try:
            # BabyAGI doesn't have native tool support in this implementation
            # This would be implemented as task capabilities
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using BabyAGI capabilities"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Create tool spec for task-driven execution
            tool_spec = ToolSpec(
                name=f"babyagi_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={
                    'description': tool_description,
                    'execution_type': 'task_driven',
                    'examples': examples or []
                },
                security_policy={
                    'risk_level': 'medium',
                    'requires_approval': True,
                    'max_iterations': self.config.get('max_iterations', 50),
                    'memory_enabled': True
                }
            )
            
            self.logger.info(f"Synthesized tool for BabyAGI agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a BabyAGI agent"""
        try:
            if agent_id in self._agents:
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed BabyAGI agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active BabyAGI agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a BabyAGI agent"""
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