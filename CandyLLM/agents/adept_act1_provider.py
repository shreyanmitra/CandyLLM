"""
Adept ACT-1 Agent Provider

Integrates Adept's ACT-1 model for computer interaction and automation,
enabling web and desktop task execution through visual understanding and action planning.
"""

import uuid
import asyncio
import json
import base64
from typing import Dict, List, Optional, Any, Union, Tuple
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
    from PIL import Image
    import io
    ADEPT_AVAILABLE = True
except ImportError:
    ADEPT_AVAILABLE = False
    # Mock classes for when dependencies are not available
    httpx = None
    requests = None
    Image = None
    io = None


class ActionType(Enum):
    """Types of actions ACT-1 can perform"""
    CLICK = "click"
    TYPE = "type"
    KEY_PRESS = "key_press"
    SCROLL = "scroll"
    DRAG = "drag"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    NAVIGATE = "navigate"
    UPLOAD_FILE = "upload_file"
    DOWNLOAD_FILE = "download_file"
    FORM_FILL = "form_fill"
    MULTI_STEP = "multi_step"


class InteractionMode(Enum):
    """Interaction modes for ACT-1"""
    WEB_BROWSER = "web_browser"
    DESKTOP_APP = "desktop_app"
    MOBILE_APP = "mobile_app"
    API_INTERFACE = "api_interface"
    COMMAND_LINE = "command_line"


@dataclass
class AdeptConfig:
    """Configuration for Adept ACT-1 agent"""
    api_key: str = ""
    model: str = "act-1"
    base_url: str = "https://api.adept.ai"
    max_steps: int = 50
    screenshot_quality: str = "high"  # "low", "medium", "high"
    wait_timeout: float = 30.0
    action_delay: float = 1.0  # Delay between actions
    browser_type: str = "chrome"  # "chrome", "firefox", "safari"
    screen_resolution: Tuple[int, int] = (1920, 1080)
    enable_vision: bool = True
    enable_planning: bool = True
    enable_error_recovery: bool = True
    safety_mode: str = "strict"  # "strict", "moderate", "permissive"
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    timeout: float = 300.0
    max_retries: int = 3


@dataclass
class ActionResult:
    """Result of an action execution"""
    action_type: ActionType
    success: bool
    confidence: float
    error_message: Optional[str] = None
    screenshot_before: Optional[str] = None  # Base64 encoded
    screenshot_after: Optional[str] = None   # Base64 encoded
    elements_detected: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    coordinates: Optional[Tuple[int, int]] = None
    text_input: Optional[str] = None


@dataclass
class TaskPlan:
    """Plan for completing a complex task"""
    task_description: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    estimated_duration: float = 0.0
    required_permissions: List[str] = field(default_factory=list)
    safety_warnings: List[str] = field(default_factory=list)
    fallback_strategies: List[str] = field(default_factory=list)


@dataclass
class ScreenContext:
    """Context about current screen state"""
    screenshot: Optional[str] = None  # Base64 encoded
    url: Optional[str] = None
    title: Optional[str] = None
    elements: List[Dict[str, Any]] = field(default_factory=list)
    text_content: Optional[str] = None
    interaction_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    detected_forms: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TaskExecution:
    """Complete task execution with results"""
    task_id: str
    task_description: str
    plan: TaskPlan
    actions_executed: List[ActionResult] = field(default_factory=list)
    final_result: Optional[str] = None
    success: bool = False
    total_duration: float = 0.0
    screenshots: List[str] = field(default_factory=list)
    error_log: List[str] = field(default_factory=list)


class AdeptACT1Agent:
    """Adept ACT-1 agent for computer interaction and automation"""
    
    def __init__(self, agent_id: str, config: AdeptConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._client = None
        self._active_sessions = {}
        self._task_history = []
        self._usage_stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'total_actions': 0,
            'successful_actions': 0,
            'screenshots_taken': 0,
            'interaction_modes_used': {mode.value: 0 for mode in InteractionMode},
            'action_types_executed': {action.value: 0 for action in ActionType},
            'domains_accessed': set(),
            'total_execution_time': 0.0,
            'error_recovery_attempts': 0
        }
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the ACT-1 agent"""
        if not ADEPT_AVAILABLE:
            return False
        
        try:
            if not self.config.api_key:
                return False
            
            # Initialize HTTP client
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=self.config.timeout
            )
            
            # Test connection
            await self._test_connection()
            
            return True
            
        except Exception as e:
            return False
    
    async def _test_connection(self):
        """Test API connection"""
        try:
            response = await self._client.get("/health")
            if response.status_code != 200:
                raise RuntimeError(f"API health check failed: {response.status_code}")
            return True
        except Exception as e:
            raise RuntimeError(f"ACT-1 API test failed: {e}")
    
    async def take_screenshot(self, session_id: Optional[str] = None) -> str:
        """Take screenshot of current screen"""
        try:
            payload = {
                "action": "screenshot",
                "session_id": session_id,
                "quality": self.config.screenshot_quality
            }
            
            response = await self._client.post("/v1/actions/screenshot", json=payload)
            response.raise_for_status()
            
            data = response.json()
            screenshot_b64 = data.get("screenshot", "")
            
            self._usage_stats['screenshots_taken'] += 1
            
            return screenshot_b64
            
        except Exception as e:
            raise RuntimeError(f"Screenshot failed: {e}")
    
    async def analyze_screen(self, screenshot: Optional[str] = None,
                           session_id: Optional[str] = None) -> ScreenContext:
        """Analyze current screen content and interactions"""
        try:
            if not screenshot:
                screenshot = await self.take_screenshot(session_id)
            
            payload = {
                "action": "analyze_screen",
                "screenshot": screenshot,
                "session_id": session_id,
                "enable_vision": self.config.enable_vision,
                "detect_elements": True,
                "extract_text": True,
                "find_interactions": True
            }
            
            response = await self._client.post("/v1/vision/analyze", json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            return ScreenContext(
                screenshot=screenshot,
                url=data.get("url"),
                title=data.get("title"),
                elements=data.get("elements", []),
                text_content=data.get("text_content"),
                interaction_opportunities=data.get("interaction_opportunities", []),
                detected_forms=data.get("detected_forms", [])
            )
            
        except Exception as e:
            return ScreenContext(
                screenshot=screenshot,
                elements=[],
                interaction_opportunities=[],
                detected_forms=[]
            )
    
    async def plan_task(self, task_description: str, context: Dict[str, Any] = None) -> TaskPlan:
        """Create execution plan for a task"""
        try:
            context = context or {}
            
            # Get current screen context if available
            screen_context = None
            if context.get('session_id'):
                screen_context = await self.analyze_screen(session_id=context['session_id'])
            
            payload = {
                "task_description": task_description,
                "screen_context": screen_context.__dict__ if screen_context else None,
                "interaction_mode": context.get('interaction_mode', InteractionMode.WEB_BROWSER.value),
                "max_steps": context.get('max_steps', self.config.max_steps),
                "safety_mode": self.config.safety_mode,
                "allowed_domains": self.config.allowed_domains,
                "blocked_domains": self.config.blocked_domains
            }
            
            response = await self._client.post("/v1/planning/create", json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            return TaskPlan(
                task_description=task_description,
                steps=data.get("steps", []),
                estimated_duration=data.get("estimated_duration", 0.0),
                required_permissions=data.get("required_permissions", []),
                safety_warnings=data.get("safety_warnings", []),
                fallback_strategies=data.get("fallback_strategies", [])
            )
            
        except Exception as e:
            # Return basic plan
            return TaskPlan(
                task_description=task_description,
                steps=[{"action": "error", "description": f"Planning failed: {e}"}],
                estimated_duration=0.0,
                safety_warnings=["Planning failed - manual execution required"]
            )
    
    async def execute_action(self, action_type: ActionType, parameters: Dict[str, Any],
                           session_id: Optional[str] = None) -> ActionResult:
        """Execute a single action"""
        try:
            start_time = datetime.now()
            
            # Take screenshot before action
            screenshot_before = await self.take_screenshot(session_id)
            
            payload = {
                "action": action_type.value,
                "parameters": parameters,
                "session_id": session_id,
                "safety_mode": self.config.safety_mode,
                "action_delay": self.config.action_delay
            }
            
            response = await self._client.post("/v1/actions/execute", json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # Take screenshot after action
            screenshot_after = await self.take_screenshot(session_id)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update usage statistics
            self._usage_stats['total_actions'] += 1
            self._usage_stats['action_types_executed'][action_type.value] += 1
            if data.get("success", False):
                self._usage_stats['successful_actions'] += 1
            
            return ActionResult(
                action_type=action_type,
                success=data.get("success", False),
                confidence=data.get("confidence", 0.0),
                error_message=data.get("error_message"),
                screenshot_before=screenshot_before,
                screenshot_after=screenshot_after,
                elements_detected=data.get("elements_detected", []),
                execution_time=execution_time,
                coordinates=tuple(data.get("coordinates", [])) if data.get("coordinates") else None,
                text_input=parameters.get("text")
            )
            
        except Exception as e:
            return ActionResult(
                action_type=action_type,
                success=False,
                confidence=0.0,
                error_message=str(e),
                execution_time=0.0
            )
    
    async def execute_task(self, task_description: str, context: Dict[str, Any] = None) -> TaskExecution:
        """Execute a complete task with planning and action execution"""
        try:
            context = context or {}
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            start_time = datetime.now()
            
            # Create task plan
            plan = await self.plan_task(task_description, context)
            
            # Initialize execution tracking
            execution = TaskExecution(
                task_id=task_id,
                task_description=task_description,
                plan=plan,
                actions_executed=[],
                screenshots=[],
                error_log=[]
            )
            
            session_id = context.get('session_id', f"session_{uuid.uuid4().hex[:8]}")
            
            # Execute each step in the plan
            for i, step in enumerate(plan.steps):
                try:
                    action_type = ActionType(step.get("action", "wait"))
                    parameters = step.get("parameters", {})
                    
                    # Execute action
                    result = await self.execute_action(action_type, parameters, session_id)
                    execution.actions_executed.append(result)
                    
                    if result.screenshot_after:
                        execution.screenshots.append(result.screenshot_after)
                    
                    # Check for errors
                    if not result.success:
                        error_msg = f"Step {i+1} failed: {result.error_message}"
                        execution.error_log.append(error_msg)
                        
                        if self.config.enable_error_recovery:
                            self._usage_stats['error_recovery_attempts'] += 1
                            # Attempt error recovery (simplified)
                            await asyncio.sleep(self.config.action_delay * 2)
                            continue
                        else:
                            break
                    
                    # Wait between actions
                    if i < len(plan.steps) - 1:
                        await asyncio.sleep(self.config.action_delay)
                        
                except Exception as e:
                    error_msg = f"Step {i+1} exception: {str(e)}"
                    execution.error_log.append(error_msg)
                    break
            
            # Determine overall success
            successful_actions = sum(1 for action in execution.actions_executed if action.success)
            execution.success = successful_actions >= len(plan.steps) * 0.8  # 80% success threshold
            
            # Calculate total duration
            execution.total_duration = (datetime.now() - start_time).total_seconds()
            
            # Generate final result summary
            if execution.success:
                execution.final_result = f"Task completed successfully. {successful_actions}/{len(plan.steps)} steps executed."
            else:
                execution.final_result = f"Task failed. {successful_actions}/{len(plan.steps)} steps executed. Errors: {'; '.join(execution.error_log)}"
            
            # Update usage statistics
            self._usage_stats['total_tasks'] += 1
            if execution.success:
                self._usage_stats['successful_tasks'] += 1
            self._usage_stats['total_execution_time'] += execution.total_duration
            
            # Track domain access
            if context.get('url'):
                domain = context['url'].split('/')[2] if '/' in context['url'] else context['url']
                self._usage_stats['domains_accessed'].add(domain)
            
            # Store task history
            self._task_history.append(execution)
            
            return execution
            
        except Exception as e:
            return TaskExecution(
                task_id=task_id,
                task_description=task_description,
                plan=TaskPlan(task_description=task_description),
                success=False,
                final_result=f"Task execution failed: {str(e)}",
                error_log=[str(e)]
            )
    
    async def click_element(self, coordinates: Tuple[int, int], session_id: Optional[str] = None) -> ActionResult:
        """Click at specific coordinates"""
        return await self.execute_action(
            ActionType.CLICK,
            {"coordinates": coordinates},
            session_id
        )
    
    async def type_text(self, text: str, session_id: Optional[str] = None) -> ActionResult:
        """Type text at current cursor position"""
        return await self.execute_action(
            ActionType.TYPE,
            {"text": text},
            session_id
        )
    
    async def press_key(self, key: str, session_id: Optional[str] = None) -> ActionResult:
        """Press a specific key"""
        return await self.execute_action(
            ActionType.KEY_PRESS,
            {"key": key},
            session_id
        )
    
    async def scroll_page(self, direction: str, amount: int = 3,
                         session_id: Optional[str] = None) -> ActionResult:
        """Scroll page in specified direction"""
        return await self.execute_action(
            ActionType.SCROLL,
            {"direction": direction, "amount": amount},
            session_id
        )
    
    async def navigate_to_url(self, url: str, session_id: Optional[str] = None) -> ActionResult:
        """Navigate to a specific URL"""
        # Security check for allowed/blocked domains
        if self.config.allowed_domains:
            domain = url.split('/')[2] if '/' in url else url
            if not any(allowed in domain for allowed in self.config.allowed_domains):
                return ActionResult(
                    action_type=ActionType.NAVIGATE,
                    success=False,
                    confidence=0.0,
                    error_message=f"Domain {domain} not in allowed domains"
                )
        
        if self.config.blocked_domains:
            domain = url.split('/')[2] if '/' in url else url
            if any(blocked in domain for blocked in self.config.blocked_domains):
                return ActionResult(
                    action_type=ActionType.NAVIGATE,
                    success=False,
                    confidence=0.0,
                    error_message=f"Domain {domain} is blocked"
                )
        
        return await self.execute_action(
            ActionType.NAVIGATE,
            {"url": url},
            session_id
        )
    
    async def fill_form(self, form_data: Dict[str, str], session_id: Optional[str] = None) -> ActionResult:
        """Fill a form with provided data"""
        return await self.execute_action(
            ActionType.FORM_FILL,
            {"form_data": form_data},
            session_id
        )
    
    async def wait_for_element(self, element_selector: str, timeout: float = None,
                             session_id: Optional[str] = None) -> ActionResult:
        """Wait for an element to appear"""
        return await self.execute_action(
            ActionType.WAIT,
            {
                "element_selector": element_selector,
                "timeout": timeout or self.config.wait_timeout
            },
            session_id
        )
    
    async def automate_workflow(self, workflow_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Automate a complex workflow"""
        try:
            context = context or {}
            
            # Break down workflow into tasks
            workflow_tasks = await self._decompose_workflow(workflow_description, context)
            
            workflow_results = []
            overall_success = True
            
            for task_desc in workflow_tasks:
                task_result = await self.execute_task(task_desc, context)
                workflow_results.append(task_result)
                
                if not task_result.success:
                    overall_success = False
                    if not self.config.enable_error_recovery:
                        break
            
            return {
                'workflow_description': workflow_description,
                'tasks_executed': workflow_tasks,
                'task_results': [result.__dict__ for result in workflow_results],
                'overall_success': overall_success,
                'total_tasks': len(workflow_tasks),
                'successful_tasks': sum(1 for result in workflow_results if result.success)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _decompose_workflow(self, workflow_description: str, context: Dict[str, Any]) -> List[str]:
        """Decompose workflow into individual tasks"""
        try:
            payload = {
                "workflow_description": workflow_description,
                "context": context,
                "max_tasks": 10
            }
            
            response = await self._client.post("/v1/planning/decompose", json=payload)
            response.raise_for_status()
            
            data = response.json()
            return data.get("tasks", [workflow_description])
            
        except Exception:
            # Fallback to simple decomposition
            return [workflow_description]
    
    def get_task_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent task execution history"""
        return [
            {
                'task_id': task.task_id,
                'description': task.task_description,
                'success': task.success,
                'duration': task.total_duration,
                'actions_count': len(task.actions_executed),
                'final_result': task.final_result
            }
            for task in self._task_history[-limit:]
        ]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = self._usage_stats.copy()
        stats['domains_accessed'] = list(stats['domains_accessed'])  # Convert set to list
        
        # Calculate success rates
        if stats['total_tasks'] > 0:
            stats['task_success_rate'] = stats['successful_tasks'] / stats['total_tasks']
        else:
            stats['task_success_rate'] = 0.0
        
        if stats['total_actions'] > 0:
            stats['action_success_rate'] = stats['successful_actions'] / stats['total_actions']
        else:
            stats['action_success_rate'] = 0.0
        
        return stats
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': {
                'model': self.config.model,
                'max_steps': self.config.max_steps,
                'browser_type': self.config.browser_type,
                'screen_resolution': self.config.screen_resolution,
                'safety_mode': self.config.safety_mode,
                'enable_vision': self.config.enable_vision,
                'enable_planning': self.config.enable_planning
            },
            'active_sessions': len(self._active_sessions),
            'task_history_count': len(self._task_history),
            'usage_stats': self.get_usage_stats(),
            'adept_available': ADEPT_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class AdeptACT1Provider(BaseAgentProvider):
    """
    Provider implementation for Adept ACT-1.
    
    Enables computer interaction and automation through visual understanding,
    action planning, and execution for web and desktop applications.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, AdeptACT1Agent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not ADEPT_AVAILABLE:
            self.logger.warning("Adept ACT-1 dependencies not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "adept_act1"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.COMPUTER_VISION,
            AgentCapability.WEB_AUTOMATION,
            AgentCapability.DESKTOP_AUTOMATION,
            AgentCapability.ACTION_PLANNING,
            AgentCapability.VISUAL_REASONING,
            AgentCapability.TASK_EXECUTION,
            AgentCapability.SCREEN_INTERACTION
        ]
    
    async def initialize(self) -> bool:
        """Initialize Adept ACT-1 provider"""
        if not ADEPT_AVAILABLE:
            self.logger.error("Adept ACT-1 dependencies not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Adept ACT-1 provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Adept ACT-1 provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Adept ACT-1 agent"""
        if not self._initialized:
            await self.initialize()
        
        if not ADEPT_AVAILABLE:
            raise RuntimeError("Adept ACT-1 dependencies not available")
        
        agent_id = f"act1_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create Adept configuration
            adept_config = AdeptConfig(
                api_key=self.config.get('api_key', ''),
                model=self.config.get('model', 'act-1'),
                base_url=self.config.get('base_url', 'https://api.adept.ai'),
                max_steps=self.config.get('max_steps', 50),
                screenshot_quality=self.config.get('screenshot_quality', 'high'),
                wait_timeout=self.config.get('wait_timeout', 30.0),
                action_delay=self.config.get('action_delay', 1.0),
                browser_type=self.config.get('browser_type', 'chrome'),
                screen_resolution=tuple(self.config.get('screen_resolution', [1920, 1080])),
                enable_vision=self.config.get('enable_vision', True),
                enable_planning=self.config.get('enable_planning', True),
                enable_error_recovery=self.config.get('enable_error_recovery', True),
                safety_mode=self.config.get('safety_mode', 'strict'),
                allowed_domains=self.config.get('allowed_domains', []),
                blocked_domains=self.config.get('blocked_domains', []),
                timeout=self.config.get('timeout', 300.0),
                max_retries=self.config.get('max_retries', 3)
            )
            
            # Create agent
            agent = AdeptACT1Agent(
                agent_id=agent_id,
                config=adept_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Adept ACT-1 agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Adept ACT-1 agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Adept ACT-1 agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute an Adept ACT-1 agent"""
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
            mode = context.get('mode', 'task')
            
            if mode == 'task':
                # Execute complete task
                result = await agent.execute_task(prompt, context)
                response_content = result.final_result
                
            elif mode == 'workflow':
                # Execute workflow
                result = await agent.automate_workflow(prompt, context)
                response_content = f"Workflow executed: {result.get('successful_tasks', 0)}/{result.get('total_tasks', 0)} tasks successful"
                
            elif mode == 'screenshot':
                # Take screenshot
                screenshot = await agent.take_screenshot(context.get('session_id'))
                result = {'screenshot': screenshot}
                response_content = "Screenshot captured successfully"
                
            elif mode == 'analyze':
                # Analyze screen
                result = await agent.analyze_screen(session_id=context.get('session_id'))
                response_content = f"Screen analysis complete. Found {len(result.elements)} elements, {len(result.interaction_opportunities)} interaction opportunities."
                
            elif mode == 'plan':
                # Create plan only
                result = await agent.plan_task(prompt, context)
                response_content = f"Task plan created with {len(result.steps)} steps, estimated duration: {result.estimated_duration:.1f}s"
                
            else:
                # Default to task execution
                result = await agent.execute_task(prompt, context)
                response_content = result.final_result
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare metadata
            metadata = {
                'execution_time_seconds': execution_time,
                'mode': mode,
                'usage_stats': agent.get_usage_stats(),
                'agent_info': agent.get_agent_info()
            }
            
            # Add mode-specific metadata
            if mode == 'task' and hasattr(result, 'success'):
                metadata['task_success'] = result.success
                metadata['actions_executed'] = len(result.actions_executed)
                metadata['screenshots_count'] = len(result.screenshots)
                metadata['error_count'] = len(result.error_log)
                metadata['total_duration'] = result.total_duration
                
            elif mode == 'workflow' and isinstance(result, dict):
                metadata['workflow_success'] = result.get('overall_success', False)
                metadata['tasks_executed'] = result.get('total_tasks', 0)
                metadata['successful_tasks'] = result.get('successful_tasks', 0)
                
            elif mode == 'analyze' and hasattr(result, 'elements'):
                metadata['elements_found'] = len(result.elements)
                metadata['interactions_available'] = len(result.interaction_opportunities)
                metadata['forms_detected'] = len(result.detected_forms)
                metadata['url'] = result.url
                metadata['title'] = result.title
                
            elif mode == 'plan' and hasattr(result, 'steps'):
                metadata['plan_steps'] = len(result.steps)
                metadata['estimated_duration'] = result.estimated_duration
                metadata['safety_warnings'] = result.safety_warnings
                metadata['required_permissions'] = result.required_permissions
            
            return AgentResponse(
                content=response_content,
                agent_id=agent_id,
                provider=self.provider_name,
                metadata=metadata,
                error=result.get('error') if isinstance(result, dict) else None
            )
            
        except Exception as e:
            self.logger.error(f"Adept ACT-1 agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def take_screenshot(self, agent_id: str, session_id: Optional[str] = None) -> str:
        """Take screenshot using agent"""
        if agent_id not in self._agents:
            return ""
        
        agent = self._agents[agent_id]
        return await agent.take_screenshot(session_id)
    
    async def execute_action(self, agent_id: str, action_type: str, parameters: Dict[str, Any],
                           session_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute specific action"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        try:
            agent = self._agents[agent_id]
            action_enum = ActionType(action_type)
            result = await agent.execute_action(action_enum, parameters, session_id)
            
            return {
                'success': result.success,
                'confidence': result.confidence,
                'error_message': result.error_message,
                'execution_time': result.execution_time,
                'coordinates': result.coordinates,
                'elements_detected': result.elements_detected
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def automate_workflow(self, agent_id: str, workflow_description: str,
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute workflow automation"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.automate_workflow(workflow_description, context)
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Adept ACT-1 agent (basic implementation)"""
        if agent_id not in self._agents:
            return False
        
        try:
            # ACT-1 doesn't have native tool support
            # This would be implemented as action sequences
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Adept ACT-1 capabilities"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Create tool spec for computer interaction
            tool_spec = ToolSpec(
                name=f"act1_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={
                    'description': tool_description,
                    'interaction_type': 'computer_automation',
                    'examples': examples or []
                },
                security_policy={
                    'risk_level': 'high',  # Computer automation is high risk
                    'requires_approval': True,
                    'allowed_domains': self.config.get('allowed_domains', []),
                    'blocked_domains': self.config.get('blocked_domains', [])
                }
            )
            
            self.logger.info(f"Synthesized tool for Adept ACT-1 agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy an Adept ACT-1 agent"""
        try:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                # Close HTTP client
                if agent._client:
                    await agent._client.aclose()
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Adept ACT-1 agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Adept ACT-1 agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about an Adept ACT-1 agent"""
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