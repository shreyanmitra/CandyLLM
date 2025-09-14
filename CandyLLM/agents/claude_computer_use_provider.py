"""
Claude Computer Use Agent Provider

Integrates Anthropic's Claude Computer Use API for computer interaction
and automation, enabling screenshot analysis, UI interaction, and task automation.
"""

import uuid
import asyncio
import json
import base64
from typing import Dict, List, Optional, Any, Callable, Union
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
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    # Mock classes for when Anthropic is not available
    class Anthropic:
        pass
    anthropic = None

try:
    from PIL import Image, ImageGrab
    import pyautogui
    import keyboard
    import mouse
    AUTOMATION_AVAILABLE = True
except ImportError:
    AUTOMATION_AVAILABLE = False
    # Mock classes for automation libraries
    class Image:
        pass
    class ImageGrab:
        pass
    pyautogui = None
    keyboard = None
    mouse = None


class ActionType(Enum):
    """Types of computer actions"""
    CLICK = "click"
    TYPE = "type"
    KEY = "key"
    SCROLL = "scroll"
    SCREENSHOT = "screenshot"
    WAIT = "wait"
    DRAG = "drag"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"


@dataclass
class ComputerAction:
    """Definition for a computer action"""
    action_type: ActionType
    coordinates: Optional[tuple] = None
    text: Optional[str] = None
    key: Optional[str] = None
    duration: float = 0.1
    scroll_direction: Optional[str] = None
    scroll_amount: int = 3
    wait_time: float = 1.0


@dataclass
class ScreenAnalysis:
    """Analysis of a screenshot"""
    description: str
    ui_elements: List[Dict[str, Any]]
    text_content: str
    actions_suggested: List[ComputerAction]
    confidence_score: float
    timestamp: datetime


class ClaudeComputerAgent:
    """Claude agent with computer interaction capabilities"""
    
    def __init__(self, agent_id: str, api_key: str,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.api_key = api_key
        self.security_manager = security_manager
        self._client = None
        self._action_history = []
        self._screenshot_history = []
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the Claude Computer Use agent"""
        if not ANTHROPIC_AVAILABLE:
            return False
        
        try:
            self._client = Anthropic(api_key=self.api_key)
            return True
        except Exception as e:
            return False
    
    async def take_screenshot(self) -> Optional[str]:
        """Take a screenshot and return base64 encoded image"""
        if not AUTOMATION_AVAILABLE:
            return None
        
        try:
            # Take screenshot
            screenshot = ImageGrab.grab()
            
            # Convert to base64
            import io
            buffer = io.BytesIO()
            screenshot.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Store in history
            screenshot_record = {
                'timestamp': datetime.now().isoformat(),
                'size': screenshot.size,
                'format': 'PNG'
            }
            self._screenshot_history.append(screenshot_record)
            
            return img_base64
            
        except Exception as e:
            return None
    
    async def analyze_screen(self, screenshot_b64: Optional[str] = None) -> ScreenAnalysis:
        """Analyze the current screen using Claude"""
        if not self._client:
            return ScreenAnalysis(
                description="Client not initialized",
                ui_elements=[],
                text_content="",
                actions_suggested=[],
                confidence_score=0.0,
                timestamp=datetime.now()
            )
        
        try:
            # Take screenshot if not provided
            if not screenshot_b64:
                screenshot_b64 = await self.take_screenshot()
            
            if not screenshot_b64:
                return ScreenAnalysis(
                    description="Failed to capture screenshot",
                    ui_elements=[],
                    text_content="",
                    actions_suggested=[],
                    confidence_score=0.0,
                    timestamp=datetime.now()
                )
            
            # Analyze with Claude
            message = await self._client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": screenshot_b64
                                }
                            },
                            {
                                "type": "text",
                                "text": """Analyze this screenshot and provide:
1. A detailed description of what's visible
2. Identify UI elements (buttons, text fields, menus, etc.) with approximate locations
3. Extract any visible text content
4. Suggest possible actions that could be taken
5. Rate your confidence in the analysis (0-1)

Format your response as JSON with keys: description, ui_elements, text_content, suggested_actions, confidence_score"""
                            }
                        ]
                    }
                ]
            )
            
            # Parse response
            response_text = message.content[0].text
            
            try:
                # Try to parse as JSON
                analysis_data = json.loads(response_text)
                
                # Convert suggested actions to ComputerAction objects
                suggested_actions = []
                for action_data in analysis_data.get('suggested_actions', []):
                    if isinstance(action_data, dict):
                        action_type = ActionType(action_data.get('type', 'click'))
                        action = ComputerAction(
                            action_type=action_type,
                            coordinates=tuple(action_data.get('coordinates', [])) if action_data.get('coordinates') else None,
                            text=action_data.get('text'),
                            key=action_data.get('key')
                        )
                        suggested_actions.append(action)
                
                return ScreenAnalysis(
                    description=analysis_data.get('description', ''),
                    ui_elements=analysis_data.get('ui_elements', []),
                    text_content=analysis_data.get('text_content', ''),
                    actions_suggested=suggested_actions,
                    confidence_score=float(analysis_data.get('confidence_score', 0.5)),
                    timestamp=datetime.now()
                )
                
            except json.JSONDecodeError:
                # Fallback: use response as description
                return ScreenAnalysis(
                    description=response_text,
                    ui_elements=[],
                    text_content="",
                    actions_suggested=[],
                    confidence_score=0.5,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            return ScreenAnalysis(
                description=f"Analysis failed: {str(e)}",
                ui_elements=[],
                text_content="",
                actions_suggested=[],
                confidence_score=0.0,
                timestamp=datetime.now()
            )
    
    async def execute_action(self, action: ComputerAction) -> Dict[str, Any]:
        """Execute a computer action"""
        if not AUTOMATION_AVAILABLE:
            return {'success': False, 'error': 'Automation libraries not available'}
        
        try:
            start_time = datetime.now()
            
            # Execute based on action type
            if action.action_type == ActionType.CLICK:
                if action.coordinates:
                    pyautogui.click(action.coordinates[0], action.coordinates[1])
                else:
                    return {'success': False, 'error': 'Coordinates required for click'}
                
            elif action.action_type == ActionType.TYPE:
                if action.text:
                    pyautogui.typewrite(action.text, interval=0.05)
                else:
                    return {'success': False, 'error': 'Text required for type action'}
                
            elif action.action_type == ActionType.KEY:
                if action.key:
                    pyautogui.press(action.key)
                else:
                    return {'success': False, 'error': 'Key required for key action'}
                
            elif action.action_type == ActionType.SCROLL:
                if action.coordinates:
                    x, y = action.coordinates
                    if action.scroll_direction == 'up':
                        pyautogui.scroll(action.scroll_amount, x=x, y=y)
                    else:
                        pyautogui.scroll(-action.scroll_amount, x=x, y=y)
                else:
                    if action.scroll_direction == 'up':
                        pyautogui.scroll(action.scroll_amount)
                    else:
                        pyautogui.scroll(-action.scroll_amount)
                        
            elif action.action_type == ActionType.RIGHT_CLICK:
                if action.coordinates:
                    pyautogui.rightClick(action.coordinates[0], action.coordinates[1])
                else:
                    return {'success': False, 'error': 'Coordinates required for right click'}
                    
            elif action.action_type == ActionType.DOUBLE_CLICK:
                if action.coordinates:
                    pyautogui.doubleClick(action.coordinates[0], action.coordinates[1])
                else:
                    return {'success': False, 'error': 'Coordinates required for double click'}
                    
            elif action.action_type == ActionType.DRAG:
                # Implementation would need start and end coordinates
                return {'success': False, 'error': 'Drag action not fully implemented'}
                
            elif action.action_type == ActionType.WAIT:
                await asyncio.sleep(action.wait_time)
                
            elif action.action_type == ActionType.SCREENSHOT:
                screenshot = await self.take_screenshot()
                return {
                    'success': True,
                    'screenshot': screenshot,
                    'execution_time': (datetime.now() - start_time).total_seconds()
                }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Record action
            action_record = {
                'action': action.__dict__,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            self._action_history.append(action_record)
            
            return {
                'success': True,
                'execution_time': execution_time,
                'action_record': action_record
            }
            
        except Exception as e:
            action_record = {
                'action': action.__dict__,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            self._action_history.append(action_record)
            
            return {
                'success': False,
                'error': str(e),
                'action_record': action_record
            }
    
    async def execute_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a complex computer task using Claude's guidance"""
        try:
            context = context or {}
            task_steps = []
            
            # Take initial screenshot
            screenshot = await self.take_screenshot()
            if not screenshot:
                return {'error': 'Failed to take initial screenshot'}
            
            # Analyze current state
            analysis = await self.analyze_screen(screenshot)
            task_steps.append({
                'step': 'initial_analysis',
                'analysis': analysis.__dict__,
                'screenshot_taken': True
            })
            
            # Plan task execution with Claude
            planning_message = await self._client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Task: {task_description}

Current screen analysis:
{analysis.description}

UI elements detected: {len(analysis.ui_elements)}
Suggested actions: {[action.action_type.value for action in analysis.actions_suggested]}

Create a step-by-step plan to accomplish this task. For each step, specify:
1. The action to take (click, type, key, etc.)
2. The target (coordinates, text, key name)
3. Expected outcome

Be specific and practical."""
                    }
                ]
            )
            
            plan_text = planning_message.content[0].text
            task_steps.append({
                'step': 'planning',
                'plan': plan_text
            })
            
            # For now, return the plan without execution
            # Full implementation would parse the plan and execute actions
            
            return {
                'task_description': task_description,
                'initial_analysis': analysis.__dict__,
                'execution_plan': plan_text,
                'task_steps': task_steps,
                'status': 'planned'
            }
            
        except Exception as e:
            return {
                'task_description': task_description,
                'error': str(e),
                'status': 'failed'
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'action_history_count': len(self._action_history),
            'screenshot_history_count': len(self._screenshot_history),
            'automation_available': AUTOMATION_AVAILABLE,
            'anthropic_available': ANTHROPIC_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class ClaudeComputerUseProvider(BaseAgentProvider):
    """
    Provider implementation for Anthropic Claude Computer Use.
    
    Enables computer interaction and automation through Claude's vision
    capabilities, including screenshot analysis, UI interaction, and task automation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, ClaudeComputerAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not ANTHROPIC_AVAILABLE:
            self.logger.warning("Anthropic not available - provider will be non-functional")
        if not AUTOMATION_AVAILABLE:
            self.logger.warning("Automation libraries not available - limited functionality")
    
    @property
    def provider_name(self) -> str:
        return "claude_computer_use"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.COMPUTER_INTERACTION,
            AgentCapability.VISION_ANALYSIS,
            AgentCapability.TASK_AUTOMATION,
            AgentCapability.UI_AUTOMATION
        ]
    
    async def initialize(self) -> bool:
        """Initialize Claude Computer Use provider"""
        if not ANTHROPIC_AVAILABLE:
            self.logger.error("Anthropic not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Claude Computer Use provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Claude Computer Use provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Claude Computer Use agent"""
        if not self._initialized:
            await self.initialize()
        
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("Anthropic not available")
        
        agent_id = f"ccu_{uuid.uuid4().hex[:8]}"
        
        try:
            # Get API key from config
            api_key = self.config.get('api_key')
            if not api_key:
                raise ValueError("Anthropic API key required for Claude Computer Use")
            
            # Create agent
            agent = ClaudeComputerAgent(
                agent_id=agent_id,
                api_key=api_key,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Claude Computer Use agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Claude Computer Use agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Claude Computer Use agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Claude Computer Use agent"""
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
            
            # Determine action type from context
            if context.get('action_type') == 'screenshot':
                # Take screenshot
                screenshot = await agent.take_screenshot()
                result = {
                    'screenshot': screenshot,
                    'action': 'screenshot_taken'
                }
                
            elif context.get('action_type') == 'analyze_screen':
                # Analyze screen
                analysis = await agent.analyze_screen()
                result = {
                    'analysis': analysis.__dict__,
                    'action': 'screen_analyzed'
                }
                
            elif context.get('action_type') == 'execute_action':
                # Execute specific action
                action_data = context.get('action', {})
                action = ComputerAction(
                    action_type=ActionType(action_data.get('type', 'click')),
                    coordinates=tuple(action_data.get('coordinates', [])) if action_data.get('coordinates') else None,
                    text=action_data.get('text'),
                    key=action_data.get('key')
                )
                result = await agent.execute_action(action)
                
            else:
                # Execute as task
                result = await agent.execute_task(prompt, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Format content
            content = ""
            if 'analysis' in result:
                analysis = result['analysis']
                content = f"Screen Analysis:\n{analysis.get('description', '')}"
                if analysis.get('ui_elements'):
                    content += f"\n\nUI Elements: {len(analysis['ui_elements'])} detected"
                if analysis.get('actions_suggested'):
                    content += f"\nSuggested Actions: {len(analysis['actions_suggested'])}"
            elif 'execution_plan' in result:
                content = f"Task: {result['task_description']}\n\nPlan:\n{result['execution_plan']}"
            elif 'screenshot' in result:
                content = "Screenshot captured successfully"
            elif 'success' in result:
                content = f"Action executed: {result.get('success', False)}"
            else:
                content = str(result)
            
            return AgentResponse(
                content=content,
                agent_id=agent_id,
                provider=self.provider_name,
                metadata={
                    'execution_time_seconds': execution_time,
                    'result': result,
                    'agent_info': agent.get_agent_info(),
                    'automation_available': AUTOMATION_AVAILABLE
                },
                error=result.get('error')
            )
            
        except Exception as e:
            self.logger.error(f"Claude Computer Use agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Claude Computer Use agent"""
        try:
            # Computer use tools are typically built-in actions
            self.logger.info(f"Tool {tool_spec.name} noted for Claude Computer Use agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Claude Computer Use agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Use Claude to generate tool specification for computer interaction
            synthesis_prompt = f"""
            Create a computer interaction tool specification for: {tool_description}
            
            Requirements:
            1. Define the tool's purpose for computer automation
            2. Specify input parameters (coordinates, text, keys, etc.)
            3. Describe expected computer actions and outcomes
            4. Include safety considerations and error handling
            5. Provide usage guidelines for UI automation
            
            {f'Examples: {examples}' if examples else ''}
            
            Generate a detailed tool specification for computer interaction.
            """
            
            # Execute synthesis using the agent
            result = await self.execute_agent(agent_id, synthesis_prompt)
            
            # Create tool spec
            tool_spec = ToolSpec(
                name=f"ccu_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'high', 'requires_approval': True}  # Computer interaction is high risk
            )
            
            self.logger.info(f"Synthesized tool for Claude Computer Use agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Claude Computer Use agent"""
        try:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                # Clean up agent resources
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Claude Computer Use agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Claude Computer Use agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Claude Computer Use agent"""
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