"""
Multi-On Agent Provider

Integrates Multi-On's web automation capabilities for browser-based task execution,
enabling intelligent navigation, form filling, data extraction, and web workflows.
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
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.action_chains import ActionChains
    MULTIO_AVAILABLE = True
except ImportError:
    MULTIO_AVAILABLE = False
    # Mock classes for when dependencies are not available
    httpx = None
    requests = None
    webdriver = None
    By = None
    WebDriverWait = None
    EC = None
    ActionChains = None


class BrowserAction(Enum):
    """Types of browser actions Multi-On can perform"""
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    EXTRACT_TEXT = "extract_text"
    EXTRACT_DATA = "extract_data"
    FILL_FORM = "fill_form"
    TAKE_SCREENSHOT = "take_screenshot"
    DOWNLOAD_FILE = "download_file"
    UPLOAD_FILE = "upload_file"
    SWITCH_TAB = "switch_tab"
    CLOSE_TAB = "close_tab"
    REFRESH_PAGE = "refresh_page"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"


class SelectorType(Enum):
    """Types of element selectors"""
    CSS = "css"
    XPATH = "xpath"
    ID = "id"
    NAME = "name"
    CLASS_NAME = "class_name"
    TAG_NAME = "tag_name"
    LINK_TEXT = "link_text"
    PARTIAL_LINK_TEXT = "partial_link_text"


@dataclass
class MultionConfig:
    """Configuration for Multi-On agent"""
    api_key: str = ""
    base_url: str = "https://api.multion.ai"
    browser_type: str = "chrome"  # "chrome", "firefox", "safari", "edge"
    headless: bool = False
    implicit_wait: float = 10.0
    page_load_timeout: float = 30.0
    script_timeout: float = 30.0
    window_size: tuple = (1920, 1080)
    user_agent: str = ""
    proxy_url: str = ""
    enable_javascript: bool = True
    enable_images: bool = True
    enable_cookies: bool = True
    max_tabs: int = 10
    download_directory: str = ""
    safety_mode: str = "strict"  # "strict", "moderate", "permissive"
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    timeout: float = 120.0
    max_retries: int = 3


@dataclass
class WebElement:
    """Web element information"""
    selector: str
    selector_type: SelectorType
    text: str = ""
    tag_name: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    is_displayed: bool = True
    is_enabled: bool = True
    location: Dict[str, int] = field(default_factory=dict)
    size: Dict[str, int] = field(default_factory=dict)


@dataclass
class BrowserState:
    """Current browser state information"""
    url: str = ""
    title: str = ""
    page_source: str = ""
    cookies: List[Dict[str, Any]] = field(default_factory=list)
    local_storage: Dict[str, str] = field(default_factory=dict)
    session_storage: Dict[str, str] = field(default_factory=dict)
    window_handles: List[str] = field(default_factory=list)
    current_window: str = ""
    elements: List[WebElement] = field(default_factory=list)


@dataclass
class ActionResult:
    """Result of a browser action"""
    action: BrowserAction
    success: bool
    message: str = ""
    data: Any = None
    screenshot: Optional[str] = None  # Base64 encoded
    execution_time: float = 0.0
    element_found: bool = False
    error_details: Optional[str] = None


@dataclass
class WebWorkflow:
    """Web automation workflow"""
    workflow_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    failure_conditions: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0


@dataclass
class WorkflowExecution:
    """Workflow execution results"""
    workflow_id: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    steps_completed: int = 0
    total_steps: int = 0
    results: List[ActionResult] = field(default_factory=list)
    final_state: Optional[BrowserState] = None
    error_log: List[str] = field(default_factory=list)


class MultionAgent:
    """Multi-On web automation agent"""
    
    def __init__(self, agent_id: str, config: MultionConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._driver = None
        self._wait = None
        self._actions = None
        self._workflows = {}
        self._executions = []
        self._usage_stats = {
            'total_sessions': 0,
            'successful_navigations': 0,
            'failed_navigations': 0,
            'actions_executed': 0,
            'forms_filled': 0,
            'data_extracted': 0,
            'screenshots_taken': 0,
            'workflows_executed': 0,
            'domains_visited': set(),
            'browser_actions': {action.value: 0 for action in BrowserAction},
            'total_execution_time': 0.0
        }
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the Multi-On agent"""
        if not MULTIO_AVAILABLE:
            return False
        
        try:
            # Setup browser driver
            await self._setup_browser()
            return True
            
        except Exception as e:
            return False
    
    async def _setup_browser(self):
        """Setup browser driver with configuration"""
        try:
            # Browser options
            if self.config.browser_type.lower() == "chrome":
                from selenium.webdriver.chrome.options import Options
                options = Options()
                if self.config.headless:
                    options.add_argument("--headless")
                options.add_argument(f"--window-size={self.config.window_size[0]},{self.config.window_size[1]}")
                if self.config.user_agent:
                    options.add_argument(f"--user-agent={self.config.user_agent}")
                if self.config.proxy_url:
                    options.add_argument(f"--proxy-server={self.config.proxy_url}")
                if not self.config.enable_images:
                    prefs = {"profile.managed_default_content_settings.images": 2}
                    options.add_experimental_option("prefs", prefs)
                if self.config.download_directory:
                    prefs = {"download.default_directory": self.config.download_directory}
                    options.add_experimental_option("prefs", prefs)
                
                self._driver = webdriver.Chrome(options=options)
                
            elif self.config.browser_type.lower() == "firefox":
                from selenium.webdriver.firefox.options import Options
                options = Options()
                if self.config.headless:
                    options.add_argument("--headless")
                options.add_argument(f"--width={self.config.window_size[0]}")
                options.add_argument(f"--height={self.config.window_size[1]}")
                
                self._driver = webdriver.Firefox(options=options)
            
            else:
                raise ValueError(f"Unsupported browser type: {self.config.browser_type}")
            
            # Configure timeouts
            self._driver.implicitly_wait(self.config.implicit_wait)
            self._driver.set_page_load_timeout(self.config.page_load_timeout)
            self._driver.set_script_timeout(self.config.script_timeout)
            
            # Setup WebDriverWait and ActionChains
            self._wait = WebDriverWait(self._driver, self.config.implicit_wait)
            self._actions = ActionChains(self._driver)
            
            self._usage_stats['total_sessions'] += 1
            
        except Exception as e:
            raise RuntimeError(f"Browser setup failed: {e}")
    
    async def navigate(self, url: str) -> ActionResult:
        """Navigate to a URL"""
        try:
            start_time = datetime.now()
            
            # Security check
            if not self._is_url_allowed(url):
                return ActionResult(
                    action=BrowserAction.NAVIGATE,
                    success=False,
                    message=f"URL not allowed: {url}",
                    error_details="Domain blocked or not in allowed list"
                )
            
            # Navigate
            self._driver.get(url)
            
            # Update domain stats
            domain = self._extract_domain(url)
            self._usage_stats['domains_visited'].add(domain)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._usage_stats['successful_navigations'] += 1
            self._usage_stats['browser_actions'][BrowserAction.NAVIGATE.value] += 1
            
            return ActionResult(
                action=BrowserAction.NAVIGATE,
                success=True,
                message=f"Successfully navigated to {url}",
                data={"url": self._driver.current_url, "title": self._driver.title},
                execution_time=execution_time
            )
            
        except Exception as e:
            self._usage_stats['failed_navigations'] += 1
            return ActionResult(
                action=BrowserAction.NAVIGATE,
                success=False,
                message=f"Navigation failed: {str(e)}",
                error_details=str(e)
            )
    
    def _is_url_allowed(self, url: str) -> bool:
        """Check if URL is allowed based on security settings"""
        domain = self._extract_domain(url)
        
        # Check blocked domains
        if self.config.blocked_domains:
            if any(blocked in domain for blocked in self.config.blocked_domains):
                return False
        
        # Check allowed domains
        if self.config.allowed_domains:
            if not any(allowed in domain for allowed in self.config.allowed_domains):
                return False
        
        return True
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return url.split('/')[2] if '/' in url else url
    
    async def click_element(self, selector: str, selector_type: SelectorType = SelectorType.CSS) -> ActionResult:
        """Click an element"""
        try:
            start_time = datetime.now()
            
            # Find element
            element = await self._find_element(selector, selector_type)
            if not element:
                return ActionResult(
                    action=BrowserAction.CLICK,
                    success=False,
                    message=f"Element not found: {selector}",
                    element_found=False
                )
            
            # Click element
            element.click()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._usage_stats['actions_executed'] += 1
            self._usage_stats['browser_actions'][BrowserAction.CLICK.value] += 1
            
            return ActionResult(
                action=BrowserAction.CLICK,
                success=True,
                message=f"Successfully clicked element: {selector}",
                element_found=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            return ActionResult(
                action=BrowserAction.CLICK,
                success=False,
                message=f"Click failed: {str(e)}",
                error_details=str(e)
            )
    
    async def type_text(self, selector: str, text: str, 
                       selector_type: SelectorType = SelectorType.CSS,
                       clear_first: bool = True) -> ActionResult:
        """Type text into an element"""
        try:
            start_time = datetime.now()
            
            # Find element
            element = await self._find_element(selector, selector_type)
            if not element:
                return ActionResult(
                    action=BrowserAction.TYPE,
                    success=False,
                    message=f"Element not found: {selector}",
                    element_found=False
                )
            
            # Clear and type
            if clear_first:
                element.clear()
            element.send_keys(text)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._usage_stats['actions_executed'] += 1
            self._usage_stats['browser_actions'][BrowserAction.TYPE.value] += 1
            
            return ActionResult(
                action=BrowserAction.TYPE,
                success=True,
                message=f"Successfully typed text into element: {selector}",
                data={"text": text},
                element_found=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            return ActionResult(
                action=BrowserAction.TYPE,
                success=False,
                message=f"Type failed: {str(e)}",
                error_details=str(e)
            )
    
    async def _find_element(self, selector: str, selector_type: SelectorType):
        """Find element by selector"""
        try:
            if selector_type == SelectorType.CSS:
                return self._wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            elif selector_type == SelectorType.XPATH:
                return self._wait.until(EC.presence_of_element_located((By.XPATH, selector)))
            elif selector_type == SelectorType.ID:
                return self._wait.until(EC.presence_of_element_located((By.ID, selector)))
            elif selector_type == SelectorType.NAME:
                return self._wait.until(EC.presence_of_element_located((By.NAME, selector)))
            elif selector_type == SelectorType.CLASS_NAME:
                return self._wait.until(EC.presence_of_element_located((By.CLASS_NAME, selector)))
            elif selector_type == SelectorType.TAG_NAME:
                return self._wait.until(EC.presence_of_element_located((By.TAG_NAME, selector)))
            elif selector_type == SelectorType.LINK_TEXT:
                return self._wait.until(EC.presence_of_element_located((By.LINK_TEXT, selector)))
            elif selector_type == SelectorType.PARTIAL_LINK_TEXT:
                return self._wait.until(EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, selector)))
            else:
                return None
        except:
            return None
    
    async def extract_text(self, selector: str, selector_type: SelectorType = SelectorType.CSS) -> ActionResult:
        """Extract text from an element"""
        try:
            start_time = datetime.now()
            
            # Find element
            element = await self._find_element(selector, selector_type)
            if not element:
                return ActionResult(
                    action=BrowserAction.EXTRACT_TEXT,
                    success=False,
                    message=f"Element not found: {selector}",
                    element_found=False
                )
            
            # Extract text
            text = element.text
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._usage_stats['data_extracted'] += 1
            self._usage_stats['browser_actions'][BrowserAction.EXTRACT_TEXT.value] += 1
            
            return ActionResult(
                action=BrowserAction.EXTRACT_TEXT,
                success=True,
                message=f"Successfully extracted text from element: {selector}",
                data={"text": text},
                element_found=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            return ActionResult(
                action=BrowserAction.EXTRACT_TEXT,
                success=False,
                message=f"Text extraction failed: {str(e)}",
                error_details=str(e)
            )
    
    async def extract_data(self, selectors: Dict[str, str],
                         selector_type: SelectorType = SelectorType.CSS) -> ActionResult:
        """Extract multiple data points from the page"""
        try:
            start_time = datetime.now()
            
            extracted_data = {}
            
            for key, selector in selectors.items():
                element = await self._find_element(selector, selector_type)
                if element:
                    extracted_data[key] = element.text
                else:
                    extracted_data[key] = None
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._usage_stats['data_extracted'] += 1
            self._usage_stats['browser_actions'][BrowserAction.EXTRACT_DATA.value] += 1
            
            return ActionResult(
                action=BrowserAction.EXTRACT_DATA,
                success=True,
                message=f"Successfully extracted data from {len(selectors)} elements",
                data=extracted_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            return ActionResult(
                action=BrowserAction.EXTRACT_DATA,
                success=False,
                message=f"Data extraction failed: {str(e)}",
                error_details=str(e)
            )
    
    async def fill_form(self, form_data: Dict[str, Any],
                       selector_type: SelectorType = SelectorType.CSS) -> ActionResult:
        """Fill a form with provided data"""
        try:
            start_time = datetime.now()
            
            filled_fields = []
            failed_fields = []
            
            for field_selector, value in form_data.items():
                try:
                    element = await self._find_element(field_selector, selector_type)
                    if element:
                        element.clear()
                        element.send_keys(str(value))
                        filled_fields.append(field_selector)
                    else:
                        failed_fields.append(field_selector)
                except:
                    failed_fields.append(field_selector)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._usage_stats['forms_filled'] += 1
            self._usage_stats['browser_actions'][BrowserAction.FILL_FORM.value] += 1
            
            success = len(failed_fields) == 0
            
            return ActionResult(
                action=BrowserAction.FILL_FORM,
                success=success,
                message=f"Form filling completed. {len(filled_fields)} fields filled, {len(failed_fields)} failed.",
                data={
                    "filled_fields": filled_fields,
                    "failed_fields": failed_fields
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            return ActionResult(
                action=BrowserAction.FILL_FORM,
                success=False,
                message=f"Form filling failed: {str(e)}",
                error_details=str(e)
            )
    
    async def take_screenshot(self) -> ActionResult:
        """Take a screenshot of the current page"""
        try:
            start_time = datetime.now()
            
            # Take screenshot
            screenshot_b64 = self._driver.get_screenshot_as_base64()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._usage_stats['screenshots_taken'] += 1
            self._usage_stats['browser_actions'][BrowserAction.TAKE_SCREENSHOT.value] += 1
            
            return ActionResult(
                action=BrowserAction.TAKE_SCREENSHOT,
                success=True,
                message="Screenshot captured successfully",
                screenshot=screenshot_b64,
                execution_time=execution_time
            )
            
        except Exception as e:
            return ActionResult(
                action=BrowserAction.TAKE_SCREENSHOT,
                success=False,
                message=f"Screenshot failed: {str(e)}",
                error_details=str(e)
            )
    
    async def scroll_page(self, direction: str = "down", amount: int = 3) -> ActionResult:
        """Scroll the page"""
        try:
            start_time = datetime.now()
            
            # Calculate scroll amount
            scroll_amount = amount * 300  # pixels
            
            if direction.lower() == "down":
                self._driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            elif direction.lower() == "up":
                self._driver.execute_script(f"window.scrollBy(0, -{scroll_amount});")
            elif direction.lower() == "top":
                self._driver.execute_script("window.scrollTo(0, 0);")
            elif direction.lower() == "bottom":
                self._driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._usage_stats['browser_actions'][BrowserAction.SCROLL.value] += 1
            
            return ActionResult(
                action=BrowserAction.SCROLL,
                success=True,
                message=f"Successfully scrolled {direction}",
                data={"direction": direction, "amount": amount},
                execution_time=execution_time
            )
            
        except Exception as e:
            return ActionResult(
                action=BrowserAction.SCROLL,
                success=False,
                message=f"Scroll failed: {str(e)}",
                error_details=str(e)
            )
    
    async def wait_for_element(self, selector: str, timeout: float = None,
                             selector_type: SelectorType = SelectorType.CSS) -> ActionResult:
        """Wait for an element to appear"""
        try:
            start_time = datetime.now()
            
            timeout = timeout or self.config.implicit_wait
            wait = WebDriverWait(self._driver, timeout)
            
            if selector_type == SelectorType.CSS:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            elif selector_type == SelectorType.XPATH:
                wait.until(EC.presence_of_element_located((By.XPATH, selector)))
            elif selector_type == SelectorType.ID:
                wait.until(EC.presence_of_element_located((By.ID, selector)))
            # Add other selector types as needed
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._usage_stats['browser_actions'][BrowserAction.WAIT.value] += 1
            
            return ActionResult(
                action=BrowserAction.WAIT,
                success=True,
                message=f"Element appeared: {selector}",
                element_found=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            return ActionResult(
                action=BrowserAction.WAIT,
                success=False,
                message=f"Wait timeout: {str(e)}",
                error_details=str(e)
            )
    
    def get_browser_state(self) -> BrowserState:
        """Get current browser state"""
        try:
            return BrowserState(
                url=self._driver.current_url,
                title=self._driver.title,
                page_source=self._driver.page_source[:1000],  # Truncated for performance
                cookies=self._driver.get_cookies(),
                window_handles=self._driver.window_handles,
                current_window=self._driver.current_window_handle
            )
        except:
            return BrowserState()
    
    async def execute_workflow(self, workflow: WebWorkflow) -> WorkflowExecution:
        """Execute a web automation workflow"""
        try:
            execution_id = f"exec_{uuid.uuid4().hex[:8]}"
            start_time = datetime.now()
            
            execution = WorkflowExecution(
                workflow_id=workflow.workflow_id,
                execution_id=execution_id,
                start_time=start_time,
                total_steps=len(workflow.steps)
            )
            
            for i, step in enumerate(workflow.steps):
                try:
                    action = BrowserAction(step.get("action"))
                    params = step.get("parameters", {})
                    
                    # Execute action based on type
                    if action == BrowserAction.NAVIGATE:
                        result = await self.navigate(params.get("url"))
                    elif action == BrowserAction.CLICK:
                        result = await self.click_element(
                            params.get("selector"),
                            SelectorType(params.get("selector_type", SelectorType.CSS.value))
                        )
                    elif action == BrowserAction.TYPE:
                        result = await self.type_text(
                            params.get("selector"),
                            params.get("text"),
                            SelectorType(params.get("selector_type", SelectorType.CSS.value))
                        )
                    elif action == BrowserAction.EXTRACT_TEXT:
                        result = await self.extract_text(
                            params.get("selector"),
                            SelectorType(params.get("selector_type", SelectorType.CSS.value))
                        )
                    elif action == BrowserAction.EXTRACT_DATA:
                        result = await self.extract_data(
                            params.get("selectors"),
                            SelectorType(params.get("selector_type", SelectorType.CSS.value))
                        )
                    elif action == BrowserAction.FILL_FORM:
                        result = await self.fill_form(
                            params.get("form_data"),
                            SelectorType(params.get("selector_type", SelectorType.CSS.value))
                        )
                    elif action == BrowserAction.SCROLL:
                        result = await self.scroll_page(
                            params.get("direction", "down"),
                            params.get("amount", 3)
                        )
                    elif action == BrowserAction.WAIT:
                        result = await self.wait_for_element(
                            params.get("selector"),
                            params.get("timeout"),
                            SelectorType(params.get("selector_type", SelectorType.CSS.value))
                        )
                    elif action == BrowserAction.TAKE_SCREENSHOT:
                        result = await self.take_screenshot()
                    else:
                        result = ActionResult(
                            action=action,
                            success=False,
                            message=f"Unsupported action: {action}"
                        )
                    
                    execution.results.append(result)
                    
                    if result.success:
                        execution.steps_completed += 1
                    else:
                        execution.error_log.append(f"Step {i+1} failed: {result.message}")
                        
                        # Stop on critical failures
                        if not step.get("continue_on_failure", False):
                            break
                    
                    # Small delay between steps
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    error_msg = f"Step {i+1} exception: {str(e)}"
                    execution.error_log.append(error_msg)
                    break
            
            execution.end_time = datetime.now()
            execution.success = execution.steps_completed >= execution.total_steps * 0.8  # 80% success rate
            execution.final_state = self.get_browser_state()
            
            self._executions.append(execution)
            self._usage_stats['workflows_executed'] += 1
            
            return execution
            
        except Exception as e:
            return WorkflowExecution(
                workflow_id=workflow.workflow_id,
                execution_id=f"failed_{uuid.uuid4().hex[:8]}",
                start_time=datetime.now(),
                success=False,
                error_log=[str(e)]
            )
    
    def create_workflow(self, name: str, description: str, steps: List[Dict[str, Any]]) -> WebWorkflow:
        """Create a new web workflow"""
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        
        workflow = WebWorkflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            steps=steps,
            estimated_duration=len(steps) * 2.0  # Rough estimate
        )
        
        self._workflows[workflow_id] = workflow
        return workflow
    
    def get_workflow(self, workflow_id: str) -> Optional[WebWorkflow]:
        """Get workflow by ID"""
        return self._workflows.get(workflow_id)
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows"""
        return [
            {
                'workflow_id': workflow.workflow_id,
                'name': workflow.name,
                'description': workflow.description,
                'steps_count': len(workflow.steps),
                'estimated_duration': workflow.estimated_duration
            }
            for workflow in self._workflows.values()
        ]
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get workflow execution history"""
        return [
            {
                'execution_id': execution.execution_id,
                'workflow_id': execution.workflow_id,
                'success': execution.success,
                'steps_completed': execution.steps_completed,
                'total_steps': execution.total_steps,
                'start_time': execution.start_time.isoformat(),
                'end_time': execution.end_time.isoformat() if execution.end_time else None
            }
            for execution in self._executions[-limit:]
        ]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = self._usage_stats.copy()
        stats['domains_visited'] = list(stats['domains_visited'])  # Convert set to list
        
        # Calculate success rates
        total_navigations = stats['successful_navigations'] + stats['failed_navigations']
        if total_navigations > 0:
            stats['navigation_success_rate'] = stats['successful_navigations'] / total_navigations
        else:
            stats['navigation_success_rate'] = 0.0
        
        return stats
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': {
                'browser_type': self.config.browser_type,
                'headless': self.config.headless,
                'window_size': self.config.window_size,
                'safety_mode': self.config.safety_mode,
                'max_tabs': self.config.max_tabs
            },
            'workflows_count': len(self._workflows),
            'executions_count': len(self._executions),
            'usage_stats': self.get_usage_stats(),
            'current_state': self.get_browser_state().__dict__ if self._driver else {},
            'multio_available': MULTIO_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }
    
    async def cleanup(self):
        """Cleanup browser resources"""
        try:
            if self._driver:
                self._driver.quit()
        except:
            pass


class MultionProvider(BaseAgentProvider):
    """
    Provider implementation for Multi-On web automation.
    
    Enables intelligent browser automation for web navigation, form filling,
    data extraction, and complex web workflows.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, MultionAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not MULTIO_AVAILABLE:
            self.logger.warning("Multi-On dependencies not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "multion"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.WEB_AUTOMATION,
            AgentCapability.BROWSER_CONTROL,
            AgentCapability.FORM_AUTOMATION,
            AgentCapability.DATA_EXTRACTION,
            AgentCapability.WORKFLOW_AUTOMATION,
            AgentCapability.SCREEN_INTERACTION
        ]
    
    async def initialize(self) -> bool:
        """Initialize Multi-On provider"""
        if not MULTIO_AVAILABLE:
            self.logger.error("Multi-On dependencies not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Multi-On provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Multi-On provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Multi-On agent"""
        if not self._initialized:
            await self.initialize()
        
        if not MULTIO_AVAILABLE:
            raise RuntimeError("Multi-On dependencies not available")
        
        agent_id = f"multion_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create Multi-On configuration
            multion_config = MultionConfig(
                api_key=self.config.get('api_key', ''),
                base_url=self.config.get('base_url', 'https://api.multion.ai'),
                browser_type=self.config.get('browser_type', 'chrome'),
                headless=self.config.get('headless', False),
                implicit_wait=self.config.get('implicit_wait', 10.0),
                page_load_timeout=self.config.get('page_load_timeout', 30.0),
                script_timeout=self.config.get('script_timeout', 30.0),
                window_size=tuple(self.config.get('window_size', [1920, 1080])),
                user_agent=self.config.get('user_agent', ''),
                proxy_url=self.config.get('proxy_url', ''),
                enable_javascript=self.config.get('enable_javascript', True),
                enable_images=self.config.get('enable_images', True),
                enable_cookies=self.config.get('enable_cookies', True),
                max_tabs=self.config.get('max_tabs', 10),
                download_directory=self.config.get('download_directory', ''),
                safety_mode=self.config.get('safety_mode', 'strict'),
                allowed_domains=self.config.get('allowed_domains', []),
                blocked_domains=self.config.get('blocked_domains', []),
                timeout=self.config.get('timeout', 120.0),
                max_retries=self.config.get('max_retries', 3)
            )
            
            # Create agent
            agent = MultionAgent(
                agent_id=agent_id,
                config=multion_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Multi-On agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Multi-On agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Multi-On agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Multi-On agent"""
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
            mode = context.get('mode', 'navigate')
            
            if mode == 'navigate':
                url = context.get('url', prompt)
                result = await agent.navigate(url)
                response_content = result.message
                
            elif mode == 'workflow':
                # Execute workflow from prompt
                workflow_steps = context.get('steps', [])
                if workflow_steps:
                    workflow = agent.create_workflow("Generated Workflow", prompt, workflow_steps)
                    result = await agent.execute_workflow(workflow)
                    response_content = f"Workflow executed: {result.steps_completed}/{result.total_steps} steps completed"
                else:
                    response_content = "No workflow steps provided"
                    result = None
                
            elif mode == 'extract':
                selectors = context.get('selectors', {})
                result = await agent.extract_data(selectors)
                response_content = f"Data extracted from {len(selectors)} elements"
                
            elif mode == 'form':
                form_data = context.get('form_data', {})
                result = await agent.fill_form(form_data)
                response_content = result.message
                
            elif mode == 'screenshot':
                result = await agent.take_screenshot()
                response_content = result.message
                
            else:
                # Default to navigation
                url = context.get('url', prompt)
                result = await agent.navigate(url)
                response_content = result.message
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare metadata
            metadata = {
                'execution_time_seconds': execution_time,
                'mode': mode,
                'browser_state': agent.get_browser_state().__dict__,
                'usage_stats': agent.get_usage_stats(),
                'agent_info': agent.get_agent_info()
            }
            
            # Add result-specific metadata
            if result:
                metadata['action_success'] = result.success
                metadata['action_type'] = result.action.value
                metadata['execution_time_action'] = result.execution_time
                metadata['element_found'] = getattr(result, 'element_found', False)
                
                if result.data:
                    metadata['extracted_data'] = result.data
                
                if result.screenshot:
                    metadata['screenshot_available'] = True
            
            return AgentResponse(
                content=response_content,
                agent_id=agent_id,
                provider=self.provider_name,
                metadata=metadata,
                error=result.error_details if result and not result.success else None
            )
            
        except Exception as e:
            self.logger.error(f"Multi-On agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def navigate_to_url(self, agent_id: str, url: str) -> Dict[str, Any]:
        """Navigate to URL"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        result = await agent.navigate(url)
        
        return {
            'success': result.success,
            'message': result.message,
            'url': result.data.get('url') if result.data else None,
            'title': result.data.get('title') if result.data else None
        }
    
    async def execute_workflow(self, agent_id: str, workflow_id: str) -> Dict[str, Any]:
        """Execute workflow by ID"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        workflow = agent.get_workflow(workflow_id)
        
        if not workflow:
            return {'error': 'Workflow not found'}
        
        result = await agent.execute_workflow(workflow)
        
        return {
            'execution_id': result.execution_id,
            'success': result.success,
            'steps_completed': result.steps_completed,
            'total_steps': result.total_steps,
            'error_log': result.error_log
        }
    
    async def extract_data(self, agent_id: str, selectors: Dict[str, str]) -> Dict[str, Any]:
        """Extract data from current page"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        result = await agent.extract_data(selectors)
        
        return {
            'success': result.success,
            'data': result.data,
            'message': result.message
        }
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Multi-On agent (basic implementation)"""
        if agent_id not in self._agents:
            return False
        
        try:
            # Multi-On doesn't have native tool support
            # This would be implemented as workflow templates
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Multi-On capabilities"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Create tool spec for web automation
            tool_spec = ToolSpec(
                name=f"multion_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={
                    'description': tool_description,
                    'automation_type': 'web_browser',
                    'examples': examples or []
                },
                security_policy={
                    'risk_level': 'high',  # Web automation is high risk
                    'requires_approval': True,
                    'allowed_domains': self.config.get('allowed_domains', []),
                    'blocked_domains': self.config.get('blocked_domains', [])
                }
            )
            
            self.logger.info(f"Synthesized tool for Multi-On agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Multi-On agent"""
        try:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                await agent.cleanup()
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Multi-On agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Multi-On agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Multi-On agent"""
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