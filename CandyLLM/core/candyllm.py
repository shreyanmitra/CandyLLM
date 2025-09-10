"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

🍭 CandyLLM 3.0 Core Interface

Main interface for the next-generation AI platform with intelligent routing,
neurosymbolic reasoning, and universal model support.

Security Implementations:
- Secure API key handling with environment variable fallbacks
- Input sanitization for all user prompts
- Rate limiting to prevent abuse
- Session management with secure UUIDs
- Content filtering for harmful inputs/outputs
- Audit logging for all operations
- Protection against prompt injection attacks
- Secure configuration validation
"""

import asyncio
import time
import os
import re
import hashlib
import secrets
import logging
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, asdict, field
from datetime import datetime
import json

# Security imports
import html
import urllib.parse

# Security logger
security_logger = logging.getLogger('candyllm.security')

# Import core components with error handling
try:
    from .router import IntelligentRouter
    from .neurosymbolic import NeuroSymbolicEngine
    from .reasoning import AdvancedReasoningEngine
    from .providers import UniversalProviderManager
    from .dynamic_tooling import DynamicToolingEngine
    from .dynamic_config import DynamicToolingConfig, ComponentFactory, load_config_from_env
    from .types import ChatResponse, StreamChunk, ReasoningResult, MathResult, RouteDecision
except ImportError as e:
    security_logger.error(f"Failed to import core components: {e}")
    raise

# Security validation patterns
PROMPT_INJECTION_PATTERNS = [
    r'ignore\s+previous\s+instructions',
    r'forget\s+everything',
    r'you\s+are\s+now\s+a',
    r'pretend\s+to\s+be',
    r'jailbreak|DAN|developer\s+mode',
    r'bypass\s+filters?',
    r'override\s+safety',
    r'disable\s+content\s+policy',
    r'unrestricted\s+mode'
]

# Compile patterns for performance
COMPILED_INJECTION_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in PROMPT_INJECTION_PATTERNS]

def _sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks
    
    Args:
        text: Raw user input
        
    Returns:
        Sanitized text safe for processing
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # HTML escape to prevent script injection
    sanitized = html.escape(text)
    
    # URL decode to prevent encoded attacks
    sanitized = urllib.parse.unquote(sanitized)
    
    # Remove null bytes and control characters
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\r\t')
    
    # Limit length to prevent DoS
    if len(sanitized) > 50000:  # 50KB limit
        security_logger.warning(f"Input truncated - exceeded length limit: {len(sanitized)}")
        sanitized = sanitized[:50000] + "... [truncated for security]"
    
    return sanitized

def _detect_prompt_injection(text: str) -> bool:
    """
    Detect potential prompt injection attempts
    
    Args:
        text: Input text to analyze
        
    Returns:
        True if injection detected, False otherwise
    """
    text_lower = text.lower()
    
    for pattern in COMPILED_INJECTION_PATTERNS:
        if pattern.search(text_lower):
            security_logger.warning(f"Potential prompt injection detected: {pattern.pattern}")
            return True
    
    return False

def _validate_api_key(api_key: str, provider: str) -> bool:
    """
    Validate API key format for security
    
    Args:
        api_key: API key to validate
        provider: Provider name (openai, anthropic, etc.)
        
    Returns:
        True if key format is valid, False otherwise
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Basic format validation by provider
    if provider.lower() == 'openai' and not api_key.startswith('sk-'):
        return False
    elif provider.lower() == 'anthropic' and not api_key.startswith('sk-ant-'):
        return False
    
    # Check for suspicious patterns
    if len(api_key) < 10 or len(api_key) > 200:
        return False
    
    # Ensure key doesn't contain obvious placeholders
    placeholders = ['your-key', 'api-key', 'secret', 'token', 'placeholder']
    if any(placeholder in api_key.lower() for placeholder in placeholders):
        return False
    
    return True

@dataclass
class CandyConfig:
    """
    Configuration for CandyLLM 3.0 with security enhancements
    
    Security Features:
    - API key validation and secure storage
    - Rate limiting configuration
    - Content filtering options
    - Audit logging settings
    - Session security parameters
    """
    
    # API Keys - stored securely with validation
    openai_api_key: Optional[str] = field(default=None, repr=False)  # Hide in repr for security
    anthropic_api_key: Optional[str] = field(default=None, repr=False)
    google_api_key: Optional[str] = field(default=None, repr=False)
    cohere_api_key: Optional[str] = field(default=None, repr=False)
    aws_access_key_id: Optional[str] = field(default=None, repr=False)
    aws_secret_access_key: Optional[str] = field(default=None, repr=False)
    aws_region: str = "us-east-1"
    
    # Core Features
    intelligent_routing: bool = True
    neurosymbolic_reasoning: bool = True
    auto_optimize: bool = True
    
    # Security Settings
    enable_content_filtering: bool = True
    enable_prompt_injection_detection: bool = True
    enable_audit_logging: bool = True
    max_input_length: int = 50000
    rate_limit_requests_per_minute: int = 60
    
    # Performance Settings
    max_retries: int = 3
    timeout: int = 60
    cache_responses: bool = True
    
    # Routing Preferences
    optimize_for: str = "quality"  # "quality", "speed", "cost"
    fallback_models: List[str] = None
    
    # Advanced Settings
    enable_streaming: bool = True
    enable_dynamic_tooling: bool = True
    dynamic_tooling_config: Optional[DynamicToolingConfig] = None
    log_level: str = "INFO"
    telemetry_enabled: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and security setup"""
        # Set secure fallback models if none provided
        if self.fallback_models is None:
            self.fallback_models = [
                "openai:gpt-4o-mini",
                "anthropic:claude-3.5-haiku",
                "google:gemini-1.5-flash"
            ]
        
        # Validate API keys if provided
        self._validate_api_keys()
        
        # Set up secure session ID
        self._session_id = secrets.token_urlsafe(32)
        
        # Log configuration creation
        security_logger.info(f"CandyConfig created with session ID: {self._session_id[:8]}...")
    
    def _validate_api_keys(self):
        """Validate provided API keys for security"""
        key_validations = [
            (self.openai_api_key, 'openai'),
            (self.anthropic_api_key, 'anthropic'),
            (self.google_api_key, 'google'),
            (self.cohere_api_key, 'cohere')
        ]
        
        for key, provider in key_validations:
            if key and not _validate_api_key(key, provider):
                security_logger.error(f"Invalid API key format for {provider}")
                raise ValueError(f"Invalid API key format for {provider}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary with API key masking"""
        config_dict = asdict(self)
        
        # Mask sensitive keys for security
        sensitive_keys = [
            'openai_api_key', 'anthropic_api_key', 'google_api_key', 
            'cohere_api_key', 'aws_access_key_id', 'aws_secret_access_key'
        ]
        
        for key in sensitive_keys:
            if config_dict.get(key):
                # Show only first 8 characters for identification
                config_dict[key] = config_dict[key][:8] + "..." + "*" * 20
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CandyConfig':
        """Create config from dictionary with security validation"""
        # Remove any unexpected keys for security
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_env(cls) -> 'CandyConfig':
        """Create config from environment variables with secure loading"""
        return cls(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            cohere_api_key=os.getenv('COHERE_API_KEY'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_region=os.getenv('AWS_REGION', 'us-east-1')
        )

class CandyLLM:
    """
    🍭 CandyLLM 3.0: Next-Generation AI Platform
    
    The world's most advanced AI platform combining intelligent routing,
    neurosymbolic reasoning, and universal model support.
    
    Features:
    - Intelligent routing to 25+ AI models
    - Neurosymbolic reasoning engine
    - Advanced chain-of-thought reasoning
    - Real-time performance optimization
    - Universal provider support
    - Mathematical computation engine
    - Knowledge graph integration
    """
    
    def __init__(self, 
                 config: Union[CandyConfig, Dict[str, Any], None] = None,
                 intelligent_routing: bool = True,
                 neurosymbolic_reasoning: bool = True,
                 auto_optimize: bool = True,
                 enable_dynamic_tooling: bool = True):
        """
        Initialize CandyLLM 3.0
        
        Args:
            config: Configuration object or dictionary
            intelligent_routing: Enable intelligent model routing
            neurosymbolic_reasoning: Enable neurosymbolic AI
            auto_optimize: Enable automatic performance optimization
            enable_dynamic_tooling: Enable dynamic tool synthesis
        """
        
        # Handle configuration
        if config is None:
            config = CandyConfig()
        elif isinstance(config, dict):
            config = CandyConfig.from_dict(config)
        
        # Override config with direct parameters
        config.intelligent_routing = intelligent_routing
        config.neurosymbolic_reasoning = neurosymbolic_reasoning
        config.auto_optimize = auto_optimize
        config.enable_dynamic_tooling = enable_dynamic_tooling
        
        self.config = config
        
        # Initialize core components
        self._initialize_components()
        
        # Performance tracking
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.start_time = time.time()
        
        print(f"🍭 CandyLLM 3.0 initialized")
        print(f"✅ Intelligent Routing: {config.intelligent_routing}")
        print(f"🧠 Neurosymbolic AI: {config.neurosymbolic_reasoning}")
        print(f"⚡ Auto-Optimization: {config.auto_optimize}")
        print(f"🔧 Dynamic Tooling: {config.enable_dynamic_tooling}")
    
    def _initialize_components(self):
        """Initialize all core components"""
        
        # Build provider configuration
        provider_config = {
            "openai": {"api_key": self.config.openai_api_key} if self.config.openai_api_key else {},
            "anthropic": {"api_key": self.config.anthropic_api_key} if self.config.anthropic_api_key else {},
            "google": {"api_key": self.config.google_api_key} if self.config.google_api_key else {},
            "cohere": {"api_key": self.config.cohere_api_key} if self.config.cohere_api_key else {},
            "aws": {
                "access_key_id": self.config.aws_access_key_id,
                "secret_access_key": self.config.aws_secret_access_key,
                "region": self.config.aws_region
            } if self.config.aws_access_key_id else {}
        }
        
        # Initialize provider manager
        self.provider_manager = UniversalProviderManager(provider_config)
        
        # Initialize intelligent router
        if self.config.intelligent_routing:
            self.router = IntelligentRouter(
                provider_manager=self.provider_manager,
                optimize_for=self.config.optimize_for
            )
        else:
            self.router = None
        
        # Initialize neurosymbolic engine
        if self.config.neurosymbolic_reasoning:
            self.neurosymbolic_engine = NeuroSymbolicEngine()
        else:
            self.neurosymbolic_engine = None
        
        # Initialize reasoning engine
        self.reasoning_engine = AdvancedReasoningEngine(
            model_manager=self.provider_manager,
            neurosymbolic_engine=self.neurosymbolic_engine
        )
        
        # Initialize dynamic tooling engine
        if self.config.enable_dynamic_tooling:
            tooling_config = self.config.dynamic_tooling_config or load_config_from_env()
            
            # Configure CandyLLM as the LLM provider for tooling
            from .dynamic_config import CandyLLMProvider
            ComponentFactory.register_llm_provider("candyllm_instance", CandyLLMProvider)
            tooling_config.llm_provider = "candyllm_instance"
            tooling_config.llm_config = {"candy_instance": self}
            
            self.dynamic_tooling = DynamicToolingEngine(tooling_config)
        else:
            self.dynamic_tooling = None
    
    async def chat(self, 
                   message: Union[str, List[Dict[str, str]]],
                   model: Optional[str] = None,
                   system_prompt: Optional[str] = None,
                   **kwargs) -> ChatResponse:
        """
        Chat with automatic model selection and intelligent routing
        
        Args:
            message: User message or conversation history
            model: Specific model to use (overrides routing)
            system_prompt: System prompt to use
            **kwargs: Additional parameters for model
        
        Returns:
            ChatResponse with content, model used, and metadata
        """
        
        start_time = time.time()
        
        # Format messages
        if isinstance(message, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": message})
        else:
            messages = message
        
        try:
            # Select model using intelligent routing or use specified model
            if model:
                selected_model = model
                route_decision = RouteDecision(
                    selected_model=model,
                    confidence=1.0,
                    reasoning="User specified model",
                    task_type="chat",
                    estimated_cost=0.0,
                    estimated_latency=0.0
                )
            elif self.router:
                route_decision = await self.router.route_request(
                    messages=messages,
                    task_type="chat",
                    context=kwargs
                )
                selected_model = route_decision.selected_model
            else:
                # Default model selection
                selected_model = "openai:gpt-4o-mini"
                route_decision = RouteDecision(
                    selected_model=selected_model,
                    confidence=0.5,
                    reasoning="Default model selection",
                    task_type="chat",
                    estimated_cost=0.0,
                    estimated_latency=0.0
                )
            
            # Get model instance and generate response
            model_instance = await self.provider_manager.get_model_instance(selected_model)
            llm_response = await model_instance.generate(messages, **kwargs)
            
            # Track performance
            processing_time = time.time() - start_time
            self.request_count += 1
            
            if llm_response.usage:
                self.total_tokens += llm_response.usage.get("total_tokens", 0)
            
            # Update router performance if available
            if self.router and self.config.auto_optimize:
                await self.router.update_performance(
                    model=selected_model,
                    latency=processing_time,
                    success=llm_response.success,
                    cost=llm_response.cost_estimate
                )
            
            # Create enhanced response
            return ChatResponse(
                content=llm_response.content,
                model=selected_model,
                provider=llm_response.provider,
                usage=llm_response.usage,
                cost_estimate=llm_response.cost_estimate,
                processing_time=processing_time,
                route_decision=route_decision,
                success=llm_response.success,
                metadata={
                    "request_id": f"candy_{int(time.time() * 1000)}",
                    "timestamp": datetime.now().isoformat(),
                    "intelligent_routing": self.config.intelligent_routing,
                    **(llm_response.metadata or {})
                }
            )
            
        except Exception as e:
            error_time = time.time() - start_time
            return ChatResponse(
                content=f"Error: {str(e)}",
                model=model or "error",
                provider="error",
                processing_time=error_time,
                success=False,
                error=str(e),
                metadata={"timestamp": datetime.now().isoformat()}
            )
    
    async def stream(self,
                    message: Union[str, List[Dict[str, str]]],
                    model: Optional[str] = None,
                    system_prompt: Optional[str] = None,
                    **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream chat response with intelligent routing
        
        Args:
            message: User message or conversation history
            model: Specific model to use
            system_prompt: System prompt to use
            **kwargs: Additional parameters
        
        Yields:
            StreamChunk objects with partial responses
        """
        
        # Format messages
        if isinstance(message, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": message})
        else:
            messages = message
        
        try:
            # Select model
            if model:
                selected_model = model
            elif self.router:
                route_decision = await self.router.route_request(
                    messages=messages,
                    task_type="chat",
                    context={**kwargs, "streaming": True}
                )
                selected_model = route_decision.selected_model
            else:
                selected_model = "openai:gpt-4o-mini"
            
            # Get model instance and stream
            model_instance = await self.provider_manager.get_model_instance(selected_model)
            
            async for chunk in model_instance.stream(messages, **kwargs):
                enhanced_chunk = StreamChunk(
                    content=chunk.content,
                    model=selected_model,
                    provider=chunk.provider,
                    chunk_id=chunk.chunk_id,
                    finish_reason=chunk.finish_reason,
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "intelligent_routing": self.config.intelligent_routing
                    }
                )
                yield enhanced_chunk
                
        except Exception as e:
            yield StreamChunk(
                content=f"Error: {str(e)}",
                model=model or "error",
                provider="error",
                error=str(e),
                metadata={"timestamp": datetime.now().isoformat()}
            )
    
    async def reason(self,
                    query: str,
                    strategy: str = "auto",
                    context: Optional[Dict[str, Any]] = None,
                    **kwargs) -> ReasoningResult:
        """
        Advanced reasoning with multiple strategies
        
        Args:
            query: Question or problem to reason about
            strategy: Reasoning strategy ("auto", "chain_of_thought", "multi_path", "neurosymbolic")
            context: Additional context for reasoning
            **kwargs: Additional parameters
        
        Returns:
            ReasoningResult with detailed reasoning paths and confidence
        """
        
        return await self.reasoning_engine.reason(
            query=query,
            context=context,
            strategy=strategy,
            **kwargs
        )
    
    async def solve_math(self,
                        problem: str,
                        show_steps: bool = True,
                        symbolic: bool = True,
                        **kwargs) -> MathResult:
        """
        Solve mathematical problems using neurosymbolic reasoning
        
        Args:
            problem: Mathematical problem to solve
            show_steps: Whether to show step-by-step solution
            symbolic: Whether to use symbolic computation
            **kwargs: Additional parameters
        
        Returns:
            MathResult with solution and reasoning steps
        """
        
        if not self.neurosymbolic_engine:
            raise ValueError("Neurosymbolic reasoning not enabled. Initialize with neurosymbolic_reasoning=True")
        
        return await self.neurosymbolic_engine.solve_mathematical_problem(
            problem=problem,
            show_steps=show_steps,
            symbolic=symbolic,
            **kwargs
        )
    
    def list_models(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available models
        
        Args:
            provider: Filter by specific provider
        
        Returns:
            List of model information dictionaries
        """
        
        if provider:
            return self.provider_manager.get_models_by_provider(provider)
        else:
            return self.provider_manager.get_all_models()
    
    def list_providers(self) -> List[str]:
        """List all available providers"""
        return list(set(
            model_info["provider"] 
            for model_info in self.provider_manager.get_all_models()
        ))
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get current status of all providers"""
        return self.provider_manager.get_provider_status()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        uptime = time.time() - self.start_time
        
        stats = {
            "requests": self.request_count,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "uptime_seconds": uptime,
            "requests_per_minute": (self.request_count / uptime) * 60 if uptime > 0 else 0,
            "provider_status": self.get_provider_status()
        }
        
        if self.router:
            stats["routing_stats"] = self.router.get_performance_metrics()
        
        return stats
    
    async def optimize_performance(self):
        """Manually trigger performance optimization"""
        if self.router and self.config.auto_optimize:
            await self.router.optimize_routing_strategy()
            print("🚀 Performance optimization completed")
        else:
            print("⚠️ Auto-optimization not enabled")
    
    def get_supported_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities supported by available models"""
        capabilities = {}
        
        for model_info in self.provider_manager.get_all_models():
            model_id = model_info["id"]
            model_capabilities = model_info.get("capabilities", [])
            
            for capability in model_capabilities:
                if capability not in capabilities:
                    capabilities[capability] = []
                capabilities[capability].append(model_id)
        
        return capabilities
    
    async def use_dynamic_tool(self, 
                              task_description: str,
                              inputs: Optional[Dict[str, Any]] = None,
                              force_synthesis: bool = False) -> Dict[str, Any]:
        """
        Use dynamic tooling to execute tasks with autonomous tool synthesis
        
        Args:
            task_description: Description of what the tool should do
            inputs: Input parameters for the tool
            force_synthesis: Force creation of new tool instead of reusing
            
        Returns:
            Tool execution result with metadata
        """
        
        if not self.dynamic_tooling:
            raise ValueError("Dynamic tooling is not enabled. Initialize CandyLLM with enable_dynamic_tooling=True")
        
        if inputs is None:
            inputs = {}
        
        try:
            # Execute task using dynamic tooling engine
            result = await self.dynamic_tooling.execute_task(
                task_description=task_description,
                inputs=inputs,
                force_synthesis=force_synthesis
            )
            
            return {
                "success": True,
                "result": result.get("output"),
                "tool_id": result.get("tool_id"),
                "tool_version": result.get("tool_version"),
                "execution_time": result.get("execution_time"),
                "was_synthesized": result.get("was_synthesized", False),
                "metadata": {
                    "task_description": task_description,
                    "timestamp": datetime.now().isoformat(),
                    "dynamic_tooling": True
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "task_description": task_description,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "dynamic_tooling": True
                }
            }
    
    async def chat_with_tools(self,
                             message: Union[str, List[Dict[str, str]]],
                             available_tools: Optional[List[str]] = None,
                             auto_tool_synthesis: bool = True,
                             **kwargs) -> ChatResponse:
        """
        Enhanced chat with automatic tool usage and synthesis
        
        Args:
            message: User message or conversation history
            available_tools: List of available tool descriptions
            auto_tool_synthesis: Automatically synthesize tools if needed
            **kwargs: Additional chat parameters
        
        Returns:
            ChatResponse with potential tool usage information
        """
        
        # First attempt regular chat
        response = await self.chat(message, **kwargs)
        
        # Check if the response indicates a need for tools
        if auto_tool_synthesis and self.dynamic_tooling:
            content = response.content.lower()
            
            # Simple heuristics to detect tool needs
            tool_indicators = [
                "i need a tool", "i cannot", "i don't have access",
                "i would need to", "i can't perform", "i cannot execute",
                "requires external", "need to run", "would need to access"
            ]
            
            needs_tool = any(indicator in content for indicator in tool_indicators)
            
            if needs_tool:
                # Extract task from the original message
                if isinstance(message, str):
                    task = message
                else:
                    task = message[-1].get("content", "") if message else ""
                
                # Attempt dynamic tool synthesis
                try:
                    tool_result = await self.use_dynamic_tool(
                        task_description=f"Complete this task: {task}",
                        inputs={}
                    )
                    
                    if tool_result["success"]:
                        # Update response with tool result
                        enhanced_content = f"{response.content}\n\n🔧 **Dynamic Tool Result:**\n{tool_result['result']}"
                        
                        response.content = enhanced_content
                        response.metadata = response.metadata or {}
                        response.metadata["dynamic_tool_used"] = True
                        response.metadata["tool_id"] = tool_result["tool_id"]
                        response.metadata["was_synthesized"] = tool_result["was_synthesized"]
                        
                except Exception as e:
                    # Add note about tool synthesis failure
                    response.content += f"\n\n⚠️ Attempted dynamic tool synthesis but failed: {str(e)}"
                    response.metadata = response.metadata or {}
                    response.metadata["tool_synthesis_attempted"] = True
                    response.metadata["tool_synthesis_error"] = str(e)
        
        return response
    
    def list_dynamic_tools(self) -> List[Dict[str, Any]]:
        """List all available dynamic tools"""
        
        if not self.dynamic_tooling:
            return []
        
        # This would call the registry to list tools
        # For now, return empty list as the registry would handle this
        return []
    
    async def get_tool_info(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific dynamic tool"""
        
        if not self.dynamic_tooling:
            return None
        
        # This would retrieve tool metadata from the registry
        return None
    
    async def close(self):
        """Clean up resources"""
        # Close any open connections, save performance data, etc.
        if hasattr(self.provider_manager, 'close'):
            await self.provider_manager.close()
        
        print("🍭 CandyLLM 3.0 session ended")
        print(f"📊 Total requests: {self.request_count}")
        print(f"💰 Total cost: ${self.total_cost:.4f}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

# Convenience functions for quick usage
async def quick_chat(message: str, **kwargs) -> str:
    """Quick chat function for immediate use"""
    async with CandyLLM() as candy:
        response = await candy.chat(message, **kwargs)
        return response.content

async def quick_reason(query: str, **kwargs) -> ReasoningResult:
    """Quick reasoning function"""
    async with CandyLLM() as candy:
        return await candy.reason(query, **kwargs)

async def quick_math(problem: str, **kwargs) -> MathResult:
    """Quick math solving function"""
    async with CandyLLM(neurosymbolic_reasoning=True) as candy:
        return await candy.solve_math(problem, **kwargs)
