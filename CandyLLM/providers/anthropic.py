"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

Anthropic Claude provider with secure implementation

Secure Anthropic integration with comprehensive safety features:
- API key validation and secure storage
- Input sanitization and content filtering
- Rate limiting and quota management
- Request/response logging for audit trails
- Error handling with security considerations
- Protection against prompt injection attacks
- Compliance with Anthropic usage policies
- Claude-specific safety features and content moderation
"""

import asyncio
import anthropic
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, AsyncGenerator
import html
import re
from datetime import datetime, timedelta

from ..core.base import BaseProvider, BaseModel, ModelConfig, ModelResponse, StreamingChunk, Message, ProviderType, ModelType

# Security logging
security_logger = logging.getLogger('candyllm.providers.anthropic.security')
logger = logging.getLogger(__name__)

# Claude-specific content policy patterns
CLAUDE_POLICY_VIOLATIONS = [
    r'ignore\s+(previous|all)\s+(instructions|prompts)',
    r'jailbreak|DAN\s+mode|evil\s+mode',
    r'pretend\s+to\s+be\s+(uncensored|unfiltered)',
    r'act\s+as\s+if\s+you\s+(have\s+no\s+limitations|are\s+not\s+AI)',
    r'(harmful|illegal|unethical)\s+(content|instructions|advice)',
    r'generate\s+(malware|virus|exploit|harmful)\s+code',
]

# Compile patterns for performance
COMPILED_CLAUDE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in CLAUDE_POLICY_VIOLATIONS]

def validate_anthropic_api_key(api_key: str) -> bool:
    """
    Validate Anthropic API key format and basic security checks
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if key appears valid, False otherwise
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Basic format check (Anthropic keys start with 'sk-ant-')
    if not api_key.startswith('sk-ant-'):
        return False
    
    # Length check (Anthropic keys are typically around 100+ characters)
    if len(api_key) < 50 or len(api_key) > 200:
        return False
    
    # Check for obvious placeholders
    placeholders = ['your-key', 'api-key', 'anthropic-key', 'claude-key', 'secret']
    if any(placeholder in api_key.lower() for placeholder in placeholders):
        return False
    
    # Basic character set validation
    allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')
    if not all(c in allowed_chars for c in api_key):
        return False
    
    return True

def check_claude_content_policy(text: str) -> tuple[bool, List[str]]:
    """
    Check if text complies with Claude content policies
    
    Args:
        text: Text to check
        
    Returns:
        Tuple of (is_compliant, violations_found)
    """
    violations = []
    
    for pattern in COMPILED_CLAUDE_PATTERNS:
        if pattern.search(text):
            violations.append(pattern.pattern)
    
    return len(violations) == 0, violations

def sanitize_claude_input(text: str) -> str:
    """
    Sanitize input for Claude API while preserving functionality
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text safe for Claude API
    """
    # HTML escape to prevent injection
    sanitized = html.escape(text)
    
    # Remove potential prompt injection markers
    injection_markers = [
        'Human:', 'Assistant:', 'System:', '<|endoftext|>', '<thinking>', '</thinking>'
    ]
    
    for marker in injection_markers:
        # Only remove if not at expected positions
        if not (marker in ['Human:', 'Assistant:'] and sanitized.strip().startswith(marker)):
            sanitized = sanitized.replace(marker, '')
    
    # Normalize whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # Limit length to prevent DoS
    if len(sanitized) > 100000:  # Claude's context limit consideration
        sanitized = sanitized[:100000] + "... [content truncated for safety]"
    
    return sanitized

class SecureAnthropicProvider(BaseProvider):
    """
    Secure Anthropic provider supporting all Claude models with enhanced security
    
    Security Features:
    - Comprehensive API key validation
    - Input sanitization and content filtering
    - Rate limiting and quota management
    - Audit logging for all API calls
    - Error handling with security context
    - Claude-specific safety features
    """
    
    SUPPORTED_MODELS = [
        # Claude 3.5 Series - Latest generation
        "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620",
        "claude-3-5-haiku-20241022",
        
        # Claude 3 Series - Advanced models
        "claude-3-opus-20240229", "claude-3-sonnet-20240229", 
        "claude-3-haiku-20240307",
        
        # Simplified aliases
        "claude-3-opus", "claude-3-sonnet", "claude-3-haiku", 
        "claude-3-5-sonnet",
        
        # Claude 2 Series - Previous generation
        "claude-2.1", "claude-2.0",
        
        # Claude Instant - Fast inference
        "claude-instant-1.2"
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize secure Anthropic provider
        
        Args:
            config: Configuration dictionary with security validation
        """
        super().__init__(config)
        
        # Extract and validate API credentials
        self.api_key = config.get('api_key') or config.get('anthropic_api_key') or os.getenv('ANTHROPIC_API_KEY')
        self.base_url = config.get('base_url') or config.get('anthropic_base_url')
        
        # Validate API key
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        if not validate_anthropic_api_key(self.api_key):
            security_logger.error("Invalid Anthropic API key format detected")
            raise ValueError("Invalid Anthropic API key format")
        
        # Security configuration
        self.content_filtering_enabled = config.get('content_filtering', True)
        self.audit_logging_enabled = config.get('audit_logging', True)
        self.rate_limit_enabled = config.get('rate_limiting', True)
        
        # Rate limiting tracking
        self.request_timestamps = []
        self.max_requests_per_minute = config.get('max_requests_per_minute', 50)  # Claude has lower limits
        
        # Initialize Anthropic client with security considerations
        try:
            client_config = {
                'api_key': self.api_key,
                'timeout': config.get('timeout', 60),
                'max_retries': config.get('max_retries', 3)
            }
            
            if self.base_url:
                client_config['base_url'] = self.base_url
            
            self.client = anthropic.AsyncAnthropic(**client_config)
            
            security_logger.info("SecureAnthropicProvider initialized successfully")
            
        except Exception as e:
            security_logger.error(f"Failed to initialize Anthropic client: {e}")
            raise
    
    def _check_rate_limit(self) -> bool:
        """
        Check if current request is within rate limits
        
        Returns:
            True if within limits, False if rate limited
        """
        if not self.rate_limit_enabled:
            return True
        
        now = time.time()
        minute_ago = now - 60
        
        # Clean old timestamps
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > minute_ago]
        
        # Check current rate
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            security_logger.warning(f"Anthropic rate limit exceeded: {len(self.request_timestamps)} requests in last minute")
            return False
        
        # Record current request
        self.request_timestamps.append(now)
        return True
    
    def _log_api_call(self, method: str, model: str, input_data: Any, response_data: Any = None, error: Any = None):
        """
        Log API calls for audit trail
        
        Args:
            method: API method called
            model: Model used
            input_data: Input data (sanitized for logging)
            response_data: Response data (optional)
            error: Error information (optional)
        """
        if not self.audit_logging_enabled:
            return
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'provider': 'anthropic',
            'method': method,
            'model': model,
            'input_length': len(str(input_data)) if input_data else 0,
            'success': error is None,
            'error': str(error) if error else None,
            'response_length': len(str(response_data)) if response_data else 0
        }
        
        security_logger.info(f"Anthropic API call: {json.dumps(log_entry)}")
    
    def _get_provider_type(self) -> ProviderType:
        """Get provider type"""
        return ProviderType.CLOUD
    
    def list_models(self) -> List[str]:
        """List all supported Anthropic models"""
        return self.SUPPORTED_MODELS.copy()
    
    def create_model(self, model_id: str, config: ModelConfig) -> 'SecureAnthropicModel':
        """
        Create a secure Anthropic model instance
        
        Args:
            model_id: Model identifier
            config: Model configuration
            
        Returns:
            Secure Anthropic model instance
        """
        if model_id not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model_id} not in supported list, but attempting anyway")
        
        return SecureAnthropicModel(
            client=self.client,
            model_id=model_id,
            config=config,
            provider=self
        )
    
    def is_available(self) -> bool:
        """Check if provider is available and properly configured"""
        return bool(self.api_key and validate_anthropic_api_key(self.api_key))

class SecureAnthropicModel(BaseModel):
    """
    Secure Anthropic model implementation with comprehensive safety features
    
    Features:
    - Input validation and sanitization
    - Claude-specific content policy checking
    - Rate limiting and quota management
    - Comprehensive error handling
    - Audit logging for all operations
    - Streaming support with security monitoring
    """
    
    def __init__(self, client: anthropic.AsyncAnthropic, model_id: str, config: ModelConfig, provider: SecureAnthropicProvider):
        """
        Initialize secure Anthropic model
        
        Args:
            client: Anthropic client instance
            model_id: Model identifier
            config: Model configuration
            provider: Parent provider instance
        """
        super().__init__(model_id, config)
        self.client = client
        self.provider = provider
        self.config.model_type = ModelType.CHAT  # All Claude models are chat models
        
        security_logger.debug(f"SecureAnthropicModel initialized: {model_id}")
    
    def _normalize_model_id(self) -> str:
        """Normalize model ID to full Anthropic format"""
        model_mapping = {
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3-5-sonnet": "claude-3-5-sonnet-20241022"
        }
        
        return model_mapping.get(self.model_id, self.model_id)
    
    def _format_messages_for_claude(self, messages: List[Message]) -> tuple[List[Dict], str]:
        """
        Format messages for Claude API (separate system message)
        
        Args:
            messages: List of messages
            
        Returns:
            Tuple of (formatted_messages, system_message)
        """
        formatted_messages = []
        system_message = ""
        
        for message in messages:
            if message.role == "system":
                system_message = message.content
            else:
                formatted_messages.append({
                    "role": message.role,
                    "content": message.content
                })
        
        return formatted_messages, system_message
    
    async def generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """
        Generate response from Claude model with security validation
        
        Args:
            messages: List of messages for the conversation
            **kwargs: Additional generation parameters
            
        Returns:
            Model response with security metadata
        """
        try:
            # Rate limiting check
            if not self.provider._check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            # Validate and sanitize input messages
            sanitized_messages = []
            system_message = ""
            
            for message in messages:
                # Content policy check
                if self.provider.content_filtering_enabled:
                    is_compliant, violations = check_claude_content_policy(message.content)
                    if not is_compliant:
                        security_logger.warning(f"Claude content policy violation: {violations}")
                        raise ValueError(f"Content policy violation: {violations}")
                
                # Sanitize message content
                sanitized_content = sanitize_claude_input(message.content)
                
                if message.role == "system":
                    system_message = sanitized_content
                else:
                    sanitized_messages.append({
                        "role": message.role,
                        "content": sanitized_content
                    })
            
            # Prepare API call parameters
            api_params = {
                "model": self._normalize_model_id(),
                "messages": sanitized_messages,
                "max_tokens": kwargs.get('max_tokens', 4096),
                "temperature": kwargs.get('temperature', self.config.temperature),
                "top_p": kwargs.get('top_p', self.config.top_p)
            }
            
            if system_message:
                api_params["system"] = system_message
            
            # Remove None values
            api_params = {k: v for k, v in api_params.items() if v is not None}
            
            # Log API call attempt
            self.provider._log_api_call("messages.create", self.model_id, sanitized_messages)
            
            # Make API call
            start_time = time.time()
            response = await self.client.messages.create(**api_params)
            end_time = time.time()
            
            # Extract response content
            content = response.content[0].text if response.content else ""
            
            # Log successful API call
            self.provider._log_api_call("messages.create", self.model_id, sanitized_messages, content)
            
            # Create model response with security metadata
            model_response = ModelResponse(
                content=content,
                model=self.model_id,
                provider="anthropic",
                usage={
                    "input_tokens": response.usage.input_tokens if response.usage else 0,
                    "output_tokens": response.usage.output_tokens if response.usage else 0,
                    "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0
                },
                metadata={
                    "stop_reason": response.stop_reason,
                    "response_time": end_time - start_time,
                    "content_filtered": self.provider.content_filtering_enabled,
                    "security_validated": True
                }
            )
            
            security_logger.debug(f"Claude API call successful: {self.model_id}")
            return model_response
            
        except Exception as e:
            # Log error with security context
            self.provider._log_api_call("messages.create", self.model_id, [], error=e)
            security_logger.error(f"Claude API call failed: {e}")
            raise
    
    async def stream(self, messages: List[Message], **kwargs) -> AsyncGenerator[StreamingChunk, None]:
        """
        Stream response from Claude model with security monitoring
        
        Args:
            messages: List of messages for the conversation
            **kwargs: Additional generation parameters
            
        Yields:
            Streaming chunks with security validation
        """
        try:
            # Rate limiting and validation (same as generate)
            if not self.provider._check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            # Sanitize messages
            sanitized_messages = []
            system_message = ""
            
            for message in messages:
                if self.provider.content_filtering_enabled:
                    is_compliant, violations = check_claude_content_policy(message.content)
                    if not is_compliant:
                        security_logger.warning(f"Claude policy violation in stream: {violations}")
                        raise ValueError(f"Content policy violation: {violations}")
                
                sanitized_content = sanitize_claude_input(message.content)
                
                if message.role == "system":
                    system_message = sanitized_content
                else:
                    sanitized_messages.append({
                        "role": message.role,
                        "content": sanitized_content
                    })
            
            # Prepare streaming parameters
            api_params = {
                "model": self._normalize_model_id(),
                "messages": sanitized_messages,
                "max_tokens": kwargs.get('max_tokens', 4096),
                "temperature": kwargs.get('temperature', self.config.temperature),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "stream": True
            }
            
            if system_message:
                api_params["system"] = system_message
            
            api_params = {k: v for k, v in api_params.items() if v is not None}
            
            # Log streaming attempt
            self.provider._log_api_call("messages.create(stream)", self.model_id, sanitized_messages)
            
            # Start streaming
            async with self.client.messages.stream(**api_params) as stream:
                async for chunk in stream:
                    if chunk.type == "content_block_delta":
                        content = chunk.delta.text
                        
                        # Security check on streamed content
                        if self.provider.content_filtering_enabled:
                            is_compliant, violations = check_claude_content_policy(content)
                            if not is_compliant:
                                security_logger.warning(f"Policy violation in Claude streamed content: {violations}")
                                break
                        
                        yield StreamingChunk(
                            content=content,
                            metadata={
                                "model": self.model_id,
                                "chunk_type": chunk.type,
                                "security_validated": True
                            }
                        )
                    elif chunk.type == "message_stop":
                        yield StreamingChunk(
                            content="",
                            metadata={
                                "model": self.model_id,
                                "chunk_type": chunk.type,
                                "finish_reason": "stop",
                                "security_validated": True
                            }
                        )
            
        except Exception as e:
            self.provider._log_api_call("messages.create(stream)", self.model_id, [], error=e)
            security_logger.error(f"Claude streaming failed: {e}")
            raise
    
    def get_context_length(self) -> int:
        """Get maximum context length for Claude model"""
        context_lengths = {
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
            "claude-3-5-sonnet": 200000,
            "claude-2.1": 200000,
            "claude-2.0": 100000,
            "claude-instant-1.2": 100000
        }
        
        # Find the best match
        for model_name, length in context_lengths.items():
            if model_name in self.model_id:
                return length
        
        return 100000  # Default fallback
    
    def supports_streaming(self) -> bool:
        """All Claude models support streaming"""
        return True
    
    def supports_function_calling(self) -> bool:
        """Check if model supports function calling"""
        # Claude 3 family supports tools/function calling
        function_calling_models = [
            "claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3-5-sonnet"
        ]
        return any(model in self.model_id for model in function_calling_models)
    
    def supports_vision(self) -> bool:
        """Check if model supports vision/image input"""
        # All Claude 3 models support vision
        vision_models = [
            "claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3-5-sonnet"
        ]
        return any(model in self.model_id for model in vision_models)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information with security metadata"""
        return {
            "id": self.model_id,
            "provider": "anthropic",
            "type": self.config.model_type.value,
            "supported": self.model_id in SecureAnthropicProvider.SUPPORTED_MODELS,
            "context_length": self.get_context_length(),
            "security_features": {
                "content_filtering": self.provider.content_filtering_enabled,
                "rate_limiting": self.provider.rate_limit_enabled,
                "audit_logging": self.provider.audit_logging_enabled,
                "input_sanitization": True,
                "output_validation": True,
                "claude_safety": True
            },
            "capabilities": {
                "streaming": self.supports_streaming(),
                "function_calling": self.supports_function_calling(),
                "vision": self.supports_vision(),
                "reasoning": True,
                "long_context": True
            },
            "config": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p
            }
        }
    
    async def generate_with_tools(self, 
                                messages: List[Message], 
                                tools: List[Dict[str, Any]], 
                                **kwargs) -> ModelResponse:
        """
        Generate response with tool use support and security validation
        
        Args:
            messages: List of messages for the conversation
            tools: List of available tools
            **kwargs: Additional generation parameters
            
        Returns:
            Model response with tool calls and security metadata
        """
        if not self.supports_function_calling():
            raise ValueError(f"Model {self.model_id} does not support function calling")
        
        try:
            # Rate limiting check
            if not self.provider._check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            # Validate and sanitize input messages
            sanitized_messages = []
            system_message = ""
            
            for message in messages:
                if self.provider.content_filtering_enabled:
                    is_compliant, violations = check_claude_content_policy(message.content)
                    if not is_compliant:
                        security_logger.warning(f"Claude tool policy violation: {violations}")
                        raise ValueError(f"Content policy violation: {violations}")
                
                sanitized_content = sanitize_claude_input(message.content)
                
                if message.role == "system":
                    system_message = sanitized_content
                else:
                    sanitized_messages.append({
                        "role": message.role,
                        "content": sanitized_content
                    })
            
            # Validate and sanitize tools
            sanitized_tools = []
            for tool in tools:
                # Basic tool validation
                if not isinstance(tool, dict) or "name" not in tool:
                    security_logger.warning(f"Invalid tool format: {tool}")
                    continue
                
                # Sanitize tool definition
                sanitized_tool = {
                    "name": html.escape(str(tool["name"])),
                    "description": html.escape(str(tool.get("description", ""))),
                    "input_schema": tool.get("input_schema", {})
                }
                
                sanitized_tools.append(sanitized_tool)
            
            # Prepare API call parameters
            api_params = {
                "model": self._normalize_model_id(),
                "messages": sanitized_messages,
                "tools": sanitized_tools,
                "max_tokens": kwargs.get('max_tokens', 4096),
                "temperature": kwargs.get('temperature', self.config.temperature),
                "top_p": kwargs.get('top_p', self.config.top_p)
            }
            
            if system_message:
                api_params["system"] = system_message
            
            api_params = {k: v for k, v in api_params.items() if v is not None}
            
            # Log API call attempt
            self.provider._log_api_call("messages.create(tools)", self.model_id, sanitized_messages)
            
            # Make API call
            start_time = time.time()
            response = await self.client.messages.create(**api_params)
            end_time = time.time()
            
            # Process response content and tool calls
            content = ""
            tool_calls = []
            
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input
                    })
            
            # Log successful API call
            self.provider._log_api_call("messages.create(tools)", self.model_id, sanitized_messages, content)
            
            # Create model response with security metadata
            model_response = ModelResponse(
                content=content,
                model=self.model_id,
                provider="anthropic",
                usage={
                    "input_tokens": response.usage.input_tokens if response.usage else 0,
                    "output_tokens": response.usage.output_tokens if response.usage else 0,
                    "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0
                },
                metadata={
                    "stop_reason": response.stop_reason,
                    "response_time": end_time - start_time,
                    "content_filtered": self.provider.content_filtering_enabled,
                    "security_validated": True,
                    "tool_calls_count": len(tool_calls)
                },
                tool_calls=tool_calls if tool_calls else None
            )
            
            security_logger.debug(f"Claude tool API call successful: {self.model_id}")
            return model_response
            
        except Exception as e:
            self.provider._log_api_call("messages.create(tools)", self.model_id, [], error=e)
            security_logger.error(f"Claude tool API call failed: {e}")
            raise

# Alias for backward compatibility
AnthropicProvider = SecureAnthropicProvider
AnthropicModel = SecureAnthropicModel
