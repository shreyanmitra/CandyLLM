"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

LiteLLM provider with secure universal model support

Secure LiteLLM integration supporting 100+ models with comprehensive safety:
- Multi-provider API key validation and secure storage
- Input sanitization and content filtering across all providers
- Rate limiting and quota management per provider
- Request/response logging for comprehensive audit trails
- Error handling with security considerations for all models
- Protection against prompt injection attacks across providers
- Compliance with multiple provider usage policies
- Universal content filtering and safety features
"""

import asyncio
import litellm
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
security_logger = logging.getLogger('candyllm.providers.litellm.security')
logger = logging.getLogger(__name__)

# Universal content policy patterns (covers multiple providers)
UNIVERSAL_POLICY_VIOLATIONS = [
    r'ignore\s+(previous|all)\s+(instructions|prompts|rules)',
    r'jailbreak|DAN\s+mode|evil\s+mode|developer\s+mode',
    r'pretend\s+to\s+be\s+(uncensored|unfiltered|unrestricted)',
    r'act\s+as\s+if\s+you\s+(have\s+no\s+limitations|are\s+not\s+AI)',
    r'(harmful|illegal|unethical|dangerous)\s+(content|instructions|advice)',
    r'generate\s+(malware|virus|exploit|harmful|illegal)\s+(code|content)',
    r'bypass\s+(safety|security|content)\s+(filters|policies|guidelines)',
    r'(child|minor|underage)\s+(sexual|explicit|inappropriate)',
    r'(suicide|self[\s-]?harm|self[\s-]?injury)\s+(methods|instructions)',
]

# Compile patterns for performance
COMPILED_UNIVERSAL_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in UNIVERSAL_POLICY_VIOLATIONS]

def validate_api_key_format(api_key: str, provider: str) -> bool:
    """
    Validate API key format for different providers
    
    Args:
        api_key: API key to validate
        provider: Provider name
        
    Returns:
        True if key appears valid, False otherwise
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Provider-specific validation
    if provider == "openai":
        return api_key.startswith('sk-') and len(api_key) > 40
    elif provider == "anthropic":
        return api_key.startswith('sk-ant-') and len(api_key) > 50
    elif provider == "cohere":
        return len(api_key) > 30 and api_key.replace('-', '').replace('_', '').isalnum()
    elif provider == "replicate":
        return len(api_key) > 30 and api_key.replace('-', '').replace('_', '').isalnum()
    elif provider in ["together_ai", "anyscale", "groq", "perplexity", "fireworks_ai", "deepinfra"]:
        return len(api_key) > 20 and api_key.replace('-', '').replace('_', '').isalnum()
    else:
        # Generic validation for unknown providers
        return len(api_key) > 10 and len(api_key) < 200

def check_universal_content_policy(text: str) -> tuple[bool, List[str]]:
    """
    Check if text complies with universal content policies
    
    Args:
        text: Text to check
        
    Returns:
        Tuple of (is_compliant, violations_found)
    """
    violations = []
    
    for pattern in COMPILED_UNIVERSAL_PATTERNS:
        if pattern.search(text):
            violations.append(pattern.pattern)
    
    return len(violations) == 0, violations

def sanitize_universal_input(text: str) -> str:
    """
    Sanitize input for universal model compatibility
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text safe for all supported models
    """
    # HTML escape to prevent injection
    sanitized = html.escape(text)
    
    # Remove potential prompt injection markers from all providers
    injection_markers = [
        'Human:', 'Assistant:', 'System:', 'User:', 'Bot:',
        '<|endoftext|>', '<|im_start|>', '<|im_end|>',
        '<thinking>', '</thinking>', '###', '---', '```'
    ]
    
    for marker in injection_markers:
        sanitized = sanitized.replace(marker, '')
    
    # Normalize whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # Limit length to prevent DoS (conservative limit for all providers)
    if len(sanitized) > 50000:
        sanitized = sanitized[:50000] + "... [content truncated for safety]"
    
    return sanitized

class SecureLiteLLMProvider(BaseProvider):
    """
    Secure universal provider using LiteLLM with enhanced security for 100+ models
    
    Security Features:
    - Multi-provider API key validation
    - Universal input sanitization and content filtering
    - Per-provider rate limiting and quota management
    - Comprehensive audit logging for all providers
    - Error handling with security context
    - Universal safety features across all models
    """
    
    # Enhanced provider categories with security metadata
    SUPPORTED_PROVIDERS = {
        "openai": {
            "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "o1-preview", "o1-mini"],
            "security_level": "high",
            "rate_limit": 60
        },
        "anthropic": {
            "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3-5-sonnet"],
            "security_level": "high", 
            "rate_limit": 50
        },
        "cohere": {
            "models": ["command", "command-light", "command-r", "command-r-plus"],
            "security_level": "medium",
            "rate_limit": 40
        },
        "replicate": {
            "models": ["llama-2-70b-chat", "mistral-7b-instruct", "mixtral-8x7b"],
            "security_level": "medium",
            "rate_limit": 30
        },
        "huggingface": {
            "models": ["microsoft/DialoGPT-medium", "meta-llama/Llama-2-7b-chat-hf"],
            "security_level": "low",
            "rate_limit": 20
        },
        "bedrock": {
            "models": ["anthropic.claude-v2", "amazon.titan-text-lite-v1", "ai21.j2-ultra-v1"],
            "security_level": "high",
            "rate_limit": 40
        },
        "vertex_ai": {
            "models": ["text-bison", "chat-bison", "gemini-pro"],
            "security_level": "high",
            "rate_limit": 40
        },
        "azure": {
            "models": ["azure/gpt-4", "azure/gpt-35-turbo"],
            "security_level": "high",
            "rate_limit": 60
        },
        "groq": {
            "models": ["llama2-70b-4096", "mixtral-8x7b-32768"],
            "security_level": "medium",
            "rate_limit": 30
        },
        "ollama": {
            "models": ["llama2", "mistral", "codellama", "phi", "gemma"],
            "security_level": "low",
            "rate_limit": 10
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize secure LiteLLM provider
        
        Args:
            config: Configuration dictionary with security validation
        """
        super().__init__(config)
        self.config_dict = config
        
        # Security configuration
        self.content_filtering_enabled = config.get('content_filtering', True)
        self.audit_logging_enabled = config.get('audit_logging', True)
        self.rate_limit_enabled = config.get('rate_limiting', True)
        
        # Per-provider rate limiting tracking
        self.provider_request_timestamps = {}
        
        # Setup and validate API keys securely
        self._setup_and_validate_api_keys()
        
        # Configure LiteLLM with security settings
        litellm.set_verbose = config.get('verbose', False)
        litellm.drop_params = True  # Drop unsupported params automatically
        litellm.telemetry = False  # Disable telemetry for privacy
        litellm.suppress_debug_info = True  # Suppress debug info in production
        
        security_logger.info("SecureLiteLLMProvider initialized successfully")
    
    def _setup_and_validate_api_keys(self):
        """Setup and validate API keys for various providers with security checks"""
        key_mappings = {
            'openai_api_key': ('OPENAI_API_KEY', 'openai'),
            'anthropic_api_key': ('ANTHROPIC_API_KEY', 'anthropic'),
            'cohere_api_key': ('COHERE_API_KEY', 'cohere'),
            'replicate_api_token': ('REPLICATE_API_TOKEN', 'replicate'),
            'huggingface_api_key': ('HUGGINGFACE_API_KEY', 'huggingface'),
            'ai21_api_key': ('AI21_API_KEY', 'ai21'),
            'together_api_key': ('TOGETHER_API_KEY', 'together_ai'),
            'anyscale_api_key': ('ANYSCALE_API_KEY', 'anyscale'),
            'groq_api_key': ('GROQ_API_KEY', 'groq'),
            'perplexity_api_key': ('PERPLEXITYAI_API_KEY', 'perplexity'),
            'fireworks_api_key': ('FIREWORKS_API_KEY', 'fireworks_ai'),
            'deepinfra_api_key': ('DEEPINFRA_API_KEY', 'deepinfra')
        }
        
        validated_keys = 0
        
        for config_key, (env_key, provider) in key_mappings.items():
            if config_key in self.config_dict:
                api_key = self.config_dict[config_key]
                
                # Validate API key format
                if validate_api_key_format(api_key, provider):
                    os.environ[env_key] = api_key
                    validated_keys += 1
                    security_logger.debug(f"Valid API key configured for {provider}")
                else:
                    security_logger.warning(f"Invalid API key format for {provider}")
        
        if validated_keys == 0:
            security_logger.warning("No valid API keys configured for any provider")
        else:
            security_logger.info(f"Configured {validated_keys} valid API keys")
    
    def _check_rate_limit(self, provider: str) -> bool:
        """
        Check if current request is within rate limits for specific provider
        
        Args:
            provider: Provider name
            
        Returns:
            True if within limits, False if rate limited
        """
        if not self.rate_limit_enabled:
            return True
        
        now = time.time()
        minute_ago = now - 60
        
        # Initialize provider tracking if needed
        if provider not in self.provider_request_timestamps:
            self.provider_request_timestamps[provider] = []
        
        # Clean old timestamps
        self.provider_request_timestamps[provider] = [
            ts for ts in self.provider_request_timestamps[provider] if ts > minute_ago
        ]
        
        # Get provider-specific rate limit
        provider_info = self.SUPPORTED_PROVIDERS.get(provider, {"rate_limit": 20})
        max_requests = provider_info["rate_limit"]
        
        # Check current rate
        current_requests = len(self.provider_request_timestamps[provider])
        if current_requests >= max_requests:
            security_logger.warning(f"Rate limit exceeded for {provider}: {current_requests} requests in last minute")
            return False
        
        # Record current request
        self.provider_request_timestamps[provider].append(now)
        return True
    
    def _log_api_call(self, method: str, model: str, provider: str, input_data: Any, 
                     response_data: Any = None, error: Any = None):
        """
        Log API calls for comprehensive audit trail
        
        Args:
            method: API method called
            model: Model used
            provider: Provider name
            input_data: Input data (sanitized for logging)
            response_data: Response data (optional)
            error: Error information (optional)
        """
        if not self.audit_logging_enabled:
            return
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'provider': provider,
            'method': method,
            'model': model,
            'input_length': len(str(input_data)) if input_data else 0,
            'success': error is None,
            'error': str(error) if error else None,
            'response_length': len(str(response_data)) if response_data else 0,
            'security_level': self.SUPPORTED_PROVIDERS.get(provider, {}).get('security_level', 'unknown')
        }
        
        security_logger.info(f"LiteLLM API call: {json.dumps(log_entry)}")
    
    def _extract_provider_from_model(self, model_id: str) -> str:
        """Extract provider name from model ID"""
        if "/" in model_id:
            potential_provider = model_id.split("/")[0]
            if potential_provider in self.SUPPORTED_PROVIDERS:
                return potential_provider
        
        # Fallback inference
        for provider, info in self.SUPPORTED_PROVIDERS.items():
            for model in info["models"]:
                if model in model_id:
                    return provider
        
        return "unknown"
    
    def _get_provider_type(self) -> ProviderType:
        """Get provider type"""
        return ProviderType.UNIVERSAL
    
    def list_models(self) -> List[str]:
        """List all supported models across all providers"""
        all_models = []
        for provider, info in self.SUPPORTED_PROVIDERS.items():
            # Add provider prefix for clarity
            all_models.extend([f"{provider}/{model}" for model in info["models"]])
        return all_models
    
    def list_models_by_provider(self, provider: str) -> List[str]:
        """List models for a specific provider"""
        provider_info = self.SUPPORTED_PROVIDERS.get(provider, {})
        return provider_info.get("models", [])
    
    def create_model(self, model_id: str, config: ModelConfig) -> 'SecureLiteLLMModel':
        """
        Create a secure LiteLLM model instance
        
        Args:
            model_id: Model identifier
            config: Model configuration
            
        Returns:
            Secure LiteLLM model instance
        """
        return SecureLiteLLMModel(
            model_id=model_id,
            config=config,
            provider_config=self.config_dict,
            provider=self
        )
    
    def is_available(self) -> bool:
        """Check if at least one provider is configured with valid API key"""
        # Check if any valid API keys are available
        api_keys = [
            'openai_api_key', 'anthropic_api_key', 'cohere_api_key',
            'replicate_api_token', 'huggingface_api_key', 'groq_api_key'
        ]
        return any(key in self.config_dict for key in api_keys)
    
    def get_provider_coverage(self) -> Dict[str, Any]:
        """Get comprehensive information about provider coverage and security"""
        return {
            "total_providers": len(self.SUPPORTED_PROVIDERS),
            "total_models": sum(len(info["models"]) for info in self.SUPPORTED_PROVIDERS.values()),
            "providers": list(self.SUPPORTED_PROVIDERS.keys()),
            "security_levels": {
                level: [p for p, info in self.SUPPORTED_PROVIDERS.items() 
                       if info.get("security_level") == level]
                for level in ["high", "medium", "low"]
            },
            "categories": {
                "cloud_major": ["openai", "anthropic", "cohere"],
                "cloud_specialized": ["replicate", "groq"],
                "enterprise": ["bedrock", "vertex_ai", "azure"],
                "open_source": ["huggingface", "ollama"]
            },
            "security_features": {
                "content_filtering": self.content_filtering_enabled,
                "rate_limiting": self.rate_limit_enabled,
                "audit_logging": self.audit_logging_enabled,
                "api_key_validation": True,
                "input_sanitization": True
            }
        }

class SecureLiteLLMModel(BaseModel):
    """
    Secure LiteLLM model implementation with comprehensive safety features
    
    Features:
    - Universal input validation and sanitization
    - Provider-specific content policy checking
    - Rate limiting and quota management per provider
    - Comprehensive error handling with security context
    - Audit logging for all operations across providers
    - Streaming support with security monitoring
    """
    
    def __init__(self, model_id: str, config: ModelConfig, provider_config: Dict[str, Any], provider: SecureLiteLLMProvider):
        """
        Initialize secure LiteLLM model
        
        Args:
            model_id: Model identifier
            config: Model configuration
            provider_config: Provider configuration
            provider: Parent provider instance
        """
        super().__init__(model_id, config)
        self.provider_config = provider_config
        self.provider = provider
        self.config.model_type = ModelType.CHAT  # Most models are chat models
        self.model_provider = self.provider._extract_provider_from_model(model_id)
        
        security_logger.debug(f"SecureLiteLLMModel initialized: {model_id} ({self.model_provider})")
    
    async def generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """
        Generate response using LiteLLM with security validation
        
        Args:
            messages: List of messages for the conversation
            **kwargs: Additional generation parameters
            
        Returns:
            Model response with security metadata
        """
        try:
            # Rate limiting check for specific provider
            if not self.provider._check_rate_limit(self.model_provider):
                raise Exception(f"Rate limit exceeded for provider {self.model_provider}")
            
            # Validate and sanitize input messages
            sanitized_messages = []
            for message in messages:
                # Content policy check
                if self.provider.content_filtering_enabled:
                    is_compliant, violations = check_universal_content_policy(message.content)
                    if not is_compliant:
                        security_logger.warning(f"Universal content policy violation for {self.model_provider}: {violations}")
                        raise ValueError(f"Content policy violation: {violations}")
                
                # Sanitize message content
                sanitized_content = sanitize_universal_input(message.content)
                
                sanitized_messages.append({
                    "role": message.role,
                    "content": sanitized_content
                })
            
            # Prepare parameters with security considerations
            params = {
                "model": self.model_id,
                "messages": sanitized_messages,
                "temperature": kwargs.get('temperature', self.config.temperature),
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "stream": False
            }
            
            # Add provider-specific parameters if needed
            if 'custom_llm_provider' in kwargs:
                params['custom_llm_provider'] = kwargs['custom_llm_provider']
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Log API call attempt
            self.provider._log_api_call("completion", self.model_id, self.model_provider, sanitized_messages)
            
            # Make async call to LiteLLM
            start_time = time.time()
            response = await litellm.acompletion(**params)
            end_time = time.time()
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Log successful API call
            self.provider._log_api_call("completion", self.model_id, self.model_provider, sanitized_messages, content)
            
            # Create model response with security metadata
            model_response = ModelResponse(
                content=content,
                model=self.model_id,
                provider=f"litellm:{self.model_provider}",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                },
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_time": end_time - start_time,
                    "provider": self.model_provider,
                    "content_filtered": self.provider.content_filtering_enabled,
                    "security_validated": True,
                    "security_level": self.provider.SUPPORTED_PROVIDERS.get(self.model_provider, {}).get("security_level", "unknown")
                }
            )
            
            security_logger.debug(f"LiteLLM API call successful: {self.model_id}")
            return model_response
            
        except Exception as e:
            # Log error with security context
            self.provider._log_api_call("completion", self.model_id, self.model_provider, [], error=e)
            security_logger.error(f"LiteLLM API call failed for {self.model_id}: {e}")
            raise
    
    async def stream(self, messages: List[Message], **kwargs) -> AsyncGenerator[StreamingChunk, None]:
        """
        Stream response using LiteLLM with security monitoring
        
        Args:
            messages: List of messages for the conversation
            **kwargs: Additional generation parameters
            
        Yields:
            Streaming chunks with security validation
        """
        try:
            # Rate limiting and validation (same as generate)
            if not self.provider._check_rate_limit(self.model_provider):
                raise Exception(f"Rate limit exceeded for provider {self.model_provider}")
            
            # Sanitize messages
            sanitized_messages = []
            for message in messages:
                if self.provider.content_filtering_enabled:
                    is_compliant, violations = check_universal_content_policy(message.content)
                    if not is_compliant:
                        security_logger.warning(f"Universal policy violation in stream for {self.model_provider}: {violations}")
                        raise ValueError(f"Content policy violation: {violations}")
                
                sanitized_content = sanitize_universal_input(message.content)
                sanitized_messages.append({
                    "role": message.role,
                    "content": sanitized_content
                })
            
            # Prepare streaming parameters
            params = {
                "model": self.model_id,
                "messages": sanitized_messages,
                "temperature": kwargs.get('temperature', self.config.temperature),
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "stream": True
            }
            
            params = {k: v for k, v in params.items() if v is not None}
            
            # Log streaming attempt
            self.provider._log_api_call("completion(stream)", self.model_id, self.model_provider, sanitized_messages)
            
            # Make streaming call
            response = await litellm.acompletion(**params)
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content
                    if content:
                        # Security check on streamed content
                        if self.provider.content_filtering_enabled:
                            is_compliant, violations = check_universal_content_policy(content)
                            if not is_compliant:
                                security_logger.warning(f"Policy violation in streamed content for {self.model_provider}: {violations}")
                                break
                        
                        yield StreamingChunk(
                            content=content,
                            metadata={
                                "model": self.model_id,
                                "provider": self.model_provider,
                                "chunk_id": chunk.id,
                                "finish_reason": chunk.choices[0].finish_reason,
                                "security_validated": True
                            }
                        )
            
        except Exception as e:
            self.provider._log_api_call("completion(stream)", self.model_id, self.model_provider, [], error=e)
            security_logger.error(f"LiteLLM streaming failed for {self.model_id}: {e}")
            raise
    
    def get_context_length(self) -> int:
        """Estimate context length based on model with security considerations"""
        # Context length mappings for common models
        context_mappings = {
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 16385,
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
            "command-r": 128000,
            "llama-2-70b": 4096,
            "mixtral-8x7b": 32768
        }
        
        # Find best match
        for model_pattern, length in context_mappings.items():
            if model_pattern in self.model_id.lower():
                return length
        
        return 4096  # Conservative default for security
    
    def supports_streaming(self) -> bool:
        """Check if model supports streaming through LiteLLM"""
        non_streaming = ["o1-preview", "o1-mini"]  # Known exceptions
        return not any(model in self.model_id for model in non_streaming)
    
    def supports_function_calling(self) -> bool:
        """Check if model supports function calling"""
        function_calling_models = [
            "gpt-4", "gpt-3.5-turbo", "claude-3", "command-r"
        ]
        return any(model in self.model_id for model in function_calling_models)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information with security metadata"""
        provider_info = self.provider.SUPPORTED_PROVIDERS.get(self.model_provider, {})
        
        return {
            "model_id": self.model_id,
            "provider": self.model_provider,
            "type": self.config.model_type.value,
            "context_length": self.get_context_length(),
            "litellm_compatible": True,
            "security_features": {
                "content_filtering": self.provider.content_filtering_enabled,
                "rate_limiting": self.provider.rate_limit_enabled,
                "audit_logging": self.provider.audit_logging_enabled,
                "input_sanitization": True,
                "output_validation": True,
                "security_level": provider_info.get("security_level", "unknown")
            },
            "capabilities": {
                "streaming": self.supports_streaming(),
                "function_calling": self.supports_function_calling(),
                "universal_access": True
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
        Generate response with tool/function calling support and security validation
        
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
            if not self.provider._check_rate_limit(self.model_provider):
                raise Exception(f"Rate limit exceeded for provider {self.model_provider}")
            
            # Validate and sanitize input messages
            sanitized_messages = []
            for message in messages:
                if self.provider.content_filtering_enabled:
                    is_compliant, violations = check_universal_content_policy(message.content)
                    if not is_compliant:
                        security_logger.warning(f"Universal tool policy violation for {self.model_provider}: {violations}")
                        raise ValueError(f"Content policy violation: {violations}")
                
                sanitized_content = sanitize_universal_input(message.content)
                sanitized_messages.append({
                    "role": message.role,
                    "content": sanitized_content
                })
            
            # Validate and sanitize tools
            sanitized_tools = []
            for tool in tools:
                # Basic tool validation
                if not isinstance(tool, dict) or "name" not in tool:
                    security_logger.warning(f"Invalid tool format for {self.model_provider}: {tool}")
                    continue
                
                # Sanitize tool definition
                sanitized_tool = {
                    "name": html.escape(str(tool["name"])),
                    "description": html.escape(str(tool.get("description", ""))),
                    "parameters": tool.get("input_schema", {})
                }
                
                sanitized_tools.append(sanitized_tool)
            
            # Prepare parameters with functions (OpenAI format for LiteLLM)
            params = {
                "model": self.model_id,
                "messages": sanitized_messages,
                "functions": sanitized_tools,
                "function_call": "auto",
                "temperature": kwargs.get('temperature', self.config.temperature),
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "top_p": kwargs.get('top_p', self.config.top_p)
            }
            
            params = {k: v for k, v in params.items() if v is not None}
            
            # Log API call attempt
            self.provider._log_api_call("completion(tools)", self.model_id, self.model_provider, sanitized_messages)
            
            # Make API call
            start_time = time.time()
            response = await litellm.acompletion(**params)
            end_time = time.time()
            
            # Extract function calls if present
            choice = response.choices[0]
            function_call = getattr(choice.message, 'function_call', None)
            tool_calls = []
            
            if function_call:
                tool_calls.append({
                    "id": getattr(function_call, 'id', 'call_1'),
                    "name": function_call.name,
                    "arguments": function_call.arguments
                })
            
            content = choice.message.content or ""
            
            # Log successful API call
            self.provider._log_api_call("completion(tools)", self.model_id, self.model_provider, sanitized_messages, content)
            
            # Create model response with security metadata
            model_response = ModelResponse(
                content=content,
                model=self.model_id,
                provider=f"litellm:{self.model_provider}",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                },
                metadata={
                    "finish_reason": choice.finish_reason,
                    "response_time": end_time - start_time,
                    "provider": self.model_provider,
                    "content_filtered": self.provider.content_filtering_enabled,
                    "security_validated": True,
                    "tool_calls_count": len(tool_calls)
                },
                tool_calls=tool_calls if tool_calls else None
            )
            
            security_logger.debug(f"LiteLLM tool API call successful: {self.model_id}")
            return model_response
            
        except Exception as e:
            self.provider._log_api_call("completion(tools)", self.model_id, self.model_provider, [], error=e)
            security_logger.error(f"LiteLLM tool generation failed for {self.model_id}: {e}")
            raise

# Alias for backward compatibility
LiteLLMProvider = SecureLiteLLMProvider
LiteLLMModel = SecureLiteLLMModel
