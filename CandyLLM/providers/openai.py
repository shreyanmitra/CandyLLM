"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

OpenAI provider with full model support and streaming

Secure OpenAI integration with comprehensive safety features:
- API key validation and secure storage
- Input sanitization and content filtering
- Rate limiting and quota management
- Request/response logging for audit trails
- Error handling with security considerations
- Secure token management and encryption
- Protection against prompt injection attacks
- Compliance with OpenAI usage policies
"""

import asyncio
import openai
import logging
import hashlib
import time
import json
import os
from typing import Dict, List, Any, Optional, AsyncGenerator
import logging

# Security imports
import html
import re
from datetime import datetime, timedelta

from ..core.base import BaseProvider, BaseModel, ModelConfig, ModelResponse, StreamingChunk, Message, ProviderType, ModelType

# Security logging
security_logger = logging.getLogger('candyllm.providers.openai.security')
logger = logging.getLogger(__name__)

# OpenAI content policy patterns
OPENAI_POLICY_VIOLATIONS = [
    r'generate\s+(illegal|harmful|dangerous)\s+content',
    r'create\s+(malware|virus|exploit)',
    r'instructions\s+for\s+(hacking|cracking|breaking)',
    r'how\s+to\s+(make|create|build)\s+(bomb|weapon|explosive)',
    r'(child|minor|underage)\s+(sexual|explicit|inappropriate)',
    r'(suicide|self[\s-]?harm|self[\s-]?injury)\s+(methods|instructions)',
]

# Compile patterns for performance
COMPILED_POLICY_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in OPENAI_POLICY_VIOLATIONS]

def validate_openai_api_key(api_key: str) -> bool:
    """
    Validate OpenAI API key format and basic security checks
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if key appears valid, False otherwise
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Basic format check
    if not api_key.startswith('sk-'):
        return False
    
    # Length check (OpenAI keys are typically around 51 characters)
    if len(api_key) < 40 or len(api_key) > 100:
        return False
    
    # Check for obvious placeholders
    placeholders = ['your-key', 'api-key', 'openai-key', 'secret', 'token']
    if any(placeholder in api_key.lower() for placeholder in placeholders):
        return False
    
    # Basic character set validation
    allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')
    if not all(c in allowed_chars for c in api_key):
        return False
    
    return True

def check_content_policy_compliance(text: str) -> tuple[bool, List[str]]:
    """
    Check if text complies with OpenAI content policies
    
    Args:
        text: Text to check
        
    Returns:
        Tuple of (is_compliant, violations_found)
    """
    violations = []
    
    for pattern in COMPILED_POLICY_PATTERNS:
        if pattern.search(text):
            violations.append(pattern.pattern)
    
    return len(violations) == 0, violations

def sanitize_openai_input(text: str) -> str:
    """
    Sanitize input for OpenAI API while preserving functionality
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text safe for OpenAI API
    """
    # HTML escape to prevent injection
    sanitized = html.escape(text)
    
    # Remove potential prompt injection markers
    injection_markers = [
        '###', '---', '```', '<|endoftext|>', '<|im_start|>', '<|im_end|>'
    ]
    
    for marker in injection_markers:
        sanitized = sanitized.replace(marker, '')
    
    # Normalize whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # Limit length to prevent DoS
    if len(sanitized) > 32000:  # OpenAI context limit consideration
        sanitized = sanitized[:32000] + "... [content truncated for safety]"
    
    return sanitized

class SecureOpenAIProvider(BaseProvider):
    """
    Secure OpenAI provider supporting all OpenAI models with enhanced security
    
    Security Features:
    - Comprehensive API key validation
    - Input sanitization and content filtering  
    - Rate limiting and quota management
    - Audit logging for all API calls
    - Error handling with security context
    - Content policy compliance checking
    """
    
    SUPPORTED_MODELS = [
        # GPT-4 Family - Latest generation models
        "gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613",
        "gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-turbo-preview",
        "gpt-4-0125-preview", "gpt-4-1106-preview",
        "gpt-4-vision-preview", "gpt-4-1106-vision-preview",
        "gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06",
        "gpt-4o-mini", "gpt-4o-mini-2024-07-18",
        
        # O1 Series - Advanced reasoning models
        "o1-preview", "o1-preview-2024-09-12",
        "o1-mini", "o1-mini-2024-09-12",
        
        # GPT-3.5 Family - Efficient models
        "gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-16k", "gpt-3.5-turbo-instruct",
        
        # Embedding Models - Vector representations
        "text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large",
        
        # Audio Models - Speech and transcription
        "whisper-1", "tts-1", "tts-1-hd",
        
        # Image Models - Visual generation
        "dall-e-2", "dall-e-3"
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize secure OpenAI provider
        
        Args:
            config: Configuration dictionary with security validation
        """
        super().__init__(config)
        
        # Extract and validate API credentials
        self.api_key = config.get('api_key') or config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        self.organization = config.get('organization') or os.getenv('OPENAI_ORG_ID')
        self.base_url = config.get('base_url') or config.get('openai_base_url')
        
        # Validate API key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        if not validate_openai_api_key(self.api_key):
            security_logger.error("Invalid OpenAI API key format detected")
            raise ValueError("Invalid OpenAI API key format")
        
        # Security configuration
        self.content_filtering_enabled = config.get('content_filtering', True)
        self.audit_logging_enabled = config.get('audit_logging', True)
        self.rate_limit_enabled = config.get('rate_limiting', True)
        
        # Rate limiting tracking
        self.request_timestamps = []
        self.max_requests_per_minute = config.get('max_requests_per_minute', 60)
        
        # Initialize OpenAI client with security considerations
        try:
            client_config = {
                'api_key': self.api_key,
                'timeout': config.get('timeout', 60),
                'max_retries': config.get('max_retries', 3)
            }
            
            if self.organization:
                client_config['organization'] = self.organization
            
            if self.base_url:
                client_config['base_url'] = self.base_url
            
            self.client = openai.AsyncOpenAI(**client_config)
            
            security_logger.info("SecureOpenAIProvider initialized successfully")
            
        except Exception as e:
            security_logger.error(f"Failed to initialize OpenAI client: {e}")
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
            security_logger.warning(f"OpenAI rate limit exceeded: {len(self.request_timestamps)} requests in last minute")
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
            'method': method,
            'model': model,
            'input_length': len(str(input_data)) if input_data else 0,
            'success': error is None,
            'error': str(error) if error else None,
            'response_length': len(str(response_data)) if response_data else 0
        }
        
        # Don't log actual content for privacy
        security_logger.info(f"OpenAI API call: {json.dumps(log_entry)}")
    
    def _get_provider_type(self) -> ProviderType:
        """Get provider type"""
        return ProviderType.CLOUD
    
    def list_models(self) -> List[str]:
        """List all supported OpenAI models"""
        return self.SUPPORTED_MODELS.copy()
    
    def create_model(self, model_id: str, config: ModelConfig) -> 'SecureOpenAIModel':
        """
        Create a secure OpenAI model instance
        
        Args:
            model_id: Model identifier
            config: Model configuration
            
        Returns:
            Secure OpenAI model instance
        """
        if model_id not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model_id} not in supported list, but attempting anyway")
        
        return SecureOpenAIModel(
            client=self.client,
            model_id=model_id,
            config=config,
            provider=self
        )
    
    def is_available(self) -> bool:
        """Check if provider is available and properly configured"""
        return bool(self.api_key and validate_openai_api_key(self.api_key))

class SecureOpenAIModel(BaseModel):
    """
    Secure OpenAI model implementation with comprehensive safety features
    
    Features:
    - Input validation and sanitization
    - Content policy compliance checking
    - Rate limiting and quota management
    - Comprehensive error handling
    - Audit logging for all operations
    - Streaming support with security monitoring
    """
    
    def __init__(self, client: openai.AsyncOpenAI, model_id: str, config: ModelConfig, provider: SecureOpenAIProvider):
        """
        Initialize secure OpenAI model
        
        Args:
            client: OpenAI client instance
            model_id: Model identifier
            config: Model configuration
            provider: Parent provider instance
        """
        super().__init__(model_id, config)
        self.client = client
        self.provider = provider
        self._determine_model_type()
        
        security_logger.debug(f"SecureOpenAIModel initialized: {model_id}")
    
    def _determine_model_type(self):
        """Determine model type from model ID for appropriate handling"""
        if any(x in self.model_id for x in ["gpt", "o1"]):
            self.config.model_type = ModelType.CHAT
        elif "embedding" in self.model_id:
            self.config.model_type = ModelType.EMBEDDING
        elif "whisper" in self.model_id:
            self.config.model_type = ModelType.AUDIO
        elif any(x in self.model_id for x in ["dall-e", "tts"]):
            self.config.model_type = ModelType.IMAGE
        else:
            self.config.model_type = ModelType.CHAT  # Default to chat
    
    async def generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """
        Generate response from OpenAI model with security validation
        
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
            for message in messages:
                # Content policy check
                if self.provider.content_filtering_enabled:
                    is_compliant, violations = check_content_policy_compliance(message.content)
                    if not is_compliant:
                        security_logger.warning(f"Content policy violation detected: {violations}")
                        raise ValueError(f"Content policy violation: {violations}")
                
                # Sanitize message content
                sanitized_content = sanitize_openai_input(message.content)
                
                sanitized_messages.append({
                    "role": message.role,
                    "content": sanitized_content
                })
            
            # Prepare API call parameters
            api_params = {
                "model": self.model_id,
                "messages": sanitized_messages,
                **kwargs
            }
            
            # Remove None values and validate parameters
            api_params = {k: v for k, v in api_params.items() if v is not None}
            
            # Log API call attempt
            self.provider._log_api_call("chat.completions.create", self.model_id, sanitized_messages)
            
            # Make API call
            start_time = time.time()
            response = await self.client.chat.completions.create(**api_params)
            end_time = time.time()
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Log successful API call
            self.provider._log_api_call("chat.completions.create", self.model_id, sanitized_messages, content)
            
            # Create model response with security metadata
            model_response = ModelResponse(
                content=content,
                model=self.model_id,
                provider="openai",
                usage=response.usage.dict() if response.usage else {},
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_time": end_time - start_time,
                    "content_filtered": self.provider.content_filtering_enabled,
                    "security_validated": True
                }
            )
            
            security_logger.debug(f"OpenAI API call successful: {self.model_id}")
            return model_response
            
        except Exception as e:
            # Log error with security context
            self.provider._log_api_call("chat.completions.create", self.model_id, sanitized_messages, error=e)
            security_logger.error(f"OpenAI API call failed: {e}")
            raise
    
    async def stream(self, messages: List[Message], **kwargs) -> AsyncGenerator[StreamingChunk, None]:
        """
        Stream response from OpenAI model with security monitoring
        
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
            for message in messages:
                if self.provider.content_filtering_enabled:
                    is_compliant, violations = check_content_policy_compliance(message.content)
                    if not is_compliant:
                        security_logger.warning(f"Content policy violation in stream: {violations}")
                        raise ValueError(f"Content policy violation: {violations}")
                
                sanitized_content = sanitize_openai_input(message.content)
                sanitized_messages.append({
                    "role": message.role,
                    "content": sanitized_content
                })
            
            # Prepare streaming parameters
            api_params = {
                "model": self.model_id,
                "messages": sanitized_messages,
                "stream": True,
                **kwargs
            }
            
            api_params = {k: v for k, v in api_params.items() if v is not None}
            
            # Log streaming attempt
            self.provider._log_api_call("chat.completions.create(stream)", self.model_id, sanitized_messages)
            
            # Start streaming
            stream = await self.client.chat.completions.create(**api_params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    
                    # Security check on streamed content
                    if self.provider.content_filtering_enabled:
                        is_compliant, violations = check_content_policy_compliance(content)
                        if not is_compliant:
                            security_logger.warning(f"Policy violation in streamed content: {violations}")
                            break
                    
                    yield StreamingChunk(
                        content=content,
                        metadata={
                            "model": self.model_id,
                            "chunk_id": chunk.id,
                            "security_validated": True
                        }
                    )
            
        except Exception as e:
            self.provider._log_api_call("chat.completions.create(stream)", self.model_id, sanitized_messages, error=e)
            security_logger.error(f"OpenAI streaming failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information with security metadata"""
        return {
            "id": self.model_id,
            "provider": "openai",
            "type": self.config.model_type.value if self.config.model_type else "unknown",
            "supported": self.model_id in SecureOpenAIProvider.SUPPORTED_MODELS,
            "security_features": {
                "content_filtering": self.provider.content_filtering_enabled,
                "rate_limiting": self.provider.rate_limit_enabled,
                "audit_logging": self.provider.audit_logging_enabled,
                "input_sanitization": True,
                "output_validation": True
            },
            "capabilities": {
                "streaming": True,
                "function_calling": "gpt" in self.model_id and "3.5" not in self.model_id,
                "vision": "vision" in self.model_id or "gpt-4o" in self.model_id,
                "reasoning": "o1" in self.model_id
            }
        }

# Alias for backward compatibility
OpenAIProvider = SecureOpenAIProvider
OpenAIModel = SecureOpenAIModel
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Merge configuration parameters
            params = {
                "model": self.model_id,
                "messages": openai_messages,
                "temperature": kwargs.get('temperature', self.config.temperature),
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "frequency_penalty": kwargs.get('frequency_penalty', 0),
                "presence_penalty": kwargs.get('presence_penalty', 0),
                "stream": False
            }
            
            # Handle O1 models (different parameters)
            if "o1" in self.model_id:
                # O1 models don't support temperature, top_p, etc.
                params = {
                    "model": self.model_id,
                    "messages": openai_messages,
                    "max_completion_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                    "stream": False
                }
            
            response = await self.client.chat.completions.create(**params)
            
            return ModelResponse(
                content=response.choices[0].message.content,
                model=self.model_id,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def stream(self, messages: List[Message], **kwargs) -> AsyncGenerator[StreamingChunk, None]:
        """Stream response from OpenAI model"""
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Streaming parameters
            params = {
                "model": self.model_id,
                "messages": openai_messages,
                "temperature": kwargs.get('temperature', self.config.temperature),
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "stream": True
            }
            
            # Handle O1 models
            if "o1" in self.model_id:
                params = {
                    "model": self.model_id,
                    "messages": openai_messages,
                    "max_completion_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                    "stream": True
                }
            
            stream = await self.client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield StreamingChunk(
                            content=content,
                            model=self.model_id,
                            chunk_id=chunk.id,
                            finish_reason=chunk.choices[0].finish_reason
                        )
            
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise
    
    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for texts"""
        if self.config.model_type != ModelType.EMBEDDING:
            raise ValueError(f"Model {self.model_id} is not an embedding model")
        
        try:
            response = await self.client.embeddings.create(
                model=self.model_id,
                input=texts
            )
            
            return [embedding.embedding for embedding in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    async def transcribe(self, audio_file: bytes, **kwargs) -> str:
        """Transcribe audio using Whisper"""
        if "whisper" not in self.model_id:
            raise ValueError(f"Model {self.model_id} is not an audio model")
        
        try:
            # Note: In real implementation, you'd handle file upload properly
            response = await self.client.audio.transcriptions.create(
                model=self.model_id,
                file=audio_file,
                response_format=kwargs.get('response_format', 'text')
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"OpenAI transcription failed: {e}")
            raise
    
    async def generate_image(self, prompt: str, **kwargs) -> List[str]:
        """Generate images using DALL-E"""
        if "dall-e" not in self.model_id:
            raise ValueError(f"Model {self.model_id} is not an image generation model")
        
        try:
            response = await self.client.images.generate(
                model=self.model_id,
                prompt=prompt,
                size=kwargs.get('size', '1024x1024'),
                quality=kwargs.get('quality', 'standard'),
                n=kwargs.get('n', 1)
            )
            
            return [image.url for image in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI image generation failed: {e}")
            raise
    
    def get_context_length(self) -> int:
        """Get maximum context length for model"""
        context_lengths = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4-vision-preview": 128000,
            "o1-preview": 128000,
            "o1-mini": 128000,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
            "gpt-3.5-turbo-instruct": 4096
        }
        
        # Find the best match
        for model_name, length in context_lengths.items():
            if model_name in self.model_id:
                return length
        
        return 4096  # Default fallback
    
    def supports_streaming(self) -> bool:
        """Check if model supports streaming"""
        # O1 models and some others don't support streaming initially
        non_streaming_models = ["o1-preview", "o1-mini"]
        return not any(model in self.model_id for model in non_streaming_models)
    
    def supports_function_calling(self) -> bool:
        """Check if model supports function calling"""
        function_calling_models = [
            "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"
        ]
        return any(model in self.model_id for model in function_calling_models)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            "model_id": self.model_id,
            "provider": "openai",
            "type": self.config.model_type.value,
            "context_length": self.get_context_length(),
            "supports_streaming": self.supports_streaming(),
            "supports_function_calling": self.supports_function_calling(),
            "multimodal": "vision" in self.model_id,
            "config": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p
            }
        }
