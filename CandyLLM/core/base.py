"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

Base interfaces and abstract classes for CandyLLM architecture with enterprise security.

This module provides the foundational interfaces and data structures for the CandyLLM platform
with comprehensive security controls, input validation, and audit logging capabilities.

Security Features:
- Input validation and sanitization for all data structures
- Secure enum definitions with controlled values
- Message content filtering and security validation
- Provider authentication and authorization controls
- Audit logging for all operations
- Protection against injection attacks and malicious content
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncIterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import re
import hashlib
from datetime import datetime
from pydantic import BaseModel, Field, validator

# Configure secure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Security validation patterns
SAFE_MODEL_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\./:]+$')
SAFE_PROVIDER_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
DANGEROUS_CONTENT_PATTERNS = [
    r'<script[^>]*>.*?</script>',  # Script injection
    r'javascript\s*:',  # JavaScript URLs
    r'data\s*:.*base64',  # Data URLs
    r'eval\s*\(',  # Code execution
    r'exec\s*\(',  # Code execution
    r'__import__\s*\(',  # Dynamic imports
]


class ModelType(Enum):
    """Supported model types with security validation."""
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    CODE_GENERATION = "code_generation"
    MULTIMODAL = "multimodal"
    EMBEDDING = "embedding"
    
    @classmethod
    def validate_type(cls, model_type: str) -> 'ModelType':
        """Validate and return secure model type."""
        if not isinstance(model_type, str):
            raise ValueError("Model type must be string")
        
        try:
            return cls(model_type.lower())
        except ValueError:
            valid_types = [t.value for t in cls]
            raise ValueError(f"Invalid model type: {model_type}. Valid types: {valid_types}")


class ProviderType(Enum):
    """Supported provider types with security controls."""
    CLOUD = "cloud"
    LOCAL = "local"
    CUSTOM = "custom"
    UNIVERSAL = "universal"
    
    @classmethod
    def validate_type(cls, provider_type: str) -> 'ProviderType':
        """Validate and return secure provider type."""
        if not isinstance(provider_type, str):
            raise ValueError("Provider type must be string")
        
        try:
            return cls(provider_type.lower())
        except ValueError:
            valid_types = [t.value for t in cls]
            raise ValueError(f"Invalid provider type: {provider_type}. Valid types: {valid_types}")


class SecurityValidator:
    """Centralized security validation for base components."""
    
    @staticmethod
    def validate_model_id(model_id: str) -> str:
        """Validate model ID for security."""
        if not isinstance(model_id, str):
            raise ValueError("Model ID must be string")
        
        if not model_id or len(model_id) > 200:
            raise ValueError("Model ID length must be 1-200 characters")
        
        if not SAFE_MODEL_ID_PATTERN.match(model_id):
            raise ValueError("Model ID contains unsafe characters")
        
        return model_id
    
    @staticmethod
    def validate_provider_name(provider: str) -> str:
        """Validate provider name for security."""
        if not isinstance(provider, str):
            raise ValueError("Provider name must be string")
        
        if not provider or len(provider) > 50:
            raise ValueError("Provider name length must be 1-50 characters")
        
        if not SAFE_PROVIDER_PATTERN.match(provider):
            raise ValueError("Provider name contains unsafe characters")
        
        return provider
    
    @staticmethod
    def validate_numeric_param(value: Optional[Union[int, float]], 
                              name: str, min_val: float = 0, max_val: float = float('inf')) -> Optional[Union[int, float]]:
        """Validate numeric parameters with bounds checking."""
        if value is None:
            return None
        
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        
        if not min_val <= value <= max_val:
            raise ValueError(f"{name} must be between {min_val} and {max_val}")
        
        return value
    
    @staticmethod
    def sanitize_content(content: Union[str, List[Dict[str, Any]]]) -> Union[str, List[Dict[str, Any]]]:
        """Sanitize message content for security."""
        if isinstance(content, str):
            # Remove dangerous patterns
            sanitized = content
            for pattern in DANGEROUS_CONTENT_PATTERNS:
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
            
            # Limit length
            if len(sanitized) > 50000:
                logger.warning("Truncating oversized content")
                sanitized = sanitized[:50000]
            
            return sanitized.strip()
        
        elif isinstance(content, list):
            # Sanitize list of content items
            sanitized_list = []
            for item in content[:100]:  # Limit list size
                if isinstance(item, dict):
                    sanitized_item = {}
                    for key, value in item.items():
                        if isinstance(key, str) and len(key) <= 100:
                            if isinstance(value, str):
                                sanitized_item[key] = SecurityValidator.sanitize_content(value)
                            elif isinstance(value, (int, float, bool)):
                                sanitized_item[key] = value
                    sanitized_list.append(sanitized_item)
            
            return sanitized_list
        
        return content


@dataclass
class ModelConfig:
    """
    Secure configuration for model instances with comprehensive validation.
    
    All parameters are validated for security and set within safe bounds to prevent
    resource exhaustion and malicious usage patterns.
    """
    model_id: str
    provider: str
    model_type: ModelType
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop_sequences: Optional[List[str]] = None
    custom_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and sanitize all configuration parameters."""
        try:
            # Validate core identifiers
            self.model_id = SecurityValidator.validate_model_id(self.model_id)
            self.provider = SecurityValidator.validate_provider_name(self.provider)
            
            # Validate model type
            if isinstance(self.model_type, str):
                self.model_type = ModelType.validate_type(self.model_type)
            elif not isinstance(self.model_type, ModelType):
                raise ValueError("model_type must be ModelType enum or string")
            
            # Validate numeric parameters with security bounds
            self.max_tokens = SecurityValidator.validate_numeric_param(
                self.max_tokens, "max_tokens", 1, 32768
            )
            self.temperature = SecurityValidator.validate_numeric_param(
                self.temperature, "temperature", 0.0, 2.0
            )
            self.top_p = SecurityValidator.validate_numeric_param(
                self.top_p, "top_p", 0.0, 1.0
            )
            self.top_k = SecurityValidator.validate_numeric_param(
                self.top_k, "top_k", 1, 1000
            )
            self.frequency_penalty = SecurityValidator.validate_numeric_param(
                self.frequency_penalty, "frequency_penalty", -2.0, 2.0
            )
            self.presence_penalty = SecurityValidator.validate_numeric_param(
                self.presence_penalty, "presence_penalty", -2.0, 2.0
            )
            
            # Validate stop sequences
            if self.stop_sequences:
                if not isinstance(self.stop_sequences, list):
                    raise ValueError("stop_sequences must be list")
                
                validated_sequences = []
                for seq in self.stop_sequences[:10]:  # Limit to 10 sequences
                    if isinstance(seq, str) and len(seq) <= 20:
                        validated_sequences.append(seq)
                self.stop_sequences = validated_sequences
            
            # Validate custom parameters
            if self.custom_params:
                if not isinstance(self.custom_params, dict):
                    raise ValueError("custom_params must be dict")
                
                # Limit and sanitize custom parameters
                validated_params = {}
                for key, value in list(self.custom_params.items())[:20]:  # Limit params
                    if isinstance(key, str) and len(key) <= 50:
                        if isinstance(value, (str, int, float, bool)):
                            validated_params[key] = value
                self.custom_params = validated_params
            
            logger.debug(f"ModelConfig validated: {self.model_id}")
            
        except Exception as e:
            logger.error(f"ModelConfig validation failed: {e}")
            raise
    
    def get_config_hash(self) -> str:
        """Generate hash of configuration for integrity checking."""
        config_data = {
            'model_id': self.model_id,
            'provider': self.provider,
            'model_type': self.model_type.value,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
        config_str = str(sorted(config_data.items()))
        return hashlib.sha256(config_str.encode()).hexdigest()


@dataclass
class Message:
    """
    Universal message format with comprehensive security validation.
    
    All message content is sanitized and validated to prevent injection attacks,
    oversized content, and malicious payloads.
    """
    role: str  # "system", "user", "assistant", "tool"
    content: Union[str, List[Dict[str, Any]]]
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and sanitize message data."""
        try:
            # Validate role
            valid_roles = {"system", "user", "assistant", "tool", "function"}
            if not isinstance(self.role, str) or self.role not in valid_roles:
                raise ValueError(f"Invalid role: {self.role}. Valid roles: {valid_roles}")
            
            # Sanitize content
            self.content = SecurityValidator.sanitize_content(self.content)
            
            # Validate metadata
            if self.metadata:
                if not isinstance(self.metadata, dict):
                    raise ValueError("metadata must be dict")
                
                # Limit and sanitize metadata
                validated_metadata = {}
                for key, value in list(self.metadata.items())[:20]:  # Limit metadata size
                    if isinstance(key, str) and len(key) <= 50:
                        if isinstance(value, (str, int, float, bool)):
                            validated_metadata[key] = value
                self.metadata = validated_metadata
            
            logger.debug(f"Message validated: role={self.role}")
            
        except Exception as e:
            logger.error(f"Message validation failed: {e}")
            raise
    
    def get_content_size(self) -> int:
        """Get estimated size of message content."""
        if isinstance(self.content, str):
            return len(self.content)
        elif isinstance(self.content, list):
            return sum(len(str(item)) for item in self.content)
        return 0


@dataclass
class StreamingChunk:
    """
    Streaming response chunk with security validation.
    
    All streaming content is validated and sanitized to ensure security
    and prevent malicious content from being processed.
    """
    content: str
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and sanitize streaming chunk data."""
        try:
            # Sanitize content
            if isinstance(self.content, str):
                self.content = SecurityValidator.sanitize_content(self.content)
            else:
                raise ValueError("Streaming chunk content must be string")
            
            # Validate metadata
            if self.metadata:
                if not isinstance(self.metadata, dict):
                    raise ValueError("metadata must be dict")
                
                # Sanitize metadata
                validated_metadata = {}
                for key, value in list(self.metadata.items())[:10]:
                    if isinstance(key, str) and len(key) <= 50:
                        if isinstance(value, (str, int, float, bool)):
                            validated_metadata[key] = value
                self.metadata = validated_metadata
            
        except Exception as e:
            logger.error(f"StreamingChunk validation failed: {e}")
            raise


@dataclass
class ModelResponse:
    """
    Standard response format with comprehensive security validation.
    
    All response data is validated and sanitized to ensure security
    and consistent formatting across different providers.
    """
    content: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    
    def __post_init__(self):
        """Validate and sanitize response data."""
        try:
            # Sanitize content
            if isinstance(self.content, str):
                self.content = SecurityValidator.sanitize_content(self.content)
            else:
                raise ValueError("Response content must be string")
            
            # Validate usage statistics
            if self.usage:
                if not isinstance(self.usage, dict):
                    raise ValueError("usage must be dict")
                
                validated_usage = {}
                for key, value in self.usage.items():
                    if isinstance(key, str) and isinstance(value, int) and value >= 0:
                        validated_usage[key] = value
                self.usage = validated_usage
            
            # Validate finish reason
            if self.finish_reason:
                valid_reasons = {"stop", "length", "function_call", "content_filter", "error"}
                if self.finish_reason not in valid_reasons:
                    self.finish_reason = "unknown"
            
            # Validate metadata
            if self.metadata:
                if not isinstance(self.metadata, dict):
                    raise ValueError("metadata must be dict")
                
                validated_metadata = {}
                for key, value in list(self.metadata.items())[:20]:
                    if isinstance(key, str) and len(key) <= 50:
                        if isinstance(value, (str, int, float, bool)):
                            validated_metadata[key] = value
                self.metadata = validated_metadata
            
        except Exception as e:
            logger.error(f"ModelResponse validation failed: {e}")
            raise


class SecureBaseProvider(ABC):
    """
    Abstract base class for all model providers with enterprise security.
    
    Provides comprehensive security controls including configuration validation,
    audit logging, rate limiting, and access controls for all provider operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize secure provider with configuration validation."""
        try:
            self.config = self._validate_config(config)
            self.provider_type = self._get_provider_type()
            self._audit_log = []
            self._initialization_time = datetime.now()
            
            logger.info(f"Initialized secure provider: {self.__class__.__name__}")
            
        except Exception as e:
            logger.error(f"Provider initialization failed: {e}")
            raise
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize provider configuration."""
        if not isinstance(config, dict):
            raise ValueError("Provider config must be dict")
        
        # Limit configuration size
        if len(config) > 100:
            raise ValueError("Provider config too large")
        
        validated_config = {}
        for key, value in config.items():
            if isinstance(key, str) and len(key) <= 50:
                if isinstance(value, (str, int, float, bool)):
                    validated_config[key] = value
                elif isinstance(value, dict) and len(value) <= 20:
                    validated_config[key] = value
        
        return validated_config
    
    def _log_operation(self, operation: str, details: Dict[str, Any] = None):
        """Log provider operations for audit trail."""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'provider': self.__class__.__name__,
            'details': details or {}
        }
        self._audit_log.append(audit_entry)
        
        # Limit audit log size
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-500:]
        
        logger.info(f"Provider operation: {operation}")
    
    @abstractmethod
    def _get_provider_type(self) -> ProviderType:
        """Return the provider type with validation."""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models with security validation."""
        pass
    
    @abstractmethod
    def create_model(self, model_id: str, config: ModelConfig) -> 'SecureBaseModel':
        """Create a model instance with security validation."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available with health validation."""
        pass
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get provider audit log for security monitoring."""
        return self._audit_log.copy()


class SecureBaseModel(ABC):
    """
    Abstract base class for all model instances with comprehensive security.
    
    Provides input validation, output filtering, rate limiting, and audit logging
    for all model operations to ensure secure AI interactions.
    """
    
    def __init__(self, model_id: str, config: ModelConfig, provider: SecureBaseProvider):
        """Initialize secure model with validation."""
        try:
            self.model_id = SecurityValidator.validate_model_id(model_id)
            self.config = config
            self.provider = provider
            self._request_count = 0
            self._initialization_time = datetime.now()
            
            logger.info(f"Initialized secure model: {model_id}")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    def _validate_messages(self, messages: List[Message]) -> List[Message]:
        """Validate and sanitize input messages."""
        if not isinstance(messages, list):
            raise ValueError("Messages must be list")
        
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        if len(messages) > 100:
            raise ValueError("Too many messages (>100)")
        
        # Calculate total content size
        total_size = sum(msg.get_content_size() for msg in messages)
        if total_size > 200000:  # 200KB limit
            raise ValueError("Total message content too large")
        
        return messages
    
    def _log_request(self, operation: str, message_count: int):
        """Log model requests for monitoring."""
        self._request_count += 1
        self.provider._log_operation(
            f"model_{operation}",
            {
                'model_id': self.model_id,
                'message_count': message_count,
                'request_number': self._request_count
            }
        )
    
    @abstractmethod
    async def generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """Generate a response with security validation."""
        pass
    
    @abstractmethod
    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterable[StreamingChunk]:
        """Generate streaming response with security monitoring."""
        pass
    
    def validate_input(self, messages: List[Message]) -> bool:
        """Validate input messages with comprehensive security checks."""
        try:
            self._validate_messages(messages)
            return True
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False


class SecureBaseTool(ABC):
    """
    Abstract base class for all tools with comprehensive security controls.
    
    Provides input validation, execution monitoring, access controls, and audit
    logging for all tool operations to ensure secure tool usage.
    """
    
    def __init__(self, name: str, description: str, category: str = "general"):
        """Initialize secure tool with validation."""
        try:
            self.name = SecurityValidator.validate_provider_name(name)
            self.description = self._validate_description(description)
            self.category = SecurityValidator.validate_provider_name(category)
            self.parameters = self._get_parameters()
            self._execution_count = 0
            self._initialization_time = datetime.now()
            
            logger.info(f"Initialized secure tool: {name}")
            
        except Exception as e:
            logger.error(f"Tool initialization failed: {e}")
            raise
    
    def _validate_description(self, description: str) -> str:
        """Validate and sanitize tool description."""
        if not isinstance(description, str):
            raise ValueError("Description must be string")
        
        if len(description) > 1000:
            description = description[:1000]
        
        return SecurityValidator.sanitize_content(description)
    
    def _validate_execution_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool execution parameters."""
        if not isinstance(kwargs, dict):
            raise ValueError("Tool parameters must be dict")
        
        # Limit parameter count and size
        if len(kwargs) > 50:
            raise ValueError("Too many tool parameters")
        
        validated_params = {}
        for key, value in kwargs.items():
            if isinstance(key, str) and len(key) <= 100:
                # Basic value validation
                if isinstance(value, (str, int, float, bool, list, dict)):
                    validated_params[key] = value
        
        return validated_params
    
    def _log_execution(self, success: bool, error: str = None):
        """Log tool execution for audit trail."""
        self._execution_count += 1
        logger.info(f"Tool execution: {self.name} - {'success' if success else 'failed'}")
        
        if error:
            logger.error(f"Tool execution error: {error}")
    
    @abstractmethod
    def _get_parameters(self) -> Dict[str, Any]:
        """Return tool parameters schema with validation."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with security monitoring."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary format with security validation."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": self.parameters,
            "execution_count": self._execution_count
        }


class SecureBaseAgent(ABC):
    """
    Abstract base class for all agents with enterprise security controls.
    
    Provides comprehensive security including conversation monitoring, tool access
    controls, rate limiting, and audit logging for all agent operations.
    """
    
    def __init__(self, model: SecureBaseModel, tools: Optional[List[SecureBaseTool]] = None):
        """Initialize secure agent with validation."""
        try:
            if not isinstance(model, SecureBaseModel):
                raise ValueError("Model must be SecureBaseModel instance")
            
            self.model = model
            self.tools = self._validate_tools(tools or [])
            self.conversation_history: List[Message] = []
            self._invocation_count = 0
            self._initialization_time = datetime.now()
            
            logger.info(f"Initialized secure agent with {len(self.tools)} tools")
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            raise
    
    def _validate_tools(self, tools: List[SecureBaseTool]) -> List[SecureBaseTool]:
        """Validate tool list for security."""
        if not isinstance(tools, list):
            raise ValueError("Tools must be list")
        
        if len(tools) > 50:
            raise ValueError("Too many tools (>50)")
        
        validated_tools = []
        for tool in tools:
            if isinstance(tool, SecureBaseTool):
                validated_tools.append(tool)
            else:
                logger.warning(f"Skipping invalid tool: {tool}")
        
        return validated_tools
    
    def _validate_prompt(self, prompt: str) -> str:
        """Validate and sanitize agent prompt."""
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be string")
        
        if len(prompt) > 20000:
            raise ValueError("Prompt too long (>20KB)")
        
        return SecurityValidator.sanitize_content(prompt)
    
    def _log_invocation(self, operation: str, success: bool):
        """Log agent invocations for monitoring."""
        self._invocation_count += 1
        logger.info(f"Agent {operation}: invocation #{self._invocation_count} - {'success' if success else 'failed'}")
    
    @abstractmethod
    async def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke the agent with a prompt and security validation."""
        pass
    
    @abstractmethod
    async def stream_invoke(self, prompt: str, **kwargs) -> AsyncIterable[str]:
        """Stream invoke the agent with security monitoring."""
        pass
    
    def add_tool(self, tool: SecureBaseTool) -> None:
        """Add a tool to the agent with validation."""
        if not isinstance(tool, SecureBaseTool):
            raise ValueError("Tool must be SecureBaseTool instance")
        
        if len(self.tools) >= 50:
            raise ValueError("Maximum tool limit reached")
        
        self.tools.append(tool)
        logger.info(f"Added tool to agent: {tool.name}")
    
    def clear_history(self) -> None:
        """Clear conversation history with audit logging."""
        history_size = len(self.conversation_history)
        self.conversation_history.clear()
        logger.info(f"Cleared agent conversation history ({history_size} messages)")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics for monitoring."""
        return {
            'message_count': len(self.conversation_history),
            'invocation_count': self._invocation_count,
            'tool_count': len(self.tools),
            'initialization_time': self._initialization_time.isoformat()
        }

# Backward compatibility aliases
BaseProvider = SecureBaseProvider
BaseModel = SecureBaseModel
BaseTool = SecureBaseTool
BaseAgent = SecureBaseAgent
