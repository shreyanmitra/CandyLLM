"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

Universal model provider with comprehensive security for 100+ LLMs

Secure universal provider supporting maximum LLM diversity with enterprise security:
- Multi-provider security validation and API key management
- Comprehensive input sanitization across all model types
- Provider-specific rate limiting and quota management
- Unified audit logging for all providers and models
- Error handling with security context preservation
- Protection against prompt injection across all providers
- Compliance with multiple provider usage policies
- Universal content filtering and safety enforcement
- Intelligent model selection with security considerations
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
from functools import lru_cache
import importlib
import logging
import time
import json
import html
import re
from datetime import datetime

from ..core.base import BaseProvider, BaseModel, ModelConfig, ProviderType, ModelType

# Security logging
security_logger = logging.getLogger('candyllm.providers.universal.security')
logger = logging.getLogger(__name__)

# Universal security patterns across all providers
UNIVERSAL_SECURITY_PATTERNS = [
    r'ignore\s+(all|previous)\s+(instructions|prompts|rules|guidelines)',
    r'jailbreak|DAN\s+mode|evil\s+mode|developer\s+mode|admin\s+mode',
    r'pretend\s+to\s+be\s+(uncensored|unfiltered|unrestricted|unlimited)',
    r'act\s+as\s+if\s+you\s+(have\s+no\s+limitations|are\s+not\s+AI|are\s+human)',
    r'bypass\s+(safety|security|content)\s+(filters|policies|guidelines|restrictions)',
    r'(harmful|illegal|unethical|dangerous|malicious)\s+(content|instructions|advice|code)',
    r'generate\s+(malware|virus|exploit|harmful|illegal|dangerous)\s+(code|content|instructions)',
    r'(child|minor|underage)\s+(sexual|explicit|inappropriate|abuse)',
    r'(suicide|self[\s-]?harm|self[\s-]?injury)\s+(methods|instructions|guidance)',
    r'(terrorist|extremist|violent)\s+(instructions|plans|methods)',
]

# Compile patterns for performance
COMPILED_SECURITY_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in UNIVERSAL_SECURITY_PATTERNS]

def validate_universal_input(text: str) -> tuple[bool, List[str]]:
    """
    Universal input validation across all providers
    
    Args:
        text: Text to validate
        
    Returns:
        Tuple of (is_valid, violations_found)
    """
    violations = []
    
    for pattern in COMPILED_SECURITY_PATTERNS:
        if pattern.search(text):
            violations.append(pattern.pattern)
    
    return len(violations) == 0, violations

def sanitize_universal_input(text: str) -> str:
    """
    Universal input sanitization for all model providers
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text safe for all providers
    """
    # HTML escape to prevent injection
    sanitized = html.escape(text)
    
    # Remove potential prompt injection markers from all providers
    injection_markers = [
        'Human:', 'Assistant:', 'System:', 'User:', 'Bot:', 'AI:',
        '<|endoftext|>', '<|im_start|>', '<|im_end|>', '<|system|>', '<|user|>', '<|assistant|>',
        '<thinking>', '</thinking>', '###', '---', '```', '***',
        '[INST]', '[/INST]', '<s>', '</s>'
    ]
    
    for marker in injection_markers:
        sanitized = sanitized.replace(marker, '')
    
    # Normalize whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # Conservative length limit for universal compatibility
    if len(sanitized) > 32000:
        sanitized = sanitized[:32000] + "... [content truncated for security]"
    
    return sanitized

class SecureUniversalModelProvider(BaseProvider):
    """
    Secure universal provider supporting maximum LLM diversity with enterprise security
    
    Security Features:
    - Multi-provider security validation
    - Universal input sanitization and content filtering
    - Provider-specific rate limiting and quota management
    - Comprehensive audit logging across all providers
    - Error handling with security context
    - Intelligent model selection with security considerations
    """
    
    SUPPORTED_PROVIDERS = {
        # Cloud AI Platforms with security metadata
        "openai": {
            "class": "SecureOpenAIProvider",
            "security_level": "high",
            "rate_limit": 60,
            "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "o1-preview", "o1-mini"]
        },
        "anthropic": {
            "class": "SecureAnthropicProvider",
            "security_level": "high", 
            "rate_limit": 50,
            "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3-5-sonnet"]
        },
        "google": {
            "class": "SecureGoogleProvider",
            "security_level": "high",
            "rate_limit": 40,
            "models": ["gemini-pro", "gemini-ultra", "gemini-flash", "palm-2"]
        },
        "aws-bedrock": {
            "class": "SecureBedrockProvider",
            "security_level": "high",
            "rate_limit": 40,
            "models": ["claude-v2", "titan", "llama2", "cohere", "nova-premier"]
        },
        "azure-openai": {
            "class": "SecureAzureOpenAIProvider",
            "security_level": "high",
            "rate_limit": 60,
            "models": ["gpt-4", "gpt-35-turbo"]
        },
        "cohere": {
            "class": "SecureCohereProvider",
            "security_level": "medium",
            "rate_limit": 40,
            "models": ["command", "command-light", "command-r", "command-r-plus"]
        },
        
        # Specialized AI Providers with security considerations
        "groq": {
            "class": "SecureGroqProvider",
            "security_level": "medium",
            "rate_limit": 30,
            "models": ["llama2-70b", "mixtral-8x7b", "gemma-7b"]
        },
        "together": {
            "class": "SecureTogetherProvider",
            "security_level": "medium",
            "rate_limit": 30,
            "models": ["llama2", "falcon", "redpajama", "mixtral"]
        },
        "anyscale": {
            "class": "SecureAnyscaleProvider",
            "security_level": "medium",
            "rate_limit": 30,
            "models": ["llama2", "mistral", "codellama"]
        },
        "fireworks": {
            "class": "SecureFireworksProvider",
            "security_level": "medium",
            "rate_limit": 30,
            "models": ["llama2", "mistral", "starcoder"]
        },
        "replicate": {
            "class": "SecureReplicateProvider",
            "security_level": "medium",
            "rate_limit": 20,
            "models": ["llama2", "stable-diffusion", "whisper"]
        },
        
        # Open Source Local with security monitoring
        "ollama": {
            "class": "SecureOllamaProvider",
            "security_level": "low",
            "rate_limit": 10,
            "models": ["llama2", "mistral", "codellama", "phi", "gemma"]
        },
        "vllm": {
            "class": "SecureVLLMProvider",
            "security_level": "low",
            "rate_limit": 10,
            "models": ["any-hf-model"]
        },
        "llamacpp": {
            "class": "SecureLlamaCppProvider",
            "security_level": "low",
            "rate_limit": 10,
            "models": ["gguf-models"]
        },
        "transformers": {
            "class": "SecureTransformersProvider",
            "security_level": "low",
            "rate_limit": 10,
            "models": ["any-hf-model"]
        },
        
        # Universal Gateway with enhanced security
        "litellm": {
            "class": "SecureLiteLLMProvider",
            "security_level": "variable",
            "rate_limit": 40,
            "providers": [
                "openai", "anthropic", "cohere", "replicate", "huggingface",
                "bedrock", "vertex_ai", "azure", "palm", "ai21", "nlp_cloud"
            ]
        },
        
        # Development/Testing Providers
        "useless": {
            "class": "SecureUselessProvider",
            "security_level": "high",  # Ironically the most secure - no external calls!
            "rate_limit": 1000,  # No rate limiting needed for useless responses
            "models": ["useless-basic", "useless-verbose", "useless-slow", "useless-streamy", "useless-error", "useless-json"],
            "description": "Development provider for testing and debugging without API costs",
            "features": ["zero_cost", "predictable_responses", "streaming_support", "debug_mode"]
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize secure universal provider
        
        Args:
            config: Configuration dictionary with security validation
        """
        super().__init__(config)
        self._providers: Dict[str, BaseProvider] = {}
        
        # Security configuration
        self.content_filtering_enabled = config.get('content_filtering', True)
        self.audit_logging_enabled = config.get('audit_logging', True)
        self.rate_limit_enabled = config.get('rate_limiting', True)
        self.security_level_required = config.get('min_security_level', 'medium')
        
        # Universal rate limiting tracking
        self.universal_request_timestamps = {}
        
        # Initialize security logging
        security_logger.info("SecureUniversalModelProvider initialized")
    
    def _check_universal_rate_limit(self, provider_name: str) -> bool:
        """
        Check universal rate limits for provider
        
        Args:
            provider_name: Provider to check
            
        Returns:
            True if within limits, False if rate limited
        """
        if not self.rate_limit_enabled:
            return True
        
        now = time.time()
        minute_ago = now - 60
        
        # Initialize provider tracking
        if provider_name not in self.universal_request_timestamps:
            self.universal_request_timestamps[provider_name] = []
        
        # Clean old timestamps
        self.universal_request_timestamps[provider_name] = [
            ts for ts in self.universal_request_timestamps[provider_name] if ts > minute_ago
        ]
        
        # Get provider-specific rate limit
        provider_info = self.SUPPORTED_PROVIDERS.get(provider_name, {"rate_limit": 10})
        max_requests = provider_info["rate_limit"]
        
        # Check current rate
        current_requests = len(self.universal_request_timestamps[provider_name])
        if current_requests >= max_requests:
            security_logger.warning(f"Universal rate limit exceeded for {provider_name}: {current_requests} requests")
            return False
        
        # Record current request
        self.universal_request_timestamps[provider_name].append(now)
        return True
    
    def _log_universal_api_call(self, provider_name: str, model_id: str, method: str, 
                               input_data: Any, response_data: Any = None, error: Any = None):
        """
        Log API calls with universal audit trail
        
        Args:
            provider_name: Provider name
            model_id: Model identifier
            method: API method
            input_data: Input data (sanitized)
            response_data: Response data (optional)
            error: Error information (optional)
        """
        if not self.audit_logging_enabled:
            return
        
        provider_info = self.SUPPORTED_PROVIDERS.get(provider_name, {})
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'universal_provider': True,
            'provider': provider_name,
            'model': model_id,
            'method': method,
            'input_length': len(str(input_data)) if input_data else 0,
            'success': error is None,
            'error': str(error) if error else None,
            'response_length': len(str(response_data)) if response_data else 0,
            'security_level': provider_info.get('security_level', 'unknown')
        }
        
        security_logger.info(f"Universal API call: {json.dumps(log_entry)}")
    
    def _validate_security_level(self, provider_name: str) -> bool:
        """
        Validate if provider meets minimum security requirements
        
        Args:
            provider_name: Provider to validate
            
        Returns:
            True if security level is acceptable
        """
        provider_info = self.SUPPORTED_PROVIDERS.get(provider_name, {})
        provider_security = provider_info.get('security_level', 'low')
        
        security_hierarchy = {'low': 0, 'medium': 1, 'high': 2, 'variable': 1}
        required_level = security_hierarchy.get(self.security_level_required, 1)
        provider_level = security_hierarchy.get(provider_security, 0)
        
        return provider_level >= required_level
    
    def _get_provider_type(self) -> ProviderType:
        """Get provider type"""
        return ProviderType.UNIVERSAL
    
    @lru_cache(maxsize=64)
    def get_provider(self, provider_name: str) -> BaseProvider:
        """
        Get provider instance with security validation and caching
        
        Args:
            provider_name: Provider to get
            
        Returns:
            Secure provider instance
        """
        if provider_name not in self._providers:
            if provider_name not in self.SUPPORTED_PROVIDERS:
                raise ValueError(f"Provider {provider_name} not supported")
            
            # Validate security level
            if not self._validate_security_level(provider_name):
                security_logger.warning(f"Provider {provider_name} does not meet minimum security level")
                raise ValueError(f"Provider {provider_name} security level insufficient")
            
            provider_config = self.SUPPORTED_PROVIDERS[provider_name]
            provider_class = provider_config["class"]
            
            # Dynamic import and instantiation with error handling
            try:
                # Import from current module's providers
                if provider_name in ["openai", "anthropic", "litellm"]:
                    module_name = provider_name
                else:
                    module_name = "litellm"  # Fallback to LiteLLM for universal coverage
                
                module = importlib.import_module(f".{module_name}", "CandyLLM.providers")
                
                # Get secure provider class
                if hasattr(module, provider_class):
                    ProviderClass = getattr(module, provider_class)
                else:
                    # Fallback to standard provider class
                    fallback_name = provider_class.replace("Secure", "")
                    ProviderClass = getattr(module, fallback_name)
                
                # Initialize with security configuration
                provider_config_dict = self.config.get(provider_name, {})
                provider_config_dict.update({
                    'content_filtering': self.content_filtering_enabled,
                    'audit_logging': self.audit_logging_enabled,
                    'rate_limiting': self.rate_limit_enabled
                })
                
                self._providers[provider_name] = ProviderClass(provider_config_dict)
                security_logger.debug(f"Initialized secure provider: {provider_name}")
                
            except ImportError as e:
                security_logger.warning(f"Failed to load provider {provider_name}: {e}")
                # Fallback to LiteLLM for universal coverage
                if provider_name != "litellm":
                    return self.get_provider("litellm")
                raise
            except Exception as e:
                security_logger.error(f"Error initializing provider {provider_name}: {e}")
                raise
        
        return self._providers[provider_name]
    
    def list_models(self) -> List[str]:
        """List all available models across all secure providers"""
        all_models = []
        for provider_name, provider_config in self.SUPPORTED_PROVIDERS.items():
            try:
                # Validate security level before listing
                if not self._validate_security_level(provider_name):
                    continue
                
                provider = self.get_provider(provider_name)
                models = provider.list_models()
                # Prefix with provider name for clarity
                all_models.extend([f"{provider_name}:{model}" for model in models])
            except Exception as e:
                security_logger.warning(f"Failed to list models for {provider_name}: {e}")
        
        self._log_universal_api_call("universal", "list_models", "list_models", None, all_models)
        return all_models
    
    @lru_cache(maxsize=32)
    def list_models_by_provider(self, provider_name: str) -> List[str]:
        """List models for specific provider with security validation"""
        try:
            # Security validation
            if not self._validate_security_level(provider_name):
                security_logger.warning(f"Provider {provider_name} blocked due to insufficient security level")
                return []
            
            provider = self.get_provider(provider_name)
            models = provider.list_models()
            
            self._log_universal_api_call(provider_name, "list_models", "list_models_by_provider", None, models)
            return models
            
        except Exception as e:
            security_logger.error(f"Failed to list models for {provider_name}: {e}")
            self._log_universal_api_call(provider_name, "list_models", "list_models_by_provider", None, error=e)
            return []
    
    def create_model(self, model_id: str, config: ModelConfig) -> BaseModel:
        """
        Create secure model instance from any supported provider
        
        Args:
            model_id: Model identifier (can be provider:model format)
            config: Model configuration
            
        Returns:
            Secure model instance
        """
        try:
            # Parse provider from model_id if formatted as "provider:model"
            if ":" in model_id:
                provider_name, actual_model_id = model_id.split(":", 1)
                config.model_id = actual_model_id
            else:
                # Try to infer provider from model name
                provider_name = self._infer_provider(model_id)
                actual_model_id = model_id
            
            # Security validation
            if not self._validate_security_level(provider_name):
                raise ValueError(f"Provider {provider_name} does not meet security requirements")
            
            # Rate limiting check
            if not self._check_universal_rate_limit(provider_name):
                raise Exception(f"Rate limit exceeded for provider {provider_name}")
            
            provider = self.get_provider(provider_name)
            model = provider.create_model(actual_model_id, config)
            
            self._log_universal_api_call(provider_name, actual_model_id, "create_model", config.__dict__, "success")
            security_logger.debug(f"Created secure model: {provider_name}:{actual_model_id}")
            
            return model
            
        except Exception as e:
            self._log_universal_api_call(provider_name if 'provider_name' in locals() else "unknown", 
                                       model_id, "create_model", config.__dict__, error=e)
            security_logger.error(f"Failed to create model {model_id}: {e}")
            raise
    
    def _infer_provider(self, model_id: str) -> str:
        """
        Infer provider from model name with security considerations
        
        Args:
            model_id: Model identifier
            
        Returns:
            Provider name
        """
        # Common model name patterns with security priority
        model_id_lower = model_id.lower()
        
        # High security providers first
        if model_id_lower.startswith("gpt-"):
            return "openai"
        elif model_id_lower.startswith("claude-"):
            return "anthropic"
        elif model_id_lower.startswith("gemini-"):
            return "google"
        elif "llama" in model_id_lower:
            # Prefer secure cloud providers for LLaMA models
            return "groq" if self._validate_security_level("groq") else "ollama"
        elif "mistral" in model_id_lower:
            return "groq" if self._validate_security_level("groq") else "ollama"
        else:
            # Default to LiteLLM for universal coverage with medium security
            return "litellm"
    
    def is_available(self) -> bool:
        """Check if at least one secure provider is available"""
        # Check high-security providers first
        for provider_name in ["openai", "anthropic", "google", "litellm"]:
            try:
                if not self._validate_security_level(provider_name):
                    continue
                provider = self.get_provider(provider_name)
                if provider.is_available():
                    return True
            except:
                continue
        return False
    
    def get_provider_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about available secure providers"""
        stats = {
            "total_providers": len(self.SUPPORTED_PROVIDERS),
            "loaded_providers": len(self._providers),
            "available_providers": [],
            "security_compliant_providers": [],
            "total_models": 0,
            "security_levels": {"high": [], "medium": [], "low": []},
            "security_features": {
                "content_filtering": self.content_filtering_enabled,
                "audit_logging": self.audit_logging_enabled,
                "rate_limiting": self.rate_limit_enabled,
                "min_security_level": self.security_level_required
            }
        }
        
        for provider_name, provider_info in self.SUPPORTED_PROVIDERS.items():
            try:
                # Check security compliance
                if self._validate_security_level(provider_name):
                    stats["security_compliant_providers"].append(provider_name)
                    
                    provider = self.get_provider(provider_name)
                    if provider.is_available():
                        stats["available_providers"].append(provider_name)
                        stats["total_models"] += len(provider.list_models())
                
                # Categorize by security level
                security_level = provider_info.get("security_level", "low")
                if security_level in stats["security_levels"]:
                    stats["security_levels"][security_level].append(provider_name)
                    
            except:
                continue
        
        return stats


class SecureModelFactory:
    """
    Secure factory for creating model instances with intelligent defaults and security validation
    """
    
    def __init__(self, universal_provider: SecureUniversalModelProvider):
        """
        Initialize secure model factory
        
        Args:
            universal_provider: Secure universal provider instance
        """
        self.provider = universal_provider
        security_logger.debug("SecureModelFactory initialized")
    
    @lru_cache(maxsize=16)
    def create_smart_model(self, 
                          task: str = "chat",
                          quality: str = "high", 
                          speed: str = "medium",
                          cost: str = "medium",
                          security_level: str = "high") -> BaseModel:
        """
        Create model with intelligent selection based on requirements and security
        
        Args:
            task: Task type (chat, code, multimodal, etc.)
            quality: Quality requirement (low, medium, high)
            speed: Speed requirement (slow, medium, fast)
            cost: Cost consideration (free, low, medium, high)
            security_level: Security requirement (low, medium, high)
            
        Returns:
            Optimally selected secure model
        """
        
        # Enhanced selection matrix with security considerations
        selection_matrix = {
            # High security options
            ("chat", "high", "medium", "low", "high"): "anthropic:claude-3-haiku",
            ("chat", "high", "medium", "medium", "high"): "anthropic:claude-3-sonnet", 
            ("chat", "high", "slow", "high", "high"): "anthropic:claude-3-opus",
            ("chat", "medium", "fast", "low", "high"): "openai:gpt-3.5-turbo",
            ("chat", "high", "fast", "medium", "high"): "openai:gpt-4-turbo",
            ("code", "high", "medium", "medium", "high"): "anthropic:claude-3-sonnet",
            ("multimodal", "high", "medium", "high", "high"): "openai:gpt-4-vision",
            
            # Medium security options
            ("chat", "medium", "fast", "low", "medium"): "groq:llama2-70b",
            ("code", "medium", "fast", "low", "medium"): "groq:mixtral-8x7b",
            ("chat", "medium", "medium", "medium", "medium"): "litellm:gpt-3.5-turbo",
            
            # Low security / local options
            ("chat", "medium", "medium", "free", "low"): "ollama:llama2",
            ("code", "medium", "medium", "free", "low"): "ollama:codellama",
        }
        
        key = (task, quality, speed, cost, security_level)
        model_id = selection_matrix.get(key)
        
        # Fallback logic with security prioritization
        if not model_id:
            if security_level == "high":
                model_id = "anthropic:claude-3-sonnet"  # Secure default
            elif security_level == "medium":
                model_id = "litellm:gpt-3.5-turbo"  # Balanced default
            else:
                model_id = "ollama:llama2"  # Local default
        
        try:
            config = ModelConfig(
                model_id=model_id.split(":")[1] if ":" in model_id else model_id,
                provider=model_id.split(":")[0] if ":" in model_id else "litellm",
                model_type=ModelType.CHAT
            )
            
            model = self.provider.create_model(model_id, config)
            
            self.provider._log_universal_api_call(
                config.provider, config.model_id, "create_smart_model",
                {"task": task, "quality": quality, "speed": speed, "cost": cost, "security_level": security_level},
                "success"
            )
            
            return model
            
        except Exception as e:
            security_logger.error(f"Failed to create smart model with requirements {key}: {e}")
            # Fallback to most secure available option
            return self._create_fallback_model()
    
    def create_model_for_task(self, task_description: str, security_level: str = "high") -> BaseModel:
        """
        Create optimal secure model based on task description analysis
        
        Args:
            task_description: Natural language task description
            security_level: Required security level
            
        Returns:
            Optimally selected secure model for the task
        """
        try:
            # Enhanced keyword-based task analysis with security awareness
            task_lower = task_description.lower()
            
            # Determine task requirements
            if any(keyword in task_lower for keyword in ["code", "programming", "debug", "software"]):
                return self.create_smart_model(task="code", quality="high", security_level=security_level)
            elif any(keyword in task_lower for keyword in ["image", "vision", "visual", "picture"]):
                return self.create_smart_model(task="multimodal", quality="high", security_level=security_level)
            elif any(keyword in task_lower for keyword in ["fast", "quick", "urgent", "immediate"]):
                return self.create_smart_model(speed="fast", cost="low", security_level=security_level)
            elif any(keyword in task_lower for keyword in ["best", "highest", "premium", "quality"]):
                return self.create_smart_model(quality="high", security_level=security_level)
            elif any(keyword in task_lower for keyword in ["cheap", "free", "budget", "cost"]):
                return self.create_smart_model(cost="low", security_level=security_level)
            else:
                # Default balanced model with specified security
                return self.create_smart_model(security_level=security_level)
                
        except Exception as e:
            security_logger.error(f"Failed to create model for task '{task_description}': {e}")
            return self._create_fallback_model()
    
    def _create_fallback_model(self) -> BaseModel:
        """
        Create a fallback model when all else fails
        
        Returns:
            Most secure available fallback model
        """
        fallback_options = [
            "anthropic:claude-3-haiku",  # High security, cost effective
            "openai:gpt-3.5-turbo",      # High security, fast
            "litellm:gpt-3.5-turbo",     # Universal fallback
            "ollama:llama2"              # Local fallback
        ]
        
        for model_id in fallback_options:
            try:
                provider_name = model_id.split(":")[0]
                if self.provider._validate_security_level(provider_name):
                    config = ModelConfig(
                        model_id=model_id.split(":")[1],
                        provider=provider_name,
                        model_type=ModelType.CHAT
                    )
                    return self.provider.create_model(model_id, config)
            except:
                continue
        
        # If all else fails, raise an error
        raise RuntimeError("No secure fallback model available")

# Aliases for backward compatibility
UniversalModelProvider = SecureUniversalModelProvider
ModelFactory = SecureModelFactory
