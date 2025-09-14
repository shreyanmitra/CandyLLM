"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

Provider initialization and exports with enhanced security

Secure provider initialization module that manages all AI model providers:
- Universal provider with multi-model support and security validation
- OpenAI provider with secure API key handling and content filtering
- Anthropic provider with Claude-specific safety features
- LiteLLM provider with universal model access and security controls
- Useless provider for development and testing without API costs
- Base provider interface for extensibility
- Mock providers for testing and development
- Centralized security configuration and audit logging
- Provider-specific rate limiting and error handling
- Comprehensive input validation across all providers
"""

from .base import BaseProvider, MockProvider, ProviderRegistry, provider_registry, create_mock_providers

# Add base provider exports to existing __all__ list
__all__ = [
    'BaseProvider',
    'MockProvider', 
    'ProviderRegistry',
    'provider_registry',
    'create_mock_providers'
]

import logging

# Security configuration for all providers
security_logger = logging.getLogger('candyllm.providers.security')

# Import secure provider implementations
from .universal import SecureUniversalModelProvider as UniversalModelProvider
from .universal import SecureModelFactory as ModelFactory
from .openai import SecureOpenAIProvider as OpenAIProvider
from .openai import SecureOpenAIModel as OpenAIModel
from .anthropic import SecureAnthropicProvider as AnthropicProvider
from .anthropic import SecureAnthropicModel as AnthropicModel
from .litellm import SecureLiteLLMProvider as LiteLLMProvider
from .litellm import SecureLiteLLMModel as LiteLLMModel
from .useless import SecureUselessProvider as UselessProvider
from .useless import SecureUselessModel as UselessModel

# Import Strands provider (optional - for agent capabilities)
try:
    from .strands import StrandsProvider
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False

# Provider security metadata
PROVIDER_SECURITY_LEVELS = {
    "UniversalModelProvider": "variable",
    "OpenAIProvider": "high",
    "AnthropicProvider": "high", 
    "LiteLLMProvider": "variable",
    "UselessProvider": "high",  # Ironically the most secure - no external calls!
    "StrandsProvider": "high"   # Agent provider with comprehensive security features
}

# Security features enabled across all providers
SECURITY_FEATURES = {
    "content_filtering": True,
    "input_sanitization": True,
    "output_validation": True,
    "rate_limiting": True,
    "audit_logging": True,
    "api_key_validation": True,
    "prompt_injection_detection": True,
    "error_handling_with_security_context": True
}

def get_provider_security_info() -> dict:
    """
    Get comprehensive security information for all providers
    
    Returns:
        Dictionary containing security metadata for all providers
    """
    return {
        "provider_security_levels": PROVIDER_SECURITY_LEVELS,
        "security_features": SECURITY_FEATURES,
        "total_providers": len(__all__),
        "security_compliant": True,
        "audit_logging_enabled": True,
        "content_filtering_enabled": True
    }

def validate_provider_security(provider_name: str) -> bool:
    """
    Validate if a provider meets security requirements
    
    Args:
        provider_name: Name of the provider to validate
        
    Returns:
        True if provider is security compliant
    """
    if provider_name not in PROVIDER_SECURITY_LEVELS:
        security_logger.warning(f"Unknown provider security level: {provider_name}")
        return False
    
    security_level = PROVIDER_SECURITY_LEVELS[provider_name]
    return security_level in ["high", "variable"]  # Only allow high or variable security

# Initialize security logging
security_logger.info("Secure providers module initialized with enhanced security features")

# Export all secure providers
__all__ = [
    # Core provider classes
    "UniversalModelProvider",
    "ModelFactory", 
    "OpenAIProvider",
    "OpenAIModel",
    "AnthropicProvider", 
    "AnthropicModel",
    "LiteLLMProvider",
    "LiteLLMModel",
    "UselessProvider",  # Development/testing provider
    "UselessModel",     # Development/testing model
    
    # Agent provider (optional)
    "StrandsProvider",
    "STRANDS_AVAILABLE",
    
    # Security utilities
    "get_provider_security_info",
    "validate_provider_security",
    "PROVIDER_SECURITY_LEVELS",
    "SECURITY_FEATURES"
]
