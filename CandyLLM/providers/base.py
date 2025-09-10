"""
Base Provider Interface for CandyLLM

This module defines the base interface that all AI providers must implement
to be compatible with CandyLLM's intelligent routing and management system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """
    Abstract base class for all AI providers in CandyLLM.
    
    All provider implementations must inherit from this class and implement
    the required abstract methods to ensure compatibility with the routing
    and management systems.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the provider.
        
        Args:
            name: Unique name for this provider
            config: Provider-specific configuration
        """
        self.name = name
        self.config = config or {}
        self.capabilities = self._get_capabilities()
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'status': 'initialized'
        }
        
        logger.info(f"Initialized provider: {name}")
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response from the AI provider.
        
        Args:
            prompt: Input prompt for generation
            **kwargs: Provider-specific parameters
            
        Returns:
            Dictionary containing the response and metadata
            Expected format:
            {
                'content': str,  # Generated content
                'success': bool,  # Whether generation succeeded
                'metadata': {
                    'model': str,  # Model used
                    'tokens_used': int,  # Tokens consumed
                    'processing_time': float,  # Time taken
                    'provider': str  # Provider name
                }
            }
        """
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the AI provider.
        
        Args:
            prompt: Input prompt for generation
            **kwargs: Provider-specific parameters
            
        Yields:
            Chunks of generated content
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model(s) supported by this provider.
        
        Returns:
            Dictionary with model information including capabilities,
            context length, supported tasks, etc.
        """
        pass
    
    @abstractmethod
    def _get_capabilities(self) -> Dict[str, Any]:
        """
        Define the capabilities of this provider.
        
        Returns:
            Dictionary describing provider capabilities:
            {
                'text_generation': bool,
                'code_generation': bool,
                'mathematical_reasoning': bool,
                'multimodal': bool,
                'streaming': bool,
                'function_calling': bool,
                'max_context_length': int,
                'supported_languages': List[str],
                'cost_per_token': float
            }
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the provider.
        
        Returns:
            Health status information
        """
        try:
            # Simple test generation
            test_response = await self.generate("test", max_tokens=1)
            
            return {
                'status': 'healthy',
                'provider': self.name,
                'timestamp': datetime.now().isoformat(),
                'test_successful': test_response.get('success', False)
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'provider': self.name,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_cost_estimate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Estimate the cost of processing a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Cost estimation information
        """
        # Simple token-based estimation
        estimated_tokens = len(prompt.split()) * 1.3  # Rough estimation
        max_tokens = kwargs.get('max_tokens', 100)
        total_tokens = estimated_tokens + max_tokens
        
        cost_per_token = self.capabilities.get('cost_per_token', 0.0)
        estimated_cost = total_tokens * cost_per_token
        
        return {
            'estimated_input_tokens': int(estimated_tokens),
            'estimated_output_tokens': max_tokens,
            'estimated_total_tokens': int(total_tokens),
            'estimated_cost_usd': estimated_cost,
            'provider': self.name
        }
    
    def supports_feature(self, feature: str) -> bool:
        """
        Check if the provider supports a specific feature.
        
        Args:
            feature: Feature name to check
            
        Returns:
            True if feature is supported
        """
        return self.capabilities.get(feature, False)
    
    def get_context_limit(self) -> int:
        """Get the maximum context length supported by this provider."""
        return self.capabilities.get('max_context_length', 4096)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this provider.
        
        Returns:
            Performance metrics dictionary
        """
        # Default implementation - providers can override
        return {
            'provider': self.name,
            'capabilities': self.capabilities,
            'metadata': self.metadata,
            'context_limit': self.get_context_limit(),
            'timestamp': datetime.now().isoformat()
        }


class MockProvider(BaseProvider):
    """
    Mock provider for testing and demonstration purposes.
    
    This provider generates simple responses and can be used for testing
    the CandyLLM system without requiring actual AI API keys.
    """
    
    def __init__(self, name: str = "mock", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.response_templates = [
            "This is a mock response from {provider}.",
            "Mock AI says: Your question about '{prompt}' is interesting.",
            "Simulated response: I understand you're asking about '{prompt}'.",
            "Mock provider {provider} generated this response."
        ]
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a mock response."""
        import random
        import asyncio
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Choose a response template
        template = random.choice(self.response_templates)
        content = template.format(
            provider=self.name,
            prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt
        )
        
        return {
            'content': content,
            'success': True,
            'metadata': {
                'model': f'mock-model-{self.name}',
                'tokens_used': len(content.split()),
                'processing_time': 0.1,
                'provider': self.name
            }
        }
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate a streaming mock response."""
        import asyncio
        
        response = await self.generate(prompt, **kwargs)
        content = response['content']
        
        # Stream word by word
        words = content.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            'name': f'MockModel-{self.name}',
            'version': '1.0',
            'description': 'Mock AI model for testing',
            'max_tokens': 1000,
            'context_length': 4096,
            'capabilities': self.capabilities
        }
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Define mock provider capabilities."""
        return {
            'text_generation': True,
            'code_generation': True,
            'mathematical_reasoning': False,
            'multimodal': False,
            'streaming': True,
            'function_calling': False,
            'max_context_length': 4096,
            'supported_languages': ['en', 'es', 'fr', 'de'],
            'cost_per_token': 0.0001
        }


class ProviderRegistry:
    """
    Registry for managing provider instances and their capabilities.
    """
    
    def __init__(self):
        self.providers = {}
        self.provider_configs = {}
    
    def register_provider(self, provider: BaseProvider, config: Dict[str, Any] = None):
        """
        Register a provider in the registry.
        
        Args:
            provider: Provider instance
            config: Additional configuration for the provider
        """
        self.providers[provider.name] = provider
        self.provider_configs[provider.name] = config or {}
        logger.info(f"Registered provider: {provider.name}")
    
    def get_provider(self, name: str) -> Optional[BaseProvider]:
        """Get a provider by name."""
        return self.providers.get(name)
    
    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        return list(self.providers.keys())
    
    def get_providers_by_capability(self, capability: str) -> List[BaseProvider]:
        """Get all providers that support a specific capability."""
        return [
            provider for provider in self.providers.values()
            if provider.supports_feature(capability)
        ]
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all registered providers."""
        results = {}
        
        for name, provider in self.providers.items():
            results[name] = await provider.health_check()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_providers': len(self.providers),
            'results': results
        }


# Global provider registry instance
provider_registry = ProviderRegistry()


def create_mock_providers() -> List[MockProvider]:
    """
    Create a set of mock providers for testing.
    
    Returns:
        List of configured mock providers
    """
    providers = [
        MockProvider("mock-gpt", {"model": "gpt-4o-mini"}),
        MockProvider("mock-claude", {"model": "claude-3.5-sonnet"}),
        MockProvider("mock-gemini", {"model": "gemini-2.0-flash"})
    ]
    
    # Register in global registry
    for provider in providers:
        provider_registry.register_provider(provider)
    
    return providers


if __name__ == "__main__":
    """
    Demo of the provider system.
    """
    import asyncio
    
    async def demo():
        print("🔧 CandyLLM Provider System Demo")
        print("=" * 35)
        
        # Create mock providers
        providers = create_mock_providers()
        
        # Test each provider
        for provider in providers:
            print(f"\n🤖 Testing {provider.name}:")
            
            # Test generation
            response = await provider.generate("What is artificial intelligence?")
            print(f"  Response: {response['content']}")
            print(f"  Success: {response['success']}")
            
            # Test capabilities
            print(f"  Capabilities: {list(provider.capabilities.keys())}")
            
            # Test health check
            health = await provider.health_check()
            print(f"  Health: {health['status']}")
        
        # Test registry
        print(f"\n📋 Registry has {len(provider_registry.list_providers())} providers")
        
        # Test capability filtering
        text_providers = provider_registry.get_providers_by_capability('text_generation')
        print(f"📝 {len(text_providers)} providers support text generation")
        
        print("\n✅ Provider system demo completed!")
    
    asyncio.run(demo())
