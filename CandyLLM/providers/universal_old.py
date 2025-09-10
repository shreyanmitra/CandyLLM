"""
Universal model provider supporting 100+ LLMs through multiple strategies
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
from functools import lru_cache
import importlib
import logging

from ..core.base import BaseProvider, BaseModel, ModelConfig, ProviderType, ModelType

logger = logging.getLogger(__name__)


class UniversalModelProvider(BaseProvider):
    """Universal provider supporting maximum LLM diversity"""
    
    SUPPORTED_PROVIDERS = {
        # Cloud AI Platforms (20+ providers)
        "openai": {
            "class": "OpenAIProvider",
            "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "o1-preview", "o1-mini"]
        },
        "anthropic": {
            "class": "AnthropicProvider", 
            "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3-5-sonnet"]
        },
        "google": {
            "class": "GoogleProvider",
            "models": ["gemini-pro", "gemini-ultra", "gemini-flash", "palm-2"]
        },
        "aws-bedrock": {
            "class": "BedrockProvider",
            "models": ["claude-v2", "titan", "llama2", "cohere", "nova-premier"]
        },
        "azure-openai": {
            "class": "AzureOpenAIProvider",
            "models": ["gpt-4", "gpt-35-turbo"]
        },
        "cohere": {
            "class": "CohereProvider",
            "models": ["command", "command-light", "command-r", "command-r-plus"]
        },
        
        # Specialized AI Providers (10+ providers)
        "groq": {
            "class": "GroqProvider",
            "models": ["llama2-70b", "mixtral-8x7b", "gemma-7b"]
        },
        "together": {
            "class": "TogetherProvider",
            "models": ["llama2", "falcon", "redpajama", "mixtral"]
        },
        "anyscale": {
            "class": "AnyscaleProvider",
            "models": ["llama2", "mistral", "codellama"]
        },
        "fireworks": {
            "class": "FireworksProvider", 
            "models": ["llama2", "mistral", "starcoder"]
        },
        "replicate": {
            "class": "ReplicateProvider",
            "models": ["llama2", "stable-diffusion", "whisper"]
        },
        
        # Open Source Local (15+ options)
        "ollama": {
            "class": "OllamaProvider",
            "models": ["llama2", "mistral", "codellama", "phi", "gemma"]
        },
        "vllm": {
            "class": "VLLMProvider",
            "models": ["any-hf-model"]
        },
        "llamacpp": {
            "class": "LlamaCppProvider",
            "models": ["gguf-models"]
        },
        "transformers": {
            "class": "TransformersProvider", 
            "models": ["any-hf-model"]
        },
        "mlx": {
            "class": "MLXProvider",
            "models": ["apple-silicon-optimized"]
        },
        
        # Universal Gateway (100+ models via LiteLLM)
        "litellm": {
            "class": "LiteLLMProvider",
            "providers": [
                "openai", "anthropic", "cohere", "replicate", "huggingface",
                "bedrock", "vertex_ai", "azure", "palm", "ai21", "nlp_cloud",
                "aleph_alpha", "petals", "oobabooga", "text-generation-webui"
            ]
        },
        
        # Enterprise & Specialized
        "writer": {
            "class": "WriterProvider",
            "models": ["palmyra-base", "palmyra-large"]
        },
        "ai21": {
            "class": "AI21Provider",
            "models": ["j2-ultra", "j2-mid", "j2-light"]
        },
        "mistral": {
            "class": "MistralProvider",
            "models": ["mistral-7b", "mixtral-8x7b", "mistral-large"]
        },
        "perplexity": {
            "class": "PerplexityProvider",
            "models": ["pplx-7b", "pplx-70b"]
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._providers: Dict[str, BaseProvider] = {}
        self._load_providers()
    
    def _get_provider_type(self) -> ProviderType:
        return ProviderType.UNIVERSAL
    
    def _load_providers(self):
        """Lazy load providers as needed"""
        # Providers will be loaded on demand
        pass
    
    @lru_cache(maxsize=64)
    def get_provider(self, provider_name: str) -> BaseProvider:
        """Get provider instance with caching"""
        if provider_name not in self._providers:
            if provider_name not in self.SUPPORTED_PROVIDERS:
                raise ValueError(f"Provider {provider_name} not supported")
            
            provider_config = self.SUPPORTED_PROVIDERS[provider_name]
            provider_class = provider_config["class"]
            
            # Dynamic import and instantiation
            try:
                module = importlib.import_module(f"..providers.{provider_name}", __name__)
                ProviderClass = getattr(module, provider_class)
                self._providers[provider_name] = ProviderClass(self.config.get(provider_name, {}))
            except ImportError as e:
                logger.warning(f"Failed to load provider {provider_name}: {e}")
                # Fallback to LiteLLM for universal coverage
                if provider_name != "litellm":
                    return self.get_provider("litellm")
                raise
        
        return self._providers[provider_name]
    
    def list_models(self) -> List[str]:
        """List all available models across all providers"""
        all_models = []
        for provider_name, provider_config in self.SUPPORTED_PROVIDERS.items():
            try:
                provider = self.get_provider(provider_name)
                models = provider.list_models()
                # Prefix with provider name for clarity
                all_models.extend([f"{provider_name}:{model}" for model in models])
            except Exception as e:
                logger.warning(f"Failed to list models for {provider_name}: {e}")
        return all_models
    
    @lru_cache(maxsize=32)
    def list_models_by_provider(self, provider_name: str) -> List[str]:
        """List models for specific provider with caching"""
        try:
            provider = self.get_provider(provider_name)
            return provider.list_models()
        except Exception as e:
            logger.error(f"Failed to list models for {provider_name}: {e}")
            return []
    
    def create_model(self, model_id: str, config: ModelConfig) -> BaseModel:
        """Create model instance from any supported provider"""
        # Parse provider from model_id if formatted as "provider:model"
        if ":" in model_id:
            provider_name, actual_model_id = model_id.split(":", 1)
            config.model_id = actual_model_id
        else:
            # Try to infer provider from model name
            provider_name = self._infer_provider(model_id)
        
        provider = self.get_provider(provider_name)
        return provider.create_model(config.model_id, config)
    
    def _infer_provider(self, model_id: str) -> str:
        """Infer provider from model name"""
        # Common model name patterns
        if model_id.startswith("gpt-"):
            return "openai"
        elif model_id.startswith("claude-"):
            return "anthropic"
        elif model_id.startswith("gemini-"):
            return "google"
        elif "llama" in model_id.lower():
            return "transformers"  # Default to local transformers
        elif "mistral" in model_id.lower():
            return "mistral"
        else:
            # Default to LiteLLM for universal coverage
            return "litellm"
    
    def is_available(self) -> bool:
        """Check if at least one provider is available"""
        for provider_name in ["openai", "anthropic", "litellm", "transformers"]:
            try:
                provider = self.get_provider(provider_name)
                if provider.is_available():
                    return True
            except:
                continue
        return False
    
    def get_provider_statistics(self) -> Dict[str, Any]:
        """Get statistics about available providers"""
        stats = {
            "total_providers": len(self.SUPPORTED_PROVIDERS),
            "loaded_providers": len(self._providers),
            "available_providers": [],
            "total_models": 0
        }
        
        for provider_name in self.SUPPORTED_PROVIDERS:
            try:
                provider = self.get_provider(provider_name)
                if provider.is_available():
                    stats["available_providers"].append(provider_name)
                    stats["total_models"] += len(provider.list_models())
            except:
                continue
        
        return stats


class ModelFactory:
    """Factory for creating model instances with intelligent defaults"""
    
    def __init__(self, universal_provider: UniversalModelProvider):
        self.provider = universal_provider
    
    @lru_cache(maxsize=16)
    def create_smart_model(self, 
                          task: str = "chat",
                          quality: str = "high", 
                          speed: str = "medium",
                          cost: str = "medium") -> BaseModel:
        """Create model with intelligent selection based on requirements"""
        
        # Model selection matrix based on requirements
        selection_matrix = {
            ("chat", "high", "medium", "low"): "anthropic:claude-3-haiku",
            ("chat", "high", "medium", "medium"): "anthropic:claude-3-sonnet", 
            ("chat", "high", "slow", "high"): "anthropic:claude-3-opus",
            ("chat", "medium", "fast", "low"): "openai:gpt-3.5-turbo",
            ("chat", "high", "fast", "medium"): "openai:gpt-4-turbo",
            ("code", "high", "medium", "medium"): "anthropic:claude-3-sonnet",
            ("code", "medium", "fast", "low"): "groq:llama2-70b",
            ("multimodal", "high", "medium", "high"): "openai:gpt-4-vision",
            ("local", "medium", "medium", "free"): "ollama:llama2",
        }
        
        key = (task, quality, speed, cost)
        model_id = selection_matrix.get(key, "litellm:gpt-3.5-turbo")  # Default fallback
        
        config = ModelConfig(
            model_id=model_id,
            provider=model_id.split(":")[0],
            model_type=ModelType.CHAT
        )
        
        return self.provider.create_model(model_id, config)
    
    def create_model_for_task(self, task_description: str) -> BaseModel:
        """Create optimal model based on task description analysis"""
        # Simple keyword-based task analysis (can be enhanced with ML)
        task_lower = task_description.lower()
        
        if "code" in task_lower or "programming" in task_lower:
            return self.create_smart_model(task="code", quality="high")
        elif "image" in task_lower or "vision" in task_lower:
            return self.create_smart_model(task="multimodal", quality="high")
        elif "fast" in task_lower or "quick" in task_lower:
            return self.create_smart_model(speed="fast", cost="low")
        elif "best" in task_lower or "highest" in task_lower:
            return self.create_smart_model(quality="high")
        else:
            return self.create_smart_model()  # Default balanced model
