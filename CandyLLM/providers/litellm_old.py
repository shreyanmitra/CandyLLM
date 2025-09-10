"""
LiteLLM provider supporting 100+ models from various providers
"""

import asyncio
import litellm
from typing import Dict, List, Any, Optional, AsyncGenerator
import logging

from ..core.base import BaseProvider, BaseModel, ModelConfig, ModelResponse, StreamingChunk, Message, ProviderType, ModelType

logger = logging.getLogger(__name__)


class LiteLLMProvider(BaseProvider):
    """
    Universal provider using LiteLLM to support 100+ models
    Covers: OpenAI, Anthropic, Cohere, Replicate, Huggingface, Bedrock, 
    VertexAI, Azure, PaLM, AI21, NLP Cloud, Aleph Alpha, Petals, etc.
    """
    
    # Major provider categories supported by LiteLLM
    SUPPORTED_PROVIDERS = {
        "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "o1-preview", "o1-mini"],
        "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3-5-sonnet"],
        "cohere": ["command", "command-light", "command-r", "command-r-plus"],
        "replicate": ["llama-2-70b-chat", "mistral-7b-instruct", "mixtral-8x7b"],
        "huggingface": ["microsoft/DialoGPT-medium", "meta-llama/Llama-2-7b-chat-hf"],
        "bedrock": ["anthropic.claude-v2", "amazon.titan-text-lite-v1", "ai21.j2-ultra-v1"],
        "vertex_ai": ["text-bison", "chat-bison", "gemini-pro"],
        "azure": ["azure/gpt-4", "azure/gpt-35-turbo"],
        "palm": ["palm/chat-bison", "palm/text-bison"],
        "ai21": ["j2-ultra", "j2-mid", "j2-light"],
        "nlp_cloud": ["dolphin"],
        "aleph_alpha": ["luminous-base", "luminous-extended"],
        "petals": ["petals-team/StableBeluga2"],
        "together_ai": ["togethercomputer/llama-2-7b-chat"],
        "anyscale": ["meta-llama/Llama-2-7b-chat-hf"],
        "groq": ["llama2-70b-4096", "mixtral-8x7b-32768"],
        "deepinfra": ["meta-llama/Llama-2-70b-chat-hf"],
        "perplexity": ["pplx-7b-chat", "pplx-70b-chat"],
        "fireworks_ai": ["accounts/fireworks/models/llama-v2-7b-chat"],
        "cloudflare": ["@cf/meta/llama-2-7b-chat-int8"],
        "ollama": ["llama2", "mistral", "codellama", "phi", "gemma"]
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config_dict = config
        
        # Set API keys from config
        self._setup_api_keys()
        
        # Configure LiteLLM
        litellm.set_verbose = config.get('verbose', False)
        litellm.drop_params = True  # Drop unsupported params automatically
        litellm.telemetry = False  # Disable telemetry for privacy
    
    def _setup_api_keys(self):
        """Setup API keys for various providers"""
        key_mappings = {
            'openai_api_key': 'OPENAI_API_KEY',
            'anthropic_api_key': 'ANTHROPIC_API_KEY',
            'cohere_api_key': 'COHERE_API_KEY',
            'replicate_api_token': 'REPLICATE_API_TOKEN',
            'huggingface_api_key': 'HUGGINGFACE_API_KEY',
            'ai21_api_key': 'AI21_API_KEY',
            'together_api_key': 'TOGETHER_API_KEY',
            'anyscale_api_key': 'ANYSCALE_API_KEY',
            'groq_api_key': 'GROQ_API_KEY',
            'perplexity_api_key': 'PERPLEXITYAI_API_KEY',
            'fireworks_api_key': 'FIREWORKS_API_KEY',
            'deepinfra_api_key': 'DEEPINFRA_API_KEY'
        }
        
        for config_key, env_key in key_mappings.items():
            if config_key in self.config_dict:
                import os
                os.environ[env_key] = self.config_dict[config_key]
    
    def _get_provider_type(self) -> ProviderType:
        return ProviderType.UNIVERSAL
    
    def list_models(self) -> List[str]:
        """List all supported models across all providers"""
        all_models = []
        for provider, models in self.SUPPORTED_PROVIDERS.items():
            # Add provider prefix for clarity
            all_models.extend([f"{provider}/{model}" for model in models])
        return all_models
    
    def list_models_by_provider(self, provider: str) -> List[str]:
        """List models for a specific provider"""
        return self.SUPPORTED_PROVIDERS.get(provider, [])
    
    def create_model(self, model_id: str, config: ModelConfig) -> 'LiteLLMModel':
        return LiteLLMModel(
            model_id=model_id,
            config=config,
            provider_config=self.config_dict
        )
    
    def is_available(self) -> bool:
        """Check if at least one provider is configured"""
        # Check if any API keys are available
        api_keys = [
            'openai_api_key', 'anthropic_api_key', 'cohere_api_key',
            'replicate_api_token', 'huggingface_api_key'
        ]
        return any(key in self.config_dict for key in api_keys)
    
    def get_provider_coverage(self) -> Dict[str, Any]:
        """Get information about provider coverage"""
        return {
            "total_providers": len(self.SUPPORTED_PROVIDERS),
            "total_models": sum(len(models) for models in self.SUPPORTED_PROVIDERS.values()),
            "providers": list(self.SUPPORTED_PROVIDERS.keys()),
            "categories": {
                "cloud_major": ["openai", "anthropic", "cohere"],
                "cloud_specialized": ["replicate", "together_ai", "anyscale", "groq"],
                "enterprise": ["bedrock", "vertex_ai", "azure"],
                "open_source": ["huggingface", "ollama", "petals"],
                "niche": ["ai21", "nlp_cloud", "aleph_alpha"]
            }
        }


class LiteLLMModel(BaseModel):
    """LiteLLM model implementation supporting 100+ models"""
    
    def __init__(self, model_id: str, config: ModelConfig, provider_config: Dict[str, Any]):
        super().__init__(model_id, config)
        self.provider_config = provider_config
        self.config.model_type = ModelType.CHAT  # Most models are chat models
    
    async def generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """Generate response using LiteLLM"""
        try:
            # Convert messages to LiteLLM format
            litellm_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Prepare parameters
            params = {
                "model": self.model_id,
                "messages": litellm_messages,
                "temperature": kwargs.get('temperature', self.config.temperature),
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "stream": False
            }
            
            # Add any provider-specific parameters
            if 'custom_llm_provider' in kwargs:
                params['custom_llm_provider'] = kwargs['custom_llm_provider']
            
            # Make async call to LiteLLM
            response = await litellm.acompletion(**params)
            
            return ModelResponse(
                content=response.choices[0].message.content,
                model=self.model_id,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else {},
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            logger.error(f"LiteLLM generation failed for {self.model_id}: {e}")
            raise
    
    async def stream(self, messages: List[Message], **kwargs) -> AsyncGenerator[StreamingChunk, None]:
        """Stream response using LiteLLM"""
        try:
            # Convert messages to LiteLLM format
            litellm_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Prepare streaming parameters
            params = {
                "model": self.model_id,
                "messages": litellm_messages,
                "temperature": kwargs.get('temperature', self.config.temperature),
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "stream": True
            }
            
            # Make streaming call
            response = await litellm.acompletion(**params)
            
            async for chunk in response:
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
            logger.error(f"LiteLLM streaming failed for {self.model_id}: {e}")
            raise
    
    def _extract_provider_from_model_id(self) -> str:
        """Extract provider name from model ID"""
        if "/" in self.model_id:
            return self.model_id.split("/")[0]
        
        # Fallback inference
        if "gpt" in self.model_id:
            return "openai"
        elif "claude" in self.model_id:
            return "anthropic"
        elif "command" in self.model_id:
            return "cohere"
        elif "llama" in self.model_id.lower():
            return "meta"
        else:
            return "unknown"
    
    def get_context_length(self) -> int:
        """Estimate context length based on model"""
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
        
        return 4096  # Conservative default
    
    def supports_streaming(self) -> bool:
        """Most models support streaming through LiteLLM"""
        non_streaming = ["o1-preview", "o1-mini"]  # Known exceptions
        return not any(model in self.model_id for model in non_streaming)
    
    def supports_function_calling(self) -> bool:
        """Check if model supports function calling"""
        function_calling_models = [
            "gpt-4", "gpt-3.5-turbo", "claude-3", "command-r"
        ]
        return any(model in self.model_id for model in function_calling_models)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            "model_id": self.model_id,
            "provider": self._extract_provider_from_model_id(),
            "type": self.config.model_type.value,
            "context_length": self.get_context_length(),
            "supports_streaming": self.supports_streaming(),
            "supports_function_calling": self.supports_function_calling(),
            "litellm_compatible": True,
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
        """Generate response with tool/function calling support"""
        if not self.supports_function_calling():
            raise ValueError(f"Model {self.model_id} does not support function calling")
        
        try:
            # Convert messages to LiteLLM format
            litellm_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Convert tools to OpenAI function format (LiteLLM standard)
            functions = []
            for tool in tools:
                functions.append({
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool.get("input_schema", {})
                })
            
            # Prepare parameters with functions
            params = {
                "model": self.model_id,
                "messages": litellm_messages,
                "functions": functions,
                "function_call": "auto",
                "temperature": kwargs.get('temperature', self.config.temperature),
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "top_p": kwargs.get('top_p', self.config.top_p)
            }
            
            response = await litellm.acompletion(**params)
            
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
            
            return ModelResponse(
                content=choice.message.content or "",
                model=self.model_id,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else {},
                finish_reason=choice.finish_reason,
                tool_calls=tool_calls if tool_calls else None
            )
            
        except Exception as e:
            logger.error(f"LiteLLM tool generation failed for {self.model_id}: {e}")
            raise
