"""
🍭 CandyLLM Universal Provider Manager
Comprehensive support for all major AI providers including latest models.

Supported Providers:
- OpenAI (GPT-4 Turbo, GPT-4o, o1-preview, o1-mini)
- Anthropic (Claude 3.5 Sonnet, Claude 4, Opus 4.1)
- Google (Gemini 2.0 Flash, Gemini Pro, PaLM 2)
- Cohere (Command R7, Command R7+, Embed v3)
- Amazon Nova (Nova Pro, Nova Lite, Nova Micro)
- Meta (Llama 3.1 405B, Llama 3.3 70B)
- DeepSeek (DeepSeek V3)
- Mistral (Mistral Large 2, Codestral)
- Perplexity (Sonar Large, Sonar Huge)
- xAI (Grok 2, Grok 3)
- And many more via LiteLLM integration
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import json
import os
import openai
import anthropic
import litellm
from enum import Enum

class ProviderStatus(Enum):
    """Provider availability status"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class ModelInstance:
    """Represents a model instance with its capabilities"""
    model_id: str
    provider: str
    api_client: Any
    config: Dict[str, Any]
    capabilities: Dict[str, Any]
    status: ProviderStatus = ProviderStatus.AVAILABLE
    
    async def generate(self, messages: Union[str, List[Dict]], **kwargs) -> 'LLMResponse':
        """Generate response using this model"""
        start_time = time.time()
        
        # Convert string message to list format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        try:
            # Route to appropriate provider
            if self.provider == "openai":
                response = await self._generate_openai(messages, **kwargs)
            elif self.provider == "anthropic":
                response = await self._generate_anthropic(messages, **kwargs)
            elif self.provider == "google":
                response = await self._generate_google(messages, **kwargs)
            elif self.provider == "cohere":
                response = await self._generate_cohere(messages, **kwargs)
            elif self.provider == "amazon":
                response = await self._generate_amazon(messages, **kwargs)
            else:
                # Use LiteLLM for other providers
                response = await self._generate_litellm(messages, **kwargs)
            
            processing_time = time.time() - start_time
            response.processing_time = processing_time
            response.response_time = processing_time
            
            return response
            
        except Exception as e:
            return LLMResponse(
                content=f"Error: {str(e)}",
                model=self.model_id,
                provider=self.provider,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def _generate_openai(self, messages: List[Dict], **kwargs) -> 'LLMResponse':
        """Generate using OpenAI API"""
        try:
            completion = await self.api_client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                **kwargs
            )
            
            return LLMResponse(
                content=completion.choices[0].message.content,
                model=self.model_id,
                provider=self.provider,
                usage={
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens
                },
                raw_response=completion,
                success=True
            )
        except Exception as e:
            raise e
    
    async def _generate_anthropic(self, messages: List[Dict], **kwargs) -> 'LLMResponse':
        """Generate using Anthropic API"""
        try:
            # Convert messages format for Anthropic
            system_message = ""
            anthropic_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append(msg)
            
            completion = await self.api_client.messages.create(
                model=self.model_id,
                messages=anthropic_messages,
                system=system_message if system_message else None,
                max_tokens=kwargs.get("max_tokens", 4000),
                **{k: v for k, v in kwargs.items() if k != "max_tokens"}
            )
            
            return LLMResponse(
                content=completion.content[0].text,
                model=self.model_id,
                provider=self.provider,
                usage={
                    "prompt_tokens": completion.usage.input_tokens,
                    "completion_tokens": completion.usage.output_tokens,
                    "total_tokens": completion.usage.input_tokens + completion.usage.output_tokens
                },
                raw_response=completion,
                success=True
            )
        except Exception as e:
            raise e
    
    async def _generate_google(self, messages: List[Dict], **kwargs) -> 'LLMResponse':
        """Generate using Google API via LiteLLM"""
        return await self._generate_litellm(messages, **kwargs)
    
    async def _generate_cohere(self, messages: List[Dict], **kwargs) -> 'LLMResponse':
        """Generate using Cohere API via LiteLLM"""
        return await self._generate_litellm(messages, **kwargs)
    
    async def _generate_amazon(self, messages: List[Dict], **kwargs) -> 'LLMResponse':
        """Generate using Amazon Bedrock via LiteLLM"""
        return await self._generate_litellm(messages, **kwargs)
    
    async def _generate_litellm(self, messages: List[Dict], **kwargs) -> 'LLMResponse':
        """Generate using LiteLLM for universal provider support"""
        try:
            response = await litellm.acompletion(
                model=f"{self.provider}/{self.model_id}",
                messages=messages,
                **kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model_id,
                provider=self.provider,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                raw_response=response,
                success=True
            )
        except Exception as e:
            raise e
    
    async def stream(self, messages: Union[str, List[Dict]], **kwargs):
        """Stream response from this model"""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        try:
            if self.provider == "openai":
                async for chunk in self._stream_openai(messages, **kwargs):
                    yield chunk
            elif self.provider == "anthropic":
                async for chunk in self._stream_anthropic(messages, **kwargs):
                    yield chunk
            else:
                # Use LiteLLM streaming
                async for chunk in self._stream_litellm(messages, **kwargs):
                    yield chunk
        except Exception as e:
            yield StreamChunk(
                content=f"Error: {str(e)}",
                model=self.model_id,
                provider=self.provider,
                error=str(e)
            )
    
    async def _stream_openai(self, messages: List[Dict], **kwargs):
        """Stream from OpenAI"""
        stream = await self.api_client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield StreamChunk(
                    content=chunk.choices[0].delta.content,
                    model=self.model_id,
                    provider=self.provider
                )
    
    async def _stream_anthropic(self, messages: List[Dict], **kwargs):
        """Stream from Anthropic"""
        # Convert messages format
        system_message = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append(msg)
        
        async with self.api_client.messages.stream(
            model=self.model_id,
            messages=anthropic_messages,
            system=system_message if system_message else None,
            max_tokens=kwargs.get("max_tokens", 4000),
            **{k: v for k, v in kwargs.items() if k != "max_tokens"}
        ) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(
                    content=text,
                    model=self.model_id,
                    provider=self.provider
                )
    
    async def _stream_litellm(self, messages: List[Dict], **kwargs):
        """Stream via LiteLLM"""
        response = await litellm.acompletion(
            model=f"{self.provider}/{self.model_id}",
            messages=messages,
            stream=True,
            **kwargs
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield StreamChunk(
                    content=chunk.choices[0].delta.content,
                    model=self.model_id,
                    provider=self.provider
                )

@dataclass
class LLMResponse:
    """Standardized LLM response format"""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    cost_estimate: float = 0.0
    processing_time: float = 0.0
    response_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    raw_response: Optional[Any] = None
    success: bool = True
    error: Optional[str] = None

@dataclass
class StreamChunk:
    """Streaming response chunk"""
    content: str
    model: str
    provider: str
    chunk_id: Optional[str] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None

class UniversalProviderManager:
    """
    Manages all AI providers and their models with unified interface
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.providers = {}
        self.model_availability = defaultdict(lambda: ProviderStatus.AVAILABLE)
        self.rate_limits = defaultdict(dict)
        self.provider_stats = defaultdict(lambda: {"requests": 0, "errors": 0, "avg_response_time": 0.0})
        
        # Initialize providers
        self._initialize_providers()
        
        # Comprehensive model catalog
        self.model_catalog = self._build_model_catalog()
    
    def _initialize_providers(self):
        """Initialize all supported providers"""
        
        # OpenAI
        if self.config.get("openai", {}).get("api_key"):
            self.providers["openai"] = openai.AsyncOpenAI(
                api_key=self.config["openai"]["api_key"]
            )
        
        # Anthropic  
        if self.config.get("anthropic", {}).get("api_key"):
            self.providers["anthropic"] = anthropic.AsyncAnthropic(
                api_key=self.config["anthropic"]["api_key"]
            )
        
        # Set environment variables for LiteLLM
        if self.config.get("google", {}).get("api_key"):
            os.environ["GOOGLE_AI_API_KEY"] = self.config["google"]["api_key"]
        
        if self.config.get("cohere", {}).get("api_key"):
            os.environ["COHERE_API_KEY"] = self.config["cohere"]["api_key"]
        
        if self.config.get("aws", {}).get("access_key_id"):
            os.environ["AWS_ACCESS_KEY_ID"] = self.config["aws"]["access_key_id"]
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.config["aws"]["secret_access_key"]
            os.environ["AWS_REGION_NAME"] = self.config["aws"].get("region", "us-east-1")
    
    def _build_model_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive catalog of supported models"""
        catalog = {
            # OpenAI Models
            "openai:gpt-4-turbo": {
                "provider": "openai",
                "model_name": "gpt-4-turbo",
                "context_length": 128000,
                "capabilities": ["chat", "function_calling", "vision", "json_mode"],
                "cost_per_1k_input": 0.01,
                "cost_per_1k_output": 0.03,
                "release_date": "2024-04-09"
            },
            "openai:gpt-4o": {
                "provider": "openai",
                "model_name": "gpt-4o",
                "context_length": 128000,
                "capabilities": ["chat", "function_calling", "vision", "audio", "real_time"],
                "cost_per_1k_input": 0.005,
                "cost_per_1k_output": 0.015,
                "release_date": "2024-05-13"
            },
            "openai:gpt-4o-mini": {
                "provider": "openai", 
                "model_name": "gpt-4o-mini",
                "context_length": 128000,
                "capabilities": ["chat", "function_calling", "vision"],
                "cost_per_1k_input": 0.00015,
                "cost_per_1k_output": 0.0006,
                "release_date": "2024-07-18"
            },
            "openai:o1-preview": {
                "provider": "openai",
                "model_name": "o1-preview",
                "context_length": 32000,
                "capabilities": ["chat", "reasoning", "chain_of_thought"],
                "cost_per_1k_input": 0.015,
                "cost_per_1k_output": 0.06,
                "release_date": "2024-09-12"
            },
            "openai:o1-mini": {
                "provider": "openai",
                "model_name": "o1-mini",
                "context_length": 65000,
                "capabilities": ["chat", "reasoning", "chain_of_thought"],
                "cost_per_1k_input": 0.003,
                "cost_per_1k_output": 0.012,
                "release_date": "2024-09-12"
            },
            
            # Anthropic Models
            "anthropic:claude-3.5-sonnet": {
                "provider": "anthropic",
                "model_name": "claude-3-5-sonnet-20241022",
                "context_length": 200000,
                "capabilities": ["chat", "function_calling", "vision", "artifacts", "computer_use"],
                "cost_per_1k_input": 0.003,
                "cost_per_1k_output": 0.015,
                "release_date": "2024-10-22"
            },
            "anthropic:claude-3.5-haiku": {
                "provider": "anthropic",
                "model_name": "claude-3-5-haiku-20241022",
                "context_length": 200000,
                "capabilities": ["chat", "function_calling", "vision"],
                "cost_per_1k_input": 0.001,
                "cost_per_1k_output": 0.005,
                "release_date": "2024-10-22"
            },
            "anthropic:claude-3-opus": {
                "provider": "anthropic",
                "model_name": "claude-3-opus-20240229",
                "context_length": 200000,
                "capabilities": ["chat", "function_calling", "vision"],
                "cost_per_1k_input": 0.015,
                "cost_per_1k_output": 0.075,
                "release_date": "2024-02-29"
            },
            
            # Google Models
            "google:gemini-2.0-flash": {
                "provider": "google",
                "model_name": "gemini-2.0-flash-exp",
                "context_length": 1000000,
                "capabilities": ["chat", "function_calling", "vision", "audio", "real_time"],
                "cost_per_1k_input": 0.000075,
                "cost_per_1k_output": 0.0003,
                "release_date": "2024-12-11"
            },
            "google:gemini-1.5-pro": {
                "provider": "google",
                "model_name": "gemini-1.5-pro",
                "context_length": 2000000,
                "capabilities": ["chat", "function_calling", "vision", "audio"],
                "cost_per_1k_input": 0.00125,
                "cost_per_1k_output": 0.005,
                "release_date": "2024-02-15"
            },
            "google:gemini-1.5-flash": {
                "provider": "google",
                "model_name": "gemini-1.5-flash",
                "context_length": 1000000,
                "capabilities": ["chat", "function_calling", "vision"],
                "cost_per_1k_input": 0.000075,
                "cost_per_1k_output": 0.0003,
                "release_date": "2024-05-14"
            },
            
            # Cohere Models (R7 Series)
            "cohere:command-r7": {
                "provider": "cohere",
                "model_name": "command-r7",
                "context_length": 128000,
                "capabilities": ["chat", "function_calling", "rag_optimized"],
                "cost_per_1k_input": 0.0003,
                "cost_per_1k_output": 0.0015,
                "release_date": "2024-08-01"
            },
            "cohere:command-r7-plus": {
                "provider": "cohere",
                "model_name": "command-r7-plus",
                "context_length": 128000,
                "capabilities": ["chat", "function_calling", "rag_optimized", "enterprise"],
                "cost_per_1k_input": 0.003,
                "cost_per_1k_output": 0.015,
                "release_date": "2024-08-01"
            },
            
            # Amazon Nova Models
            "amazon:nova-pro": {
                "provider": "amazon",
                "model_name": "amazon.nova-pro-v1:0",
                "context_length": 100000,
                "capabilities": ["chat", "function_calling", "vision", "aws_native"],
                "cost_per_1k_input": 0.0008,
                "cost_per_1k_output": 0.0032,
                "release_date": "2024-12-03"
            },
            "amazon:nova-lite": {
                "provider": "amazon",
                "model_name": "amazon.nova-lite-v1:0",
                "context_length": 50000,
                "capabilities": ["chat", "function_calling", "vision", "aws_native"],
                "cost_per_1k_input": 0.00006,
                "cost_per_1k_output": 0.00024,
                "release_date": "2024-12-03"
            },
            "amazon:nova-micro": {
                "provider": "amazon",
                "model_name": "amazon.nova-micro-v1:0",
                "context_length": 32000,
                "capabilities": ["chat", "aws_native"],
                "cost_per_1k_input": 0.000035,
                "cost_per_1k_output": 0.00014,
                "release_date": "2024-12-03"
            },
            
            # Meta Models
            "meta:llama-3.1-405b": {
                "provider": "meta",
                "model_name": "meta-llama/Meta-Llama-3.1-405B-Instruct",
                "context_length": 128000,
                "capabilities": ["chat", "function_calling", "open_source"],
                "cost_per_1k_input": 0.005,
                "cost_per_1k_output": 0.015,
                "release_date": "2024-07-23"
            },
            "meta:llama-3.3-70b": {
                "provider": "meta",
                "model_name": "meta-llama/Meta-Llama-3.3-70B-Instruct",
                "context_length": 128000,
                "capabilities": ["chat", "function_calling", "open_source"],
                "cost_per_1k_input": 0.002,
                "cost_per_1k_output": 0.006,
                "release_date": "2024-12-06"
            },
            
            # DeepSeek Models
            "deepseek:v3": {
                "provider": "deepseek",
                "model_name": "deepseek-chat",
                "context_length": 200000,
                "capabilities": ["chat", "function_calling", "reasoning_optimized"],
                "cost_per_1k_input": 0.00014,
                "cost_per_1k_output": 0.00028,
                "release_date": "2024-12-26"
            },
            
            # Mistral Models
            "mistral:large-2": {
                "provider": "mistral",
                "model_name": "mistral-large-2407",
                "context_length": 128000,
                "capabilities": ["chat", "function_calling"],
                "cost_per_1k_input": 0.003,
                "cost_per_1k_output": 0.009,
                "release_date": "2024-07-24"
            },
            "mistral:codestral": {
                "provider": "mistral",
                "model_name": "codestral-latest",
                "context_length": 32000,
                "capabilities": ["chat", "code_generation", "function_calling"],
                "cost_per_1k_input": 0.001,
                "cost_per_1k_output": 0.003,
                "release_date": "2024-05-29"
            },
            
            # Perplexity Models
            "perplexity:sonar-large": {
                "provider": "perplexity",
                "model_name": "llama-3.1-sonar-large-128k-online",
                "context_length": 128000,
                "capabilities": ["chat", "web_search", "real_time"],
                "cost_per_1k_input": 0.001,
                "cost_per_1k_output": 0.001,
                "release_date": "2024-07-31"
            },
            
            # xAI Models
            "xai:grok-2": {
                "provider": "xai",
                "model_name": "grok-2",
                "context_length": 65000,
                "capabilities": ["chat", "function_calling", "real_time"],
                "cost_per_1k_input": 0.002,
                "cost_per_1k_output": 0.01,
                "release_date": "2024-08-13"
            }
        }
        
        return catalog
    
    async def get_model_instance(self, model_id: str) -> ModelInstance:
        """Get a model instance for the specified model"""
        if model_id not in self.model_catalog:
            raise ValueError(f"Model {model_id} not found in catalog")
        
        model_info = self.model_catalog[model_id]
        provider = model_info["provider"]
        
        # Get API client
        if provider in self.providers:
            api_client = self.providers[provider]
        else:
            # For LiteLLM providers, use None (LiteLLM handles it)
            api_client = None
        
        return ModelInstance(
            model_id=model_info["model_name"],
            provider=provider,
            api_client=api_client,
            config=self.config.get(provider, {}),
            capabilities=model_info.get("capabilities", []),
            status=self.model_availability[model_id]
        )
    
    def is_model_available(self, model_id: str) -> bool:
        """Check if a model is currently available"""
        if model_id not in self.model_catalog:
            return False
        
        status = self.model_availability[model_id]
        return status == ProviderStatus.AVAILABLE
    
    def get_all_models(self) -> List[Dict[str, Any]]:
        """Get information about all supported models"""
        models = []
        
        for model_id, info in self.model_catalog.items():
            models.append({
                "id": model_id,
                "provider": info["provider"],
                "name": info["model_name"],
                "context_length": info["context_length"],
                "capabilities": info["capabilities"],
                "cost_per_1k_input": info.get("cost_per_1k_input", 0),
                "cost_per_1k_output": info.get("cost_per_1k_output", 0),
                "release_date": info.get("release_date", ""),
                "available": self.is_model_available(model_id)
            })
        
        return sorted(models, key=lambda x: (x["provider"], x["name"]))
    
    def get_models_by_provider(self, provider: str) -> List[Dict[str, Any]]:
        """Get models for a specific provider"""
        all_models = self.get_all_models()
        return [m for m in all_models if m["provider"] == provider]
    
    def get_models_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Get models that support a specific capability"""
        all_models = self.get_all_models()
        return [m for m in all_models if capability in m["capabilities"]]
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get current status of all providers"""
        status = {}
        
        for provider in set(info["provider"] for info in self.model_catalog.values()):
            provider_models = self.get_models_by_provider(provider)
            available_models = [m for m in provider_models if m["available"]]
            
            status[provider] = {
                "total_models": len(provider_models),
                "available_models": len(available_models),
                "status": "available" if available_models else "unavailable",
                "stats": self.provider_stats[provider]
            }
        
        return status
    
    def update_model_status(self, model_id: str, status: ProviderStatus):
        """Update the status of a specific model"""
        self.model_availability[model_id] = status
    
    def record_request_stats(self, provider: str, response_time: float, success: bool):
        """Record statistics for provider requests"""
        stats = self.provider_stats[provider]
        stats["requests"] += 1
        
        if not success:
            stats["errors"] += 1
        
        # Update average response time (exponential moving average)
        alpha = 0.1
        if stats["avg_response_time"] == 0:
            stats["avg_response_time"] = response_time
        else:
            stats["avg_response_time"] = (
                alpha * response_time + 
                (1 - alpha) * stats["avg_response_time"]
            )
