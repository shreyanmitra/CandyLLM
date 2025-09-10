"""
Anthropic provider with Claude model support
"""

import asyncio
import anthropic
from typing import Dict, List, Any, Optional, AsyncGenerator
import logging

from ..core.base import BaseProvider, BaseModel, ModelConfig, ModelResponse, StreamingChunk, Message, ProviderType, ModelType

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Anthropic provider supporting all Claude models"""
    
    SUPPORTED_MODELS = [
        # Claude 3 Family
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229", 
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022",
        
        # Simplified aliases
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku", 
        "claude-3-5-sonnet",
        
        # Earlier models
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2"
    ]
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key') or config.get('anthropic_api_key')
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        # Initialize Anthropic client
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
    
    def _get_provider_type(self) -> ProviderType:
        return ProviderType.CLOUD
    
    def list_models(self) -> List[str]:
        return self.SUPPORTED_MODELS.copy()
    
    def create_model(self, model_id: str, config: ModelConfig) -> 'AnthropicModel':
        if model_id not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model_id} not in supported list, but attempting anyway")
        
        return AnthropicModel(
            client=self.client,
            model_id=model_id,
            config=config
        )
    
    def is_available(self) -> bool:
        return bool(self.api_key)


class AnthropicModel(BaseModel):
    """Anthropic Claude model implementation"""
    
    def __init__(self, client: anthropic.AsyncAnthropic, model_id: str, config: ModelConfig):
        super().__init__(model_id, config)
        self.client = client
        self.config.model_type = ModelType.CHAT  # All Claude models are chat models
    
    async def generate(self, messages: List[Message], **kwargs) -> ModelResponse:
        """Generate response from Claude model"""
        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            system_message = None
            
            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    anthropic_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # Prepare parameters
            params = {
                "model": self._normalize_model_id(),
                "messages": anthropic_messages,
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens or 4096),
                "temperature": kwargs.get('temperature', self.config.temperature),
                "top_p": kwargs.get('top_p', self.config.top_p)
            }
            
            if system_message:
                params["system"] = system_message
            
            response = await self.client.messages.create(**params)
            
            return ModelResponse(
                content=response.content[0].text,
                model=self.model_id,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                finish_reason=response.stop_reason
            )
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    async def stream(self, messages: List[Message], **kwargs) -> AsyncGenerator[StreamingChunk, None]:
        """Stream response from Claude model"""
        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            system_message = None
            
            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    anthropic_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # Prepare streaming parameters
            params = {
                "model": self._normalize_model_id(),
                "messages": anthropic_messages,
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens or 4096),
                "temperature": kwargs.get('temperature', self.config.temperature),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "stream": True
            }
            
            if system_message:
                params["system"] = system_message
            
            async with self.client.messages.stream(**params) as stream:
                async for chunk in stream:
                    if chunk.type == "content_block_delta":
                        yield StreamingChunk(
                            content=chunk.delta.text,
                            model=self.model_id,
                            chunk_id=getattr(chunk, 'id', None),
                            finish_reason=None
                        )
                    elif chunk.type == "message_stop":
                        yield StreamingChunk(
                            content="",
                            model=self.model_id,
                            chunk_id=getattr(chunk, 'id', None),
                            finish_reason="stop"
                        )
            
        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise
    
    def _normalize_model_id(self) -> str:
        """Normalize model ID to full Anthropic format"""
        model_mapping = {
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3-5-sonnet": "claude-3-5-sonnet-20241022"
        }
        
        return model_mapping.get(self.model_id, self.model_id)
    
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
        """Get comprehensive model information"""
        return {
            "model_id": self.model_id,
            "provider": "anthropic",
            "type": self.config.model_type.value,
            "context_length": self.get_context_length(),
            "supports_streaming": self.supports_streaming(),
            "supports_function_calling": self.supports_function_calling(),
            "supports_vision": self.supports_vision(),
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
        """Generate response with tool use support"""
        if not self.supports_function_calling():
            raise ValueError(f"Model {self.model_id} does not support function calling")
        
        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            system_message = None
            
            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    anthropic_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # Convert tools to Anthropic format
            anthropic_tools = []
            for tool in tools:
                anthropic_tools.append({
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool.get("input_schema", {})
                })
            
            # Prepare parameters with tools
            params = {
                "model": self._normalize_model_id(),
                "messages": anthropic_messages,
                "tools": anthropic_tools,
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens or 4096),
                "temperature": kwargs.get('temperature', self.config.temperature),
                "top_p": kwargs.get('top_p', self.config.top_p)
            }
            
            if system_message:
                params["system"] = system_message
            
            response = await self.client.messages.create(**params)
            
            # Handle tool use in response
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
            
            return ModelResponse(
                content=content,
                model=self.model_id,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                finish_reason=response.stop_reason,
                tool_calls=tool_calls if tool_calls else None
            )
            
        except Exception as e:
            logger.error(f"Anthropic tool generation failed: {e}")
            raise
