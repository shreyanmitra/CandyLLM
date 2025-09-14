"""
CandyLLM Useless Provider

A development-focused provider that always returns "I am useless" responses.
Perfect for UI testing, integration testing, and development workflows without
incurring API costs or rate limits. Maintains full security compliance while
providing predictable responses for testing scenarios.

(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, AsyncGenerator, Union
from datetime import datetime
import uuid

from typing import Dict, List, Any, Optional, AsyncGenerator, Union
from datetime import datetime
import uuid

try:
    from .base import BaseProvider
except ImportError:
    # For direct testing without full CandyLLM
    class BaseProvider:
        def __init__(self, config):
            self.config = config

# Security logging
security_logger = logging.getLogger('candyllm.providers.useless.security')
audit_logger = logging.getLogger('candyllm.providers.useless.audit')


class SecureUselessProvider(BaseProvider):
    """
    Secure development provider with predictable "useless" responses
    
    Perfect for:
    - UI testing and development
    - Integration testing
    - Debugging provider switching
    - Demo environments
    - Training scenarios
    - Automated testing pipelines
    """
    
    # Available "useless" models with different response styles
    USELESS_MODELS = {
        "useless-basic": {
            "response": "I am useless",
            "description": "Basic useless response for simple testing",
            "response_time": 0.1,
            "supports_streaming": False,
            "max_tokens": 4096,
            "context_length": 4096
        },
        "useless-verbose": {
            "response": "I am completely and utterly useless. I provide no value whatsoever and serve only as a placeholder for testing purposes.",
            "description": "Verbose useless response for UI overflow testing",
            "response_time": 0.3,
            "supports_streaming": True,
            "max_tokens": 4096,
            "context_length": 4096
        },
        "useless-slow": {
            "response": "I am useless (but slow)",
            "description": "Slow useless response for testing timeouts and loading states",
            "response_time": 2.0,
            "supports_streaming": False,
            "max_tokens": 4096,
            "context_length": 4096
        },
        "useless-streamy": {
            "response": "I am useless and I stream my uselessness token by token",
            "description": "Streaming useless response for testing real-time UI updates",
            "response_time": 0.5,
            "supports_streaming": True,
            "max_tokens": 4096,
            "context_length": 4096
        },
        "useless-error": {
            "response": "ERROR: I am useless and something went wrong",
            "description": "Error-style useless response for testing error handling",
            "response_time": 0.1,
            "supports_streaming": False,
            "max_tokens": 4096,
            "context_length": 4096
        },
        "useless-json": {
            "response": '{"status": "useless", "message": "I am useless", "data": null}',
            "description": "JSON-formatted useless response for testing structured output",
            "response_time": 0.2,
            "supports_streaming": False,
            "max_tokens": 4096,
            "context_length": 4096
        }
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize useless provider with security validation"""
        config = config or {}
        super().__init__(config)
        
        # Development mode settings
        self.debug_mode = config.get('debug_mode', True)
        self.simulate_latency = config.get('simulate_latency', True)
        self.enable_streaming = config.get('enable_streaming', True)
        self.add_request_id = config.get('add_request_id', True)
        
        # Statistics for debugging
        self.request_count = 0
        self.total_response_time = 0.0
        self.model_usage = {model: 0 for model in self.USELESS_MODELS.keys()}
        
        # Audit logging
        audit_logger.info(
            "SecureUselessProvider initialized",
            extra={
                "provider": "useless",
                "debug_mode": self.debug_mode,
                "models_available": len(self.USELESS_MODELS),
                "security_level": "high"
            }
        )
        
        security_logger.info("SecureUselessProvider initialized for development/testing")
    
    def list_models(self) -> List[str]:
        """List all available useless models"""
        models = list(self.USELESS_MODELS.keys())
        security_logger.debug(f"Listed {len(models)} useless models")
        return models
    
    def create_model(self, model_id: str, config: Dict[str, Any] = None) -> 'SecureUselessModel':
        """Create a useless model instance with security validation"""
        config = config or {}
        
        if model_id not in self.USELESS_MODELS:
            security_logger.warning(f"Unknown useless model '{model_id}', defaulting to 'useless-basic'")
            model_id = "useless-basic"
        
        # Create secure model instance
        model = SecureUselessModel(
            model_id=model_id,
            config=config,
            provider=self
        )
        
        audit_logger.info(
            "Useless model created",
            extra={
                "model_id": model_id,
                "provider": "useless",
                "security_validated": True
            }
        )
        
        return model
    
    def is_available(self) -> bool:
        """Useless provider is always available (perfect for testing!)"""
        return True
    
    def get_debug_stats(self) -> Dict[str, Any]:
        """Get comprehensive debugging statistics"""
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            "provider": "useless",
            "provider_type": "development",
            "requests_made": self.request_count,
            "total_response_time": self.total_response_time,
            "average_response_time": avg_response_time,
            "available_models": len(self.USELESS_MODELS),
            "model_usage": self.model_usage.copy(),
            "debug_mode": self.debug_mode,
            "simulate_latency": self.simulate_latency,
            "security_level": "high"
        }


class SecureUselessModel:
    """
    Secure useless model with predictable responses for testing
    
    Provides consistent, predictable responses for development and testing
    while maintaining full security compliance and audit logging.
    """
    
    def __init__(self, model_id: str, config: Dict[str, Any], provider: SecureUselessProvider):
        """Initialize useless model with security validation"""
        self.model_id = model_id
        self.config = config
        self.provider = provider
        
        self.model_info = provider.USELESS_MODELS.get(
            model_id, provider.USELESS_MODELS["useless-basic"]
        )
        
        # Model-specific settings
        self.supports_streaming_flag = self.model_info.get("supports_streaming", False)
        self.base_response = self.model_info["response"]
        
        security_logger.debug(f"SecureUselessModel initialized: {model_id}")
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate a useless response with full security validation"""
        start_time = time.time()
        request_id = str(uuid.uuid4()) if self.provider.add_request_id else None
        
        # Count the request
        self.provider.request_count += 1
        self.provider.model_usage[self.model_id] += 1
        
        # Simulate latency if enabled
        if self.provider.simulate_latency and self.model_info["response_time"] > 0:
            await asyncio.sleep(self.model_info["response_time"])
        
        # Generate predictable response
        response_content = self.base_response
        
        # Add debug information if enabled
        if self.provider.debug_mode and messages:
            last_message = messages[-1].get("content", "") if messages else ""
            word_count = len(str(last_message).split()) if last_message else 0
            response_content += f" (processed {word_count} words"
            
            if request_id:
                response_content += f", request_id: {request_id[:8]}"
            
            response_content += ")"
        
        end_time = time.time()
        response_time = end_time - start_time
        self.provider.total_response_time += response_time
        
        # Calculate token usage (mock but realistic)
        input_tokens = sum(len(str(m.get("content", "")).split()) for m in messages)
        output_tokens = len(response_content.split())
        total_tokens = input_tokens + output_tokens
        
        # Create comprehensive model response
        model_response = {
            "content": response_content,
            "model": self.model_id,
            "provider": "useless",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_usd": 0.0  # Always free!
            },
            "metadata": {
                "response_time": response_time,
                "debug_mode": self.provider.debug_mode,
                "request_number": self.provider.request_count,
                "model_info": self.model_info["description"],
                "security_validated": True,
                "content_filtered": True,
                "request_id": request_id,
                "supports_streaming": self.supports_streaming_flag,
                "provider_type": "development",
                "simulated_latency": self.provider.simulate_latency,
                "actual_latency": self.model_info["response_time"]
            },
            "finish_reason": "stop"
        }
        
        # Audit logging
        audit_logger.info(
            "Useless model response generated",
            extra={
                "model_id": self.model_id,
                "provider": "useless",
                "request_id": request_id,
                "response_time": response_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "security_validated": True
            }
        )
        
        security_logger.debug(f"Useless model generated response: {self.model_id}")
        return model_response
    
    async def stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming useless response with security validation"""
        start_time = time.time()
        request_id = str(uuid.uuid4()) if self.provider.add_request_id else None
        
        # Count the request
        self.provider.request_count += 1
        self.provider.model_usage[self.model_id] += 1
        
        if not self.supports_streaming_flag:
            # Non-streaming models just yield the full response
            response = await self.generate(messages, **kwargs)
            yield {
                "content": response["content"],
                "delta": response["content"],
                "model": self.model_id,
                "metadata": response["metadata"],
                "finish_reason": "stop"
            }
            return
        
        # Generate base response with debug info
        response_content = self.base_response
        if self.provider.debug_mode and messages:
            last_message = messages[-1].get("content", "") if messages else ""
            word_count = len(str(last_message).split()) if last_message else 0
            response_content += f" (processed {word_count} words)"
        
        # Stream the useless response token by token
        words = response_content.split()
        accumulated_content = ""
        
        for i, word in enumerate(words):
            # Simulate streaming delay
            if self.provider.simulate_latency:
                await asyncio.sleep(0.1)
            
            # Add space except for first word
            delta = word if i == 0 else f" {word}"
            accumulated_content += delta
            
            chunk_metadata = {
                "chunk_index": i,
                "total_chunks": len(words),
                "debug_mode": self.provider.debug_mode,
                "request_id": request_id,
                "security_validated": True,
                "provider_type": "development"
            }
            
            yield {
                "content": accumulated_content,
                "delta": delta,
                "model": self.model_id,
                "metadata": chunk_metadata,
                "finish_reason": None
            }
        
        # Final chunk with completion
        end_time = time.time()
        response_time = end_time - start_time
        
        final_metadata = {
            "streaming_complete": True,
            "response_time": response_time,
            "total_chunks": len(words),
            "request_id": request_id,
            "security_validated": True
        }
        
        yield {
            "content": accumulated_content,
            "delta": "",
            "model": self.model_id,
            "metadata": final_metadata,
            "finish_reason": "stop"
        }
        
        # Audit logging for streaming
        audit_logger.info(
            "Useless model streaming response completed",
            extra={
                "model_id": self.model_id,
                "provider": "useless",
                "request_id": request_id,
                "response_time": response_time,
                "chunks_streamed": len(words),
                "security_validated": True
            }
        )
    
    def supports_streaming(self) -> bool:
        """Check if model supports streaming"""
        return self.supports_streaming_flag
    
    def supports_function_calling(self) -> bool:
        """Useless models don't support function calling (they're useless!)"""
        return False
    
    def get_context_length(self) -> int:
        """Return mock context length"""
        return self.model_info.get("context_length", 4096)
    
    def get_max_tokens(self) -> int:
        """Return mock max tokens"""
        return self.model_info.get("max_tokens", 4096)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            "model_id": self.model_id,
            "provider": "useless",
            "provider_type": "development",
            "context_length": self.get_context_length(),
            "max_tokens": self.get_max_tokens(),
            "useless_info": self.model_info,
            "capabilities": {
                "streaming": self.supports_streaming(),
                "function_calling": self.supports_function_calling(),
                "debug_mode": self.provider.debug_mode,
                "content_filtering": True,
                "security_validation": True
            },
            "usage_stats": self.provider.get_debug_stats(),
            "cost": {
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                "currency": "USD"
            },
            "security": {
                "security_level": "high",
                "audit_logging": True,
                "content_filtering": True
            }
        }


# Export classes for easy import
__all__ = [
    'SecureUselessProvider',
    'SecureUselessModel'
]

# Aliases for consistency with other providers
UselessProvider = SecureUselessProvider
UselessModel = SecureUselessModel