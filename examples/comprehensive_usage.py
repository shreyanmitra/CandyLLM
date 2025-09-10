"""
CandyLLM 2.0 Usage Examples
Comprehensive examples showing all features including tools with LRU caching
"""

import asyncio
from CandyLLM import CandyLLM, tool, register_tool, get_tool, list_tools

# Example 1: Basic Usage with Auto Model Selection
async def basic_usage():
    """Basic chat with automatic model selection"""
    print("=== Basic Usage Example ===")
    
    # Initialize CandyLLM
    candy = CandyLLM({
        'openai_api_key': 'your-openai-key',
        'anthropic_api_key': 'your-anthropic-key'
    })
    
    # Auto-select optimal model for the task
    candy.auto_model(task="chat", quality="high", speed="fast", cost="low")
    
    # Simple chat
    response = await candy.chat("What is quantum computing?")
    print(f"Response: {response.content}")
    print(f"Model used: {response.model}")
    print(f"Tokens: {response.usage}")


# Example 2: Manual Model Selection from 100+ Models
async def model_selection_examples():
    """Examples of selecting specific models"""
    print("\n=== Model Selection Examples ===")
    
    candy = CandyLLM()
    
    # List all available models (100+ models)
    all_models = candy.list_models()
    print(f"Total models available: {len(all_models)}")
    print(f"First 10 models: {all_models[:10]}")
    
    # List models by provider
    openai_models = candy.list_models("openai")
    anthropic_models = candy.list_models("anthropic") 
    print(f"OpenAI models: {len(openai_models)}")
    print(f"Anthropic models: {len(anthropic_models)}")
    
    # Use specific models
    models_to_test = [
        "openai:gpt-4-turbo",
        "anthropic:claude-3-sonnet", 
        "litellm:groq/llama2-70b-4096",
        "transformers:microsoft/DialoGPT-medium"
    ]
    
    for model_id in models_to_test:
        try:
            candy.set_model(model_id)
            response = await candy.chat("Hello!")
            print(f"{model_id}: {response.content[:50]}...")
        except Exception as e:
            print(f"{model_id}: Error - {e}")


# Example 3: Tool Registration and Usage with LRU Caching
def tool_examples():
    """Comprehensive tool system examples"""
    print("\n=== Tool System Examples ===")
    
    # Method 1: Function decorator registration
    @tool(name="weather", description="Get weather information", category="utility")
    def get_weather(city: str, units: str = "celsius") -> dict:
        """Get current weather for a city"""
        # Simulate weather API call
        return {
            "city": city,
            "temperature": 22,
            "units": units,
            "condition": "sunny"
        }
    
    # Method 2: Direct function registration  
    def calculate_factorial(n: int) -> int:
        """Calculate factorial of a number"""
        if n <= 1:
            return 1
        return n * calculate_factorial(n - 1)
    
    candy = CandyLLM()
    candy.register_tool(calculate_factorial, name="factorial", category="math")
    
    # Method 3: Class-based tool registration
    class DatabaseTool:
        _name = "database"
        _description = "Database operations"
        _category = "data"
        
        def get_config(self):
            from CandyLLM.core.base import ToolConfig
            return ToolConfig(
                name=self._name,
                description=self._description,
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "table": {"type": "string"}
                    },
                    "required": ["query"]
                }
            )
        
        async def execute(self, query: str, table: str = "users"):
            from CandyLLM.core.base import ToolResult
            # Simulate database query
            return ToolResult(
                success=True,
                data={"results": [{"id": 1, "name": "John"}], "count": 1},
                message=f"Query executed on {table}"
            )
    
    register_tool(DatabaseTool)
    
    # List and use tools
    all_tools = list_tools()
    print(f"Total tools registered: {len(all_tools)}")
    
    for tool_meta in all_tools:
        print(f"- {tool_meta.name}: {tool_meta.description} (category: {tool_meta.category})")
    
    # Use tools (cached with LRU)
    weather_tool = get_tool("weather")
    factorial_tool = get_tool("factorial") 
    db_tool = get_tool("database")
    
    print(f"Weather tool cached: {weather_tool is not None}")
    print(f"Factorial tool cached: {factorial_tool is not None}")
    print(f"Database tool cached: {db_tool is not None}")
    
    # Tool registry statistics (shows LRU cache performance)
    stats = candy.tools.get_tool_usage_stats()
    print(f"Tool cache stats: {stats}")


# Example 4: Streaming Responses
async def streaming_example():
    """Example of streaming responses"""
    print("\n=== Streaming Example ===")
    
    candy = CandyLLM()
    candy.auto_model(task="chat", speed="fast")
    
    print("Streaming response:")
    async for chunk in candy.stream_chat("Write a short poem about AI"):
        print(chunk.content, end="", flush=True)
    print("\n")


# Example 5: Advanced Configuration and Provider Statistics
async def advanced_configuration():
    """Advanced configuration and system statistics"""
    print("\n=== Advanced Configuration ===")
    
    # Comprehensive configuration
    config = {
        # API Keys
        'openai_api_key': 'your-openai-key',
        'anthropic_api_key': 'your-anthropic-key',
        'cohere_api_key': 'your-cohere-key',
        'groq_api_key': 'your-groq-key',
        
        # LiteLLM configuration for 100+ models
        'litellm': {
            'drop_params': True,
            'set_verbose': False
        },
        
        # Provider-specific settings
        'openai': {
            'organization': 'your-org-id',
            'timeout': 30
        },
        'anthropic': {
            'timeout': 30
        }
    }
    
    candy = CandyLLM(config)
    
    # Get comprehensive system statistics
    stats = candy.get_statistics()
    print("System Statistics:")
    print(f"- Total providers: {stats['total_providers']}")
    print(f"- Available providers: {len(stats['available_providers'])}")
    print(f"- Total models: {stats['total_models']}")
    print(f"- Tool cache hits: {stats['cache_hits']}")
    print(f"- Tool cache misses: {stats['cache_misses']}")
    
    # Provider-specific information
    providers = candy.list_providers()
    print(f"All providers: {providers}")


# Example 6: Core Compatibility
def core_compatibility():
    """Show compatibility with different interfaces"""
    print("\n=== Core Compatibility ===")
    
    # Modern CandyLLM interface
    from CandyLLM import CandyLLM
    
    # Core wrapper usage (CandyLLM with compatibility)
    llm = CandyLLM()
    llm.setConfig(
        openai_api_key='your-key',
        model_alias='gpt4',
        temperature=0.7
    )
    
    # Sync call
    try:
        response = llm.answer("Hello from interface!")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Interface needs API key: {e}")

    # New interface with core methods
    candy = CandyLLM()
    candy.setConfig(openai_api_key='your-key')  # Core method
    
    try:
        response = candy.answer("Hello from new interface with core method!")
        print(f"New interface, core method: {response}")
    except Exception as e:
        print(f"Needs API key: {e}")


# Example 7: Multi-Agent Pattern Setup (Phase 2 preview)
async def multi_agent_preview():
    """Preview of upcoming multi-agent capabilities"""
    print("\n=== Multi-Agent Preview (Phase 2) ===")
    
    # This shows the architecture we're building towards
    candy = CandyLLM()
    
    # Register specialized agents (future feature)
    print("Coming in Phase 2:")
    print("- Research Agent: claude-3-opus for deep analysis")
    print("- Coding Agent: gpt-4-turbo for programming tasks") 
    print("- Creative Agent: claude-3-5-sonnet for writing")
    print("- Speed Agent: groq/llama2-70b for quick responses")
    
    # Tool-equipped agents
    print("- Web Search Agent with real-time data access")
    print("- Code Execution Agent with sandbox environment")
    print("- Data Analysis Agent with visualization tools")


# Example 8: Performance and Caching Demonstration
async def performance_demo():
    """Demonstrate LRU caching performance benefits"""
    print("\n=== Performance & Caching Demo ===")
    
    candy = CandyLLM()
    
    # Register multiple tools to test caching
    for i in range(20):
        @tool(name=f"tool_{i}", category="test")
        def temp_tool(x: int) -> int:
            return x * i
    
    # First access (cache miss)
    import time
    start = time.time()
    for i in range(10):
        tool = get_tool(f"tool_{i}")
    first_access_time = time.time() - start
    
    # Second access (cache hit)
    start = time.time()
    for i in range(10):
        tool = get_tool(f"tool_{i}")
    second_access_time = time.time() - start
    
    print(f"First access (cache miss): {first_access_time:.4f}s")
    print(f"Second access (cache hit): {second_access_time:.4f}s")
    print(f"Speed improvement: {first_access_time/second_access_time:.2f}x faster")
    
    # Cache statistics
    cache_stats = candy.tools.get_tool_usage_stats()
    print(f"Cache efficiency: {cache_stats['cache_hits']/(cache_stats['cache_hits']+cache_stats['cache_misses'])*100:.1f}%")


# Main execution
async def main():
    """Run all examples"""
    print("🍭 CandyLLM 2.0 - Comprehensive Examples")
    print("=" * 50)
    
    # Note: Most examples need API keys to run fully
    # These will demonstrate the structure and features
    
    try:
        await basic_usage()
    except Exception as e:
        print(f"Basic usage needs API key: {e}")
    
    try:
        await model_selection_examples()
    except Exception as e:
        print(f"Model selection demo: {e}")
    
    tool_examples()  # This works without API keys
    
    try:
        await streaming_example()
    except Exception as e:
        print(f"Streaming needs API key: {e}")
    
    try:
        await advanced_configuration()
    except Exception as e:
        print(f"Advanced config demo: {e}")
    
    core_compatibility()  # Works without API keys
    
    await multi_agent_preview()  # Information only
    
    await performance_demo()  # Works without API keys


if __name__ == "__main__":
    # Run the comprehensive examples
    asyncio.run(main())


# Quick usage snippets for documentation
"""
Quick Start Examples:

# 1. Simple chat with auto model selection
from CandyLLM import quick_chat
response = quick_chat("What is the capital of France?")

# 2. List all 100+ available models  
from CandyLLM import list_all_models
models = list_all_models()
print(f"Available models: {len(models)}")

# 3. Get model information
from CandyLLM import get_model_info
info = get_model_info("openai:gpt-4-turbo")
print(info)

# 4. Register a simple tool with LRU caching
from CandyLLM import tool

@tool(name="add", category="math")
def add_numbers(a: int, b: int) -> int:
    return a + b

# 5. Use streaming
candy = CandyLLM()
async for chunk in candy.stream_chat("Tell me a story"):
    print(chunk.content, end="")
"""
