# 🍭 CandyLLM

Yet another LLM wrapper, because why not? This one has advanced neurosymbolic reasoning, supports 100+ models, uses a state-of-the-art agentic framework, and can be deployed with any cloud provider, though. 👀

[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/candyllm)](https://pypi.org/project/candyllm/)

## 🌟 Why CandyLLM?

CandyLLM is a comprehensive AI platform designed for real-world applications:

- **🎯 Production-Ready**: Enterprise monitoring, event systems, security, and scalability
- **🧠 Intelligent**: Automatic model selection, reasoning optimization, and performance learning
- **🌐 Universal**: Support for 100+ models across all major providers
- **📊 Observable**: Complete event tracking, metrics, and debugging capabilities
- **🔧 Extensible**: Plugin architecture, custom tools, and flexible configuration
- **⚡ High-Performance**: Intelligent caching, load balancing, and optimization


## 📦 Installation

### Standard Installation
```bash
# Install CandyLLM 3.0
pip install candyllm

# Install with all optional dependencies
pip install "candyllm[full]"

# Install specific feature sets
pip install "candyllm[enterprise]"  # Enterprise features
pip install "candyllm[analytics]"   # Analytics and monitoring
pip install "candyllm[multimodal]"  # Image, audio, document processing
```

### Development Installation
```bash
# Clone the repository
git clone https://github.com/shreyanmitra/CandyLLM.git
cd CandyLLM

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Build documentation
cd docs && make html
```

### Docker Installation
```bash
# Pull the official image
docker pull candyllm/candyllm:latest

# Run with Docker Compose
docker-compose up -d

# Build from source
docker build -t candyllm .
```

## 🏃‍♀️ Quick Start

### 🔥 30-Second Demo

Get started with CandyLLM in seconds:

```python
from candyllm import CandyLLM

# Simple usage - auto-selects best model
candy = CandyLLM()
response = candy.answer("What is the capital of France?")
print(response)

# Intelligent auto-routing
candy = CandyLLM(intelligent_routing=True)
response = candy.answer("Explain quantum computing in simple terms")
print(f"Model used: {response.model}")
print(f"Response: {response.content}")
```

### 🧠 Neurosymbolic AI

```python
from candyllm import CandyLLM

# Enable neurosymbolic reasoning
llm = CandyLLM(
    neurosymbolic=True,
    reasoning_depth=5
)

# Complex reasoning with knowledge integration
response = llm.query(
    "Analyze the relationship between climate change and economic inflation, "
    "considering mathematical models and causal relationships.",
    reasoning_type="causal_inference"
)

print(response.answer)           # Main response
print(response.reasoning_path)   # Step-by-step symbolic logic
print(response.knowledge_graph)  # Visual reasoning structure
print(response.confidence_scores) # Reasoning confidence per step
```

### 🤖 Latest AI Models (100+ Supported)

```python
from candyllm import CandyLLM

# Latest state-of-the-art models
models = {
    "gpt5": CandyLLM(provider="openai", model="gpt-5"),
    "claude_opus": CandyLLM(provider="anthropic", model="claude-3-5-sonnet-20241022"),
    "nova_pro": CandyLLM(provider="bedrock", model="nova-pro"),
    "gemini_2": CandyLLM(provider="google", model="gemini-2.0-flash-exp"),
    "command_r7": CandyLLM(provider="cohere", model="command-r7"),
    "deepseek_v3": CandyLLM(provider="deepseek", model="deepseek-chat"),
    "llama_33": CandyLLM(provider="meta", model="llama-3.3-70b")
}

# Auto-select optimal model
auto_llm = CandyLLM(provider="auto")
response = auto_llm.answer("Complex scientific question")
print(f"Selected: {response.model_used}")
```

### 🛠️ Tool System with Auto-Registration

```python
from CandyLLM.tools import tool
from CandyLLM import LLMWrapper

# Define tools with decorator
@tool(description="Get current weather for a city")
def get_weather(city: str, country: str = "US") -> str:
    # Your weather API call here
    return f"Weather in {city}, {country}: Sunny, 75°F"

# Use tools with LLM
llm = LLMWrapper(source="OpenAI", modelName="gpt-4-turbo")
llm.register_tool("weather", get_weather)

# Tools are automatically cached with LRU
response = llm.answer("What's the weather in Paris?")
print(response)  # AI will use the weather tool automatically

# Check tool usage statistics
from CandyLLM.tools import ToolRegistry
ToolRegistry.get_stats()  # Shows cache hits, usage patterns
```

### 🎨 UI Integration

```python
from candyllm import CandyLLM
from CandyLLM import LLMWrapper

# Launch enhanced legacy UI (drop-in replacement)
def my_preprocessor(text):
    return text.upper()

def my_postprocessor(prompt, response):
    return f"Response length: {len(response)} characters"

# Your existing v1 code works with 100+ models now!
LLMWrapper.getUI(
    preprocessor_fn=my_preprocessor,
    postprocessor_fn=my_postprocessor,
    selfOutput=True,
    selfOutputLabel="Analytics",
    launch=True
)

# Or launch the new advanced UI
LLMWrapper.getAdvancedUI(launch=True)
# Features: Tabbed interface, 100+ models, streaming, tool integration
```

### ⚡ Streaming & Async Support

```python
from CandyLLM import LLMWrapper
import asyncio

llm = LLMWrapper(source="OpenAI", modelName="gpt-4-turbo")

# Stream responses
for chunk in llm.stream("Write a long story about AI"):
    print(chunk, end="", flush=True)

# Async support
async def async_example():
    response = await llm.answer_async("What is machine learning?")
    print(response)
    
    async for chunk in llm.stream_async("Explain neural networks"):
        print(chunk, end="", flush=True)

asyncio.run(async_example())
```

### 🎯 Smart Model Selection

```python
from candyllm import CandyLLM, EventType

# Create instance with event monitoring
candy = CandyLLM(neurosymbolic_reasoning=True)

# Track which models are selected
candy.on_event(EventType.MODEL_CHANGED, 
               lambda e: print(f"Switched to: {e.data['new_model']}"))

# Different tasks automatically use optimal models
math_response = candy.answer("Solve: ∫(x² + 3x)dx")  # → Math-optimized model
creative_response = candy.answer("Write a haiku about AI")  # → Creative model
code_response = candy.answer("Fix this Python code: print('hello world'")  # → Code model

print(f"Math model: {candy.last_model_used}")
```

### 📊 Event Monitoring

```python
from candyllm import CandyLLM, EventType

candy = CandyLLM()

# Monitor all AI interactions
@candy.on_event(EventType.QUERY_RECEIVED)
def log_queries(event_data):
    print(f"Query: {event_data.data['prompt']}")
    print(f"User: {event_data.user_id}")

@candy.on_event(EventType.RESPONSE_SENT)  
def track_responses(event_data):
    print(f"Response time: {event_data.data['duration']}ms")
    print(f"Model: {event_data.data['model']}")

# Use normally - events fire automatically
response = candy.answer("What's the weather like?")
```

### 🖥️ CLI Interface

```bash
# Quick commands
candyllm ui                    # Enhanced legacy UI
candyllm ui advanced           # Advanced UI 2.0
candyllm models list           # List all 100+ models
candyllm models test gpt-4     # Test specific model
candyllm tools list            # List registered tools

# Neurosymbolic operations
candyllm neuro analyze "complex reasoning task"
candyllm tools generate "task description"

# Analytics and monitoring
candyllm analytics dashboard
candyllm costs monthly-report
```

### 🌟 Advanced Features Demo

```python
import asyncio
from candyllm import CandyLLM, CandyConfig, EventType

async def advanced_demo():
    # Enterprise configuration
    config = CandyConfig(
        # Multi-provider setup
        openai_api_key="your-openai-key",
        anthropic_api_key="your-anthropic-key",
        google_api_key="your-google-key",
        
        # Advanced features
        intelligent_routing=True,
        neurosymbolic_reasoning=True,
        auto_optimize=True,
        cache_enabled=True,
        
        # Performance optimization
        optimize_for="balanced",  # "quality", "speed", "cost", "balanced"
        max_retries=3,
        timeout=30,
        
        # Fallback strategy
        fallback_models=[
            "openai:gpt-4o-mini",
            "anthropic:claude-3.5-haiku",
            "google:gemini-1.5-flash"
        ]
    )
    
    candy = CandyLLM(config=config)
    
    # Start tracked session
    session_id = candy.start_session(user_id="demo-user")
    
    # Register event handlers for monitoring
    candy.on_event(EventType.QUERY_RECEIVED, 
                   lambda e: print(f"🔍 Processing: {e.data['prompt'][:50]}..."))
    
    candy.on_event(EventType.RESPONSE_RECEIVED, 
                   lambda e: print(f"✅ Generated in {e.data['duration']:.2f}s"))
    
    # Complex reasoning task
    math_result = await candy.solve_math(
        "If f(x) = x³ + 2x² - x + 1, find f'(x) and solve f'(x) = 0"
    )
    print(f"Mathematical solution: {math_result.symbolic_result}")
    
    # Multi-path reasoning
    logic_result = await candy.reason(
        "If all birds can fly, and penguins are birds, can penguins fly?",
        strategy="multi_path"
    )
    print(f"Reasoning paths explored: {len(logic_result.reasoning_paths)}")
    
    # Model comparison
    comparison = await candy.compare_models(
        "Explain quantum entanglement",
        models=["gpt-4", "claude-3-opus", "gemini-pro"]
    )
    
    for model, result in comparison.results.items():
        print(f"{model}: {result.confidence:.2f} confidence")
    
    candy.end_session()

# Run the demo
asyncio.run(advanced_demo())
```

### 📊 Event-Driven Monitoring

Set up comprehensive monitoring and analytics:

```python
from candyllm import CandyLLM, EventType
from candyllm.events import LoggingHandler, MetricsHandler, AlertHandler

# Initialize with monitoring
candy = CandyLLM()

# File logging
log_handler = LoggingHandler("ai_interactions.log")
candy.on_event(EventType.QUERY_RECEIVED, log_handler)
candy.on_event(EventType.RESPONSE_SENT, log_handler)

# Real-time metrics
metrics_handler = MetricsHandler()
candy.on_event(EventType.RESPONSE_RECEIVED, metrics_handler)

# Alert system
alert_handler = AlertHandler(
    alert_events=[EventType.RESPONSE_ERROR, EventType.MODEL_ERROR],
    alert_callback=lambda event: print(f"🚨 ALERT: {event.data}")
)
candy.on_event(EventType.RESPONSE_ERROR, alert_handler)

# Custom analytics
def performance_tracker(event_data):
    if event_data.event_type == EventType.RESPONSE_RECEIVED:
        duration = event_data.data.get('duration', 0)
        if duration > 5.0:
            print(f"⚠️ Slow response: {duration:.2f}s")

candy.on_event(EventType.RESPONSE_RECEIVED, performance_tracker)

# Use normally - all events are automatically tracked
response = candy.answer("Explain quantum computing")

# Get comprehensive metrics
stats = metrics_handler.get_metrics()
print(f"Performance metrics: {stats}")
```

### 🤖 Strands Agents with Claude LLM

**Real-World Example: Temperature Comparison with Chain-of-Thought Reasoning**

This example demonstrates using Claude LLM through Strands Agents with sophisticated chain-of-thought reasoning to compare current temperatures between Paris and New York:

```python
import asyncio
from CandyLLM.agents import AgentProviderManager
from CandyLLM.core.types import Message

async def weather_comparison_example():
    """
    Advanced example: Use Claude LLM with Strands Agents to compare 
    temperatures between Paris and New York using chain-of-thought reasoning
    with dynamically synthesized tools
    """
    
    # Initialize Strands agent provider
    manager = AgentProviderManager()
    strands_provider = manager.get_provider("strands")
    
    # Enable dynamic tool synthesis - Strands will generate tools as needed
    tool_synthesis_config = {
        "enable_dynamic_synthesis": True,
        "tool_libraries": ["weather", "web_search", "calculations"],
        "synthesis_model": "claude-3-5-sonnet-20241022",  # Use Claude for tool generation
        "allowed_apis": [
            "openweathermap.org",
            "weatherapi.com", 
            "api.weather.gov"
        ],
        "security_level": "sandbox"  # Safe execution environment
    }
    
    # Create Claude-powered agent with dynamic tool synthesis capabilities
    agent_config = {
        "name": "weather_analyst",
        "model_provider": "anthropic",  # Using Claude LLM
        "model_name": "claude-3-5-sonnet-20241022",  # Latest Claude model
        "instructions": """You are a sophisticated weather analyst with advanced reasoning capabilities.
        
        When comparing temperatures between cities, use this chain-of-thought approach:
        
        1. **Tool Assessment**: Determine what tools you need (weather APIs, calculations, etc.)
        2. **Dynamic Tool Request**: If you need a tool that doesn't exist, request its synthesis
        3. **Data Collection**: Gather current temperature data for both cities using synthesized tools
        4. **Analysis Framework**: 
           - Extract exact temperatures in both Celsius and Fahrenheit
           - Note the time of data collection for accuracy
           - Consider any additional context (humidity, weather conditions)
        5. **Mathematical Comparison**: 
           - Calculate the precise difference in both temperature scales
           - Determine which city is warmer/cooler
           - Express the difference as both absolute values and percentages
        6. **Contextual Reasoning**:
           - Consider seasonal expectations for each location
           - Factor in typical climate patterns
           - Assess if the difference is significant or minor
        7. **Clear Communication**: 
           - Present findings in a structured, easy-to-understand format
           - Include both raw data and interpreted insights
           - Highlight the key takeaway
        
        When you need a tool that doesn't exist, simply describe what you need and the system 
        will synthesize it for you. Always show your reasoning step-by-step and be precise 
        with numerical calculations.""",
        "tool_synthesis": tool_synthesis_config,  # Enable dynamic tool creation
        "max_turns": 15,
        "provider_config": {
            "api_key": "${ANTHROPIC_API_KEY}",  # Set your API key
            "max_tokens": 2500,
            "temperature": 0.3  # Lower temperature for more focused reasoning
        }
    }
    
    # Create the agent
    agent = await strands_provider.create_agent(**agent_config)
    
    # Complex query requiring chain-of-thought reasoning and dynamic tool synthesis
    query = Message(
        role="user", 
        content="""What is the difference in current temperature between Paris, France and New York, USA today? 
        
        I need you to:
        1. Synthesize appropriate tools to get real-time weather data for both cities
        2. Use those tools to gather accurate temperature information
        3. Apply systematic chain-of-thought reasoning to compare the temperatures
        4. Explain what this difference means practically
        5. Provide context about whether this is typical for this time of year
        6. Give me both Celsius and Fahrenheit comparisons
        
        Show me your complete reasoning process step by step, including how you 
        determine what tools you need and request their synthesis."""
    )
    
    print("🌡️ Weather Comparison Analysis with Dynamic Tool Synthesis...")
    print("=" * 70)
    print(f"Query: {query.content}")
    print("=" * 70)
    
    # Get Claude's response with dynamic tool synthesis and full reasoning
    response = await strands_provider.run_agent(agent.agent_id, [query])
    
    print("🤖 Claude's Analysis with Tool Synthesis:")
    print(response.content)
    
    # Check what tools were dynamically created
    synthesized_tools = await strands_provider.get_synthesized_tools(agent.agent_id)
    if synthesized_tools:
        print("\n🔧 Dynamically Synthesized Tools:")
        for tool in synthesized_tools:
            print(f"  - {tool['name']}: {tool['description']}")
    
    # Follow-up question to demonstrate tool reuse and reasoning persistence
    follow_up = Message(
        role="user",
        content="""Now synthesize additional tools if needed to help me understand: 
        
        Based on your temperature analysis, what would be the optimal clothing 
        recommendations for outdoor activities in each city? Consider wind chill, 
        humidity effects, and UV index if available.
        
        If you need tools for clothing recommendations or comfort calculations, 
        please synthesize them."""
    )
    
    print("\n" + "=" * 70)
    print("Follow-up Query:", follow_up.content)
    print("=" * 70)
    
    follow_up_response = await strands_provider.run_agent(agent.agent_id, [follow_up])
    
    print("🤖 Claude's Enhanced Recommendations:")
    print(follow_up_response.content)
    
    # Check if additional tools were synthesized for the follow-up
    final_tools = await strands_provider.get_synthesized_tools(agent.agent_id)
    if len(final_tools) > len(synthesized_tools):
        print("\n🔧 Additional Tools Synthesized for Follow-up:")
        for tool in final_tools[len(synthesized_tools):]:
            print(f"  - {tool['name']}: {tool['description']}")
    
    # Show tool synthesis performance metrics
    synthesis_stats = await strands_provider.get_tool_synthesis_stats(agent.agent_id)
    print(f"\n📊 Tool Synthesis Performance:")
    print(f"  - Tools synthesized: {synthesis_stats['total_synthesized']}")
    print(f"  - Synthesis time: {synthesis_stats['avg_synthesis_time']:.2f}s")
    print(f"  - Success rate: {synthesis_stats['success_rate']:.1%}")
    
    # Cleanup
    await strands_provider.cleanup_agent(agent.agent_id)
    
    print("\n✅ Analysis with dynamic tool synthesis complete!")

# Run the example
if __name__ == "__main__":
    asyncio.run(weather_comparison_example())
```

**Expected Output Example:**
```
🌡️ Weather Comparison Analysis with Dynamic Tool Synthesis...
======================================================================
Query: What is the difference in current temperature between Paris, France and New York, USA today?
======================================================================

🤖 Claude's Analysis with Tool Synthesis:
I'll analyze the temperature difference between Paris and New York using systematic reasoning 
and dynamic tool synthesis.

**Step 1: Tool Assessment & Synthesis**
I need tools to get current weather data. Let me synthesize appropriate tools:

[Strands dynamically synthesizes weather API tool]
✅ Synthesized: get_weather_data(city, country, include_details=True)
✅ Synthesized: temperature_converter(temp, from_unit, to_unit) 
✅ Synthesized: calculate_temperature_difference(temp1, temp2, unit)

**Step 2: Data Collection**
Using synthesized weather tool to gather data:

[Agent calls synthesized tools]
- Paris, France: 18°C (64°F), partly cloudy, humidity 65%, wind 12 km/h
- New York, USA: 22°C (72°F), sunny, humidity 58%, wind 8 km/h
- Data timestamp: 2025-09-14T15:30:00 UTC

**Step 3: Mathematical Analysis** 
Using temperature calculation tools:
- Celsius difference: 22°C - 18°C = 4°C
- Fahrenheit difference: 72°F - 64°F = 8°F  
- Percentage difference: (4/18) × 100 = 22.2% warmer in New York
- Relative warmth factor: New York feels moderately warmer

**Step 4: Contextual Reasoning**
September weather analysis:
- Paris: 18°C is typical early fall, slightly below seasonal average
- New York: 22°C is pleasant fall weather, near seasonal norm
- 4°C difference is noticeable but not extreme
- Both cities in comfortable ranges for outdoor activities

**Step 5: Key Insights**
New York is currently 4°C (8°F) warmer than Paris - a moderate difference that 
would require different clothing choices but both cities have pleasant weather.

🔧 Dynamically Synthesized Tools:
  - get_weather_data: Real-time weather API with comprehensive data
  - temperature_converter: Convert between temperature units
  - calculate_temperature_difference: Precise temperature analysis

======================================================================
Follow-up Query: Now synthesize additional tools if needed to help me understand...
======================================================================

🤖 Claude's Enhanced Recommendations:
Let me synthesize additional tools for clothing and comfort analysis:

[Strands synthesizes comfort analysis tools]
✅ Synthesized: clothing_comfort_calculator(temp, humidity, wind, activity_level)
✅ Synthesized: uv_index_analyzer(location, time_of_day)
✅ Synthesized: wind_chill_calculator(temp, wind_speed)

**Clothing Analysis Using Synthesized Tools:**

**For Paris (18°C, partly cloudy, 65% humidity, 12 km/h wind):**
[Tool calculates comfort factors]
- Effective temperature: 16°C (feels cooler due to wind)
- Comfort recommendation: Light jacket or cardigan essential
- Suggested outfit: Long sleeves + light jacket, long pants, closed shoes
- UV consideration: Low UV due to cloud cover

**For New York (22°C, sunny, 58% humidity, 8 km/h wind):**
[Tool calculates comfort factors] 
- Effective temperature: 23°C (feels slightly warmer in sun)
- Comfort recommendation: Light layers, sun protection needed
- Suggested outfit: T-shirt or light long-sleeve, light pants/shorts, sunglasses
- UV consideration: Moderate UV, sunscreen recommended

🔧 Additional Tools Synthesized for Follow-up:
  - clothing_comfort_calculator: Factors in humidity, wind, activity level
  - uv_index_analyzer: UV exposure analysis for clothing decisions
  - wind_chill_calculator: Real-feel temperature calculation

📊 Tool Synthesis Performance:
  - Tools synthesized: 6
  - Synthesis time: 1.3s
  - Success rate: 100.0%

✅ Analysis with dynamic tool synthesis complete!
```

This enhanced example showcases:
- **🧠 Chain-of-Thought Reasoning**: Claude systematically breaks down problems
- **⚡ Dynamic Tool Synthesis**: Tools created on-demand based on requirements
- **🔧 Intelligent Tool Selection**: System determines optimal tools for each task
- **💬 Conversational Tool Building**: Follow-up questions trigger additional tool creation
- **📊 Performance Monitoring**: Real-time metrics on tool synthesis efficiency
- **🎯 Adaptive Problem Solving**: Tools evolve based on conversation needs

## 🆘 Troubleshooting & Support

### Common Issues

#### Installation Problems
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools

# Install with all dependencies
pip install "candyllm[full]"

# For Apple Silicon Macs
pip install --no-deps candyllm
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### API Key Configuration
```python
# Verify API keys are set
import os
print("OpenAI:", "✅" if os.getenv('OPENAI_API_KEY') else "❌")
print("Anthropic:", "✅" if os.getenv('ANTHROPIC_API_KEY') else "❌")

# Test API access
from candyllm import CandyLLM
candy = CandyLLM()
try:
    response = candy.answer("Hello")
    print("✅ API access working")
except Exception as e:
    print(f"❌ API access failed: {e}")
```

#### Model Access Issues
```python
# List available models
candy = CandyLLM()
try:
    # Check what models are accessible
    response = candy.answer("Test", task="QAWithoutRAG")
    print("✅ Model access working")
except Exception as e:
    print(f"❌ Model access failed: {e}")
```

### Getting Help

- **Documentation**: [GitHub Wiki](https://github.com/shreyanmitra/CandyLLM/wiki)
- **Issues**: [GitHub Issues](https://github.com/shreyanmitra/CandyLLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/shreyanmitra/CandyLLM/discussions)
- **Examples**: Check the [examples/](examples/) directory

## 📚 API Reference

### Core Classes

#### CandyLLM
```python
class CandyLLM:
    def __init__(self, config: dict = None, **kwargs):
        """Initialize CandyLLM instance"""
    
    def answer(self, prompt: str, **kwargs) -> str:
        """Generate response to prompt"""
    
    def setConfig(self, accessKey: str = None, testing: bool = True, 
                  source: str = "HuggingFace", modelName: str = "Llama8b"):
        """Set configuration"""
    
    def auto_model(self, task: str = "general", quality: str = "medium"):
        """Auto-select optimal model"""
```

#### LLMWrapper (Legacy)
```python
class LLMWrapper:
    def __init__(self, source: str, modelName: str, accessKey: str = None):
        """Legacy constructor"""
    
    def answer(self, prompt: str, task: str = "QAWithoutRAG") -> str:
        """Legacy answer method"""
    
    @staticmethod
    def getUI(**kwargs):
        """Launch web interface"""
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/shreyanmitra/CandyLLM.git
cd CandyLLM
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Shreyan Mitra** - Creator and Lead Developer
- **AIEA Lab at UC Santa Cruz** - Research Support
- **Open Source Community** - Contributions and feedback

---

**🍭 CandyLLM - Making AI as sweet and simple as candy!**
