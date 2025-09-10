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
