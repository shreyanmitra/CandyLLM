"""
Comprehensive Strands Agent Provider Examples

This file demonstrates the full capabilities of the Strands Agent Provider
integrated with CandyLLM, showcasing all major features from basic agent
creation to advanced multi-agent workflows.

Requirements:
- pip install strands[agents]
- AWS credentials configured for Bedrock (default)
- Optional: API keys for other providers (OpenAI, Anthropic, etc.)
"""

import asyncio
import json
from typing import Dict, Any, List
from pathlib import Path

# CandyLLM imports
from CandyLLM.agents import AgentProviderManager
from CandyLLM.agents.strands_provider import StrandsAgentProvider
from CandyLLM.core.types import Message

def setup_strands_provider() -> StrandsAgentProvider:
    """Initialize the Strands agent provider with basic configuration"""
    provider = StrandsAgentProvider()
    return provider

# Example 1: Basic Agent Creation and Chat
async def basic_agent_example():
    """Demonstrate basic agent creation and simple conversation"""
    print("=== Basic Agent Example ===")
    
    provider = setup_strands_provider()
    
    # Create a simple agent with default Bedrock configuration
    agent_config = {
        "name": "basic_assistant",
        "model_provider": "bedrock",
        "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "instructions": "You are a helpful assistant that provides clear, concise answers.",
        "max_turns": 10
    }
    
    agent = await provider.create_agent(**agent_config)
    
    # Simple conversation
    messages = [
        Message(role="user", content="What is the capital of France?"),
        Message(role="user", content="What's the population of that city?")
    ]
    
    for message in messages:
        response = await provider.run_agent(agent.agent_id, [message])
        print(f"User: {message.content}")
        print(f"Agent: {response.content}\n")
    
    await provider.cleanup_agent(agent.agent_id)

# Example 2: Agent with Custom Tools
async def agent_with_tools_example():
    """Demonstrate agent with custom Python tools"""
    print("=== Agent with Custom Tools Example ===")
    
    provider = setup_strands_provider()
    
    # Define custom tools
    tools = [
        {
            "type": "python",
            "function": {
                "name": "calculator",
                "description": "Perform basic arithmetic calculations",
                "code": """
def calculator(expression: str) -> str:
    \"\"\"Safely evaluate arithmetic expressions\"\"\"
    try:
        # Only allow basic arithmetic operations
        allowed_chars = set('0123456789+-*/().')
        if not all(c in allowed_chars or c.isspace() for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
"""
            }
        },
        {
            "type": "python",
            "function": {
                "name": "word_counter",
                "description": "Count words in a text",
                "code": """
def word_counter(text: str) -> str:
    \"\"\"Count words in the given text\"\"\"
    words = text.split()
    return f"Word count: {len(words)}"
"""
            }
        }
    ]
    
    agent_config = {
        "name": "calculator_agent",
        "model_provider": "bedrock",
        "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "instructions": "You are a helpful assistant with access to calculator and text analysis tools. Use them when appropriate.",
        "tools": tools,
        "max_turns": 15
    }
    
    agent = await provider.create_agent(**agent_config)
    
    # Test the tools
    test_messages = [
        Message(role="user", content="Calculate 25 * 4 + 10"),
        Message(role="user", content="Count the words in this sentence: 'The quick brown fox jumps over the lazy dog'"),
        Message(role="user", content="What's 15% of 200?")
    ]
    
    for message in test_messages:
        response = await provider.run_agent(agent.agent_id, [message])
        print(f"User: {message.content}")
        print(f"Agent: {response.content}\n")
    
    await provider.cleanup_agent(agent.agent_id)

# Example 3: Streaming Agent Response
async def streaming_agent_example():
    """Demonstrate streaming responses from agent"""
    print("=== Streaming Agent Example ===")
    
    provider = setup_strands_provider()
    
    agent_config = {
        "name": "streaming_writer",
        "model_provider": "bedrock",
        "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "instructions": "You are a creative writer. Write detailed, engaging content.",
        "stream": True
    }
    
    agent = await provider.create_agent(**agent_config)
    
    message = Message(role="user", content="Write a short story about a robot learning to paint")
    
    print("User: Write a short story about a robot learning to paint")
    print("Agent: ", end="", flush=True)
    
    async for chunk in provider.stream_agent(agent.agent_id, [message]):
        if hasattr(chunk, 'content') and chunk.content:
            print(chunk.content, end="", flush=True)
    
    print("\n")
    await provider.cleanup_agent(agent.agent_id)

# Example 4: Multi-Modal Agent
async def multimodal_agent_example():
    """Demonstrate agent with multi-modal capabilities"""
    print("=== Multi-Modal Agent Example ===")
    
    provider = setup_strands_provider()
    
    agent_config = {
        "name": "vision_agent",
        "model_provider": "bedrock",
        "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "instructions": "You are a vision assistant that can analyze images and describe what you see.",
        "multimodal": True
    }
    
    agent = await provider.create_agent(**agent_config)
    
    # Multi-modal message with text and image (placeholder)
    multimodal_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/..."  # Placeholder
                }
            }
        ]
    }
    
    # For this example, we'll use a text-only message
    message = Message(role="user", content="Describe what you would look for when analyzing an image of a sunset")
    
    response = await provider.run_agent(agent.agent_id, [message])
    print(f"User: {message.content}")
    print(f"Agent: {response.content}\n")
    
    await provider.cleanup_agent(agent.agent_id)

# Example 5: Multi-Agent Swarm
async def swarm_example():
    """Demonstrate multi-agent swarm pattern"""
    print("=== Multi-Agent Swarm Example ===")
    
    provider = setup_strands_provider()
    
    # Create specialized agents for different tasks
    researcher_config = {
        "name": "researcher",
        "model_provider": "bedrock",
        "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "instructions": "You are a research specialist. Gather and analyze information thoroughly.",
        "role": "researcher"
    }
    
    writer_config = {
        "name": "writer",
        "model_provider": "bedrock", 
        "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "instructions": "You are a content writer. Create engaging, well-structured content.",
        "role": "writer"
    }
    
    reviewer_config = {
        "name": "reviewer",
        "model_provider": "bedrock",
        "model_name": "anthropic.claude-3-sonnet-20240229-v1:0", 
        "instructions": "You are a quality reviewer. Provide constructive feedback and improvements.",
        "role": "reviewer"
    }
    
    # Create swarm configuration
    swarm_config = {
        "name": "content_creation_swarm",
        "agents": [researcher_config, writer_config, reviewer_config],
        "coordination_strategy": "sequential",  # researcher -> writer -> reviewer
        "shared_context": True
    }
    
    swarm = await provider.create_swarm(**swarm_config)
    
    task = "Create a comprehensive article about the benefits of renewable energy"
    
    result = await provider.run_swarm(
        swarm.swarm_id,
        task,
        max_iterations=3
    )
    
    print(f"Task: {task}")
    print(f"Swarm Result: {result.content}\n")
    
    await provider.cleanup_swarm(swarm.swarm_id)

# Example 6: Graph Workflow
async def graph_workflow_example():
    """Demonstrate graph-based workflow coordination"""
    print("=== Graph Workflow Example ===")
    
    provider = setup_strands_provider()
    
    # Define workflow nodes (agents)
    nodes = [
        {
            "id": "analyzer",
            "agent_config": {
                "name": "data_analyzer",
                "model_provider": "bedrock",
                "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
                "instructions": "Analyze data and extract key insights.",
            }
        },
        {
            "id": "visualizer",
            "agent_config": {
                "name": "chart_creator",
                "model_provider": "bedrock",
                "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
                "instructions": "Create visualization recommendations based on data analysis.",
            }
        },
        {
            "id": "reporter",
            "agent_config": {
                "name": "report_writer",
                "model_provider": "bedrock",
                "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
                "instructions": "Write comprehensive reports based on analysis and visualizations.",
            }
        }
    ]
    
    # Define workflow edges (dependencies)
    edges = [
        {"from": "analyzer", "to": "visualizer"},
        {"from": "analyzer", "to": "reporter"},
        {"from": "visualizer", "to": "reporter"}
    ]
    
    graph_config = {
        "name": "data_analysis_workflow",
        "nodes": nodes,
        "edges": edges,
        "execution_strategy": "parallel_where_possible"
    }
    
    workflow = await provider.create_graph_workflow(**graph_config)
    
    input_data = {
        "data": "Sales data: Q1: $100k, Q2: $120k, Q3: $110k, Q4: $140k",
        "requirements": "Create analysis with visualization recommendations and executive summary"
    }
    
    result = await provider.run_graph_workflow(workflow.workflow_id, input_data)
    
    print("Input Data:", input_data)
    print(f"Workflow Result: {result.content}\n")
    
    await provider.cleanup_graph_workflow(workflow.workflow_id)

# Example 7: Different Model Providers
async def multiple_providers_example():
    """Demonstrate using different model providers"""
    print("=== Multiple Model Providers Example ===")
    
    provider = setup_strands_provider()
    
    # Test different providers (skip if not available)
    provider_configs = [
        {
            "name": "bedrock_agent",
            "model_provider": "bedrock",
            "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "provider_config": {
                "region": "us-east-1"
            }
        },
        {
            "name": "openai_agent", 
            "model_provider": "openai",
            "model_name": "gpt-4",
            "provider_config": {
                "api_key": "${OPENAI_API_KEY}",
                "temperature": 0.7
            }
        },
        {
            "name": "anthropic_agent",
            "model_provider": "anthropic", 
            "model_name": "claude-3-sonnet-20240229",
            "provider_config": {
                "api_key": "${ANTHROPIC_API_KEY}",
                "max_tokens": 1000
            }
        }
    ]
    
    question = "Explain quantum computing in simple terms"
    
    for config in provider_configs:
        try:
            agent = await provider.create_agent(
                name=config["name"],
                model_provider=config["model_provider"],
                model_name=config["model_name"],
                instructions="You are a helpful technical explainer.",
                **config.get("provider_config", {})
            )
            
            message = Message(role="user", content=question)
            response = await provider.run_agent(agent.agent_id, [message])
            
            print(f"Provider: {config['model_provider']}")
            print(f"Model: {config['model_name']}")
            print(f"Response: {response.content[:200]}...\n")
            
            await provider.cleanup_agent(agent.agent_id)
            
        except Exception as e:
            print(f"Provider {config['model_provider']} not available: {e}\n")

# Example 8: Session Management
async def session_management_example():
    """Demonstrate persistent session management"""
    print("=== Session Management Example ===")
    
    provider = setup_strands_provider()
    
    # Create agent with session persistence
    agent_config = {
        "name": "persistent_agent",
        "model_provider": "bedrock",
        "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "instructions": "You are a helpful assistant with memory of our conversation.",
        "session_config": {
            "type": "file",
            "path": "./sessions",
            "persist": True
        }
    }
    
    agent = await provider.create_agent(**agent_config)
    
    # First conversation
    messages = [
        Message(role="user", content="My name is Alice. Remember this."),
        Message(role="user", content="What's my favorite color? I'll tell you: it's blue.")
    ]
    
    for message in messages:
        response = await provider.run_agent(agent.agent_id, [message])
        print(f"User: {message.content}")
        print(f"Agent: {response.content}\n")
    
    # Save session state
    session_state = await provider.get_session_state(agent.agent_id)
    print(f"Session saved with {len(session_state.get('messages', []))} messages\n")
    
    # Later conversation (session should be restored)
    follow_up = Message(role="user", content="What's my name and favorite color?")
    response = await provider.run_agent(agent.agent_id, [follow_up])
    print(f"User: {follow_up.content}")
    print(f"Agent: {response.content}\n")
    
    await provider.cleanup_agent(agent.agent_id)

# Example 9: Advanced Tool Integration (MCP)
async def mcp_tools_example():
    """Demonstrate Model Context Protocol (MCP) tool integration"""
    print("=== MCP Tools Example ===")
    
    provider = setup_strands_provider()
    
    # Configure MCP tools
    mcp_tools = [
        {
            "type": "mcp",
            "server": "filesystem",
            "config": {
                "allowed_directories": ["./examples", "./docs"]
            }
        },
        {
            "type": "mcp", 
            "server": "web_search",
            "config": {
                "api_key": "${SEARCH_API_KEY}",
                "safe_search": True
            }
        }
    ]
    
    agent_config = {
        "name": "mcp_agent",
        "model_provider": "bedrock",
        "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "instructions": "You have access to filesystem and web search tools. Use them to help users.",
        "tools": mcp_tools
    }
    
    try:
        agent = await provider.create_agent(**agent_config)
        
        message = Message(
            role="user", 
            content="List the Python files in the examples directory and search for information about Strands agents"
        )
        
        response = await provider.run_agent(agent.agent_id, [message])
        print(f"User: {message.content}")
        print(f"Agent: {response.content}\n")
        
        await provider.cleanup_agent(agent.agent_id)
        
    except Exception as e:
        print(f"MCP tools not available: {e}\n")

# Example 10: Real-World Use Case - Customer Support Bot
async def customer_support_example():
    """Real-world customer support bot with multiple capabilities"""
    print("=== Customer Support Bot Example ===")
    
    provider = setup_strands_provider()
    
    # Define support tools
    support_tools = [
        {
            "type": "python",
            "function": {
                "name": "check_order_status",
                "description": "Check the status of a customer order",
                "code": """
def check_order_status(order_id: str) -> str:
    \"\"\"Mock order status checker\"\"\"
    # In real implementation, this would query a database
    mock_orders = {
        "ORD123": "Shipped - Expected delivery: Tomorrow",
        "ORD456": "Processing - Will ship within 2 business days", 
        "ORD789": "Delivered - Delivered yesterday at 3:45 PM"
    }
    
    status = mock_orders.get(order_id, "Order not found")
    return f"Order {order_id}: {status}"
"""
            }
        },
        {
            "type": "python",
            "function": {
                "name": "schedule_callback",
                "description": "Schedule a callback for the customer",
                "code": """
def schedule_callback(phone_number: str, preferred_time: str) -> str:
    \"\"\"Mock callback scheduler\"\"\"
    return f"Callback scheduled for {phone_number} at {preferred_time}. You will receive a confirmation SMS shortly."
"""
            }
        }
    ]
    
    agent_config = {
        "name": "support_bot",
        "model_provider": "bedrock",
        "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "instructions": """You are a helpful customer support agent for an e-commerce company. 
        You can check order status, schedule callbacks, and provide general assistance.
        Always be polite, helpful, and try to resolve customer issues efficiently.
        If you cannot help with something, offer to escalate to a human agent.""",
        "tools": support_tools,
        "max_turns": 20
    }
    
    agent = await provider.create_agent(**agent_config)
    
    # Simulate customer support conversation
    support_conversation = [
        Message(role="user", content="Hi, I need help with my order ORD123"),
        Message(role="user", content="When will it arrive?"),
        Message(role="user", content="Great! Can you also schedule a callback for tomorrow at 2 PM? My number is 555-0123"),
        Message(role="user", content="Thank you for your help!")
    ]
    
    for message in support_conversation:
        response = await provider.run_agent(agent.agent_id, [message])
        print(f"Customer: {message.content}")
        print(f"Support Bot: {response.content}\n")
    
    await provider.cleanup_agent(agent.agent_id)

async def main():
    """Run all examples"""
    print("🤖 Strands Agent Provider Examples\n")
    print("=" * 50)
    
    examples = [
        basic_agent_example,
        agent_with_tools_example,
        streaming_agent_example,
        multimodal_agent_example,
        swarm_example,
        graph_workflow_example,
        multiple_providers_example,
        session_management_example,
        mcp_tools_example,
        customer_support_example
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            print(f"\n{i}/{len(examples)} - {example.__name__.replace('_', ' ').title()}")
            await example()
            print("✅ Completed successfully\n")
        except Exception as e:
            print(f"❌ Error in {example.__name__}: {e}\n")
    
    print("🎉 All examples completed!")

if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())