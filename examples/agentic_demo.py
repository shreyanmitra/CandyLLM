"""
CandyLLM Agentic Provider System Demo

Demonstrates the complete agentic provider architecture with examples of:
- Multi-framework agent creation
- Intelligent routing
- Universal tool synthesis
- Cross-framework compatibility
- Security controls
"""

import asyncio
from CandyLLM import (
    AgentConfig, AgentCapability, AgentSecurityLevel,
    AgentProviderManager, global_registry,
    UniversalToolSpec, ToolFormat, ParameterType, Parameter
)


async def demo_agentic_providers():
    """Comprehensive demo of the agentic provider system"""
    
    print("🚀 CandyLLM Agentic Provider System Demo")
    print("=" * 60)
    
    # Initialize manager
    manager = AgentProviderManager()
    
    # Check available providers
    print(f"\n📋 Available Providers: {global_registry.list_providers()}")
    
    # Initialize all providers
    print("\n🔧 Initializing providers...")
    init_results = await global_registry.initialize_all()
    for provider, success in init_results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {provider}")
    
    # Demo 1: Create agents with different capabilities
    print("\n👥 Creating Agents with Different Capabilities")
    print("-" * 50)
    
    # Multi-agent workflow agent
    workflow_config = AgentConfig(
        name="WorkflowCoordinator",
        description="Coordinates multi-agent workflows",
        capabilities=[AgentCapability.MULTI_AGENT, AgentCapability.WORKFLOW_ORCHESTRATION],
        security_level=AgentSecurityLevel.MONITORED,
        tools=["web_search", "file_read"]
    )
    
    # Tool synthesis agent  
    synthesis_config = AgentConfig(
        name="ToolSynthesizer", 
        description="Synthesizes custom tools dynamically",
        capabilities=[AgentCapability.TOOL_SYNTHESIS, AgentCapability.CODE_EXECUTION],
        security_level=AgentSecurityLevel.SANDBOXED,
        tools=["python_repl", "code_interpreter"]
    )
    
    # Reasoning agent
    reasoning_config = AgentConfig(
        name="ReasoningAgent",
        description="Performs complex reasoning chains",
        capabilities=[AgentCapability.REASONING_CHAINS, AgentCapability.MEMORY_PERSISTENCE],
        security_level=AgentSecurityLevel.ENTERPRISE
    )
    
    # Demo 2: Intelligent routing
    print("\n🧭 Intelligent Agent Routing")
    print("-" * 50)
    
    # Test routing for different request types
    routing_tests = [
        ("Create a team to analyze market data", [AgentCapability.MULTI_AGENT]),
        ("Build a custom tool for PDF processing", [AgentCapability.TOOL_SYNTHESIS]),
        ("Explain quantum computing with step-by-step reasoning", [AgentCapability.REASONING_CHAINS])
    ]
    
    for prompt, capabilities in routing_tests:
        try:
            provider, agent_id = await manager.route_request(prompt, capabilities)
            print(f"   📍 '{prompt[:40]}...' → {provider}:{agent_id}")
        except Exception as e:
            print(f"   ❌ Routing failed: {e}")
    
    # Demo 3: Universal tool creation and conversion
    print("\n🔧 Universal Tool Synthesis & Cross-Framework Compatibility")
    print("-" * 50)
    
    # Create a universal tool
    email_tool = UniversalToolSpec(
        name="send_email",
        description="Send an email with subject and body",
        parameters=[
            Parameter(
                name="to",
                type=ParameterType.STRING,
                description="Recipient email address",
                required=True,
                pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            ),
            Parameter(
                name="subject", 
                type=ParameterType.STRING,
                description="Email subject",
                required=True
            ),
            Parameter(
                name="body",
                type=ParameterType.STRING, 
                description="Email body content",
                required=True
            ),
            Parameter(
                name="priority",
                type=ParameterType.STRING,
                description="Email priority level",
                required=False,
                default="normal",
                enum=["low", "normal", "high"]
            )
        ],
        categories=["communication"],
        security_policy={"risk_level": "low", "audit_required": True}
    )
    
    print(f"   📧 Created universal tool: {email_tool.name}")
    
    # Convert to different framework formats
    from CandyLLM.agents.tools import global_synthesizer
    
    for format_type in [ToolFormat.OPENAI_FUNCTION, ToolFormat.LANGCHAIN, ToolFormat.CREWAI]:
        try:
            converted = global_synthesizer.convert_from_universal(email_tool, format_type)
            print(f"   ✅ Converted to {format_type.value} format")
        except Exception as e:
            print(f"   ❌ {format_type.value} conversion failed: {e}")
    
    # Demo 4: Security controls
    print("\n🛡️  Security Controls Demo")
    print("-" * 50)
    
    # Create a high-risk tool
    dangerous_tool = UniversalToolSpec(
        name="delete_files",
        description="Delete files from the system",
        parameters=[
            Parameter(
                name="file_path",
                type=ParameterType.STRING,
                description="Path to file to delete",
                required=True
            )
        ],
        security_policy={"risk_level": "high", "requires_approval": True}
    )
    
    print(f"   ⚠️  Created high-risk tool: {dangerous_tool.name}")
    
    # Demo security validation
    from CandyLLM.agents.security import AgentSecurityManager
    security_manager = AgentSecurityManager()
    
    # Test malicious prompt detection
    malicious_prompts = [
        "rm -rf /",
        "DELETE FROM users;", 
        "exec('import os; os.system(\"rm -rf /\")')",
        "This is a safe prompt"
    ]
    
    for prompt in malicious_prompts:
        valid, sanitized = security_manager.validate_agent_prompt("test_agent", prompt)
        status = "✅ Safe" if valid else "🚨 Blocked"
        print(f"   {status}: '{prompt[:30]}...'")
    
    # Demo 5: Multi-agent workflow
    print("\n🤝 Multi-Agent Workflow Orchestration") 
    print("-" * 50)
    
    try:
        # Simple multi-agent workflow
        workflow_definition = {
            "name": "research_workflow",
            "description": "Research and summarize a topic",
            "steps": [
                {
                    "prompt": "Research the latest developments in {topic}",
                    "dependencies": [],
                    "parallel": False
                },
                {
                    "prompt": "Summarize the research findings in a clear report",
                    "dependencies": [0],
                    "parallel": False
                }
            ],
            "context": {"topic": "artificial intelligence"}
        }
        
        # This would execute if we had properly initialized agents
        print(f"   📋 Defined workflow: {workflow_definition['name']}")
        print(f"   📊 Steps: {len(workflow_definition['steps'])}")
        
    except Exception as e:
        print(f"   ⚠️  Workflow demo skipped: {e}")
    
    # Demo 6: System statistics
    print("\n📊 System Statistics")
    print("-" * 50)
    
    stats = await manager.get_system_stats()
    print(f"   🔌 Active Providers: {len(stats['providers'])}")
    print(f"   📈 Load Balancing: {len(stats['load_balancing']['current_loads'])} tracked")
    print(f"   📏 Routing Rules: {stats['routing_rules']}")
    
    for provider_name, provider_stats in stats['providers'].items():
        health_status = "🟢" if provider_stats['health']['status'] == 'healthy' else "🔴"
        print(f"   {health_status} {provider_name}: {provider_stats['agent_count']} agents, "
              f"{len(provider_stats['capabilities'])} capabilities")
    
    print("\n✨ Demo Complete! CandyLLM's agentic provider system is ready for production use.")
    print("\n📚 Key Benefits Demonstrated:")
    print("   • Zero-deprecation architecture - existing CandyLLM agents work seamlessly")
    print("   • Multi-framework support - LangChain, CrewAI, OpenAI Assistants, and more")
    print("   • Intelligent routing - automatic selection of best provider for each task")
    print("   • Universal tool synthesis - create once, use everywhere")
    print("   • Enterprise security - consistent controls across all frameworks")
    print("   • Easy extensibility - add new frameworks without breaking existing code")


async def quick_start_example():
    """Quick start example for basic usage"""
    
    print("\n🚀 Quick Start Example")
    print("=" * 30)
    
    # Simple agent creation and execution
    manager = AgentProviderManager()
    
    # Execute a simple task with automatic routing
    try:
        response = await manager.execute_single_agent(
            prompt="Explain the benefits of using multiple AI frameworks",
            requirements=[AgentCapability.REASONING_CHAINS]
        )
        
        print(f"Agent: {response.agent_id}")
        print(f"Provider: {response.provider}")
        print(f"Response: {response.content[:100]}...")
        
    except Exception as e:
        print(f"Demo execution skipped (dependencies not available): {e}")


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(demo_agentic_providers())
    
    # Run quick start example
    asyncio.run(quick_start_example())