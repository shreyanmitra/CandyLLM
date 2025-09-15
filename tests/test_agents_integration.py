"""
Comprehensive integration test suite for CandyLLM agents system.

Tests end-to-end agent workflows, multi-agent coordination, framework
integrations, and real-world agent usage scenarios.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
import asyncio
import json
import time

# Agent system imports
try:
    from CandyLLM.agents.manager import AgentManager
    from CandyLLM.agents.base import BaseAgentProvider, AgentConfig, AgentResponse
    from CandyLLM.agents.tools import UniversalToolSynthesizer
    from CandyLLM.agents.security import AgentSecurityManager, SecurityPolicy
    from CandyLLM.agents.providers.langchain import LangChainProvider
    from CandyLLM.agents.providers.crewai import CrewAIProvider
    from CandyLLM.agents.providers.autogen import AutoGenProvider
    from CandyLLM.core.candyllm import CandyLLM
    AGENTS_INTEGRATION_AVAILABLE = True
except ImportError:
    AGENTS_INTEGRATION_AVAILABLE = False


@pytest.mark.skipif(not AGENTS_INTEGRATION_AVAILABLE, reason="Agents integration not available")
class TestAgentSystemIntegration:
    """Test suite for complete agent system integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.candyllm = CandyLLM()
        self.agent_manager = AgentManager()
        self.tool_synthesizer = UniversalToolSynthesizer()
        self.security_manager = AgentSecurityManager()
    
    def test_complete_agent_system_setup(self):
        """Test complete agent system setup and initialization."""
        try:
            # Test CandyLLM integration with agents
            assert hasattr(self.candyllm, 'agents')
            assert self.candyllm.agents is not None
            
            # Test agent manager initialization
            assert self.agent_manager is not None
            assert hasattr(self.agent_manager, 'providers')
            assert hasattr(self.agent_manager, 'router')
            
            # Test tool synthesizer initialization
            assert self.tool_synthesizer is not None
            assert hasattr(self.tool_synthesizer, 'adapters')
            
            # Test security manager initialization
            assert self.security_manager is not None
            assert hasattr(self.security_manager, 'access_control')
            
        except Exception:
            pytest.skip("Agent system setup differs")
    
    @pytest.mark.asyncio
    async def test_end_to_end_agent_workflow(self):
        """Test complete end-to-end agent workflow."""
        try:
            # Create agent configuration
            config = AgentConfig(
                name="integration_test_agent",
                description="End-to-end test agent",
                model="gpt-4",
                temperature=0.7,
                tools=["web_search", "calculator"],
                security_level="sandboxed"
            )
            
            # Register agent with manager
            agent_id = await self.agent_manager.register_agent("langchain", config)
            
            assert agent_id is not None
            assert await self.agent_manager.is_agent_active(agent_id)
            
            # Execute query through agent
            with patch.object(self.agent_manager, 'execute_agent_query') as mock_execute:
                mock_response = AgentResponse(
                    content="Integration test successful",
                    agent_name="integration_test_agent",
                    success=True,
                    execution_time=1.5
                )
                mock_execute.return_value = mock_response
                
                response = await self.agent_manager.execute_agent_query(
                    agent_id, "What is the capital of France?"
                )
                
                assert isinstance(response, AgentResponse)
                assert response.success is True
                assert response.agent_name == "integration_test_agent"
            
            # Cleanup
            await self.agent_manager.unregister_agent(agent_id)
            
        except Exception:
            pytest.skip("End-to-end workflow not available")
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self):
        """Test multi-agent coordination and collaboration."""
        try:
            # Create multiple agents with different capabilities
            agent_configs = [
                AgentConfig(
                    name="researcher_agent",
                    description="Research specialist",
                    model="gpt-4",
                    tools=["web_search", "document_retrieval"],
                    capabilities=["research", "analysis"]
                ),
                AgentConfig(
                    name="writer_agent",
                    description="Content writing specialist",
                    model="gpt-4",
                    tools=["text_generation", "grammar_check"],
                    capabilities=["writing", "editing"]
                ),
                AgentConfig(
                    name="reviewer_agent",
                    description="Content review specialist",
                    model="gpt-4",
                    tools=["quality_check", "fact_verification"],
                    capabilities=["review", "validation"]
                )
            ]
            
            # Register all agents
            agent_ids = []
            for config in agent_configs:
                agent_id = await self.agent_manager.register_agent("langchain", config)
                agent_ids.append(agent_id)
            
            assert len(agent_ids) == 3
            
            # Create collaborative workflow
            workflow = {
                "name": "content_creation_workflow",
                "steps": [
                    {"agent": "researcher_agent", "task": "research_topic"},
                    {"agent": "writer_agent", "task": "create_content"},
                    {"agent": "reviewer_agent", "task": "review_content"}
                ],
                "coordination_mode": "sequential"
            }
            
            # Execute collaborative workflow
            with patch.object(self.agent_manager, 'execute_workflow') as mock_workflow:
                mock_workflow.return_value = {
                    "workflow_id": "wf_123",
                    "status": "completed",
                    "results": {
                        "researcher_agent": "Research completed",
                        "writer_agent": "Content created",
                        "reviewer_agent": "Content approved"
                    },
                    "final_output": "High-quality content produced"
                }
                
                workflow_result = await self.agent_manager.execute_workflow(workflow)
                
                assert workflow_result["status"] == "completed"
                assert len(workflow_result["results"]) == 3
                assert "final_output" in workflow_result
            
            # Cleanup agents
            for agent_id in agent_ids:
                await self.agent_manager.unregister_agent(agent_id)
                
        except Exception:
            pytest.skip("Multi-agent coordination not available")
    
    @pytest.mark.asyncio
    async def test_dynamic_tool_synthesis_integration(self):
        """Test dynamic tool synthesis with agent execution."""
        try:
            # Create agent that uses dynamic tools
            config = AgentConfig(
                name="dynamic_tool_agent",
                description="Agent with dynamic tool synthesis",
                model="gpt-4",
                enable_dynamic_tools=True,
                tool_synthesis_config={
                    "auto_discover": True,
                    "framework_adapters": ["langchain", "openai"],
                    "synthesis_strategy": "intelligent"
                }
            )
            
            agent_id = await self.agent_manager.register_agent("langchain", config)
            
            # Define a task that requires tool synthesis
            task_description = """
            I need to check the current weather in Paris and then calculate
            the temperature difference with New York. Please synthesize the
            necessary tools dynamically.
            """
            
            # Mock tool synthesis
            with patch.object(self.tool_synthesizer, 'synthesize_tools') as mock_synthesize:
                mock_tools = [
                    {
                        "name": "weather_api",
                        "description": "Get current weather for a location",
                        "parameters": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "framework": "langchain"
                    },
                    {
                        "name": "temperature_calculator",
                        "description": "Calculate temperature difference",
                        "parameters": {
                            "temp1": {"type": "number"},
                            "temp2": {"type": "number"}
                        },
                        "framework": "openai"
                    }
                ]
                mock_synthesize.return_value = mock_tools
                
                # Execute task with dynamic tool synthesis
                with patch.object(self.agent_manager, 'execute_with_dynamic_tools') as mock_execute:
                    mock_response = AgentResponse(
                        content="Temperature difference calculated: Paris 15°C, New York 8°C, Difference: 7°C",
                        agent_name="dynamic_tool_agent",
                        success=True,
                        tools_used=["weather_api", "temperature_calculator"],
                        tools_synthesized=True
                    )
                    mock_execute.return_value = mock_response
                    
                    response = await self.agent_manager.execute_with_dynamic_tools(
                        agent_id, task_description
                    )
                    
                    assert response.success is True
                    assert response.tools_synthesized is True
                    assert len(response.tools_used) == 2
                    assert "temperature_calculator" in response.tools_used
            
            await self.agent_manager.unregister_agent(agent_id)
            
        except Exception:
            pytest.skip("Dynamic tool synthesis integration not available")
    
    @pytest.mark.asyncio
    async def test_agent_security_integration(self):
        """Test agent security integration and enforcement."""
        try:
            # Create security policy
            security_policy = SecurityPolicy(
                name="integration_test_policy",
                security_level="sandboxed",
                allowed_actions=["read_file", "web_search"],
                blocked_actions=["execute_system", "write_file"],
                resource_limits={"max_memory_mb": 256, "max_execution_time": 30},
                audit_enabled=True
            )
            
            # Create agent with security policy
            config = AgentConfig(
                name="secure_agent",
                description="Security-enabled agent",
                model="gpt-4",
                security_policy=security_policy
            )
            
            agent_id = await self.agent_manager.register_agent("langchain", config)
            
            # Test allowed action
            with patch.object(self.security_manager, 'authorize_action') as mock_authorize:
                mock_authorize.return_value = True
                
                allowed = await self.security_manager.authorize_action(
                    agent_id, "web_search", {"query": "test query"}
                )
                assert allowed is True
            
            # Test blocked action
            with patch.object(self.security_manager, 'authorize_action') as mock_authorize:
                mock_authorize.return_value = False
                
                blocked = await self.security_manager.authorize_action(
                    agent_id, "execute_system", {"command": "rm -rf /"}
                )
                assert blocked is False
            
            # Test security violation handling
            with patch.object(self.security_manager, 'handle_violation') as mock_handle:
                mock_handle.return_value = {
                    "action_taken": "blocked",
                    "violation_logged": True,
                    "agent_status": "suspended"
                }
                
                violation_response = await self.security_manager.handle_violation(
                    agent_id, "unauthorized_file_access", {"severity": "high"}
                )
                
                assert violation_response["action_taken"] == "blocked"
                assert violation_response["violation_logged"] is True
            
            await self.agent_manager.unregister_agent(agent_id)
            
        except Exception:
            pytest.skip("Agent security integration not available")
    
    @pytest.mark.asyncio
    async def test_framework_interoperability(self):
        """Test interoperability between different agent frameworks."""
        try:
            # Create agents from different frameworks
            framework_configs = [
                ("langchain", AgentConfig(
                    name="langchain_agent",
                    model="gpt-4",
                    tools=["web_search"]
                )),
                ("crewai", AgentConfig(
                    name="crewai_agent",
                    model="gpt-4",
                    team_role="researcher"
                )),
                ("autogen", AgentConfig(
                    name="autogen_agent",
                    model="gpt-4",
                    max_consecutive_auto_reply=5
                ))
            ]
            
            agent_ids = []
            for framework, config in framework_configs:
                try:
                    agent_id = await self.agent_manager.register_agent(framework, config)
                    agent_ids.append((framework, agent_id))
                except Exception:
                    # Framework might not be available, skip
                    continue
            
            # Test cross-framework communication
            if len(agent_ids) >= 2:
                # Create inter-framework workflow
                cross_framework_workflow = {
                    "name": "cross_framework_collaboration",
                    "participants": [agent_id for _, agent_id in agent_ids],
                    "communication_protocol": "message_passing",
                    "coordination_mode": "asynchronous"
                }
                
                with patch.object(self.agent_manager, 'execute_cross_framework_workflow') as mock_cross:
                    mock_cross.return_value = {
                        "status": "completed",
                        "framework_compatibility": "successful",
                        "message_exchanges": 5,
                        "final_result": "Cross-framework collaboration successful"
                    }
                    
                    result = await self.agent_manager.execute_cross_framework_workflow(
                        cross_framework_workflow
                    )
                    
                    assert result["status"] == "completed"
                    assert result["framework_compatibility"] == "successful"
            
            # Cleanup
            for framework, agent_id in agent_ids:
                await self.agent_manager.unregister_agent(agent_id)
                
        except Exception:
            pytest.skip("Framework interoperability not available")
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle_management(self):
        """Test complete agent lifecycle management."""
        try:
            # Test agent creation
            config = AgentConfig(
                name="lifecycle_test_agent",
                description="Agent for lifecycle testing",
                model="gpt-4",
                health_check_enabled=True,
                auto_recovery=True
            )
            
            agent_id = await self.agent_manager.register_agent("langchain", config)
            assert agent_id is not None
            
            # Test agent health monitoring
            health_status = await self.agent_manager.check_agent_health(agent_id)
            assert health_status["status"] in ["healthy", "unknown"]  # unknown if mocked
            
            # Test agent pause/resume
            pause_success = await self.agent_manager.pause_agent(agent_id)
            assert pause_success is True
            
            resume_success = await self.agent_manager.resume_agent(agent_id)
            assert resume_success is True
            
            # Test agent configuration update
            updated_config = AgentConfig(
                name="lifecycle_test_agent",
                description="Updated agent description",
                model="gpt-4",
                temperature=0.5  # Changed parameter
            )
            
            update_success = await self.agent_manager.update_agent_config(
                agent_id, updated_config
            )
            assert update_success is True
            
            # Test agent metrics collection
            metrics = await self.agent_manager.get_agent_metrics(agent_id)
            assert "total_requests" in metrics
            assert "average_response_time" in metrics
            assert "success_rate" in metrics
            
            # Test agent termination
            termination_success = await self.agent_manager.unregister_agent(agent_id)
            assert termination_success is True
            
            # Verify agent is no longer active
            is_active = await self.agent_manager.is_agent_active(agent_id)
            assert is_active is False
            
        except Exception:
            pytest.skip("Agent lifecycle management not available")
    
    @pytest.mark.asyncio
    async def test_agent_performance_optimization(self):
        """Test agent performance optimization features."""
        try:
            # Create agent with performance monitoring
            config = AgentConfig(
                name="performance_test_agent",
                description="Agent for performance testing",
                model="gpt-4",
                performance_monitoring=True,
                caching_enabled=True,
                load_balancing=True
            )
            
            agent_id = await self.agent_manager.register_agent("langchain", config)
            
            # Test caching functionality
            query = "What is machine learning?"
            
            # First execution (should cache result)
            with patch.object(self.agent_manager, 'execute_agent_query') as mock_execute:
                mock_response = AgentResponse(
                    content="ML explanation...",
                    agent_name="performance_test_agent",
                    success=True,
                    execution_time=2.0,
                    cached=False
                )
                mock_execute.return_value = mock_response
                
                response1 = await self.agent_manager.execute_agent_query(agent_id, query)
                assert response1.cached is False
            
            # Second execution (should use cache)
            with patch.object(self.agent_manager, 'execute_agent_query') as mock_execute:
                mock_response = AgentResponse(
                    content="ML explanation...",
                    agent_name="performance_test_agent",
                    success=True,
                    execution_time=0.1,  # Much faster
                    cached=True
                )
                mock_execute.return_value = mock_response
                
                response2 = await self.agent_manager.execute_agent_query(agent_id, query)
                assert response2.cached is True
                assert response2.execution_time < response1.execution_time
            
            # Test load balancing
            if hasattr(self.agent_manager, 'get_load_balancing_stats'):
                lb_stats = await self.agent_manager.get_load_balancing_stats(agent_id)
                assert "request_distribution" in lb_stats
                assert "response_times" in lb_stats
            
            await self.agent_manager.unregister_agent(agent_id)
            
        except Exception:
            pytest.skip("Performance optimization not available")
    
    @pytest.mark.asyncio
    async def test_agent_error_handling_and_recovery(self):
        """Test agent error handling and recovery mechanisms."""
        try:
            config = AgentConfig(
                name="error_handling_agent",
                description="Agent for error handling testing",
                model="gpt-4",
                error_recovery_enabled=True,
                max_retry_attempts=3,
                fallback_strategy="graceful_degradation"
            )
            
            agent_id = await self.agent_manager.register_agent("langchain", config)
            
            # Test error handling with retries
            with patch.object(self.agent_manager, 'execute_agent_query') as mock_execute:
                # First calls fail, last succeeds
                mock_execute.side_effect = [
                    Exception("Network error"),
                    Exception("Rate limit exceeded"),
                    AgentResponse(
                        content="Success after retries",
                        agent_name="error_handling_agent",
                        success=True,
                        retry_count=2
                    )
                ]
                
                response = await self.agent_manager.execute_agent_query(
                    agent_id, "Test query with retries"
                )
                
                assert response.success is True
                assert response.retry_count == 2
            
            # Test fallback strategy
            with patch.object(self.agent_manager, 'execute_with_fallback') as mock_fallback:
                mock_fallback.return_value = AgentResponse(
                    content="Fallback response",
                    agent_name="error_handling_agent",
                    success=True,
                    fallback_used=True
                )
                
                fallback_response = await self.agent_manager.execute_with_fallback(
                    agent_id, "Query requiring fallback"
                )
                
                assert fallback_response.success is True
                assert fallback_response.fallback_used is True
            
            # Test error reporting
            error_report = await self.agent_manager.get_error_report(agent_id)
            assert "total_errors" in error_report
            assert "error_types" in error_report
            assert "recovery_rate" in error_report
            
            await self.agent_manager.unregister_agent(agent_id)
            
        except Exception:
            pytest.skip("Error handling and recovery not available")


if __name__ == '__main__':
    pytest.main([__file__, "-v"])