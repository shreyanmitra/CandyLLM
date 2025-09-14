"""
Comprehensive Test Suite for Strands Agent Provider

Tests all functionality of the Strands provider integration with CandyLLM,
including agent creation, tools, multi-agent systems, and error handling.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# CandyLLM imports
from CandyLLM.agents.strands_provider import StrandsAgentProvider
from CandyLLM.agents.base import BaseAgentProvider
from CandyLLM.core.types import Message


class TestStrandsProvider:
    """Test suite for Strands agent provider"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.provider = StrandsAgentProvider()
    
    def test_provider_inheritance(self):
        """Test that StrandsProvider inherits from BaseAgentProvider"""
        assert isinstance(self.provider, BaseAgentProvider)
        assert hasattr(self.provider, 'create_agent')
        assert hasattr(self.provider, 'run_agent')
        assert hasattr(self.provider, 'cleanup_agent')
    
    def test_provider_info(self):
        """Test provider information and capabilities"""
        info = self.provider.get_provider_info()
        
        assert info['name'] == 'strands'
        assert info['version'] is not None
        assert 'capabilities' in info
        
        capabilities = info['capabilities']
        assert 'streaming' in capabilities
        assert 'multimodal' in capabilities
        assert 'tools' in capabilities
        assert 'multi_agent' in capabilities
    
    def test_health_check(self):
        """Test provider health check"""
        health = self.provider.health_check()
        
        assert isinstance(health, dict)
        assert 'status' in health
        assert 'dependencies' in health
        assert health['status'] in ['healthy', 'unhealthy', 'degraded']
    
    @patch('CandyLLM.providers.strands.StrandsProvider')
    async def test_create_agent_basic(self, mock_strands):
        """Test basic agent creation"""
        # Mock Strands agent
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent_123"
        mock_agent.name = "test_agent"
        mock_strands_instance = Mock()
        mock_strands_instance.create_agent = AsyncMock(return_value=mock_agent)
        mock_strands.return_value = mock_strands_instance
        
        # Test agent creation
        with patch.object(self.provider, '_strands_provider', mock_strands_instance):
            agent = await self.provider.create_agent(
                name="test_agent",
                model_provider="bedrock",
                model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                instructions="Test instructions"
            )
            
            assert agent.agent_id == "test_agent_123"
            assert agent.name == "test_agent"
    
    @patch('CandyLLM.providers.strands.StrandsProvider')
    async def test_create_agent_with_tools(self, mock_strands):
        """Test agent creation with custom tools"""
        mock_agent = Mock()
        mock_agent.agent_id = "tool_agent_123"
        mock_strands_instance = Mock()
        mock_strands_instance.create_agent = AsyncMock(return_value=mock_agent)
        mock_strands.return_value = mock_strands_instance
        
        tools = [
            {
                "type": "python",
                "function": {
                    "name": "calculator",
                    "description": "Basic calculator",
                    "code": "def calculator(x, y): return x + y"
                }
            }
        ]
        
        with patch.object(self.provider, '_strands_provider', mock_strands_instance):
            agent = await self.provider.create_agent(
                name="tool_agent",
                model_provider="bedrock",
                model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                instructions="Test instructions",
                tools=tools
            )
            
            assert agent.agent_id == "tool_agent_123"
    
    @patch('CandyLLM.providers.strands.StrandsProvider')
    async def test_run_agent(self, mock_strands):
        """Test running an agent with messages"""
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_response.role = "assistant"
        
        mock_strands_instance = Mock()
        mock_strands_instance.run_agent = AsyncMock(return_value=mock_response)
        mock_strands.return_value = mock_strands_instance
        
        with patch.object(self.provider, '_strands_provider', mock_strands_instance):
            messages = [Message(role="user", content="Hello")]
            response = await self.provider.run_agent("test_agent_123", messages)
            
            assert response.content == "Test response"
            assert response.role == "assistant"
    
    @patch('CandyLLM.providers.strands.StrandsProvider')
    async def test_stream_agent(self, mock_strands):
        """Test streaming agent responses"""
        mock_chunks = [
            Mock(content="Hello"),
            Mock(content=" there"),
            Mock(content="!")
        ]
        
        mock_strands_instance = Mock()
        mock_strands_instance.stream_agent = AsyncMock()
        mock_strands_instance.stream_agent.return_value = iter(mock_chunks)
        mock_strands.return_value = mock_strands_instance
        
        with patch.object(self.provider, '_strands_provider', mock_strands_instance):
            messages = [Message(role="user", content="Hello")]
            chunks = []
            
            async for chunk in self.provider.stream_agent("test_agent_123", messages):
                chunks.append(chunk)
            
            assert len(chunks) == 3
            assert chunks[0].content == "Hello"
    
    @patch('CandyLLM.providers.strands.StrandsProvider')
    async def test_create_swarm(self, mock_strands):
        """Test multi-agent swarm creation"""
        mock_swarm = Mock()
        mock_swarm.swarm_id = "swarm_123"
        mock_swarm.name = "test_swarm"
        
        mock_strands_instance = Mock()
        mock_strands_instance.create_swarm = AsyncMock(return_value=mock_swarm)
        mock_strands.return_value = mock_strands_instance
        
        agents = [
            {
                "name": "agent1",
                "model_provider": "bedrock",
                "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
                "instructions": "Agent 1 instructions"
            },
            {
                "name": "agent2", 
                "model_provider": "bedrock",
                "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
                "instructions": "Agent 2 instructions"
            }
        ]
        
        with patch.object(self.provider, '_strands_provider', mock_strands_instance):
            swarm = await self.provider.create_swarm(
                name="test_swarm",
                agents=agents,
                coordination_strategy="sequential"
            )
            
            assert swarm.swarm_id == "swarm_123"
            assert swarm.name == "test_swarm"
    
    @patch('CandyLLM.providers.strands.StrandsProvider')
    async def test_create_graph_workflow(self, mock_strands):
        """Test graph workflow creation"""
        mock_workflow = Mock()
        mock_workflow.workflow_id = "workflow_123"
        mock_workflow.name = "test_workflow"
        
        mock_strands_instance = Mock()
        mock_strands_instance.create_graph_workflow = AsyncMock(return_value=mock_workflow)
        mock_strands.return_value = mock_strands_instance
        
        nodes = [
            {
                "id": "agent1",
                "agent_config": {
                    "name": "analyzer",
                    "model_provider": "bedrock",
                    "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "instructions": "Analyze data"
                }
            }
        ]
        
        edges = []
        
        with patch.object(self.provider, '_strands_provider', mock_strands_instance):
            workflow = await self.provider.create_graph_workflow(
                name="test_workflow",
                nodes=nodes,
                edges=edges
            )
            
            assert workflow.workflow_id == "workflow_123"
            assert workflow.name == "test_workflow"
    
    @patch('CandyLLM.providers.strands.StrandsProvider')
    async def test_session_management(self, mock_strands):
        """Test session state management"""
        mock_session_state = {
            "messages": [{"role": "user", "content": "Hello"}],
            "context": {"user_id": "123"}
        }
        
        mock_strands_instance = Mock()
        mock_strands_instance.get_session_state = AsyncMock(return_value=mock_session_state)
        mock_strands_instance.save_session_state = AsyncMock()
        mock_strands.return_value = mock_strands_instance
        
        with patch.object(self.provider, '_strands_provider', mock_strands_instance):
            # Test getting session state
            state = await self.provider.get_session_state("test_agent_123")
            assert state == mock_session_state
            
            # Test saving session state
            new_state = {"messages": [{"role": "assistant", "content": "Hi"}]}
            await self.provider.save_session_state("test_agent_123", new_state)
            
            mock_strands_instance.save_session_state.assert_called_once()
    
    @patch('CandyLLM.providers.strands.StrandsProvider')
    async def test_cleanup_agent(self, mock_strands):
        """Test agent cleanup"""
        mock_strands_instance = Mock()
        mock_strands_instance.cleanup_agent = AsyncMock()
        mock_strands.return_value = mock_strands_instance
        
        with patch.object(self.provider, '_strands_provider', mock_strands_instance):
            await self.provider.cleanup_agent("test_agent_123")
            mock_strands_instance.cleanup_agent.assert_called_once_with("test_agent_123")
    
    def test_model_provider_validation(self):
        """Test model provider validation"""
        # Test valid providers
        valid_providers = [
            "bedrock", "openai", "anthropic", "litellm", 
            "ollama", "cohere", "mistral"
        ]
        
        for provider in valid_providers:
            assert self.provider._validate_model_provider(provider) is True
        
        # Test invalid provider
        assert self.provider._validate_model_provider("invalid_provider") is False
    
    def test_tool_validation(self):
        """Test tool configuration validation"""
        # Valid Python tool
        valid_tool = {
            "type": "python",
            "function": {
                "name": "test_func",
                "description": "Test function",
                "code": "def test_func(): return 'test'"
            }
        }
        
        assert self.provider._validate_tool(valid_tool) is True
        
        # Invalid tool (missing required fields)
        invalid_tool = {
            "type": "python",
            "function": {
                "name": "test_func"
                # Missing description and code
            }
        }
        
        assert self.provider._validate_tool(invalid_tool) is False
    
    @patch('CandyLLM.providers.strands.StrandsProvider')
    async def test_error_handling(self, mock_strands):
        """Test error handling in various scenarios"""
        mock_strands_instance = Mock()
        mock_strands_instance.create_agent = AsyncMock(side_effect=Exception("Test error"))
        mock_strands.return_value = mock_strands_instance
        
        with patch.object(self.provider, '_strands_provider', mock_strands_instance):
            with pytest.raises(Exception):
                await self.provider.create_agent(
                    name="error_agent",
                    model_provider="bedrock",
                    model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                    instructions="Test instructions"
                )
    
    def test_configuration_options(self):
        """Test various configuration options"""
        # Test with custom configuration
        custom_config = {
            "region": "us-west-2",
            "timeout": 30,
            "max_retries": 3
        }
        
        provider = StrandsAgentProvider(config=custom_config)
        assert provider.config == custom_config
    
    @patch('CandyLLM.providers.strands.StrandsProvider')
    async def test_multiple_model_providers(self, mock_strands):
        """Test using different model providers"""
        mock_agent = Mock()
        mock_agent.agent_id = "multi_provider_agent"
        
        mock_strands_instance = Mock()
        mock_strands_instance.create_agent = AsyncMock(return_value=mock_agent)
        mock_strands.return_value = mock_strands_instance
        
        providers_to_test = [
            ("bedrock", "anthropic.claude-3-sonnet-20240229-v1:0"),
            ("openai", "gpt-4"),
            ("anthropic", "claude-3-sonnet-20240229"),
            ("litellm", "gpt-3.5-turbo")
        ]
        
        with patch.object(self.provider, '_strands_provider', mock_strands_instance):
            for provider_name, model_name in providers_to_test:
                agent = await self.provider.create_agent(
                    name=f"{provider_name}_agent",
                    model_provider=provider_name,
                    model_name=model_name,
                    instructions="Test instructions"
                )
                
                assert agent.agent_id == "multi_provider_agent"


class TestStrandsIntegration:
    """Integration tests for Strands provider"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.provider = StrandsAgentProvider()
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle(self):
        """Test complete agent lifecycle: create -> run -> cleanup"""
        with patch('CandyLLM.providers.strands.StrandsProvider') as mock_strands:
            # Mock agent
            mock_agent = Mock()
            mock_agent.agent_id = "lifecycle_test_agent"
            mock_agent.name = "lifecycle_agent"
            
            # Mock response
            mock_response = Mock()
            mock_response.content = "Test response"
            mock_response.role = "assistant"
            
            # Setup mocks
            mock_strands_instance = Mock()
            mock_strands_instance.create_agent = AsyncMock(return_value=mock_agent)
            mock_strands_instance.run_agent = AsyncMock(return_value=mock_response)
            mock_strands_instance.cleanup_agent = AsyncMock()
            mock_strands.return_value = mock_strands_instance
            
            with patch.object(self.provider, '_strands_provider', mock_strands_instance):
                # Create agent
                agent = await self.provider.create_agent(
                    name="lifecycle_agent",
                    model_provider="bedrock",
                    model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                    instructions="Test lifecycle"
                )
                
                assert agent.agent_id == "lifecycle_test_agent"
                
                # Run agent
                messages = [Message(role="user", content="Test message")]
                response = await self.provider.run_agent(agent.agent_id, messages)
                
                assert response.content == "Test response"
                
                # Cleanup agent
                await self.provider.cleanup_agent(agent.agent_id)
                
                # Verify all methods were called
                mock_strands_instance.create_agent.assert_called_once()
                mock_strands_instance.run_agent.assert_called_once()
                mock_strands_instance.cleanup_agent.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self):
        """Test multi-agent workflow integration"""
        with patch('CandyLLM.providers.strands.StrandsProvider') as mock_strands:
            # Mock swarm
            mock_swarm = Mock()
            mock_swarm.swarm_id = "integration_swarm"
            
            # Mock swarm result
            mock_result = Mock()
            mock_result.content = "Swarm task completed"
            
            # Setup mocks
            mock_strands_instance = Mock()
            mock_strands_instance.create_swarm = AsyncMock(return_value=mock_swarm)
            mock_strands_instance.run_swarm = AsyncMock(return_value=mock_result)
            mock_strands_instance.cleanup_swarm = AsyncMock()
            mock_strands.return_value = mock_strands_instance
            
            with patch.object(self.provider, '_strands_provider', mock_strands_instance):
                # Create swarm
                agents = [
                    {
                        "name": "agent1",
                        "model_provider": "bedrock", 
                        "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
                        "instructions": "Agent 1"
                    },
                    {
                        "name": "agent2",
                        "model_provider": "bedrock",
                        "model_name": "anthropic.claude-3-sonnet-20240229-v1:0", 
                        "instructions": "Agent 2"
                    }
                ]
                
                swarm = await self.provider.create_swarm(
                    name="integration_swarm",
                    agents=agents,
                    coordination_strategy="sequential"
                )
                
                # Run swarm task
                result = await self.provider.run_swarm(
                    swarm.swarm_id,
                    "Test task",
                    max_iterations=2
                )
                
                assert result.content == "Swarm task completed"
                
                # Cleanup
                await self.provider.cleanup_swarm(swarm.swarm_id)


class TestStrandsPerformance:
    """Performance tests for Strands provider"""
    
    def setup_method(self):
        """Set up performance test fixtures"""
        self.provider = StrandsAgentProvider()
    
    @pytest.mark.asyncio
    async def test_concurrent_agents(self):
        """Test creating and running multiple agents concurrently"""
        with patch('CandyLLM.providers.strands.StrandsProvider') as mock_strands:
            # Mock multiple agents
            mock_agents = [Mock(agent_id=f"agent_{i}") for i in range(5)]
            mock_responses = [Mock(content=f"Response {i}") for i in range(5)]
            
            mock_strands_instance = Mock()
            mock_strands_instance.create_agent = AsyncMock(side_effect=mock_agents)
            mock_strands_instance.run_agent = AsyncMock(side_effect=mock_responses)
            mock_strands_instance.cleanup_agent = AsyncMock()
            mock_strands.return_value = mock_strands_instance
            
            with patch.object(self.provider, '_strands_provider', mock_strands_instance):
                # Create multiple agents concurrently
                agent_tasks = [
                    self.provider.create_agent(
                        name=f"concurrent_agent_{i}",
                        model_provider="bedrock",
                        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                        instructions=f"Agent {i}"
                    )
                    for i in range(5)
                ]
                
                agents = await asyncio.gather(*agent_tasks)
                assert len(agents) == 5
                
                # Run agents concurrently
                run_tasks = [
                    self.provider.run_agent(
                        agent.agent_id,
                        [Message(role="user", content=f"Message {i}")]
                    )
                    for i, agent in enumerate(agents)
                ]
                
                responses = await asyncio.gather(*run_tasks)
                assert len(responses) == 5
                
                # Cleanup concurrently
                cleanup_tasks = [
                    self.provider.cleanup_agent(agent.agent_id)
                    for agent in agents
                ]
                
                await asyncio.gather(*cleanup_tasks)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])