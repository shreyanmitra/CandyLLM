"""
Replicate Agent Provider

Integrates Replicate's cloud platform for running ML models with agent capabilities,
providing custom deployments, scaling, and access to thousands of open-source models.
"""

import uuid
import asyncio
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass, field

from .base import (
    BaseAgentProvider, 
    AgentConfig, 
    AgentResponse, 
    ToolSpec, 
    AgentCapability,
    AgentSecurityLevel
)
from .security import AgentSecurityManager

try:
    import replicate
    from replicate import Client, AsyncClient
    from replicate.exceptions import ReplicateError
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    # Mock classes for when Replicate is not available
    replicate = None
    Client = None
    AsyncClient = None
    ReplicateError = None


@dataclass
class ReplicateConfig:
    """Configuration for Replicate agent"""
    api_token: str = ""
    default_model: str = "meta/llama-2-70b-chat"
    timeout: float = 300.0  # 5 minutes default for model runs
    webhook_url: Optional[str] = None
    enable_streaming: bool = True
    enable_deployments: bool = True
    enable_model_scaling: bool = True
    max_concurrent_runs: int = 10
    poll_interval: float = 1.0  # Polling interval for non-streaming runs
    base_url: str = "https://api.replicate.com"


@dataclass
class ReplicateModel:
    """Represents a Replicate model"""
    owner: str
    name: str
    version: Optional[str] = None
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    

@dataclass
class ReplicatePrediction:
    """Represents a Replicate prediction/run"""
    id: str
    model: str
    status: str
    input: Dict[str, Any]
    output: Optional[Any] = None
    error: Optional[str] = None
    logs: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ReplicateAgent:
    """Replicate agent with model deployment and scaling capabilities"""
    
    def __init__(self, agent_id: str, config: ReplicateConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._client = None
        self._async_client = None
        self._models = {}
        self._deployments = {}
        self._predictions = {}
        self._usage_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'total_processing_time': 0.0,
            'models_used': set(),
            'deployments_created': 0
        }
        self._interaction_history = []
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the Replicate agent"""
        if not REPLICATE_AVAILABLE:
            return False
        
        try:
            if not self.config.api_token:
                return False
            
            # Initialize clients
            self._client = Client(api_token=self.config.api_token)
            self._async_client = AsyncClient(api_token=self.config.api_token)
            
            # Test connection
            await self._test_connection()
            
            return True
            
        except Exception as e:
            return False
    
    async def _test_connection(self):
        """Test API connection"""
        try:
            # List models to test connection
            models = await self._async_client.models.list()
            return True
        except Exception as e:
            raise RuntimeError(f"Replicate API test failed: {e}")
    
    async def run_prediction(self, model: str, input_data: Dict[str, Any], 
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a prediction on a Replicate model"""
        try:
            context = context or {}
            start_time = datetime.now()
            
            # Prepare prediction parameters
            prediction_params = {
                'input': input_data,
                'webhook': context.get('webhook_url', self.config.webhook_url)
            }
            
            # Add stream parameter if supported
            if self.config.enable_streaming and context.get('stream', True):
                prediction_params['stream'] = True
            
            # Remove None values
            prediction_params = {k: v for k, v in prediction_params.items() if v is not None}
            
            # Run prediction
            if prediction_params.get('stream'):
                return await self._stream_prediction(model, **prediction_params)
            else:
                prediction = await self._async_client.predictions.create(
                    model=model,
                    **prediction_params
                )
            
            # Wait for completion if not streaming
            if not prediction_params.get('stream'):
                prediction = await self._wait_for_completion(prediction)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update usage statistics
            self._usage_stats['total_predictions'] += 1
            self._usage_stats['total_processing_time'] += execution_time
            self._usage_stats['models_used'].add(model)
            
            if prediction.status == 'succeeded':
                self._usage_stats['successful_predictions'] += 1
            else:
                self._usage_stats['failed_predictions'] += 1
            
            # Store prediction
            replicate_pred = ReplicatePrediction(
                id=prediction.id,
                model=model,
                status=prediction.status,
                input=input_data,
                output=prediction.output,
                error=prediction.error,
                logs=prediction.logs if hasattr(prediction, 'logs') else None,
                created_at=prediction.created_at if hasattr(prediction, 'created_at') else None,
                started_at=prediction.started_at if hasattr(prediction, 'started_at') else None,
                completed_at=prediction.completed_at if hasattr(prediction, 'completed_at') else None
            )
            
            self._predictions[prediction.id] = replicate_pred
            
            result = {
                'prediction_id': prediction.id,
                'output': prediction.output,
                'status': prediction.status,
                'execution_time': execution_time,
                'model_used': model,
                'logs': prediction.logs if hasattr(prediction, 'logs') else None,
                'error': prediction.error,
                'metrics': {
                    'processing_time': execution_time,
                    'status': prediction.status
                }
            }
            
            return result
            
        except Exception as e:
            self._usage_stats['failed_predictions'] += 1
            return {'error': str(e)}
    
    async def _stream_prediction(self, model: str, **prediction_params) -> Dict[str, Any]:
        """Handle streaming predictions"""
        try:
            output_chunks = []
            prediction_id = None
            
            async for event in self._async_client.stream(model, **prediction_params):
                if hasattr(event, 'id'):
                    prediction_id = event.id
                
                if hasattr(event, 'output') and event.output:
                    output_chunks.append(event.output)
            
            full_output = ''.join(str(chunk) for chunk in output_chunks)
            
            return {
                'prediction_id': prediction_id,
                'output': full_output,
                'status': 'succeeded',
                'stream': True,
                'chunks_received': len(output_chunks)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _wait_for_completion(self, prediction, max_wait_time: float = None) -> Any:
        """Wait for prediction completion with polling"""
        max_wait = max_wait_time or self.config.timeout
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < max_wait:
            # Refresh prediction status
            try:
                prediction = await self._async_client.predictions.get(prediction.id)
                
                if prediction.status in ['succeeded', 'failed', 'canceled']:
                    return prediction
                
                # Wait before next poll
                await asyncio.sleep(self.config.poll_interval)
                
            except Exception:
                break
        
        # Timeout reached
        prediction.status = 'timeout'
        prediction.error = f'Prediction timed out after {max_wait} seconds'
        return prediction
    
    async def chat_completion(self, messages: List[Dict[str, str]], 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Chat completion using a Replicate language model"""
        try:
            context = context or {}
            model = context.get('model', self.config.default_model)
            
            # Convert messages to prompt format (model-dependent)
            prompt = self._format_chat_prompt(messages, model)
            
            # Prepare input for the model
            model_input = {
                'prompt': prompt,
                'max_new_tokens': context.get('max_tokens', 1000),
                'temperature': context.get('temperature', 0.7),
                'top_p': context.get('top_p', 0.9),
                'top_k': context.get('top_k', 50)
            }
            
            # Add model-specific parameters
            if 'system_prompt' in context:
                model_input['system_prompt'] = context['system_prompt']
            
            # Run prediction
            result = await self.run_prediction(model, model_input, context)
            
            if result.get('error'):
                return result
            
            # Extract text response from output
            response_text = self._extract_text_response(result['output'], model)
            
            return {
                'response': response_text,
                'prediction_id': result['prediction_id'],
                'model_used': model,
                'execution_time': result['execution_time'],
                'status': result['status'],
                'logs': result.get('logs'),
                'metrics': result.get('metrics', {})
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]], model: str) -> str:
        """Format chat messages into prompt for specific model"""
        if 'llama' in model.lower():
            # Llama-style formatting
            formatted = ""
            for msg in messages:
                role = msg['role']
                content = msg['content']
                
                if role == 'system':
                    formatted += f"<|system|>\n{content}\n\n"
                elif role == 'user':
                    formatted += f"<|user|>\n{content}\n\n"
                elif role == 'assistant':
                    formatted += f"<|assistant|>\n{content}\n\n"
            
            formatted += "<|assistant|>\n"
            return formatted
        
        elif 'mistral' in model.lower():
            # Mistral-style formatting
            formatted = ""
            for msg in messages:
                role = msg['role']
                content = msg['content']
                
                if role == 'system':
                    formatted += f"[INST] {content} [/INST]\n"
                elif role == 'user':
                    formatted += f"[INST] {content} [/INST]\n"
                elif role == 'assistant':
                    formatted += f"{content}\n"
            
            return formatted
        
        else:
            # Generic formatting
            formatted = ""
            for msg in messages:
                role = msg['role']
                content = msg['content']
                formatted += f"{role.title()}: {content}\n"
            
            formatted += "Assistant:"
            return formatted
    
    def _extract_text_response(self, output: Any, model: str) -> str:
        """Extract text response from model output"""
        if isinstance(output, str):
            return output
        elif isinstance(output, list):
            return ''.join(str(item) for item in output)
        elif isinstance(output, dict):
            # Try common keys
            for key in ['text', 'output', 'response', 'generated_text']:
                if key in output:
                    return str(output[key])
            return str(output)
        else:
            return str(output)
    
    async def create_deployment(self, model: str, hardware: str = "cpu", 
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a model deployment for faster inference"""
        if not self.config.enable_deployments:
            return {'error': 'Deployments not enabled'}
        
        try:
            context = context or {}
            
            deployment = await self._async_client.deployments.create(
                name=f"{self.agent_id}-{uuid.uuid4().hex[:8]}",
                model=model,
                hardware=hardware,
                min_instances=context.get('min_instances', 0),
                max_instances=context.get('max_instances', 1)
            )
            
            deployment_info = {
                'deployment_id': deployment.name,
                'model': model,
                'hardware': hardware,
                'status': deployment.current_release.status if hasattr(deployment, 'current_release') else 'unknown',
                'created_at': datetime.now().isoformat()
            }
            
            self._deployments[deployment.name] = deployment_info
            self._usage_stats['deployments_created'] += 1
            
            return deployment_info
            
        except Exception as e:
            return {'error': str(e)}
    
    async def get_deployment_status(self, deployment_name: str) -> Dict[str, Any]:
        """Get deployment status"""
        try:
            deployment = await self._async_client.deployments.get(deployment_name)
            
            status_info = {
                'deployment_name': deployment_name,
                'status': deployment.current_release.status if hasattr(deployment, 'current_release') else 'unknown',
                'model': deployment.current_release.model if hasattr(deployment, 'current_release') else None,
                'hardware': deployment.current_release.hardware if hasattr(deployment, 'current_release') else None
            }
            
            return status_info
            
        except Exception as e:
            return {'error': str(e)}
    
    async def list_models(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """List available models"""
        try:
            context = context or {}
            models = []
            
            async for model in self._async_client.models.list():
                model_info = ReplicateModel(
                    owner=model.owner,
                    name=model.name,
                    description=model.description if hasattr(model, 'description') else None
                )
                
                models.append({
                    'owner': model_info.owner,
                    'name': model_info.name,
                    'full_name': f"{model_info.owner}/{model_info.name}",
                    'description': model_info.description
                })
                
                # Store model info
                self._models[f"{model_info.owner}/{model_info.name}"] = model_info
            
            return {
                'models': models,
                'total_count': len(models)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed model information"""
        try:
            model = await self._async_client.models.get(model_name)
            
            return {
                'owner': model.owner,
                'name': model.name,
                'description': model.description if hasattr(model, 'description') else None,
                'visibility': model.visibility if hasattr(model, 'visibility') else None,
                'github_url': model.github_url if hasattr(model, 'github_url') else None,
                'paper_url': model.paper_url if hasattr(model, 'paper_url') else None,
                'license_url': model.license_url if hasattr(model, 'license_url') else None,
                'cover_image_url': model.cover_image_url if hasattr(model, 'cover_image_url') else None
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def cancel_prediction(self, prediction_id: str) -> Dict[str, Any]:
        """Cancel a running prediction"""
        try:
            prediction = await self._async_client.predictions.cancel(prediction_id)
            
            return {
                'prediction_id': prediction_id,
                'status': prediction.status,
                'canceled': True
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_prediction_history(self) -> List[Dict[str, Any]]:
        """Get prediction history"""
        return [
            {
                'id': pred.id,
                'model': pred.model,
                'status': pred.status,
                'created_at': pred.created_at.isoformat() if pred.created_at else None,
                'completed_at': pred.completed_at.isoformat() if pred.completed_at else None,
                'error': pred.error
            }
            for pred in self._predictions.values()
        ]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = self._usage_stats.copy()
        stats['models_used'] = list(stats['models_used'])  # Convert set to list
        
        # Calculate success rate
        total = stats['total_predictions']
        if total > 0:
            stats['success_rate'] = stats['successful_predictions'] / total
            stats['average_processing_time'] = stats['total_processing_time'] / total
        else:
            stats['success_rate'] = 0.0
            stats['average_processing_time'] = 0.0
        
        return stats
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': self.config.__dict__,
            'stored_models': len(self._models),
            'active_deployments': len(self._deployments),
            'prediction_history': len(self._predictions),
            'usage_stats': self.get_usage_stats(),
            'total_interactions': len(self._interaction_history),
            'replicate_available': REPLICATE_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class ReplicateAgentProvider(BaseAgentProvider):
    """
    Provider implementation for Replicate.
    
    Enables cloud-based ML model deployment with custom scaling,
    access to thousands of open-source models, and enterprise deployment capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, ReplicateAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not REPLICATE_AVAILABLE:
            self.logger.warning("Replicate not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "replicate"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.MODEL_DEPLOYMENT,
            AgentCapability.CLOUD_SCALING,
            AgentCapability.CUSTOM_MODELS,
            AgentCapability.STREAMING,
            AgentCapability.BATCH_PROCESSING,
            AgentCapability.HARDWARE_OPTIMIZATION
        ]
    
    async def initialize(self) -> bool:
        """Initialize Replicate provider"""
        if not REPLICATE_AVAILABLE:
            self.logger.error("Replicate not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Replicate provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Replicate provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Replicate agent"""
        if not self._initialized:
            await self.initialize()
        
        if not REPLICATE_AVAILABLE:
            raise RuntimeError("Replicate not available")
        
        agent_id = f"replicate_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create Replicate configuration
            replicate_config = ReplicateConfig(
                api_token=self.config.get('api_token', ''),
                default_model=self.config.get('default_model', 'meta/llama-2-70b-chat'),
                timeout=self.config.get('timeout', 300.0),
                webhook_url=self.config.get('webhook_url'),
                enable_streaming=self.config.get('enable_streaming', True),
                enable_deployments=self.config.get('enable_deployments', True),
                enable_model_scaling=self.config.get('enable_model_scaling', True),
                max_concurrent_runs=self.config.get('max_concurrent_runs', 10),
                poll_interval=self.config.get('poll_interval', 1.0),
                base_url=self.config.get('base_url', 'https://api.replicate.com')
            )
            
            # Create agent
            agent = ReplicateAgent(
                agent_id=agent_id,
                config=replicate_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Replicate agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Replicate agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Replicate agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Replicate agent"""
        if agent_id not in self._agents:
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error="Agent not found"
            )
        
        agent = self._agents[agent_id]
        context = context or {}
        
        try:
            start_time = datetime.now()
            
            # Determine execution mode
            mode = context.get('mode', 'chat')
            
            if mode == 'chat':
                # Chat completion mode
                messages = context.get('messages', [{'role': 'user', 'content': prompt}])
                if not any(msg['content'] == prompt for msg in messages):
                    messages.append({'role': 'user', 'content': prompt})
                
                result = await agent.chat_completion(messages, context)
                
            elif mode == 'prediction':
                # Direct model prediction mode
                model = context.get('model', agent.config.default_model)
                input_data = context.get('input', {'prompt': prompt})
                result = await agent.run_prediction(model, input_data, context)
                
            else:
                # Default to chat
                messages = [{'role': 'user', 'content': prompt}]
                result = await agent.chat_completion(messages, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                content=result.get('response', result.get('output', '')),
                agent_id=agent_id,
                provider=self.provider_name,
                metadata={
                    'execution_time_seconds': execution_time,
                    'mode': mode,
                    'model_used': result.get('model_used'),
                    'prediction_id': result.get('prediction_id'),
                    'status': result.get('status'),
                    'logs': result.get('logs'),
                    'metrics': result.get('metrics', {}),
                    'usage_stats': agent.get_usage_stats(),
                    'agent_info': agent.get_agent_info()
                },
                error=result.get('error')
            )
            
        except Exception as e:
            self.logger.error(f"Replicate agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Replicate agent (basic implementation)"""
        if agent_id not in self._agents:
            return False
        
        try:
            # Replicate doesn't have native tool support like OpenAI
            # This would need to be implemented as a custom wrapper
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Replicate agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Use Replicate agent to generate tool specification
            synthesis_prompt = f"""
            Create a detailed tool specification for: {tool_description}
            
            Requirements:
            1. Define the tool's purpose and functionality
            2. Specify input parameters with types and validation
            3. Describe expected outputs and return formats
            4. Include error handling and edge cases
            5. Provide implementation guidelines for Python
            6. Consider cloud deployment and scaling requirements
            7. Optimize for Replicate's model execution environment
            
            {f'Examples: {examples}' if examples else ''}
            
            Generate a comprehensive tool specification for Replicate integration.
            """
            
            agent = self._agents[agent_id]
            messages = [{'role': 'user', 'content': synthesis_prompt}]
            result = await agent.chat_completion(messages)
            
            # Create tool spec
            tool_spec = ToolSpec(
                name=f"replicate_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            self.logger.info(f"Synthesized tool for Replicate agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def create_deployment(self, agent_id: str, model: str, hardware: str = "cpu", 
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a model deployment"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.create_deployment(model, hardware, context)
    
    async def get_deployment_status(self, agent_id: str, deployment_name: str) -> Dict[str, Any]:
        """Get deployment status"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.get_deployment_status(deployment_name)
    
    async def list_models(self, agent_id: str) -> Dict[str, Any]:
        """List available models"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.list_models()
    
    async def get_model_info(self, agent_id: str, model_name: str) -> Dict[str, Any]:
        """Get model information"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.get_model_info(model_name)
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Replicate agent"""
        try:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                
                # Cancel any running predictions
                for prediction in agent._predictions.values():
                    if prediction.status in ['starting', 'processing']:
                        try:
                            await agent.cancel_prediction(prediction.id)
                        except:
                            pass
                
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Replicate agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Replicate agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Replicate agent"""
        if agent_id not in self._agents:
            return {}
        
        agent = self._agents[agent_id]
        config = self._agent_configs[agent_id]
        
        return {
            'agent_id': agent_id,
            'provider': self.provider_name,
            'config': config.__dict__,
            'agent_info': agent.get_agent_info()
        }