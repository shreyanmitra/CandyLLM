"""
Azure OpenAI Service Agent Provider

Integrates Azure OpenAI Service with enterprise features and compliance,
including private endpoints, managed identity, and enterprise security controls.
"""

import uuid
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Union
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
    import openai
    from azure.identity import DefaultAzureCredential, ClientSecretCredential
    from azure.keyvault.secrets import SecretClient
    from azure.monitor.opentelemetry import configure_azure_monitor
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
    # Mock classes for when Azure OpenAI is not available
    openai = None
    DefaultAzureCredential = None
    ClientSecretCredential = None
    SecretClient = None
    configure_azure_monitor = None


@dataclass
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI Service"""
    azure_endpoint: str
    api_version: str = "2024-02-15-preview"
    deployment_name: str = "gpt-4"
    api_key: Optional[str] = None
    use_managed_identity: bool = False
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    tenant_id: Optional[str] = None
    key_vault_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1500
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    enable_monitoring: bool = True
    enable_content_filtering: bool = True
    private_endpoint: bool = False


@dataclass
class AzureConversation:
    """Azure OpenAI conversation session"""
    conversation_id: str
    user_id: str
    deployment_name: str
    created_at: datetime
    last_interaction: datetime
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)
    content_filter_results: List[Dict[str, Any]] = field(default_factory=list)


class AzureOpenAIAgent:
    """Azure OpenAI Service agent with enterprise features"""
    
    def __init__(self, agent_id: str, config: AzureOpenAIConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._client = None
        self._credential = None
        self._conversations: Dict[str, AzureConversation] = {}
        self._function_registry: Dict[str, Callable] = {}
        self._interaction_history = []
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the Azure OpenAI agent"""
        if not AZURE_OPENAI_AVAILABLE:
            return False
        
        try:
            # Setup authentication
            await self._setup_authentication()
            
            # Initialize OpenAI client for Azure
            if self.config.use_managed_identity and self._credential:
                # Use managed identity
                token = self._credential.get_token("https://cognitiveservices.azure.com/.default")
                self._client = openai.AzureOpenAI(
                    azure_endpoint=self.config.azure_endpoint,
                    api_version=self.config.api_version,
                    azure_ad_token=token.token
                )
            elif self.config.api_key:
                # Use API key
                self._client = openai.AzureOpenAI(
                    azure_endpoint=self.config.azure_endpoint,
                    api_key=self.config.api_key,
                    api_version=self.config.api_version
                )
            else:
                return False
            
            # Setup monitoring if enabled
            if self.config.enable_monitoring:
                try:
                    configure_azure_monitor()
                except:
                    pass  # Monitoring setup is optional
            
            return True
            
        except Exception as e:
            return False
    
    async def _setup_authentication(self):
        """Setup Azure authentication"""
        try:
            if self.config.use_managed_identity:
                if self.config.client_id and self.config.client_secret and self.config.tenant_id:
                    # Service principal authentication
                    self._credential = ClientSecretCredential(
                        tenant_id=self.config.tenant_id,
                        client_id=self.config.client_id,
                        client_secret=self.config.client_secret
                    )
                else:
                    # Default managed identity
                    self._credential = DefaultAzureCredential()
            
            # Retrieve secrets from Key Vault if configured
            if self.config.key_vault_url and self._credential:
                secret_client = SecretClient(
                    vault_url=self.config.key_vault_url,
                    credential=self._credential
                )
                
                try:
                    # Retrieve API key from Key Vault
                    api_key_secret = secret_client.get_secret("openai-api-key")
                    self.config.api_key = api_key_secret.value
                except:
                    pass  # API key might be provided directly
                    
        except Exception as e:
            pass  # Authentication setup errors are handled in initialize
    
    async def create_conversation(self, user_id: str = None, system_prompt: str = None) -> str:
        """Create a new conversation session"""
        conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"
        
        conversation = AzureConversation(
            conversation_id=conversation_id,
            user_id=user_id,
            deployment_name=self.config.deployment_name,
            created_at=datetime.now(),
            last_interaction=datetime.now()
        )
        
        # Add system prompt if provided
        if system_prompt:
            conversation.message_history.append({
                "role": "system",
                "content": system_prompt,
                "timestamp": datetime.now().isoformat()
            })
        
        self._conversations[conversation_id] = conversation
        return conversation_id
    
    async def chat_completion(self, messages: List[Dict[str, str]], 
                            conversation_id: str = None,
                            functions: List[Dict[str, Any]] = None,
                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute chat completion with Azure OpenAI"""
        if not self._client:
            return {'error': 'Client not initialized'}
        
        try:
            start_time = datetime.now()
            context = context or {}
            
            # Prepare request parameters
            request_params = {
                'model': self.config.deployment_name,
                'messages': messages,
                'temperature': context.get('temperature', self.config.temperature),
                'max_tokens': context.get('max_tokens', self.config.max_tokens),
                'top_p': context.get('top_p', self.config.top_p),
                'frequency_penalty': context.get('frequency_penalty', self.config.frequency_penalty),
                'presence_penalty': context.get('presence_penalty', self.config.presence_penalty)
            }
            
            # Add functions if provided
            if functions:
                request_params['functions'] = functions
                request_params['function_call'] = context.get('function_call', 'auto')
            
            # Execute chat completion
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.chat.completions.create(**request_params)
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract response data
            message = response.choices[0].message
            content = message.content
            function_call = getattr(message, 'function_call', None)
            
            # Extract usage information
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            # Extract content filter results if available
            content_filter_results = []
            if hasattr(response.choices[0], 'content_filter_results'):
                content_filter_results = response.choices[0].content_filter_results
            
            # Update conversation if provided
            if conversation_id and conversation_id in self._conversations:
                conversation = self._conversations[conversation_id]
                conversation.last_interaction = datetime.now()
                conversation.message_history.extend(messages)
                conversation.message_history.append({
                    "role": "assistant",
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update token usage
                for key, value in usage.items():
                    conversation.token_usage[key] = conversation.token_usage.get(key, 0) + value
                
                # Record content filter results
                if content_filter_results:
                    conversation.content_filter_results.append({
                        'timestamp': datetime.now().isoformat(),
                        'results': content_filter_results
                    })
            
            # Record interaction
            interaction_record = {
                'conversation_id': conversation_id,
                'execution_time': execution_time,
                'token_usage': usage,
                'content_filter_results': content_filter_results,
                'function_call': function_call is not None,
                'timestamp': datetime.now().isoformat()
            }
            self._interaction_history.append(interaction_record)
            
            result = {
                'content': content,
                'function_call': function_call.__dict__ if function_call else None,
                'usage': usage,
                'content_filter_results': content_filter_results,
                'execution_time': execution_time,
                'interaction_record': interaction_record
            }
            
            # Handle function calls
            if function_call and function_call.name in self._function_registry:
                try:
                    function_args = json.loads(function_call.arguments)
                    function_result = await self._execute_function(function_call.name, function_args)
                    result['function_result'] = function_result
                except Exception as e:
                    result['function_error'] = str(e)
            
            return result
            
        except Exception as e:
            interaction_record = {
                'conversation_id': conversation_id,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            self._interaction_history.append(interaction_record)
            
            return {
                'content': '',
                'error': str(e),
                'interaction_record': interaction_record
            }
    
    async def _execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a registered function"""
        if function_name not in self._function_registry:
            raise ValueError(f"Function {function_name} not registered")
        
        function = self._function_registry[function_name]
        
        if asyncio.iscoroutinefunction(function):
            return await function(**arguments)
        else:
            return function(**arguments)
    
    def register_function(self, name: str, function: Callable, description: str, parameters: Dict[str, Any]):
        """Register a function for use with function calling"""
        self._function_registry[name] = function
        
        # Return function schema for OpenAI
        return {
            "name": name,
            "description": description,
            "parameters": parameters
        }
    
    async def generate_embedding(self, text: str, model: str = "text-embedding-ada-002") -> Dict[str, Any]:
        """Generate embeddings using Azure OpenAI"""
        if not self._client:
            return {'error': 'Client not initialized'}
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.embeddings.create(
                    model=model,
                    input=text
                )
            )
            
            return {
                'embedding': response.data[0].embedding,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_conversation_info(self, conversation_id: str) -> Dict[str, Any]:
        """Get information about a conversation"""
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return {}
        
        return {
            'conversation_id': conversation.conversation_id,
            'user_id': conversation.user_id,
            'deployment_name': conversation.deployment_name,
            'created_at': conversation.created_at.isoformat(),
            'last_interaction': conversation.last_interaction.isoformat(),
            'message_count': len(conversation.message_history),
            'token_usage': conversation.token_usage,
            'content_filter_events': len(conversation.content_filter_results)
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': self.config.__dict__,
            'active_conversations': len(self._conversations),
            'registered_functions': list(self._function_registry.keys()),
            'total_interactions': len(self._interaction_history),
            'azure_openai_available': AZURE_OPENAI_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class AzureOpenAIAgentProvider(BaseAgentProvider):
    """
    Provider implementation for Azure OpenAI Service.
    
    Enables enterprise-grade OpenAI capabilities with Azure security,
    compliance, private endpoints, and managed identity integration.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, AzureOpenAIAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not AZURE_OPENAI_AVAILABLE:
            self.logger.warning("Azure OpenAI not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "azure_openai"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.CONVERSATIONAL_AI,
            AgentCapability.FUNCTION_CALLING,
            AgentCapability.ENTERPRISE_FEATURES,
            AgentCapability.CONTENT_FILTERING,
            AgentCapability.EMBEDDINGS
        ]
    
    async def initialize(self) -> bool:
        """Initialize Azure OpenAI provider"""
        if not AZURE_OPENAI_AVAILABLE:
            self.logger.error("Azure OpenAI not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Azure OpenAI Service provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Azure OpenAI agent"""
        if not self._initialized:
            await self.initialize()
        
        if not AZURE_OPENAI_AVAILABLE:
            raise RuntimeError("Azure OpenAI not available")
        
        agent_id = f"aoi_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create Azure OpenAI configuration
            azure_config = AzureOpenAIConfig(
                azure_endpoint=self.config.get('azure_endpoint'),
                api_version=self.config.get('api_version', '2024-02-15-preview'),
                deployment_name=self.config.get('deployment_name', 'gpt-4'),
                api_key=self.config.get('api_key'),
                use_managed_identity=self.config.get('use_managed_identity', False),
                client_id=self.config.get('client_id'),
                client_secret=self.config.get('client_secret'),
                tenant_id=self.config.get('tenant_id'),
                key_vault_url=self.config.get('key_vault_url'),
                temperature=self.config.get('temperature', 0.7),
                max_tokens=self.config.get('max_tokens', 1500),
                top_p=self.config.get('top_p', 1.0),
                frequency_penalty=self.config.get('frequency_penalty', 0.0),
                presence_penalty=self.config.get('presence_penalty', 0.0),
                enable_monitoring=self.config.get('enable_monitoring', True),
                enable_content_filtering=self.config.get('enable_content_filtering', True),
                private_endpoint=self.config.get('private_endpoint', False)
            )
            
            if not azure_config.azure_endpoint:
                raise ValueError("Azure endpoint required for Azure OpenAI Service")
            
            # Create agent
            agent = AzureOpenAIAgent(
                agent_id=agent_id,
                config=azure_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Azure OpenAI agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Azure OpenAI agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Azure OpenAI agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute an Azure OpenAI agent"""
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
            
            # Prepare messages
            messages = []
            if context.get('system_prompt'):
                messages.append({"role": "system", "content": context['system_prompt']})
            messages.append({"role": "user", "content": prompt})
            
            # Get conversation ID
            conversation_id = context.get('conversation_id')
            if not conversation_id:
                conversation_id = await agent.create_conversation(
                    user_id=context.get('user_id'),
                    system_prompt=context.get('system_prompt')
                )
            
            # Prepare functions if any tools are specified
            functions = []
            for function_name in context.get('functions', []):
                if function_name in agent._function_registry:
                    # Would need function schema here
                    pass
            
            # Execute chat completion
            result = await agent.chat_completion(
                messages=messages,
                conversation_id=conversation_id,
                functions=functions if functions else None,
                context=context
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                content=result.get('content', ''),
                agent_id=agent_id,
                provider=self.provider_name,
                metadata={
                    'execution_time_seconds': execution_time,
                    'conversation_id': conversation_id,
                    'usage': result.get('usage', {}),
                    'content_filter_results': result.get('content_filter_results', []),
                    'function_call': result.get('function_call'),
                    'function_result': result.get('function_result'),
                    'agent_info': agent.get_agent_info()
                },
                error=result.get('error')
            )
            
        except Exception as e:
            self.logger.error(f"Azure OpenAI agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Azure OpenAI agent"""
        if agent_id not in self._agents:
            return False
        
        agent = self._agents[agent_id]
        
        try:
            # Create function schema
            parameters = {
                "type": "object",
                "properties": tool_spec.parameters,
                "required": list(tool_spec.parameters.keys())
            }
            
            # Register function
            function_schema = agent.register_function(
                name=tool_spec.name,
                function=tool_spec.function,
                description=tool_spec.description,
                parameters=parameters
            )
            
            self.logger.info(f"Registered tool {tool_spec.name} with Azure OpenAI agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Azure OpenAI agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Use Azure OpenAI to generate tool specification
            synthesis_prompt = f"""
            Create a detailed tool specification for: {tool_description}
            
            Requirements:
            1. Define the tool's purpose and functionality
            2. Specify input parameters with types and descriptions
            3. Describe expected outputs and return formats
            4. Include error handling and validation
            5. Provide implementation guidelines
            6. Consider enterprise security and compliance requirements
            
            {f'Examples: {examples}' if examples else ''}
            
            Generate a comprehensive tool specification for enterprise Azure deployment.
            """
            
            messages = [{"role": "user", "content": synthesis_prompt}]
            agent = self._agents[agent_id]
            result = await agent.chat_completion(messages)
            
            # Create tool spec
            tool_spec = ToolSpec(
                name=f"aoi_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            self.logger.info(f"Synthesized tool for Azure OpenAI agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def create_conversation(self, agent_id: str, user_id: str = None, system_prompt: str = None) -> str:
        """Create a conversation session with Azure OpenAI agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        return await agent.create_conversation(user_id, system_prompt)
    
    async def get_conversation_info(self, agent_id: str, conversation_id: str) -> Dict[str, Any]:
        """Get information about a conversation"""
        if agent_id not in self._agents:
            return {}
        
        agent = self._agents[agent_id]
        return agent.get_conversation_info(conversation_id)
    
    async def generate_embedding(self, agent_id: str, text: str, model: str = None) -> Dict[str, Any]:
        """Generate embeddings using Azure OpenAI"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.generate_embedding(text, model or "text-embedding-ada-002")
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy an Azure OpenAI agent"""
        try:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                # Clean up agent resources
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Azure OpenAI agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Azure OpenAI agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about an Azure OpenAI agent"""
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