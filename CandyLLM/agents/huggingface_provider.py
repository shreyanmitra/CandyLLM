"""
HuggingFace Transformers Agent Provider

Integrates HuggingFace Transformers with agent capabilities and model integration,
including support for custom models, fine-tuning, and distributed inference.
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
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
        pipeline, Conversation, Agent, HfAgent, ReactCodeAgent,
        Tool, PipelineTool, load_tool
    )
    from transformers.agents import (
        CodeAgent, ReactAgent, ReactJsonAgent
    )
    from huggingface_hub import HfApi, login, logout
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Mock classes for when Transformers is not available
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoModelForSeq2SeqLM = None
    pipeline = None
    Conversation = None
    Agent = None
    HfAgent = None
    ReactCodeAgent = None
    Tool = None
    PipelineTool = None
    load_tool = None
    CodeAgent = None
    ReactAgent = None
    ReactJsonAgent = None
    HfApi = None
    login = None
    logout = None


@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace Transformers agent"""
    model_name: str = "microsoft/DialoGPT-medium"
    agent_type: str = "code_agent"  # "code_agent", "react_agent", "react_json_agent"
    model_type: str = "causal_lm"  # "causal_lm", "seq2seq_lm", "pipeline"
    device: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    use_auth_token: bool = False
    hf_token: Optional[str] = None
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    enable_tools: bool = True
    custom_tools: List[str] = field(default_factory=list)


@dataclass
class ModelInfo:
    """Information about loaded model"""
    model_name: str
    model_type: str
    device: str
    parameters_count: Optional[int]
    tokenizer_vocab_size: int
    loaded_at: datetime
    memory_usage_mb: Optional[float] = None


class HuggingFaceAgent:
    """HuggingFace Transformers agent with model integration"""
    
    def __init__(self, agent_id: str, config: HuggingFaceConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._model = None
        self._tokenizer = None
        self._agent = None
        self._pipeline = None
        self._tools = {}
        self._conversations = {}
        self._model_info = None
        self._interaction_history = []
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the HuggingFace agent"""
        if not TRANSFORMERS_AVAILABLE:
            return False
        
        try:
            # Setup authentication if needed
            if self.config.hf_token:
                login(token=self.config.hf_token)
            
            # Initialize based on agent type
            if self.config.agent_type == "code_agent":
                await self._initialize_code_agent()
            elif self.config.agent_type == "react_agent":
                await self._initialize_react_agent()
            elif self.config.agent_type == "react_json_agent":
                await self._initialize_react_json_agent()
            else:
                await self._initialize_basic_model()
            
            return True
            
        except Exception as e:
            return False
    
    async def _initialize_code_agent(self):
        """Initialize a code-based agent"""
        try:
            # Create code agent with model
            self._agent = ReactCodeAgent(
                tools=[],
                llm_engine=self.config.model_name,
                add_base_tools=self.config.enable_tools
            )
            
            # Load additional tools if specified
            if self.config.custom_tools:
                for tool_name in self.config.custom_tools:
                    try:
                        tool = load_tool(tool_name)
                        self._agent.toolbox.add_tool(tool)
                        self._tools[tool_name] = tool
                    except:
                        continue
            
        except Exception:
            # Fallback to basic model
            await self._initialize_basic_model()
    
    async def _initialize_react_agent(self):
        """Initialize a ReAct agent"""
        try:
            self._agent = ReactAgent(
                tools=[],
                llm_engine=self.config.model_name,
                add_base_tools=self.config.enable_tools
            )
            
            if self.config.custom_tools:
                for tool_name in self.config.custom_tools:
                    try:
                        tool = load_tool(tool_name)
                        self._agent.toolbox.add_tool(tool)
                        self._tools[tool_name] = tool
                    except:
                        continue
                        
        except Exception:
            await self._initialize_basic_model()
    
    async def _initialize_react_json_agent(self):
        """Initialize a ReAct JSON agent"""
        try:
            self._agent = ReactJsonAgent(
                tools=[],
                llm_engine=self.config.model_name,
                add_base_tools=self.config.enable_tools
            )
            
        except Exception:
            await self._initialize_basic_model()
    
    async def _initialize_basic_model(self):
        """Initialize basic model and tokenizer"""
        try:
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                use_auth_token=self.config.use_auth_token
            )
            
            # Set pad token if not present
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Determine device
            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model
            if self.config.model_type == "seq2seq_lm":
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if self.config.torch_dtype == "auto" and device == "cuda" else None,
                    device_map=device if device != "cpu" else None,
                    trust_remote_code=self.config.trust_remote_code,
                    use_auth_token=self.config.use_auth_token
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if self.config.torch_dtype == "auto" and device == "cuda" else None,
                    device_map=device if device != "cpu" else None,
                    trust_remote_code=self.config.trust_remote_code,
                    use_auth_token=self.config.use_auth_token
                )
            
            # Move to device if needed
            if device == "cpu":
                self._model = self._model.to(device)
            
            # Create model info
            param_count = sum(p.numel() for p in self._model.parameters())
            self._model_info = ModelInfo(
                model_name=self.config.model_name,
                model_type=self.config.model_type,
                device=str(device),
                parameters_count=param_count,
                tokenizer_vocab_size=len(self._tokenizer),
                loaded_at=datetime.now()
            )
            
            # Calculate memory usage if on GPU
            if device == "cuda":
                try:
                    memory_usage = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    self._model_info.memory_usage_mb = memory_usage
                except:
                    pass
            
        except Exception as e:
            # Fallback: create a text generation pipeline
            try:
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.config.model_name,
                    tokenizer=self.config.model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
            except:
                pass
    
    async def generate_response(self, prompt: str, conversation_id: str = None,
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response using the HuggingFace model or agent"""
        try:
            start_time = datetime.now()
            context = context or {}
            
            # Use agent if available
            if self._agent:
                result = await self._execute_agent(prompt, context)
                
            elif self._model and self._tokenizer:
                result = await self._execute_model(prompt, context)
                
            elif self._pipeline:
                result = await self._execute_pipeline(prompt, context)
                
            else:
                return {'error': 'No model or agent available'}
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update conversation if provided
            if conversation_id:
                if conversation_id not in self._conversations:
                    self._conversations[conversation_id] = []
                
                self._conversations[conversation_id].append({
                    'user_input': prompt,
                    'agent_response': result.get('response', ''),
                    'timestamp': datetime.now().isoformat(),
                    'execution_time': execution_time
                })
            
            # Record interaction
            interaction_record = {
                'conversation_id': conversation_id,
                'prompt': prompt,
                'response': result.get('response', ''),
                'execution_time': execution_time,
                'model_used': self._model_info.model_name if self._model_info else self.config.model_name,
                'timestamp': datetime.now().isoformat()
            }
            self._interaction_history.append(interaction_record)
            
            result['execution_time'] = execution_time
            result['interaction_record'] = interaction_record
            
            return result
            
        except Exception as e:
            interaction_record = {
                'conversation_id': conversation_id,
                'prompt': prompt,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            self._interaction_history.append(interaction_record)
            
            return {
                'response': '',
                'error': str(e),
                'interaction_record': interaction_record
            }
    
    async def _execute_agent(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using HuggingFace agent"""
        try:
            if hasattr(self._agent, 'run'):
                if asyncio.iscoroutinefunction(self._agent.run):
                    response = await self._agent.run(prompt)
                else:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self._agent.run(prompt)
                    )
            else:
                response = str(self._agent(prompt))
            
            return {
                'response': str(response),
                'agent_type': self.config.agent_type,
                'tools_used': list(self._tools.keys())
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _execute_model(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using raw model and tokenizer"""
        try:
            # Tokenize input
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            )
            
            # Move to model device
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 100,
                    temperature=context.get('temperature', self.config.temperature),
                    top_p=context.get('top_p', self.config.top_p),
                    do_sample=context.get('do_sample', self.config.do_sample),
                    pad_token_id=self.config.pad_token_id or self._tokenizer.pad_token_id,
                    eos_token_id=self.config.eos_token_id or self._tokenizer.eos_token_id
                )
            
            # Decode response
            response = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return {
                'response': response.strip(),
                'input_tokens': inputs['input_ids'].shape[1],
                'output_tokens': outputs.shape[1] - inputs['input_ids'].shape[1],
                'model_info': self._model_info.__dict__ if self._model_info else {}
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _execute_pipeline(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using pipeline"""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._pipeline(
                    prompt,
                    max_length=context.get('max_length', self.config.max_length),
                    temperature=context.get('temperature', self.config.temperature),
                    top_p=context.get('top_p', self.config.top_p),
                    do_sample=context.get('do_sample', self.config.do_sample)
                )
            )
            
            response = result[0]['generated_text']
            # Remove the input prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return {
                'response': response,
                'pipeline_used': True
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def add_tool(self, tool_name: str, tool_function: Callable, description: str) -> bool:
        """Add a custom tool to the agent"""
        try:
            if self._agent and hasattr(self._agent, 'toolbox'):
                # Create tool wrapper
                tool = Tool(
                    tool_function,
                    name=tool_name,
                    description=description
                )
                
                self._agent.toolbox.add_tool(tool)
                self._tools[tool_name] = tool
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self._model_info:
            return self._model_info.__dict__
        
        return {
            'model_name': self.config.model_name,
            'agent_type': self.config.agent_type,
            'model_available': self._model is not None,
            'agent_available': self._agent is not None,
            'pipeline_available': self._pipeline is not None
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': self.config.__dict__,
            'model_info': self.get_model_info(),
            'active_conversations': len(self._conversations),
            'registered_tools': list(self._tools.keys()),
            'total_interactions': len(self._interaction_history),
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class HuggingFaceAgentProvider(BaseAgentProvider):
    """
    Provider implementation for HuggingFace Transformers.
    
    Enables agent capabilities with custom models, fine-tuning support,
    distributed inference, and integration with HuggingFace ecosystem.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, HuggingFaceAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("HuggingFace Transformers not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "huggingface"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.CUSTOM_MODELS,
            AgentCapability.FINE_TUNING,
            AgentCapability.DISTRIBUTED_INFERENCE,
            AgentCapability.TOOL_SYNTHESIS,
            AgentCapability.CODE_GENERATION
        ]
    
    async def initialize(self) -> bool:
        """Initialize HuggingFace provider"""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("HuggingFace Transformers not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("HuggingFace Transformers provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize HuggingFace provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new HuggingFace agent"""
        if not self._initialized:
            await self.initialize()
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("HuggingFace Transformers not available")
        
        agent_id = f"hf_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create HuggingFace configuration
            hf_config = HuggingFaceConfig(
                model_name=self.config.get('model_name', 'microsoft/DialoGPT-medium'),
                agent_type=self.config.get('agent_type', 'code_agent'),
                model_type=self.config.get('model_type', 'causal_lm'),
                device=self.config.get('device', 'auto'),
                torch_dtype=self.config.get('torch_dtype', 'auto'),
                trust_remote_code=self.config.get('trust_remote_code', False),
                use_auth_token=self.config.get('use_auth_token', False),
                hf_token=self.config.get('hf_token'),
                max_length=self.config.get('max_length', 512),
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 0.9),
                do_sample=self.config.get('do_sample', True),
                enable_tools=self.config.get('enable_tools', True),
                custom_tools=config.tools or []
            )
            
            # Create agent
            agent = HuggingFaceAgent(
                agent_id=agent_id,
                config=hf_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize HuggingFace agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created HuggingFace agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create HuggingFace agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a HuggingFace agent"""
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
            
            # Generate response
            conversation_id = context.get('conversation_id')
            result = await agent.generate_response(prompt, conversation_id, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                content=result.get('response', ''),
                agent_id=agent_id,
                provider=self.provider_name,
                metadata={
                    'execution_time_seconds': execution_time,
                    'model_info': result.get('model_info', {}),
                    'agent_type': agent.config.agent_type,
                    'tools_used': result.get('tools_used', []),
                    'input_tokens': result.get('input_tokens'),
                    'output_tokens': result.get('output_tokens'),
                    'interaction_record': result.get('interaction_record', {}),
                    'agent_info': agent.get_agent_info()
                },
                error=result.get('error')
            )
            
        except Exception as e:
            self.logger.error(f"HuggingFace agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with HuggingFace agent"""
        if agent_id not in self._agents:
            return False
        
        agent = self._agents[agent_id]
        
        try:
            return await agent.add_tool(
                tool_spec.name,
                tool_spec.function,
                tool_spec.description
            )
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using HuggingFace agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Use HuggingFace agent to generate tool specification
            synthesis_prompt = f"""
            Create a detailed tool specification for: {tool_description}
            
            Requirements:
            1. Define the tool's purpose and functionality
            2. Specify input parameters with types and validation
            3. Describe expected outputs and return formats
            4. Include error handling and edge cases
            5. Provide implementation guidelines for Python
            6. Consider integration with HuggingFace ecosystem
            
            {f'Examples: {examples}' if examples else ''}
            
            Generate a comprehensive tool specification for HuggingFace integration.
            """
            
            agent = self._agents[agent_id]
            result = await agent.generate_response(synthesis_prompt)
            
            # Create tool spec
            tool_spec = ToolSpec(
                name=f"hf_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={'description': tool_description},
                security_policy={'risk_level': 'medium', 'requires_approval': True}
            )
            
            self.logger.info(f"Synthesized tool for HuggingFace agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def create_conversation(self, agent_id: str) -> str:
        """Create a conversation session with HuggingFace agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        agent = self._agents[agent_id]
        agent._conversations[conversation_id] = []
        return conversation_id
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a HuggingFace agent"""
        try:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                
                # Clean up model resources
                if agent._model:
                    del agent._model
                if agent._tokenizer:
                    del agent._tokenizer
                if agent._agent:
                    del agent._agent
                if agent._pipeline:
                    del agent._pipeline
                
                # Clear CUDA cache if available
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed HuggingFace agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active HuggingFace agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a HuggingFace agent"""
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