"""
Camel-AI Agent Provider

Integrates Camel-AI's communicative agent framework with role-playing capabilities,
enabling multi-agent debate, collaborative problem solving, and social simulation.
"""

import uuid
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging

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
    import httpx
    import requests
    from openai import OpenAI, AsyncOpenAI
    import numpy as np
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    # Mock classes for when dependencies are not available
    httpx = None
    requests = None
    OpenAI = None
    AsyncOpenAI = None
    np = None


class RoleType(Enum):
    """Types of roles in Camel-AI conversations"""
    USER = "user"
    ASSISTANT = "assistant"
    CRITIC = "critic"
    MEDIATOR = "mediator"
    EXPERT = "expert"
    DEVIL_ADVOCATE = "devil_advocate"
    FACILITATOR = "facilitator"
    OBSERVER = "observer"
    SPECIALIST = "specialist"
    DECISION_MAKER = "decision_maker"


class ConversationPhase(Enum):
    """Phases of Camel-AI conversation"""
    INITIALIZATION = "initialization"
    ROLE_ASSIGNMENT = "role_assignment"
    PROBLEM_DEFINITION = "problem_definition"
    DEBATE = "debate"
    COLLABORATION = "collaboration"
    SYNTHESIS = "synthesis"
    CONSENSUS = "consensus"
    RESOLUTION = "resolution"
    REFLECTION = "reflection"
    COMPLETION = "completion"


class MessageType(Enum):
    """Types of messages in Camel-AI conversations"""
    INSTRUCTION = "instruction"
    RESPONSE = "response"
    QUESTION = "question"
    PROPOSAL = "proposal"
    CRITICISM = "criticism"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    CLARIFICATION = "clarification"
    SUMMARY = "summary"
    DECISION = "decision"


@dataclass
class CamelConfig:
    """Configuration for Camel-AI agent"""
    api_key: str = ""
    model: str = "gpt-4"
    base_url: str = "https://api.openai.com/v1"
    max_tokens: int = 2000
    temperature: float = 0.7
    max_turns: int = 20
    max_agents: int = 10
    enable_debate: bool = True
    enable_collaboration: bool = True
    enable_consensus: bool = True
    debate_rounds: int = 5
    consensus_threshold: float = 0.8
    role_specialization: bool = True
    enable_reflection: bool = True
    reflection_frequency: int = 3
    max_conversation_time: float = 1800.0  # 30 minutes
    enable_social_simulation: bool = True
    personality_diversity: float = 0.7
    communication_style_variety: bool = True
    enable_meta_conversation: bool = True
    conflict_resolution_enabled: bool = True
    automatic_role_assignment: bool = True


@dataclass
class Role:
    """Role definition in Camel-AI conversation"""
    role_id: str
    role_type: RoleType
    name: str
    description: str
    personality: str = ""
    expertise: List[str] = field(default_factory=list)
    communication_style: str = "professional"
    objectives: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    background: str = ""
    decision_weight: float = 1.0
    active: bool = True
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConversationMessage:
    """Message in Camel-AI conversation"""
    message_id: str
    sender_role_id: str
    content: str
    message_type: MessageType
    timestamp: datetime = field(default_factory=datetime.now)
    turn_number: int = 0
    addressing: List[str] = field(default_factory=list)  # role_ids being addressed
    references: List[str] = field(default_factory=list)  # message_ids being referenced
    confidence: float = 1.0
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationState:
    """State of ongoing Camel-AI conversation"""
    conversation_id: str
    topic: str
    phase: ConversationPhase
    active_roles: List[str]  # role_ids
    messages: List[ConversationMessage] = field(default_factory=list)
    turn_number: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    consensus_reached: bool = False
    resolution: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateOutcome:
    """Outcome of a debate session"""
    debate_id: str
    topic: str
    participants: List[str]  # role_ids
    winner: Optional[str] = None
    consensus_reached: bool = False
    final_position: Optional[str] = None
    key_arguments: List[str] = field(default_factory=list)
    resolution_strategy: Optional[str] = None
    satisfaction_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class CollaborationSession:
    """Collaborative problem-solving session"""
    session_id: str
    problem: str
    participants: List[str]  # role_ids
    approach: str = "collaborative"
    solutions: List[str] = field(default_factory=list)
    final_solution: Optional[str] = None
    implementation_plan: Optional[str] = None
    success_metrics: Dict[str, float] = field(default_factory=dict)


class CamelAgent:
    """Camel-AI communicative agent with role-playing capabilities"""
    
    def __init__(self, agent_id: str, config: CamelConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._client = None
        self._async_client = None
        self._roles = {}  # role_id -> Role
        self._conversations = {}  # conversation_id -> ConversationState
        self._debates = {}  # debate_id -> DebateOutcome
        self._collaborations = {}  # session_id -> CollaborationSession
        self._usage_stats = {
            'total_conversations': 0,
            'total_messages': 0,
            'total_roles_created': 0,
            'total_debates': 0,
            'total_collaborations': 0,
            'consensus_reached_count': 0,
            'successful_resolutions': 0,
            'average_conversation_length': 0.0,
            'average_consensus_time': 0.0,
            'role_performance': {},
            'communication_effectiveness': 0.0,
            'conflict_resolution_rate': 0.0,
            'total_conversation_time': 0.0
        }
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the Camel-AI agent"""
        if not CAMEL_AVAILABLE:
            return False
        
        try:
            if not self.config.api_key:
                return False
            
            # Initialize OpenAI clients
            client_kwargs = {
                'api_key': self.config.api_key,
                'base_url': self.config.base_url
            }
            
            self._client = OpenAI(**client_kwargs)
            self._async_client = AsyncOpenAI(**client_kwargs)
            
            # Test connection
            await self._test_connection()
            
            # Initialize default roles if automatic assignment is enabled
            if self.config.automatic_role_assignment:
                await self._initialize_default_roles()
            
            return True
            
        except Exception as e:
            return False
    
    async def _test_connection(self):
        """Test API connection"""
        try:
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Camel-AI API test failed: {e}")
    
    async def _initialize_default_roles(self):
        """Initialize default roles for common scenarios"""
        default_roles = [
            {
                'role_type': RoleType.USER,
                'name': 'User Representative',
                'description': 'Represents user interests and requirements',
                'personality': 'pragmatic and goal-oriented',
                'expertise': ['user experience', 'requirements'],
                'communication_style': 'direct'
            },
            {
                'role_type': RoleType.EXPERT,
                'name': 'Domain Expert',
                'description': 'Provides specialized knowledge and technical expertise',
                'personality': 'analytical and thorough',
                'expertise': ['technical knowledge', 'domain expertise'],
                'communication_style': 'detailed'
            },
            {
                'role_type': RoleType.CRITIC,
                'name': 'Critical Reviewer',
                'description': 'Identifies flaws, risks, and areas for improvement',
                'personality': 'skeptical and detail-oriented',
                'expertise': ['risk assessment', 'quality assurance'],
                'communication_style': 'challenging'
            },
            {
                'role_type': RoleType.MEDIATOR,
                'name': 'Conversation Mediator',
                'description': 'Facilitates discussion and manages conflicts',
                'personality': 'diplomatic and balanced',
                'expertise': ['conflict resolution', 'facilitation'],
                'communication_style': 'collaborative'
            }
        ]
        
        for role_data in default_roles:
            await self.create_role(
                role_type=role_data['role_type'],
                name=role_data['name'],
                description=role_data['description'],
                personality=role_data['personality'],
                expertise=role_data['expertise'],
                communication_style=role_data['communication_style']
            )
    
    async def create_role(self, role_type: RoleType, name: str, description: str,
                         personality: str = "", expertise: List[str] = None,
                         communication_style: str = "professional",
                         objectives: List[str] = None, constraints: List[str] = None,
                         background: str = "", decision_weight: float = 1.0) -> str:
        """Create a new role"""
        try:
            role_id = f"role_{uuid.uuid4().hex[:8]}"
            
            role = Role(
                role_id=role_id,
                role_type=role_type,
                name=name,
                description=description,
                personality=personality,
                expertise=expertise or [],
                communication_style=communication_style,
                objectives=objectives or [],
                constraints=constraints or [],
                background=background,
                decision_weight=decision_weight,
                performance_metrics={
                    'messages_sent': 0,
                    'agreements_received': 0,
                    'influence_score': 0.0,
                    'collaboration_rating': 0.0,
                    'consensus_contribution': 0.0
                }
            )
            
            self._roles[role_id] = role
            self._usage_stats['total_roles_created'] += 1
            self._usage_stats['role_performance'][role_id] = role.performance_metrics.copy()
            
            return role_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to create role: {e}")
    
    async def start_conversation(self, topic: str, participants: List[str] = None,
                                conversation_type: str = "general") -> str:
        """Start a new conversation"""
        try:
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
            
            # Use all roles if participants not specified
            if participants is None:
                participants = list(self._roles.keys())
            
            # Validate participants
            valid_participants = [p for p in participants if p in self._roles]
            if not valid_participants:
                raise ValueError("No valid participants found")
            
            conversation = ConversationState(
                conversation_id=conversation_id,
                topic=topic,
                phase=ConversationPhase.INITIALIZATION,
                active_roles=valid_participants,
                metadata={'conversation_type': conversation_type}
            )
            
            self._conversations[conversation_id] = conversation
            self._usage_stats['total_conversations'] += 1
            
            # Initialize conversation with topic introduction
            await self._advance_conversation_phase(conversation_id, ConversationPhase.PROBLEM_DEFINITION)
            
            return conversation_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to start conversation: {e}")
    
    async def send_message(self, conversation_id: str, sender_role_id: str,
                          content: str, message_type: MessageType = MessageType.RESPONSE,
                          addressing: List[str] = None, references: List[str] = None) -> str:
        """Send a message in a conversation"""
        try:
            if conversation_id not in self._conversations:
                raise ValueError("Conversation not found")
            
            if sender_role_id not in self._roles:
                raise ValueError("Role not found")
            
            conversation = self._conversations[conversation_id]
            
            if sender_role_id not in conversation.active_roles:
                raise ValueError("Role not active in conversation")
            
            message_id = f"msg_{uuid.uuid4().hex[:8]}"
            
            message = ConversationMessage(
                message_id=message_id,
                sender_role_id=sender_role_id,
                content=content,
                message_type=message_type,
                turn_number=conversation.turn_number + 1,
                addressing=addressing or [],
                references=references or []
            )
            
            conversation.messages.append(message)
            conversation.turn_number += 1
            conversation.last_activity = datetime.now()
            
            # Update role performance
            role = self._roles[sender_role_id]
            role.performance_metrics['messages_sent'] += 1
            
            self._usage_stats['total_messages'] += 1
            
            # Process message for conversation advancement
            await self._process_conversation_message(conversation_id, message)
            
            return message_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to send message: {e}")
    
    async def generate_role_response(self, conversation_id: str, role_id: str,
                                   context: Dict[str, Any] = None) -> str:
        """Generate a response for a role in conversation"""
        try:
            if conversation_id not in self._conversations:
                raise ValueError("Conversation not found")
            
            if role_id not in self._roles:
                raise ValueError("Role not found")
            
            conversation = self._conversations[conversation_id]
            role = self._roles[role_id]
            context = context or {}
            
            # Build conversation context
            recent_messages = conversation.messages[-10:]  # Last 10 messages
            conversation_context = "\n".join([
                f"{self._roles[msg.sender_role_id].name}: {msg.content}"
                for msg in recent_messages
            ])
            
            # Build role prompt
            role_prompt = f"""
            You are {role.name}, a {role.role_type.value} in a conversation.
            
            Your role description: {role.description}
            Your personality: {role.personality}
            Your expertise: {', '.join(role.expertise)}
            Your communication style: {role.communication_style}
            
            Conversation topic: {conversation.topic}
            Current phase: {conversation.phase.value}
            
            Recent conversation:
            {conversation_context}
            
            Respond as {role.name} would, staying true to your role, personality, and expertise.
            Consider the conversation phase and contribute meaningfully to {conversation.topic}.
            
            {self._get_phase_specific_instructions(conversation.phase)}
            """
            
            # Generate response
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": role_prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            generated_content = response.choices[0].message.content
            
            # Send the generated message
            message_id = await self.send_message(
                conversation_id=conversation_id,
                sender_role_id=role_id,
                content=generated_content,
                message_type=MessageType.RESPONSE
            )
            
            return generated_content
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate role response: {e}")
    
    def _get_phase_specific_instructions(self, phase: ConversationPhase) -> str:
        """Get instructions specific to conversation phase"""
        instructions = {
            ConversationPhase.PROBLEM_DEFINITION: "Focus on clearly defining and understanding the problem or topic.",
            ConversationPhase.DEBATE: "Present arguments, challenge ideas constructively, and defend your position.",
            ConversationPhase.COLLABORATION: "Work together to find solutions and build on others' ideas.",
            ConversationPhase.SYNTHESIS: "Help combine different perspectives into coherent insights.",
            ConversationPhase.CONSENSUS: "Work towards agreement and find common ground.",
            ConversationPhase.RESOLUTION: "Focus on finalizing decisions and next steps.",
            ConversationPhase.REFLECTION: "Reflect on the conversation process and outcomes."
        }
        return instructions.get(phase, "Contribute meaningfully to the conversation.")
    
    async def _process_conversation_message(self, conversation_id: str, message: ConversationMessage):
        """Process a message for conversation advancement"""
        try:
            conversation = self._conversations[conversation_id]
            
            # Analyze message for phase transition triggers
            if self._should_advance_phase(conversation, message):
                next_phase = self._get_next_phase(conversation.phase)
                if next_phase:
                    await self._advance_conversation_phase(conversation_id, next_phase)
            
            # Check for consensus indicators
            if self.config.enable_consensus and conversation.phase == ConversationPhase.CONSENSUS:
                consensus_level = await self._assess_consensus(conversation_id)
                if consensus_level >= self.config.consensus_threshold:
                    conversation.consensus_reached = True
                    await self._advance_conversation_phase(conversation_id, ConversationPhase.RESOLUTION)
            
            # Update communication effectiveness
            await self._update_communication_metrics(conversation_id, message)
            
        except Exception as e:
            pass  # Non-critical processing
    
    def _should_advance_phase(self, conversation: ConversationState, message: ConversationMessage) -> bool:
        """Determine if conversation should advance to next phase"""
        # Simple heuristics for phase advancement
        if conversation.phase == ConversationPhase.PROBLEM_DEFINITION and len(conversation.messages) >= 3:
            return True
        elif conversation.phase == ConversationPhase.DEBATE and len(conversation.messages) >= self.config.debate_rounds * len(conversation.active_roles):
            return True
        elif conversation.phase == ConversationPhase.COLLABORATION and len(conversation.messages) >= 10:
            return True
        elif conversation.phase == ConversationPhase.SYNTHESIS and len(conversation.messages) >= 5:
            return True
        
        return False
    
    def _get_next_phase(self, current_phase: ConversationPhase) -> Optional[ConversationPhase]:
        """Get the next logical phase"""
        phase_sequence = [
            ConversationPhase.INITIALIZATION,
            ConversationPhase.PROBLEM_DEFINITION,
            ConversationPhase.DEBATE,
            ConversationPhase.COLLABORATION,
            ConversationPhase.SYNTHESIS,
            ConversationPhase.CONSENSUS,
            ConversationPhase.RESOLUTION,
            ConversationPhase.REFLECTION,
            ConversationPhase.COMPLETION
        ]
        
        try:
            current_index = phase_sequence.index(current_phase)
            if current_index < len(phase_sequence) - 1:
                return phase_sequence[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    async def _advance_conversation_phase(self, conversation_id: str, new_phase: ConversationPhase):
        """Advance conversation to new phase"""
        try:
            conversation = self._conversations[conversation_id]
            conversation.phase = new_phase
            
            # Generate phase transition message
            phase_message = f"Conversation advancing to {new_phase.value} phase."
            
            system_message = ConversationMessage(
                message_id=f"sys_{uuid.uuid4().hex[:8]}",
                sender_role_id="system",
                content=phase_message,
                message_type=MessageType.INSTRUCTION,
                turn_number=conversation.turn_number + 1
            )
            
            conversation.messages.append(system_message)
            conversation.turn_number += 1
            
        except Exception as e:
            pass  # Non-critical operation
    
    async def _assess_consensus(self, conversation_id: str) -> float:
        """Assess level of consensus in conversation"""
        try:
            conversation = self._conversations[conversation_id]
            
            # Simple consensus assessment based on agreement messages
            total_messages = len(conversation.messages)
            agreement_messages = sum(1 for msg in conversation.messages 
                                   if msg.message_type == MessageType.AGREEMENT)
            
            if total_messages > 0:
                return agreement_messages / total_messages
            
            return 0.0
            
        except Exception as e:
            return 0.0
    
    async def _update_communication_metrics(self, conversation_id: str, message: ConversationMessage):
        """Update communication effectiveness metrics"""
        try:
            # Simple metrics update
            self._usage_stats['communication_effectiveness'] = min(1.0, 
                self._usage_stats['communication_effectiveness'] + 0.01)
            
            # Update role performance
            if message.sender_role_id in self._roles:
                role = self._roles[message.sender_role_id]
                role.performance_metrics['influence_score'] += 0.1
                
        except Exception as e:
            pass
    
    async def start_debate(self, topic: str, participants: List[str], 
                          debate_format: str = "structured") -> str:
        """Start a structured debate"""
        try:
            debate_id = f"debate_{uuid.uuid4().hex[:8]}"
            
            # Create debate conversation
            conversation_id = await self.start_conversation(
                topic=f"Debate: {topic}",
                participants=participants,
                conversation_type="debate"
            )
            
            debate = DebateOutcome(
                debate_id=debate_id,
                topic=topic,
                participants=participants
            )
            
            self._debates[debate_id] = debate
            self._usage_stats['total_debates'] += 1
            
            # Set conversation to debate phase
            if conversation_id in self._conversations:
                await self._advance_conversation_phase(conversation_id, ConversationPhase.DEBATE)
            
            return debate_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to start debate: {e}")
    
    async def start_collaboration(self, problem: str, participants: List[str],
                                 approach: str = "collaborative") -> str:
        """Start a collaborative problem-solving session"""
        try:
            session_id = f"collab_{uuid.uuid4().hex[:8]}"
            
            # Create collaboration conversation
            conversation_id = await self.start_conversation(
                topic=f"Collaboration: {problem}",
                participants=participants,
                conversation_type="collaboration"
            )
            
            collaboration = CollaborationSession(
                session_id=session_id,
                problem=problem,
                participants=participants,
                approach=approach
            )
            
            self._collaborations[session_id] = collaboration
            self._usage_stats['total_collaborations'] += 1
            
            # Set conversation to collaboration phase
            if conversation_id in self._conversations:
                await self._advance_conversation_phase(conversation_id, ConversationPhase.COLLABORATION)
            
            return session_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to start collaboration: {e}")
    
    async def simulate_conversation(self, conversation_id: str, max_turns: int = None) -> Dict[str, Any]:
        """Simulate automatic conversation between roles"""
        try:
            if conversation_id not in self._conversations:
                raise ValueError("Conversation not found")
            
            conversation = self._conversations[conversation_id]
            max_turns = max_turns or self.config.max_turns
            start_time = datetime.now()
            
            # Simulate conversation turns
            for turn in range(max_turns):
                if conversation.phase == ConversationPhase.COMPLETION:
                    break
                
                # Check time limit
                if (datetime.now() - start_time).total_seconds() > self.config.max_conversation_time:
                    break
                
                # Get active role for this turn
                role_index = turn % len(conversation.active_roles)
                current_role = conversation.active_roles[role_index]
                
                # Generate and send response
                await self.generate_role_response(conversation_id, current_role)
                
                # Small delay between messages
                await asyncio.sleep(0.5)
            
            # Generate final summary
            summary = await self._generate_conversation_summary(conversation_id)
            
            # Update statistics
            conversation_time = (datetime.now() - start_time).total_seconds()
            self._usage_stats['total_conversation_time'] += conversation_time
            self._usage_stats['average_conversation_length'] = (
                (self._usage_stats['average_conversation_length'] * (self._usage_stats['total_conversations'] - 1) +
                 len(conversation.messages)) / self._usage_stats['total_conversations']
            )
            
            return {
                'conversation_id': conversation_id,
                'turns_completed': conversation.turn_number,
                'final_phase': conversation.phase.value,
                'consensus_reached': conversation.consensus_reached,
                'summary': summary,
                'participants': len(conversation.active_roles),
                'total_messages': len(conversation.messages),
                'duration_seconds': conversation_time
            }
            
        except Exception as e:
            raise RuntimeError(f"Conversation simulation failed: {e}")
    
    async def _generate_conversation_summary(self, conversation_id: str) -> str:
        """Generate summary of conversation"""
        try:
            conversation = self._conversations[conversation_id]
            
            # Get key messages
            key_messages = conversation.messages[-5:] if len(conversation.messages) > 5 else conversation.messages
            
            summary_prompt = f"""
            Summarize this conversation about: {conversation.topic}
            
            Key messages:
            {chr(10).join([f"{self._roles.get(msg.sender_role_id, {}).get('name', 'Unknown')}: {msg.content}" for msg in key_messages])}
            
            Phase reached: {conversation.phase.value}
            Consensus reached: {conversation.consensus_reached}
            
            Provide a comprehensive summary of the discussion, key points, and outcomes.
            """
            
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Summary generation failed. Conversation had {len(conversation.messages)} messages across {conversation.turn_number} turns."
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """Get conversation by ID"""
        return self._conversations.get(conversation_id)
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID"""
        return self._roles.get(role_id)
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations"""
        return [
            {
                'conversation_id': conv.conversation_id,
                'topic': conv.topic,
                'phase': conv.phase.value,
                'participants': len(conv.active_roles),
                'messages': len(conv.messages),
                'consensus_reached': conv.consensus_reached,
                'start_time': conv.start_time.isoformat()
            }
            for conv in self._conversations.values()
        ]
    
    def list_roles(self) -> List[Dict[str, Any]]:
        """List all roles"""
        return [
            {
                'role_id': role.role_id,
                'name': role.name,
                'role_type': role.role_type.value,
                'description': role.description,
                'expertise': role.expertise,
                'active': role.active,
                'performance': role.performance_metrics
            }
            for role in self._roles.values()
        ]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self._usage_stats.copy()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': {
                'model': self.config.model,
                'max_turns': self.config.max_turns,
                'enable_debate': self.config.enable_debate,
                'enable_collaboration': self.config.enable_collaboration,
                'enable_consensus': self.config.enable_consensus,
                'role_specialization': self.config.role_specialization
            },
            'roles_count': len(self._roles),
            'conversations_count': len(self._conversations),
            'debates_count': len(self._debates),
            'collaborations_count': len(self._collaborations),
            'usage_stats': self.get_usage_stats(),
            'camel_available': CAMEL_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class CamelAIProvider(BaseAgentProvider):
    """
    Provider implementation for Camel-AI.
    
    Enables communicative agents with role-playing capabilities for multi-agent debate,
    collaborative problem solving, and social simulation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, CamelAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not CAMEL_AVAILABLE:
            self.logger.warning("Camel-AI dependencies not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "camel-ai"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.MULTI_AGENT_CONVERSATION,
            AgentCapability.ROLE_PLAYING,
            AgentCapability.COLLABORATIVE_REASONING,
            AgentCapability.DEBATE_FACILITATION,
            AgentCapability.CONSENSUS_BUILDING,
            AgentCapability.SOCIAL_SIMULATION
        ]
    
    async def initialize(self) -> bool:
        """Initialize Camel-AI provider"""
        if not CAMEL_AVAILABLE:
            self.logger.error("Camel-AI dependencies not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Camel-AI provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Camel-AI provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Camel-AI agent"""
        if not self._initialized:
            await self.initialize()
        
        if not CAMEL_AVAILABLE:
            raise RuntimeError("Camel-AI dependencies not available")
        
        agent_id = f"camel_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create Camel-AI configuration
            camel_config = CamelConfig(
                api_key=self.config.get('api_key', ''),
                model=self.config.get('model', 'gpt-4'),
                base_url=self.config.get('base_url', 'https://api.openai.com/v1'),
                max_tokens=self.config.get('max_tokens', 2000),
                temperature=self.config.get('temperature', 0.7),
                max_turns=self.config.get('max_turns', 20),
                max_agents=self.config.get('max_agents', 10),
                enable_debate=self.config.get('enable_debate', True),
                enable_collaboration=self.config.get('enable_collaboration', True),
                enable_consensus=self.config.get('enable_consensus', True),
                debate_rounds=self.config.get('debate_rounds', 5),
                consensus_threshold=self.config.get('consensus_threshold', 0.8),
                role_specialization=self.config.get('role_specialization', True),
                enable_reflection=self.config.get('enable_reflection', True),
                reflection_frequency=self.config.get('reflection_frequency', 3),
                max_conversation_time=self.config.get('max_conversation_time', 1800.0),
                enable_social_simulation=self.config.get('enable_social_simulation', True),
                personality_diversity=self.config.get('personality_diversity', 0.7),
                communication_style_variety=self.config.get('communication_style_variety', True),
                enable_meta_conversation=self.config.get('enable_meta_conversation', True),
                conflict_resolution_enabled=self.config.get('conflict_resolution_enabled', True),
                automatic_role_assignment=self.config.get('automatic_role_assignment', True)
            )
            
            # Create agent
            agent = CamelAgent(
                agent_id=agent_id,
                config=camel_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Camel-AI agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Camel-AI agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Camel-AI agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Camel-AI agent"""
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
            
            # Determine execution type
            execution_type = context.get('execution_type', 'conversation')
            
            if execution_type == 'debate':
                result = await self._execute_debate(agent, prompt, context)
            elif execution_type == 'collaboration':
                result = await self._execute_collaboration(agent, prompt, context)
            else:
                result = await self._execute_conversation(agent, prompt, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare response
            response_content = result.get('summary', 'Conversation completed')
            
            # Prepare metadata
            metadata = {
                'execution_time_seconds': execution_time,
                'execution_type': execution_type,
                'conversation_id': result.get('conversation_id'),
                'turns_completed': result.get('turns_completed', 0),
                'participants': result.get('participants', 0),
                'consensus_reached': result.get('consensus_reached', False),
                'final_phase': result.get('final_phase'),
                'usage_stats': agent.get_usage_stats(),
                'agent_info': agent.get_agent_info()
            }
            
            return AgentResponse(
                content=response_content,
                agent_id=agent_id,
                provider=self.provider_name,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Camel-AI agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def _execute_conversation(self, agent: CamelAgent, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a general conversation"""
        participants = context.get('participants')
        max_turns = context.get('max_turns')
        
        # Start conversation
        conversation_id = await agent.start_conversation(prompt, participants)
        
        # Simulate conversation
        result = await agent.simulate_conversation(conversation_id, max_turns)
        
        return result
    
    async def _execute_debate(self, agent: CamelAgent, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a debate"""
        participants = context.get('participants')
        debate_format = context.get('debate_format', 'structured')
        
        # Start debate
        debate_id = await agent.start_debate(prompt, participants, debate_format)
        
        # Find associated conversation and simulate
        conversations = agent.list_conversations()
        debate_conversation = next((c for c in conversations if f"Debate: {prompt}" in c['topic']), None)
        
        if debate_conversation:
            result = await agent.simulate_conversation(debate_conversation['conversation_id'])
            result['debate_id'] = debate_id
            return result
        
        return {'debate_id': debate_id, 'error': 'Could not simulate debate'}
    
    async def _execute_collaboration(self, agent: CamelAgent, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a collaboration session"""
        participants = context.get('participants')
        approach = context.get('approach', 'collaborative')
        
        # Start collaboration
        session_id = await agent.start_collaboration(prompt, participants, approach)
        
        # Find associated conversation and simulate
        conversations = agent.list_conversations()
        collab_conversation = next((c for c in conversations if f"Collaboration: {prompt}" in c['topic']), None)
        
        if collab_conversation:
            result = await agent.simulate_conversation(collab_conversation['conversation_id'])
            result['session_id'] = session_id
            return result
        
        return {'session_id': session_id, 'error': 'Could not simulate collaboration'}
    
    async def create_role(self, agent_id: str, role_type: str, name: str, description: str, **kwargs) -> str:
        """Create a role for an agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        
        try:
            role_type_enum = RoleType(role_type)
        except ValueError:
            role_type_enum = RoleType.ASSISTANT
        
        return await agent.create_role(
            role_type=role_type_enum,
            name=name,
            description=description,
            **kwargs
        )
    
    async def start_conversation(self, agent_id: str, topic: str, participants: List[str] = None) -> str:
        """Start a conversation"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        return await agent.start_conversation(topic, participants)
    
    async def start_debate(self, agent_id: str, topic: str, participants: List[str]) -> str:
        """Start a debate"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        return await agent.start_debate(topic, participants)
    
    async def start_collaboration(self, agent_id: str, problem: str, participants: List[str]) -> str:
        """Start a collaboration session"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        return await agent.start_collaboration(problem, participants)
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Camel-AI agent (basic implementation)"""
        if agent_id not in self._agents:
            return False
        
        try:
            # Camel-AI doesn't have native tool support in this implementation
            # This would be implemented as role capabilities
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Camel-AI capabilities"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Create tool spec for communicative agents
            tool_spec = ToolSpec(
                name=f"camel_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={
                    'description': tool_description,
                    'execution_type': 'communicative',
                    'role_based': True,
                    'examples': examples or []
                },
                security_policy={
                    'risk_level': 'medium',
                    'requires_approval': True,
                    'multi_agent': True,
                    'consensus_required': True
                }
            )
            
            self.logger.info(f"Synthesized tool for Camel-AI agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Camel-AI agent"""
        try:
            if agent_id in self._agents:
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Camel-AI agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Camel-AI agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Camel-AI agent"""
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