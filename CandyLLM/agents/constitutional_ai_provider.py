"""
Constitutional AI Agent Provider

Integrates Anthropic's Constitutional AI principles with agent capabilities,
emphasizing safety, alignment, ethical reasoning, and harmless AI behavior.
"""

import uuid
import asyncio
import json
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

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
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    # Mock classes for when dependencies are not available
    anthropic = None
    Anthropic = None
    AsyncAnthropic = None


class ConstitutionalPrinciple(Enum):
    """Core constitutional principles for AI safety"""
    HARMLESSNESS = "harmlessness"
    HELPFULNESS = "helpfulness"
    HONESTY = "honesty"
    AUTONOMY_RESPECT = "autonomy_respect"
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    PRIVACY = "privacy"
    SAFETY = "safety"
    BENEFICIAL = "beneficial"
    NON_MALEFICENCE = "non_maleficence"


class EthicalViolationType(Enum):
    """Types of ethical violations to detect and prevent"""
    HARMFUL_CONTENT = "harmful_content"
    BIAS_DISCRIMINATION = "bias_discrimination"
    PRIVACY_VIOLATION = "privacy_violation"
    DECEPTION = "deception"
    MANIPULATION = "manipulation"
    ILLEGAL_ACTIVITY = "illegal_activity"
    UNSAFE_ADVICE = "unsafe_advice"
    MISINFORMATION = "misinformation"


@dataclass
class ConstitutionalConfig:
    """Configuration for Constitutional AI agent"""
    api_key: str = ""
    model: str = "claude-3-opus-20240229"  # Default to most capable model
    max_tokens: int = 4096
    temperature: float = 0.3  # Lower for more consistent ethical behavior
    top_p: float = 0.9
    top_k: int = 40
    system_prompt: str = ""
    constitutional_principles: List[ConstitutionalPrinciple] = field(default_factory=lambda: list(ConstitutionalPrinciple))
    safety_threshold: float = 0.8  # Threshold for safety checks
    enable_critique_revision: bool = True
    enable_chain_of_thought: bool = True
    enable_self_reflection: bool = True
    max_revision_rounds: int = 3
    ethical_guidelines: Dict[str, str] = field(default_factory=dict)
    prohibited_topics: List[str] = field(default_factory=list)
    required_disclaimers: List[str] = field(default_factory=list)
    timeout: float = 120.0
    base_url: Optional[str] = None


@dataclass
class ConstitutionalCheck:
    """Result of constitutional AI safety check"""
    principle: ConstitutionalPrinciple
    passed: bool
    confidence: float
    reasoning: str
    violations: List[EthicalViolationType] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class EthicalAssessment:
    """Comprehensive ethical assessment of AI response"""
    overall_safe: bool
    safety_score: float
    constitutional_checks: List[ConstitutionalCheck] = field(default_factory=list)
    potential_harms: List[str] = field(default_factory=list)
    ethical_concerns: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    requires_revision: bool = False


@dataclass
class RevisionRound:
    """Single round of critique and revision"""
    round_number: int
    original_response: str
    critique: str
    revised_response: str
    ethical_assessment: EthicalAssessment
    improvement_score: float


@dataclass
class ConstitutionalResponse:
    """Enhanced response with constitutional AI safeguards"""
    content: str
    ethical_assessment: EthicalAssessment
    revision_history: List[RevisionRound] = field(default_factory=list)
    constitutional_reasoning: str = ""
    safety_disclaimers: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)
    confidence_level: float = 1.0
    model_used: Optional[str] = None


class ConstitutionalAIAgent:
    """Constitutional AI agent with safety, alignment, and ethical reasoning"""
    
    def __init__(self, agent_id: str, config: ConstitutionalConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._client = None
        self._async_client = None
        self._conversations = {}
        self._ethical_history = []
        self._violation_log = []
        self._usage_stats = {
            'total_requests': 0,
            'total_revisions': 0,
            'safety_violations_prevented': 0,
            'ethical_assessments': 0,
            'constitutional_checks': 0,
            'revision_rounds': 0,
            'principles_applied': {principle.value: 0 for principle in ConstitutionalPrinciple},
            'violation_types_detected': {violation.value: 0 for violation in EthicalViolationType}
        }
        self._created_at = datetime.now()
        
        # Initialize constitutional principles if empty
        if not self.config.constitutional_principles:
            self.config.constitutional_principles = list(ConstitutionalPrinciple)
        
        # Initialize default ethical guidelines
        if not self.config.ethical_guidelines:
            self.config.ethical_guidelines = self._get_default_ethical_guidelines()
        
    def _get_default_ethical_guidelines(self) -> Dict[str, str]:
        """Get default ethical guidelines"""
        return {
            'harmlessness': 'Avoid content that could cause physical, emotional, or psychological harm',
            'helpfulness': 'Provide accurate, useful, and constructive information',
            'honesty': 'Be truthful and acknowledge uncertainty when appropriate',
            'autonomy_respect': 'Respect human autonomy and decision-making capacity',
            'fairness': 'Treat all individuals and groups fairly without discrimination',
            'transparency': 'Be clear about capabilities, limitations, and reasoning',
            'privacy': 'Respect privacy and confidentiality of personal information',
            'safety': 'Prioritize physical and digital safety in all recommendations',
            'beneficial': 'Aim to benefit humanity and promote wellbeing',
            'non_maleficence': 'Do no harm through action or inaction'
        }
        
    async def initialize(self) -> bool:
        """Initialize the Constitutional AI agent"""
        if not ANTHROPIC_AVAILABLE:
            return False
        
        try:
            if not self.config.api_key:
                return False
            
            # Initialize Anthropic clients
            client_kwargs = {'api_key': self.config.api_key}
            if self.config.base_url:
                client_kwargs['base_url'] = self.config.base_url
            
            self._client = Anthropic(**client_kwargs)
            self._async_client = AsyncAnthropic(**client_kwargs)
            
            # Test connection
            await self._test_connection()
            
            return True
            
        except Exception as e:
            return False
    
    async def _test_connection(self):
        """Test API connection"""
        try:
            response = await self._async_client.messages.create(
                model=self.config.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Constitutional AI API test failed: {e}")
    
    async def _assess_ethics(self, content: str, context: Dict[str, Any] = None) -> EthicalAssessment:
        """Perform comprehensive ethical assessment"""
        try:
            constitutional_checks = []
            
            # Check each constitutional principle
            for principle in self.config.constitutional_principles:
                check = await self._check_constitutional_principle(content, principle, context)
                constitutional_checks.append(check)
                self._usage_stats['principles_applied'][principle.value] += 1
            
            # Calculate overall safety
            passed_checks = sum(1 for check in constitutional_checks if check.passed)
            safety_score = passed_checks / len(constitutional_checks) if constitutional_checks else 0.0
            overall_safe = safety_score >= self.config.safety_threshold
            
            # Collect potential harms and concerns
            potential_harms = []
            ethical_concerns = []
            improvements = []
            
            for check in constitutional_checks:
                if not check.passed:
                    ethical_concerns.append(f"{check.principle.value}: {check.reasoning}")
                    improvements.extend(check.suggestions)
                    
                    # Count violation types
                    for violation in check.violations:
                        self._usage_stats['violation_types_detected'][violation.value] += 1
                        potential_harms.append(f"Potential {violation.value.replace('_', ' ')}")
            
            # Determine if revision is required
            requires_revision = not overall_safe or safety_score < self.config.safety_threshold
            
            assessment = EthicalAssessment(
                overall_safe=overall_safe,
                safety_score=safety_score,
                constitutional_checks=constitutional_checks,
                potential_harms=list(set(potential_harms)),
                ethical_concerns=list(set(ethical_concerns)),
                improvements=list(set(improvements)),
                requires_revision=requires_revision
            )
            
            self._usage_stats['ethical_assessments'] += 1
            if requires_revision:
                self._usage_stats['safety_violations_prevented'] += 1
            
            return assessment
            
        except Exception as e:
            # Return safe default assessment
            return EthicalAssessment(
                overall_safe=False,
                safety_score=0.0,
                constitutional_checks=[],
                potential_harms=[f"Assessment error: {str(e)}"],
                ethical_concerns=["Failed to assess ethical implications"],
                improvements=["Manual review required"],
                requires_revision=True
            )
    
    async def _check_constitutional_principle(self, content: str, principle: ConstitutionalPrinciple,
                                            context: Dict[str, Any] = None) -> ConstitutionalCheck:
        """Check content against a specific constitutional principle"""
        try:
            guideline = self.config.ethical_guidelines.get(principle.value, "")
            
            check_prompt = f"""
            Evaluate the following content against the constitutional principle of {principle.value}:
            
            Principle Guideline: {guideline}
            
            Content to evaluate:
            {content}
            
            Please assess:
            1. Does this content align with the principle? (Yes/No)
            2. Confidence level (0.0 to 1.0)
            3. Reasoning for your assessment
            4. Any specific ethical violations
            5. Suggestions for improvement if needed
            
            Respond in JSON format with keys: passed, confidence, reasoning, violations, suggestions
            """
            
            response = await self._async_client.messages.create(
                model=self.config.model,
                max_tokens=500,
                temperature=0.1,  # Low temperature for consistent assessment
                messages=[
                    {"role": "user", "content": check_prompt}
                ]
            )
            
            # Parse response (simplified - would use proper JSON parsing in production)
            response_text = response.content[0].text
            
            # Extract assessment (simplified parsing)
            passed = "yes" in response_text.lower() and "no" not in response_text.lower()
            confidence = 0.8  # Default confidence
            reasoning = response_text[:200]  # Truncated reasoning
            violations = []
            suggestions = []
            
            # Try to extract more structured data if available
            try:
                if '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    json_data = json.loads(response_text[json_start:json_end])
                    
                    passed = json_data.get('passed', passed)
                    confidence = float(json_data.get('confidence', confidence))
                    reasoning = json_data.get('reasoning', reasoning)
                    violations = [EthicalViolationType(v) for v in json_data.get('violations', []) 
                                if v in [vt.value for vt in EthicalViolationType]]
                    suggestions = json_data.get('suggestions', [])
            except:
                pass  # Fall back to defaults
            
            self._usage_stats['constitutional_checks'] += 1
            
            return ConstitutionalCheck(
                principle=principle,
                passed=passed,
                confidence=confidence,
                reasoning=reasoning,
                violations=violations,
                suggestions=suggestions
            )
            
        except Exception as e:
            # Return safe default check
            return ConstitutionalCheck(
                principle=principle,
                passed=False,
                confidence=0.0,
                reasoning=f"Check failed: {str(e)}",
                violations=[EthicalViolationType.HARMFUL_CONTENT],
                suggestions=["Manual review required"]
            )
    
    async def _critique_and_revise(self, content: str, ethical_assessment: EthicalAssessment,
                                 context: Dict[str, Any] = None) -> RevisionRound:
        """Critique content and generate revised version"""
        try:
            # Generate critique
            critique_prompt = f"""
            Please provide a constructive critique of the following content based on constitutional AI principles:
            
            Content:
            {content}
            
            Ethical Concerns:
            {chr(10).join(ethical_assessment.ethical_concerns)}
            
            Improvement Suggestions:
            {chr(10).join(ethical_assessment.improvements)}
            
            Provide specific feedback on how to make this content more aligned with constitutional AI principles while maintaining helpfulness.
            """
            
            critique_response = await self._async_client.messages.create(
                model=self.config.model,
                max_tokens=1000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": critique_prompt}
                ]
            )
            
            critique = critique_response.content[0].text
            
            # Generate revised content
            revision_prompt = f"""
            Based on the following critique, please revise the content to be more aligned with constitutional AI principles:
            
            Original Content:
            {content}
            
            Critique:
            {critique}
            
            Please provide a revised version that addresses the concerns while remaining helpful and informative.
            Focus on safety, harmlessness, honesty, and beneficial outcomes.
            """
            
            revision_response = await self._async_client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "user", "content": revision_prompt}
                ]
            )
            
            revised_content = revision_response.content[0].text
            
            # Assess the revised content
            revised_assessment = await self._assess_ethics(revised_content, context)
            
            # Calculate improvement score
            improvement_score = revised_assessment.safety_score - ethical_assessment.safety_score
            
            self._usage_stats['total_revisions'] += 1
            self._usage_stats['revision_rounds'] += 1
            
            return RevisionRound(
                round_number=1,  # Will be updated by caller
                original_response=content,
                critique=critique,
                revised_response=revised_content,
                ethical_assessment=revised_assessment,
                improvement_score=improvement_score
            )
            
        except Exception as e:
            # Return minimal revision round
            return RevisionRound(
                round_number=1,
                original_response=content,
                critique=f"Critique failed: {str(e)}",
                revised_response=content,
                ethical_assessment=ethical_assessment,
                improvement_score=0.0
            )
    
    async def generate_constitutional_response(self, prompt: str, context: Dict[str, Any] = None) -> ConstitutionalResponse:
        """Generate response with constitutional AI safeguards"""
        try:
            context = context or {}
            start_time = datetime.now()
            
            # Build system prompt with constitutional principles
            system_prompt = self._build_constitutional_system_prompt()
            
            # Initial generation
            messages = []
            if system_prompt:
                messages.append({"role": "user", "content": f"System: {system_prompt}\n\nUser: {prompt}"})
            else:
                messages.append({"role": "user", "content": prompt})
            
            response = await self._async_client.messages.create(
                model=context.get('model', self.config.model),
                max_tokens=context.get('max_tokens', self.config.max_tokens),
                temperature=context.get('temperature', self.config.temperature),
                top_p=context.get('top_p', self.config.top_p),
                top_k=context.get('top_k', self.config.top_k),
                messages=messages
            )
            
            initial_content = response.content[0].text
            
            # Ethical assessment
            ethical_assessment = await self._assess_ethics(initial_content, context)
            
            revision_history = []
            current_content = initial_content
            current_assessment = ethical_assessment
            
            # Iterative revision if needed
            if self.config.enable_critique_revision and current_assessment.requires_revision:
                for round_num in range(1, self.config.max_revision_rounds + 1):
                    revision_round = await self._critique_and_revise(current_content, current_assessment, context)
                    revision_round.round_number = round_num
                    revision_history.append(revision_round)
                    
                    current_content = revision_round.revised_response
                    current_assessment = revision_round.ethical_assessment
                    
                    # Stop if satisfactory
                    if not current_assessment.requires_revision:
                        break
            
            # Generate constitutional reasoning if enabled
            constitutional_reasoning = ""
            if self.config.enable_self_reflection:
                constitutional_reasoning = await self._generate_constitutional_reasoning(
                    prompt, current_content, current_assessment
                )
            
            # Add safety disclaimers
            safety_disclaimers = self._generate_safety_disclaimers(current_assessment)
            
            # Generate alternative approaches if needed
            alternative_approaches = []
            if not current_assessment.overall_safe:
                alternative_approaches = await self._generate_alternative_approaches(prompt, context)
            
            # Calculate confidence level
            confidence_level = min(1.0, current_assessment.safety_score * 1.2)  # Boost confidence for safe content
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self._usage_stats['total_requests'] += 1
            
            return ConstitutionalResponse(
                content=current_content,
                ethical_assessment=current_assessment,
                revision_history=revision_history,
                constitutional_reasoning=constitutional_reasoning,
                safety_disclaimers=safety_disclaimers,
                alternative_approaches=alternative_approaches,
                confidence_level=confidence_level,
                model_used=context.get('model', self.config.model)
            )
            
        except Exception as e:
            # Return safe fallback response
            fallback_assessment = EthicalAssessment(
                overall_safe=False,
                safety_score=0.0,
                constitutional_checks=[],
                potential_harms=[f"Generation error: {str(e)}"],
                ethical_concerns=["Unable to generate safe response"],
                improvements=["Manual intervention required"],
                requires_revision=True
            )
            
            return ConstitutionalResponse(
                content="I apologize, but I encountered an error while generating a safe response. Please try again with a different prompt.",
                ethical_assessment=fallback_assessment,
                revision_history=[],
                constitutional_reasoning="Error in constitutional processing",
                safety_disclaimers=["This response may not meet constitutional AI standards"],
                alternative_approaches=["Please rephrase your request"],
                confidence_level=0.0,
                model_used=self.config.model
            )
    
    def _build_constitutional_system_prompt(self) -> str:
        """Build system prompt incorporating constitutional principles"""
        principles_text = "\n".join([
            f"- {principle.value.replace('_', ' ').title()}: {self.config.ethical_guidelines.get(principle.value, '')}"
            for principle in self.config.constitutional_principles
        ])
        
        return f"""
        You are a Constitutional AI assistant guided by the following principles:
        
        {principles_text}
        
        Always prioritize safety, harmlessness, and beneficial outcomes in your responses.
        Be honest about uncertainties and limitations.
        Respect human autonomy and decision-making.
        Avoid content that could cause harm or promote unethical behavior.
        
        If a request conflicts with these principles, explain why and suggest ethical alternatives.
        """
    
    async def _generate_constitutional_reasoning(self, prompt: str, response: str,
                                               assessment: EthicalAssessment) -> str:
        """Generate reasoning about constitutional AI decision-making"""
        try:
            reasoning_prompt = f"""
            Explain the constitutional AI reasoning behind this response:
            
            User Request: {prompt}
            AI Response: {response}
            Safety Score: {assessment.safety_score}
            
            Explain:
            1. How constitutional principles were applied
            2. Ethical considerations taken into account
            3. Why this response aligns with beneficial AI behavior
            4. Any limitations or caveats
            
            Keep the explanation concise and accessible.
            """
            
            reasoning_response = await self._async_client.messages.create(
                model=self.config.model,
                max_tokens=300,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": reasoning_prompt}
                ]
            )
            
            return reasoning_response.content[0].text
            
        except Exception:
            return "Constitutional AI principles were applied to ensure safety and beneficial outcomes."
    
    def _generate_safety_disclaimers(self, assessment: EthicalAssessment) -> List[str]:
        """Generate appropriate safety disclaimers"""
        disclaimers = []
        
        if not assessment.overall_safe:
            disclaimers.append("This response may not fully meet constitutional AI safety standards.")
        
        if assessment.safety_score < 0.9:
            disclaimers.append("Please exercise caution and critical thinking when considering this information.")
        
        if any("misinformation" in concern.lower() for concern in assessment.ethical_concerns):
            disclaimers.append("Please verify information from authoritative sources.")
        
        if any("safety" in concern.lower() for concern in assessment.ethical_concerns):
            disclaimers.append("Consider consulting relevant experts before taking action based on this advice.")
        
        # Add required disclaimers from config
        disclaimers.extend(self.config.required_disclaimers)
        
        return list(set(disclaimers))  # Remove duplicates
    
    async def _generate_alternative_approaches(self, prompt: str, context: Dict[str, Any] = None) -> List[str]:
        """Generate alternative approaches for ethically problematic requests"""
        try:
            alt_prompt = f"""
            The following request may have ethical concerns. Suggest 3 alternative approaches that would be more aligned with constitutional AI principles:
            
            Request: {prompt}
            
            Provide constructive alternatives that:
            1. Address the underlying need safely
            2. Promote beneficial outcomes
            3. Respect ethical boundaries
            
            List each alternative clearly.
            """
            
            alt_response = await self._async_client.messages.create(
                model=self.config.model,
                max_tokens=400,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": alt_prompt}
                ]
            )
            
            alt_text = alt_response.content[0].text
            
            # Extract alternatives (simplified parsing)
            alternatives = []
            lines = alt_text.split('\n')
            for line in lines:
                if any(marker in line for marker in ['1.', '2.', '3.', '-', '•']):
                    alternative = line.strip()
                    if len(alternative) > 10:  # Filter out short lines
                        alternatives.append(alternative)
            
            return alternatives[:3]  # Return up to 3 alternatives
            
        except Exception:
            return [
                "Consider rephrasing your request to focus on beneficial outcomes",
                "Explore educational resources on this topic from authoritative sources",
                "Consult with relevant experts or professionals for guidance"
            ]
    
    async def analyze_conversation_ethics(self, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze ethical implications of an entire conversation"""
        try:
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in conversation_history
            ])
            
            # Assess overall conversation
            conversation_assessment = await self._assess_ethics(conversation_text)
            
            # Analyze individual messages
            message_assessments = []
            for i, msg in enumerate(conversation_history):
                if msg['role'] == 'assistant':  # Only assess AI responses
                    assessment = await self._assess_ethics(msg['content'])
                    message_assessments.append({
                        'message_index': i,
                        'assessment': assessment
                    })
            
            # Calculate conversation metrics
            total_messages = len([msg for msg in conversation_history if msg['role'] == 'assistant'])
            safe_messages = sum(1 for assessment in message_assessments if assessment['assessment'].overall_safe)
            conversation_safety_rate = safe_messages / total_messages if total_messages > 0 else 1.0
            
            return {
                'conversation_assessment': conversation_assessment,
                'message_assessments': message_assessments,
                'safety_metrics': {
                    'total_ai_messages': total_messages,
                    'safe_messages': safe_messages,
                    'safety_rate': conversation_safety_rate,
                    'average_safety_score': sum(assessment['assessment'].safety_score for assessment in message_assessments) / len(message_assessments) if message_assessments else 0.0
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_ethical_history(self) -> List[Dict[str, Any]]:
        """Get history of ethical assessments"""
        return self._ethical_history
    
    def get_violation_log(self) -> List[Dict[str, Any]]:
        """Get log of detected violations"""
        return self._violation_log
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self._usage_stats.copy()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': {
                'model': self.config.model,
                'constitutional_principles': [p.value for p in self.config.constitutional_principles],
                'safety_threshold': self.config.safety_threshold,
                'enable_critique_revision': self.config.enable_critique_revision,
                'max_revision_rounds': self.config.max_revision_rounds
            },
            'active_conversations': len(self._conversations),
            'ethical_assessments': len(self._ethical_history),
            'violation_log_entries': len(self._violation_log),
            'usage_stats': self.get_usage_stats(),
            'anthropic_available': ANTHROPIC_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class ConstitutionalAIProvider(BaseAgentProvider):
    """
    Provider implementation for Constitutional AI.
    
    Emphasizes safety, alignment, ethical reasoning, and harmless AI behavior
    through constitutional training principles and iterative critique-revision.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, ConstitutionalAIAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not ANTHROPIC_AVAILABLE:
            self.logger.warning("Anthropic dependencies not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "constitutional_ai"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.ETHICAL_REASONING,
            AgentCapability.SAFETY_ASSESSMENT,
            AgentCapability.CONTENT_MODERATION,
            AgentCapability.BIAS_DETECTION,
            AgentCapability.CONSTITUTIONAL_AI,
            AgentCapability.HARMFUL_CONTENT_DETECTION
        ]
    
    async def initialize(self) -> bool:
        """Initialize Constitutional AI provider"""
        if not ANTHROPIC_AVAILABLE:
            self.logger.error("Anthropic dependencies not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Constitutional AI provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Constitutional AI provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Constitutional AI agent"""
        if not self._initialized:
            await self.initialize()
        
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("Anthropic dependencies not available")
        
        agent_id = f"constitutional_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create Constitutional AI configuration
            constitutional_config = ConstitutionalConfig(
                api_key=self.config.get('api_key', ''),
                model=self.config.get('model', 'claude-3-opus-20240229'),
                max_tokens=self.config.get('max_tokens', 4096),
                temperature=self.config.get('temperature', 0.3),
                top_p=self.config.get('top_p', 0.9),
                top_k=self.config.get('top_k', 40),
                system_prompt=self.config.get('system_prompt', ''),
                constitutional_principles=[
                    ConstitutionalPrinciple(p) for p in self.config.get('constitutional_principles', [])
                ] or list(ConstitutionalPrinciple),
                safety_threshold=self.config.get('safety_threshold', 0.8),
                enable_critique_revision=self.config.get('enable_critique_revision', True),
                enable_chain_of_thought=self.config.get('enable_chain_of_thought', True),
                enable_self_reflection=self.config.get('enable_self_reflection', True),
                max_revision_rounds=self.config.get('max_revision_rounds', 3),
                ethical_guidelines=self.config.get('ethical_guidelines', {}),
                prohibited_topics=self.config.get('prohibited_topics', []),
                required_disclaimers=self.config.get('required_disclaimers', []),
                timeout=self.config.get('timeout', 120.0),
                base_url=self.config.get('base_url')
            )
            
            # Create agent
            agent = ConstitutionalAIAgent(
                agent_id=agent_id,
                config=constitutional_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Constitutional AI agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Constitutional AI agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Constitutional AI agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Constitutional AI agent"""
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
            
            # Generate constitutional response
            constitutional_response = await agent.generate_constitutional_response(prompt, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare metadata
            metadata = {
                'execution_time_seconds': execution_time,
                'ethical_assessment': {
                    'overall_safe': constitutional_response.ethical_assessment.overall_safe,
                    'safety_score': constitutional_response.ethical_assessment.safety_score,
                    'potential_harms': constitutional_response.ethical_assessment.potential_harms,
                    'ethical_concerns': constitutional_response.ethical_assessment.ethical_concerns,
                    'improvements': constitutional_response.ethical_assessment.improvements,
                    'requires_revision': constitutional_response.ethical_assessment.requires_revision
                },
                'constitutional_reasoning': constitutional_response.constitutional_reasoning,
                'revision_count': len(constitutional_response.revision_history),
                'confidence_level': constitutional_response.confidence_level,
                'safety_disclaimers': constitutional_response.safety_disclaimers,
                'alternative_approaches': constitutional_response.alternative_approaches,
                'model_used': constitutional_response.model_used,
                'usage_stats': agent.get_usage_stats(),
                'agent_info': agent.get_agent_info()
            }
            
            # Add revision history if available
            if constitutional_response.revision_history:
                metadata['revision_history'] = [
                    {
                        'round_number': round.round_number,
                        'improvement_score': round.improvement_score,
                        'ethical_assessment_after': {
                            'safety_score': round.ethical_assessment.safety_score,
                            'overall_safe': round.ethical_assessment.overall_safe
                        }
                    }
                    for round in constitutional_response.revision_history
                ]
            
            # Add constitutional checks details
            metadata['constitutional_checks'] = [
                {
                    'principle': check.principle.value,
                    'passed': check.passed,
                    'confidence': check.confidence,
                    'violations': [v.value for v in check.violations]
                }
                for check in constitutional_response.ethical_assessment.constitutional_checks
            ]
            
            return AgentResponse(
                content=constitutional_response.content,
                agent_id=agent_id,
                provider=self.provider_name,
                metadata=metadata,
                error=None
            )
            
        except Exception as e:
            self.logger.error(f"Constitutional AI agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def assess_content_ethics(self, agent_id: str, content: str) -> Dict[str, Any]:
        """Assess ethical implications of content"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        assessment = await agent._assess_ethics(content)
        
        return {
            'overall_safe': assessment.overall_safe,
            'safety_score': assessment.safety_score,
            'constitutional_checks': [
                {
                    'principle': check.principle.value,
                    'passed': check.passed,
                    'confidence': check.confidence,
                    'reasoning': check.reasoning,
                    'violations': [v.value for v in check.violations],
                    'suggestions': check.suggestions
                }
                for check in assessment.constitutional_checks
            ],
            'potential_harms': assessment.potential_harms,
            'ethical_concerns': assessment.ethical_concerns,
            'improvements': assessment.improvements,
            'requires_revision': assessment.requires_revision
        }
    
    async def analyze_conversation_ethics(self, agent_id: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze ethical implications of conversation"""
        if agent_id not in self._agents:
            return {'error': 'Agent not found'}
        
        agent = self._agents[agent_id]
        return await agent.analyze_conversation_ethics(conversation_history)
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Constitutional AI agent (with ethical assessment)"""
        if agent_id not in self._agents:
            return False
        
        try:
            # Assess tool safety
            agent = self._agents[agent_id]
            tool_description = f"Tool: {tool_spec.name}\nDescription: {tool_spec.description}\nParameters: {tool_spec.parameters}"
            
            assessment = await agent._assess_ethics(tool_description)
            
            if not assessment.overall_safe:
                self.logger.warning(f"Tool {tool_spec.name} failed ethical assessment for agent {agent_id}")
                return False
            
            # Constitutional AI doesn't have native tool support
            # This would be implemented as ethical tool usage guidelines
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Constitutional AI principles"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Generate tool specification with constitutional principles
            agent = self._agents[agent_id]
            synthesis_prompt = f"""
            Create a tool specification that adheres to constitutional AI principles for: {tool_description}
            
            Requirements:
            1. Define the tool's purpose and functionality
            2. Specify input parameters with ethical validation
            3. Include safety checks and harm prevention
            4. Define constitutional constraints and limitations
            5. Provide ethical guidelines for usage
            6. Include bias detection and fairness measures
            7. Specify transparency and accountability features
            
            Constitutional Principles to consider:
            - Harmlessness: Prevent potential harm
            - Helpfulness: Provide beneficial functionality
            - Honesty: Be transparent about capabilities and limitations
            - Fairness: Avoid bias and discrimination
            - Privacy: Respect user privacy and data protection
            
            {f'Examples: {examples}' if examples else ''}
            
            Generate a comprehensive, ethically-aligned tool specification.
            """
            
            constitutional_response = await agent.generate_constitutional_response(synthesis_prompt)
            
            # Create tool spec with constitutional safeguards
            tool_spec = ToolSpec(
                name=f"constitutional_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={
                    'description': tool_description,
                    'constitutional_safeguards': constitutional_response.safety_disclaimers,
                    'ethical_guidelines': constitutional_response.constitutional_reasoning
                },
                security_policy={
                    'risk_level': 'high' if not constitutional_response.ethical_assessment.overall_safe else 'medium',
                    'requires_approval': True,
                    'constitutional_compliance': constitutional_response.ethical_assessment.overall_safe,
                    'safety_score': constitutional_response.ethical_assessment.safety_score
                }
            )
            
            self.logger.info(f"Synthesized constitutional tool for agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Constitutional AI agent"""
        try:
            if agent_id in self._agents:
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Constitutional AI agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Constitutional AI agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Constitutional AI agent"""
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