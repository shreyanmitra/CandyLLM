"""
🍭 CandyLLM 3.0 Type Definitions

Comprehensive type system for CandyLLM 3.0 with all response formats,
data structures, and configuration types.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime
from enum import Enum
import json

class TaskType(Enum):
    """Different types of AI tasks"""
    CHAT = "chat"
    COMPLETION = "completion"
    REASONING = "reasoning"
    MATH = "math"
    CODE = "code"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"

class ReasoningMode(Enum):
    """Different reasoning approaches"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    ABDUCTIVE = "abductive"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    CAUSAL = "causal"
    ANALOGICAL = "analogical"
    METACOGNITIVE = "metacognitive"

class OptimizationTarget(Enum):
    """Performance optimization targets"""
    QUALITY = "quality"
    SPEED = "speed"
    COST = "cost"
    BALANCED = "balanced"

@dataclass
class RouteDecision:
    """Information about intelligent routing decision"""
    selected_model: str
    confidence: float
    reasoning: str
    task_type: str
    estimated_cost: float
    estimated_latency: float
    alternative_models: List[str] = field(default_factory=list)
    model_scores: Dict[str, float] = field(default_factory=dict)
    routing_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChatResponse:
    """Standard chat response format"""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    cost_estimate: float = 0.0
    processing_time: float = 0.0
    route_decision: Optional[RouteDecision] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning_chain: Optional[List[str]] = None
    confidence_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage,
            "cost_estimate": self.cost_estimate,
            "processing_time": self.processing_time,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
            "reasoning_chain": self.reasoning_chain,
            "confidence_score": self.confidence_score
        }
        
        if self.route_decision:
            result["route_decision"] = {
                "selected_model": self.route_decision.selected_model,
                "confidence": self.route_decision.confidence,
                "reasoning": self.route_decision.reasoning,
                "task_type": self.route_decision.task_type,
                "estimated_cost": self.route_decision.estimated_cost,
                "estimated_latency": self.route_decision.estimated_latency
            }
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class StreamChunk:
    """Streaming response chunk"""
    content: str
    model: str
    provider: str
    chunk_id: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class ReasoningStep:
    """Single step in reasoning chain"""
    step_id: str
    content: str
    reasoning_type: ReasoningMode
    confidence: float
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningPath:
    """Complete reasoning path with multiple steps"""
    path_id: str
    steps: List[ReasoningStep]
    conclusion: str
    overall_confidence: float
    reasoning_chain: List[str]
    supporting_evidence: List[str]
    limitations: List[str]
    alternative_explanations: List[str]
    path_type: ReasoningMode = ReasoningMode.SEQUENTIAL
    processing_time: float = 0.0

@dataclass
class ReasoningResult:
    """Complete reasoning result with multiple paths"""
    query: str
    primary_answer: str
    reasoning_paths: List[ReasoningPath]
    confidence_score: float
    evidence_quality: float
    reasoning_transparency: Dict[str, Any]
    alternative_answers: List[str]
    uncertainty_factors: List[str]
    recommendations: List[str]
    model_used: Optional[str] = None
    processing_time: float = 0.0
    
    def get_best_path(self) -> Optional[ReasoningPath]:
        """Get the reasoning path with highest confidence"""
        if not self.reasoning_paths:
            return None
        return max(self.reasoning_paths, key=lambda p: p.overall_confidence)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of reasoning results"""
        return {
            "query": self.query,
            "answer": self.primary_answer,
            "confidence": self.confidence_score,
            "num_paths": len(self.reasoning_paths),
            "evidence_quality": self.evidence_quality,
            "model_used": self.model_used,
            "processing_time": self.processing_time
        }

@dataclass
class MathResult:
    """Mathematical reasoning and computation result"""
    problem: str
    symbolic_result: Optional[str] = None
    numeric_result: Optional[Union[float, int, str]] = None
    steps: List[str] = field(default_factory=list)
    solution_method: str = "symbolic"
    confidence: float = 1.0
    verification_result: Optional[bool] = None
    error_analysis: Optional[str] = None
    computational_complexity: Optional[str] = None
    alternative_solutions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "problem": self.problem,
            "symbolic_result": self.symbolic_result,
            "numeric_result": self.numeric_result,
            "steps": self.steps,
            "solution_method": self.solution_method,
            "confidence": self.confidence,
            "verification_result": self.verification_result,
            "error_analysis": self.error_analysis,
            "computational_complexity": self.computational_complexity,
            "alternative_solutions": self.alternative_solutions,
            "metadata": self.metadata
        }

@dataclass
class KnowledgeTriple:
    """Knowledge graph triple (subject, predicate, object)"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"({self.subject}) --[{self.predicate}]--> ({self.object})"

@dataclass
class ModelCapabilities:
    """Model capability information"""
    model_id: str
    provider: str
    context_length: int
    supports_chat: bool = True
    supports_completion: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    supports_streaming: bool = True
    supports_json_mode: bool = False
    max_output_tokens: Optional[int] = None
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    average_latency: float = 0.0
    quality_score: float = 0.0
    specialized_tasks: List[str] = field(default_factory=list)
    release_date: Optional[str] = None

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    model_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency: float = 0.0
    average_cost: float = 0.0
    average_quality_score: float = 0.0
    total_tokens_processed: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        return 1.0 - self.success_rate

@dataclass
class TaskClassification:
    """Task classification result"""
    task_type: TaskType
    confidence: float
    reasoning: str
    complexity_score: float
    estimated_tokens: int
    required_capabilities: List[str]
    recommended_models: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CostEstimate:
    """Cost estimation for a request"""
    input_tokens: int
    estimated_output_tokens: int
    total_estimated_tokens: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    total_estimated_cost: float
    currency: str = "USD"
    provider: str = ""
    model: str = ""

@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    relevance_score: float = 0.0
    coherence_score: float = 0.0
    factual_accuracy: float = 0.0
    completeness_score: float = 0.0
    overall_quality: float = 0.0
    assessment_method: str = "automatic"
    human_feedback: Optional[Dict[str, Any]] = None

@dataclass
class NeurosymbolicResult:
    """Result from neurosymbolic reasoning"""
    query: str
    symbolic_reasoning: List[str]
    neural_response: str
    combined_result: str
    confidence_scores: Dict[str, float]
    knowledge_graph_usage: List[KnowledgeTriple]
    mathematical_computation: Optional[MathResult] = None
    logical_inference_steps: List[str] = field(default_factory=list)
    causal_analysis: Optional[Dict[str, Any]] = None

@dataclass
class BatchRequest:
    """Batch processing request"""
    requests: List[Dict[str, Any]]
    batch_id: str
    priority: int = 1
    max_parallel: int = 5
    timeout: int = 300
    retry_failed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchResponse:
    """Batch processing response"""
    batch_id: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    responses: List[Union[ChatResponse, ReasoningResult, MathResult]]
    processing_time: float
    total_cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate batch success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

# Utility functions for type conversion and validation

def validate_chat_response(response: Dict[str, Any]) -> ChatResponse:
    """Validate and convert dictionary to ChatResponse"""
    required_fields = ["content", "model", "provider"]
    
    for field in required_fields:
        if field not in response:
            raise ValueError(f"Missing required field: {field}")
    
    return ChatResponse(
        content=response["content"],
        model=response["model"],
        provider=response["provider"],
        usage=response.get("usage"),
        cost_estimate=response.get("cost_estimate", 0.0),
        processing_time=response.get("processing_time", 0.0),
        success=response.get("success", True),
        error=response.get("error"),
        metadata=response.get("metadata", {}),
        reasoning_chain=response.get("reasoning_chain"),
        confidence_score=response.get("confidence_score")
    )

def create_error_response(error_message: str, 
                         model: str = "error",
                         provider: str = "error") -> ChatResponse:
    """Create standardized error response"""
    return ChatResponse(
        content=f"Error: {error_message}",
        model=model,
        provider=provider,
        success=False,
        error=error_message,
        metadata={"timestamp": datetime.now().isoformat()}
    )

def merge_reasoning_results(results: List[ReasoningResult]) -> ReasoningResult:
    """Merge multiple reasoning results into one"""
    if not results:
        raise ValueError("No results to merge")
    
    if len(results) == 1:
        return results[0]
    
    # Combine all reasoning paths
    all_paths = []
    all_alternatives = []
    all_uncertainties = []
    all_recommendations = []
    
    for result in results:
        all_paths.extend(result.reasoning_paths)
        all_alternatives.extend(result.alternative_answers)
        all_uncertainties.extend(result.uncertainty_factors)
        all_recommendations.extend(result.recommendations)
    
    # Use the result with highest confidence as primary
    primary_result = max(results, key=lambda r: r.confidence_score)
    
    # Calculate ensemble confidence
    ensemble_confidence = sum(r.confidence_score for r in results) / len(results)
    
    return ReasoningResult(
        query=primary_result.query,
        primary_answer=primary_result.primary_answer,
        reasoning_paths=all_paths,
        confidence_score=ensemble_confidence,
        evidence_quality=sum(r.evidence_quality for r in results) / len(results),
        reasoning_transparency={
            "ensemble_reasoning": True,
            "num_merged_results": len(results),
            "individual_confidences": [r.confidence_score for r in results]
        },
        alternative_answers=list(set(all_alternatives)),
        uncertainty_factors=list(set(all_uncertainties)),
        recommendations=list(set(all_recommendations))
    )

# Type aliases for convenience
ModelID = str
ProviderName = str
APIKey = str
ConfigDict = Dict[str, Any]
MessageList = List[Dict[str, str]]
TokenCount = int
CostAmount = float

# Constants
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4000

# Model ID patterns
OPENAI_MODELS = ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini"]
ANTHROPIC_MODELS = ["claude-3.5-sonnet", "claude-3.5-haiku", "claude-3-opus"]
GOOGLE_MODELS = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
COHERE_MODELS = ["command-r7", "command-r7-plus"]
AMAZON_MODELS = ["nova-pro", "nova-lite", "nova-micro"]
META_MODELS = ["llama-3.1-405b", "llama-3.3-70b"]
DEEPSEEK_MODELS = ["deepseek-v3"]
MISTRAL_MODELS = ["large-2", "codestral"]

ALL_SUPPORTED_MODELS = (
    [f"openai:{m}" for m in OPENAI_MODELS] +
    [f"anthropic:{m}" for m in ANTHROPIC_MODELS] +
    [f"google:{m}" for m in GOOGLE_MODELS] +
    [f"cohere:{m}" for m in COHERE_MODELS] +
    [f"amazon:{m}" for m in AMAZON_MODELS] +
    [f"meta:{m}" for m in META_MODELS] +
    [f"deepseek:{m}" for m in DEEPSEEK_MODELS] +
    [f"mistral:{m}" for m in MISTRAL_MODELS]
)
