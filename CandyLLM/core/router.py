"""
🍭 CandyLLM Intelligent Router
Advanced AI model routing system with performance optimization and task-specific selection.
"""

import time
import asyncio
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import re
import numpy as np
from enum import Enum

class TaskType(Enum):
    """Types of AI tasks for intelligent routing"""
    CHAT = "chat"
    CODE = "code"
    REASONING = "reasoning"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    MATH = "math"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QA = "question_answering"
    MULTIMODAL = "multimodal"
    TOOL_USE = "tool_use"
    RESEARCH = "research"

class ModelCapability(Enum):
    """Model capabilities for routing decisions"""
    SPEED = "speed"
    QUALITY = "quality"
    COST = "cost"
    CONTEXT_LENGTH = "context_length"
    REASONING = "reasoning"
    CODE = "code"
    MATH = "math"
    MULTIMODAL = "multimodal"
    TOOLS = "tools"
    LANGUAGES = "languages"

@dataclass
class ModelProfile:
    """Performance profile for a model"""
    model_id: str
    provider: str
    capabilities: Dict[ModelCapability, float]  # 0-1 scores
    avg_response_time: float
    cost_per_token: float
    max_context_length: int
    supported_features: List[str]
    recent_performance: deque
    success_rate: float = 1.0
    load_factor: float = 0.0
    
class TaskClassifier:
    """Classifies user inputs into task types"""
    
    def __init__(self):
        self.task_patterns = {
            TaskType.CODE: [
                r'\b(code|program|function|class|debug|fix|implement)\b',
                r'\b(python|javascript|java|c\+\+|html|css|sql)\b',
                r'```|`[^`]+`',
                r'\b(algorithm|data structure|api|framework)\b'
            ],
            TaskType.MATH: [
                r'\b(calculate|solve|equation|formula|math|statistics)\b',
                r'[0-9]+\s*[\+\-\*\/\=\^]\s*[0-9]+',
                r'\b(derivative|integral|matrix|probability)\b',
                r'\b(geometry|algebra|calculus|theorem)\b'
            ],
            TaskType.REASONING: [
                r'\b(analyze|reason|logic|conclude|infer|deduce)\b',
                r'\b(because|therefore|however|although|consequently)\b',
                r'\b(step by step|think through|break down)\b',
                r'\b(argument|premise|conclusion|hypothesis)\b'
            ],
            TaskType.CREATIVE: [
                r'\b(write|create|generate|story|poem|creative)\b',
                r'\b(imagine|creative|artistic|novel|narrative)\b',
                r'\b(character|plot|dialogue|setting)\b',
                r'\b(brainstorm|innovative|original)\b'
            ],
            TaskType.ANALYSIS: [
                r'\b(analyze|examine|evaluate|assess|review)\b',
                r'\b(data|statistics|trends|patterns|insights)\b',
                r'\b(compare|contrast|summarize|conclude)\b',
                r'\b(report|findings|results|metrics)\b'
            ],
            TaskType.TRANSLATION: [
                r'\b(translate|translation|language|deutsch|français|español)\b',
                r'\b(chinese|japanese|korean|arabic|hindi|russian)\b',
                r'\b(from .+ to .+)\b',
                r'\b(in .+ language)\b'
            ],
            TaskType.QA: [
                r'\b(what|who|when|where|why|how)\b',
                r'\b(question|answer|explain|define|describe)\b',
                r'\?',
                r'\b(tell me|help me understand)\b'
            ],
            TaskType.RESEARCH: [
                r'\b(research|investigate|study|explore|survey)\b',
                r'\b(literature review|academic|scholarly|paper)\b',
                r'\b(sources|references|citations|bibliography)\b',
                r'\b(methodology|findings|conclusion)\b'
            ]
        }
    
    def classify_task(self, text: str, context: Dict = None) -> Tuple[TaskType, float]:
        """
        Classify input text into task type
        
        Returns:
            Tuple of (TaskType, confidence_score)
        """
        text_lower = text.lower()
        scores = defaultdict(float)
        
        # Pattern matching
        for task_type, patterns in self.task_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                scores[task_type] += matches * 0.2
        
        # Context analysis
        if context:
            if context.get('has_code', False):
                scores[TaskType.CODE] += 0.3
            if context.get('has_math', False):
                scores[TaskType.MATH] += 0.3
            if context.get('has_images', False):
                scores[TaskType.MULTIMODAL] += 0.5
        
        # Length analysis
        word_count = len(text.split())
        if word_count > 200:
            scores[TaskType.ANALYSIS] += 0.2
        if word_count < 20:
            scores[TaskType.QA] += 0.1
        
        # Default fallback
        if not scores:
            return TaskType.CHAT, 0.5
        
        # Return highest scoring task
        best_task = max(scores.items(), key=lambda x: x[1])
        return best_task[0], min(best_task[1], 1.0)

class PerformancePredictor:
    """Predicts model performance for specific tasks"""
    
    def __init__(self):
        self.task_model_history = defaultdict(lambda: defaultdict(list))
        self.model_load_tracking = defaultdict(lambda: deque(maxlen=100))
    
    def record_performance(self, model_id: str, task_type: TaskType, 
                         response_time: float, quality_score: float, success: bool):
        """Record model performance for learning"""
        self.task_model_history[task_type][model_id].append({
            'response_time': response_time,
            'quality_score': quality_score,
            'success': success,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        history = self.task_model_history[task_type][model_id]
        if len(history) > 50:
            self.task_model_history[task_type][model_id] = history[-50:]
    
    def predict_performance(self, model_id: str, task_type: TaskType) -> Dict[str, float]:
        """Predict model performance for a task"""
        history = self.task_model_history[task_type][model_id]
        
        if not history:
            # Default predictions for new model/task combinations
            return {
                'predicted_response_time': 5.0,
                'predicted_quality': 0.7,
                'predicted_success_rate': 0.9,
                'confidence': 0.1
            }
        
        recent_history = [h for h in history if time.time() - h['timestamp'] < 86400]  # Last 24h
        if not recent_history:
            recent_history = history[-10:]  # Fall back to last 10 records
        
        return {
            'predicted_response_time': statistics.mean([h['response_time'] for h in recent_history]),
            'predicted_quality': statistics.mean([h['quality_score'] for h in recent_history]),
            'predicted_success_rate': sum(1 for h in recent_history if h['success']) / len(recent_history),
            'confidence': min(len(recent_history) / 20, 1.0)
        }

class IntelligentRouter:
    """
    Intelligent model routing system that selects optimal models based on:
    - Task type and requirements
    - Model capabilities and performance
    - Real-time load and availability
    - Cost optimization
    - User preferences
    """
    
    def __init__(self, provider_manager):
        self.provider_manager = provider_manager
        self.classifier = TaskClassifier()
        self.predictor = PerformancePredictor()
        
        # Model profiles - updated with latest models
        self.model_profiles = self._initialize_model_profiles()
        
        # Routing statistics
        self.routing_stats = defaultdict(int)
        self.performance_history = deque(maxlen=1000)
        
        # Advanced features
        self.load_balancing_enabled = True
        self.cost_optimization_enabled = True
        self.quality_preference = 0.7  # 0 = cost focused, 1 = quality focused
        
    def _initialize_model_profiles(self) -> Dict[str, ModelProfile]:
        """Initialize comprehensive model profiles with latest models"""
        profiles = {}
        
        # OpenAI Models (Latest)
        profiles["openai:gpt-4o"] = ModelProfile(
            model_id="gpt-4o",
            provider="openai",
            capabilities={
                ModelCapability.SPEED: 0.9,
                ModelCapability.QUALITY: 0.98,
                ModelCapability.COST: 0.4,
                ModelCapability.CONTEXT_LENGTH: 0.95,
                ModelCapability.REASONING: 0.97,
                ModelCapability.CODE: 0.95,
                ModelCapability.MATH: 0.9,
                ModelCapability.MULTIMODAL: 0.95,
                ModelCapability.TOOLS: 0.98
            },
            avg_response_time=2.1,
            cost_per_token=0.000015,
            max_context_length=128000,
            supported_features=["function_calling", "vision", "json_mode", "structured_outputs"],
            recent_performance=deque(maxlen=100)
        )
        
        profiles["openai:gpt-4-turbo"] = ModelProfile(
            model_id="gpt-4-turbo",
            provider="openai",
            capabilities={
                ModelCapability.SPEED: 0.7,
                ModelCapability.QUALITY: 0.95,
                ModelCapability.COST: 0.3,
                ModelCapability.CONTEXT_LENGTH: 0.9,
                ModelCapability.REASONING: 0.95,
                ModelCapability.CODE: 0.9,
                ModelCapability.MATH: 0.85,
                ModelCapability.MULTIMODAL: 0.9,
                ModelCapability.TOOLS: 0.95
            },
            avg_response_time=3.5,
            cost_per_token=0.00003,
            max_context_length=128000,
            supported_features=["function_calling", "vision", "json_mode"],
            recent_performance=deque(maxlen=100)
        )
        
        profiles["openai:gpt-4o"] = ModelProfile(
            model_id="gpt-4o",
            provider="openai", 
            capabilities={
                ModelCapability.SPEED: 0.85,
                ModelCapability.QUALITY: 0.9,
                ModelCapability.COST: 0.4,
                ModelCapability.CONTEXT_LENGTH: 0.9,
                ModelCapability.REASONING: 0.9,
                ModelCapability.CODE: 0.85,
                ModelCapability.MATH: 0.8,
                ModelCapability.MULTIMODAL: 0.95,
                ModelCapability.TOOLS: 0.9
            },
            avg_response_time=2.0,
            cost_per_token=0.000015,
            max_context_length=128000,
            supported_features=["function_calling", "vision", "audio", "real_time"],
            recent_performance=deque(maxlen=100)
        )
        
        profiles["openai:o1-preview"] = ModelProfile(
            model_id="o1-preview",
            provider="openai",
            capabilities={
                ModelCapability.SPEED: 0.3,
                ModelCapability.QUALITY: 0.98,
                ModelCapability.COST: 0.1,
                ModelCapability.CONTEXT_LENGTH: 0.85,
                ModelCapability.REASONING: 0.99,
                ModelCapability.CODE: 0.95,
                ModelCapability.MATH: 0.98,
                ModelCapability.MULTIMODAL: 0.0,
                ModelCapability.TOOLS: 0.0
            },
            avg_response_time=15.0,
            cost_per_token=0.00015,
            max_context_length=32000,
            supported_features=["chain_of_thought"],
            recent_performance=deque(maxlen=100)
        )
        
        # Anthropic Models
        profiles["anthropic:claude-3.5-sonnet"] = ModelProfile(
            model_id="claude-3.5-sonnet",
            provider="anthropic",
            capabilities={
                ModelCapability.SPEED: 0.8,
                ModelCapability.QUALITY: 0.92,
                ModelCapability.COST: 0.5,
                ModelCapability.CONTEXT_LENGTH: 0.95,
                ModelCapability.REASONING: 0.9,
                ModelCapability.CODE: 0.9,
                ModelCapability.MATH: 0.85,
                ModelCapability.MULTIMODAL: 0.85,
                ModelCapability.TOOLS: 0.9
            },
            avg_response_time=2.5,
            cost_per_token=0.000015,
            max_context_length=200000,
            supported_features=["function_calling", "vision", "artifacts"],
            recent_performance=deque(maxlen=100)
        )
        
        # Google Models
        profiles["google:gemini-2.0-flash"] = ModelProfile(
            model_id="gemini-2.0-flash",
            provider="google",
            capabilities={
                ModelCapability.SPEED: 0.95,
                ModelCapability.QUALITY: 0.85,
                ModelCapability.COST: 0.8,
                ModelCapability.CONTEXT_LENGTH: 0.9,
                ModelCapability.REASONING: 0.8,
                ModelCapability.CODE: 0.8,
                ModelCapability.MATH: 0.8,
                ModelCapability.MULTIMODAL: 0.9,
                ModelCapability.TOOLS: 0.85
            },
            avg_response_time=1.0,
            cost_per_token=0.000001,
            max_context_length=1000000,
            supported_features=["function_calling", "vision", "audio", "real_time"],
            recent_performance=deque(maxlen=100)
        )
        
        # Cohere Models (R7 Series)
        profiles["cohere:command-r7"] = ModelProfile(
            model_id="command-r7",
            provider="cohere",
            capabilities={
                ModelCapability.SPEED: 0.8,
                ModelCapability.QUALITY: 0.88,
                ModelCapability.COST: 0.6,
                ModelCapability.CONTEXT_LENGTH: 0.9,
                ModelCapability.REASONING: 0.85,
                ModelCapability.CODE: 0.75,
                ModelCapability.MATH: 0.8,
                ModelCapability.MULTIMODAL: 0.0,
                ModelCapability.TOOLS: 0.9
            },
            avg_response_time=2.2,
            cost_per_token=0.000008,
            max_context_length=128000,
            supported_features=["function_calling", "rag_optimized", "enterprise"],
            recent_performance=deque(maxlen=100)
        )
        
        # Amazon Nova Models
        profiles["amazon:nova-pro"] = ModelProfile(
            model_id="nova-pro",
            provider="amazon",
            capabilities={
                ModelCapability.SPEED: 0.75,
                ModelCapability.QUALITY: 0.85,
                ModelCapability.COST: 0.7,
                ModelCapability.CONTEXT_LENGTH: 0.8,
                ModelCapability.REASONING: 0.8,
                ModelCapability.CODE: 0.8,
                ModelCapability.MATH: 0.75,
                ModelCapability.MULTIMODAL: 0.8,
                ModelCapability.TOOLS: 0.8
            },
            avg_response_time=3.0,
            cost_per_token=0.00001,
            max_context_length=100000,
            supported_features=["function_calling", "vision", "aws_native"],
            recent_performance=deque(maxlen=100)
        )
        
        profiles["amazon:nova-lite"] = ModelProfile(
            model_id="nova-lite",
            provider="amazon",
            capabilities={
                ModelCapability.SPEED: 0.9,
                ModelCapability.QUALITY: 0.7,
                ModelCapability.COST: 0.9,
                ModelCapability.CONTEXT_LENGTH: 0.7,
                ModelCapability.REASONING: 0.65,
                ModelCapability.CODE: 0.7,
                ModelCapability.MATH: 0.6,
                ModelCapability.MULTIMODAL: 0.7,
                ModelCapability.TOOLS: 0.75
            },
            avg_response_time=1.5,
            cost_per_token=0.000003,
            max_context_length=50000,
            supported_features=["function_calling", "vision", "aws_native"],
            recent_performance=deque(maxlen=100)
        )
        
        # Meta Models
        profiles["meta:llama-3.1-405b"] = ModelProfile(
            model_id="llama-3.1-405b",
            provider="meta",
            capabilities={
                ModelCapability.SPEED: 0.5,
                ModelCapability.QUALITY: 0.9,
                ModelCapability.COST: 0.95,  # Open source
                ModelCapability.CONTEXT_LENGTH: 0.85,
                ModelCapability.REASONING: 0.85,
                ModelCapability.CODE: 0.85,
                ModelCapability.MATH: 0.8,
                ModelCapability.MULTIMODAL: 0.0,
                ModelCapability.TOOLS: 0.8
            },
            avg_response_time=8.0,
            cost_per_token=0.0,  # Open source
            max_context_length=128000,
            supported_features=["function_calling", "open_source"],
            recent_performance=deque(maxlen=100)
        )
        
        # DeepSeek Models
        profiles["deepseek:v3"] = ModelProfile(
            model_id="deepseek-v3",
            provider="deepseek",
            capabilities={
                ModelCapability.SPEED: 0.7,
                ModelCapability.QUALITY: 0.88,
                ModelCapability.COST: 0.9,
                ModelCapability.CONTEXT_LENGTH: 0.9,
                ModelCapability.REASONING: 0.9,
                ModelCapability.CODE: 0.95,
                ModelCapability.MATH: 0.9,
                ModelCapability.MULTIMODAL: 0.0,
                ModelCapability.TOOLS: 0.85
            },
            avg_response_time=4.0,
            cost_per_token=0.000001,
            max_context_length=200000,
            supported_features=["function_calling", "reasoning_optimized"],
            recent_performance=deque(maxlen=100)
        )
        
        return profiles
    
    async def route_request(self, message: str, context: Dict = None, 
                          preferences: Dict = None, **kwargs) -> 'ModelInstance':
        """
        Intelligently route request to optimal model
        
        Args:
            message: Input message
            context: Session context
            preferences: User preferences (speed, quality, cost)
            **kwargs: Additional parameters
            
        Returns:
            Selected model instance
        """
        start_time = time.time()
        
        # 1. Classify the task
        task_type, confidence = self.classifier.classify_task(message, context)
        
        # 2. Get user preferences or defaults
        prefs = preferences or {}
        speed_weight = prefs.get('speed', 0.3)
        quality_weight = prefs.get('quality', 0.4)
        cost_weight = prefs.get('cost', 0.3)
        
        # 3. Score all available models
        model_scores = {}
        for model_id, profile in self.model_profiles.items():
            if not self._is_model_available(model_id):
                continue
                
            score = self._calculate_model_score(
                profile, task_type, speed_weight, quality_weight, cost_weight
            )
            model_scores[model_id] = score
        
        # 4. Apply performance predictions
        for model_id in model_scores:
            performance = self.predictor.predict_performance(model_id, task_type)
            # Adjust score based on predicted performance
            adjustment = (performance['predicted_success_rate'] - 0.5) * 0.2
            model_scores[model_id] += adjustment
        
        # 5. Apply load balancing
        if self.load_balancing_enabled:
            for model_id in model_scores:
                load_factor = self.model_profiles[model_id].load_factor
                model_scores[model_id] *= (1.0 - load_factor * 0.3)
        
        # 6. Select best model
        if not model_scores:
            # Fallback to default model
            selected_model = "openai:gpt-4o"
        else:
            selected_model = max(model_scores.items(), key=lambda x: x[1])[0]
        
        # 7. Record routing decision
        self.routing_stats[selected_model] += 1
        routing_time = time.time() - start_time
        
        self.performance_history.append({
            'timestamp': start_time,
            'task_type': task_type.value,
            'selected_model': selected_model,
            'routing_time': routing_time,
            'confidence': confidence,
            'scores': model_scores
        })
        
        # 8. Get model instance
        return await self.provider_manager.get_model_instance(selected_model)
    
    def _calculate_model_score(self, profile: ModelProfile, task_type: TaskType,
                             speed_weight: float, quality_weight: float, 
                             cost_weight: float) -> float:
        """Calculate composite score for a model given task and preferences"""
        
        # Base capability scores for the task
        task_capability_map = {
            TaskType.CODE: [ModelCapability.CODE, ModelCapability.REASONING],
            TaskType.MATH: [ModelCapability.MATH, ModelCapability.REASONING],
            TaskType.REASONING: [ModelCapability.REASONING, ModelCapability.QUALITY],
            TaskType.CREATIVE: [ModelCapability.QUALITY],
            TaskType.ANALYSIS: [ModelCapability.REASONING, ModelCapability.QUALITY],
            TaskType.TRANSLATION: [ModelCapability.LANGUAGES, ModelCapability.QUALITY],
            TaskType.QA: [ModelCapability.QUALITY, ModelCapability.SPEED],
            TaskType.MULTIMODAL: [ModelCapability.MULTIMODAL, ModelCapability.QUALITY],
            TaskType.TOOL_USE: [ModelCapability.TOOLS, ModelCapability.REASONING],
            TaskType.RESEARCH: [ModelCapability.REASONING, ModelCapability.CONTEXT_LENGTH]
        }
        
        # Calculate task-specific capability score
        relevant_capabilities = task_capability_map.get(task_type, [ModelCapability.QUALITY])
        capability_score = statistics.mean([
            profile.capabilities.get(cap, 0.5) for cap in relevant_capabilities
        ])
        
        # Performance scores
        speed_score = 1.0 / (1.0 + profile.avg_response_time / 5.0)  # Normalize around 5s
        quality_score = profile.capabilities.get(ModelCapability.QUALITY, 0.5)
        cost_score = profile.capabilities.get(ModelCapability.COST, 0.5)
        
        # Composite score
        composite_score = (
            capability_score * 0.4 +
            speed_score * speed_weight +
            quality_score * quality_weight +
            cost_score * cost_weight
        )
        
        # Apply success rate multiplier
        composite_score *= profile.success_rate
        
        return composite_score
    
    def _is_model_available(self, model_id: str) -> bool:
        """Check if model is currently available"""
        return self.provider_manager.is_model_available(model_id)
    
    def should_use_reasoning(self, message: str) -> bool:
        """Determine if enhanced reasoning should be applied"""
        reasoning_indicators = [
            r'\b(think step by step|analyze|reasoning|logic|solve)\b',
            r'\b(why|how|explain|because)\b',
            r'\b(problem|solution|approach|method)\b'
        ]
        
        text_lower = message.lower()
        for pattern in reasoning_indicators:
            if re.search(pattern, text_lower):
                return True
        
        # Complex mathematical expressions
        if re.search(r'[0-9]+\s*[\+\-\*\/\=\^]\s*[0-9]+', message):
            return True
        
        # Long, complex queries
        if len(message.split()) > 50:
            return True
        
        return False
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        total_requests = sum(self.routing_stats.values())
        
        if total_requests == 0:
            return {"total_requests": 0}
        
        # Model usage distribution
        model_distribution = {
            model: count / total_requests 
            for model, count in self.routing_stats.items()
        }
        
        # Recent performance metrics
        recent_history = [h for h in self.performance_history 
                         if time.time() - h['timestamp'] < 3600]  # Last hour
        
        avg_routing_time = statistics.mean([h['routing_time'] for h in recent_history]) if recent_history else 0
        
        # Task type distribution
        task_distribution = defaultdict(int)
        for h in recent_history:
            task_distribution[h['task_type']] += 1
        
        task_percentages = {
            task: count / len(recent_history) 
            for task, count in task_distribution.items()
        } if recent_history else {}
        
        return {
            "total_requests": total_requests,
            "model_distribution": model_distribution,
            "avg_routing_time": avg_routing_time,
            "task_distribution": task_percentages,
            "recent_requests": len(recent_history),
            "available_models": len([m for m in self.model_profiles.keys() 
                                   if self._is_model_available(m)])
        }
    
    def update_model_performance(self, model_id: str, task_type: TaskType,
                               response_time: float, quality_score: float, 
                               success: bool):
        """Update model performance data for learning"""
        if model_id in self.model_profiles:
            profile = self.model_profiles[model_id]
            
            # Update response time (exponential moving average)
            alpha = 0.1
            profile.avg_response_time = (
                alpha * response_time + 
                (1 - alpha) * profile.avg_response_time
            )
            
            # Update success rate
            profile.recent_performance.append(success)
            if len(profile.recent_performance) >= 10:
                profile.success_rate = sum(profile.recent_performance) / len(profile.recent_performance)
        
        # Update predictor
        self.predictor.record_performance(model_id, task_type, response_time, quality_score, success)
    
    def get_supported_models(self) -> List[Dict[str, Any]]:
        """Get list of all supported models with their capabilities"""
        models = []
        for model_id, profile in self.model_profiles.items():
            models.append({
                "model_id": profile.model_id,
                "provider": profile.provider,
                "capabilities": {cap.value: score for cap, score in profile.capabilities.items()},
                "avg_response_time": profile.avg_response_time,
                "cost_per_token": profile.cost_per_token,
                "max_context_length": profile.max_context_length,
                "supported_features": profile.supported_features,
                "success_rate": profile.success_rate,
                "available": self._is_model_available(model_id)
            })
        return models
    
    async def compare_models(self, prompt: str, models: List[str] = None, 
                           task_type: TaskType = None) -> Dict[str, Any]:
        """
        Compare multiple models for the same prompt
        
        Args:
            prompt: The input prompt to test
            models: List of model IDs to compare (uses top 3 if None)
            task_type: Specific task type (auto-detected if None)
            
        Returns:
            Comprehensive comparison results
        """
        if task_type is None:
            task_type, _ = self.classifier.classify_task(prompt)
        
        if models is None:
            # Auto-select top 3 models for this task
            model_scores = []
            for model_id in self.model_profiles.keys():
                if self._is_model_available(model_id):
                    score = self._calculate_model_score(model_id, task_type, {})
                    model_scores.append((model_id, score))
            
            model_scores.sort(key=lambda x: x[1], reverse=True)
            models = [model_id for model_id, _ in model_scores[:3]]
        
        comparison_results = {
            "prompt": prompt,
            "task_type": task_type.value,
            "models_compared": len(models),
            "results": {},
            "winner": None,
            "comparison_metrics": {}
        }
        
        model_results = []
        
        # Test each model
        for model_id in models:
            start_time = time.time()
            
            try:
                # Get provider and make request
                provider_name = model_id.split(':')[0] if ':' in model_id else 'openai'
                provider = self.provider_manager.get_provider(provider_name)
                
                if provider:
                    response = await provider.generate_async(
                        prompt=prompt,
                        model=model_id.split(':')[-1],
                        max_tokens=1000
                    )
                    
                    response_time = time.time() - start_time
                    
                    # Calculate quality metrics
                    quality_score = self._calculate_response_quality(prompt, response, task_type)
                    
                    result = {
                        "model_id": model_id,
                        "response": response,
                        "response_time": response_time,
                        "quality_score": quality_score,
                        "success": True,
                        "error": None,
                        "token_count": len(response.split()) * 1.3,  # Rough estimate
                        "cost_estimate": self._estimate_cost(model_id, prompt, response)
                    }
                    
                else:
                    result = {
                        "model_id": model_id,
                        "response": None,
                        "response_time": 0,
                        "quality_score": 0,
                        "success": False,
                        "error": f"Provider {provider_name} not available",
                        "token_count": 0,
                        "cost_estimate": 0
                    }
                    
            except Exception as e:
                result = {
                    "model_id": model_id,
                    "response": None,
                    "response_time": time.time() - start_time,
                    "quality_score": 0,
                    "success": False,
                    "error": str(e),
                    "token_count": 0,
                    "cost_estimate": 0
                }
            
            model_results.append(result)
            comparison_results["results"][model_id] = result
        
        # Determine winner and calculate metrics
        successful_results = [r for r in model_results if r["success"]]
        
        if successful_results:
            # Multi-criteria scoring
            for result in successful_results:
                score = (
                    result["quality_score"] * 0.4 +
                    (1 / max(result["response_time"], 0.1)) * 0.3 +
                    (1 / max(result["cost_estimate"], 0.001)) * 0.3
                )
                result["overall_score"] = score
            
            winner = max(successful_results, key=lambda x: x["overall_score"])
            comparison_results["winner"] = winner["model_id"]
            
            # Calculate comparison metrics
            comparison_results["comparison_metrics"] = {
                "avg_response_time": statistics.mean([r["response_time"] for r in successful_results]),
                "avg_quality_score": statistics.mean([r["quality_score"] for r in successful_results]),
                "total_cost": sum([r["cost_estimate"] for r in successful_results]),
                "success_rate": len(successful_results) / len(model_results),
                "speed_winner": min(successful_results, key=lambda x: x["response_time"])["model_id"],
                "quality_winner": max(successful_results, key=lambda x: x["quality_score"])["model_id"],
                "cost_winner": min(successful_results, key=lambda x: x["cost_estimate"])["model_id"]
            }
        
        return comparison_results
    
    def _calculate_response_quality(self, prompt: str, response: str, task_type: TaskType) -> float:
        """Calculate response quality score based on task type"""
        if not response:
            return 0.0
        
        quality_score = 0.5  # Base score
        
        # Length appropriateness
        response_length = len(response.split())
        prompt_length = len(prompt.split())
        
        if task_type == TaskType.CODE:
            # Look for code indicators
            if '```' in response or 'def ' in response or 'function' in response:
                quality_score += 0.2
            if 'import ' in response or '#' in response:
                quality_score += 0.1
                
        elif task_type == TaskType.MATH:
            # Look for mathematical content
            if any(char in response for char in '=+-*/^()'):
                quality_score += 0.2
            if any(word in response.lower() for word in ['solution', 'answer', 'result']):
                quality_score += 0.1
                
        elif task_type == TaskType.REASONING:
            # Look for reasoning indicators
            reasoning_words = ['because', 'therefore', 'however', 'first', 'second', 'conclusion']
            found_reasoning = sum(1 for word in reasoning_words if word in response.lower())
            quality_score += min(found_reasoning * 0.05, 0.2)
            
        elif task_type == TaskType.CREATIVE:
            # Creativity indicators
            if response_length > prompt_length * 2:
                quality_score += 0.2
            if any(word in response.lower() for word in ['imagine', 'story', 'character']):
                quality_score += 0.1
        
        # Response completeness
        if response_length > 20:
            quality_score += 0.1
        if response_length > 50:
            quality_score += 0.1
            
        return min(quality_score, 1.0)
    
    def _estimate_cost(self, model_id: str, prompt: str, response: str) -> float:
        """Estimate cost for the request"""
        if model_id not in self.model_profiles:
            return 0.001  # Default estimate
            
        profile = self.model_profiles[model_id]
        
        # Rough token estimation
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = len(response.split()) * 1.3
        total_tokens = input_tokens + output_tokens
        
        return total_tokens * profile.cost_per_token
    
    async def auto_select_model(self, prompt: str, preferences: Dict = None) -> str:
        """
        Automatically select the best model for a prompt
        
        Args:
            prompt: Input prompt
            preferences: User preferences (quality_priority, speed_priority, cost_priority)
            
        Returns:
            Selected model ID
        """
        # Classify task
        task_type, confidence = self.classifier.classify_task(prompt)
        
        # Default preferences
        prefs = {
            'quality_priority': 0.4,
            'speed_priority': 0.3,
            'cost_priority': 0.3
        }
        if preferences:
            prefs.update(preferences)
        
        # Score all available models
        model_scores = []
        for model_id in self.model_profiles.keys():
            if self._is_model_available(model_id):
                score = self._calculate_model_score_with_preferences(
                    model_id, task_type, prefs
                )
                model_scores.append((model_id, score))
        
        if not model_scores:
            return "openai:gpt-4o"  # Fallback
            
        # Return best model
        model_scores.sort(key=lambda x: x[1], reverse=True)
        selected_model = model_scores[0][0]
        
        # Record selection
        self.routing_stats[selected_model] += 1
        
        return selected_model
    
    def _calculate_model_score_with_preferences(self, model_id: str, task_type: TaskType, 
                                              preferences: Dict) -> float:
        """Calculate model score with user preferences"""
        if model_id not in self.model_profiles:
            return 0.0
            
        profile = self.model_profiles[model_id]
        
        # Get task-specific capability
        task_capability_map = {
            TaskType.CODE: ModelCapability.CODE,
            TaskType.MATH: ModelCapability.MATH,
            TaskType.REASONING: ModelCapability.REASONING,
            TaskType.CREATIVE: ModelCapability.QUALITY,
            TaskType.ANALYSIS: ModelCapability.REASONING,
            TaskType.MULTIMODAL: ModelCapability.MULTIMODAL,
            TaskType.TOOL_USE: ModelCapability.TOOLS
        }
        
        primary_capability = task_capability_map.get(task_type, ModelCapability.QUALITY)
        task_score = profile.capabilities.get(primary_capability, 0.5)
        
        # Calculate weighted score
        quality_score = profile.capabilities.get(ModelCapability.QUALITY, 0.5)
        speed_score = profile.capabilities.get(ModelCapability.SPEED, 0.5)
        cost_score = profile.capabilities.get(ModelCapability.COST, 0.5)
        
        # Apply user preferences
        final_score = (
            task_score * 0.4 +  # Task-specific capability
            quality_score * preferences['quality_priority'] +
            speed_score * preferences['speed_priority'] +
            cost_score * preferences['cost_priority']
        )
        
        # Apply success rate and load factors
        final_score *= profile.success_rate
        final_score *= (1 - profile.load_factor * 0.2)  # Penalize high load
        
        return final_score
