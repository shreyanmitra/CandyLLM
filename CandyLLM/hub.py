"""
CandyLLM Hub - Main Interface for Advanced AI Capabilities

This module provides the main interface for CandyLLM's advanced AI features including:
- Intelligent model routing and selection
- Neurosymbolic reasoning engine
- Mathematical computation and symbolic reasoning
- Knowledge graph operations
- Multi-modal AI capabilities

Author: CandyLLM Team
License: MIT
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json

from .core.router import IntelligentRouter
from .core.neurosymbolic import create_neurosymbolic_engine, NeuroSymbolicEngine
from .providers.base import BaseProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CandyLLM:
    """
    Main CandyLLM interface providing access to all advanced AI capabilities.
    
    Features:
    - Intelligent model routing and auto-selection
    - Neurosymbolic reasoning with knowledge graphs
    - Mathematical problem solving
    - Multi-modal AI integration
    - Performance analytics and optimization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize CandyLLM with configuration.
        
        Args:
            config: Configuration dictionary with settings for various components
        """
        self.config = config or {}
        self.providers = {}
        self.router = None
        self.neurosymbolic_engine = None
        
        # Performance tracking
        self.session_stats = {
            'start_time': datetime.now(),
            'total_queries': 0,
            'successful_queries': 0,
            'reasoning_operations': 0,
            'knowledge_items_added': 0
        }
        
        logger.info("Initializing CandyLLM Hub with advanced AI capabilities")
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize core components."""
        try:
            # Initialize router
            router_config = self.config.get('router', {})
            self.router = IntelligentRouter(
                providers=self.providers,
                default_provider=router_config.get('default_provider'),
                enable_caching=router_config.get('enable_caching', True),
                cache_ttl=router_config.get('cache_ttl', 3600)
            )
            
            # Initialize neurosymbolic engine
            neurosymbolic_config = self.config.get('neurosymbolic', {})
            self.neurosymbolic_engine = create_neurosymbolic_engine(
                provider_manager=self,
                config=neurosymbolic_config
            )
            
            logger.info("All CandyLLM components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def add_provider(self, name: str, provider: BaseProvider):
        """
        Add an AI provider to the system.
        
        Args:
            name: Unique name for the provider
            provider: Provider instance implementing BaseProvider interface
        """
        self.providers[name] = provider
        if self.router:
            self.router.providers = self.providers
        logger.info(f"Added provider: {name}")
    
    async def query(self, text: str, provider: str = None, **kwargs) -> Dict[str, Any]:
        """
        Send a query to an AI provider with intelligent routing.
        
        Args:
            text: Query text
            provider: Specific provider to use (optional, auto-selected if None)
            **kwargs: Additional parameters for the provider
            
        Returns:
            Response from the AI provider with metadata
        """
        self.session_stats['total_queries'] += 1
        start_time = datetime.now()
        
        try:
            if provider:
                # Use specific provider
                if provider not in self.providers:
                    raise ValueError(f"Provider '{provider}' not found")
                
                response = await self.providers[provider].generate(text, **kwargs)
            else:
                # Use intelligent routing
                if not self.router:
                    raise RuntimeError("Router not initialized")
                
                response = await self.router.route_query(text, **kwargs)
            
            # Add metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            response['metadata'] = {
                'processing_time': processing_time,
                'timestamp': start_time.isoformat(),
                'provider_used': response.get('provider', provider),
                'session_query_count': self.session_stats['total_queries']
            }
            
            self.session_stats['successful_queries'] += 1
            return response
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'error': str(e),
                'success': False,
                'metadata': {
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'timestamp': start_time.isoformat(),
                    'session_query_count': self.session_stats['total_queries']
                }
            }
    
    async def reason(self, query: str, reasoning_type: str = "auto", 
                    context: Dict = None, max_depth: int = 5) -> Dict[str, Any]:
        """
        Perform advanced neurosymbolic reasoning on a query.
        
        Args:
            query: Question or problem to reason about
            reasoning_type: Type of reasoning ("auto", "mathematical", "logical", etc.)
            context: Additional context for reasoning
            max_depth: Maximum depth of reasoning chain
            
        Returns:
            Reasoning result with steps and conclusions
        """
        if not self.neurosymbolic_engine:
            raise RuntimeError("Neurosymbolic engine not initialized")
        
        self.session_stats['reasoning_operations'] += 1
        start_time = datetime.now()
        
        try:
            reasoning_chain = await self.neurosymbolic_engine.reason(
                query=query,
                reasoning_type=reasoning_type,
                context=context,
                max_depth=max_depth
            )
            
            # Convert to dictionary format
            result = {
                'success': True,
                'query': query,
                'reasoning_type': reasoning_chain.reasoning_type.value,
                'conclusion': reasoning_chain.final_conclusion,
                'confidence': reasoning_chain.overall_confidence,
                'steps': len(reasoning_chain.steps),
                'reasoning_steps': [
                    {
                        'step_number': step.step_number,
                        'explanation': step.explanation,
                        'confidence': step.confidence,
                        'operation': step.logical_operation
                    }
                    for step in reasoning_chain.steps
                ],
                'knowledge_sources': reasoning_chain.knowledge_sources,
                'audit_trail': reasoning_chain.get_audit_trail(),
                'metadata': {
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'timestamp': start_time.isoformat()
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'metadata': {
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'timestamp': start_time.isoformat()
                }
            }
    
    async def solve_math(self, problem: str, context: Dict = None) -> Dict[str, Any]:
        """
        Solve mathematical problems using symbolic computation.
        
        Args:
            problem: Mathematical problem or equation
            context: Additional context or constraints
            
        Returns:
            Mathematical solution with steps
        """
        if not self.neurosymbolic_engine:
            raise RuntimeError("Neurosymbolic engine not initialized")
        
        try:
            result = await self.neurosymbolic_engine.mathematical_reasoner.solve_mathematical_problem(
                problem, context
            )
            
            return {
                'success': True,
                'problem': problem,
                'solution': result.get('final_answer'),
                'steps': result.get('steps', []),
                'confidence': result.get('confidence', 1.0),
                'symbolic_form': result.get('symbolic_expression'),
                'metadata': {
                    'solver_used': result.get('solver_type', 'symbolic'),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Mathematical solving failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'problem': problem
            }
    
    def add_knowledge(self, subject: str, predicate: str, object: str, 
                     confidence: float = 1.0, source: str = "user") -> bool:
        """
        Add knowledge to the knowledge graph.
        
        Args:
            subject: Subject of the knowledge triple
            predicate: Relationship/predicate
            object: Object of the knowledge triple
            confidence: Confidence level (0.0 to 1.0)
            source: Source of the knowledge
            
        Returns:
            True if knowledge was added successfully
        """
        if not self.neurosymbolic_engine:
            logger.warning("Neurosymbolic engine not initialized")
            return False
        
        success = self.neurosymbolic_engine.add_knowledge(
            subject, predicate, object, confidence, source
        )
        
        if success:
            self.session_stats['knowledge_items_added'] += 1
        
        return success
    
    async def compare_models(self, query: str, providers: List[str] = None, 
                           criteria: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Compare multiple AI models on a given query.
        
        Args:
            query: Query to test models with
            providers: List of provider names to compare (all if None)
            criteria: Evaluation criteria weights
            
        Returns:
            Comparison results with scores and recommendations
        """
        if not self.router:
            raise RuntimeError("Router not initialized")
        
        try:
            comparison = await self.router.compare_models(
                query=query,
                providers=providers,
                criteria=criteria
            )
            
            return {
                'success': True,
                'query': query,
                'comparison': comparison,
                'recommendation': comparison.get('recommended_model'),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'providers_tested': len(comparison.get('results', {}))
                }
            }
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query
            }
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        current_time = datetime.now()
        session_duration = (current_time - self.session_stats['start_time']).total_seconds()
        
        stats = self.session_stats.copy()
        stats['session_duration'] = session_duration
        stats['current_time'] = current_time.isoformat()
        stats['success_rate'] = (
            stats['successful_queries'] / max(stats['total_queries'], 1)
        )
        
        # Add component stats
        if self.router:
            stats['router_stats'] = self.router.get_performance_metrics()
        
        if self.neurosymbolic_engine:
            stats['neurosymbolic_stats'] = self.neurosymbolic_engine.get_performance_summary()
        
        return stats
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self.providers.keys())
    
    def get_supported_reasoning_types(self) -> List[str]:
        """Get list of supported reasoning types."""
        return [
            "auto", "mathematical", "logical", "causal", 
            "deductive", "inductive", "abductive"
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        # Check providers
        for name, provider in self.providers.items():
            try:
                # Simple test query
                test_response = await provider.generate("test", max_tokens=1)
                health['components'][f'provider_{name}'] = 'healthy'
            except Exception as e:
                health['components'][f'provider_{name}'] = f'error: {e}'
                health['overall_status'] = 'degraded'
        
        # Check router
        if self.router:
            health['components']['router'] = 'healthy'
        else:
            health['components']['router'] = 'not_initialized'
            health['overall_status'] = 'degraded'
        
        # Check neurosymbolic engine
        if self.neurosymbolic_engine:
            health['components']['neurosymbolic_engine'] = 'healthy'
        else:
            health['components']['neurosymbolic_engine'] = 'not_initialized'
            health['overall_status'] = 'degraded'
        
        return health


# Convenience functions for quick access

async def quick_query(text: str, provider: str = None, config: Dict = None) -> str:
    """
    Quick query function for simple use cases.
    
    Args:
        text: Query text
        provider: Specific provider to use
        config: Configuration for CandyLLM
        
    Returns:
        Response text
    """
    candy = CandyLLM(config)
    response = await candy.query(text, provider)
    return response.get('content', response.get('error', 'No response'))


async def quick_reason(query: str, reasoning_type: str = "auto", config: Dict = None) -> str:
    """
    Quick reasoning function for simple use cases.
    
    Args:
        query: Question to reason about
        reasoning_type: Type of reasoning to apply
        config: Configuration for CandyLLM
        
    Returns:
        Reasoning conclusion
    """
    candy = CandyLLM(config)
    response = await candy.reason(query, reasoning_type)
    return response.get('conclusion', response.get('error', 'No conclusion'))


async def quick_math(problem: str, config: Dict = None) -> str:
    """
    Quick math solving function for simple use cases.
    
    Args:
        problem: Mathematical problem
        config: Configuration for CandyLLM
        
    Returns:
        Mathematical solution
    """
    candy = CandyLLM(config)
    response = await candy.solve_math(problem)
    return response.get('solution', response.get('error', 'No solution'))


if __name__ == "__main__":
    """
    Demo of CandyLLM capabilities.
    """
    async def demo():
        print("🍭 CandyLLM Advanced AI Hub Demo")
        print("=" * 40)
        
        # Initialize CandyLLM
        config = {
            'neurosymbolic': {
                'max_knowledge_items': 1000,
                'initial_knowledge': [
                    {
                        'subject': 'Python',
                        'predicate': 'is_language_for',
                        'object': 'AI_development',
                        'confidence': 0.9
                    }
                ]
            }
        }
        
        candy = CandyLLM(config)
        
        # Demo reasoning
        reasoning_result = await candy.reason(
            "What is Python good for?", 
            reasoning_type="logical"
        )
        print(f"🧠 Reasoning: {reasoning_result['conclusion']}")
        
        # Demo math solving
        math_result = await candy.solve_math("x^2 + 4x + 4 = 0")
        print(f"🧮 Math: {math_result.get('solution', 'No solution')}")
        
        # Demo knowledge addition
        candy.add_knowledge("CandyLLM", "enables", "advanced_AI", confidence=0.95)
        
        # Show stats
        stats = candy.get_session_stats()
        print(f"📊 Session: {stats['total_queries']} queries, {stats['reasoning_operations']} reasoning ops")
        
        print("✅ Demo completed!")
    
    asyncio.run(demo())
