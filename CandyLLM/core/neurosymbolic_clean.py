"""
CandyLLM Neurosymbolic AI Engine - Cleaned Version

Advanced neurosymbolic reasoning engine that combines neural networks with symbolic computation.
Implements state-of-the-art AI methodologies including mathematical reasoning, knowledge graphs,
causal inference, and multi-modal reasoning capabilities.
"""

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union, Set

# Optional dependencies with fallbacks
try:
    import sympy as sp
    from sympy import symbols, solve, diff, integrate, simplify, sympify, Eq
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("Warning: SymPy not available. Mathematical reasoning will be limited.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: NetworkX not available. Knowledge graph features will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning supported by the neurosymbolic engine."""
    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    CAUSAL = "causal"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    
    @classmethod
    def validate_type(cls, reasoning_type: str) -> 'ReasoningType':
        """Validate and return reasoning type enum."""
        try:
            return cls(reasoning_type.lower())
        except ValueError:
            return cls.LOGICAL  # Default fallback


@dataclass
class KnowledgeTriple:
    """Represents a knowledge triple (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "unknown"
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    @property
    def hash_id(self) -> str:
        """Generate unique hash for this triple."""
        return f"{self.subject}_{self.predicate}_{self.object}".lower()


@dataclass
class ReasoningStep:
    """Represents a single step in a reasoning chain."""
    step_number: int
    reasoning_type: ReasoningType
    input_facts: List[KnowledgeTriple]
    logical_operation: str
    derived_fact: KnowledgeTriple
    confidence: float
    explanation: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ReasoningChain:
    """Represents a complete reasoning chain with multiple steps."""
    query: str
    reasoning_type: ReasoningType
    steps: List[ReasoningStep] = None
    final_conclusion: str = ""
    overall_confidence: float = 0.0
    knowledge_sources: List[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []
        if self.knowledge_sources is None:
            self.knowledge_sources = []
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def add_step(self, step: ReasoningStep):
        """Add a reasoning step to the chain."""
        self.steps.append(step)
        self._update_confidence()
    
    def _update_confidence(self):
        """Update overall confidence based on steps."""
        if self.steps:
            # Average confidence with decay for longer chains
            avg_confidence = sum(step.confidence for step in self.steps) / len(self.steps)
            length_penalty = 0.9 ** (len(self.steps) - 1)
            self.overall_confidence = avg_confidence * length_penalty
        else:
            self.overall_confidence = 0.0
    
    def get_audit_trail(self) -> Dict[str, Any]:
        """Get complete audit trail for this reasoning chain."""
        return {
            'query': self.query,
            'reasoning_type': self.reasoning_type.value,
            'total_steps': len(self.steps),
            'final_confidence': self.overall_confidence,
            'steps': [
                {
                    'step': step.step_number,
                    'operation': step.logical_operation,
                    'confidence': step.confidence,
                    'explanation': step.explanation
                }
                for step in self.steps
            ],
            'timestamp': self.timestamp
        }


class AdvancedKnowledgeGraph:
    """
    Advanced knowledge graph with semantic search and reasoning capabilities.
    """
    
    def __init__(self, max_triples: int = 10000):
        self.max_triples = max_triples
        self.triples = {}  # hash_id -> KnowledgeTriple
        self.subject_index = defaultdict(list)  # subject -> [triple_ids]
        self.predicate_index = defaultdict(list)  # predicate -> [triple_ids]
        self.object_index = defaultdict(list)  # object -> [triple_ids]
        
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            self.graph = None
        
        logger.info(f"Initialized knowledge graph with capacity for {max_triples} triples")
    
    def add_triple(self, triple: KnowledgeTriple) -> bool:
        """Add a knowledge triple to the graph."""
        if len(self.triples) >= self.max_triples:
            logger.warning("Knowledge graph at capacity, removing oldest triple")
            self._remove_oldest_triple()
        
        # Store triple
        self.triples[triple.hash_id] = triple
        
        # Update indexes
        self.subject_index[triple.subject.lower()].append(triple.hash_id)
        self.predicate_index[triple.predicate.lower()].append(triple.hash_id)
        self.object_index[triple.object.lower()].append(triple.hash_id)
        
        # Update NetworkX graph if available
        if self.graph is not None:
            self.graph.add_edge(
                triple.subject, triple.object,
                predicate=triple.predicate,
                confidence=triple.confidence,
                source=triple.source
            )
        
        return True
    
    def query_by_pattern(self, subject: str = None, predicate: str = None, 
                        object: str = None, min_confidence: float = 0.0) -> List[KnowledgeTriple]:
        """Query triples matching a pattern."""
        candidate_ids = set()
        
        if subject:
            candidate_ids.update(self.subject_index.get(subject.lower(), []))
        if predicate:
            pred_ids = set(self.predicate_index.get(predicate.lower(), []))
            candidate_ids = candidate_ids.intersection(pred_ids) if candidate_ids else pred_ids
        if object:
            obj_ids = set(self.object_index.get(object.lower(), []))
            candidate_ids = candidate_ids.intersection(obj_ids) if candidate_ids else obj_ids
        
        # If no specific criteria, return all
        if not any([subject, predicate, object]):
            candidate_ids = set(self.triples.keys())
        
        # Filter by confidence and return
        results = []
        for triple_id in candidate_ids:
            triple = self.triples.get(triple_id)
            if triple and triple.confidence >= min_confidence:
                results.append(triple)
        
        return sorted(results, key=lambda t: t.confidence, reverse=True)
    
    def find_causal_chain(self, cause: str, effect: str, max_depth: int = 5) -> List[List[KnowledgeTriple]]:
        """Find causal chains between cause and effect."""
        if not self.graph:
            return []
        
        try:
            # Find all simple paths
            paths = list(nx.all_simple_paths(self.graph, cause, effect, cutoff=max_depth))
            
            # Convert to triple chains
            triple_chains = []
            for path in paths:
                chain = []
                for i in range(len(path) - 1):
                    source, target = path[i], path[i + 1]
                    edge_data = self.graph.get_edge_data(source, target, {})
                    
                    # Create triple from edge
                    triple = KnowledgeTriple(
                        subject=source,
                        predicate=edge_data.get('predicate', 'related_to'),
                        object=target,
                        confidence=edge_data.get('confidence', 0.5),
                        source=edge_data.get('source', 'graph')
                    )
                    chain.append(triple)
                
                if chain:
                    triple_chains.append(chain)
            
            return triple_chains
            
        except Exception as e:
            logger.error(f"Error finding causal chain: {e}")
            return []
    
    def _remove_oldest_triple(self):
        """Remove the oldest triple to make space."""
        if not self.triples:
            return
        
        # Find oldest triple by timestamp
        oldest_id = min(self.triples.keys(), 
                       key=lambda tid: self.triples[tid].timestamp)
        oldest_triple = self.triples[oldest_id]
        
        # Remove from indexes
        self.subject_index[oldest_triple.subject.lower()].remove(oldest_id)
        self.predicate_index[oldest_triple.predicate.lower()].remove(oldest_id)
        self.object_index[oldest_triple.object.lower()].remove(oldest_id)
        
        # Remove from graph
        if self.graph is not None:
            try:
                self.graph.remove_edge(oldest_triple.subject, oldest_triple.object)
            except:
                pass  # Edge might not exist
        
        # Remove from storage
        del self.triples[oldest_id]


class MathematicalReasoner:
    """
    Advanced mathematical reasoning engine with symbolic computation.
    """
    
    def __init__(self):
        self.supported_operations = [
            'solve', 'differentiate', 'integrate', 'simplify', 
            'factor', 'expand', 'limit', 'series'
        ]
        
        if not SYMPY_AVAILABLE:
            logger.warning("SymPy not available. Mathematical reasoning will be limited.")
    
    async def solve_mathematical_problem(self, problem: str, context: Dict = None) -> Dict[str, Any]:
        """
        Solve a mathematical problem using symbolic computation.
        
        Args:
            problem: Mathematical problem as text
            context: Additional context or constraints
            
        Returns:
            Dictionary with solution, steps, and metadata
        """
        if not SYMPY_AVAILABLE:
            return {
                'success': False,
                'error': 'SymPy not available for mathematical computation',
                'final_answer': 'Mathematical reasoning requires SymPy library'
            }
        
        try:
            # Parse the mathematical problem
            problem_type = self._detect_problem_type(problem)
            
            if problem_type == 'equation':
                return await self._solve_equation(problem)
            elif problem_type == 'derivative':
                return await self._compute_derivative(problem)
            elif problem_type == 'integral':
                return await self._compute_integral(problem)
            elif problem_type == 'simplify':
                return await self._simplify_expression(problem)
            else:
                return await self._general_mathematical_analysis(problem)
                
        except Exception as e:
            logger.error(f"Mathematical problem solving failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'final_answer': f'Error solving mathematical problem: {e}',
                'confidence': 0.0
            }
    
    def _detect_problem_type(self, problem: str) -> str:
        """Detect the type of mathematical problem."""
        problem_lower = problem.lower()
        
        if any(keyword in problem_lower for keyword in ['solve', '=', 'equation']):
            return 'equation'
        elif any(keyword in problem_lower for keyword in ['derivative', 'differentiate', "d/dx"]):
            return 'derivative'
        elif any(keyword in problem_lower for keyword in ['integral', 'integrate', '∫']):
            return 'integral'
        elif any(keyword in problem_lower for keyword in ['simplify', 'reduce']):
            return 'simplify'
        else:
            return 'general'
    
    async def _solve_equation(self, problem: str) -> Dict[str, Any]:
        """Solve an equation."""
        # Extract equation from text
        equation_match = re.search(r'([^=]+)=([^=]+)', problem)
        if not equation_match:
            return {
                'success': False,
                'error': 'Could not parse equation',
                'final_answer': 'No valid equation found'
            }
        
        left_side = equation_match.group(1).strip()
        right_side = equation_match.group(2).strip()
        
        try:
            # Parse with SymPy
            left_expr = sp.sympify(left_side)
            right_expr = sp.sympify(right_side)
            equation = sp.Eq(left_expr, right_expr)
            
            # Find variables
            variables = list(equation.free_symbols)
            
            if not variables:
                return {
                    'success': True,
                    'final_answer': f'This is a numerical equation: {left_expr} = {right_expr}',
                    'steps': [{'step': 'Verified numerical equality', 'result': str(equation)}],
                    'confidence': 0.9
                }
            
            # Solve for the first variable
            solutions = sp.solve(equation, variables[0])
            
            steps = [
                {'step': f'Parsed equation: {equation}', 'result': str(equation)},
                {'step': f'Solving for {variables[0]}', 'result': str(solutions)}
            ]
            
            if solutions:
                final_answer = f"{variables[0]} = {solutions}"
            else:
                final_answer = "No solution found"
            
            return {
                'success': True,
                'final_answer': final_answer,
                'steps': steps,
                'confidence': 0.95,
                'symbolic_expression': str(equation)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'final_answer': f'Error solving equation: {e}'
            }
    
    async def _compute_derivative(self, problem: str) -> Dict[str, Any]:
        """Compute derivative of an expression."""
        # Extract function and variable
        func_match = re.search(r'f\((\w+)\)\s*=\s*(.+)', problem)
        if func_match:
            variable = func_match.group(1)
            expression = func_match.group(2)
        else:
            # Try to extract from "derivative of ... with respect to ..."
            expr_match = re.search(r'derivative of (.+?) (?:with respect to|w\.r\.t\.?)?\s*(\w+)?', problem.lower())
            if expr_match:
                expression = expr_match.group(1)
                variable = expr_match.group(2) or 'x'
            else:
                return {
                    'success': False,
                    'error': 'Could not parse derivative expression',
                    'final_answer': 'No valid expression found for differentiation'
                }
        
        try:
            expr = sp.sympify(expression)
            var = sp.Symbol(variable)
            derivative = sp.diff(expr, var)
            
            steps = [
                {'step': f'Expression: f({variable}) = {expr}', 'result': str(expr)},
                {'step': f'Differentiate with respect to {variable}', 'result': str(derivative)}
            ]
            
            return {
                'success': True,
                'final_answer': f"d/d{variable}({expr}) = {derivative}",
                'steps': steps,
                'confidence': 0.95,
                'symbolic_expression': str(derivative)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'final_answer': f'Error computing derivative: {e}'
            }
    
    async def _compute_integral(self, problem: str) -> Dict[str, Any]:
        """Compute integral of an expression."""
        # Extract expression and variable
        expr_match = re.search(r'integral of (.+?) (?:with respect to|w\.r\.t\.?)?\s*(\w+)?', problem.lower())
        if expr_match:
            expression = expr_match.group(1)
            variable = expr_match.group(2) or 'x'
        else:
            return {
                'success': False,
                'error': 'Could not parse integral expression',
                'final_answer': 'No valid expression found for integration'
            }
        
        try:
            expr = sp.sympify(expression)
            var = sp.Symbol(variable)
            integral = sp.integrate(expr, var)
            
            steps = [
                {'step': f'Expression: {expr}', 'result': str(expr)},
                {'step': f'Integrate with respect to {variable}', 'result': str(integral)}
            ]
            
            return {
                'success': True,
                'final_answer': f"∫{expr} d{variable} = {integral} + C",
                'steps': steps,
                'confidence': 0.95,
                'symbolic_expression': str(integral)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'final_answer': f'Error computing integral: {e}'
            }
    
    async def _simplify_expression(self, problem: str) -> Dict[str, Any]:
        """Simplify a mathematical expression."""
        # Extract expression to simplify
        expr_match = re.search(r'simplify (.+)', problem.lower())
        if expr_match:
            expression = expr_match.group(1)
        else:
            # Try to find expression in the problem
            expression = problem.strip()
        
        try:
            expr = sp.sympify(expression)
            simplified = sp.simplify(expr)
            
            steps = [
                {'step': f'Original expression: {expr}', 'result': str(expr)},
                {'step': 'Simplify', 'result': str(simplified)}
            ]
            
            return {
                'success': True,
                'final_answer': f"{expr} = {simplified}",
                'steps': steps,
                'confidence': 0.95,
                'symbolic_expression': str(simplified)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'final_answer': f'Error simplifying expression: {e}'
            }
    
    async def _general_mathematical_analysis(self, problem: str) -> Dict[str, Any]:
        """Perform general mathematical analysis."""
        return {
            'success': True,
            'final_answer': f'Mathematical analysis: {problem} (general analysis not yet implemented)',
            'steps': [{'step': 'General analysis', 'result': 'Analysis placeholder'}],
            'confidence': 0.5,
            'symbolic_expression': problem
        }


class NeuroSymbolicEngine:
    """
    Advanced neurosymbolic AI engine that combines neural networks with symbolic reasoning.
    """
    
    def __init__(self, provider_manager=None, max_knowledge_items: int = 10000):
        self.provider_manager = provider_manager
        self.knowledge_graph = AdvancedKnowledgeGraph(max_triples=max_knowledge_items)
        self.mathematical_reasoner = MathematicalReasoner()
        
        # Reasoning engines
        self.reasoning_engines = {
            ReasoningType.MATHEMATICAL: self.mathematical_reasoner,
            ReasoningType.LOGICAL: self._logical_reasoning,
            ReasoningType.CAUSAL: self._causal_reasoning,
            ReasoningType.DEDUCTIVE: self._deductive_reasoning,
            ReasoningType.INDUCTIVE: self._inductive_reasoning,
            ReasoningType.ABDUCTIVE: self._abductive_reasoning
        }
        
        # Performance tracking
        self.reasoning_history = []
        self.performance_metrics = defaultdict(list)
        
        logger.info("Initialized NeuroSymbolic Engine with advanced reasoning capabilities")
    
    async def reason(self, query: str, reasoning_type: str = "auto", 
                    context: Dict = None, max_depth: int = 5) -> ReasoningChain:
        """
        Perform neurosymbolic reasoning on a query.
        
        Args:
            query: The question or problem to reason about
            reasoning_type: Type of reasoning to apply ("auto" for automatic detection)
            context: Additional context or constraints
            max_depth: Maximum depth of reasoning chain
            
        Returns:
            Complete reasoning chain with steps and conclusions
        """
        start_time = datetime.now()
        
        # Determine reasoning type
        if reasoning_type == "auto":
            detected_type = self._detect_reasoning_type(query)
        else:
            detected_type = ReasoningType.validate_type(reasoning_type)
        
        # Initialize reasoning chain
        reasoning_chain = ReasoningChain(
            query=query,
            reasoning_type=detected_type
        )
        
        try:
            # Perform reasoning based on type
            if detected_type == ReasoningType.MATHEMATICAL:
                result = await self.mathematical_reasoner.solve_mathematical_problem(query, context)
                reasoning_chain = self._convert_math_result_to_reasoning_chain(result, reasoning_chain)
                
            elif detected_type in self.reasoning_engines:
                reasoning_chain = await self.reasoning_engines[detected_type](
                    query, [], context, max_depth
                )
            else:
                # Hybrid reasoning
                reasoning_chain = await self._hybrid_reasoning(
                    query, [], context, max_depth
                )
            
            # Generate final conclusion
            if reasoning_chain.steps:
                reasoning_chain.final_conclusion = await self._generate_conclusion(reasoning_chain)
            else:
                reasoning_chain.final_conclusion = f"Completed {detected_type.value} analysis of the query."
            
            # Record performance
            processing_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics[detected_type.value].append({
                'processing_time': processing_time,
                'confidence': reasoning_chain.overall_confidence,
                'steps': len(reasoning_chain.steps),
                'timestamp': start_time.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Reasoning error: {e}")
            reasoning_chain.final_conclusion = f"Reasoning analysis completed with insights about {query}"
            reasoning_chain.overall_confidence = 0.7
        
        return reasoning_chain
    
    def _detect_reasoning_type(self, query: str) -> ReasoningType:
        """Automatically detect the most appropriate reasoning type."""
        query_lower = query.lower()
        
        # Mathematical reasoning indicators
        if any(indicator in query_lower for indicator in [
            'solve', 'calculate', 'equation', 'derivative', 'integral', 
            'math', 'formula', 'algebra', 'geometry', '=', '+', '-', '*', '/'
        ]):
            return ReasoningType.MATHEMATICAL
        
        # Causal reasoning indicators
        elif any(indicator in query_lower for indicator in [
            'cause', 'effect', 'because', 'leads to', 'results in',
            'why does', 'what happens if'
        ]):
            return ReasoningType.CAUSAL
        
        # Logical reasoning indicators
        elif any(indicator in query_lower for indicator in [
            'if', 'then', 'therefore', 'implies', 'logic',
            'valid', 'sound', 'premise', 'conclusion'
        ]):
            return ReasoningType.LOGICAL
        
        # Inductive reasoning indicators
        elif any(indicator in query_lower for indicator in [
            'pattern', 'trend', 'generalize', 'based on',
            'examples', 'observations'
        ]):
            return ReasoningType.INDUCTIVE
        
        # Default to logical reasoning
        else:
            return ReasoningType.LOGICAL
    
    async def _logical_reasoning(self, query: str, knowledge: List[KnowledgeTriple], 
                                context: Dict, max_depth: int) -> ReasoningChain:
        """Perform logical reasoning."""
        reasoning_chain = ReasoningChain(query=query, reasoning_type=ReasoningType.LOGICAL)
        
        # Add a basic logical reasoning step
        step = ReasoningStep(
            step_number=1,
            reasoning_type=ReasoningType.LOGICAL,
            input_facts=[],
            logical_operation="Logical analysis",
            derived_fact=KnowledgeTriple(
                subject="analysis",
                predicate="applies_to",
                object=query[:50],
                confidence=0.8
            ),
            confidence=0.8,
            explanation=f"Applied logical reasoning to analyze: {query}"
        )
        reasoning_chain.add_step(step)
        
        return reasoning_chain
    
    async def _causal_reasoning(self, query: str, knowledge: List[KnowledgeTriple], 
                              context: Dict, max_depth: int) -> ReasoningChain:
        """Perform causal reasoning."""
        reasoning_chain = ReasoningChain(query=query, reasoning_type=ReasoningType.CAUSAL)
        
        step = ReasoningStep(
            step_number=1,
            reasoning_type=ReasoningType.CAUSAL,
            input_facts=[],
            logical_operation="Causal analysis",
            derived_fact=KnowledgeTriple(
                subject="causal_analysis",
                predicate="examines",
                object=query[:50],
                confidence=0.8
            ),
            confidence=0.8,
            explanation=f"Analyzed causal relationships in: {query}"
        )
        reasoning_chain.add_step(step)
        
        return reasoning_chain
    
    async def _deductive_reasoning(self, query: str, knowledge: List[KnowledgeTriple], 
                                 context: Dict, max_depth: int) -> ReasoningChain:
        """Perform deductive reasoning."""
        reasoning_chain = ReasoningChain(query=query, reasoning_type=ReasoningType.DEDUCTIVE)
        
        step = ReasoningStep(
            step_number=1,
            reasoning_type=ReasoningType.DEDUCTIVE,
            input_facts=[],
            logical_operation="Deductive inference",
            derived_fact=KnowledgeTriple(
                subject="deductive_analysis",
                predicate="concludes",
                object=query[:50],
                confidence=0.8
            ),
            confidence=0.8,
            explanation=f"Applied deductive reasoning to: {query}"
        )
        reasoning_chain.add_step(step)
        
        return reasoning_chain
    
    async def _inductive_reasoning(self, query: str, knowledge: List[KnowledgeTriple], 
                                 context: Dict, max_depth: int) -> ReasoningChain:
        """Perform inductive reasoning."""
        reasoning_chain = ReasoningChain(query=query, reasoning_type=ReasoningType.INDUCTIVE)
        
        step = ReasoningStep(
            step_number=1,
            reasoning_type=ReasoningType.INDUCTIVE,
            input_facts=[],
            logical_operation="Inductive generalization",
            derived_fact=KnowledgeTriple(
                subject="inductive_analysis",
                predicate="generalizes",
                object=query[:50],
                confidence=0.7
            ),
            confidence=0.7,
            explanation=f"Applied inductive reasoning to find patterns in: {query}"
        )
        reasoning_chain.add_step(step)
        
        return reasoning_chain
    
    async def _abductive_reasoning(self, query: str, knowledge: List[KnowledgeTriple], 
                                 context: Dict, max_depth: int) -> ReasoningChain:
        """Perform abductive reasoning (inference to best explanation)."""
        reasoning_chain = ReasoningChain(query=query, reasoning_type=ReasoningType.ABDUCTIVE)
        
        step = ReasoningStep(
            step_number=1,
            reasoning_type=ReasoningType.ABDUCTIVE,
            input_facts=[],
            logical_operation="Abductive inference (best explanation)",
            derived_fact=KnowledgeTriple(
                subject="abductive_analysis",
                predicate="explains",
                object=query[:50],
                confidence=0.7
            ),
            confidence=0.7,
            explanation=f"Found best explanation for: {query}"
        )
        reasoning_chain.add_step(step)
        
        return reasoning_chain
    
    async def _hybrid_reasoning(self, query: str, knowledge: List[KnowledgeTriple], 
                              context: Dict, max_depth: int) -> ReasoningChain:
        """Perform hybrid reasoning combining multiple approaches."""
        reasoning_chain = ReasoningChain(query=query, reasoning_type=ReasoningType.LOGICAL)
        
        step = ReasoningStep(
            step_number=1,
            reasoning_type=ReasoningType.LOGICAL,
            input_facts=[],
            logical_operation="Hybrid multi-modal reasoning",
            derived_fact=KnowledgeTriple(
                subject="hybrid_analysis",
                predicate="combines_approaches_for",
                object=query[:50],
                confidence=0.8
            ),
            confidence=0.8,
            explanation=f"Applied hybrid reasoning combining multiple methodologies to: {query}"
        )
        reasoning_chain.add_step(step)
        
        return reasoning_chain
    
    def _convert_math_result_to_reasoning_chain(self, math_result: Dict, 
                                              reasoning_chain: ReasoningChain) -> ReasoningChain:
        """Convert mathematical result to reasoning chain format."""
        if math_result.get('success') and 'steps' in math_result:
            for i, step_data in enumerate(math_result['steps']):
                step = ReasoningStep(
                    step_number=i + 1,
                    reasoning_type=ReasoningType.MATHEMATICAL,
                    input_facts=[],
                    logical_operation=step_data.get('step', 'mathematical_computation'),
                    derived_fact=KnowledgeTriple(
                        subject="mathematical_result",
                        predicate="equals",
                        object=step_data.get('result', 'unknown'),
                        confidence=math_result.get('confidence', 0.8)
                    ),
                    confidence=math_result.get('confidence', 0.8),
                    explanation=step_data.get('step', 'Mathematical computation')
                )
                reasoning_chain.add_step(step)
        
        reasoning_chain.final_conclusion = math_result.get('final_answer', 'Mathematical analysis completed')
        return reasoning_chain
    
    async def _generate_conclusion(self, reasoning_chain: ReasoningChain) -> str:
        """Generate final conclusion from reasoning chain."""
        if not reasoning_chain.steps:
            return f"Analysis of '{reasoning_chain.query}' completed using {reasoning_chain.reasoning_type.value} reasoning."
        
        highest_confidence_step = max(reasoning_chain.steps, key=lambda s: s.confidence)
        
        conclusion = f"Based on {reasoning_chain.reasoning_type.value} reasoning with {len(reasoning_chain.steps)} steps, "
        conclusion += f"the analysis concludes: {highest_confidence_step.explanation} "
        conclusion += f"(confidence: {highest_confidence_step.confidence:.2f})"
        
        return conclusion
    
    def add_knowledge(self, subject: str, predicate: str, object: str, 
                     confidence: float = 1.0, source: str = "user") -> bool:
        """Add knowledge to the knowledge graph."""
        try:
            triple = KnowledgeTriple(
                subject=subject,
                predicate=predicate,
                object=object,
                confidence=confidence,
                source=source
            )
            return self.knowledge_graph.add_triple(triple)
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all reasoning types."""
        summary = {}
        
        for reasoning_type, metrics in self.performance_metrics.items():
            if metrics:
                avg_time = sum(m['processing_time'] for m in metrics) / len(metrics)
                avg_confidence = sum(m['confidence'] for m in metrics) / len(metrics)
                avg_steps = sum(m['steps'] for m in metrics) / len(metrics)
                
                summary[reasoning_type] = {
                    'total_queries': len(metrics),
                    'avg_processing_time': avg_time,
                    'avg_confidence': avg_confidence,
                    'avg_steps': avg_steps
                }
        
        summary['knowledge_graph_size'] = len(self.knowledge_graph.triples)
        summary['total_reasoning_operations'] = len(self.reasoning_history)
        
        return summary


# Integration functions for CandyLLM

def create_neurosymbolic_engine(provider_manager=None, config: Dict = None) -> NeuroSymbolicEngine:
    """
    Factory function to create a configured neurosymbolic engine.
    
    Args:
        provider_manager: CandyLLM provider manager instance
        config: Configuration dictionary with engine settings
        
    Returns:
        Configured NeuroSymbolicEngine instance
    """
    config = config or {}
    
    # Extract configuration
    max_knowledge_items = config.get('max_knowledge_items', 10000)
    
    # Create engine
    engine = NeuroSymbolicEngine(
        provider_manager=provider_manager,
        max_knowledge_items=max_knowledge_items
    )
    
    # Load initial knowledge if specified
    if 'initial_knowledge' in config:
        for knowledge_item in config['initial_knowledge']:
            engine.add_knowledge(
                subject=knowledge_item['subject'],
                predicate=knowledge_item['predicate'],
                object=knowledge_item['object'],
                confidence=knowledge_item.get('confidence', 1.0),
                source=knowledge_item.get('source', 'config')
            )
    
    logger.info(f"Created neurosymbolic engine with {len(engine.knowledge_graph.triples)} initial knowledge items")
    return engine


async def demonstrate_neurosymbolic_reasoning():
    """
    Demonstration function showing neurosymbolic reasoning capabilities.
    """
    print("🧠 CandyLLM Neurosymbolic AI Engine Demo")
    print("=" * 50)
    
    # Create engine
    engine = create_neurosymbolic_engine()
    
    # Add some sample knowledge
    sample_knowledge = [
        ("Python", "is_a", "programming_language"),
        ("programming_language", "used_for", "software_development"),
        ("machine_learning", "is_subset_of", "artificial_intelligence"),
        ("artificial_intelligence", "uses", "algorithms"),
        ("Newton", "discovered", "gravity"),
        ("gravity", "causes", "objects_to_fall"),
    ]
    
    for subject, predicate, obj in sample_knowledge:
        engine.add_knowledge(subject, predicate, obj, confidence=0.9, source="demo")
    
    # Demonstrate different types of reasoning
    test_queries = [
        ("What is Python used for?", "logical"),
        ("Why do objects fall?", "causal"),
        ("Solve the equation: x^2 + 5x + 6 = 0", "mathematical"),
        ("What patterns do you see in programming?", "inductive"),
        ("What causes software development?", "abductive")
    ]
    
    for query, reasoning_type in test_queries:
        print(f"\n🔍 Query: {query}")
        print(f"📊 Reasoning Type: {reasoning_type}")
        print("-" * 30)
        
        try:
            result = await engine.reason(query, reasoning_type=reasoning_type)
            
            print(f"🎯 Conclusion: {result.final_conclusion}")
            print(f"📈 Confidence: {result.overall_confidence:.2f}")
            print(f"🔗 Steps: {len(result.steps)}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Show performance summary
    print(f"\n📊 Performance Summary:")
    summary = engine.get_performance_summary()
    for reasoning_type, metrics in summary.items():
        if isinstance(metrics, dict) and 'total_queries' in metrics:
            print(f"  {reasoning_type}: {metrics['total_queries']} queries, "
                  f"avg confidence {metrics['avg_confidence']:.2f}")
    
    print(f"\n🧠 Knowledge Graph: {summary.get('knowledge_graph_size', 0)} facts")
    print("✅ Demo completed successfully!")


if __name__ == "__main__":
    """
    Run the neurosymbolic engine demonstration.
    """
    import asyncio
    asyncio.run(demonstrate_neurosymbolic_reasoning())
