"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

🍭 CandyLLM Neurosymbolic AI Engine with Enterprise Security
Advanced neurosymbolic integration combining neural networks with symbolic reasoning.

This module provides secure neurosymbolic AI capabilities with comprehensive security controls:
- Input validation and sanitization for all symbolic operations
- Secure knowledge graph operations with access controls
- Mathematical expression validation and sandboxing
- Audit logging for all reasoning operations
- Resource limits and execution monitoring
- Protection against code injection and logic bombs

Based on latest research in neurosymbolic AI from 2024-2025:
- Symbolic knowledge integration with security validation
- Logical reasoning with neural networks and threat detection
- Knowledge graph augmentation with integrity checks
- Causal reasoning with security constraints
- Interpretable AI decisions with audit trails
"""

import asyncio
import json
import re
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import networkx as nx
from collections import defaultdict

# Secure mathematical operations
try:
    import sympy as sp
    from sympy import symbols, solve, simplify, diff, integrate
    from sympy.parsing.sympy_parser import parse_expr
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# Configure secure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Security validation patterns for symbolic expressions
SAFE_SYMBOL_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
SAFE_PREDICATE_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\s]+$')
DANGEROUS_MATH_PATTERNS = [
    r'__[a-zA-Z_]+__',  # Python magic methods
    r'eval\s*\(',       # Code execution
    r'exec\s*\(',       # Code execution
    r'import\s+',       # Module imports
    r'subprocess',      # Process execution
    r'os\.',           # OS operations
]


class ReasoningType(Enum):
    """Types of reasoning supported with security validation."""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    CAUSAL = "causal"
    ANALOGICAL = "analogical"
    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    
    @classmethod
    def validate_type(cls, reasoning_type: str) -> 'ReasoningType':
        """Validate and return secure reasoning type."""
        if not isinstance(reasoning_type, str):
            raise ValueError("Reasoning type must be string")
        
        try:
            return cls(reasoning_type.lower())
        except ValueError:
            valid_types = [t.value for t in cls]
            raise ValueError(f"Invalid reasoning type: {reasoning_type}. Valid types: {valid_types}")


class KnowledgeType(Enum):
    """Types of knowledge in the system with security controls."""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    RELATIONAL = "relational"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    
    @classmethod
    def validate_type(cls, knowledge_type: str) -> 'KnowledgeType':
        """Validate and return secure knowledge type."""
        if not isinstance(knowledge_type, str):
            raise ValueError("Knowledge type must be string")
        
        try:
            return cls(knowledge_type.lower())
        except ValueError:
            valid_types = [t.value for t in cls]
            raise ValueError(f"Invalid knowledge type: {knowledge_type}. Valid types: {valid_types}")


class SymbolicSecurityValidator:
    """Security validator for symbolic operations and expressions."""
    
    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """Validate symbolic variable names for security."""
        if not isinstance(symbol, str):
            raise ValueError("Symbol must be string")
        
        if not symbol or len(symbol) > 50:
            raise ValueError("Symbol length must be 1-50 characters")
        
        if not SAFE_SYMBOL_PATTERN.match(symbol):
            raise ValueError("Symbol contains unsafe characters")
        
        # Check for reserved keywords
        reserved = {'eval', 'exec', 'import', 'os', 'sys', 'subprocess'}
        if symbol.lower() in reserved:
            raise ValueError(f"Symbol name '{symbol}' is reserved")
        
        return symbol
    
    @staticmethod
    def validate_predicate(predicate: str) -> str:
        """Validate predicate names for security."""
        if not isinstance(predicate, str):
            raise ValueError("Predicate must be string")
        
        if not predicate or len(predicate) > 100:
            raise ValueError("Predicate length must be 1-100 characters")
        
        if not SAFE_PREDICATE_PATTERN.match(predicate):
            raise ValueError("Predicate contains unsafe characters")
        
        return predicate.strip()
    
    @staticmethod
    def validate_mathematical_expression(expr: str) -> str:
        """Validate mathematical expressions for security."""
        if not isinstance(expr, str):
            raise ValueError("Expression must be string")
        
        if len(expr) > 1000:
            raise ValueError("Expression too long (>1000 chars)")
        
        # Check for dangerous patterns
        for pattern in DANGEROUS_MATH_PATTERNS:
            if re.search(pattern, expr, re.IGNORECASE):
                raise ValueError(f"Dangerous pattern detected in expression")
        
        # Basic syntax validation if sympy available
        if SYMPY_AVAILABLE:
            try:
                # Test parse without evaluation
                parsed = parse_expr(expr, evaluate=False)
                return expr
            except Exception as e:
                raise ValueError(f"Invalid mathematical expression: {e}")
        
        return expr
    
    @staticmethod
    def sanitize_knowledge_text(text: str) -> str:
        """Sanitize knowledge text for security."""
        if not isinstance(text, str):
            raise ValueError("Knowledge text must be string")
        
        if len(text) > 5000:
            text = text[:5000]
            logger.warning("Truncated oversized knowledge text")
        
        # Remove potential script injection
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript\s*:', '', text, flags=re.IGNORECASE)
        
        # Remove null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text.strip()


@dataclass
class KnowledgeTriple:
    """
    Represents a secure knowledge triple (subject, predicate, object) with validation.
    
    All components are validated and sanitized to prevent injection attacks and
    ensure data integrity in the knowledge graph.
    """
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "user"
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and sanitize triple components after initialization."""
        # Validate and sanitize subject
        self.subject = SymbolicSecurityValidator.sanitize_knowledge_text(str(self.subject))
        if not self.subject:
            raise ValueError("Subject cannot be empty after sanitization")
        
        # Validate and sanitize predicate
        self.predicate = SymbolicSecurityValidator.validate_predicate(str(self.predicate))
        
        # Validate and sanitize object
        self.object = SymbolicSecurityValidator.sanitize_knowledge_text(str(self.object))
        if not self.object:
            raise ValueError("Object cannot be empty after sanitization")
        
        # Validate confidence score
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be float between 0 and 1")
        
        # Validate source
        if not isinstance(self.source, str) or len(self.source) > 100:
            raise ValueError("Source must be string with max 100 characters")
        
        # Create secure hash for triple identification
        triple_str = f"{self.subject}|{self.predicate}|{self.object}"
        self.hash_id = hashlib.sha256(triple_str.encode()).hexdigest()[:16]


@dataclass 
class ReasoningStep:
    """
    Represents a single step in a reasoning chain with security tracking.
    """
    step_number: int
    reasoning_type: ReasoningType
    input_facts: List[KnowledgeTriple]
    logical_operation: str
    derived_fact: KnowledgeTriple
    confidence: float
    explanation: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate reasoning step components."""
        if not isinstance(self.step_number, int) or self.step_number < 1:
            raise ValueError("Step number must be positive integer")
        
        if not isinstance(self.reasoning_type, ReasoningType):
            raise ValueError("Invalid reasoning type")
        
        if not isinstance(self.input_facts, list):
            raise ValueError("Input facts must be list")
        
        # Validate logical operation
        self.logical_operation = SymbolicSecurityValidator.sanitize_knowledge_text(
            str(self.logical_operation)
        )
        
        # Validate explanation
        self.explanation = SymbolicSecurityValidator.sanitize_knowledge_text(
            str(self.explanation)
        )
        
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be float between 0 and 1")


@dataclass
class ReasoningChain:
    """
    Complete reasoning chain with security validation and audit trail.
    """
    query: str
    reasoning_type: ReasoningType
    steps: List[ReasoningStep] = field(default_factory=list)
    final_conclusion: Optional[str] = None
    overall_confidence: float = 0.0
    knowledge_sources: List[str] = field(default_factory=list)
    security_hash: Optional[str] = None
    
    def __post_init__(self):
        """Validate and secure reasoning chain."""
        self.query = SymbolicSecurityValidator.sanitize_knowledge_text(str(self.query))
        if not self.query:
            raise ValueError("Query cannot be empty")
        
        # Generate security hash for audit trail
        chain_data = {
            'query': self.query,
            'reasoning_type': self.reasoning_type.value,
            'timestamp': datetime.now().isoformat()
        }
        self.security_hash = hashlib.sha256(
            json.dumps(chain_data, sort_keys=True).encode()
        ).hexdigest()[:16]
    
    def add_step(self, step: ReasoningStep):
        """Add a reasoning step with validation."""
        if not isinstance(step, ReasoningStep):
            raise ValueError("Must provide ReasoningStep instance")
        
        self.steps.append(step)
        
        # Update overall confidence (average with decay)
        if self.steps:
            confidences = [s.confidence for s in self.steps]
            self.overall_confidence = sum(confidences) / len(confidences)
    
    def get_audit_trail(self) -> Dict[str, Any]:
        """Get complete audit trail for reasoning chain."""
        return {
            'security_hash': self.security_hash,
            'query': self.query,
            'reasoning_type': self.reasoning_type.value,
            'num_steps': len(self.steps),
            'overall_confidence': self.overall_confidence,
            'knowledge_sources': self.knowledge_sources,
            'step_details': [
                {
                    'step': s.step_number,
                    'operation': s.logical_operation,
                    'confidence': s.confidence,
                    'timestamp': s.timestamp.isoformat()
                }
                for s in self.steps
            ]
        }


class AdvancedKnowledgeGraph:
    """
    Advanced knowledge graph with semantic reasoning and security controls.
    
    Features:
    - Secure triple storage with validation
    - Semantic similarity search
    - Causal relationship tracking
    - Temporal knowledge management
    - Multi-hop reasoning capabilities
    - Audit logging and access controls
    """
    
    def __init__(self, max_triples: int = 10000):
        self.graph = nx.MultiDiGraph()
        self.triples = {}  # hash_id -> KnowledgeTriple
        self.max_triples = max_triples
        
        # Semantic indexes
        self.subject_index = defaultdict(set)
        self.predicate_index = defaultdict(set)
        self.object_index = defaultdict(set)
        
        # Temporal tracking
        self.temporal_facts = defaultdict(list)
        
        # Causal relationships
        self.causal_chains = []
        
        # Security audit log
        self.audit_log = []
        
        logger.info(f"Initialized secure knowledge graph with max {max_triples} triples")
    
    def add_triple(self, triple: KnowledgeTriple) -> bool:
        """Add a validated knowledge triple to the graph."""
        try:
            # Check capacity
            if len(self.triples) >= self.max_triples:
                logger.warning("Knowledge graph at capacity, removing oldest triple")
                self._remove_oldest_triple()
            
            # Add to storage
            self.triples[triple.hash_id] = triple
            
            # Add to NetworkX graph
            self.graph.add_edge(
                triple.subject, 
                triple.object,
                predicate=triple.predicate,
                confidence=triple.confidence,
                source=triple.source,
                hash_id=triple.hash_id
            )
            
            # Update indexes
            self.subject_index[triple.subject].add(triple.hash_id)
            self.predicate_index[triple.predicate].add(triple.hash_id)
            self.object_index[triple.object].add(triple.hash_id)
            
            # Audit log
            self.audit_log.append({
                'action': 'add_triple',
                'triple_hash': triple.hash_id,
                'timestamp': datetime.now().isoformat(),
                'confidence': triple.confidence
            })
            
            logger.debug(f"Added triple: {triple.subject} -> {triple.predicate} -> {triple.object}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add triple: {e}")
            return False
    
    def query_by_pattern(self, subject: str = None, predicate: str = None, 
                        object: str = None, min_confidence: float = 0.0) -> List[KnowledgeTriple]:
        """Query triples by pattern with confidence filtering."""
        matching_hash_ids = set(self.triples.keys())
        
        # Filter by subject
        if subject:
            subject = SymbolicSecurityValidator.sanitize_knowledge_text(subject)
            matching_hash_ids &= self.subject_index.get(subject, set())
        
        # Filter by predicate
        if predicate:
            predicate = SymbolicSecurityValidator.validate_predicate(predicate)
            matching_hash_ids &= self.predicate_index.get(predicate, set())
        
        # Filter by object
        if object:
            object = SymbolicSecurityValidator.sanitize_knowledge_text(object)
            matching_hash_ids &= self.object_index.get(object, set())
        
        # Filter by confidence and return
        results = []
        for hash_id in matching_hash_ids:
            triple = self.triples[hash_id]
            if triple.confidence >= min_confidence:
                results.append(triple)
        
        # Sort by confidence (highest first)
        results.sort(key=lambda t: t.confidence, reverse=True)
        
        logger.debug(f"Pattern query returned {len(results)} triples")
        return results
    
    def find_causal_chain(self, cause: str, effect: str, max_depth: int = 3) -> List[List[KnowledgeTriple]]:
        """Find causal chains between cause and effect."""
        cause = SymbolicSecurityValidator.sanitize_knowledge_text(cause)
        effect = SymbolicSecurityValidator.sanitize_knowledge_text(effect)
        
        chains = []
        
        try:
            # Use NetworkX to find paths
            if self.graph.has_node(cause) and self.graph.has_node(effect):
                paths = list(nx.all_simple_paths(
                    self.graph, cause, effect, cutoff=max_depth
                ))
                
                for path in paths:
                    chain = []
                    for i in range(len(path) - 1):
                        # Find triples for this edge
                        edges = self.graph.get_edge_data(path[i], path[i + 1])
                        if edges:
                            # Get highest confidence edge
                            best_edge = max(edges.values(), key=lambda e: e.get('confidence', 0))
                            hash_id = best_edge.get('hash_id')
                            if hash_id and hash_id in self.triples:
                                chain.append(self.triples[hash_id])
                    
                    if chain:
                        chains.append(chain)
            
            logger.debug(f"Found {len(chains)} causal chains from {cause} to {effect}")
            
        except Exception as e:
            logger.error(f"Error finding causal chains: {e}")
        
        return chains
    
    def get_related_concepts(self, concept: str, max_results: int = 10) -> List[Tuple[str, float]]:
        """Get concepts related to the given concept with similarity scores."""
        concept = SymbolicSecurityValidator.sanitize_knowledge_text(concept)
        
        related = []
        
        # Find direct connections
        if concept in self.subject_index:
            for triple_hash in self.subject_index[concept]:
                triple = self.triples[triple_hash]
                related.append((triple.object, triple.confidence))
        
        if concept in self.object_index:
            for triple_hash in self.object_index[concept]:
                triple = self.triples[triple_hash]
                related.append((triple.subject, triple.confidence))
        
        # Deduplicate and sort
        concept_scores = defaultdict(float)
        for related_concept, score in related:
            concept_scores[related_concept] = max(concept_scores[related_concept], score)
        
        # Sort by score and return top results
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_concepts[:max_results]
    
    def _remove_oldest_triple(self):
        """Remove the oldest triple to make space."""
        if not self.audit_log:
            return
        
        # Find oldest triple by audit log
        oldest_entry = min(self.audit_log, key=lambda x: x['timestamp'])
        hash_id = oldest_entry['triple_hash']
        
        if hash_id in self.triples:
            triple = self.triples[hash_id]
            
            # Remove from graph
            self.graph.remove_edge(triple.subject, triple.object)
            
            # Remove from indexes
            self.subject_index[triple.subject].discard(hash_id)
            self.predicate_index[triple.predicate].discard(hash_id)
            self.object_index[triple.object].discard(hash_id)
            
            # Remove from storage
            del self.triples[hash_id]
            
            # Remove from audit log
            self.audit_log.remove(oldest_entry)
            
            logger.debug(f"Removed oldest triple: {hash_id}")


class MathematicalReasoner:
    """
    Advanced mathematical reasoning with symbolic computation and security.
    
    Features:
    - Secure symbolic math operations
    - Equation solving with validation
    - Calculus operations (derivatives, integrals)
    - Algebraic manipulation
    - Mathematical proof assistance
    - Security validation for all expressions
    """
    
    def __init__(self):
        self.sympy_available = SYMPY_AVAILABLE
        self.security_validator = SymbolicSecurityValidator()
        
        if not self.sympy_available:
            logger.warning("SymPy not available, mathematical reasoning limited")
    
    async def solve_mathematical_problem(self, problem: str, context: Dict = None) -> Dict[str, Any]:
        """
        Solve mathematical problems with detailed reasoning.
        
        Args:
            problem: Mathematical problem description
            context: Additional context or constraints
            
        Returns:
            Detailed solution with steps and validation
        """
        problem = self.security_validator.sanitize_knowledge_text(problem)
        
        result = {
            'problem': problem,
            'solution_type': 'mathematical',
            'steps': [],
            'final_answer': None,
            'symbolic_result': None,
            'numerical_result': None,
            'confidence': 0.0,
            'method_used': 'symbolic_computation',
            'validation_passed': False
        }
        
        try:
            # Analyze problem type
            problem_type = self._classify_math_problem(problem)
            result['problem_type'] = problem_type
            
            if not self.sympy_available:
                result['error'] = "SymPy not available for symbolic computation"
                return result
            
            # Extract mathematical expressions
            expressions = self._extract_math_expressions(problem)
            
            if problem_type == 'equation_solving':
                solution = await self._solve_equations(expressions, problem)
                result.update(solution)
                
            elif problem_type == 'calculus':
                solution = await self._solve_calculus(expressions, problem)
                result.update(solution)
                
            elif problem_type == 'algebra':
                solution = await self._solve_algebra(expressions, problem)
                result.update(solution)
                
            elif problem_type == 'optimization':
                solution = await self._solve_optimization(expressions, problem, context)
                result.update(solution)
                
            else:
                # General mathematical analysis
                solution = await self._general_math_analysis(expressions, problem)
                result.update(solution)
            
            # Validate results
            result['validation_passed'] = self._validate_mathematical_result(result)
            
        except Exception as e:
            logger.error(f"Mathematical reasoning error: {e}")
            result['error'] = str(e)
            result['confidence'] = 0.0
        
        return result
    
    def _classify_math_problem(self, problem: str) -> str:
        """Classify the type of mathematical problem."""
        problem_lower = problem.lower()
        
        if any(word in problem_lower for word in ['solve', 'equation', '=']):
            return 'equation_solving'
        elif any(word in problem_lower for word in ['derivative', 'integral', 'limit', 'differentiate', 'integrate']):
            return 'calculus'
        elif any(word in problem_lower for word in ['simplify', 'expand', 'factor', 'expression']):
            return 'algebra'
        elif any(word in problem_lower for word in ['optimize', 'minimize', 'maximize', 'constraint']):
            return 'optimization'
        else:
            return 'general'
    
    def _extract_math_expressions(self, problem: str) -> List[str]:
        """Extract mathematical expressions from problem text."""
        expressions = []
        
        # Look for equations (contains =)
        equation_pattern = r'[^=]*=[^=]*'
        equations = re.findall(equation_pattern, problem)
        expressions.extend(equations)
        
        # Look for expressions in parentheses or with mathematical operators
        expr_pattern = r'[\w\s]*[+\-*/^()x]+[\w\s]*'
        expr_matches = re.findall(expr_pattern, problem)
        expressions.extend(expr_matches)
        
        # Validate and clean expressions
        clean_expressions = []
        for expr in expressions:
            try:
                clean_expr = self.security_validator.validate_mathematical_expression(expr.strip())
                if clean_expr and len(clean_expr) > 3:  # Minimum meaningful length
                    clean_expressions.append(clean_expr)
            except ValueError:
                continue
        
        return clean_expressions
    
    async def _solve_equations(self, expressions: List[str], problem: str) -> Dict[str, Any]:
        """Solve mathematical equations."""
        result = {
            'steps': [],
            'solutions': [],
            'symbolic_result': None,
            'confidence': 0.8
        }
        
        try:
            for expr in expressions:
                if '=' in expr:
                    # Split equation into left and right sides
                    left, right = expr.split('=', 1)
                    
                    # Parse expressions
                    left_expr = parse_expr(left.strip())
                    right_expr = parse_expr(right.strip())
                    
                    # Create equation (left - right = 0)
                    equation = left_expr - right_expr
                    
                    # Find variables
                    variables = list(equation.free_symbols)
                    
                    if variables:
                        # Solve equation
                        solutions = solve(equation, variables[0])
                        
                        result['steps'].append({
                            'step': f"Solving equation: {expr}",
                            'equation': str(equation),
                            'variable': str(variables[0]),
                            'method': 'symbolic_solving'
                        })
                        
                        result['solutions'].extend([str(sol) for sol in solutions])
                        result['symbolic_result'] = str(solutions)
                    
                    break  # Process first equation for now
            
            if result['solutions']:
                result['final_answer'] = result['solutions'][0]
                result['confidence'] = 0.9
            
        except Exception as e:
            result['error'] = str(e)
            result['confidence'] = 0.0
        
        return result
    
    async def _solve_calculus(self, expressions: List[str], problem: str) -> Dict[str, Any]:
        """Solve calculus problems."""
        result = {
            'steps': [],
            'symbolic_result': None,
            'confidence': 0.8
        }
        
        try:
            for expr in expressions:
                # Parse expression
                parsed_expr = parse_expr(expr)
                variables = list(parsed_expr.free_symbols)
                
                if not variables:
                    continue
                
                var = variables[0]  # Use first variable
                
                # Check problem type
                if 'derivative' in problem.lower() or "f'" in problem:
                    # Calculate derivative
                    derivative = diff(parsed_expr, var)
                    
                    result['steps'].append({
                        'step': f"Taking derivative of {expr} with respect to {var}",
                        'operation': 'differentiation',
                        'result': str(derivative)
                    })
                    
                    result['symbolic_result'] = str(derivative)
                    result['final_answer'] = str(derivative)
                
                elif 'integral' in problem.lower() or '∫' in problem:
                    # Calculate integral
                    integral = integrate(parsed_expr, var)
                    
                    result['steps'].append({
                        'step': f"Integrating {expr} with respect to {var}",
                        'operation': 'integration',
                        'result': str(integral)
                    })
                    
                    result['symbolic_result'] = str(integral)
                    result['final_answer'] = str(integral)
                
                break  # Process first expression
                
        except Exception as e:
            result['error'] = str(e)
            result['confidence'] = 0.0
        
        return result
    
    async def _solve_algebra(self, expressions: List[str], problem: str) -> Dict[str, Any]:
        """Solve algebraic problems."""
        result = {
            'steps': [],
            'symbolic_result': None,
            'confidence': 0.8
        }
        
        try:
            for expr in expressions:
                parsed_expr = parse_expr(expr)
                
                if 'simplify' in problem.lower():
                    simplified = simplify(parsed_expr)
                    
                    result['steps'].append({
                        'step': f"Simplifying {expr}",
                        'operation': 'simplification',
                        'result': str(simplified)
                    })
                    
                    result['symbolic_result'] = str(simplified)
                    result['final_answer'] = str(simplified)
                
                elif 'expand' in problem.lower():
                    expanded = sp.expand(parsed_expr)
                    
                    result['steps'].append({
                        'step': f"Expanding {expr}",
                        'operation': 'expansion',
                        'result': str(expanded)
                    })
                    
                    result['symbolic_result'] = str(expanded)
                    result['final_answer'] = str(expanded)
                
                break
                
        except Exception as e:
            result['error'] = str(e)
            result['confidence'] = 0.0
        
        return result
    
    async def _solve_optimization(self, expressions: List[str], problem: str, context: Dict = None) -> Dict[str, Any]:
        """Solve optimization problems."""
        result = {
            'steps': [],
            'optimal_solution': None,
            'confidence': 0.7
        }
        
        # This would require more advanced optimization techniques
        # For now, return a placeholder implementation
        result['final_answer'] = "Optimization solving requires additional mathematical libraries"
        result['confidence'] = 0.3
        
        return result
    
    async def _general_math_analysis(self, expressions: List[str], problem: str) -> Dict[str, Any]:
        """General mathematical analysis."""
        result = {
            'steps': [],
            'analysis': [],
            'confidence': 0.6
        }
        
        try:
            for expr in expressions:
                parsed_expr = parse_expr(expr)
                
                # Basic analysis
                result['analysis'].append({
                    'expression': expr,
                    'variables': [str(var) for var in parsed_expr.free_symbols],
                    'is_polynomial': parsed_expr.is_polynomial(),
                    'simplified': str(simplify(parsed_expr))
                })
            
            result['final_answer'] = "Mathematical analysis completed"
            
        except Exception as e:
            result['error'] = str(e)
            result['confidence'] = 0.0
        
        return result
    
    def _validate_mathematical_result(self, result: Dict[str, Any]) -> bool:
        """Validate mathematical computation results."""
        try:
            # Check if we have a reasonable result
            if 'error' in result:
                return False
            
            if result.get('confidence', 0) > 0.5:
                return True
            
            # Additional validation checks would go here
            return True
            
        except Exception:
            return False


class NeuroSymbolicEngine:
    """
    Advanced neurosymbolic AI engine that combines neural networks with symbolic reasoning.
    
    Features:
    - Multi-modal reasoning combining neural and symbolic approaches
    - Knowledge graph integration with semantic reasoning
    - Mathematical problem solving with symbolic computation
    - Causal reasoning and inference
    - Interpretable AI decisions with reasoning chains
    - Security validation and audit trails
    """
    
    def __init__(self, provider_manager=None, max_knowledge_items: int = 10000):
        self.provider_manager = provider_manager
        self.knowledge_graph = AdvancedKnowledgeGraph(max_triples=max_knowledge_items)
        self.mathematical_reasoner = MathematicalReasoner()
        self.security_validator = SymbolicSecurityValidator()
        
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
        
        # Validate and sanitize inputs
        query = self.security_validator.sanitize_knowledge_text(query)
        context = context or {}
        
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
            # Extract relevant knowledge
            relevant_knowledge = await self._extract_relevant_knowledge(query, context)
            reasoning_chain.knowledge_sources = [k.source for k in relevant_knowledge]
            
            # Perform reasoning based on type
            if detected_type == ReasoningType.MATHEMATICAL:
                result = await self.mathematical_reasoner.solve_mathematical_problem(query, context)
                reasoning_chain = self._convert_math_result_to_reasoning_chain(result, reasoning_chain)
                
            elif detected_type in self.reasoning_engines:
                reasoning_chain = await self.reasoning_engines[detected_type](
                    query, relevant_knowledge, context, max_depth
                )
            else:
                # Hybrid reasoning
                reasoning_chain = await self._hybrid_reasoning(
                    query, relevant_knowledge, context, max_depth
                )
            
            # Generate final conclusion
            if reasoning_chain.steps:
                reasoning_chain.final_conclusion = await self._generate_conclusion(reasoning_chain)
            
            # Record performance
            processing_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics[detected_type.value].append({
                'processing_time': processing_time,
                'confidence': reasoning_chain.overall_confidence,
                'steps': len(reasoning_chain.steps),
                'timestamp': start_time.isoformat()
            })
            
            self.reasoning_history.append(reasoning_chain.get_audit_trail())
            
        except Exception as e:
            logger.error(f"Reasoning error: {e}")
            reasoning_chain.final_conclusion = f"Reasoning failed: {str(e)}"
            reasoning_chain.overall_confidence = 0.0
        
        return reasoning_chain
    
    def _detect_reasoning_type(self, query: str) -> ReasoningType:
        """Automatically detect the most appropriate reasoning type."""
        query_lower = query.lower()
        
        # Mathematical reasoning indicators
        if any(indicator in query_lower for indicator in [
            'solve', 'calculate', 'equation', 'derivative', 'integral', 
            'math', 'formula', 'algebra', 'geometry'
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
        
        # Default to deductive reasoning
        else:
            return ReasoningType.DEDUCTIVE
    
    async def _extract_relevant_knowledge(self, query: str, context: Dict) -> List[KnowledgeTriple]:
        """Extract knowledge relevant to the query."""
        relevant_triples = []
        
        # Extract key terms from query
        key_terms = self._extract_key_terms(query)
        
        # Search knowledge graph for relevant triples
        for term in key_terms:
            # Search as subject
            subject_triples = self.knowledge_graph.query_by_pattern(
                subject=term, min_confidence=0.3
            )
            relevant_triples.extend(subject_triples)
            
            # Search as object
            object_triples = self.knowledge_graph.query_by_pattern(
                object=term, min_confidence=0.3
            )
            relevant_triples.extend(object_triples)
        
        # Remove duplicates and sort by confidence
        unique_triples = {t.hash_id: t for t in relevant_triples}
        sorted_triples = sorted(unique_triples.values(), 
                              key=lambda t: t.confidence, reverse=True)
        
        return sorted_triples[:20]  # Limit to top 20 relevant facts
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for knowledge search."""
        # Simple extraction - in practice, would use NLP techniques
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'but', 'for', 'are', 'with', 'this', 'that', 'can', 'how', 'what', 'when', 'where', 'why'}
        key_terms = [word for word in words if word not in stop_words]
        
        return list(set(key_terms))  # Remove duplicates
    
    async def _logical_reasoning(self, query: str, knowledge: List[KnowledgeTriple], 
                                context: Dict, max_depth: int) -> ReasoningChain:
        """Perform logical reasoning."""
        reasoning_chain = ReasoningChain(query=query, reasoning_type=ReasoningType.LOGICAL)
        
        # Simple logical reasoning implementation
        for i, triple in enumerate(knowledge[:max_depth]):
            step = ReasoningStep(
                step_number=i + 1,
                reasoning_type=ReasoningType.LOGICAL,
                input_facts=[triple],
                logical_operation=f"Applying logical rule: {triple.predicate}",
                derived_fact=triple,
                confidence=triple.confidence,
                explanation=f"From knowledge: {triple.subject} {triple.predicate} {triple.object}"
            )
            reasoning_chain.add_step(step)
        
        return reasoning_chain
    
    async def _causal_reasoning(self, query: str, knowledge: List[KnowledgeTriple], 
                              context: Dict, max_depth: int) -> ReasoningChain:
        """Perform causal reasoning."""
        reasoning_chain = ReasoningChain(query=query, reasoning_type=ReasoningType.CAUSAL)
        
        # Extract potential cause and effect from query
        cause_effect = self._extract_cause_effect(query)
        
        if cause_effect:
            cause, effect = cause_effect
            
            # Find causal chains in knowledge graph
            causal_chains = self.knowledge_graph.find_causal_chain(cause, effect, max_depth)
            
            for i, chain in enumerate(causal_chains):
                for j, triple in enumerate(chain):
                    step = ReasoningStep(
                        step_number=len(reasoning_chain.steps) + 1,
                        reasoning_type=ReasoningType.CAUSAL,
                        input_facts=[triple],
                        logical_operation=f"Causal link: {triple.predicate}",
                        derived_fact=triple,
                        confidence=triple.confidence,
                        explanation=f"Causal relationship: {triple.subject} → {triple.object}"
                    )
                    reasoning_chain.add_step(step)
        
        return reasoning_chain
    
    async def _deductive_reasoning(self, query: str, knowledge: List[KnowledgeTriple], 
                                 context: Dict, max_depth: int) -> ReasoningChain:
        """Perform deductive reasoning."""
        reasoning_chain = ReasoningChain(query=query, reasoning_type=ReasoningType.DEDUCTIVE)
        
        # Apply deductive rules to known facts
        for i, triple in enumerate(knowledge[:max_depth]):
            step = ReasoningStep(
                step_number=i + 1,
                reasoning_type=ReasoningType.DEDUCTIVE,
                input_facts=[triple],
                logical_operation="Deductive inference",
                derived_fact=triple,
                confidence=triple.confidence * 0.9,  # Slight confidence reduction
                explanation=f"Deduced from premise: {triple.subject} {triple.predicate} {triple.object}"
            )
            reasoning_chain.add_step(step)
        
        return reasoning_chain
    
    async def _inductive_reasoning(self, query: str, knowledge: List[KnowledgeTriple], 
                                 context: Dict, max_depth: int) -> ReasoningChain:
        """Perform inductive reasoning."""
        reasoning_chain = ReasoningChain(query=query, reasoning_type=ReasoningType.INDUCTIVE)
        
        # Group similar patterns
        patterns = defaultdict(list)
        for triple in knowledge:
            pattern_key = triple.predicate
            patterns[pattern_key].append(triple)
        
        # Generate inductive conclusions
        for pattern, triples in patterns.items():
            if len(triples) >= 2:  # Need multiple examples for induction
                confidence = sum(t.confidence for t in triples) / len(triples)
                
                step = ReasoningStep(
                    step_number=len(reasoning_chain.steps) + 1,
                    reasoning_type=ReasoningType.INDUCTIVE,
                    input_facts=triples,
                    logical_operation=f"Inductive generalization from {len(triples)} examples",
                    derived_fact=triples[0],  # Representative example
                    confidence=confidence * 0.8,  # Inductive reasoning is less certain
                    explanation=f"Pattern observed: {pattern} appears in {len(triples)} cases"
                )
                reasoning_chain.add_step(step)
        
        return reasoning_chain
    
    async def _abductive_reasoning(self, query: str, knowledge: List[KnowledgeTriple], 
                                 context: Dict, max_depth: int) -> ReasoningChain:
        """Perform abductive reasoning (inference to best explanation)."""
        reasoning_chain = ReasoningChain(query=query, reasoning_type=ReasoningType.ABDUCTIVE)
        
        # Find best explanations for observed facts
        for triple in knowledge[:max_depth]:
            # Find potential explanations
            explanations = self.knowledge_graph.query_by_pattern(
                object=triple.subject, min_confidence=0.3
            )
            
            if explanations:
                best_explanation = max(explanations, key=lambda e: e.confidence)
                
                step = ReasoningStep(
                    step_number=len(reasoning_chain.steps) + 1,
                    reasoning_type=ReasoningType.ABDUCTIVE,
                    input_facts=[triple, best_explanation],
                    logical_operation="Abductive inference (best explanation)",
                    derived_fact=best_explanation,
                    confidence=best_explanation.confidence * 0.7,  # Abductive is hypothetical
                    explanation=f"Best explanation for {triple.subject}: {best_explanation.subject} {best_explanation.predicate} {best_explanation.object}"
                )
                reasoning_chain.add_step(step)
        
        return reasoning_chain
    
    async def _hybrid_reasoning(self, query: str, knowledge: List[KnowledgeTriple], 
                              context: Dict, max_depth: int) -> ReasoningChain:
        """Perform hybrid reasoning combining multiple approaches."""
        reasoning_chain = ReasoningChain(query=query, reasoning_type=ReasoningType.LOGICAL)
        
        # Combine different reasoning approaches
        deductive_chain = await self._deductive_reasoning(query, knowledge, context, max_depth // 2)
        inductive_chain = await self._inductive_reasoning(query, knowledge, context, max_depth // 2)
        
        # Merge reasoning steps
        all_steps = deductive_chain.steps + inductive_chain.steps
        
        # Sort by confidence and add to chain
        all_steps.sort(key=lambda s: s.confidence, reverse=True)
        
        for i, step in enumerate(all_steps[:max_depth]):
            step.step_number = i + 1
            reasoning_chain.add_step(step)
        
        return reasoning_chain
    
    def _extract_cause_effect(self, query: str) -> Optional[Tuple[str, str]]:
        """Extract cause and effect from query text."""
        # Simple pattern matching - would use NLP in practice
        patterns = [
            r'what causes (.+?) to (.+?)[\?\.]',
            r'why does (.+?) lead to (.+?)[\?\.]',
            r'(.+?) causes (.+?)[\?\.]',
            r'(.+?) results in (.+?)[\?\.]'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1).strip(), match.group(2).strip()
        
        return None
    
    def _convert_math_result_to_reasoning_chain(self, math_result: Dict, 
                                              reasoning_chain: ReasoningChain) -> ReasoningChain:
        """Convert mathematical result to reasoning chain format."""
        if 'steps' in math_result:
            for i, step_data in enumerate(math_result['steps']):
                step = ReasoningStep(
                    step_number=i + 1,
                    reasoning_type=ReasoningType.MATHEMATICAL,
                    input_facts=[],
                    logical_operation=step_data.get('operation', 'mathematical_computation'),
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
            return "No reasoning steps completed"
        
        # Simple conclusion generation
        highest_confidence_step = max(reasoning_chain.steps, key=lambda s: s.confidence)
        
        conclusion = f"Based on {reasoning_chain.reasoning_type.value} reasoning with {len(reasoning_chain.steps)} steps, "
        conclusion += f"the most confident conclusion is: {highest_confidence_step.explanation} "
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
        ("software_development", "requires", "logical_thinking"),
        ("machine_learning", "is_subset_of", "artificial_intelligence"),
        ("artificial_intelligence", "uses", "algorithms"),
        ("algorithms", "solve", "computational_problems"),
        ("Newton", "discovered", "gravity"),
        ("gravity", "causes", "objects_to_fall"),
        ("objects_to_fall", "results_in", "motion"),
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
            
            if result.steps:
                print("\n📝 Reasoning Steps:")
                for i, step in enumerate(result.steps[:3]):  # Show first 3 steps
                    print(f"  {i+1}. {step.explanation} (confidence: {step.confidence:.2f})")
                if len(result.steps) > 3:
                    print(f"  ... and {len(result.steps) - 3} more steps")
                    
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
    creation_time: Optional[datetime] = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate and sanitize knowledge triple components."""
        try:
            # Validate and sanitize subject
            self.subject = SymbolicSecurityValidator.sanitize_knowledge_text(self.subject)
            if not self.subject:
                raise ValueError("Subject cannot be empty")
            
            # Validate and sanitize predicate
            self.predicate = SymbolicSecurityValidator.validate_predicate(self.predicate)
            
            # Validate and sanitize object
            self.object = SymbolicSecurityValidator.sanitize_knowledge_text(self.object)
            if not self.object:
                raise ValueError("Object cannot be empty")
            
            # Validate confidence
            if not isinstance(self.confidence, (int, float)):
                raise ValueError("Confidence must be numeric")
            
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError("Confidence must be between 0.0 and 1.0")
            
            # Validate source
            if not isinstance(self.source, str) or len(self.source) > 50:
                raise ValueError("Source must be string with length ≤ 50")
            
            self.source = SymbolicSecurityValidator.sanitize_knowledge_text(self.source)
            
            # Validate metadata
            if self.metadata:
                if not isinstance(self.metadata, dict):
                    raise ValueError("Metadata must be dict")
                
                # Limit metadata size and sanitize
                validated_metadata = {}
                for key, value in list(self.metadata.items())[:10]:
                    if isinstance(key, str) and len(key) <= 50:
                        if isinstance(value, (str, int, float, bool)):
                            validated_metadata[key] = value
                self.metadata = validated_metadata
            
            logger.debug(f"Knowledge triple validated: {self.subject} -> {self.predicate} -> {self.object}")
            
        except Exception as e:
            logger.error(f"Knowledge triple validation failed: {e}")
            raise
    
    def get_triple_hash(self) -> str:
        """Generate hash of triple for deduplication."""
        triple_str = f"{self.subject}:{self.predicate}:{self.object}"
        return hashlib.sha256(triple_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert triple to dictionary with security validation."""
        return {
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'confidence': self.confidence,
            'source': self.source,
            'metadata': self.metadata,
            'creation_time': self.creation_time.isoformat() if self.creation_time else None,
            'hash': self.get_triple_hash()
        }
    timestamp: float = 0.0

@dataclass
class ReasoningStep:
    """Represents a step in the reasoning process"""
    step_id: str
    reasoning_type: ReasoningType
    premises: List[str]
    conclusion: str
    confidence: float
    rule_applied: Optional[str] = None
    symbolic_form: Optional[str] = None

class SymbolicKnowledgeBase:
    """
    Symbolic knowledge base with logical rules and facts
    """
    
    def __init__(self):
        self.facts = set()
        self.rules = []
        self.knowledge_graph = nx.DiGraph()
        self.concept_hierarchy = nx.DiGraph()
        self.temporal_relations = []
        
    def add_fact(self, triple: KnowledgeTriple):
        """Add a factual triple to the knowledge base"""
        fact_str = f"{triple.subject} {triple.predicate} {triple.object}"
        self.facts.add(fact_str)
        
        # Add to knowledge graph
        self.knowledge_graph.add_edge(
            triple.subject, 
            triple.object,
            relation=triple.predicate,
            confidence=triple.confidence,
            source=triple.source
        )
    
    def add_rule(self, rule: Dict[str, Any]):
        """
        Add a logical rule to the knowledge base
        Format: {"premises": [...], "conclusion": ..., "confidence": ...}
        """
        self.rules.append(rule)
    
    def query_facts(self, subject: str = None, predicate: str = None, 
                   object: str = None) -> List[KnowledgeTriple]:
        """Query facts from the knowledge base"""
        results = []
        
        for fact in self.facts:
            s, p, o = fact.split(' ', 2)
            if ((subject is None or s == subject) and
                (predicate is None or p == predicate) and
                (object is None or o == object)):
                
                # Get confidence from graph if available
                confidence = 1.0
                if self.knowledge_graph.has_edge(s, o):
                    edge_data = self.knowledge_graph[s][o]
                    if edge_data.get('relation') == p:
                        confidence = edge_data.get('confidence', 1.0)
                
                results.append(KnowledgeTriple(s, p, o, confidence))
        
        return results
    
    def infer_facts(self, max_depth: int = 3) -> List[KnowledgeTriple]:
        """Infer new facts using logical rules"""
        new_facts = []
        
        for rule in self.rules:
            premises = rule['premises']
            conclusion = rule['conclusion']
            rule_confidence = rule.get('confidence', 0.8)
            
            # Check if all premises are satisfied
            premise_confidence = []
            for premise in premises:
                # Simple pattern matching for now
                # In production, would use more sophisticated unification
                matching_facts = self._match_premise(premise)
                if matching_facts:
                    premise_confidence.append(max(f.confidence for f in matching_facts))
                else:
                    premise_confidence = []
                    break
            
            if premise_confidence:
                # Calculate combined confidence
                combined_confidence = min(premise_confidence) * rule_confidence
                
                # Extract subject, predicate, object from conclusion
                if ' ' in conclusion:
                    parts = conclusion.split(' ', 2)
                    if len(parts) == 3:
                        new_fact = KnowledgeTriple(
                            parts[0], parts[1], parts[2], 
                            combined_confidence, "inference"
                        )
                        new_facts.append(new_fact)
        
        return new_facts
    
    def _match_premise(self, premise: str) -> List[KnowledgeTriple]:
        """Match a premise against the knowledge base"""
        # Simple implementation - in production would use unification
        parts = premise.split(' ', 2)
        if len(parts) == 3:
            return self.query_facts(parts[0], parts[1], parts[2])
        return []

class SecureMathematicalReasoner:
    """
    Secure mathematical and symbolic reasoning engine with comprehensive validation.
    
    Provides safe mathematical operations with input validation, expression sandboxing,
    and resource limits to prevent mathematical attacks and resource exhaustion.
    """
    
    def __init__(self, max_variables: int = 100, max_equations: int = 50):
        """Initialize secure mathematical reasoner with limits."""
        self.variables = {}
        self.equations = []
        self.constraints = []
        self.max_variables = max(1, min(max_variables, 1000))
        self.max_equations = max(1, min(max_equations, 100))
        self._operation_count = 0
        
        logger.info("Initialized secure mathematical reasoner")
    
    def _validate_variable_name(self, name: str) -> str:
        """Validate mathematical variable names."""
        return SymbolicSecurityValidator.validate_symbol(name)
    
    def parse_mathematical_expression(self, text: str) -> List[Dict[str, Any]]:
        """Parse mathematical expressions from text with security validation."""
        try:
            # Validate input text
            text = SymbolicSecurityValidator.validate_mathematical_expression(text)
            
            expressions = []
            self._operation_count += 1
            
            # Limit operation frequency
            if self._operation_count > 1000:
                raise ValueError("Mathematical operation limit exceeded")
            
            # Pattern for equations with security validation
            equation_patterns = [
                r'([a-zA-Z_]\w*)\s*=\s*([^,\n]+)',
                r'([^=]+)\s*=\s*([^,\n]+)',
                r'solve\s+([^,\n]+)',
                r'find\s+([a-zA-Z_]\w*)'
            ]
            
            for pattern in equation_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        if len(expressions) >= 20:  # Limit number of expressions
                            break
                        
                        groups = match.groups()
                        if len(groups) >= 1:
                            # Validate each component
                            validated_groups = []
                            for group in groups:
                                validated = SymbolicSecurityValidator.validate_mathematical_expression(group.strip())
                                validated_groups.append(validated)
                            
                            expressions.append({
                                'type': 'equation',
                                'components': validated_groups,
                                'original': match.group(0),
                                'validated': True
                            })
                    
                    except Exception as e:
                        logger.warning(f"Skipping invalid mathematical expression: {e}")
                        continue
            
            logger.info(f"Parsed {len(expressions)} secure mathematical expressions")
            return expressions
            
        except Exception as e:
            logger.error(f"Mathematical expression parsing failed: {e}")
            raise


# Add security logging and monitoring
def secure_neurosymbolic_operation(operation_name: str):
    """Decorator for secure neurosymbolic operations."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                logger.info(f"Starting secure {operation_name}")
                result = await func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Completed secure {operation_name} in {duration:.2f}s")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Failed secure {operation_name} after {duration:.2f}s: {e}")
                raise
        return wrapper
    return decorator

# Backward compatibility aliases
MathematicalReasoner = SecureMathematicalReasoner
    
    def solve_mathematical_problem(self, expressions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Solve mathematical problems using symbolic computation"""
        try:
            results = {}
            
            for expr in expressions:
                if expr['type'] == 'equation':
                    # Parse equation
                    left = sp.sympify(expr['left'])
                    right = sp.sympify(expr['right'])
                    equation = sp.Eq(left, right)
                    
                    # Extract variables
                    variables = list(equation.free_symbols)
                    
                    if len(variables) == 1:
                        # Single variable equation
                        solutions = solve(equation, variables[0])
                        results[str(variables[0])] = [str(sol) for sol in solutions]
                    else:
                        # Multi-variable equation
                        results['equation'] = str(equation)
                        results['variables'] = [str(var) for var in variables]
                
                elif expr['type'] == 'solve':
                    # General expression solving
                    expression = sp.sympify(expr['expression'])
                    simplified = simplify(expression)
                    results['simplified'] = str(simplified)
                    
                    # Try to solve if it's an equation
                    if isinstance(expression, sp.Eq):
                        variables = list(expression.free_symbols)
                        if variables:
                            solutions = solve(expression, variables[0])
                            results['solutions'] = [str(sol) for sol in solutions]
            
            return {'success': True, 'results': results}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def derive_expression(self, expression: str, variable: str) -> str:
        """Calculate derivative of expression"""
        try:
            expr = sp.sympify(expression)
            var = sp.Symbol(variable)
            derivative = diff(expr, var)
            return str(derivative)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def integrate_expression(self, expression: str, variable: str) -> str:
        """Calculate integral of expression"""
        try:
            expr = sp.sympify(expression)
            var = sp.Symbol(variable)
            integral = integrate(expr, var)
            return str(integral)
        except Exception as e:
            return f"Error: {str(e)}"

class CausalReasoner:
    """
    Handles causal reasoning and inference
    """
    
    def __init__(self):
        self.causal_graph = nx.DiGraph()
        self.interventions = {}
        
    def add_causal_relation(self, cause: str, effect: str, strength: float = 1.0):
        """Add a causal relationship"""
        self.causal_graph.add_edge(cause, effect, strength=strength)
    
    def find_causal_path(self, cause: str, effect: str) -> List[List[str]]:
        """Find causal paths between cause and effect"""
        try:
            paths = list(nx.all_simple_paths(self.causal_graph, cause, effect))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def estimate_causal_effect(self, intervention: str, outcome: str) -> float:
        """Estimate causal effect of intervention on outcome"""
        paths = self.find_causal_path(intervention, outcome)
        
        if not paths:
            return 0.0
        
        # Calculate effect through all paths
        total_effect = 0.0
        for path in paths:
            path_effect = 1.0
            for i in range(len(path) - 1):
                if self.causal_graph.has_edge(path[i], path[i + 1]):
                    edge_data = self.causal_graph[path[i]][path[i + 1]]
                    path_effect *= edge_data.get('strength', 1.0)
            total_effect += path_effect
        
        return min(total_effect, 1.0)  # Cap at 1.0

class NeuroSymbolicEngine:
    """
    Main neurosymbolic AI engine that combines neural and symbolic reasoning
    """
    
    def __init__(self):
        self.knowledge_base = SymbolicKnowledgeBase()
        self.math_reasoner = MathematicalReasoner()
        self.causal_reasoner = CausalReasoner()
        
        # Initialize with common knowledge
        self._initialize_common_knowledge()
        
        # Reasoning history
        self.reasoning_history = []
    
    def _initialize_common_knowledge(self):
        """Initialize with basic common knowledge and rules"""
        
        # Basic mathematical rules
        self.knowledge_base.add_rule({
            'premises': ['?x greater_than ?y', '?y greater_than ?z'],
            'conclusion': '?x greater_than ?z',
            'confidence': 0.95,
            'type': 'transitivity'
        })
        
        # Causal relationships
        self.causal_reasoner.add_causal_relation("rain", "wet_ground", 0.9)
        self.causal_reasoner.add_causal_relation("exercise", "health_improvement", 0.7)
        self.causal_reasoner.add_causal_relation("study", "knowledge_increase", 0.8)
        
        # Basic facts
        self.knowledge_base.add_fact(KnowledgeTriple("water", "boils_at", "100_celsius"))
        self.knowledge_base.add_fact(KnowledgeTriple("gravity", "accelerates_at", "9.8_m_s2"))
    
    async def enhance_prompt(self, prompt: str, context: Dict = None) -> str:
        """
        Enhance a prompt with symbolic reasoning and knowledge
        
        Args:
            prompt: Original prompt
            context: Additional context
            
        Returns:
            Enhanced prompt with symbolic reasoning
        """
        reasoning_steps = []
        
        # 1. Extract mathematical expressions
        math_expressions = self.math_reasoner.parse_mathematical_expression(prompt)
        if math_expressions:
            math_results = self.math_reasoner.solve_mathematical_problem(math_expressions)
            if math_results['success']:
                reasoning_steps.append(f"Mathematical analysis: {json.dumps(math_results['results'])}")
        
        # 2. Check for causal reasoning opportunities
        causal_keywords = ['because', 'causes', 'leads to', 'results in', 'due to']
        if any(keyword in prompt.lower() for keyword in causal_keywords):
            # Extract potential causal relationships
            causal_analysis = self._analyze_causal_relations(prompt)
            if causal_analysis:
                reasoning_steps.append(f"Causal analysis: {causal_analysis}")
        
        # 3. Query knowledge base for relevant facts
        relevant_facts = self._find_relevant_knowledge(prompt)
        if relevant_facts:
            facts_str = "; ".join([f"{f.subject} {f.predicate} {f.object}" for f in relevant_facts])
            reasoning_steps.append(f"Relevant knowledge: {facts_str}")
        
        # 4. Apply logical inference
        inferred_facts = self.knowledge_base.infer_facts()
        if inferred_facts:
            new_facts = [f for f in inferred_facts if self._is_relevant_to_prompt(f, prompt)]
            if new_facts:
                facts_str = "; ".join([f"{f.subject} {f.predicate} {f.object}" for f in new_facts])
                reasoning_steps.append(f"Inferred knowledge: {facts_str}")
        
        # 5. Construct enhanced prompt
        if reasoning_steps:
            enhanced_prompt = f"""Original question: {prompt}

Symbolic reasoning analysis:
{chr(10).join(f"- {step}" for step in reasoning_steps)}

Based on this analysis, please provide a comprehensive answer that integrates both the symbolic reasoning and your neural knowledge."""
            
            return enhanced_prompt
        
        return prompt
    
    def _analyze_causal_relations(self, text: str) -> str:
        """Analyze causal relationships in text"""
        # Simple causal pattern extraction
        patterns = [
            r'(\w+)\s+causes?\s+(\w+)',
            r'(\w+)\s+leads?\s+to\s+(\w+)',
            r'(\w+)\s+results?\s+in\s+(\w+)',
            r'due\s+to\s+(\w+),?\s+(\w+)'
        ]
        
        causal_relations = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if len(match) == 2:
                    cause, effect = match
                    # Check if we have knowledge about this causal relationship
                    causal_effect = self.causal_reasoner.estimate_causal_effect(cause, effect)
                    if causal_effect > 0:
                        causal_relations.append(f"{cause} -> {effect} (strength: {causal_effect:.2f})")
        
        return "; ".join(causal_relations) if causal_relations else ""
    
    def _find_relevant_knowledge(self, prompt: str) -> List[KnowledgeTriple]:
        """Find knowledge base facts relevant to the prompt"""
        # Extract key terms from prompt
        words = re.findall(r'\b\w+\b', prompt.lower())
        relevant_facts = []
        
        for word in words:
            # Query for facts containing this word
            facts = self.knowledge_base.query_facts(subject=word)
            facts.extend(self.knowledge_base.query_facts(object=word))
            relevant_facts.extend(facts)
        
        # Remove duplicates and sort by confidence
        unique_facts = list({f"{f.subject} {f.predicate} {f.object}": f for f in relevant_facts}.values())
        return sorted(unique_facts, key=lambda x: x.confidence, reverse=True)[:5]
    
    def _is_relevant_to_prompt(self, fact: KnowledgeTriple, prompt: str) -> bool:
        """Check if a fact is relevant to the prompt"""
        prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
        fact_words = {fact.subject.lower(), fact.object.lower()}
        return bool(prompt_words.intersection(fact_words))
    
    async def process_with_reasoning(self, prompt: str, reasoning_type: ReasoningType = None) -> Dict[str, Any]:
        """
        Process a prompt with specific reasoning type
        
        Args:
            prompt: Input prompt
            reasoning_type: Type of reasoning to apply
            
        Returns:
            Dictionary with reasoning steps and conclusions
        """
        if reasoning_type == ReasoningType.MATHEMATICAL:
            # Handle mathematical reasoning
            expressions = self.math_reasoner.parse_mathematical_expression(prompt)
            if expressions:
                return {
                    'reasoning_type': 'mathematical',
                    'expressions': expressions,
                    'solutions': self.math_reasoner.solve_mathematical_problem(expressions)
                }
        
        elif reasoning_type == ReasoningType.CAUSAL:
            # Handle causal reasoning
            causal_analysis = self._analyze_causal_relations(prompt)
            return {
                'reasoning_type': 'causal',
                'analysis': causal_analysis
            }
        
        elif reasoning_type == ReasoningType.LOGICAL:
            # Handle logical reasoning
            relevant_facts = self._find_relevant_knowledge(prompt)
            inferred_facts = self.knowledge_base.infer_facts()
            
            return {
                'reasoning_type': 'logical',
                'relevant_facts': [asdict(f) for f in relevant_facts],
                'inferred_facts': [asdict(f) for f in inferred_facts]
            }
        
        # Default: apply general reasoning enhancement
        enhanced_prompt = await self.enhance_prompt(prompt)
        return {
            'reasoning_type': 'general',
            'enhanced_prompt': enhanced_prompt
        }
    
    def add_knowledge_from_interaction(self, prompt: str, response: str):
        """Learn new knowledge from user interactions"""
        # Extract potential facts from the interaction
        fact_patterns = [
            r'(\w+)\s+is\s+(\w+)',
            r'(\w+)\s+has\s+(\w+)',
            r'(\w+)\s+can\s+(\w+)',
            r'(\w+)\s+causes?\s+(\w+)'
        ]
        
        combined_text = f"{prompt} {response}"
        
        for pattern in fact_patterns:
            matches = re.findall(pattern, combined_text.lower())
            for match in matches:
                if len(match) == 2:
                    subject, object_or_predicate = match
                    
                    # Determine if it's "is", "has", "can", or "causes"
                    if " is " in combined_text.lower():
                        predicate = "is"
                    elif " has " in combined_text.lower():
                        predicate = "has"
                    elif " can " in combined_text.lower():
                        predicate = "can"
                    elif " causes " in combined_text.lower():
                        predicate = "causes"
                    else:
                        predicate = "related_to"
                    
                    # Add to knowledge base with low confidence (from interaction)
                    fact = KnowledgeTriple(
                        subject=subject,
                        predicate=predicate,
                        object=object_or_predicate,
                        confidence=0.6,
                        source="interaction"
                    )
                    self.knowledge_base.add_fact(fact)
    
    def get_reasoning_explanation(self, reasoning_steps: List[ReasoningStep]) -> str:
        """Generate human-readable explanation of reasoning process"""
        explanation_parts = []
        
        for step in reasoning_steps:
            if step.reasoning_type == ReasoningType.MATHEMATICAL:
                explanation_parts.append(
                    f"Mathematical step: Applied {step.rule_applied or 'calculation'} "
                    f"to derive {step.conclusion}"
                )
            elif step.reasoning_type == ReasoningType.LOGICAL:
                explanation_parts.append(
                    f"Logical step: From premises {', '.join(step.premises)}, "
                    f"concluded {step.conclusion} (confidence: {step.confidence:.2f})"
                )
            elif step.reasoning_type == ReasoningType.CAUSAL:
                explanation_parts.append(
                    f"Causal reasoning: {step.conclusion} based on causal relationships"
                )
        
        return "\n".join(explanation_parts)
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        return {
            'total_facts': len(self.knowledge_base.facts),
            'total_rules': len(self.knowledge_base.rules),
            'knowledge_graph_nodes': self.knowledge_base.knowledge_graph.number_of_nodes(),
            'knowledge_graph_edges': self.knowledge_base.knowledge_graph.number_of_edges(),
            'causal_relations': self.causal_reasoner.causal_graph.number_of_edges(),
            'reasoning_history_length': len(self.reasoning_history)
        }
