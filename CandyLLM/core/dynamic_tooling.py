"""
Architecture-agnostic dynamic tooling system for CandyLLM.

This module implements autonomous tool synthesis, secure execution, registry management,
and lifecycle control without being tied to any specific cloud provider.
"""

import asyncio
import json
import os
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import docker
import subprocess
import hashlib
from datetime import datetime, timedelta

from ..types import CandyResponse, ToolMetadata, ToolResult


class DecisionAction(Enum):
    """Actions available for build-vs-search policy."""
    REUSE = "reuse"
    ADAPT = "adapt" 
    SYNTHESIZE = "synthesize"


class PromotionStatus(Enum):
    """Tool promotion pipeline statuses."""
    PENDING = "pending"
    TESTING = "testing"
    SECURITY_SCAN = "security_scan"
    POLICY_CHECK = "policy_check"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ToolSpec:
    """Tool specification for synthesis."""
    name: str
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    capabilities: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    risk_level: str = "low"


@dataclass
class CandidateTool:
    """A synthesized tool awaiting promotion."""
    spec: ToolSpec
    code: str
    tests: str
    manifest: Dict[str, Any]
    sbom: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    promotion_status: PromotionStatus = PromotionStatus.PENDING


@dataclass
class ToolMetrics:
    """Tool usage and quality metrics."""
    usage_count: int = 0
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0
    last_used: Optional[datetime] = None
    reuse_frequency: float = 0.0
    quality_score: float = 1.0


# Abstract interfaces for pluggable components

class SandboxRunner(ABC):
    """Abstract base for secure code execution."""
    
    @abstractmethod
    async def execute(self, code: str, inputs: Dict[str, Any], 
                     timeout: int = 30) -> Dict[str, Any]:
        """Execute code in a secure sandbox."""
        pass
    
    @abstractmethod
    async def test(self, code: str, tests: str, timeout: int = 60) -> bool:
        """Run tests in sandbox."""
        pass


class ToolStorage(ABC):
    """Abstract storage for tool artifacts."""
    
    @abstractmethod
    async def store_tool(self, tool_id: str, version: str, 
                        artifacts: Dict[str, Any]) -> str:
        """Store tool artifacts and return storage path."""
        pass
    
    @abstractmethod
    async def get_tool(self, tool_id: str, version: str) -> Dict[str, Any]:
        """Retrieve tool artifacts."""
        pass
    
    @abstractmethod
    async def list_tools(self, category: Optional[str] = None) -> List[ToolMetadata]:
        """List available tools."""
        pass


class EmbeddingProvider(ABC):
    """Abstract embedding service for tool discovery."""
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: List[float], 
                    top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar tools by embedding."""
        pass


class LLMProvider(ABC):
    """Abstract LLM provider for synthesis."""
    
    @abstractmethod
    async def generate_spec(self, task_description: str) -> ToolSpec:
        """Generate tool specification from task."""
        pass
    
    @abstractmethod
    async def generate_code(self, spec: ToolSpec) -> str:
        """Generate tool implementation."""
        pass
    
    @abstractmethod
    async def generate_tests(self, spec: ToolSpec, code: str) -> str:
        """Generate comprehensive tests."""
        pass


# Concrete implementations

class DockerSandboxRunner(SandboxRunner):
    """Docker-based secure sandbox execution."""
    
    def __init__(self, image: str = "python:3.11-slim", 
                 network_mode: str = "none"):
        self.image = image
        self.network_mode = network_mode
        self.client = docker.from_env()
    
    async def execute(self, code: str, inputs: Dict[str, Any], 
                     timeout: int = 30) -> Dict[str, Any]:
        """Execute code in isolated Docker container."""
        
        # Create temporary directory for code
        with tempfile.TemporaryDirectory() as temp_dir:
            code_path = Path(temp_dir) / "tool.py"
            input_path = Path(temp_dir) / "inputs.json"
            
            # Write code and inputs
            code_path.write_text(code)
            input_path.write_text(json.dumps(inputs))
            
            # Create execution wrapper
            wrapper_code = f"""
import json
import sys
import traceback
from pathlib import Path

try:
    # Load inputs
    with open('/app/inputs.json') as f:
        inputs = json.load(f)
    
    # Import and execute tool
    sys.path.insert(0, '/app')
    from tool import main
    
    result = main(**inputs)
    
    # Output result
    print(json.dumps({{"success": True, "result": result}}))
    
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e), "traceback": traceback.format_exc()}}))
"""
            
            wrapper_path = Path(temp_dir) / "wrapper.py"
            wrapper_path.write_text(wrapper_code)
            
            # Run in container
            try:
                container = self.client.containers.run(
                    image=self.image,
                    command=["python", "/app/wrapper.py"],
                    volumes={temp_dir: {"bind": "/app", "mode": "ro"}},
                    network_mode=self.network_mode,
                    user="65534:65534",  # nobody user
                    read_only=True,
                    mem_limit="512m",
                    nano_cpus=1_000_000_000,  # 1 CPU
                    remove=True,
                    timeout=timeout,
                    detach=False
                )
                
                output = container.decode('utf-8').strip()
                return json.loads(output)
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Container execution failed: {str(e)}"
                }
    
    async def test(self, code: str, tests: str, timeout: int = 60) -> bool:
        """Run tests in sandbox."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            code_path = Path(temp_dir) / "tool.py"
            test_path = Path(temp_dir) / "test_tool.py"
            
            code_path.write_text(code)
            test_path.write_text(tests)
            
            try:
                container = self.client.containers.run(
                    image=self.image,
                    command=["python", "-m", "pytest", "/app/test_tool.py", "-v"],
                    volumes={temp_dir: {"bind": "/app", "mode": "ro"}},
                    network_mode=self.network_mode,
                    user="65534:65534",
                    read_only=True,
                    mem_limit="512m",
                    remove=True,
                    timeout=timeout,
                    detach=False
                )
                
                return True
                
            except Exception:
                return False


class FileSystemStorage(ToolStorage):
    """File system-based tool storage."""
    
    def __init__(self, base_path: str = "~/.candyllm/tools"):
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Registry metadata file
        self.registry_file = self.base_path / "registry.json"
        if not self.registry_file.exists():
            self.registry_file.write_text(json.dumps({}))
    
    async def store_tool(self, tool_id: str, version: str, 
                        artifacts: Dict[str, Any]) -> str:
        """Store tool artifacts to filesystem."""
        
        tool_dir = self.base_path / tool_id / version
        tool_dir.mkdir(parents=True, exist_ok=True)
        
        # Store each artifact
        for name, content in artifacts.items():
            if isinstance(content, str):
                (tool_dir / name).write_text(content)
            else:
                (tool_dir / name).write_text(json.dumps(content, indent=2))
        
        # Update registry
        registry = json.loads(self.registry_file.read_text())
        if tool_id not in registry:
            registry[tool_id] = {}
        
        registry[tool_id][version] = {
            "path": str(tool_dir),
            "stored_at": datetime.now().isoformat(),
            "artifacts": list(artifacts.keys())
        }
        
        self.registry_file.write_text(json.dumps(registry, indent=2))
        return str(tool_dir)
    
    async def get_tool(self, tool_id: str, version: str) -> Dict[str, Any]:
        """Retrieve tool artifacts."""
        
        tool_dir = self.base_path / tool_id / version
        if not tool_dir.exists():
            raise FileNotFoundError(f"Tool {tool_id}@{version} not found")
        
        artifacts = {}
        for file_path in tool_dir.iterdir():
            if file_path.is_file():
                try:
                    # Try to parse as JSON first
                    artifacts[file_path.name] = json.loads(file_path.read_text())
                except json.JSONDecodeError:
                    # Fall back to plain text
                    artifacts[file_path.name] = file_path.read_text()
        
        return artifacts
    
    async def list_tools(self, category: Optional[str] = None) -> List[ToolMetadata]:
        """List available tools."""
        
        registry = json.loads(self.registry_file.read_text())
        tools = []
        
        for tool_id, versions in registry.items():
            for version, metadata in versions.items():
                try:
                    # Load manifest for tool metadata
                    tool_dir = Path(metadata["path"])
                    manifest_path = tool_dir / "manifest.json"
                    
                    if manifest_path.exists():
                        manifest = json.loads(manifest_path.read_text())
                        
                        # Filter by category if specified
                        if category and manifest.get("category") != category:
                            continue
                        
                        tools.append(ToolMetadata(
                            name=tool_id,
                            version=version,
                            description=manifest.get("description", ""),
                            category=manifest.get("category", "general"),
                            input_schema=manifest.get("input_schema", {}),
                            output_schema=manifest.get("output_schema", {}),
                            risk_level=manifest.get("risk_level", "low"),
                            quality_score=manifest.get("quality_score", 1.0)
                        ))
                except Exception:
                    continue
        
        return tools


class BuildVsSearchPolicy:
    """Decides whether to reuse, adapt, or synthesize tools."""
    
    def __init__(self, reuse_threshold: float = 0.8, 
                 synthesis_threshold: float = 0.3):
        self.reuse_threshold = reuse_threshold
        self.synthesis_threshold = synthesis_threshold
    
    async def decide(self, task_spec: ToolSpec, 
                    candidates: List[ToolMetadata]) -> DecisionAction:
        """Make build-vs-search decision."""
        
        if not candidates:
            return DecisionAction.SYNTHESIZE
        
        # Find best candidate by similarity (simplified)
        best_candidate = max(candidates, 
                           key=lambda c: self._calculate_similarity(task_spec, c))
        
        similarity = self._calculate_similarity(task_spec, best_candidate)
        
        if similarity >= self.reuse_threshold:
            return DecisionAction.REUSE
        elif similarity >= self.synthesis_threshold:
            return DecisionAction.ADAPT
        else:
            return DecisionAction.SYNTHESIZE
    
    def _calculate_similarity(self, spec: ToolSpec, 
                            candidate: ToolMetadata) -> float:
        """Calculate similarity between spec and candidate tool."""
        
        # Simplified similarity based on description overlap
        spec_words = set(spec.description.lower().split())
        candidate_words = set(candidate.description.lower().split())
        
        if not spec_words or not candidate_words:
            return 0.0
        
        intersection = spec_words.intersection(candidate_words)
        union = spec_words.union(candidate_words)
        
        return len(intersection) / len(union) if union else 0.0


class PromotionGate:
    """Verifies and approves tools for registry admission."""
    
    def __init__(self, sandbox: SandboxRunner):
        self.sandbox = sandbox
        self.security_checks = [
            self._check_imports,
            self._check_network_calls,
            self._check_file_operations
        ]
    
    async def evaluate(self, candidate: CandidateTool) -> bool:
        """Evaluate candidate tool for promotion."""
        
        try:
            # Run tests
            tests_pass = await self.sandbox.test(candidate.code, candidate.tests)
            if not tests_pass:
                return False
            
            # Security checks
            for check in self.security_checks:
                if not await check(candidate.code):
                    return False
            
            # Static analysis (simplified)
            if not self._static_analysis(candidate.code):
                return False
            
            return True
            
        except Exception:
            return False
    
    async def _check_imports(self, code: str) -> bool:
        """Check for dangerous imports."""
        dangerous = ["subprocess", "os.system", "eval", "exec", "__import__"]
        return not any(danger in code for danger in dangerous)
    
    async def _check_network_calls(self, code: str) -> bool:
        """Check for network operations."""
        network_patterns = ["requests.", "urllib.", "socket.", "http."]
        return not any(pattern in code for pattern in network_patterns)
    
    async def _check_file_operations(self, code: str) -> bool:
        """Check for file system operations."""
        file_patterns = ["open(", "file(", "Path(", "os.path"]
        # Allow read-only operations in specific contexts
        return True  # Simplified for demo
    
    def _static_analysis(self, code: str) -> bool:
        """Run static analysis checks."""
        # In real implementation, use tools like bandit, pylint
        return True  # Simplified for demo


class LifecycleManager:
    """Manages tool caching, eviction, and lifecycle policies."""
    
    def __init__(self, storage: ToolStorage, max_tools: int = 1000):
        self.storage = storage
        self.max_tools = max_tools
        self.metrics: Dict[str, ToolMetrics] = {}
    
    async def update_metrics(self, tool_id: str, version: str, 
                           latency_ms: float, success: bool):
        """Update tool usage metrics."""
        
        key = f"{tool_id}@{version}"
        if key not in self.metrics:
            self.metrics[key] = ToolMetrics()
        
        metrics = self.metrics[key]
        metrics.usage_count += 1
        metrics.last_used = datetime.now()
        
        # Update success rate (exponential moving average)
        alpha = 0.1
        metrics.success_rate = (alpha * (1 if success else 0) + 
                              (1 - alpha) * metrics.success_rate)
        
        # Update average latency
        metrics.avg_latency_ms = (metrics.avg_latency_ms * (metrics.usage_count - 1) + 
                                latency_ms) / metrics.usage_count
        
        # Update quality score (composite metric)
        metrics.quality_score = (metrics.success_rate * 0.6 + 
                               min(1.0, 1000.0 / metrics.avg_latency_ms) * 0.3 +
                               min(1.0, metrics.usage_count / 10.0) * 0.1)
    
    async def evict_tools(self) -> List[str]:
        """Evict low-value tools based on LFU + quality."""
        
        if len(self.metrics) <= self.max_tools:
            return []
        
        # Score tools for eviction (lower = more likely to evict)
        scored_tools = []
        for tool_key, metrics in self.metrics.items():
            # Recency factor
            days_since_use = 0
            if metrics.last_used:
                days_since_use = (datetime.now() - metrics.last_used).days
            
            recency_factor = max(0.1, 1.0 - days_since_use / 30.0)
            
            score = (metrics.usage_count * 0.4 + 
                    metrics.quality_score * 0.4 + 
                    recency_factor * 0.2)
            
            scored_tools.append((score, tool_key))
        
        # Sort by score and evict lowest
        scored_tools.sort()
        to_evict = scored_tools[:len(scored_tools) - self.max_tools]
        
        evicted = []
        for _, tool_key in to_evict:
            # Remove from metrics
            del self.metrics[tool_key]
            evicted.append(tool_key)
        
        return evicted


class DynamicToolingEngine:
    """Main engine for autonomous dynamic tooling."""
    
    def __init__(self, 
                 llm_provider: LLMProvider,
                 sandbox: SandboxRunner,
                 storage: ToolStorage,
                 embedding_provider: Optional[EmbeddingProvider] = None):
        
        self.llm = llm_provider
        self.sandbox = sandbox
        self.storage = storage
        self.embedding = embedding_provider
        
        self.policy = BuildVsSearchPolicy()
        self.promotion_gate = PromotionGate(sandbox)
        self.lifecycle = LifecycleManager(storage)
        
        self.pending_tools: Dict[str, CandidateTool] = {}
    
    async def solve_with_dynamic_tooling(self, task_description: str) -> CandyResponse:
        """Solve a task using dynamic tooling approach."""
        
        start_time = time.time()
        
        try:
            # 1. Generate tool specification
            spec = await self.llm.generate_spec(task_description)
            
            # 2. Search for existing tools
            candidates = await self._discover_tools(spec)
            
            # 3. Make build-vs-search decision
            action = await self.policy.decide(spec, candidates)
            
            if action == DecisionAction.REUSE and candidates:
                # Use existing tool
                tool = candidates[0]
                result = await self._invoke_tool(tool, spec.inputs)
                
                # Update metrics
                latency = (time.time() - start_time) * 1000
                await self.lifecycle.update_metrics(
                    tool.name, tool.version, latency, result.get("success", False)
                )
                
                return CandyResponse(
                    content=result.get("result", ""),
                    metadata={
                        "action": "reused",
                        "tool_used": f"{tool.name}@{tool.version}",
                        "latency_ms": latency
                    }
                )
            
            elif action == DecisionAction.SYNTHESIZE:
                # Create new tool
                tool_id = await self._synthesize_tool(spec)
                
                if tool_id:
                    # Promote and use
                    promoted = await self._promote_tool(tool_id)
                    if promoted:
                        # Tool is now available for use
                        tools = await self.storage.list_tools()
                        new_tool = next(t for t in tools if t.name == tool_id)
                        
                        result = await self._invoke_tool(new_tool, spec.inputs)
                        
                        latency = (time.time() - start_time) * 1000
                        await self.lifecycle.update_metrics(
                            new_tool.name, new_tool.version, latency, 
                            result.get("success", False)
                        )
                        
                        return CandyResponse(
                            content=result.get("result", ""),
                            metadata={
                                "action": "synthesized",
                                "tool_created": f"{new_tool.name}@{new_tool.version}",
                                "latency_ms": latency
                            }
                        )
            
            # Fallback
            return CandyResponse(
                content="Unable to solve task with dynamic tooling",
                metadata={"action": "failed", "reason": "no suitable approach"}
            )
            
        except Exception as e:
            return CandyResponse(
                content=f"Dynamic tooling error: {str(e)}",
                metadata={"action": "error", "error": str(e)}
            )
    
    async def _discover_tools(self, spec: ToolSpec) -> List[ToolMetadata]:
        """Discover relevant tools for the specification."""
        
        # Get all tools in relevant category
        tools = await self.storage.list_tools()
        
        # If we have embeddings, use semantic search
        if self.embedding:
            try:
                query_embedding = await self.embedding.embed(spec.description)
                semantic_results = await self.embedding.search(query_embedding)
                
                # Match with available tools
                semantic_tool_ids = {r["tool_id"] for r in semantic_results}
                tools = [t for t in tools if t.name in semantic_tool_ids]
            except Exception:
                pass
        
        # Filter by capability overlap
        relevant_tools = []
        for tool in tools:
            if any(cap in spec.capabilities for cap in tool.input_schema.get("capabilities", [])):
                relevant_tools.append(tool)
        
        return relevant_tools[:10]  # Top 10 candidates
    
    async def _synthesize_tool(self, spec: ToolSpec) -> Optional[str]:
        """Synthesize a new tool from specification."""
        
        try:
            # Generate code and tests
            code = await self.llm.generate_code(spec)
            tests = await self.llm.generate_tests(spec, code)
            
            # Create manifest
            manifest = {
                "name": spec.name,
                "version": "0.1.0",
                "description": spec.description,
                "input_schema": spec.inputs,
                "output_schema": spec.outputs,
                "capabilities": spec.capabilities,
                "requirements": spec.requirements,
                "risk_level": spec.risk_level,
                "created_at": datetime.now().isoformat(),
                "generator": "candyllm-dynamic-tooling"
            }
            
            # Create SBOM (simplified)
            sbom = {
                "bomFormat": "CycloneDX",
                "specVersion": "1.4",
                "components": [
                    {
                        "type": "library",
                        "name": req,
                        "version": "latest"
                    } for req in spec.requirements
                ]
            }
            
            # Create candidate
            candidate = CandidateTool(
                spec=spec,
                code=code,
                tests=tests,
                manifest=manifest,
                sbom=sbom
            )
            
            # Store as pending
            tool_id = f"synthesized_{int(time.time())}_{spec.name}"
            self.pending_tools[tool_id] = candidate
            
            return tool_id
            
        except Exception:
            return None
    
    async def _promote_tool(self, tool_id: str) -> bool:
        """Promote a pending tool to the registry."""
        
        if tool_id not in self.pending_tools:
            return False
        
        candidate = self.pending_tools[tool_id]
        
        # Run promotion gate
        if await self.promotion_gate.evaluate(candidate):
            # Store in registry
            artifacts = {
                "tool.py": candidate.code,
                "test_tool.py": candidate.tests,
                "manifest.json": candidate.manifest,
                "sbom.json": candidate.sbom
            }
            
            await self.storage.store_tool(
                candidate.spec.name, 
                candidate.manifest["version"],
                artifacts
            )
            
            # Remove from pending
            del self.pending_tools[tool_id]
            return True
        
        return False
    
    async def _invoke_tool(self, tool: ToolMetadata, 
                          inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool with given inputs."""
        
        try:
            # Get tool artifacts
            artifacts = await self.storage.get_tool(tool.name, tool.version)
            code = artifacts.get("tool.py", "")
            
            # Execute in sandbox
            result = await self.sandbox.execute(code, inputs)
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
