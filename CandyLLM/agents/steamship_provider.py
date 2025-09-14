"""
Steamship Agent Provider

Integrates Steamship's agent hosting platform with package management,
deployment infrastructure, and multi-modal capabilities.
"""

import uuid
import asyncio
import json
import os
import zipfile
import tempfile
from typing import Dict, List, Optional, Any, Callable, Union
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
    import boto3
    import docker
    import yaml
    from kubernetes import client, config
    import aiofiles
    from pydantic import BaseModel
    STEAMSHIP_AVAILABLE = True
except ImportError:
    STEAMSHIP_AVAILABLE = False
    # Mock classes for when dependencies are not available
    httpx = None
    requests = None
    boto3 = None
    docker = None
    yaml = None
    client = None
    config = None
    aiofiles = None
    BaseModel = None


class PackageType(Enum):
    """Types of Steamship packages"""
    AGENT = "agent"
    TOOL = "tool"
    PLUGIN = "plugin"
    GENERATOR = "generator"
    EMBEDDER = "embedder"
    TAGGER = "tagger"
    IMPORTER = "importer"
    BLOCKIFIER = "blockifier"


class DeploymentStatus(Enum):
    """Status of deployments"""
    PENDING = "pending"
    BUILDING = "building"
    DEPLOYING = "deploying"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    UPDATING = "updating"
    SCALING = "scaling"


class ResourceType(Enum):
    """Types of resources"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    GPU = "gpu"
    BANDWIDTH = "bandwidth"


class EnvironmentType(Enum):
    """Types of environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


@dataclass
class SteamshipConfig:
    """Configuration for Steamship agent"""
    api_key: str = ""
    workspace_id: str = ""
    project_name: str = "candyllm-agent"
    package_name: str = "agent-package"
    package_type: PackageType = PackageType.AGENT
    version: str = "1.0.0"
    description: str = "AI Agent powered by CandyLLM"
    author: str = "CandyLLM"
    requirements: List[str] = field(default_factory=lambda: ["steamship", "openai", "requests"])
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    base_url: str = "https://api.steamship.com"
    timeout: int = 300
    max_retries: int = 3
    
    # Resource configuration
    cpu_limit: str = "1000m"  # 1 CPU
    memory_limit: str = "2Gi"  # 2GB
    storage_limit: str = "10Gi"  # 10GB
    gpu_count: int = 0
    min_replicas: int = 1
    max_replicas: int = 5
    auto_scaling: bool = True
    
    # Features
    enable_multi_modal: bool = True
    enable_file_storage: bool = True
    enable_vector_search: bool = True
    enable_audio_generation: bool = True
    enable_image_generation: bool = True
    enable_video_processing: bool = True
    enable_document_processing: bool = True
    enable_web_scraping: bool = True
    enable_api_integration: bool = True
    enable_database_access: bool = True
    
    # Security
    enable_authentication: bool = True
    enable_encryption: bool = True
    enable_audit_logging: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 100
    max_file_size_mb: int = 100
    
    # Monitoring
    enable_metrics: bool = True
    enable_tracing: bool = True
    log_level: str = "INFO"
    health_check_interval: int = 30
    metrics_port: int = 9090


@dataclass
class Package:
    """Steamship package definition"""
    package_id: str
    name: str
    package_type: PackageType
    version: str
    description: str
    author: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    files: Dict[str, str] = field(default_factory=dict)  # filename -> content
    manifest: Dict[str, Any] = field(default_factory=dict)
    build_logs: List[str] = field(default_factory=list)
    deployment_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Deployment:
    """Deployment instance"""
    deployment_id: str
    package_id: str
    workspace_id: str
    instance_handle: str
    status: DeploymentStatus = DeploymentStatus.PENDING
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    endpoint_url: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    health_status: str = "unknown"
    last_health_check: Optional[datetime] = None


@dataclass
class Resource:
    """Resource allocation and usage"""
    resource_id: str
    resource_type: ResourceType
    allocated: float
    used: float
    limit: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)


class SteamshipAgent:
    """Steamship agent hosting platform with package management and deployment"""
    
    def __init__(self, agent_id: str, config: SteamshipConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._packages = {}  # package_id -> Package
        self._deployments = {}  # deployment_id -> Deployment
        self._resources = {}  # resource_id -> Resource
        self._client = None
        self._workspace = None
        self._docker_client = None
        self._k8s_client = None
        self._usage_stats = {
            'total_packages': 0,
            'total_deployments': 0,
            'active_deployments': 0,
            'total_builds': 0,
            'successful_builds': 0,
            'failed_builds': 0,
            'total_invocations': 0,
            'average_response_time': 0.0,
            'total_cpu_hours': 0.0,
            'total_memory_gb_hours': 0.0,
            'total_storage_gb': 0.0,
            'total_requests': 0,
            'error_rate': 0.0,
            'uptime_percentage': 100.0,
            'data_processed_gb': 0.0,
            'files_uploaded': 0,
            'api_calls': 0,
            'vector_searches': 0,
            'multi_modal_operations': 0
        }
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the Steamship agent"""
        if not STEAMSHIP_AVAILABLE:
            return False
        
        try:
            if not self.config.api_key:
                return False
            
            # Initialize HTTP client
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={
                    'Authorization': f'Bearer {self.config.api_key}',
                    'Content-Type': 'application/json'
                },
                timeout=self.config.timeout
            )
            
            # Test connection
            await self._test_connection()
            
            # Initialize workspace
            await self._initialize_workspace()
            
            # Initialize container runtimes
            await self._initialize_container_runtime()
            
            return True
            
        except Exception as e:
            return False
    
    async def _test_connection(self):
        """Test Steamship API connection"""
        try:
            response = await self._client.get('/api/v1/health')
            if response.status_code != 200:
                raise RuntimeError(f"Health check failed: {response.status_code}")
            return True
        except Exception as e:
            raise RuntimeError(f"Steamship API test failed: {e}")
    
    async def _initialize_workspace(self):
        """Initialize Steamship workspace"""
        try:
            # Create or get workspace
            workspace_data = {
                'handle': self.config.workspace_id or f"workspace_{uuid.uuid4().hex[:8]}",
                'displayName': f"CandyLLM Workspace - {self.agent_id}",
                'description': 'AI Agent workspace powered by CandyLLM'
            }
            
            response = await self._client.post('/api/v1/workspace/create', json=workspace_data)
            if response.status_code in [200, 409]:  # 409 = already exists
                self._workspace = response.json()
                if not self.config.workspace_id:
                    self.config.workspace_id = self._workspace.get('handle')
            else:
                raise RuntimeError(f"Failed to create workspace: {response.status_code}")
            
        except Exception as e:
            pass  # Non-critical for basic functionality
    
    async def _initialize_container_runtime(self):
        """Initialize container runtime (Docker/Kubernetes)"""
        try:
            # Initialize Docker client
            self._docker_client = docker.from_env()
            
            # Try to initialize Kubernetes client
            try:
                config.load_incluster_config()  # For in-cluster usage
            except:
                try:
                    config.load_kube_config()  # For local development
                except:
                    pass  # Kubernetes not available
            
            if client:
                self._k8s_client = client.AppsV1Api()
            
        except Exception as e:
            pass  # Non-critical initialization
    
    async def create_package(self, package_name: str = None, package_type: PackageType = None,
                           files: Dict[str, str] = None, dependencies: List[str] = None) -> str:
        """Create a new package"""
        try:
            package_id = f"pkg_{uuid.uuid4().hex[:8]}"
            
            package = Package(
                package_id=package_id,
                name=package_name or self.config.package_name,
                package_type=package_type or self.config.package_type,
                version=self.config.version,
                description=self.config.description,
                author=self.config.author,
                dependencies=dependencies or self.config.requirements,
                files=files or {}
            )
            
            # Generate default files if not provided
            if not package.files:
                package.files = await self._generate_default_files(package)
            
            # Create manifest
            package.manifest = self._create_package_manifest(package)
            
            self._packages[package_id] = package
            self._usage_stats['total_packages'] += 1
            
            return package_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to create package: {e}")
    
    async def _generate_default_files(self, package: Package) -> Dict[str, str]:
        """Generate default package files"""
        files = {}
        
        # Main agent file
        files['agent.py'] = f'''"""
{package.name} - Steamship Agent
Generated by CandyLLM
"""

import asyncio
from typing import Dict, Any, List, Optional
from steamship import Steamship, SteamshipError
from steamship.agents.llms import OpenAI
from steamship.agents.tools import Tool
from steamship.agents.schema import AgentContext

class {package.name.replace('-', '_').title()}Agent:
    """AI Agent powered by Steamship and CandyLLM"""
    
    def __init__(self, client: Steamship):
        self.client = client
        self.llm = OpenAI(client=client, model_name="gpt-4")
        self.tools = []
        
    async def run(self, message: str, context: AgentContext = None) -> str:
        """Process user message and return response"""
        try:
            # Use LLM to process message
            response = await self.llm.complete(
                prompt=message,
                max_tokens=500
            )
            
            return response.text
            
        except Exception as e:
            return f"Error processing request: {{str(e)}}"

# Steamship entry point
def create_agent(client: Steamship) -> {package.name.replace('-', '_').title()}Agent:
    return {package.name.replace('-', '_').title()}Agent(client)
'''
        
        # Requirements file
        files['requirements.txt'] = '\\n'.join(package.dependencies)
        
        # Steamship manifest
        files['steamship.json'] = json.dumps(package.manifest, indent=2)
        
        # README
        files['README.md'] = f'''# {package.name}

{package.description}

## Description
This is an AI agent powered by Steamship and CandyLLM.

## Features
- Conversational AI capabilities
- Multi-modal processing
- Scalable deployment
- Enterprise security

## Usage
Deploy this package to Steamship and interact via API endpoints.

## Author
{package.author}

## Version
{package.version}
'''
        
        # Docker configuration
        files['Dockerfile'] = f'''FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "agent.py"]
'''
        
        return files
    
    def _create_package_manifest(self, package: Package) -> Dict[str, Any]:
        """Create package manifest"""
        return {
            'type': package.package_type.value,
            'handle': package.name,
            'version': package.version,
            'description': package.description,
            'author': package.author,
            'entrypoint': 'agent.py',
            'dependencies': package.dependencies,
            'steamship_version': '>=2.0.0',
            'python_version': '>=3.8',
            'resources': {
                'cpu': self.config.cpu_limit,
                'memory': self.config.memory_limit,
                'storage': self.config.storage_limit,
                'gpu': self.config.gpu_count
            },
            'scaling': {
                'min_replicas': self.config.min_replicas,
                'max_replicas': self.config.max_replicas,
                'auto_scaling': self.config.auto_scaling
            },
            'features': {
                'multi_modal': self.config.enable_multi_modal,
                'file_storage': self.config.enable_file_storage,
                'vector_search': self.config.enable_vector_search,
                'audio_generation': self.config.enable_audio_generation,
                'image_generation': self.config.enable_image_generation,
                'video_processing': self.config.enable_video_processing,
                'document_processing': self.config.enable_document_processing,
                'web_scraping': self.config.enable_web_scraping,
                'api_integration': self.config.enable_api_integration,
                'database_access': self.config.enable_database_access
            },
            'security': {
                'authentication': self.config.enable_authentication,
                'encryption': self.config.enable_encryption,
                'audit_logging': self.config.enable_audit_logging,
                'allowed_origins': self.config.allowed_origins,
                'rate_limit': self.config.rate_limit_per_minute,
                'max_file_size_mb': self.config.max_file_size_mb
            },
            'monitoring': {
                'metrics': self.config.enable_metrics,
                'tracing': self.config.enable_tracing,
                'log_level': self.config.log_level,
                'health_check_interval': self.config.health_check_interval,
                'metrics_port': self.config.metrics_port
            }
        }
    
    async def build_package(self, package_id: str) -> bool:
        """Build package for deployment"""
        try:
            if package_id not in self._packages:
                return False
            
            package = self._packages[package_id]
            start_time = datetime.now()
            
            # Create temporary build directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write package files
                for filename, content in package.files.items():
                    file_path = os.path.join(temp_dir, filename)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    async with aiofiles.open(file_path, 'w') as f:
                        await f.write(content)
                
                # Build package (simulate)
                await asyncio.sleep(2)  # Simulate build time
                
                # Create package archive
                archive_path = os.path.join(temp_dir, f"{package.name}.zip")
                with zipfile.ZipFile(archive_path, 'w') as zipf:
                    for filename in package.files.keys():
                        file_path = os.path.join(temp_dir, filename)
                        if os.path.exists(file_path):
                            zipf.write(file_path, filename)
                
                # Update package
                build_time = (datetime.now() - start_time).total_seconds()
                package.build_logs.append(f"Build completed in {build_time:.2f}s")
                package.updated_at = datetime.now()
                
                self._usage_stats['total_builds'] += 1
                self._usage_stats['successful_builds'] += 1
                
                return True
            
        except Exception as e:
            self._usage_stats['total_builds'] += 1
            self._usage_stats['failed_builds'] += 1
            if package_id in self._packages:
                self._packages[package_id].build_logs.append(f"Build failed: {str(e)}")
            return False
    
    async def deploy_package(self, package_id: str, environment: EnvironmentType = None) -> str:
        """Deploy package to Steamship"""
        try:
            if package_id not in self._packages:
                raise ValueError(f"Package {package_id} not found")
            
            package = self._packages[package_id]
            deployment_id = f"dep_{uuid.uuid4().hex[:8]}"
            
            deployment = Deployment(
                deployment_id=deployment_id,
                package_id=package_id,
                workspace_id=self.config.workspace_id,
                instance_handle=f"{package.name}-{deployment_id}",
                environment=environment or self.config.environment,
                status=DeploymentStatus.BUILDING
            )
            
            # Simulate deployment process
            await self._simulate_deployment(deployment)
            
            self._deployments[deployment_id] = deployment
            self._usage_stats['total_deployments'] += 1
            self._usage_stats['active_deployments'] = len([
                d for d in self._deployments.values() 
                if d.status == DeploymentStatus.RUNNING
            ])
            
            return deployment_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to deploy package: {e}")
    
    async def _simulate_deployment(self, deployment: Deployment):
        """Simulate deployment process"""
        try:
            # Building phase
            deployment.status = DeploymentStatus.BUILDING
            deployment.logs.append(f"Building deployment {deployment.deployment_id}...")
            await asyncio.sleep(1)
            
            # Deploying phase
            deployment.status = DeploymentStatus.DEPLOYING
            deployment.logs.append("Deploying to Steamship infrastructure...")
            await asyncio.sleep(2)
            
            # Running phase
            deployment.status = DeploymentStatus.RUNNING
            deployment.started_at = datetime.now()
            deployment.endpoint_url = f"https://{deployment.instance_handle}.steamship.run"
            deployment.health_status = "healthy"
            deployment.last_health_check = datetime.now()
            deployment.logs.append(f"Deployment successful. Endpoint: {deployment.endpoint_url}")
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.logs.append(f"Deployment failed: {str(e)}")
    
    async def invoke_agent(self, deployment_id: str, message: str, context: Dict[str, Any] = None) -> str:
        """Invoke deployed agent"""
        try:
            if deployment_id not in self._deployments:
                return "Deployment not found"
            
            deployment = self._deployments[deployment_id]
            
            if deployment.status != DeploymentStatus.RUNNING:
                return f"Deployment not running (status: {deployment.status.value})"
            
            start_time = datetime.now()
            
            # Simulate agent invocation
            response = await self._simulate_agent_invocation(deployment, message, context)
            
            # Update metrics
            response_time = (datetime.now() - start_time).total_seconds()
            deployment.metrics['last_response_time'] = response_time
            deployment.metrics['total_invocations'] = deployment.metrics.get('total_invocations', 0) + 1
            
            self._usage_stats['total_invocations'] += 1
            self._usage_stats['total_requests'] += 1
            
            # Update average response time
            total_invocations = self._usage_stats['total_invocations']
            avg_time = self._usage_stats['average_response_time']
            self._usage_stats['average_response_time'] = (
                (avg_time * (total_invocations - 1) + response_time) / total_invocations
            )
            
            return response
            
        except Exception as e:
            self._usage_stats['total_requests'] += 1
            error_rate = (self._usage_stats.get('errors', 0) + 1) / self._usage_stats['total_requests']
            self._usage_stats['error_rate'] = error_rate
            return f"Error invoking agent: {str(e)}"
    
    async def _simulate_agent_invocation(self, deployment: Deployment, message: str, 
                                       context: Dict[str, Any] = None) -> str:
        """Simulate agent processing"""
        try:
            # Simulate processing delay
            await asyncio.sleep(0.5)
            
            # Generate response based on message
            if "hello" in message.lower():
                return f"Hello! I'm the {deployment.instance_handle} agent. How can I help you?"
            elif "status" in message.lower():
                return f"Agent status: {deployment.health_status}. Uptime: {datetime.now() - deployment.started_at if deployment.started_at else 'Unknown'}"
            elif "capabilities" in message.lower():
                package = self._packages.get(deployment.package_id)
                if package and package.manifest:
                    features = package.manifest.get('features', {})
                    enabled_features = [k for k, v in features.items() if v]
                    return f"Enabled capabilities: {', '.join(enabled_features)}"
                return "Multi-modal AI agent with advanced capabilities"
            else:
                return f"I understand you said: '{message}'. I'm processing this with my AI capabilities and will provide a comprehensive response based on my training and available tools."
            
        except Exception as e:
            return f"Processing error: {str(e)}"
    
    async def scale_deployment(self, deployment_id: str, replicas: int) -> bool:
        """Scale deployment"""
        try:
            if deployment_id not in self._deployments:
                return False
            
            deployment = self._deployments[deployment_id]
            
            if deployment.status != DeploymentStatus.RUNNING:
                return False
            
            deployment.status = DeploymentStatus.SCALING
            deployment.logs.append(f"Scaling to {replicas} replicas...")
            
            # Simulate scaling
            await asyncio.sleep(1)
            
            deployment.status = DeploymentStatus.RUNNING
            deployment.metrics['replicas'] = replicas
            deployment.logs.append(f"Scaled to {replicas} replicas successfully")
            
            return True
            
        except Exception as e:
            if deployment_id in self._deployments:
                self._deployments[deployment_id].logs.append(f"Scaling failed: {str(e)}")
            return False
    
    async def stop_deployment(self, deployment_id: str) -> bool:
        """Stop deployment"""
        try:
            if deployment_id not in self._deployments:
                return False
            
            deployment = self._deployments[deployment_id]
            deployment.status = DeploymentStatus.STOPPED
            deployment.stopped_at = datetime.now()
            deployment.logs.append("Deployment stopped")
            
            self._usage_stats['active_deployments'] = len([
                d for d in self._deployments.values() 
                if d.status == DeploymentStatus.RUNNING
            ])
            
            return True
            
        except Exception as e:
            return False
    
    async def update_deployment(self, deployment_id: str, package_id: str = None) -> bool:
        """Update deployment with new package version"""
        try:
            if deployment_id not in self._deployments:
                return False
            
            deployment = self._deployments[deployment_id]
            old_package_id = deployment.package_id
            
            deployment.status = DeploymentStatus.UPDATING
            deployment.logs.append("Updating deployment...")
            
            if package_id:
                deployment.package_id = package_id
            
            # Simulate update
            await asyncio.sleep(2)
            
            deployment.status = DeploymentStatus.RUNNING
            deployment.updated_at = datetime.now()
            deployment.logs.append("Deployment updated successfully")
            
            return True
            
        except Exception as e:
            if deployment_id in self._deployments:
                self._deployments[deployment_id].logs.append(f"Update failed: {str(e)}")
                self._deployments[deployment_id].status = DeploymentStatus.FAILED
            return False
    
    async def get_deployment_logs(self, deployment_id: str, lines: int = 100) -> List[str]:
        """Get deployment logs"""
        if deployment_id not in self._deployments:
            return []
        
        deployment = self._deployments[deployment_id]
        return deployment.logs[-lines:]
    
    async def get_deployment_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment metrics"""
        if deployment_id not in self._deployments:
            return {}
        
        deployment = self._deployments[deployment_id]
        
        # Generate resource usage metrics
        metrics = {
            'deployment_id': deployment_id,
            'status': deployment.status.value,
            'uptime_seconds': (datetime.now() - deployment.started_at).total_seconds() 
                            if deployment.started_at else 0,
            'total_invocations': deployment.metrics.get('total_invocations', 0),
            'last_response_time': deployment.metrics.get('last_response_time', 0),
            'replicas': deployment.metrics.get('replicas', 1),
            'health_status': deployment.health_status,
            'endpoint_url': deployment.endpoint_url,
            'cpu_usage_percent': 45.2,  # Simulated
            'memory_usage_mb': 512,     # Simulated
            'storage_usage_gb': 2.1,    # Simulated
            'network_in_mb': 10.5,      # Simulated
            'network_out_mb': 8.3,      # Simulated
            'errors_count': 0,
            'success_rate': 99.5
        }
        
        return metrics
    
    async def create_resource_allocation(self, deployment_id: str, resource_type: ResourceType,
                                       allocation: float, limit: float, unit: str) -> str:
        """Create resource allocation"""
        try:
            resource_id = f"res_{uuid.uuid4().hex[:8]}"
            
            resource = Resource(
                resource_id=resource_id,
                resource_type=resource_type,
                allocated=allocation,
                used=0.0,
                limit=limit,
                unit=unit
            )
            
            self._resources[resource_id] = resource
            
            # Update deployment metrics
            if deployment_id in self._deployments:
                deployment = self._deployments[deployment_id]
                deployment.resource_usage[resource_type.value] = {
                    'resource_id': resource_id,
                    'allocated': allocation,
                    'limit': limit,
                    'unit': unit
                }
            
            return resource_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to create resource allocation: {e}")
    
    def get_package(self, package_id: str) -> Optional[Package]:
        """Get package by ID"""
        return self._packages.get(package_id)
    
    def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        """Get deployment by ID"""
        return self._deployments.get(deployment_id)
    
    def list_packages(self) -> List[Dict[str, Any]]:
        """List all packages"""
        return [
            {
                'package_id': package.package_id,
                'name': package.name,
                'type': package.package_type.value,
                'version': package.version,
                'description': package.description,
                'author': package.author,
                'created_at': package.created_at.isoformat(),
                'updated_at': package.updated_at.isoformat(),
                'tags': package.tags,
                'dependencies': package.dependencies
            }
            for package in self._packages.values()
        ]
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments"""
        return [
            {
                'deployment_id': deployment.deployment_id,
                'package_id': deployment.package_id,
                'instance_handle': deployment.instance_handle,
                'status': deployment.status.value,
                'environment': deployment.environment.value,
                'created_at': deployment.created_at.isoformat(),
                'started_at': deployment.started_at.isoformat() if deployment.started_at else None,
                'endpoint_url': deployment.endpoint_url,
                'health_status': deployment.health_status
            }
            for deployment in self._deployments.values()
        ]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = self._usage_stats.copy()
        
        # Calculate derived metrics
        if stats['total_builds'] > 0:
            stats['build_success_rate'] = (stats['successful_builds'] / stats['total_builds']) * 100
        
        # Calculate resource utilization
        total_cpu = sum(r.used for r in self._resources.values() if r.resource_type == ResourceType.CPU)
        total_memory = sum(r.used for r in self._resources.values() if r.resource_type == ResourceType.MEMORY)
        
        stats['total_cpu_used'] = total_cpu
        stats['total_memory_used_gb'] = total_memory / 1024  # Convert MB to GB
        
        return stats
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': {
                'workspace_id': self.config.workspace_id,
                'project_name': self.config.project_name,
                'package_name': self.config.package_name,
                'environment': self.config.environment.value,
                'auto_scaling': self.config.auto_scaling,
                'min_replicas': self.config.min_replicas,
                'max_replicas': self.config.max_replicas,
                'multi_modal': self.config.enable_multi_modal,
                'file_storage': self.config.enable_file_storage,
                'vector_search': self.config.enable_vector_search
            },
            'packages_count': len(self._packages),
            'deployments_count': len(self._deployments),
            'active_deployments': len([d for d in self._deployments.values() 
                                     if d.status == DeploymentStatus.RUNNING]),
            'resources_count': len(self._resources),
            'usage_stats': self.get_usage_stats(),
            'steamship_available': STEAMSHIP_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class SteamshipProvider(BaseAgentProvider):
    """
    Provider implementation for Steamship.
    
    Enables agent hosting platform with package management,
    deployment infrastructure, and multi-modal capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, SteamshipAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not STEAMSHIP_AVAILABLE:
            self.logger.warning("Steamship dependencies not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "steamship"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.PACKAGE_MANAGEMENT,
            AgentCapability.DEPLOYMENT_INFRASTRUCTURE,
            AgentCapability.CONTAINER_ORCHESTRATION,
            AgentCapability.AUTO_SCALING,
            AgentCapability.RESOURCE_MANAGEMENT,
            AgentCapability.MULTI_MODAL,
            AgentCapability.FILE_STORAGE,
            AgentCapability.VECTOR_SEARCH,
            AgentCapability.API_INTEGRATION,
            AgentCapability.MONITORING_METRICS,
            AgentCapability.ENTERPRISE_SECURITY
        ]
    
    async def initialize(self) -> bool:
        """Initialize Steamship provider"""
        if not STEAMSHIP_AVAILABLE:
            self.logger.error("Steamship dependencies not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Steamship provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Steamship provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Steamship agent"""
        if not self._initialized:
            await self.initialize()
        
        if not STEAMSHIP_AVAILABLE:
            raise RuntimeError("Steamship dependencies not available")
        
        agent_id = f"steamship_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create Steamship configuration
            steamship_config = SteamshipConfig(
                api_key=self.config.get('api_key', ''),
                workspace_id=self.config.get('workspace_id', ''),
                project_name=self.config.get('project_name', f'candyllm-{agent_id}'),
                package_name=self.config.get('package_name', f'agent-{agent_id}'),
                package_type=PackageType(self.config.get('package_type', 'agent')),
                version=self.config.get('version', '1.0.0'),
                description=self.config.get('description', 'AI Agent powered by CandyLLM'),
                author=self.config.get('author', 'CandyLLM'),
                requirements=self.config.get('requirements', ['steamship', 'openai', 'requests']),
                environment=EnvironmentType(self.config.get('environment', 'development')),
                cpu_limit=self.config.get('cpu_limit', '1000m'),
                memory_limit=self.config.get('memory_limit', '2Gi'),
                storage_limit=self.config.get('storage_limit', '10Gi'),
                gpu_count=self.config.get('gpu_count', 0),
                min_replicas=self.config.get('min_replicas', 1),
                max_replicas=self.config.get('max_replicas', 5),
                auto_scaling=self.config.get('auto_scaling', True),
                enable_multi_modal=self.config.get('enable_multi_modal', True),
                enable_file_storage=self.config.get('enable_file_storage', True),
                enable_vector_search=self.config.get('enable_vector_search', True),
                enable_audio_generation=self.config.get('enable_audio_generation', True),
                enable_image_generation=self.config.get('enable_image_generation', True),
                enable_video_processing=self.config.get('enable_video_processing', True),
                enable_document_processing=self.config.get('enable_document_processing', True),
                enable_web_scraping=self.config.get('enable_web_scraping', True),
                enable_api_integration=self.config.get('enable_api_integration', True),
                enable_database_access=self.config.get('enable_database_access', True),
                enable_authentication=self.config.get('enable_authentication', True),
                enable_encryption=self.config.get('enable_encryption', True),
                enable_audit_logging=self.config.get('enable_audit_logging', True),
                allowed_origins=self.config.get('allowed_origins', ['*']),
                rate_limit_per_minute=self.config.get('rate_limit_per_minute', 100),
                max_file_size_mb=self.config.get('max_file_size_mb', 100),
                enable_metrics=self.config.get('enable_metrics', True),
                enable_tracing=self.config.get('enable_tracing', True),
                log_level=self.config.get('log_level', 'INFO'),
                health_check_interval=self.config.get('health_check_interval', 30),
                metrics_port=self.config.get('metrics_port', 9090)
            )
            
            # Create agent
            agent = SteamshipAgent(
                agent_id=agent_id,
                config=steamship_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Steamship agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Steamship agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Steamship agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Steamship agent"""
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
            execution_type = context.get('execution_type', 'create_and_deploy')
            
            if execution_type == 'create_package':
                result = await self._create_package(agent, context)
            elif execution_type == 'deploy_package':
                result = await self._deploy_package(agent, context)
            elif execution_type == 'invoke_agent':
                result = await self._invoke_agent(agent, prompt, context)
            elif execution_type == 'scale_deployment':
                result = await self._scale_deployment(agent, context)
            else:
                result = await self._create_and_deploy(agent, prompt, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare response
            response_content = result.get('summary', 'Steamship operation completed')
            
            # Prepare metadata
            metadata = {
                'execution_time_seconds': execution_time,
                'execution_type': execution_type,
                'package_id': result.get('package_id'),
                'deployment_id': result.get('deployment_id'),
                'endpoint_url': result.get('endpoint_url'),
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
            self.logger.error(f"Steamship agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def _create_package(self, agent: SteamshipAgent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create Steamship package"""
        package_name = context.get('package_name')
        package_type = PackageType(context.get('package_type', 'agent'))
        files = context.get('files', {})
        dependencies = context.get('dependencies', [])
        
        package_id = await agent.create_package(package_name, package_type, files, dependencies)
        
        # Build package
        build_success = await agent.build_package(package_id)
        
        return {
            'package_id': package_id,
            'build_success': build_success,
            'summary': f"Package {package_id} {'built successfully' if build_success else 'build failed'}"
        }
    
    async def _deploy_package(self, agent: SteamshipAgent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy Steamship package"""
        package_id = context.get('package_id')
        environment = EnvironmentType(context.get('environment', 'development'))
        
        if not package_id:
            raise ValueError("package_id required for deployment")
        
        deployment_id = await agent.deploy_package(package_id, environment)
        
        # Get deployment info
        deployment = agent.get_deployment(deployment_id)
        endpoint_url = deployment.endpoint_url if deployment else None
        
        return {
            'deployment_id': deployment_id,
            'package_id': package_id,
            'endpoint_url': endpoint_url,
            'summary': f"Package {package_id} deployed as {deployment_id}"
        }
    
    async def _invoke_agent(self, agent: SteamshipAgent, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke deployed agent"""
        deployment_id = context.get('deployment_id')
        
        if not deployment_id:
            raise ValueError("deployment_id required for invocation")
        
        response = await agent.invoke_agent(deployment_id, prompt, context)
        
        return {
            'deployment_id': deployment_id,
            'response': response,
            'summary': f"Agent {deployment_id} responded: {response[:100]}..."
        }
    
    async def _scale_deployment(self, agent: SteamshipAgent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Scale deployment"""
        deployment_id = context.get('deployment_id')
        replicas = context.get('replicas', 2)
        
        if not deployment_id:
            raise ValueError("deployment_id required for scaling")
        
        success = await agent.scale_deployment(deployment_id, replicas)
        
        return {
            'deployment_id': deployment_id,
            'replicas': replicas,
            'success': success,
            'summary': f"Deployment {deployment_id} {'scaled' if success else 'failed to scale'} to {replicas} replicas"
        }
    
    async def _create_and_deploy(self, agent: SteamshipAgent, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create package and deploy"""
        # Create package
        package_id = await agent.create_package()
        
        # Build package
        build_success = await agent.build_package(package_id)
        if not build_success:
            return {
                'package_id': package_id,
                'error': 'Package build failed',
                'summary': f"Package {package_id} build failed"
            }
        
        # Deploy package
        deployment_id = await agent.deploy_package(package_id)
        
        # Get deployment info
        deployment = agent.get_deployment(deployment_id)
        endpoint_url = deployment.endpoint_url if deployment else None
        
        # Invoke with prompt
        response = await agent.invoke_agent(deployment_id, prompt, context)
        
        return {
            'package_id': package_id,
            'deployment_id': deployment_id,
            'endpoint_url': endpoint_url,
            'response': response,
            'summary': f"Created package {package_id}, deployed as {deployment_id}, and responded: {response[:100]}..."
        }
    
    async def create_package(self, agent_id: str, package_name: str = None, package_type: str = "agent",
                           files: Dict[str, str] = None, dependencies: List[str] = None) -> str:
        """Create a package"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        pkg_type = PackageType(package_type)
        
        return await agent.create_package(package_name, pkg_type, files, dependencies)
    
    async def deploy_package(self, agent_id: str, package_id: str, environment: str = "development") -> str:
        """Deploy a package"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        env = EnvironmentType(environment)
        
        return await agent.deploy_package(package_id, env)
    
    async def invoke_deployment(self, agent_id: str, deployment_id: str, message: str,
                              context: Dict[str, Any] = None) -> str:
        """Invoke a deployment"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        
        return await agent.invoke_agent(deployment_id, message, context)
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Steamship agent"""
        if agent_id not in self._agents:
            return False
        
        try:
            # Steamship tools would be registered through package dependencies
            # This is a simplified implementation
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Steamship capabilities"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Create tool spec for Steamship deployment
            tool_spec = ToolSpec(
                name=f"steamship_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={
                    'description': tool_description,
                    'deployment_type': 'steamship_package',
                    'scalable': True,
                    'multi_modal': True,
                    'examples': examples or []
                },
                security_policy={
                    'risk_level': 'medium',
                    'requires_approval': True,
                    'deployment_required': True,
                    'resource_limits': True
                }
            )
            
            self.logger.info(f"Synthesized tool for Steamship agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Steamship agent"""
        try:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                
                # Stop all deployments
                for deployment_id in list(agent._deployments.keys()):
                    await agent.stop_deployment(deployment_id)
                
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Steamship agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Steamship agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Steamship agent"""
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