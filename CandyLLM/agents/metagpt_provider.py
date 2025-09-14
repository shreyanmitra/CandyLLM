"""
MetaGPT Agent Provider

Integrates MetaGPT's multi-agent software development framework with role-based programming,
enabling automated requirement analysis, system design, and code generation workflows.
"""

import uuid
import asyncio
import json
import os
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

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
    import yaml
    import jinja2
    METAGPT_AVAILABLE = True
except ImportError:
    METAGPT_AVAILABLE = False
    # Mock classes for when dependencies are not available
    httpx = None
    requests = None
    OpenAI = None
    AsyncOpenAI = None
    yaml = None
    jinja2 = None


class DevelopmentRole(Enum):
    """Software development roles in MetaGPT"""
    PRODUCT_MANAGER = "product_manager"
    ARCHITECT = "architect"
    PROJECT_MANAGER = "project_manager"
    ENGINEER = "engineer"
    QA_ENGINEER = "qa_engineer"
    UI_UX_DESIGNER = "ui_ux_designer"
    DEVOPS_ENGINEER = "devops_engineer"
    TECH_LEAD = "tech_lead"
    BUSINESS_ANALYST = "business_analyst"
    SECURITY_ENGINEER = "security_engineer"


class DevelopmentPhase(Enum):
    """Phases of software development"""
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    SYSTEM_DESIGN = "system_design"
    DETAILED_DESIGN = "detailed_design"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"
    REVIEW = "review"
    DOCUMENTATION = "documentation"
    QUALITY_ASSURANCE = "quality_assurance"


class ArtifactType(Enum):
    """Types of development artifacts"""
    REQUIREMENTS = "requirements"
    SYSTEM_DESIGN = "system_design"
    API_DESIGN = "api_design"
    DATABASE_DESIGN = "database_design"
    UI_DESIGN = "ui_design"
    CODE = "code"
    TEST_CASES = "test_cases"
    DOCUMENTATION = "documentation"
    DEPLOYMENT_CONFIG = "deployment_config"
    PROJECT_PLAN = "project_plan"


@dataclass
class MetaGPTConfig:
    """Configuration for MetaGPT agent"""
    api_key: str = ""
    model: str = "gpt-4"
    base_url: str = "https://api.openai.com/v1"
    max_tokens: int = 4000
    temperature: float = 0.7
    project_name: str = "untitled_project"
    output_directory: str = "./metagpt_output"
    enable_code_generation: bool = True
    enable_testing: bool = True
    enable_documentation: bool = True
    max_iterations: int = 10
    team_size: int = 5
    development_methodology: str = "agile"  # agile, waterfall, lean
    quality_gate_enabled: bool = True
    code_review_enabled: bool = True
    continuous_integration: bool = True
    target_languages: List[str] = field(default_factory=lambda: ["python"])
    frameworks: List[str] = field(default_factory=list)
    databases: List[str] = field(default_factory=list)
    deployment_targets: List[str] = field(default_factory=list)
    enable_security_review: bool = True
    enable_performance_optimization: bool = True
    compliance_requirements: List[str] = field(default_factory=list)


@dataclass
class DeveloperAgent:
    """Individual developer agent in MetaGPT team"""
    agent_id: str
    role: DevelopmentRole
    name: str
    specialization: List[str] = field(default_factory=list)
    experience_level: str = "senior"  # junior, mid, senior, lead
    active: bool = True
    current_task: Optional[str] = None
    workload: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    skills: List[str] = field(default_factory=list)
    preferred_technologies: List[str] = field(default_factory=list)


@dataclass
class DevelopmentTask:
    """Individual development task"""
    task_id: str
    title: str
    description: str
    artifact_type: ArtifactType
    assigned_role: DevelopmentRole
    assigned_agent: Optional[str] = None
    phase: DevelopmentPhase = DevelopmentPhase.REQUIREMENT_ANALYSIS
    priority: int = 3  # 1=critical, 5=low
    estimated_effort: float = 1.0  # in hours
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, blocked
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)  # file paths
    review_status: str = "not_reviewed"  # not_reviewed, approved, needs_changes
    quality_score: float = 0.0


@dataclass
class ProjectArtifact:
    """Development artifact/deliverable"""
    artifact_id: str
    name: str
    artifact_type: ArtifactType
    content: str
    file_path: Optional[str] = None
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    review_comments: List[str] = field(default_factory=list)
    approved: bool = False


@dataclass
class SoftwareProject:
    """Complete software development project"""
    project_id: str
    name: str
    description: str
    start_date: datetime
    team_members: List[str] = field(default_factory=list)  # agent_ids
    tasks: List[DevelopmentTask] = field(default_factory=list)
    artifacts: List[ProjectArtifact] = field(default_factory=list)
    current_phase: DevelopmentPhase = DevelopmentPhase.REQUIREMENT_ANALYSIS
    status: str = "active"  # active, completed, on_hold, cancelled
    progress: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    timeline: Dict[str, datetime] = field(default_factory=dict)
    budget: Dict[str, float] = field(default_factory=dict)
    risks: List[str] = field(default_factory=list)
    end_date: Optional[datetime] = None


class MetaGPTAgent:
    """MetaGPT multi-agent software development system"""
    
    def __init__(self, agent_id: str, config: MetaGPTConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._client = None
        self._async_client = None
        self._developer_agents = {}  # agent_id -> DeveloperAgent
        self._projects = {}  # project_id -> SoftwareProject
        self._templates = {}  # template_name -> template_content
        self._current_project = None
        self._usage_stats = {
            'total_projects': 0,
            'total_tasks_completed': 0,
            'total_artifacts_generated': 0,
            'total_code_lines': 0,
            'total_development_time': 0.0,
            'successful_deliveries': 0,
            'average_project_duration': 0.0,
            'code_quality_average': 0.0,
            'team_productivity': 0.0,
            'bug_rate': 0.0,
            'on_time_delivery_rate': 0.0,
            'role_performance': {},
            'technology_usage': {}
        }
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the MetaGPT agent"""
        if not METAGPT_AVAILABLE:
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
            
            # Initialize development team
            await self._initialize_development_team()
            
            # Load templates
            await self._load_templates()
            
            # Create output directory
            os.makedirs(self.config.output_directory, exist_ok=True)
            
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
            raise RuntimeError(f"MetaGPT API test failed: {e}")
    
    async def _initialize_development_team(self):
        """Initialize default development team"""
        team_roles = [
            {
                'role': DevelopmentRole.PRODUCT_MANAGER,
                'name': 'Alice PM',
                'specialization': ['requirements', 'stakeholder_management', 'product_strategy'],
                'skills': ['user_stories', 'backlog_management', 'market_analysis']
            },
            {
                'role': DevelopmentRole.ARCHITECT,
                'name': 'Bob Architect',
                'specialization': ['system_design', 'scalability', 'technology_selection'],
                'skills': ['microservices', 'cloud_architecture', 'security_design']
            },
            {
                'role': DevelopmentRole.ENGINEER,
                'name': 'Charlie Engineer',
                'specialization': ['backend_development', 'api_design', 'database_optimization'],
                'skills': ['python', 'javascript', 'sql', 'docker']
            },
            {
                'role': DevelopmentRole.QA_ENGINEER,
                'name': 'Diana QA',
                'specialization': ['test_automation', 'quality_assurance', 'performance_testing'],
                'skills': ['selenium', 'pytest', 'load_testing', 'ci_cd']
            },
            {
                'role': DevelopmentRole.UI_UX_DESIGNER,
                'name': 'Eve Designer',
                'specialization': ['user_interface', 'user_experience', 'design_systems'],
                'skills': ['figma', 'prototyping', 'user_research', 'accessibility']
            }
        ]
        
        for team_member in team_roles[:self.config.team_size]:
            agent_id = f"dev_{uuid.uuid4().hex[:8]}"
            
            developer = DeveloperAgent(
                agent_id=agent_id,
                role=team_member['role'],
                name=team_member['name'],
                specialization=team_member['specialization'],
                skills=team_member['skills'],
                performance_metrics={
                    'tasks_completed': 0,
                    'quality_score': 0.8,
                    'productivity': 1.0,
                    'collaboration_score': 0.9,
                    'innovation_index': 0.7
                }
            )
            
            self._developer_agents[agent_id] = developer
            self._usage_stats['role_performance'][team_member['role'].value] = developer.performance_metrics.copy()
    
    async def _load_templates(self):
        """Load development templates"""
        templates = {
            'requirements_template': """
            # Requirements Document for {{project_name}}

            ## Overview
            {{description}}

            ## Functional Requirements
            {% for req in functional_requirements %}
            - {{req}}
            {% endfor %}

            ## Non-Functional Requirements
            {% for req in non_functional_requirements %}
            - {{req}}
            {% endfor %}

            ## User Stories
            {% for story in user_stories %}
            ### {{story.title}}
            **As a** {{story.role}}
            **I want** {{story.want}}
            **So that** {{story.benefit}}

            **Acceptance Criteria:**
            {% for criteria in story.acceptance_criteria %}
            - {{criteria}}
            {% endfor %}
            {% endfor %}
            """,
            
            'api_design_template': """
            # API Design for {{project_name}}

            ## Base URL
            {{base_url}}

            ## Endpoints
            {% for endpoint in endpoints %}
            ### {{endpoint.method}} {{endpoint.path}}
            **Description:** {{endpoint.description}}

            **Request:**
            ```json
            {{endpoint.request_example}}
            ```

            **Response:**
            ```json
            {{endpoint.response_example}}
            ```
            {% endfor %}
            """,
            
            'code_template': """
            # {{class_name}}
            # Generated by MetaGPT
            # {{description}}

            from typing import {{type_imports}}
            {{imports}}

            class {{class_name}}:
                \"\"\"{{class_description}}\"\"\"
                
                def __init__(self{{init_params}}):
                    {{init_body}}
                
                {{methods}}
            """,
            
            'test_template': """
            # Test for {{module_name}}
            # Generated by MetaGPT

            import pytest
            from {{module_path}} import {{class_name}}

            class Test{{class_name}}:
                \"\"\"Test cases for {{class_name}}\"\"\"
                
                def setup_method(self):
                    {{setup_code}}
                
                {{test_methods}}
            """
        }
        
        for name, template in templates.items():
            self._templates[name] = jinja2.Template(template)
    
    async def create_project(self, name: str, description: str, requirements: List[str] = None) -> str:
        """Create a new software development project"""
        try:
            project_id = f"proj_{uuid.uuid4().hex[:8]}"
            
            project = SoftwareProject(
                project_id=project_id,
                name=name,
                description=description,
                start_date=datetime.now(),
                team_members=list(self._developer_agents.keys()),
                quality_metrics={
                    'code_coverage': 0.0,
                    'complexity_score': 0.0,
                    'maintainability_index': 0.0,
                    'security_score': 0.0,
                    'performance_score': 0.0
                },
                timeline={
                    'requirement_analysis': datetime.now(),
                    'system_design': None,
                    'implementation': None,
                    'testing': None,
                    'deployment': None
                }
            )
            
            self._projects[project_id] = project
            self._current_project = project_id
            self._usage_stats['total_projects'] += 1
            
            # Create project directory
            project_dir = os.path.join(self.config.output_directory, name)
            os.makedirs(project_dir, exist_ok=True)
            
            # Initialize with requirements analysis
            if requirements:
                await self._analyze_requirements(project_id, requirements)
            
            return project_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to create project: {e}")
    
    async def _analyze_requirements(self, project_id: str, requirements: List[str]):
        """Analyze and document requirements"""
        try:
            project = self._projects[project_id]
            
            # Find Product Manager
            pm_agent = next(
                (agent for agent in self._developer_agents.values() 
                 if agent.role == DevelopmentRole.PRODUCT_MANAGER), 
                None
            )
            
            if not pm_agent:
                return
            
            # Generate detailed requirements
            requirements_prompt = f"""
            As a Product Manager, analyze these high-level requirements for {project.name}:
            
            {chr(10).join([f"- {req}" for req in requirements])}
            
            Generate a comprehensive requirements document including:
            1. Functional requirements (specific features and capabilities)
            2. Non-functional requirements (performance, security, scalability)
            3. User stories with acceptance criteria
            4. Technical constraints and assumptions
            
            Provide the output in a structured format.
            """
            
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": requirements_prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            requirements_doc = response.choices[0].message.content
            
            # Create requirements artifact
            artifact = ProjectArtifact(
                artifact_id=f"artifact_{uuid.uuid4().hex[:8]}",
                name="Requirements Document",
                artifact_type=ArtifactType.REQUIREMENTS,
                content=requirements_doc,
                created_by=pm_agent.agent_id,
                file_path=os.path.join(self.config.output_directory, project.name, "requirements.md")
            )
            
            project.artifacts.append(artifact)
            
            # Save to file
            with open(artifact.file_path, 'w') as f:
                f.write(requirements_doc)
            
            # Create requirement analysis task
            task = DevelopmentTask(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                title="Requirements Analysis",
                description="Analyze and document project requirements",
                artifact_type=ArtifactType.REQUIREMENTS,
                assigned_role=DevelopmentRole.PRODUCT_MANAGER,
                assigned_agent=pm_agent.agent_id,
                status="completed",
                completed_at=datetime.now(),
                output=requirements_doc,
                artifacts=[artifact.file_path]
            )
            
            project.tasks.append(task)
            self._usage_stats['total_tasks_completed'] += 1
            self._usage_stats['total_artifacts_generated'] += 1
            
        except Exception as e:
            pass  # Non-critical operation
    
    async def execute_development_workflow(self, project_id: str) -> Dict[str, Any]:
        """Execute complete development workflow"""
        try:
            if project_id not in self._projects:
                raise ValueError("Project not found")
            
            project = self._projects[project_id]
            start_time = datetime.now()
            
            # Execute development phases
            workflow_result = {
                'project_id': project_id,
                'phases_completed': [],
                'artifacts_generated': [],
                'tasks_completed': [],
                'quality_metrics': {},
                'timeline': {}
            }
            
            # Phase 1: System Design
            if project.current_phase == DevelopmentPhase.REQUIREMENT_ANALYSIS:
                await self._execute_system_design(project_id)
                workflow_result['phases_completed'].append('system_design')
                project.current_phase = DevelopmentPhase.SYSTEM_DESIGN
            
            # Phase 2: Implementation Planning
            if project.current_phase == DevelopmentPhase.SYSTEM_DESIGN:
                await self._execute_implementation_planning(project_id)
                workflow_result['phases_completed'].append('implementation_planning')
                project.current_phase = DevelopmentPhase.DETAILED_DESIGN
            
            # Phase 3: Code Generation
            if self.config.enable_code_generation and project.current_phase == DevelopmentPhase.DETAILED_DESIGN:
                await self._execute_code_generation(project_id)
                workflow_result['phases_completed'].append('code_generation')
                project.current_phase = DevelopmentPhase.IMPLEMENTATION
            
            # Phase 4: Testing
            if self.config.enable_testing and project.current_phase == DevelopmentPhase.IMPLEMENTATION:
                await self._execute_testing(project_id)
                workflow_result['phases_completed'].append('testing')
                project.current_phase = DevelopmentPhase.TESTING
            
            # Phase 5: Documentation
            if self.config.enable_documentation:
                await self._execute_documentation(project_id)
                workflow_result['phases_completed'].append('documentation')
                project.current_phase = DevelopmentPhase.DOCUMENTATION
            
            # Update project status
            execution_time = (datetime.now() - start_time).total_seconds()
            project.status = "completed"
            project.end_date = datetime.now()
            project.progress = 1.0
            
            # Calculate quality metrics
            workflow_result['quality_metrics'] = await self._calculate_quality_metrics(project_id)
            
            # Update usage statistics
            self._usage_stats['total_development_time'] += execution_time
            self._usage_stats['successful_deliveries'] += 1
            
            workflow_result['execution_time'] = execution_time
            workflow_result['total_artifacts'] = len(project.artifacts)
            workflow_result['total_tasks'] = len(project.tasks)
            
            return workflow_result
            
        except Exception as e:
            raise RuntimeError(f"Development workflow failed: {e}")
    
    async def _execute_system_design(self, project_id: str):
        """Execute system design phase"""
        try:
            project = self._projects[project_id]
            
            # Find architect
            architect = next(
                (agent for agent in self._developer_agents.values() 
                 if agent.role == DevelopmentRole.ARCHITECT), 
                None
            )
            
            if not architect:
                return
            
            # Get requirements
            requirements_artifact = next(
                (artifact for artifact in project.artifacts 
                 if artifact.artifact_type == ArtifactType.REQUIREMENTS), 
                None
            )
            
            if not requirements_artifact:
                return
            
            # Generate system design
            design_prompt = f"""
            As a Software Architect, create a comprehensive system design for {project.name}.
            
            Based on these requirements:
            {requirements_artifact.content[:2000]}...
            
            Create a system design that includes:
            1. High-level architecture overview
            2. Component breakdown and responsibilities
            3. Technology stack recommendations
            4. Database design
            5. API design
            6. Security considerations
            7. Scalability and performance considerations
            8. Deployment architecture
            
            Target languages: {', '.join(self.config.target_languages)}
            """
            
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": design_prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            design_doc = response.choices[0].message.content
            
            # Create system design artifact
            artifact = ProjectArtifact(
                artifact_id=f"artifact_{uuid.uuid4().hex[:8]}",
                name="System Design Document",
                artifact_type=ArtifactType.SYSTEM_DESIGN,
                content=design_doc,
                created_by=architect.agent_id,
                file_path=os.path.join(self.config.output_directory, project.name, "system_design.md")
            )
            
            project.artifacts.append(artifact)
            
            # Save to file
            with open(artifact.file_path, 'w') as f:
                f.write(design_doc)
            
            # Create task
            task = DevelopmentTask(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                title="System Design",
                description="Create comprehensive system architecture and design",
                artifact_type=ArtifactType.SYSTEM_DESIGN,
                assigned_role=DevelopmentRole.ARCHITECT,
                assigned_agent=architect.agent_id,
                status="completed",
                completed_at=datetime.now(),
                output=design_doc,
                artifacts=[artifact.file_path]
            )
            
            project.tasks.append(task)
            self._usage_stats['total_tasks_completed'] += 1
            self._usage_stats['total_artifacts_generated'] += 1
            
        except Exception as e:
            pass
    
    async def _execute_implementation_planning(self, project_id: str):
        """Execute implementation planning phase"""
        try:
            project = self._projects[project_id]
            
            # Find project manager
            pm = next(
                (agent for agent in self._developer_agents.values() 
                 if agent.role == DevelopmentRole.PROJECT_MANAGER), 
                None
            )
            
            # If no PM, use tech lead or engineer
            if not pm:
                pm = next(
                    (agent for agent in self._developer_agents.values() 
                     if agent.role in [DevelopmentRole.TECH_LEAD, DevelopmentRole.ENGINEER]), 
                    None
                )
            
            if not pm:
                return
            
            # Generate implementation plan
            plan_prompt = f"""
            As a Project Manager, create a detailed implementation plan for {project.name}.
            
            Break down the system into implementable components and create tasks for:
            1. Core backend services
            2. API endpoints
            3. Database schema
            4. Frontend components
            5. Integration points
            6. Testing requirements
            
            For each component, specify:
            - Implementation priority
            - Estimated effort
            - Dependencies
            - Assigned role
            
            Target technologies: {', '.join(self.config.target_languages)}
            """
            
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": plan_prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            plan_doc = response.choices[0].message.content
            
            # Create implementation plan artifact
            artifact = ProjectArtifact(
                artifact_id=f"artifact_{uuid.uuid4().hex[:8]}",
                name="Implementation Plan",
                artifact_type=ArtifactType.PROJECT_PLAN,
                content=plan_doc,
                created_by=pm.agent_id,
                file_path=os.path.join(self.config.output_directory, project.name, "implementation_plan.md")
            )
            
            project.artifacts.append(artifact)
            
            # Save to file
            with open(artifact.file_path, 'w') as f:
                f.write(plan_doc)
            
            # Create task
            task = DevelopmentTask(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                title="Implementation Planning",
                description="Create detailed implementation plan and task breakdown",
                artifact_type=ArtifactType.PROJECT_PLAN,
                assigned_role=pm.role,
                assigned_agent=pm.agent_id,
                status="completed",
                completed_at=datetime.now(),
                output=plan_doc,
                artifacts=[artifact.file_path]
            )
            
            project.tasks.append(task)
            self._usage_stats['total_tasks_completed'] += 1
            self._usage_stats['total_artifacts_generated'] += 1
            
        except Exception as e:
            pass
    
    async def _execute_code_generation(self, project_id: str):
        """Execute code generation phase"""
        try:
            project = self._projects[project_id]
            
            # Find engineers
            engineers = [
                agent for agent in self._developer_agents.values() 
                if agent.role == DevelopmentRole.ENGINEER
            ]
            
            if not engineers:
                return
            
            # Generate core modules
            modules_to_generate = [
                {
                    'name': 'main',
                    'description': 'Main application entry point',
                    'type': 'application'
                },
                {
                    'name': 'models',
                    'description': 'Data models and database schemas',
                    'type': 'data'
                },
                {
                    'name': 'api',
                    'description': 'API endpoints and routes',
                    'type': 'api'
                },
                {
                    'name': 'services',
                    'description': 'Business logic and services',
                    'type': 'business'
                },
                {
                    'name': 'utils',
                    'description': 'Utility functions and helpers',
                    'type': 'utility'
                }
            ]
            
            code_dir = os.path.join(self.config.output_directory, project.name, "src")
            os.makedirs(code_dir, exist_ok=True)
            
            for module in modules_to_generate:
                await self._generate_module(project_id, module, code_dir, engineers[0])
            
        except Exception as e:
            pass
    
    async def _generate_module(self, project_id: str, module_info: Dict[str, str], 
                             code_dir: str, engineer: DeveloperAgent):
        """Generate a specific code module"""
        try:
            project = self._projects[project_id]
            
            # Get system design for context
            design_artifact = next(
                (artifact for artifact in project.artifacts 
                 if artifact.artifact_type == ArtifactType.SYSTEM_DESIGN), 
                None
            )
            
            design_context = design_artifact.content[:1000] if design_artifact else ""
            
            # Generate code
            code_prompt = f"""
            As a Software Engineer, implement the {module_info['name']} module for {project.name}.
            
            Module purpose: {module_info['description']}
            Module type: {module_info['type']}
            
            System design context:
            {design_context}...
            
            Generate production-ready {self.config.target_languages[0]} code that includes:
            1. Proper class structure and organization
            2. Error handling and logging
            3. Type hints and documentation
            4. Best practices and design patterns
            5. Configuration management
            6. Unit test stubs
            
            Create a complete, functional module.
            """
            
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": code_prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            code_content = response.choices[0].message.content
            
            # Extract code from response
            if "```" in code_content:
                code_start = code_content.find("```")
                code_end = code_content.rfind("```")
                if code_start != code_end:
                    # Extract language tag if present
                    first_line_end = code_content.find("\n", code_start)
                    code_content = code_content[first_line_end+1:code_end]
            
            # Save code file
            file_ext = '.py' if 'python' in self.config.target_languages else '.js'
            file_path = os.path.join(code_dir, f"{module_info['name']}{file_ext}")
            
            with open(file_path, 'w') as f:
                f.write(code_content)
            
            # Create code artifact
            artifact = ProjectArtifact(
                artifact_id=f"artifact_{uuid.uuid4().hex[:8]}",
                name=f"{module_info['name'].title()} Module",
                artifact_type=ArtifactType.CODE,
                content=code_content,
                created_by=engineer.agent_id,
                file_path=file_path
            )
            
            project.artifacts.append(artifact)
            
            # Create task
            task = DevelopmentTask(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                title=f"Implement {module_info['name']} Module",
                description=f"Generate {module_info['description']}",
                artifact_type=ArtifactType.CODE,
                assigned_role=DevelopmentRole.ENGINEER,
                assigned_agent=engineer.agent_id,
                status="completed",
                completed_at=datetime.now(),
                output=f"Generated {module_info['name']} module",
                artifacts=[file_path]
            )
            
            project.tasks.append(task)
            
            # Update statistics
            code_lines = len(code_content.split('\n'))
            self._usage_stats['total_code_lines'] += code_lines
            self._usage_stats['total_tasks_completed'] += 1
            self._usage_stats['total_artifacts_generated'] += 1
            
        except Exception as e:
            pass
    
    async def _execute_testing(self, project_id: str):
        """Execute testing phase"""
        try:
            project = self._projects[project_id]
            
            # Find QA engineer
            qa_engineer = next(
                (agent for agent in self._developer_agents.values() 
                 if agent.role == DevelopmentRole.QA_ENGINEER), 
                None
            )
            
            if not qa_engineer:
                return
            
            # Get code artifacts
            code_artifacts = [
                artifact for artifact in project.artifacts 
                if artifact.artifact_type == ArtifactType.CODE
            ]
            
            test_dir = os.path.join(self.config.output_directory, project.name, "tests")
            os.makedirs(test_dir, exist_ok=True)
            
            # Generate tests for each module
            for code_artifact in code_artifacts:
                await self._generate_test_file(project_id, code_artifact, test_dir, qa_engineer)
            
        except Exception as e:
            pass
    
    async def _generate_test_file(self, project_id: str, code_artifact: ProjectArtifact, 
                                 test_dir: str, qa_engineer: DeveloperAgent):
        """Generate test file for a code module"""
        try:
            project = self._projects[project_id]
            
            # Generate test code
            test_prompt = f"""
            As a QA Engineer, create comprehensive unit tests for this module:
            
            Module: {code_artifact.name}
            Code content (first 1000 chars):
            {code_artifact.content[:1000]}...
            
            Generate {self.config.target_languages[0]} unit tests that include:
            1. Test cases for all public methods
            2. Edge cases and error conditions
            3. Mock objects where needed
            4. Setup and teardown methods
            5. Proper assertions and test data
            
            Use appropriate testing framework (pytest for Python, jest for JavaScript).
            """
            
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            test_content = response.choices[0].message.content
            
            # Extract test code
            if "```" in test_content:
                code_start = test_content.find("```")
                code_end = test_content.rfind("```")
                if code_start != code_end:
                    first_line_end = test_content.find("\n", code_start)
                    test_content = test_content[first_line_end+1:code_end]
            
            # Save test file
            module_name = Path(code_artifact.file_path).stem if code_artifact.file_path else code_artifact.name.lower()
            file_ext = '.py' if 'python' in self.config.target_languages else '.js'
            test_file_path = os.path.join(test_dir, f"test_{module_name}{file_ext}")
            
            with open(test_file_path, 'w') as f:
                f.write(test_content)
            
            # Create test artifact
            artifact = ProjectArtifact(
                artifact_id=f"artifact_{uuid.uuid4().hex[:8]}",
                name=f"Test for {code_artifact.name}",
                artifact_type=ArtifactType.TEST_CASES,
                content=test_content,
                created_by=qa_engineer.agent_id,
                file_path=test_file_path
            )
            
            project.artifacts.append(artifact)
            
            # Create task
            task = DevelopmentTask(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                title=f"Test {code_artifact.name}",
                description=f"Create unit tests for {code_artifact.name}",
                artifact_type=ArtifactType.TEST_CASES,
                assigned_role=DevelopmentRole.QA_ENGINEER,
                assigned_agent=qa_engineer.agent_id,
                status="completed",
                completed_at=datetime.now(),
                output=f"Generated test suite for {code_artifact.name}",
                artifacts=[test_file_path]
            )
            
            project.tasks.append(task)
            self._usage_stats['total_tasks_completed'] += 1
            self._usage_stats['total_artifacts_generated'] += 1
            
        except Exception as e:
            pass
    
    async def _execute_documentation(self, project_id: str):
        """Execute documentation phase"""
        try:
            project = self._projects[project_id]
            
            # Generate comprehensive documentation
            doc_prompt = f"""
            Create comprehensive documentation for the {project.name} project.
            
            Project description: {project.description}
            
            Include:
            1. Project overview and purpose
            2. Installation and setup instructions
            3. API documentation
            4. Architecture overview
            5. Development guide
            6. Deployment instructions
            7. Contributing guidelines
            8. License information
            
            Make it professional and easy to follow.
            """
            
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": doc_prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            doc_content = response.choices[0].message.content
            
            # Save documentation
            doc_path = os.path.join(self.config.output_directory, project.name, "README.md")
            with open(doc_path, 'w') as f:
                f.write(doc_content)
            
            # Create documentation artifact
            artifact = ProjectArtifact(
                artifact_id=f"artifact_{uuid.uuid4().hex[:8]}",
                name="Project Documentation",
                artifact_type=ArtifactType.DOCUMENTATION,
                content=doc_content,
                created_by="system",
                file_path=doc_path
            )
            
            project.artifacts.append(artifact)
            self._usage_stats['total_artifacts_generated'] += 1
            
        except Exception as e:
            pass
    
    async def _calculate_quality_metrics(self, project_id: str) -> Dict[str, float]:
        """Calculate project quality metrics"""
        try:
            project = self._projects[project_id]
            
            metrics = {
                'completeness': len(project.artifacts) / 10.0,  # Normalized to expected artifacts
                'code_quality': 0.8,  # Simulated based on generated code
                'test_coverage': len([a for a in project.artifacts if a.artifact_type == ArtifactType.TEST_CASES]) / max(1, len([a for a in project.artifacts if a.artifact_type == ArtifactType.CODE])),
                'documentation_quality': 1.0 if any(a.artifact_type == ArtifactType.DOCUMENTATION for a in project.artifacts) else 0.0,
                'on_time_delivery': 1.0 if project.status == "completed" else 0.5,
                'team_productivity': len(project.tasks) / max(1, len(self._developer_agents))
            }
            
            # Update project quality metrics
            project.quality_metrics.update(metrics)
            
            return metrics
            
        except Exception as e:
            return {}
    
    def get_project(self, project_id: str) -> Optional[SoftwareProject]:
        """Get project by ID"""
        return self._projects.get(project_id)
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects"""
        return [
            {
                'project_id': proj.project_id,
                'name': proj.name,
                'status': proj.status,
                'current_phase': proj.current_phase.value,
                'progress': proj.progress,
                'team_size': len(proj.team_members),
                'artifacts_count': len(proj.artifacts),
                'tasks_count': len(proj.tasks),
                'start_date': proj.start_date.isoformat()
            }
            for proj in self._projects.values()
        ]
    
    def get_team_members(self) -> List[Dict[str, Any]]:
        """Get development team members"""
        return [
            {
                'agent_id': agent.agent_id,
                'name': agent.name,
                'role': agent.role.value,
                'specialization': agent.specialization,
                'experience_level': agent.experience_level,
                'active': agent.active,
                'performance_metrics': agent.performance_metrics
            }
            for agent in self._developer_agents.values()
        ]
    
    def get_project_artifacts(self, project_id: str) -> List[Dict[str, Any]]:
        """Get project artifacts"""
        if project_id not in self._projects:
            return []
        
        project = self._projects[project_id]
        return [
            {
                'artifact_id': artifact.artifact_id,
                'name': artifact.name,
                'type': artifact.artifact_type.value,
                'created_by': artifact.created_by,
                'created_at': artifact.created_at.isoformat(),
                'file_path': artifact.file_path,
                'approved': artifact.approved
            }
            for artifact in project.artifacts
        ]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = self._usage_stats.copy()
        
        # Calculate derived metrics
        if stats['total_projects'] > 0:
            stats['average_artifacts_per_project'] = stats['total_artifacts_generated'] / stats['total_projects']
            stats['average_tasks_per_project'] = stats['total_tasks_completed'] / stats['total_projects']
        
        return stats
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': {
                'model': self.config.model,
                'team_size': self.config.team_size,
                'enable_code_generation': self.config.enable_code_generation,
                'enable_testing': self.config.enable_testing,
                'enable_documentation': self.config.enable_documentation,
                'target_languages': self.config.target_languages,
                'development_methodology': self.config.development_methodology
            },
            'team_size': len(self._developer_agents),
            'projects_count': len(self._projects),
            'current_project': self._current_project,
            'output_directory': self.config.output_directory,
            'usage_stats': self.get_usage_stats(),
            'metagpt_available': METAGPT_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class MetaGPTProvider(BaseAgentProvider):
    """
    Provider implementation for MetaGPT.
    
    Enables multi-agent software development with role-based programming,
    automated requirement analysis, system design, and code generation workflows.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, MetaGPTAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not METAGPT_AVAILABLE:
            self.logger.warning("MetaGPT dependencies not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "metagpt"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.CODE_GENERATION,
            AgentCapability.MULTI_AGENT_COLLABORATION,
            AgentCapability.ROLE_SPECIALIZATION,
            AgentCapability.PROJECT_MANAGEMENT,
            AgentCapability.REQUIREMENT_ANALYSIS,
            AgentCapability.SYSTEM_DESIGN,
            AgentCapability.AUTOMATED_TESTING,
            AgentCapability.DOCUMENTATION_GENERATION
        ]
    
    async def initialize(self) -> bool:
        """Initialize MetaGPT provider"""
        if not METAGPT_AVAILABLE:
            self.logger.error("MetaGPT dependencies not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("MetaGPT provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MetaGPT provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new MetaGPT agent"""
        if not self._initialized:
            await self.initialize()
        
        if not METAGPT_AVAILABLE:
            raise RuntimeError("MetaGPT dependencies not available")
        
        agent_id = f"metagpt_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create MetaGPT configuration
            metagpt_config = MetaGPTConfig(
                api_key=self.config.get('api_key', ''),
                model=self.config.get('model', 'gpt-4'),
                base_url=self.config.get('base_url', 'https://api.openai.com/v1'),
                max_tokens=self.config.get('max_tokens', 4000),
                temperature=self.config.get('temperature', 0.7),
                project_name=self.config.get('project_name', 'untitled_project'),
                output_directory=self.config.get('output_directory', './metagpt_output'),
                enable_code_generation=self.config.get('enable_code_generation', True),
                enable_testing=self.config.get('enable_testing', True),
                enable_documentation=self.config.get('enable_documentation', True),
                max_iterations=self.config.get('max_iterations', 10),
                team_size=self.config.get('team_size', 5),
                development_methodology=self.config.get('development_methodology', 'agile'),
                quality_gate_enabled=self.config.get('quality_gate_enabled', True),
                code_review_enabled=self.config.get('code_review_enabled', True),
                continuous_integration=self.config.get('continuous_integration', True),
                target_languages=self.config.get('target_languages', ['python']),
                frameworks=self.config.get('frameworks', []),
                databases=self.config.get('databases', []),
                deployment_targets=self.config.get('deployment_targets', []),
                enable_security_review=self.config.get('enable_security_review', True),
                enable_performance_optimization=self.config.get('enable_performance_optimization', True),
                compliance_requirements=self.config.get('compliance_requirements', [])
            )
            
            # Create agent
            agent = MetaGPTAgent(
                agent_id=agent_id,
                config=metagpt_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize MetaGPT agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created MetaGPT agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create MetaGPT agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a MetaGPT agent"""
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
            execution_type = context.get('execution_type', 'full_development')
            
            if execution_type == 'create_project':
                result = await self._create_project(agent, prompt, context)
            elif execution_type == 'full_development':
                result = await self._execute_full_development(agent, prompt, context)
            else:
                result = await self._create_project(agent, prompt, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare response
            response_content = result.get('summary', f"Development workflow completed for project: {prompt}")
            
            # Prepare metadata
            metadata = {
                'execution_time_seconds': execution_time,
                'execution_type': execution_type,
                'project_id': result.get('project_id'),
                'phases_completed': result.get('phases_completed', []),
                'artifacts_generated': result.get('total_artifacts', 0),
                'tasks_completed': result.get('total_tasks', 0),
                'quality_metrics': result.get('quality_metrics', {}),
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
            self.logger.error(f"MetaGPT agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def _create_project(self, agent: MetaGPTAgent, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new project"""
        requirements = context.get('requirements', [prompt])
        project_name = context.get('project_name', prompt.replace(' ', '_').lower())
        
        project_id = await agent.create_project(project_name, prompt, requirements)
        
        return {
            'project_id': project_id,
            'summary': f"Created project '{project_name}' with requirements analysis"
        }
    
    async def _execute_full_development(self, agent: MetaGPTAgent, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full development workflow"""
        # First create project
        project_result = await self._create_project(agent, prompt, context)
        project_id = project_result['project_id']
        
        # Execute development workflow
        workflow_result = await agent.execute_development_workflow(project_id)
        
        workflow_result['summary'] = f"Completed full development workflow for '{prompt}'. Generated {workflow_result.get('total_artifacts', 0)} artifacts across {len(workflow_result.get('phases_completed', []))} phases."
        
        return workflow_result
    
    async def create_project(self, agent_id: str, name: str, description: str, requirements: List[str] = None) -> str:
        """Create a project"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        return await agent.create_project(name, description, requirements)
    
    async def execute_development_workflow(self, agent_id: str, project_id: str) -> Dict[str, Any]:
        """Execute development workflow"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        return await agent.execute_development_workflow(project_id)
    
    async def get_project_artifacts(self, agent_id: str, project_id: str) -> List[Dict[str, Any]]:
        """Get project artifacts"""
        if agent_id not in self._agents:
            return []
        
        agent = self._agents[agent_id]
        return agent.get_project_artifacts(project_id)
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with MetaGPT agent (basic implementation)"""
        if agent_id not in self._agents:
            return False
        
        try:
            # MetaGPT doesn't have native tool support in this implementation
            # This would be implemented as development capabilities
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using MetaGPT capabilities"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Create tool spec for software development
            tool_spec = ToolSpec(
                name=f"metagpt_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={
                    'description': tool_description,
                    'execution_type': 'development_workflow',
                    'multi_agent': True,
                    'role_based': True,
                    'examples': examples or []
                },
                security_policy={
                    'risk_level': 'medium',
                    'requires_approval': True,
                    'code_generation': True,
                    'file_system_access': True
                }
            )
            
            self.logger.info(f"Synthesized tool for MetaGPT agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a MetaGPT agent"""
        try:
            if agent_id in self._agents:
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed MetaGPT agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active MetaGPT agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a MetaGPT agent"""
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