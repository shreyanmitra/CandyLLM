"""
Universal Tool Synthesis System

Framework-agnostic tool specification and conversion system for cross-framework
compatibility in the CandyLLM agentic provider ecosystem.
"""

import uuid
import json
import inspect
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import importlib.util

from .base import ToolSpec, AgentCapability


class ToolFormat(Enum):
    """Supported tool formats across frameworks"""
    CANDYLLM = "candyllm"
    LANGCHAIN = "langchain"
    CREWAI = "crewai"
    OPENAI_FUNCTION = "openai_function"
    ANTHROPIC_TOOL = "anthropic_tool"
    AUTOGEN = "autogen"
    HAYSTACK = "haystack"


class ParameterType(Enum):
    """Universal parameter types"""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ANY = "any"


@dataclass
class Parameter:
    """Universal parameter specification"""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Any = None
    enum: List[Any] = field(default_factory=list)
    format: Optional[str] = None  # e.g., "date-time", "email"
    minimum: Optional[Union[int, float]] = None
    maximum: Optional[Union[int, float]] = None
    pattern: Optional[str] = None  # regex pattern for strings


@dataclass
class UniversalToolSpec:
    """Enhanced universal tool specification with cross-framework compatibility"""
    name: str
    description: str
    parameters: List[Parameter]
    returns: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    security_policy: Dict[str, Any] = field(default_factory=dict)
    implementation: Optional[Callable] = None
    source_code: Optional[str] = None
    
    # Framework-specific adaptations
    framework_adaptations: Dict[ToolFormat, Dict[str, Any]] = field(default_factory=dict)
    
    # Versioning and metadata
    version: str = "1.0.0"
    author: str = ""
    license: str = ""
    documentation_url: str = ""
    repository_url: str = ""


class FrameworkToolAdapter(ABC):
    """Abstract base for framework-specific tool adapters"""
    
    @property
    @abstractmethod
    def target_format(self) -> ToolFormat:
        """Target framework format"""
        pass
    
    @abstractmethod
    def from_universal(self, universal_spec: UniversalToolSpec) -> Any:
        """Convert from universal spec to framework-specific format"""
        pass
    
    @abstractmethod
    def to_universal(self, framework_tool: Any) -> UniversalToolSpec:
        """Convert from framework-specific format to universal spec"""
        pass
    
    def supports_feature(self, feature: str) -> bool:
        """Check if framework supports a specific feature"""
        return True  # Default implementation


class LangChainAdapter(FrameworkToolAdapter):
    """Adapter for LangChain tools"""
    
    @property
    def target_format(self) -> ToolFormat:
        return ToolFormat.LANGCHAIN
    
    def from_universal(self, universal_spec: UniversalToolSpec) -> Any:
        """Convert universal spec to LangChain BaseTool"""
        try:
            from langchain.tools import BaseTool
            from pydantic import Field
            
            # Create dynamic Pydantic model for arguments
            class DynamicArgs:
                pass
            
            # Add fields to the args class
            for param in universal_spec.parameters:
                if param.required:
                    setattr(DynamicArgs, param.name, Field(..., description=param.description))
                else:
                    setattr(DynamicArgs, param.name, Field(param.default, description=param.description))
            
            class UniversalLangChainTool(BaseTool):
                name: str = universal_spec.name
                description: str = universal_spec.description
                args_schema: Type = DynamicArgs
                
                def _run(self, **kwargs) -> str:
                    if universal_spec.implementation:
                        try:
                            result = universal_spec.implementation(**kwargs)
                            return str(result)
                        except Exception as e:
                            return f"Tool execution failed: {e}"
                    return f"Tool {self.name} executed with {kwargs}"
                
                async def _arun(self, **kwargs) -> str:
                    if universal_spec.implementation:
                        try:
                            if asyncio.iscoroutinefunction(universal_spec.implementation):
                                result = await universal_spec.implementation(**kwargs)
                            else:
                                result = universal_spec.implementation(**kwargs)
                            return str(result)
                        except Exception as e:
                            return f"Tool execution failed: {e}"
                    return f"Tool {self.name} executed with {kwargs}"
            
            return UniversalLangChainTool()
            
        except ImportError:
            raise RuntimeError("LangChain not available for tool conversion")
    
    def to_universal(self, framework_tool: Any) -> UniversalToolSpec:
        """Convert LangChain tool to universal spec"""
        try:
            # Extract parameters from args_schema
            parameters = []
            if hasattr(framework_tool, 'args_schema') and framework_tool.args_schema:
                schema = framework_tool.args_schema.schema()
                for prop_name, prop_def in schema.get('properties', {}).items():
                    param = Parameter(
                        name=prop_name,
                        type=ParameterType(prop_def.get('type', 'string')),
                        description=prop_def.get('description', ''),
                        required=prop_name in schema.get('required', []),
                        default=prop_def.get('default')
                    )
                    parameters.append(param)
            
            return UniversalToolSpec(
                name=framework_tool.name,
                description=framework_tool.description,
                parameters=parameters,
                implementation=framework_tool._run if hasattr(framework_tool, '_run') else None
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert LangChain tool: {e}")


class CrewAIAdapter(FrameworkToolAdapter):
    """Adapter for CrewAI tools"""
    
    @property
    def target_format(self) -> ToolFormat:
        return ToolFormat.CREWAI
    
    def from_universal(self, universal_spec: UniversalToolSpec) -> Any:
        """Convert universal spec to CrewAI BaseTool"""
        try:
            from crewai.tools import BaseTool
            
            class UniversalCrewAITool(BaseTool):
                name: str = universal_spec.name
                description: str = universal_spec.description
                
                def _run(self, *args, **kwargs) -> str:
                    if universal_spec.implementation:
                        try:
                            if args and not kwargs:
                                result = universal_spec.implementation(*args)
                            elif kwargs and not args:
                                result = universal_spec.implementation(**kwargs)
                            else:
                                result = universal_spec.implementation(*args, **kwargs)
                            return str(result)
                        except Exception as e:
                            return f"Tool execution failed: {e}"
                    return f"Tool {self.name} executed"
            
            return UniversalCrewAITool()
            
        except ImportError:
            raise RuntimeError("CrewAI not available for tool conversion")
    
    def to_universal(self, framework_tool: Any) -> UniversalToolSpec:
        """Convert CrewAI tool to universal spec"""
        return UniversalToolSpec(
            name=framework_tool.name,
            description=framework_tool.description,
            parameters=[],  # CrewAI tools don't have structured parameters
            implementation=framework_tool._run if hasattr(framework_tool, '_run') else None
        )


class OpenAIFunctionAdapter(FrameworkToolAdapter):
    """Adapter for OpenAI Function Calling format"""
    
    @property
    def target_format(self) -> ToolFormat:
        return ToolFormat.OPENAI_FUNCTION
    
    def from_universal(self, universal_spec: UniversalToolSpec) -> Dict[str, Any]:
        """Convert universal spec to OpenAI function definition"""
        properties = {}
        required = []
        
        for param in universal_spec.parameters:
            prop_def = {
                "type": param.type.value,
                "description": param.description
            }
            
            if param.enum:
                prop_def["enum"] = param.enum
            if param.minimum is not None:
                prop_def["minimum"] = param.minimum
            if param.maximum is not None:
                prop_def["maximum"] = param.maximum
            if param.pattern:
                prop_def["pattern"] = param.pattern
            
            properties[param.name] = prop_def
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": universal_spec.name,
                "description": universal_spec.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def to_universal(self, framework_tool: Dict[str, Any]) -> UniversalToolSpec:
        """Convert OpenAI function to universal spec"""
        function_def = framework_tool.get('function', {})
        parameters = []
        
        props = function_def.get('parameters', {}).get('properties', {})
        required_params = function_def.get('parameters', {}).get('required', [])
        
        for param_name, param_def in props.items():
            param = Parameter(
                name=param_name,
                type=ParameterType(param_def.get('type', 'string')),
                description=param_def.get('description', ''),
                required=param_name in required_params,
                enum=param_def.get('enum', []),
                minimum=param_def.get('minimum'),
                maximum=param_def.get('maximum'),
                pattern=param_def.get('pattern')
            )
            parameters.append(param)
        
        return UniversalToolSpec(
            name=function_def.get('name', ''),
            description=function_def.get('description', ''),
            parameters=parameters
        )


class ToolSynthesizer:
    """Advanced tool synthesis engine with multi-framework support"""
    
    def __init__(self):
        self.adapters: Dict[ToolFormat, FrameworkToolAdapter] = {
            ToolFormat.LANGCHAIN: LangChainAdapter(),
            ToolFormat.CREWAI: CrewAIAdapter(),
            ToolFormat.OPENAI_FUNCTION: OpenAIFunctionAdapter()
        }
        self.tool_registry: Dict[str, UniversalToolSpec] = {}
        
    def register_adapter(self, adapter: FrameworkToolAdapter):
        """Register a new framework adapter"""
        self.adapters[adapter.target_format] = adapter
    
    def register_tool(self, tool_spec: UniversalToolSpec):
        """Register a universal tool"""
        self.tool_registry[tool_spec.name] = tool_spec
    
    def convert_tool(self, tool: Any, source_format: ToolFormat, target_format: ToolFormat) -> Any:
        """Convert a tool between frameworks"""
        # First convert to universal format
        if source_format not in self.adapters:
            raise ValueError(f"No adapter available for source format: {source_format}")
        
        source_adapter = self.adapters[source_format]
        universal_spec = source_adapter.to_universal(tool)
        
        # Then convert to target format
        return self.convert_from_universal(universal_spec, target_format)
    
    def convert_from_universal(self, universal_spec: UniversalToolSpec, target_format: ToolFormat) -> Any:
        """Convert from universal spec to target framework format"""
        if target_format not in self.adapters:
            raise ValueError(f"No adapter available for target format: {target_format}")
        
        target_adapter = self.adapters[target_format]
        return target_adapter.from_universal(universal_spec)
    
    def convert_to_universal(self, tool: Any, source_format: ToolFormat) -> UniversalToolSpec:
        """Convert from framework format to universal spec"""
        if source_format not in self.adapters:
            raise ValueError(f"No adapter available for source format: {source_format}")
        
        source_adapter = self.adapters[source_format]
        return source_adapter.to_universal(tool)
    
    async def synthesize_tool_from_description(self, description: str, examples: List[str] = None,
                                             target_formats: List[ToolFormat] = None) -> Dict[ToolFormat, Any]:
        """Synthesize a tool from natural language description"""
        # This would use an LLM to generate the tool implementation
        # For now, create a basic template
        
        tool_name = f"synthesized_{uuid.uuid4().hex[:8]}"
        
        # Parse parameters from description (simplified)
        parameters = [
            Parameter(
                name="input",
                type=ParameterType.STRING,
                description="Input for the tool",
                required=True
            )
        ]
        
        # Create basic implementation
        def basic_implementation(input: str) -> str:
            return f"Processed: {input}"
        
        universal_spec = UniversalToolSpec(
            name=tool_name,
            description=description,
            parameters=parameters,
            implementation=basic_implementation,
            examples=examples or [],
            categories=["synthesized"],
            security_policy={"risk_level": "low", "requires_review": True}
        )
        
        # Convert to target formats
        target_formats = target_formats or list(self.adapters.keys())
        results = {}
        
        for format in target_formats:
            try:
                results[format] = self.convert_from_universal(universal_spec, format)
            except Exception as e:
                results[format] = f"Conversion failed: {e}"
        
        return results
    
    def create_tool_bundle(self, tools: List[UniversalToolSpec], 
                          target_format: ToolFormat) -> List[Any]:
        """Create a bundle of tools in target framework format"""
        bundle = []
        for tool in tools:
            try:
                converted = self.convert_from_universal(tool, target_format)
                bundle.append(converted)
            except Exception as e:
                # Log error but continue with other tools
                print(f"Failed to convert tool {tool.name}: {e}")
        
        return bundle
    
    def validate_tool_compatibility(self, tool_spec: UniversalToolSpec, 
                                  target_format: ToolFormat) -> Dict[str, Any]:
        """Validate if a tool is compatible with a target framework"""
        adapter = self.adapters.get(target_format)
        if not adapter:
            return {"compatible": False, "reason": "No adapter available"}
        
        compatibility_issues = []
        
        # Check parameter types
        for param in tool_spec.parameters:
            if param.type == ParameterType.ANY and not adapter.supports_feature("any_type"):
                compatibility_issues.append(f"Parameter {param.name} uses unsupported 'any' type")
        
        # Check security features
        if tool_spec.security_policy and not adapter.supports_feature("security_policies"):
            compatibility_issues.append("Security policies not supported")
        
        return {
            "compatible": len(compatibility_issues) == 0,
            "issues": compatibility_issues,
            "recommendations": self._get_compatibility_recommendations(compatibility_issues)
        }
    
    def _get_compatibility_recommendations(self, issues: List[str]) -> List[str]:
        """Get recommendations for fixing compatibility issues"""
        recommendations = []
        
        for issue in issues:
            if "any" in issue.lower():
                recommendations.append("Consider using more specific parameter types")
            elif "security" in issue.lower():
                recommendations.append("Implement security checks within tool implementation")
        
        return recommendations
    
    def export_tool_catalog(self, format: str = "json") -> str:
        """Export the tool registry catalog"""
        if format == "json":
            catalog = {
                "tools": {
                    name: {
                        "name": spec.name,
                        "description": spec.description,
                        "version": spec.version,
                        "categories": spec.categories,
                        "parameters": [
                            {
                                "name": p.name,
                                "type": p.type.value,
                                "description": p.description,
                                "required": p.required
                            }
                            for p in spec.parameters
                        ]
                    }
                    for name, spec in self.tool_registry.items()
                },
                "adapters": list(self.adapters.keys()),
                "total_tools": len(self.tool_registry)
            }
            return json.dumps(catalog, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global tool synthesizer instance
global_synthesizer = ToolSynthesizer()


def register_universal_tool(tool_spec: UniversalToolSpec):
    """Register a universal tool with the global synthesizer"""
    global_synthesizer.register_tool(tool_spec)


def convert_tool_for_provider(tool: Any, source_format: ToolFormat, 
                            target_format: ToolFormat) -> Any:
    """Convert a tool for use with a different provider"""
    return global_synthesizer.convert_tool(tool, source_format, target_format)


def synthesize_tool(description: str, examples: List[str] = None,
                   target_formats: List[ToolFormat] = None) -> Dict[ToolFormat, Any]:
    """Synthesize a tool from description for multiple frameworks"""
    return asyncio.run(
        global_synthesizer.synthesize_tool_from_description(
            description, examples, target_formats
        )
    )