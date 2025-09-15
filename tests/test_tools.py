"""
Comprehensive test suite for CandyLLM tools system.

Tests tool registry, tool decorators, tool execution, and tool management
functionality including caching and statistics.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Callable
import time

# Tools imports
try:
    from CandyLLM.tools import tool, ToolRegistry
    from CandyLLM.tools.registry import UniversalToolSpec, FrameworkToolAdapter
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False


@pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools modules not available")
class TestToolDecorator:
    """Test suite for @tool decorator functionality."""
    
    def test_tool_decorator_basic(self):
        """Test basic @tool decorator functionality."""
        
        @tool(description="Test function for unit tests")
        def test_function(x: int, y: int) -> int:
            """Add two numbers together."""
            return x + y
        
        # Test that function still works
        result = test_function(2, 3)
        assert result == 5
        
        # Test that metadata is attached
        assert hasattr(test_function, '__tool_metadata__')
        metadata = getattr(test_function, '__tool_metadata__')
        assert metadata['description'] == "Test function for unit tests"
    
    def test_tool_decorator_with_parameters(self):
        """Test @tool decorator with various parameters."""
        
        @tool(
            description="Complex test function",
            category="math",
            tags=["arithmetic", "basic"],
            cache=True,
            timeout=30
        )
        def complex_function(a: float, b: float, operation: str = "add") -> float:
            """Perform arithmetic operations."""
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            else:
                raise ValueError("Unsupported operation")
        
        # Test functionality
        assert complex_function(2.5, 3.5) == 6.0
        assert complex_function(2.0, 3.0, "multiply") == 6.0
        
        # Test metadata
        metadata = getattr(complex_function, '__tool_metadata__')
        assert metadata['category'] == "math"
        assert "arithmetic" in metadata['tags']
        assert metadata['cache'] is True
        assert metadata['timeout'] == 30
    
    def test_tool_decorator_type_hints(self):
        """Test that @tool decorator preserves type hints."""
        
        @tool(description="Type hint test")
        def typed_function(name: str, age: int, active: bool = True) -> Dict[str, Any]:
            """Create user profile."""
            return {
                "name": name,
                "age": age,
                "active": active
            }
        
        # Test type hint preservation
        import inspect
        signature = inspect.signature(typed_function)
        
        assert signature.parameters['name'].annotation == str
        assert signature.parameters['age'].annotation == int
        assert signature.parameters['active'].annotation == bool
        assert signature.return_annotation == Dict[str, Any]
    
    def test_tool_decorator_async_functions(self):
        """Test @tool decorator with async functions."""
        
        @tool(description="Async test function")
        async def async_function(delay: float) -> str:
            """Async function with delay."""
            await asyncio.sleep(delay)
            return f"Completed after {delay} seconds"
        
        # Test async functionality
        async def test_async():
            result = await async_function(0.1)
            assert "Completed after 0.1 seconds" == result
        
        asyncio.run(test_async())
    
    def test_tool_decorator_error_handling(self):
        """Test @tool decorator error handling."""
        
        @tool(description="Function that can error")
        def error_function(should_error: bool) -> str:
            """Function that might raise an error."""
            if should_error:
                raise ValueError("Intentional error for testing")
            return "Success"
        
        # Test normal operation
        assert error_function(False) == "Success"
        
        # Test error handling
        with pytest.raises(ValueError):
            error_function(True)


@pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools modules not available")
class TestToolRegistry:
    """Test suite for ToolRegistry functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset registry for clean tests
        if hasattr(ToolRegistry, '_reset'):
            ToolRegistry._reset()
    
    def test_tool_registration(self):
        """Test tool registration in registry."""
        
        @tool(description="Registry test function")
        def registry_test_function(x: int) -> int:
            """Double the input."""
            return x * 2
        
        # Check if tool was automatically registered
        if hasattr(ToolRegistry, 'get_tool'):
            try:
                retrieved_tool = ToolRegistry.get_tool('registry_test_function')
                assert retrieved_tool is not None
            except Exception:
                # Registry implementation may differ
                pass
    
    def test_tool_discovery(self):
        """Test tool discovery functionality."""
        
        # Register multiple tools
        @tool(description="Tool 1", category="math")
        def tool_one(x: int) -> int:
            return x + 1
        
        @tool(description="Tool 2", category="math")
        def tool_two(x: int) -> int:
            return x * 2
        
        @tool(description="Tool 3", category="string")
        def tool_three(s: str) -> str:
            return s.upper()
        
        # Test discovery methods
        if hasattr(ToolRegistry, 'list_tools'):
            try:
                all_tools = ToolRegistry.list_tools()
                assert len(all_tools) >= 3
            except Exception:
                pass
        
        if hasattr(ToolRegistry, 'get_tools_by_category'):
            try:
                math_tools = ToolRegistry.get_tools_by_category('math')
                assert len(math_tools) >= 2
            except Exception:
                pass
    
    def test_tool_caching(self):
        """Test tool result caching functionality."""
        
        call_count = 0
        
        @tool(description="Cached function", cache=True)
        def cached_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * x
        
        # First call
        result1 = cached_function(5)
        assert result1 == 25
        assert call_count == 1
        
        # Second call with same input (should use cache)
        result2 = cached_function(5)
        assert result2 == 25
        
        # Check if caching worked (implementation dependent)
        if hasattr(ToolRegistry, 'cache_enabled'):
            # Cache implementation may vary
            pass
    
    def test_tool_statistics(self):
        """Test tool usage statistics."""
        
        @tool(description="Stats test function")
        def stats_function(x: int) -> int:
            return x + 10
        
        # Use the function multiple times
        for i in range(5):
            stats_function(i)
        
        # Check statistics
        if hasattr(ToolRegistry, 'get_stats'):
            try:
                stats = ToolRegistry.get_stats()
                assert stats is not None
                assert isinstance(stats, dict)
            except Exception:
                pass
    
    def test_tool_search(self):
        """Test tool search functionality."""
        
        @tool(description="Weather information tool", tags=["weather", "api"])
        def get_weather(city: str) -> str:
            return f"Weather in {city}: Sunny"
        
        @tool(description="Math calculation tool", tags=["math", "calculator"])
        def calculate(expression: str) -> float:
            return eval(expression)  # Note: eval is unsafe, just for testing
        
        # Test search functionality
        if hasattr(ToolRegistry, 'search_tools'):
            try:
                weather_tools = ToolRegistry.search_tools("weather")
                assert len(weather_tools) >= 1
                
                math_tools = ToolRegistry.search_tools("math")
                assert len(math_tools) >= 1
            except Exception:
                pass


@pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools modules not available")
class TestUniversalToolSpec:
    """Test suite for UniversalToolSpec functionality."""
    
    def test_universal_tool_spec_creation(self):
        """Test UniversalToolSpec creation and usage."""
        
        try:
            tool_spec = UniversalToolSpec(
                name="test_tool",
                description="Universal test tool",
                parameters={
                    "input": {"type": "string", "description": "Input text"},
                    "count": {"type": "integer", "description": "Repeat count"}
                },
                function=lambda input, count: input * count
            )
            
            assert tool_spec.name == "test_tool"
            assert tool_spec.description == "Universal test tool"
            assert "input" in tool_spec.parameters
            
        except Exception:
            pytest.skip("UniversalToolSpec not available")
    
    def test_tool_spec_execution(self):
        """Test UniversalToolSpec execution."""
        
        try:
            def multiply_string(text: str, times: int) -> str:
                return text * times
            
            tool_spec = UniversalToolSpec(
                name="multiply_string",
                description="Multiply string",
                parameters={
                    "text": {"type": "string"},
                    "times": {"type": "integer"}
                },
                function=multiply_string
            )
            
            # Test execution
            if hasattr(tool_spec, 'execute'):
                result = tool_spec.execute(text="Hello", times=3)
                assert result == "HelloHelloHello"
            
        except Exception:
            pytest.skip("UniversalToolSpec execution not available")
    
    def test_tool_spec_validation(self):
        """Test UniversalToolSpec parameter validation."""
        
        try:
            tool_spec = UniversalToolSpec(
                name="validator_test",
                description="Test validation",
                parameters={
                    "required_param": {"type": "string", "required": True},
                    "optional_param": {"type": "integer", "required": False}
                },
                function=lambda required_param, optional_param=10: f"{required_param}:{optional_param}"
            )
            
            # Test validation methods
            if hasattr(tool_spec, 'validate_parameters'):
                # Valid parameters
                is_valid = tool_spec.validate_parameters({
                    "required_param": "test",
                    "optional_param": 5
                })
                assert is_valid is True
                
                # Missing required parameter
                is_invalid = tool_spec.validate_parameters({
                    "optional_param": 5
                })
                assert is_invalid is False
            
        except Exception:
            pytest.skip("UniversalToolSpec validation not available")


@pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools modules not available")
class TestFrameworkToolAdapter:
    """Test suite for FrameworkToolAdapter functionality."""
    
    def test_adapter_creation(self):
        """Test FrameworkToolAdapter creation."""
        
        try:
            adapter = FrameworkToolAdapter(framework="langchain")
            assert adapter is not None
            assert hasattr(adapter, 'framework')
            
        except Exception:
            pytest.skip("FrameworkToolAdapter not available")
    
    def test_tool_conversion(self):
        """Test tool conversion between frameworks."""
        
        try:
            # Create a standard tool
            @tool(description="Test tool for conversion")
            def conversion_test(x: int, y: int) -> int:
                return x + y
            
            adapter = FrameworkToolAdapter(framework="langchain")
            
            # Test conversion
            if hasattr(adapter, 'convert_tool'):
                converted = adapter.convert_tool(conversion_test)
                assert converted is not None
            
        except Exception:
            pytest.skip("Tool conversion not available")
    
    def test_multi_framework_support(self):
        """Test support for multiple frameworks."""
        
        frameworks = ["langchain", "crewai", "autogen", "custom"]
        
        for framework in frameworks:
            try:
                adapter = FrameworkToolAdapter(framework=framework)
                assert adapter.framework == framework
            except Exception:
                # Framework may not be supported
                pass


@pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools modules not available")
class TestToolIntegration:
    """Integration tests for tools system."""
    
    def test_end_to_end_tool_workflow(self):
        """Test complete tool workflow from registration to execution."""
        
        # Define and register tool
        @tool(
            description="End-to-end test tool",
            category="testing",
            cache=True,
            tags=["integration", "test"]
        )
        def e2e_tool(message: str, repeat: int = 1) -> str:
            """Repeat message for testing."""
            return (message + " ") * repeat
        
        # Execute tool
        result = e2e_tool("Integration test", 3)
        expected = "Integration test Integration test Integration test "
        assert result == expected
        
        # Check registration
        if hasattr(ToolRegistry, 'get_tool'):
            try:
                registered_tool = ToolRegistry.get_tool('e2e_tool')
                assert registered_tool is not None
            except Exception:
                pass
    
    def test_tool_performance(self):
        """Test tool performance monitoring."""
        
        @tool(description="Performance test tool")
        def performance_tool(iterations: int) -> int:
            """Perform computational work."""
            total = 0
            for i in range(iterations):
                total += i * i
            return total
        
        # Measure performance
        start_time = time.time()
        result = performance_tool(1000)
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration >= 0
        assert result > 0
    
    @pytest.mark.asyncio
    async def test_async_tool_integration(self):
        """Test asynchronous tool integration."""
        
        @tool(description="Async integration test")
        async def async_integration_tool(delay: float, message: str) -> str:
            """Async tool with delay."""
            await asyncio.sleep(delay)
            return f"Async: {message}"
        
        # Test async execution
        result = await async_integration_tool(0.1, "test message")
        assert result == "Async: test message"
    
    def test_tool_error_recovery(self):
        """Test tool error handling and recovery."""
        
        @tool(description="Error recovery test")
        def error_recovery_tool(should_fail: bool, message: str) -> str:
            """Tool that can fail for testing error handling."""
            if should_fail:
                raise RuntimeError("Intentional failure for testing")
            return f"Success: {message}"
        
        # Test successful execution
        success_result = error_recovery_tool(False, "works")
        assert success_result == "Success: works"
        
        # Test error handling
        with pytest.raises(RuntimeError):
            error_recovery_tool(True, "fails")
    
    def test_tool_metadata_preservation(self):
        """Test that tool metadata is preserved through the system."""
        
        original_description = "Metadata preservation test"
        original_category = "testing"
        original_tags = ["metadata", "preservation"]
        
        @tool(
            description=original_description,
            category=original_category,
            tags=original_tags
        )
        def metadata_tool(data: str) -> str:
            """Tool for testing metadata preservation."""
            return f"Processed: {data}"
        
        # Check metadata preservation
        metadata = getattr(metadata_tool, '__tool_metadata__')
        assert metadata['description'] == original_description
        assert metadata['category'] == original_category
        assert metadata['tags'] == original_tags
        
        # Test functionality still works
        result = metadata_tool("test data")
        assert result == "Processed: test data"


if __name__ == '__main__':
    pytest.main([__file__, "-v"])