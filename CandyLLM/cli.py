#!/usr/bin/env python3
"""
🍭 CandyLLM CLI - Command Line Interface
Launch UIs and manage CandyLLM from the terminal

Usage:
    candyllm ui                    # Launch enhanced legacy UI
    candyllm ui advanced           # Launch advanced UI 2.0
    candyllm ui --port 8080        # Custom port
    candyllm models list           # List available models
    candyllm tools list            # List registered tools
    candyllm --version             # Show version
"""

import argparse
import sys
import os
from typing import Optional

# Add CandyLLM to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def create_parser():
    """Create the CLI argument parser"""
    parser = argparse.ArgumentParser(
        prog='candyllm',
        description='🍭 CandyLLM - Universal LLM Interface',
        epilog='For more help: https://github.com/yourusername/CandyLLM'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='CandyLLM 2.0.0'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # UI Commands
    ui_parser = subparsers.add_parser('ui', help='Launch user interfaces')
    ui_parser.add_argument(
        'ui_type', 
        nargs='?', 
        default='legacy',
        choices=['legacy', 'advanced', 'auto'],
        help='UI type to launch (default: legacy)'
    )
    ui_parser.add_argument(
        '--port', 
        type=int, 
        default=7860,
        help='Port to run on (default: 7860)'
    )
    ui_parser.add_argument(
        '--host', 
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    ui_parser.add_argument(
        '--share', 
        action='store_true',
        help='Create public Gradio link'
    )
    ui_parser.add_argument(
        '--no-launch', 
        action='store_true',
        help='Create interface without launching'
    )
    
    # Model Commands
    models_parser = subparsers.add_parser('models', help='Model management')
    models_parser.add_argument(
        'action',
        choices=['list', 'test', 'info'],
        help='Action to perform'
    )
    models_parser.add_argument(
        'model_name',
        nargs='?',
        help='Model name for test/info actions'
    )
    
    # Tools Commands
    tools_parser = subparsers.add_parser('tools', help='Tool management')
    tools_parser.add_argument(
        'action',
        choices=['list', 'register', 'test'],
        help='Action to perform'
    )
    tools_parser.add_argument(
        'tool_name',
        nargs='?',
        help='Tool name for register/test actions'
    )
    
    # Config Commands
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument(
        'action',
        choices=['show', 'set', 'reset'],
        help='Configuration action'
    )
    config_parser.add_argument(
        'key',
        nargs='?',
        help='Configuration key'
    )
    config_parser.add_argument(
        'value',
        nargs='?',
        help='Configuration value'
    )
    
    return parser

def launch_ui(ui_type: str, port: int, host: str, share: bool, launch: bool):
    """Launch the specified UI type"""
    try:
        from CandyLLM import LLMWrapper
        
        print(f"🍭 Launching CandyLLM {ui_type.title()} UI...")
        print(f"🌐 Server: http://{host}:{port}")
        if share:
            print("🔗 Public link will be generated...")
        
        if ui_type == 'legacy':
            interface = LLMWrapper.getUI(launch=False)
            if launch:
                interface.launch(
                    server_name=host,
                    server_port=port,
                    share=share,
                    debug=True
                )
                
        elif ui_type == 'advanced':
            interface = LLMWrapper.getAdvancedUI(
                launch=launch,
                server_name=host,
                server_port=port,
                share=share
            )
            
        elif ui_type == 'auto':
            # Try advanced first, fall back to legacy
            try:
                interface = LLMWrapper.getAdvancedUI(
                    launch=launch,
                    server_name=host,
                    server_port=port,
                    share=share
                )
            except:
                print("⚠️  Advanced UI not available, using legacy UI")
                interface = LLMWrapper.getUI(launch=False)
                if launch:
                    interface.launch(
                        server_name=host,
                        server_port=port,
                        share=share
                    )
        
        if not launch:
            print("✅ Interface created successfully (not launched)")
            return interface
            
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        print("💡 Try: pip install gradio>=4.0.0")
        sys.exit(1)

def list_models():
    """List available models"""
    try:
        from CandyLLM import LLMWrapper
        from CandyLLM.providers import get_all_available_models
        
        print("🤖 Available Models:")
        print("=" * 50)
        
        # Legacy aliases
        print("\n📚 Legacy Models (Aliases):")
        for name in sorted(LLMWrapper.aliases.keys()):
            print(f"   • {name}")
        
        # All provider models
        try:
            all_models = get_all_available_models()
            
            print(f"\n🌐 All Provider Models ({len(all_models)} total):")
            
            current_provider = None
            for model in sorted(all_models):
                provider = model.split(':')[0] if ':' in model else 'unknown'
                if provider != current_provider:
                    current_provider = provider
                    print(f"\n   {provider.upper()}:")
                print(f"     • {model}")
                
        except ImportError:
            print("\n💡 Install providers for full model list:")
            print("   pip install openai anthropic litellm")
            
    except Exception as e:
        print(f"❌ Error listing models: {e}")

def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    try:
        from CandyLLM import LLMWrapper
        from CandyLLM.providers import get_model_info as provider_get_info
        
        print(f"ℹ️  Model Information: {model_name}")
        print("=" * 50)
        
        try:
            info = provider_get_info(model_name)
            
            print(f"📝 Name: {info.get('name', model_name)}")
            print(f"🏢 Provider: {info.get('provider', 'Unknown')}")
            print(f"📊 Context Length: {info.get('context_length', 'Unknown')}")
            print(f"💰 Cost (Input): ${info.get('input_cost_per_1k', 'Unknown')}/1K tokens")
            print(f"💰 Cost (Output): ${info.get('output_cost_per_1k', 'Unknown')}/1K tokens")
            print(f"🎯 Capabilities: {', '.join(info.get('capabilities', []))}")
            print(f"📅 Release Date: {info.get('release_date', 'Unknown')}")
            print(f"📝 Description: {info.get('description', 'No description available')}")
            
        except Exception as e:
            print(f"⚠️  Detailed info not available: {e}")
            print(f"🔍 Basic test will be performed instead...")
            test_model(model_name)
            
    except Exception as e:
        print(f"❌ Error getting model info: {e}")

def test_model(model_name: str):
    """Test a specific model"""
    try:
        from CandyLLM import LLMWrapper
        
        print(f"🧪 Testing model: {model_name}")
        
        # Create wrapper
        if ':' in model_name:
            source, name = model_name.split(':', 1)
            llm = LLMWrapper(source=source, modelName=name, modelNameType="path")
        else:
            llm = LLMWrapper(modelName=model_name)
        
        # Test query
        test_prompt = "Hello! Please respond with exactly: 'Test successful'"
        print(f"📝 Prompt: {test_prompt}")
        
        response = llm.answer(test_prompt, max_tokens=50)
        print(f"🤖 Response: {response}")
        
        if "test successful" in response.lower():
            print("✅ Model test PASSED")
        else:
            print("⚠️  Model test completed (response may vary)")
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")

def register_tool_from_cli(tool_path: str):
    """Register a tool from a Python file"""
    try:
        import importlib.util
        from CandyLLM.tools import ToolRegistry
        
        print(f"📦 Registering tool from: {tool_path}")
        
        # Load the module
        if not os.path.exists(tool_path):
            print(f"❌ File not found: {tool_path}")
            return
        
        spec = importlib.util.spec_from_file_location("custom_tools", tool_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find functions with @tool decorator or register manually
        registered_count = 0
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and not name.startswith('_'):
                try:
                    # Try to register the function
                    ToolRegistry.register(name, obj, description=f"Tool from {tool_path}")
                    registered_count += 1
                    print(f"✅ Registered: {name}")
                except Exception as e:
                    print(f"⚠️  Skipped {name}: {e}")
        
        print(f"🎉 Successfully registered {registered_count} tools")
        
    except Exception as e:
        print(f"❌ Error registering tool: {e}")

def test_tool(tool_name: str):
    """Test a registered tool"""
    try:
        from CandyLLM.tools import ToolRegistry
        
        print(f"🧪 Testing tool: {tool_name}")
        
        tools = ToolRegistry.list_tools()
        if tool_name not in tools:
            print(f"❌ Tool '{tool_name}' not found")
            print(f"Available tools: {list(tools.keys())}")
            return
        
        tool_info = tools[tool_name]
        tool_func = tool_info.get('function')
        
        if not tool_func:
            print(f"❌ Tool function not available")
            return
        
        # Try to call with no arguments first
        try:
            result = tool_func()
            print(f"✅ Tool test result: {result}")
        except TypeError as e:
            print(f"⚠️  Tool requires arguments: {e}")
            print(f"💡 Try calling with appropriate parameters")
        except Exception as e:
            print(f"❌ Tool execution failed: {e}")
            
    except Exception as e:
        print(f"❌ Error testing tool: {e}")

def list_tools():
    """List registered tools"""
    try:
        from CandyLLM.tools import ToolRegistry
        
        tools = ToolRegistry.list_tools()
        
        print("🔧 Registered Tools:")
        print("=" * 50)
        
        if not tools:
            print("📭 No tools registered yet")
            print("\n💡 Register tools using:")
            print("   from CandyLLM.tools import tool")
            print("   @tool")
            print("   def my_function(): pass")
            print("\n💡 Or register from CLI:")
            print("   candyllm tools register path/to/tools.py")
            return
        
        for name, info in tools.items():
            print(f"\n📦 {name}")
            print(f"   Description: {info.get('description', 'No description')}")
            print(f"   Function: {info.get('function', 'Unknown')}")
            print(f"   Cache hits: {info.get('cache_hits', 0)}")
            
    except Exception as e:
        print(f"❌ Error listing tools: {e}")

def manage_config(action: str, key: str = None, value: str = None):
    """Manage CandyLLM configuration"""
    try:
        from CandyLLM.config import ConfigManager
        
        config = ConfigManager()
        
        if action == 'show':
            print("⚙️  CandyLLM Configuration:")
            print("=" * 50)
            
            if key:
                # Show specific key
                val = config.get(key)
                if val is not None:
                    print(f"{key}: {val}")
                else:
                    print(f"❌ Key '{key}' not found")
            else:
                # Show all configuration
                all_config = config.get_all()
                if not all_config:
                    print("📭 No configuration set")
                    print("\n💡 Set configuration with:")
                    print("   candyllm config set openai_api_key <your-key>")
                else:
                    for k, v in all_config.items():
                        # Mask sensitive values
                        if 'key' in k.lower() or 'token' in k.lower():
                            masked = f"{v[:4]}...{v[-4:]}" if len(v) > 8 else "***"
                            print(f"   {k}: {masked}")
                        else:
                            print(f"   {k}: {v}")
        
        elif action == 'set':
            if not key or value is None:
                print("❌ Both key and value required for set operation")
                print("💡 Usage: candyllm config set <key> <value>")
                return
            
            config.set(key, value)
            print(f"✅ Set {key}")
            
        elif action == 'reset':
            if key:
                config.delete(key)
                print(f"✅ Reset {key}")
            else:
                config.reset_all()
                print("✅ Reset all configuration")
                
    except Exception as e:
        print(f"❌ Error managing configuration: {e}")

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        print("🍭 CandyLLM - Universal LLM Interface")
        print("\nQuick start:")
        print("   candyllm ui              # Launch legacy UI")
        print("   candyllm ui advanced     # Launch advanced UI")
        print("   candyllm models list     # List models")
        print("   candyllm --help          # Full help")
        return
    
    # Handle commands
    if args.command == 'ui':
        launch_ui(
            ui_type=args.ui_type,
            port=args.port,
            host=args.host,
            share=args.share,
            launch=not args.no_launch
        )
        
    elif args.command == 'models':
        if args.action == 'list':
            list_models()
        elif args.action == 'test':
            if not args.model_name:
                print("❌ Model name required for testing")
                sys.exit(1)
            test_model(args.model_name)
        elif args.action == 'info':
            if not args.model_name:
                print("❌ Model name required for info")
                sys.exit(1)
            get_model_info(args.model_name)
            
    elif args.command == 'tools':
        if args.action == 'list':
            list_tools()
        elif args.action == 'register':
            if not args.tool_name:
                print("❌ Tool file path required for registration")
                print("💡 Usage: candyllm tools register path/to/tools.py")
                sys.exit(1)
            register_tool_from_cli(args.tool_name)
        elif args.action == 'test':
            if not args.tool_name:
                print("❌ Tool name required for testing")
                sys.exit(1)
            test_tool(args.tool_name)
            
    elif args.command == 'config':
        manage_config(args.action, args.key, args.value)

if __name__ == '__main__':
    main()
