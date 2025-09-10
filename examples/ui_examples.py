"""
🍭 CandyLLM UI Examples
Complete demonstration of both legacy and advanced UI interfaces

This example shows how to launch different UI types and integrate
with the comprehensive 100+ model system.
"""

import sys
import os

# Add CandyLLM to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from CandyLLM import CandyLLM
from CandyLLM.ui import launch_ui, quick_ui

def demo_advanced_ui():
    """
    🔄 Advanced UI Demo
    
    Shows the advanced interface with modern features:
    - 100+ model support
    - Streaming responses
    - Auto-provider detection
    - Enhanced error handling
    """
    
    print("🍭 Launching Advanced UI...")
    print("✨ Features: 100+ models, streaming, auto-detection")
    print("🔗 Access at: http://localhost:7860")
    
    # Custom preprocessor to clean input
    def clean_input(text):
        return text.strip().replace("\\n", "\n")
    
    # Custom postprocessor for analytics
    def analyze_response(prompt, response):
        return f"Length: {len(response)} chars | Tokens: ~{len(response.split())}"
    
    # Launch with compatible signature
    CandyLLM.getUI(
        preprocessor_fn=clean_input,
        postprocessor_fn=analyze_response,
        selfOutput=True,
        selfOutputLabel="Response Analytics",
        selfOutputType="Text",
        launch=True
    )

def demo_modern_ui():
    """
    🚀 Modern UI Demo
    
    Shows the comprehensive modern interface with:
    - Tabbed interface (Chat, Config, Tools, Analytics)
    - Real-time streaming
    - Tool integration
    - Performance monitoring
    - Session management
    """
    
    print("🚀 Launching Modern CandyLLM UI...")
    print("🎯 Features: Enterprise dashboard, tools, analytics")
    print("🔗 Access at: http://localhost:7861")
    
    # Launch advanced UI
    CandyLLM.getAdvancedUI(
        launch=True,
        server_port=7861,
        share=True
    )

def demo_convenience_functions():
    """
    ⚡ Quick Launch Demos
    
    Shows convenience functions for rapid deployment
    """
    
    print("⚡ Quick UI Launcher Examples:")
    
    # Quick legacy UI
    print("\n1. Quick Legacy UI:")
    # quick_ui(ui_type="legacy", launch=False)
    
    # Quick advanced UI  
    print("2. Quick Advanced UI:")
    # quick_ui(ui_type="advanced", launch=False)
    
    # Generic launcher
    print("3. Generic Launcher:")
    # launch_ui(ui_type="auto", launch=False)

def demo_multi_ui_setup():
    """
    🔀 Multi-UI Setup
    
    Shows how to run multiple UI instances for different use cases
    """
    
    print("🔀 Multi-UI Setup Demo:")
    print("📊 Standard UI: Port 7860 (classic interface)")
    print("🚀 Advanced UI: Port 7861 (modern features)")
    
    # Note: In practice, you'd run these in separate processes
    print("\n💡 To run multiple UIs simultaneously:")
    print("   Process 1: python -c \"from CandyLLM import CandyLLM; CandyLLM.getUI(launch=True)\"")
    print("   Process 2: python -c \"from CandyLLM import CandyLLM; CandyLLM.getAdvancedUI(launch=True)\"")

def demo_programmatic_usage():
    """
    🔧 Programmatic UI Integration
    
    Shows how to integrate UI components into existing applications
    """
    
    print("🔧 Programmatic Integration Examples:")
    
    # Get UI without launching for custom integration
    interface = CandyLLM.getUI(launch=False)
    print(f"📱 Interface created: {type(interface)}")
    
    advanced_interface = CandyLLM.getAdvancedUI(launch=False)
    if advanced_interface:
        print(f"🚀 Advanced interface created: {type(advanced_interface)}")
    else:
        print("❌ Advanced UI dependencies not available")
    
    print("\n💼 Integration patterns:")
    print("   - Embed in Flask/FastAPI apps")
    print("   - Custom Gradio Blocks composition")
    print("   - Jupyter notebook integration")
    print("   - Desktop app embedding")

if __name__ == "__main__":
    print("🍭 CandyLLM UI Examples")
    print("=" * 50)
    
    # Show available demos
    demos = {
        "1": ("Advanced UI", demo_advanced_ui),
        "2": ("Modern UI", demo_modern_ui), 
        "3": ("Convenience Functions", demo_convenience_functions),
        "4": ("Multi-UI Setup", demo_multi_ui_setup),
        "5": ("Programmatic Usage", demo_programmatic_usage)
    }
    
    print("\nAvailable Demos:")
    for key, (name, _) in demos.items():
        print(f"   {key}. {name}")
    
    choice = input("\nSelect demo (1-5) or 'all' for overview: ").strip()
    
    if choice == "all":
        print("\n🎯 Running all demos (overview mode)...")
        for name, func in demos.values():
            print(f"\n{'='*20} {name} {'='*20}")
            if func == demo_advanced_ui or func == demo_modern_ui:
                print(f"Skipping actual launch for {name} (use individual demo)")
            else:
                func()
    elif choice in demos:
        name, func = demos[choice]
        print(f"\n🚀 Running: {name}")
        func()
    else:
        print("❌ Invalid choice. Please run again with 1-5 or 'all'")
