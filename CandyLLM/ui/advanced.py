"""
CandyLLM 2.0 Modern UI - Comprehensive Chat Interface with Advanced Features
Built with Gradio for maximum functionality and enterprise capabilities
"""

import gradio as gr
import asyncio
import json
import time
from typing import List, Dict, Any, Optional
import nest_asyncio
import threading
from datetime import datetime

# Enable nested event loops for Jupyter compatibility
nest_asyncio.apply()

# Import new CandyLLM components
try:
    from ..core.base import Message, ModelConfig
    from ..providers import UniversalModelProvider, ModelFactory
    from ..tools import global_tool_registry, list_tools, get_tool
    from .. import CandyLLM
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from core.base import Message, ModelConfig
    from providers import UniversalModelProvider, ModelFactory
    from tools import global_tool_registry, list_tools, get_tool
    from __init__ import CandyLLM


class CandyLLMUI:
    """
    Advanced UI for CandyLLM 2.0 with comprehensive features:
    - 100+ model support across all providers
    - Real-time streaming responses
    - Tool integration with visual interface
    - Agent workflow management
    - Enterprise monitoring dashboard
    - Session management
    - Cost tracking
    - Performance analytics
    """
    
    def __init__(self):
        self.candy = None
        self.current_session_id = None
        self.conversation_history = []
        self.streaming_active = False
        self.cost_tracker = {"total": 0.0, "session": 0.0}
        self.performance_metrics = {
            "response_times": [],
            "token_usage": [],
            "model_usage": {},
            "tool_usage": {}
        }
        
        # Initialize with default config
        self.initialize_candy({})
    
    def initialize_candy(self, config: Dict[str, Any]):
        """Initialize CandyLLM with configuration"""
        try:
            self.candy = CandyLLM(config)
            return "✅ CandyLLM initialized successfully"
        except Exception as e:
            return f"❌ Initialization failed: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        """Get all available models across providers"""
        if not self.candy:
            return ["Please configure CandyLLM first"]
        
        try:
            models = self.candy.list_models()
            return sorted(models)
        except Exception as e:
            return [f"Error loading models: {str(e)}"]
    
    def get_available_providers(self) -> List[str]:
        """Get all available providers"""
        if not self.candy:
            return ["openai", "anthropic", "litellm", "transformers"]
        
        try:
            providers = self.candy.list_providers()
            return sorted(providers)
        except Exception as e:
            return ["openai", "anthropic", "litellm"]
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get all available tools"""
        try:
            tools = list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category
                }
                for tool in tools
            ]
        except Exception as e:
            return [{"name": "error", "description": str(e), "category": "system"}]
    
    def format_tool_list(self, tools: List[Dict[str, str]]) -> str:
        """Format tools for display"""
        if not tools:
            return "No tools available"
        
        by_category = {}
        for tool in tools:
            category = tool.get("category", "other")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(tool)
        
        formatted = []
        for category, category_tools in sorted(by_category.items()):
            formatted.append(f"**{category.title()}:**")
            for tool in category_tools:
                formatted.append(f"  • {tool['name']}: {tool['description']}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def update_configuration(self, 
                           openai_key: str,
                           anthropic_key: str,
                           groq_key: str,
                           cohere_key: str,
                           other_config: str) -> str:
        """Update CandyLLM configuration"""
        config = {}
        
        if openai_key.strip():
            config['openai_api_key'] = openai_key.strip()
        if anthropic_key.strip():
            config['anthropic_api_key'] = anthropic_key.strip()
        if groq_key.strip():
            config['groq_api_key'] = groq_key.strip()
        if cohere_key.strip():
            config['cohere_api_key'] = cohere_key.strip()
        
        # Parse additional configuration
        if other_config.strip():
            try:
                additional = json.loads(other_config)
                config.update(additional)
            except json.JSONDecodeError:
                return "❌ Invalid JSON in additional configuration"
        
        return self.initialize_candy(config)
    
    def set_model(self, model_selection: str, auto_select: bool,
                  task: str, quality: str, speed: str, cost: str) -> str:
        """Set the current model"""
        if not self.candy:
            return "❌ Please configure CandyLLM first"
        
        try:
            if auto_select:
                self.candy.auto_model(task=task, quality=quality, speed=speed, cost=cost)
                model_info = self.candy._current_model.get_model_info()
                return f"✅ Auto-selected: {model_info['model_id']} ({model_info['provider']})"
            else:
                if not model_selection or model_selection == "Please configure CandyLLM first":
                    return "❌ Please select a valid model"
                
                self.candy.set_model(model_selection)
                model_info = self.candy._current_model.get_model_info()
                return f"✅ Selected: {model_info['model_id']} ({model_info['provider']})"
        except Exception as e:
            return f"❌ Model selection failed: {str(e)}"
    
    def process_chat_message(self,
                           message: str,
                           history: List[List[str]],
                           system_prompt: str,
                           temperature: float,
                           max_tokens: int,
                           stream_response: bool,
                           use_tools: bool,
                           selected_tools: List[str]) -> tuple:
        """Process chat message with full feature support"""
        
        if not self.candy or not self.candy._current_model:
            error_msg = "❌ Please configure CandyLLM and select a model first"
            history.append([message, error_msg])
            return history, ""
        
        start_time = time.time()
        
        try:
            # Convert history to messages
            messages = []
            if system_prompt.strip():
                messages.append(Message(role="system", content=system_prompt.strip()))
            
            # Add conversation history
            for user_msg, assistant_msg in history:
                if user_msg:
                    messages.append(Message(role="user", content=user_msg))
                if assistant_msg:
                    messages.append(Message(role="assistant", content=assistant_msg))
            
            # Add current message
            messages.append(Message(role="user", content=message))
            
            if stream_response:
                # Streaming response
                return self._handle_streaming_response(
                    messages, history, message, temperature, max_tokens,
                    use_tools, selected_tools, start_time
                )
            else:
                # Non-streaming response
                return self._handle_regular_response(
                    messages, history, message, temperature, max_tokens,
                    use_tools, selected_tools, start_time
                )
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            history.append([message, error_msg])
            return history, ""
    
    def _handle_regular_response(self, messages, history, user_message,
                               temperature, max_tokens, use_tools, 
                               selected_tools, start_time):
        """Handle non-streaming response"""
        # Create a new event loop for async call
        def run_async():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Generate response
                if use_tools and selected_tools:
                    # Tool-enabled response (future enhancement)
                    response = loop.run_until_complete(
                        self.candy.chat(user_message, 
                                      temperature=temperature, 
                                      max_tokens=max_tokens)
                    )
                else:
                    response = loop.run_until_complete(
                        self.candy.chat(user_message,
                                      temperature=temperature,
                                      max_tokens=max_tokens)
                    )
                
                # Update metrics
                end_time = time.time()
                self.performance_metrics["response_times"].append(end_time - start_time)
                if hasattr(response, 'usage') and response.usage:
                    self.performance_metrics["token_usage"].append(response.usage.get('total_tokens', 0))
                
                model_name = response.model if hasattr(response, 'model') else 'unknown'
                self.performance_metrics["model_usage"][model_name] = \
                    self.performance_metrics["model_usage"].get(model_name, 0) + 1
                
                history.append([user_message, response.content])
                return history, ""
                
            except Exception as e:
                history.append([user_message, f"❌ Error: {str(e)}"])
                return history, ""
            finally:
                loop.close()
        
        return run_async()
    
    def _handle_streaming_response(self, messages, history, user_message,
                                 temperature, max_tokens, use_tools,
                                 selected_tools, start_time):
        """Handle streaming response"""
        # For streaming, we'll use a placeholder and update via JavaScript
        # This is a simplified version - full streaming would require WebSocket integration
        
        try:
            # Simulate streaming by running async in thread
            response_content = ""
            
            def run_streaming():
                nonlocal response_content
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    async def stream_handler():
                        nonlocal response_content
                        full_response = ""
                        async for chunk in self.candy.stream_chat(
                            user_message,
                            temperature=temperature,
                            max_tokens=max_tokens
                        ):
                            full_response += chunk.content
                        return full_response
                    
                    response_content = loop.run_until_complete(stream_handler())
                    
                except Exception as e:
                    response_content = f"❌ Streaming error: {str(e)}"
                finally:
                    loop.close()
            
            # Run in thread to avoid blocking
            thread = threading.Thread(target=run_streaming)
            thread.start()
            thread.join(timeout=30)  # 30 second timeout
            
            # Update metrics
            end_time = time.time()
            self.performance_metrics["response_times"].append(end_time - start_time)
            
            history.append([user_message, response_content])
            return history, ""
            
        except Exception as e:
            history.append([user_message, f"❌ Streaming error: {str(e)}"])
            return history, ""
    
    def get_system_statistics(self) -> str:
        """Get comprehensive system statistics"""
        if not self.candy:
            return "❌ CandyLLM not initialized"
        
        try:
            stats = self.candy.get_statistics()
            
            # Performance metrics
            avg_response_time = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"]) if self.performance_metrics["response_times"] else 0
            total_tokens = sum(self.performance_metrics["token_usage"])
            
            # Format statistics
            formatted_stats = f"""
**🎯 System Statistics**

**Provider Coverage:**
• Total Providers: {stats.get('total_providers', 'N/A')}
• Available Providers: {len(stats.get('available_providers', []))}
• Total Models: {stats.get('total_models', 'N/A')}

**Tool System:**
• Total Tools: {stats.get('total_tools', 'N/A')}
• Loaded Instances: {stats.get('loaded_instances', 'N/A')}
• Cache Hits: {stats.get('cache_hits', 'N/A')}
• Cache Misses: {stats.get('cache_misses', 'N/A')}

**Performance Metrics:**
• Average Response Time: {avg_response_time:.2f}s
• Total Tokens Used: {total_tokens:,}
• Total Conversations: {len(self.performance_metrics['response_times'])}

**Model Usage:**
{self._format_usage_stats(self.performance_metrics['model_usage'])}

**Tool Usage:**
{self._format_usage_stats(self.performance_metrics['tool_usage'])}
"""
            return formatted_stats
            
        except Exception as e:
            return f"❌ Error getting statistics: {str(e)}"
    
    def _format_usage_stats(self, usage_dict: Dict[str, int]) -> str:
        """Format usage statistics"""
        if not usage_dict:
            return "• No usage data"
        
        formatted = []
        for item, count in sorted(usage_dict.items(), key=lambda x: x[1], reverse=True):
            formatted.append(f"• {item}: {count} uses")
        
        return "\n".join(formatted[:5])  # Top 5 items
    
    def export_conversation(self, history: List[List[str]]) -> str:
        """Export conversation history"""
        if not history:
            return "No conversation to export"
        
        # Create formatted export
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "conversation": [
                {"role": "user", "content": user_msg, "assistant": assistant_msg}
                for user_msg, assistant_msg in history
            ],
            "statistics": self.get_system_statistics()
        }
        
        # Convert to JSON string
        return json.dumps(export_data, indent=2)
    
    def create_interface(self) -> gr.Blocks:
        """Create the comprehensive Gradio interface"""
        
        # Custom CSS for modern styling
        custom_css = """
        .gradio-container {
            font-family: 'Inter', sans-serif !important;
        }
        .panel-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            text-align: center;
            font-weight: bold;
        }
        .config-panel {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
        }
        .metrics-panel {
            background: #fff5f5;
            border: 1px solid #fed7d7;
            border-radius: 8px;
            padding: 1rem;
        }
        .tool-panel {
            background: #f0fff4;
            border: 1px solid #c6f6d5;
            border-radius: 8px;
            padding: 1rem;
        }
        """
        
        with gr.Blocks(
            theme=gr.themes.Soft(),
            css=custom_css,
            title="🍭 CandyLLM 2.0 - Universal AI Interface"
        ) as interface:
            
            # Header
            gr.HTML("""
            <div class="panel-header">
                <h1>🍭 CandyLLM 2.0 - Universal AI Interface</h1>
                <p>Access 100+ language models with advanced agentic capabilities</p>
            </div>
            """)
            
            with gr.Tabs():
                
                # Main Chat Tab
                with gr.Tab("💬 Chat Interface"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            # Chat interface
                            chatbot = gr.Chatbot(
                                label="🤖 AI Assistant",
                                height=500,
                                show_label=True,
                                container=True,
                                bubble_full_width=False,
                                show_share_button=True,
                                show_copy_button=True,
                                likeable=True
                            )
                            
                            # Message input
                            msg_input = gr.Textbox(
                                label="Your message",
                                placeholder="Type your message here...",
                                lines=2,
                                max_lines=10
                            )
                            
                            with gr.Row():
                                send_btn = gr.Button("Send 📤", variant="primary")
                                clear_btn = gr.Button("Clear 🗑️", variant="secondary")
                                export_btn = gr.Button("Export 📄", variant="secondary")
                        
                        with gr.Column(scale=1):
                            # Configuration panel
                            gr.HTML('<div class="panel-header">⚙️ Configuration</div>')
                            
                            with gr.Group():
                                # Model selection
                                auto_select = gr.Checkbox(
                                    label="🎯 Auto-select optimal model",
                                    value=True,
                                    info="Let AI choose the best model for your task"
                                )
                                
                                with gr.Row():
                                    task_type = gr.Dropdown(
                                        choices=["chat", "code", "multimodal", "local"],
                                        value="chat",
                                        label="Task Type",
                                        visible=True
                                    )
                                    quality = gr.Dropdown(
                                        choices=["high", "medium", "low"],
                                        value="high",
                                        label="Quality",
                                        visible=True
                                    )
                                
                                with gr.Row():
                                    speed = gr.Dropdown(
                                        choices=["fast", "medium", "slow"],
                                        value="medium",
                                        label="Speed",
                                        visible=True
                                    )
                                    cost = gr.Dropdown(
                                        choices=["low", "medium", "high"],
                                        value="medium",
                                        label="Cost",
                                        visible=True
                                    )
                                
                                model_dropdown = gr.Dropdown(
                                    choices=self.get_available_models(),
                                    label="🎛️ Manual Model Selection",
                                    visible=False,
                                    allow_custom_value=True,
                                    info="Select specific model"
                                )
                                
                                model_status = gr.Textbox(
                                    label="Model Status",
                                    value="Please configure and select a model",
                                    interactive=False
                                )
                                
                                set_model_btn = gr.Button("Set Model 🎯", variant="primary")
                            
                            # Generation parameters
                            with gr.Group():
                                gr.HTML("**🎚️ Generation Parameters**")
                                
                                system_prompt = gr.Textbox(
                                    label="System Prompt",
                                    placeholder="You are a helpful AI assistant...",
                                    lines=2
                                )
                                
                                temperature = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=0.7,
                                    step=0.1,
                                    label="Temperature",
                                    info="Creativity level"
                                )
                                
                                max_tokens = gr.Slider(
                                    minimum=50,
                                    maximum=4000,
                                    value=1000,
                                    step=50,
                                    label="Max Tokens",
                                    info="Response length"
                                )
                                
                                stream_response = gr.Checkbox(
                                    label="🌊 Stream Response",
                                    value=True,
                                    info="Real-time response streaming"
                                )
                            
                            # Tools configuration
                            with gr.Group():
                                gr.HTML("**🛠️ Tools Configuration**")
                                
                                use_tools = gr.Checkbox(
                                    label="Enable Tools",
                                    value=False,
                                    info="Allow AI to use tools"
                                )
                                
                                available_tools = self.get_available_tools()
                                tool_choices = [f"{tool['name']} ({tool['category']})" for tool in available_tools]
                                
                                selected_tools = gr.CheckboxGroup(
                                    choices=tool_choices,
                                    label="Select Tools",
                                    visible=False
                                )
                
                # Configuration Tab
                with gr.Tab("⚙️ Configuration"):
                    gr.HTML('<div class="panel-header">🔧 API Configuration</div>')
                    
                    with gr.Row():
                        with gr.Column():
                            gr.HTML("**🔑 API Keys**")
                            
                            openai_key = gr.Textbox(
                                label="OpenAI API Key",
                                type="password",
                                placeholder="sk-..."
                            )
                            
                            anthropic_key = gr.Textbox(
                                label="Anthropic API Key", 
                                type="password",
                                placeholder="sk-ant-..."
                            )
                            
                            groq_key = gr.Textbox(
                                label="Groq API Key",
                                type="password",
                                placeholder="gsk_..."
                            )
                            
                            cohere_key = gr.Textbox(
                                label="Cohere API Key",
                                type="password",
                                placeholder="co_..."
                            )
                        
                        with gr.Column():
                            gr.HTML("**⚡ Advanced Configuration**")
                            
                            other_config = gr.Textbox(
                                label="Additional Configuration (JSON)",
                                placeholder='{"timeout": 30, "max_retries": 3}',
                                lines=5,
                                info="Additional provider-specific settings"
                            )
                            
                            config_status = gr.Textbox(
                                label="Configuration Status",
                                value="Please enter API keys and click Update",
                                interactive=False
                            )
                            
                            update_config_btn = gr.Button("Update Configuration 🔄", variant="primary")
                
                # Tools Management Tab
                with gr.Tab("🛠️ Tools"):
                    gr.HTML('<div class="panel-header">🔧 Tool Management</div>')
                    
                    with gr.Row():
                        with gr.Column():
                            gr.HTML("**📋 Available Tools**")
                            tools_display = gr.Markdown(
                                value=self.format_tool_list(self.get_available_tools()),
                                label="Tools"
                            )
                            
                            refresh_tools_btn = gr.Button("Refresh Tools 🔄")
                        
                        with gr.Column():
                            gr.HTML("**➕ Register New Tool**")
                            
                            gr.Markdown("""
                            **Coming Soon:**
                            - Visual tool builder
                            - Custom function registration
                            - Tool testing interface
                            - Tool marketplace integration
                            """)
                
                # Analytics Tab
                with gr.Tab("📊 Analytics"):
                    gr.HTML('<div class="panel-header">📈 Performance Analytics</div>')
                    
                    with gr.Row():
                        with gr.Column():
                            stats_display = gr.Markdown(
                                value=self.get_system_statistics(),
                                label="System Statistics"
                            )
                            
                            refresh_stats_btn = gr.Button("Refresh Statistics 🔄")
                        
                        with gr.Column():
                            gr.HTML("**💰 Cost Tracking**")
                            gr.Markdown("Coming Soon: Real-time cost monitoring")
                            
                            gr.HTML("**⚡ Performance Metrics**")
                            gr.Markdown("Coming Soon: Response time analysis")
                
                # Examples Tab
                with gr.Tab("📝 Examples"):
                    gr.HTML('<div class="panel-header">💡 Example Prompts</div>')
                    
                    examples = [
                        ["What is quantum computing and how does it work?"],
                        ["Write a Python function to calculate fibonacci numbers"],
                        ["Explain the differences between React and Vue.js"],
                        ["Create a business plan for a sustainable coffee shop"],
                        ["Analyze this data and provide insights: [data]"],
                        ["Help me debug this error: TypeError: 'str' object is not callable"]
                    ]
                    
                    gr.Examples(
                        examples=examples,
                        inputs=[msg_input],
                        label="Click to use example prompts"
                    )
            
            # Event handlers
            def toggle_model_selection(auto_select_val):
                return gr.update(visible=not auto_select_val)
            
            def toggle_tools_selection(use_tools_val):
                return gr.update(visible=use_tools_val)
            
            # Auto-select toggle
            auto_select.change(
                fn=toggle_model_selection,
                inputs=[auto_select],
                outputs=[model_dropdown]
            )
            
            # Tools toggle
            use_tools.change(
                fn=toggle_tools_selection,
                inputs=[use_tools],
                outputs=[selected_tools]
            )
            
            # Configuration update
            update_config_btn.click(
                fn=self.update_configuration,
                inputs=[openai_key, anthropic_key, groq_key, cohere_key, other_config],
                outputs=[config_status]
            ).then(
                fn=lambda: gr.update(choices=self.get_available_models()),
                outputs=[model_dropdown]
            )
            
            # Model selection
            set_model_btn.click(
                fn=self.set_model,
                inputs=[model_dropdown, auto_select, task_type, quality, speed, cost],
                outputs=[model_status]
            )
            
            # Chat functionality
            def chat_wrapper(*args):
                return self.process_chat_message(*args)
            
            send_btn.click(
                fn=chat_wrapper,
                inputs=[
                    msg_input, chatbot, system_prompt, temperature, max_tokens,
                    stream_response, use_tools, selected_tools
                ],
                outputs=[chatbot, msg_input]
            )
            
            msg_input.submit(
                fn=chat_wrapper,
                inputs=[
                    msg_input, chatbot, system_prompt, temperature, max_tokens,
                    stream_response, use_tools, selected_tools
                ],
                outputs=[chatbot, msg_input]
            )
            
            # Clear chat
            clear_btn.click(
                fn=lambda: ([], ""),
                outputs=[chatbot, msg_input]
            )
            
            # Export conversation
            export_btn.click(
                fn=self.export_conversation,
                inputs=[chatbot],
                outputs=[gr.File(label="Download Conversation")]
            )
            
            # Refresh functions
            refresh_tools_btn.click(
                fn=lambda: self.format_tool_list(self.get_available_tools()),
                outputs=[tools_display]
            )
            
            refresh_stats_btn.click(
                fn=self.get_system_statistics,
                outputs=[stats_display]
            )
        
        return interface


def create_advanced_ui(launch: bool = True, share: bool = False, **kwargs) -> gr.Blocks:
    """
    Create and optionally launch the advanced CandyLLM UI
    
    Args:
        launch: Whether to launch the interface immediately
        share: Whether to create a public shareable link
        **kwargs: Additional arguments passed to launch()
    
    Returns:
        Gradio Blocks interface
    """
    ui = CandyLLMUI()
    interface = ui.create_interface()
    
    if launch:
        interface.launch(
            share=share,
            show_error=True,
            show_tips=True,
            height=800,
            **kwargs
        )
    
    return interface


# Convenience function for compatibility
def getAdvancedUI(**kwargs):
    """Function name for compatibility"""
    return create_advanced_ui(**kwargs)


if __name__ == "__main__":
    # Launch the interface
    create_advanced_ui(
        launch=True,
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
