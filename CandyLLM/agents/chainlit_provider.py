"""
Chainlit Agent Provider

Integrates Chainlit's conversational AI interface with chat UI, async support,
and integration capabilities for building interactive AI applications.
"""

import uuid
import asyncio
import json
import os
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
    from openai import OpenAI, AsyncOpenAI
    import websockets
    import aiofiles
    from fastapi import FastAPI, WebSocket
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    import uvicorn
    CHAINLIT_AVAILABLE = True
except ImportError:
    CHAINLIT_AVAILABLE = False
    # Mock classes for when dependencies are not available
    httpx = None
    requests = None
    OpenAI = None
    AsyncOpenAI = None
    websockets = None
    aiofiles = None
    FastAPI = None
    WebSocket = None
    StaticFiles = None
    HTMLResponse = None
    uvicorn = None


class MessageType(Enum):
    """Types of messages in Chainlit"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    ERROR = "error"
    THINKING = "thinking"
    STREAMING = "streaming"
    COMPLETED = "completed"


class SessionStatus(Enum):
    """Status of chat sessions"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ENDED = "ended"
    ERROR = "error"
    TIMEOUT = "timeout"


class UIComponent(Enum):
    """Types of UI components"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    FILE = "file"
    BUTTON = "button"
    INPUT = "input"
    CHART = "chart"
    TABLE = "table"
    CODE = "code"
    MARKDOWN = "markdown"


@dataclass
class ChainlitConfig:
    """Configuration for Chainlit agent"""
    api_key: str = ""
    model: str = "gpt-4"
    base_url: str = "https://api.openai.com/v1"
    max_tokens: int = 2000
    temperature: float = 0.7
    host: str = "localhost"
    port: int = 8000
    title: str = "Chainlit AI Assistant"
    description: str = "Interactive AI Assistant powered by Chainlit"
    favicon: str = "🤖"
    theme: str = "light"  # light, dark, auto
    enable_streaming: bool = True
    enable_file_upload: bool = True
    enable_image_display: bool = True
    enable_audio_playback: bool = True
    enable_video_playback: bool = True
    max_file_size_mb: int = 100
    session_timeout: int = 3600  # 1 hour
    max_concurrent_sessions: int = 100
    enable_chat_history: bool = True
    enable_user_feedback: bool = True
    enable_authentication: bool = False
    custom_css: str = ""
    custom_js: str = ""
    websocket_enabled: bool = True
    rate_limit_per_minute: int = 60
    enable_markdown: bool = True
    enable_latex: bool = True
    enable_syntax_highlighting: bool = True


@dataclass
class Message:
    """Message in Chainlit conversation"""
    message_id: str
    session_id: str
    message_type: MessageType
    content: str
    author: str = "user"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ui_components: List[Dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    response_time: float = 0.0
    feedback: Optional[Dict[str, Any]] = None


@dataclass
class Session:
    """Chat session in Chainlit"""
    session_id: str
    user_id: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    messages: List[Message] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    total_messages: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    websocket: Optional[Any] = None


@dataclass
class UIElement:
    """UI element for display"""
    element_id: str
    component_type: UIComponent
    content: Any
    title: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    interactive: bool = False
    callback: Optional[str] = None


class ChainlitAgent:
    """Chainlit conversational AI interface with chat UI and async support"""
    
    def __init__(self, agent_id: str, config: ChainlitConfig,
                 security_manager: Optional[AgentSecurityManager] = None):
        self.agent_id = agent_id
        self.config = config
        self.security_manager = security_manager
        self._client = None
        self._async_client = None
        self._app = None
        self._server_task = None
        self._sessions = {}  # session_id -> Session
        self._active_websockets = {}  # session_id -> WebSocket
        self._message_handlers = {}  # message_type -> handler_function
        self._ui_callbacks = {}  # callback_id -> callback_function
        self._running = False
        self._usage_stats = {
            'total_sessions': 0,
            'total_messages': 0,
            'total_tokens_used': 0,
            'average_session_duration': 0.0,
            'average_response_time': 0.0,
            'user_satisfaction': 0.0,
            'active_sessions': 0,
            'messages_per_session': 0.0,
            'file_uploads': 0,
            'ui_interactions': 0,
            'streaming_sessions': 0,
            'websocket_connections': 0,
            'authentication_attempts': 0,
            'error_rate': 0.0
        }
        self._created_at = datetime.now()
        
    async def initialize(self) -> bool:
        """Initialize the Chainlit agent"""
        if not CHAINLIT_AVAILABLE:
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
            
            # Initialize FastAPI app
            await self._initialize_app()
            
            # Setup message handlers
            self._setup_message_handlers()
            
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
            raise RuntimeError(f"Chainlit API test failed: {e}")
    
    async def _initialize_app(self):
        """Initialize FastAPI application"""
        try:
            self._app = FastAPI(
                title=self.config.title,
                description=self.config.description
            )
            
            # Add static files if needed
            # self._app.mount("/static", StaticFiles(directory="static"), name="static")
            
            # Setup routes
            await self._setup_routes()
            
        except Exception as e:
            pass  # Non-critical initialization
    
    async def _setup_routes(self):
        """Setup FastAPI routes"""
        try:
            @self._app.get("/", response_class=HTMLResponse)
            async def get_chat_interface():
                return await self._generate_chat_html()
            
            @self._app.websocket("/ws/{session_id}")
            async def websocket_endpoint(websocket: WebSocket, session_id: str):
                await self._handle_websocket(websocket, session_id)
            
            @self._app.post("/api/message/{session_id}")
            async def send_message(session_id: str, message_data: dict):
                return await self._handle_api_message(session_id, message_data)
            
            @self._app.get("/api/session/{session_id}")
            async def get_session(session_id: str):
                return await self._get_session_info(session_id)
            
            @self._app.post("/api/feedback")
            async def submit_feedback(feedback_data: dict):
                return await self._handle_feedback(feedback_data)
            
        except Exception as e:
            pass
    
    def _setup_message_handlers(self):
        """Setup message type handlers"""
        self._message_handlers = {
            MessageType.USER: self._handle_user_message,
            MessageType.SYSTEM: self._handle_system_message,
            MessageType.FUNCTION: self._handle_function_message
        }
    
    async def _generate_chat_html(self) -> str:
        """Generate chat interface HTML"""
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.config.title}</title>
            <link rel="icon" href="data:text/plain;charset=utf-8;base64,{self.config.favicon}" />
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: {'#f5f5f5' if self.config.theme == 'light' else '#1a1a1a'};
                    color: {'#333' if self.config.theme == 'light' else '#fff'};
                }}
                .chat-container {{
                    max-width: 800px;
                    margin: 0 auto;
                    background: {'white' if self.config.theme == 'light' else '#2d2d2d'};
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    overflow: hidden;
                }}
                .chat-header {{
                    padding: 20px;
                    background: {'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' if self.config.theme == 'light' else 'linear-gradient(135deg, #434343 0%, #000000 100%)'};
                    color: white;
                    text-align: center;
                }}
                .chat-messages {{
                    height: 400px;
                    overflow-y: auto;
                    padding: 20px;
                }}
                .message {{
                    margin-bottom: 15px;
                    padding: 10px 15px;
                    border-radius: 18px;
                    max-width: 70%;
                }}
                .user-message {{
                    background: #007AFF;
                    color: white;
                    margin-left: auto;
                }}
                .assistant-message {{
                    background: {'#f1f1f1' if self.config.theme == 'light' else '#3a3a3a'};
                    color: {'#333' if self.config.theme == 'light' else '#fff'};
                }}
                .chat-input {{
                    display: flex;
                    padding: 20px;
                    border-top: 1px solid {'#eee' if self.config.theme == 'light' else '#444'};
                }}
                .input-field {{
                    flex: 1;
                    padding: 10px 15px;
                    border: 1px solid {'#ddd' if self.config.theme == 'light' else '#555'};
                    border-radius: 25px;
                    outline: none;
                    background: {'white' if self.config.theme == 'light' else '#2d2d2d'};
                    color: {'#333' if self.config.theme == 'light' else '#fff'};
                }}
                .send-button {{
                    padding: 10px 20px;
                    margin-left: 10px;
                    background: #007AFF;
                    color: white;
                    border: none;
                    border-radius: 25px;
                    cursor: pointer;
                }}
                .send-button:hover {{
                    background: #0056CC;
                }}
                .typing-indicator {{
                    font-style: italic;
                    color: #666;
                    padding: 10px 15px;
                }}
                {self.config.custom_css}
            </style>
        </head>
        <body>
            <div class="chat-container">
                <div class="chat-header">
                    <h1>{self.config.title}</h1>
                    <p>{self.config.description}</p>
                </div>
                <div class="chat-messages" id="chatMessages">
                    <div class="message assistant-message">
                        Hello! I'm your AI assistant. How can I help you today?
                    </div>
                </div>
                <div class="chat-input">
                    <input type="text" class="input-field" id="messageInput" placeholder="Type your message..." />
                    <button class="send-button" onclick="sendMessage()">Send</button>
                </div>
            </div>
            
            <script>
                const sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
                const websocket = new WebSocket(`ws://localhost:{self.config.port}/ws/${{sessionId}}`);
                
                websocket.onmessage = function(event) {{
                    const data = JSON.parse(event.data);
                    displayMessage(data.content, 'assistant');
                }};
                
                function sendMessage() {{
                    const input = document.getElementById('messageInput');
                    const message = input.value.trim();
                    if (message) {{
                        displayMessage(message, 'user');
                        websocket.send(JSON.stringify({{
                            type: 'user_message',
                            content: message,
                            session_id: sessionId
                        }}));
                        input.value = '';
                    }}
                }}
                
                function displayMessage(content, type) {{
                    const messagesDiv = document.getElementById('chatMessages');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${{type}}-message`;
                    messageDiv.textContent = content;
                    messagesDiv.appendChild(messageDiv);
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                }}
                
                document.getElementById('messageInput').addEventListener('keypress', function(e) {{
                    if (e.key === 'Enter') {{
                        sendMessage();
                    }}
                }});
                
                {self.config.custom_js}
            </script>
        </body>
        </html>
        """
        return html_template
    
    async def _handle_websocket(self, websocket: WebSocket, session_id: str):
        """Handle WebSocket connection"""
        try:
            await websocket.accept()
            
            # Create or get session
            if session_id not in self._sessions:
                await self._create_session(session_id)
            
            session = self._sessions[session_id]
            session.websocket = websocket
            self._active_websockets[session_id] = websocket
            
            self._usage_stats['websocket_connections'] += 1
            self._usage_stats['active_sessions'] = len(self._active_websockets)
            
            try:
                while True:
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    if message_data.get('type') == 'user_message':
                        await self._process_websocket_message(session_id, message_data)
                    
            except Exception as e:
                pass
            finally:
                # Clean up
                if session_id in self._active_websockets:
                    del self._active_websockets[session_id]
                self._usage_stats['active_sessions'] = len(self._active_websockets)
                
        except Exception as e:
            pass
    
    async def _process_websocket_message(self, session_id: str, message_data: dict):
        """Process WebSocket message"""
        try:
            content = message_data.get('content', '')
            session = self._sessions[session_id]
            
            # Create user message
            user_message = Message(
                message_id=f"msg_{uuid.uuid4().hex[:8]}",
                session_id=session_id,
                message_type=MessageType.USER,
                content=content,
                author="user"
            )
            
            session.messages.append(user_message)
            session.total_messages += 1
            self._usage_stats['total_messages'] += 1
            
            # Generate response
            if self.config.enable_streaming:
                await self._stream_response(session_id, content)
            else:
                response = await self._generate_response(content, session_id)
                await self._send_websocket_response(session_id, response)
            
        except Exception as e:
            await self._send_websocket_error(session_id, str(e))
    
    async def _stream_response(self, session_id: str, user_input: str):
        """Stream response to WebSocket"""
        try:
            websocket = self._active_websockets.get(session_id)
            if not websocket:
                return
            
            # Stream response
            response_chunks = []
            
            async for chunk in self._generate_streaming_response(user_input, session_id):
                response_chunks.append(chunk)
                
                # Send chunk via WebSocket
                await websocket.send_text(json.dumps({
                    'type': 'streaming',
                    'content': chunk,
                    'session_id': session_id
                }))
            
            # Send completion
            final_response = ''.join(response_chunks)
            await websocket.send_text(json.dumps({
                'type': 'completed',
                'content': final_response,
                'session_id': session_id
            }))
            
            # Store complete message
            assistant_message = Message(
                message_id=f"msg_{uuid.uuid4().hex[:8]}",
                session_id=session_id,
                message_type=MessageType.ASSISTANT,
                content=final_response,
                author="assistant"
            )
            
            session = self._sessions[session_id]
            session.messages.append(assistant_message)
            
            self._usage_stats['streaming_sessions'] += 1
            
        except Exception as e:
            await self._send_websocket_error(session_id, str(e))
    
    async def _generate_streaming_response(self, user_input: str, session_id: str):
        """Generate streaming response"""
        try:
            session = self._sessions[session_id]
            
            # Build conversation history
            messages = []
            for msg in session.messages[-10:]:  # Last 10 messages
                role = "user" if msg.message_type == MessageType.USER else "assistant"
                messages.append({"role": role, "content": msg.content})
            
            messages.append({"role": "user", "content": user_input})
            
            # Stream response
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
        except Exception as e:
            yield f"Error generating response: {str(e)}"
    
    async def _send_websocket_response(self, session_id: str, content: str):
        """Send response via WebSocket"""
        try:
            websocket = self._active_websockets.get(session_id)
            if websocket:
                await websocket.send_text(json.dumps({
                    'type': 'assistant_message',
                    'content': content,
                    'session_id': session_id
                }))
            
        except Exception as e:
            pass
    
    async def _send_websocket_error(self, session_id: str, error: str):
        """Send error via WebSocket"""
        try:
            websocket = self._active_websockets.get(session_id)
            if websocket:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'content': f"Error: {error}",
                    'session_id': session_id
                }))
            
        except Exception as e:
            pass
    
    async def _create_session(self, session_id: str, user_id: Optional[str] = None) -> Session:
        """Create a new chat session"""
        try:
            session = Session(
                session_id=session_id,
                user_id=user_id
            )
            
            self._sessions[session_id] = session
            self._usage_stats['total_sessions'] += 1
            
            return session
            
        except Exception as e:
            raise RuntimeError(f"Failed to create session: {e}")
    
    async def _generate_response(self, user_input: str, session_id: str) -> str:
        """Generate response for user input"""
        try:
            session = self._sessions[session_id]
            start_time = datetime.now()
            
            # Build conversation history
            messages = []
            for msg in session.messages[-10:]:  # Last 10 messages
                role = "user" if msg.message_type == MessageType.USER else "assistant"
                messages.append({"role": role, "content": msg.content})
            
            messages.append({"role": "user", "content": user_input})
            
            # Generate response
            response = await self._async_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 100
            
            # Create assistant message
            assistant_message = Message(
                message_id=f"msg_{uuid.uuid4().hex[:8]}",
                session_id=session_id,
                message_type=MessageType.ASSISTANT,
                content=content,
                author="assistant",
                tokens_used=tokens_used,
                response_time=(datetime.now() - start_time).total_seconds()
            )
            
            session.messages.append(assistant_message)
            session.total_tokens += tokens_used
            session.last_activity = datetime.now()
            
            # Update statistics
            self._usage_stats['total_tokens_used'] += tokens_used
            self._usage_stats['average_response_time'] = (
                (self._usage_stats['average_response_time'] * (self._usage_stats['total_messages'] - 1) +
                 assistant_message.response_time) / self._usage_stats['total_messages']
            )
            
            return content
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
    
    async def _handle_user_message(self, message: Message) -> str:
        """Handle user message"""
        return await self._generate_response(message.content, message.session_id)
    
    async def _handle_system_message(self, message: Message) -> str:
        """Handle system message"""
        return "System message processed"
    
    async def _handle_function_message(self, message: Message) -> str:
        """Handle function message"""
        return "Function call processed"
    
    async def _handle_api_message(self, session_id: str, message_data: dict) -> dict:
        """Handle API message"""
        try:
            if session_id not in self._sessions:
                await self._create_session(session_id)
            
            content = message_data.get('content', '')
            response = await self._generate_response(content, session_id)
            
            return {
                'session_id': session_id,
                'response': response,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _get_session_info(self, session_id: str) -> dict:
        """Get session information"""
        try:
            if session_id not in self._sessions:
                return {'error': 'Session not found'}
            
            session = self._sessions[session_id]
            
            return {
                'session_id': session_id,
                'status': session.status.value,
                'message_count': len(session.messages),
                'total_tokens': session.total_tokens,
                'start_time': session.start_time.isoformat(),
                'last_activity': session.last_activity.isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _handle_feedback(self, feedback_data: dict) -> dict:
        """Handle user feedback"""
        try:
            session_id = feedback_data.get('session_id')
            message_id = feedback_data.get('message_id')
            rating = feedback_data.get('rating', 0)
            comment = feedback_data.get('comment', '')
            
            # Store feedback
            if session_id in self._sessions:
                session = self._sessions[session_id]
                for message in session.messages:
                    if message.message_id == message_id:
                        message.feedback = {
                            'rating': rating,
                            'comment': comment,
                            'timestamp': datetime.now().isoformat()
                        }
                        break
            
            # Update satisfaction score
            if rating > 0:
                current_avg = self._usage_stats['user_satisfaction']
                total_feedback = self._usage_stats.get('total_feedback', 0) + 1
                self._usage_stats['user_satisfaction'] = (
                    (current_avg * (total_feedback - 1) + rating) / total_feedback
                )
                self._usage_stats['total_feedback'] = total_feedback
            
            return {'status': 'feedback_received'}
            
        except Exception as e:
            return {'error': str(e)}
    
    async def start_server(self) -> bool:
        """Start the Chainlit server"""
        try:
            if self._running:
                return True
            
            if not self._app:
                await self._initialize_app()
            
            # Start server
            config = uvicorn.Config(
                app=self._app,
                host=self.config.host,
                port=self.config.port,
                log_level="info"
            )
            
            server = uvicorn.Server(config)
            self._server_task = asyncio.create_task(server.serve())
            self._running = True
            
            print(f"Chainlit server started at http://{self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            return False
    
    async def stop_server(self):
        """Stop the Chainlit server"""
        try:
            if self._server_task:
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    pass
            
            self._running = False
            
        except Exception as e:
            pass
    
    async def send_message_to_session(self, session_id: str, content: str, 
                                    message_type: MessageType = MessageType.ASSISTANT) -> bool:
        """Send message to a specific session"""
        try:
            if session_id not in self._sessions:
                return False
            
            websocket = self._active_websockets.get(session_id)
            if websocket:
                await websocket.send_text(json.dumps({
                    'type': message_type.value,
                    'content': content,
                    'session_id': session_id
                }))
                return True
            
            return False
            
        except Exception as e:
            return False
    
    async def add_ui_element(self, session_id: str, element: UIElement) -> bool:
        """Add UI element to session"""
        try:
            if session_id not in self._sessions:
                return False
            
            session = self._sessions[session_id]
            
            # Add to last message if it exists
            if session.messages:
                last_message = session.messages[-1]
                last_message.ui_components.append({
                    'element_id': element.element_id,
                    'type': element.component_type.value,
                    'content': element.content,
                    'title': element.title,
                    'description': element.description,
                    'interactive': element.interactive,
                    'callback': element.callback
                })
            
            self._usage_stats['ui_interactions'] += 1
            return True
            
        except Exception as e:
            return False
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        return self._sessions.get(session_id)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions"""
        return [
            {
                'session_id': session.session_id,
                'user_id': session.user_id,
                'status': session.status.value,
                'message_count': len(session.messages),
                'total_tokens': session.total_tokens,
                'start_time': session.start_time.isoformat(),
                'last_activity': session.last_activity.isoformat()
            }
            for session in self._sessions.values()
        ]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = self._usage_stats.copy()
        
        # Calculate derived metrics
        if stats['total_sessions'] > 0:
            stats['messages_per_session'] = stats['total_messages'] / stats['total_sessions']
        
        return stats
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'config': {
                'model': self.config.model,
                'host': self.config.host,
                'port': self.config.port,
                'title': self.config.title,
                'enable_streaming': self.config.enable_streaming,
                'enable_file_upload': self.config.enable_file_upload,
                'theme': self.config.theme,
                'max_concurrent_sessions': self.config.max_concurrent_sessions
            },
            'running': self._running,
            'sessions_count': len(self._sessions),
            'active_websockets': len(self._active_websockets),
            'server_url': f"http://{self.config.host}:{self.config.port}" if self._running else None,
            'usage_stats': self.get_usage_stats(),
            'chainlit_available': CHAINLIT_AVAILABLE,
            'created_at': self._created_at.isoformat()
        }


class ChainlitProvider(BaseAgentProvider):
    """
    Provider implementation for Chainlit.
    
    Enables conversational AI interface with chat UI, async support,
    and integration capabilities for building interactive AI applications.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._agents: Dict[str, ChainlitAgent] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._security_manager = None
        
        if not CHAINLIT_AVAILABLE:
            self.logger.warning("Chainlit dependencies not available - provider will be non-functional")
    
    @property
    def provider_name(self) -> str:
        return "chainlit"
    
    @property
    def supported_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.CONVERSATIONAL_INTERFACE,
            AgentCapability.STREAMING_RESPONSES,
            AgentCapability.WEBSOCKET_SUPPORT,
            AgentCapability.UI_COMPONENTS,
            AgentCapability.FILE_UPLOAD,
            AgentCapability.MULTIMODAL_DISPLAY,
            AgentCapability.SESSION_MANAGEMENT,
            AgentCapability.USER_FEEDBACK
        ]
    
    async def initialize(self) -> bool:
        """Initialize Chainlit provider"""
        if not CHAINLIT_AVAILABLE:
            self.logger.error("Chainlit dependencies not available")
            return False
        
        try:
            # Initialize security manager
            self._security_manager = AgentSecurityManager(self.config.get('security', {}))
            
            self._initialized = True
            self.logger.info("Chainlit provider initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Chainlit provider: {e}")
            return False
    
    async def create_agent(self, config: AgentConfig) -> str:
        """Create a new Chainlit agent"""
        if not self._initialized:
            await self.initialize()
        
        if not CHAINLIT_AVAILABLE:
            raise RuntimeError("Chainlit dependencies not available")
        
        agent_id = f"chainlit_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create Chainlit configuration
            chainlit_config = ChainlitConfig(
                api_key=self.config.get('api_key', ''),
                model=self.config.get('model', 'gpt-4'),
                base_url=self.config.get('base_url', 'https://api.openai.com/v1'),
                max_tokens=self.config.get('max_tokens', 2000),
                temperature=self.config.get('temperature', 0.7),
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 8000),
                title=self.config.get('title', 'Chainlit AI Assistant'),
                description=self.config.get('description', 'Interactive AI Assistant powered by Chainlit'),
                favicon=self.config.get('favicon', '🤖'),
                theme=self.config.get('theme', 'light'),
                enable_streaming=self.config.get('enable_streaming', True),
                enable_file_upload=self.config.get('enable_file_upload', True),
                enable_image_display=self.config.get('enable_image_display', True),
                enable_audio_playback=self.config.get('enable_audio_playback', True),
                enable_video_playback=self.config.get('enable_video_playback', True),
                max_file_size_mb=self.config.get('max_file_size_mb', 100),
                session_timeout=self.config.get('session_timeout', 3600),
                max_concurrent_sessions=self.config.get('max_concurrent_sessions', 100),
                enable_chat_history=self.config.get('enable_chat_history', True),
                enable_user_feedback=self.config.get('enable_user_feedback', True),
                enable_authentication=self.config.get('enable_authentication', False),
                custom_css=self.config.get('custom_css', ''),
                custom_js=self.config.get('custom_js', ''),
                websocket_enabled=self.config.get('websocket_enabled', True),
                rate_limit_per_minute=self.config.get('rate_limit_per_minute', 60),
                enable_markdown=self.config.get('enable_markdown', True),
                enable_latex=self.config.get('enable_latex', True),
                enable_syntax_highlighting=self.config.get('enable_syntax_highlighting', True)
            )
            
            # Create agent
            agent = ChainlitAgent(
                agent_id=agent_id,
                config=chainlit_config,
                security_manager=self._security_manager
            )
            
            # Initialize agent
            if not await agent.initialize():
                raise RuntimeError("Failed to initialize Chainlit agent")
            
            self._agents[agent_id] = agent
            self._agent_configs[agent_id] = config
            
            self.logger.info(f"Created Chainlit agent {agent_id}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create Chainlit agent: {e}")
            raise
    
    async def execute_agent(self, agent_id: str, prompt: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute a Chainlit agent"""
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
            execution_type = context.get('execution_type', 'start_server')
            
            if execution_type == 'start_server':
                result = await self._start_server(agent)
            elif execution_type == 'send_message':
                result = await self._send_message(agent, prompt, context)
            elif execution_type == 'create_session':
                result = await self._create_session(agent, context)
            else:
                result = await self._start_server(agent)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare response
            response_content = result.get('summary', 'Chainlit operation completed')
            
            # Prepare metadata
            metadata = {
                'execution_time_seconds': execution_time,
                'execution_type': execution_type,
                'server_running': result.get('server_running', False),
                'server_url': result.get('server_url'),
                'session_id': result.get('session_id'),
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
            self.logger.error(f"Chainlit agent execution failed: {e}")
            return AgentResponse(
                content="",
                agent_id=agent_id,
                provider=self.provider_name,
                error=str(e)
            )
    
    async def _start_server(self, agent: ChainlitAgent) -> Dict[str, Any]:
        """Start Chainlit server"""
        success = await agent.start_server()
        
        return {
            'server_running': success,
            'server_url': f"http://{agent.config.host}:{agent.config.port}" if success else None,
            'summary': f"Chainlit server {'started' if success else 'failed to start'} at http://{agent.config.host}:{agent.config.port}"
        }
    
    async def _send_message(self, agent: ChainlitAgent, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Send message to session"""
        session_id = context.get('session_id', 'default')
        message_type = MessageType(context.get('message_type', 'assistant'))
        
        success = await agent.send_message_to_session(session_id, content, message_type)
        
        return {
            'message_sent': success,
            'session_id': session_id,
            'summary': f"Message {'sent' if success else 'failed to send'} to session {session_id}"
        }
    
    async def _create_session(self, agent: ChainlitAgent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create new session"""
        session_id = context.get('session_id', f"session_{uuid.uuid4().hex[:8]}")
        user_id = context.get('user_id')
        
        session = await agent._create_session(session_id, user_id)
        
        return {
            'session_id': session.session_id,
            'summary': f"Created session {session_id}"
        }
    
    async def start_server(self, agent_id: str) -> bool:
        """Start server for agent"""
        if agent_id not in self._agents:
            return False
        
        agent = self._agents[agent_id]
        return await agent.start_server()
    
    async def stop_server(self, agent_id: str) -> bool:
        """Stop server for agent"""
        if agent_id not in self._agents:
            return False
        
        agent = self._agents[agent_id]
        await agent.stop_server()
        return True
    
    async def create_session(self, agent_id: str, session_id: str = None, user_id: str = None) -> str:
        """Create a session"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        
        session = await agent._create_session(session_id, user_id)
        return session.session_id
    
    async def send_message_to_session(self, agent_id: str, session_id: str, content: str,
                                    message_type: str = "assistant") -> bool:
        """Send message to session"""
        if agent_id not in self._agents:
            return False
        
        agent = self._agents[agent_id]
        msg_type = MessageType(message_type)
        
        return await agent.send_message_to_session(session_id, content, msg_type)
    
    async def register_tool(self, agent_id: str, tool_spec: ToolSpec) -> bool:
        """Register a tool with Chainlit agent (basic implementation)"""
        if agent_id not in self._agents:
            return False
        
        try:
            # Chainlit doesn't have native tool support in this implementation
            # This would be implemented as UI callbacks
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool: {e}")
            return False
    
    async def synthesize_tool(self, agent_id: str, tool_description: str, examples: List[str] = None) -> ToolSpec:
        """Synthesize a tool using Chainlit capabilities"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            # Create tool spec for conversational interface
            tool_spec = ToolSpec(
                name=f"chainlit_synthesized_{uuid.uuid4().hex[:8]}",
                description=tool_description,
                parameters={
                    'description': tool_description,
                    'execution_type': 'conversational',
                    'ui_based': True,
                    'interactive': True,
                    'examples': examples or []
                },
                security_policy={
                    'risk_level': 'low',
                    'requires_approval': False,
                    'websocket_enabled': True,
                    'session_based': True
                }
            )
            
            self.logger.info(f"Synthesized tool for Chainlit agent {agent_id}")
            return tool_spec
            
        except Exception as e:
            self.logger.error(f"Tool synthesis failed: {e}")
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Clean up and destroy a Chainlit agent"""
        try:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                await agent.stop_server()
                del self._agents[agent_id]
            
            if agent_id in self._agent_configs:
                del self._agent_configs[agent_id]
            
            if self._security_manager:
                self._security_manager.cleanup_agent(agent_id)
            
            self.logger.info(f"Destroyed Chainlit agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def list_agents(self) -> List[str]:
        """List all active Chainlit agent IDs"""
        return list(self._agents.keys())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a Chainlit agent"""
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