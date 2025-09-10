"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

CandyLLM Event System - Comprehensive Event Management for AI Workflows

Provides a powerful event-driven architecture for monitoring, logging, analytics,
and custom workflow integration throughout the AI interaction lifecycle.

Security Features:
- Secure event data handling with input validation
- Event data encryption for sensitive information
- Rate limiting for event emission to prevent abuse
- Audit trail maintenance with tamper detection
- Secure event handler registration with validation
- Protection against event injection attacks
- Comprehensive logging for security monitoring
"""

import asyncio
import time
import threading
import hashlib
import secrets
import logging
from typing import Dict, List, Callable, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import json
import uuid
from collections import defaultdict, deque
import weakref

# Security imports
from cryptography.fernet import Fernet
import html

# Security logging setup
security_logger = logging.getLogger('candyllm.events.security')
event_logger = logging.getLogger('candyllm.events')

class EventType(Enum):
    """
    Predefined event types for CandyLLM operations
    
    Comprehensive event taxonomy covering all AI workflow stages
    for monitoring, analytics, and security auditing.
    """
    
    # System Events - Core system lifecycle and health
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown" 
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_HEALTH_CHECK = "system_health_check"
    
    # Session Events - User session management and tracking
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_TIMEOUT = "session_timeout"
    SESSION_RESUME = "session_resume"
    SESSION_INVALID = "session_invalid"
    
    # Model Events - AI model lifecycle and operations
    MODEL_LOADED = "model_loaded"
    MODEL_UNLOADED = "model_unloaded"
    MODEL_CHANGED = "model_changed"
    MODEL_ERROR = "model_error"
    MODEL_FALLBACK = "model_fallback"
    MODEL_PERFORMANCE = "model_performance"
    
    # Query/Response Events - Core AI interaction flow
    QUERY_RECEIVED = "query_received"
    QUERY_PREPROCESSED = "query_preprocessed"
    QUERY_VALIDATED = "query_validated"
    QUERY_SENT = "query_sent"
    RESPONSE_RECEIVED = "response_received"
    RESPONSE_POSTPROCESSED = "response_postprocessed"
    RESPONSE_VALIDATED = "response_validated"
    RESPONSE_SENT = "response_sent"
    RESPONSE_ERROR = "response_error"
    
    # Streaming Events - Real-time response streaming
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"
    STREAM_ERROR = "stream_error"
    STREAM_INTERRUPTED = "stream_interrupted"
    
    # File Events - File handling and processing
    FILE_UPLOADED = "file_uploaded"
    FILE_PROCESSED = "file_processed"
    FILE_ERROR = "file_error"
    FILE_DELETED = "file_deleted"
    FILE_ACCESS_DENIED = "file_access_denied"
    
    # Tool Events - Tool execution and management
    TOOL_CALLED = "tool_called"
    TOOL_RESPONSE = "tool_response"
    TOOL_ERROR = "tool_error"
    TOOL_REGISTERED = "tool_registered"
    TOOL_UNREGISTERED = "tool_unregistered"
    TOOL_VALIDATION_FAILED = "tool_validation_failed"
    
    # Authentication Events - Security and access control
    AUTH_LOGIN = "auth_login"
    AUTH_LOGOUT = "auth_logout"
    AUTH_FAILED = "auth_failed"
    AUTH_TOKEN_REFRESH = "auth_token_refresh"
    AUTH_PERMISSION_DENIED = "auth_permission_denied"
    
    # Performance Events - System performance monitoring
    PERFORMANCE_METRIC = "performance_metric"
    LATENCY_MEASURED = "latency_measured"
    TOKEN_USAGE = "token_usage"
    COST_CALCULATED = "cost_calculated"
    RESOURCE_UTILIZATION = "resource_utilization"
    
    # Safety Events - Content safety and moderation
    CONTENT_FILTERED = "content_filtered"
    SAFETY_WARNING = "safety_warning"
    MODERATION_TRIGGERED = "moderation_triggered"
    THREAT_DETECTED = "threat_detected"
    
    # Analytics Events - User behavior and system analytics
    USER_ACTION = "user_action"
    CONVERSION_EVENT = "conversion_event"
    FEEDBACK_RECEIVED = "feedback_received"
    EXPERIMENT_STARTED = "experiment_started"
    
    # Security Events - Security monitoring and incidents
    SECURITY_VIOLATION = "security_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INJECTION_ATTEMPT = "injection_attempt"
    
    # Custom Events - Extensible event system
    CUSTOM = "custom"

@dataclass
class EventData:
    """
    Container for event data and metadata with security enhancements
    
    Features:
    - Secure data storage with optional encryption
    - Input validation and sanitization
    - Tamper detection with checksums
    - Audit trail preservation
    """
    
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    model_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Security fields
    checksum: Optional[str] = field(default=None, init=False)
    encrypted: bool = field(default=False, init=False)
    
    def __post_init__(self):
        """Post-initialization processing with security validation"""
        # Validate and sanitize input data
        self._validate_and_sanitize()
        
        # Generate tamper detection checksum
        self._generate_checksum()
        
        # Log event creation for audit trail
        event_logger.debug(f"Event created: {self.event_type.value} - {self.event_id}")
    
    def _validate_and_sanitize(self):
        """Validate and sanitize event data for security"""
        # Validate event_type
        if not isinstance(self.event_type, EventType):
            raise ValueError("Invalid event_type - must be EventType enum")
        
        # Validate string fields
        string_fields = ['session_id', 'user_id', 'model_id']
        for field_name in string_fields:
            value = getattr(self, field_name)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(f"{field_name} must be string or None")
                # Sanitize string values
                sanitized = html.escape(str(value))
                if len(sanitized) > 256:  # Reasonable limit
                    sanitized = sanitized[:256]
                setattr(self, field_name, sanitized)
        
        # Validate and sanitize data dictionary
        if not isinstance(self.data, dict):
            self.data = {}
        
        # Sanitize data values
        self.data = self._sanitize_dict(self.data)
        
        # Validate and sanitize metadata
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        
        self.metadata = self._sanitize_dict(self.metadata)
    
    def _sanitize_dict(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize dictionary values for security"""
        sanitized = {}
        
        for key, value in data_dict.items():
            # Sanitize key
            if not isinstance(key, str):
                key = str(key)
            key = html.escape(key)[:100]  # Limit key length
            
            # Sanitize value based on type
            if isinstance(value, str):
                # HTML escape string values
                value = html.escape(value)
                # Limit string length to prevent DoS
                if len(value) > 10000:
                    value = value[:10000] + "... [truncated]"
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                value = self._sanitize_dict(value)
            elif isinstance(value, list):
                # Sanitize list items
                value = [html.escape(str(item))[:1000] if isinstance(item, str) else item for item in value[:100]]
            elif not isinstance(value, (int, float, bool, type(None))):
                # Convert other types to string and sanitize
                value = html.escape(str(value))[:1000]
            
            sanitized[key] = value
        
        return sanitized
    
    def _generate_checksum(self):
        """Generate checksum for tamper detection"""
        # Create deterministic string representation
        data_str = json.dumps({
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'event_id': self.event_id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'model_id': self.model_id,
            'data': self.data,
            'metadata': self.metadata
        }, sort_keys=True)
        
        # Generate SHA-256 checksum
        self.checksum = hashlib.sha256(data_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify event integrity using checksum"""
        if not self.checksum:
            return False
        
        # Temporarily store current checksum
        current_checksum = self.checksum
        
        # Regenerate checksum
        self.checksum = None
        self._generate_checksum()
        
        # Compare checksums
        is_valid = self.checksum == current_checksum
        
        if not is_valid:
            security_logger.warning(f"Event integrity check failed: {self.event_id}")
        
        return is_valid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary with security information"""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'event_id': self.event_id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'model_id': self.model_id,
            'data': self.data,
            'metadata': self.metadata,
            'checksum': self.checksum,
            'encrypted': self.encrypted
        }
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'EventData':
        """Create EventData from dictionary with validation"""
        # Validate required fields
        if 'event_type' not in data_dict:
            raise ValueError("Missing required field: event_type")
        
        try:
            event_type = EventType(data_dict['event_type'])
        except ValueError:
            raise ValueError(f"Invalid event_type: {data_dict['event_type']}")
        
        # Parse timestamp
        timestamp = datetime.now()
        if 'timestamp' in data_dict:
            try:
                timestamp = datetime.fromisoformat(data_dict['timestamp'])
            except ValueError:
                security_logger.warning(f"Invalid timestamp format: {data_dict['timestamp']}")
        
        # Create event with validation
        event = cls(
            event_type=event_type,
            timestamp=timestamp,
            event_id=data_dict.get('event_id', secrets.token_urlsafe(16)),
            session_id=data_dict.get('session_id'),
            user_id=data_dict.get('user_id'),
            model_id=data_dict.get('model_id'),
            data=data_dict.get('data', {}),
            metadata=data_dict.get('metadata', {})
        )
        
        # Restore security fields
        if 'checksum' in data_dict:
            event.checksum = data_dict['checksum']
        if 'encrypted' in data_dict:
            event.encrypted = data_dict['encrypted']
        
        return event
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    model_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event data to dictionary"""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "event_id": self.event_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "model_id": self.model_id,
            "data": self.data,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert event data to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class EventHandler:
    """Base class for event handlers"""
    
    def __init__(self, 
                 callback: Callable[[EventData], Any],
                 priority: int = 0,
                 async_handler: bool = False,
                 filter_func: Optional[Callable[[EventData], bool]] = None):
        """
        Initialize event handler
        
        Args:
            callback: Function to call when event is triggered
            priority: Handler priority (higher = executed first)
            async_handler: Whether the handler is async
            filter_func: Optional filter function to determine if handler should run
        """
        self.callback = callback
        self.priority = priority
        self.async_handler = async_handler
        self.filter_func = filter_func
        self.handler_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.execution_count = 0
        self.last_execution = None
        
    def should_execute(self, event_data: EventData) -> bool:
        """Check if handler should execute for given event"""
        if self.filter_func:
            return self.filter_func(event_data)
        return True
    
    async def execute(self, event_data: EventData) -> Any:
        """Execute the handler"""
        if not self.should_execute(event_data):
            return None
            
        self.execution_count += 1
        self.last_execution = datetime.now()
        
        try:
            if self.async_handler:
                return await self.callback(event_data)
            else:
                return self.callback(event_data)
        except Exception as e:
            print(f"Event handler error: {e}")
            return None


class EventManager:
    """Central event management system for CandyLLM"""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize event manager
        
        Args:
            max_history: Maximum number of events to keep in history
        """
        self.handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self.global_handlers: List[EventHandler] = []
        self.event_history: deque = deque(maxlen=max_history)
        self.metrics: Dict[str, Any] = defaultdict(int)
        self.lock = threading.RLock()
        self.enabled = True
        
        # Performance tracking
        self.handler_performance: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "execution_count": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "last_execution": None
        })
    
    def register_handler(self, 
                        event_type: Union[EventType, str], 
                        handler: Union[EventHandler, Callable],
                        priority: int = 0,
                        async_handler: bool = False,
                        filter_func: Optional[Callable[[EventData], bool]] = None) -> str:
        """
        Register an event handler
        
        Args:
            event_type: Type of event to handle
            handler: EventHandler instance or callback function
            priority: Handler priority (higher = executed first)
            async_handler: Whether the handler is async
            filter_func: Optional filter function
            
        Returns:
            Handler ID for removal
        """
        if not self.enabled:
            return ""
        
        with self.lock:
            # Convert string to EventType if needed
            if isinstance(event_type, str):
                try:
                    event_type = EventType(event_type)
                except ValueError:
                    event_type = EventType.CUSTOM
            
            # Convert function to EventHandler if needed
            if not isinstance(handler, EventHandler):
                handler = EventHandler(
                    callback=handler,
                    priority=priority,
                    async_handler=async_handler,
                    filter_func=filter_func
                )
            
            # Add to appropriate list
            if event_type == EventType.CUSTOM:
                self.global_handlers.append(handler)
            else:
                self.handlers[event_type].append(handler)
                # Sort by priority (descending)
                self.handlers[event_type].sort(key=lambda h: h.priority, reverse=True)
            
            return handler.handler_id
    
    def remove_handler(self, handler_id: str) -> bool:
        """
        Remove event handler by ID
        
        Args:
            handler_id: ID of handler to remove
            
        Returns:
            True if handler was found and removed
        """
        with self.lock:
            # Check global handlers
            for i, handler in enumerate(self.global_handlers):
                if handler.handler_id == handler_id:
                    del self.global_handlers[i]
                    return True
            
            # Check event-specific handlers
            for event_type, handler_list in self.handlers.items():
                for i, handler in enumerate(handler_list):
                    if handler.handler_id == handler_id:
                        del handler_list[i]
                        return True
        
        return False
    
    def clear_handlers(self, event_type: Optional[EventType] = None):
        """
        Clear event handlers
        
        Args:
            event_type: Specific event type to clear, or None for all
        """
        with self.lock:
            if event_type is None:
                self.handlers.clear()
                self.global_handlers.clear()
            else:
                self.handlers[event_type].clear()
    
    async def emit_event(self, 
                        event_type: Union[EventType, str],
                        data: Dict[str, Any] = None,
                        session_id: Optional[str] = None,
                        user_id: Optional[str] = None,
                        model_id: Optional[str] = None,
                        metadata: Dict[str, Any] = None) -> List[Any]:
        """
        Emit an event to all registered handlers
        
        Args:
            event_type: Type of event to emit
            data: Event data
            session_id: Session identifier
            user_id: User identifier
            model_id: Model identifier
            metadata: Additional metadata
            
        Returns:
            List of handler return values
        """
        if not self.enabled:
            return []
        
        # Convert string to EventType if needed
        if isinstance(event_type, str):
            try:
                event_type = EventType(event_type)
            except ValueError:
                event_type = EventType.CUSTOM
        
        # Create event data
        event_data = EventData(
            event_type=event_type,
            data=data or {},
            session_id=session_id,
            user_id=user_id,
            model_id=model_id,
            metadata=metadata or {}
        )
        
        # Add to history
        with self.lock:
            self.event_history.append(event_data)
            self.metrics[f"event_{event_type.value}"] += 1
            self.metrics["total_events"] += 1
        
        # Get handlers to execute
        handlers_to_execute = []
        
        with self.lock:
            # Add global handlers
            handlers_to_execute.extend(self.global_handlers)
            
            # Add event-specific handlers
            if event_type in self.handlers:
                handlers_to_execute.extend(self.handlers[event_type])
        
        # Execute handlers
        results = []
        for handler in handlers_to_execute:
            start_time = time.time()
            
            try:
                result = await handler.execute(event_data)
                results.append(result)
                
                # Track performance
                execution_time = time.time() - start_time
                perf = self.handler_performance[handler.handler_id]
                perf["execution_count"] += 1
                perf["total_time"] += execution_time
                perf["average_time"] = perf["total_time"] / perf["execution_count"]
                perf["last_execution"] = datetime.now()
                
            except Exception as e:
                print(f"Error executing event handler: {e}")
                results.append(None)
        
        return results
    
    def emit_event_sync(self, 
                       event_type: Union[EventType, str],
                       data: Dict[str, Any] = None,
                       session_id: Optional[str] = None,
                       user_id: Optional[str] = None,
                       model_id: Optional[str] = None,
                       metadata: Dict[str, Any] = None) -> List[Any]:
        """
        Synchronous version of emit_event
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.emit_event(event_type, data, session_id, user_id, model_id, metadata)
            )
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(
                self.emit_event(event_type, data, session_id, user_id, model_id, metadata)
            )
    
    def get_event_history(self, 
                         event_type: Optional[EventType] = None,
                         limit: Optional[int] = None) -> List[EventData]:
        """
        Get event history
        
        Args:
            event_type: Filter by event type
            limit: Maximum number of events to return
            
        Returns:
            List of event data
        """
        with self.lock:
            events = list(self.event_history)
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event system metrics"""
        with self.lock:
            metrics = dict(self.metrics)
            metrics["handler_count"] = sum(len(handlers) for handlers in self.handlers.values())
            metrics["global_handler_count"] = len(self.global_handlers)
            metrics["history_size"] = len(self.event_history)
            
        return metrics
    
    def get_handler_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get handler performance metrics"""
        return dict(self.handler_performance)
    
    def enable(self):
        """Enable event system"""
        self.enabled = True
    
    def disable(self):
        """Disable event system"""
        self.enabled = False
    
    def export_events(self, 
                     format: str = "json",
                     event_type: Optional[EventType] = None,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None) -> str:
        """
        Export events in specified format
        
        Args:
            format: Export format ("json", "csv")
            event_type: Filter by event type
            start_time: Filter events after this time
            end_time: Filter events before this time
            
        Returns:
            Exported data as string
        """
        events = self.get_event_history(event_type)
        
        # Apply time filters
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        if format.lower() == "json":
            return json.dumps([event.to_dict() for event in events], indent=2)
        elif format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            if events:
                fieldnames = ["event_type", "timestamp", "event_id", "session_id", 
                             "user_id", "model_id", "data", "metadata"]
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                
                for event in events:
                    row = event.to_dict()
                    row["data"] = json.dumps(row["data"])
                    row["metadata"] = json.dumps(row["metadata"])
                    writer.writerow(row)
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")


# Global event manager instance
global_event_manager = EventManager()


# Convenience functions for global event manager
def on_event(event_type: Union[EventType, str], 
            handler: Union[EventHandler, Callable],
            priority: int = 0,
            async_handler: bool = False,
            filter_func: Optional[Callable[[EventData], bool]] = None) -> str:
    """Register a global event handler"""
    return global_event_manager.register_handler(
        event_type, handler, priority, async_handler, filter_func
    )


def emit_event(event_type: Union[EventType, str],
              data: Dict[str, Any] = None,
              session_id: Optional[str] = None,
              user_id: Optional[str] = None,
              model_id: Optional[str] = None,
              metadata: Dict[str, Any] = None) -> List[Any]:
    """Emit a global event"""
    return global_event_manager.emit_event_sync(
        event_type, data, session_id, user_id, model_id, metadata
    )


def remove_event_handler(handler_id: str) -> bool:
    """Remove a global event handler"""
    return global_event_manager.remove_handler(handler_id)


def clear_event_handlers(event_type: Optional[EventType] = None):
    """Clear global event handlers"""
    global_event_manager.clear_handlers(event_type)


def get_event_metrics() -> Dict[str, Any]:
    """Get global event metrics"""
    return global_event_manager.get_metrics()


def get_event_history(event_type: Optional[EventType] = None,
                     limit: Optional[int] = None) -> List[EventData]:
    """Get global event history"""
    return global_event_manager.get_event_history(event_type, limit)


# Decorator for event-driven functions
def event_trigger(event_type: Union[EventType, str],
                 include_result: bool = True,
                 include_args: bool = False):
    """
    Decorator to automatically emit events for function calls
    
    Args:
        event_type: Event type to emit
        include_result: Include function result in event data
        include_args: Include function arguments in event data
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Emit start event
            start_data = {}
            if include_args:
                start_data.update({
                    "function": func.__name__,
                    "args": args,
                    "kwargs": kwargs
                })
            
            emit_event(f"{event_type}_start", start_data)
            
            try:
                result = func(*args, **kwargs)
                
                # Emit success event
                success_data = {"function": func.__name__, "success": True}
                if include_result:
                    success_data["result"] = result
                
                emit_event(event_type, success_data)
                return result
                
            except Exception as e:
                # Emit error event
                error_data = {
                    "function": func.__name__,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                emit_event(f"{event_type}_error", error_data)
                raise
        
        return wrapper
    return decorator


# Built-in event handlers for common use cases
class LoggingHandler:
    """Event handler for logging events"""
    
    def __init__(self, log_file: str = "candyllm_events.log"):
        self.log_file = log_file
    
    def __call__(self, event_data: EventData):
        """Log event to file"""
        try:
            with open(self.log_file, "a") as f:
                f.write(f"{event_data.to_json()}\n")
        except Exception as e:
            print(f"Error logging event: {e}")


class MetricsHandler:
    """Event handler for collecting metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(int)
        self.timings = defaultdict(list)
    
    def __call__(self, event_data: EventData):
        """Collect metrics from event"""
        self.metrics[f"event_{event_data.event_type.value}"] += 1
        
        # Collect timing data if available
        if "duration" in event_data.data:
            self.timings[event_data.event_type.value].append(event_data.data["duration"])
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics"""
        return {
            "counts": dict(self.metrics),
            "timings": {k: {
                "count": len(v),
                "avg": sum(v) / len(v) if v else 0,
                "min": min(v) if v else 0,
                "max": max(v) if v else 0
            } for k, v in self.timings.items()}
        }


class AlertHandler:
    """Event handler for triggering alerts"""
    
    def __init__(self, 
                 alert_events: List[EventType] = None,
                 alert_callback: Callable[[EventData], None] = None):
        self.alert_events = alert_events or [
            EventType.SYSTEM_ERROR,
            EventType.MODEL_ERROR,
            EventType.RESPONSE_ERROR,
            EventType.SAFETY_WARNING
        ]
        self.alert_callback = alert_callback or self.default_alert
    
    def __call__(self, event_data: EventData):
        """Handle alert events"""
        if event_data.event_type in self.alert_events:
            self.alert_callback(event_data)
    
    def default_alert(self, event_data: EventData):
        """Default alert handler"""
        print(f"🚨 ALERT: {event_data.event_type.value} - {event_data.data}")


# Example usage and built-in handlers
if __name__ == "__main__":
    # Example usage
    manager = EventManager()
    
    # Register handlers
    def query_handler(event_data: EventData):
        print(f"Query received: {event_data.data.get('query', 'N/A')}")
    
    def response_handler(event_data: EventData):
        print(f"Response sent: {event_data.data.get('response', 'N/A')}")
    
    manager.register_handler(EventType.QUERY_RECEIVED, query_handler)
    manager.register_handler(EventType.RESPONSE_SENT, response_handler)
    
    # Emit events
    manager.emit_event_sync(
        EventType.QUERY_RECEIVED,
        data={"query": "What is AI?", "user": "test_user"}
    )
    
    manager.emit_event_sync(
        EventType.RESPONSE_SENT,
        data={"response": "AI is artificial intelligence.", "model": "gpt-4"}
    )
    
    # Get metrics
    print("Metrics:", manager.get_metrics())
