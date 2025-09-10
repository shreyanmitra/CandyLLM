"""
🍭 CandyLLM Enterprise Features
Production-ready monitoring, analytics, and scaling capabilities
"""

import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import sqlite3
from pathlib import Path
import psutil
import logging

@dataclass
class RequestMetrics:
    """Metrics for a single LLM request"""
    request_id: str
    timestamp: datetime
    model_name: str
    provider: str
    input_tokens: int
    output_tokens: int
    response_time: float
    cost_estimate: float
    status: str  # success, error, timeout
    error_message: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_requests: int
    queue_size: int
    cache_hit_rate: float

class MetricsCollector:
    """
    Real-time metrics collection and analysis
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Path.home() / ".candyllm" / "metrics.db")
        self.request_metrics: deque = deque(maxsize=10000)  # In-memory buffer
        self.system_metrics: deque = deque(maxsize=1000)
        self.active_requests: Dict[str, float] = {}  # request_id -> start_time
        
        # Initialize database
        self._init_database()
        
        # Start background metrics collection
        self._start_background_collection()
    
    def _init_database(self):
        """Initialize SQLite database for persistent metrics"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_metrics (
                    request_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    model_name TEXT,
                    provider TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    response_time REAL,
                    cost_estimate REAL,
                    status TEXT,
                    error_message TEXT,
                    user_id TEXT,
                    session_id TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    timestamp TEXT PRIMARY KEY,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    active_requests INTEGER,
                    queue_size INTEGER,
                    cache_hit_rate REAL
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_request_timestamp ON request_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_request_model ON request_metrics(model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)")
    
    def _start_background_collection(self):
        """Start background thread for system metrics collection"""
        def collect_system_metrics():
            while True:
                try:
                    metrics = SystemMetrics(
                        timestamp=datetime.now(),
                        cpu_usage=psutil.cpu_percent(),
                        memory_usage=psutil.virtual_memory().percent,
                        disk_usage=psutil.disk_usage('/').percent,
                        active_requests=len(self.active_requests),
                        queue_size=0,  # Will be updated by queue manager
                        cache_hit_rate=self._get_cache_hit_rate()
                    )
                    
                    self.system_metrics.append(metrics)
                    self._persist_system_metrics(metrics)
                    
                except Exception as e:
                    logging.error(f"Error collecting system metrics: {e}")
                
                time.sleep(30)  # Collect every 30 seconds
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def start_request(self, request_id: str) -> None:
        """Mark the start of a request"""
        self.active_requests[request_id] = time.time()
    
    def end_request(self, metrics: RequestMetrics) -> None:
        """Record completed request metrics"""
        # Calculate response time if not provided
        if metrics.request_id in self.active_requests:
            start_time = self.active_requests.pop(metrics.request_id)
            if metrics.response_time == 0:
                metrics.response_time = time.time() - start_time
        
        # Add to in-memory buffer
        self.request_metrics.append(metrics)
        
        # Persist to database
        self._persist_request_metrics(metrics)
    
    def _persist_request_metrics(self, metrics: RequestMetrics):
        """Persist request metrics to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO request_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.request_id,
                    metrics.timestamp.isoformat(),
                    metrics.model_name,
                    metrics.provider,
                    metrics.input_tokens,
                    metrics.output_tokens,
                    metrics.response_time,
                    metrics.cost_estimate,
                    metrics.status,
                    metrics.error_message,
                    metrics.user_id,
                    metrics.session_id
                ))
        except Exception as e:
            logging.error(f"Error persisting request metrics: {e}")
    
    def _persist_system_metrics(self, metrics: SystemMetrics):
        """Persist system metrics to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO system_metrics VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp.isoformat(),
                    metrics.cpu_usage,
                    metrics.memory_usage,
                    metrics.disk_usage,
                    metrics.active_requests,
                    metrics.queue_size,
                    metrics.cache_hit_rate
                ))
        except Exception as e:
            logging.error(f"Error persisting system metrics: {e}")
    
    def _get_cache_hit_rate(self) -> float:
        """Get current cache hit rate from tool registry"""
        try:
            from .tools import ToolRegistry
            stats = ToolRegistry.get_stats()
            return stats.get('cache_hit_rate', 0.0)
        except:
            return 0.0
    
    def get_request_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get request statistics for the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM request_metrics 
                WHERE datetime(timestamp) > datetime(?)
                ORDER BY timestamp DESC
            """, (cutoff.isoformat(),))
            
            rows = cursor.fetchall()
        
        if not rows:
            return {"total_requests": 0}
        
        # Calculate statistics
        total_requests = len(rows)
        successful_requests = sum(1 for row in rows if row[8] == 'success')
        total_tokens = sum(row[4] + row[5] for row in rows)  # input + output tokens
        total_cost = sum(row[7] for row in rows)
        avg_response_time = sum(row[6] for row in rows) / total_requests
        
        # Group by model
        model_stats = defaultdict(int)
        provider_stats = defaultdict(int)
        
        for row in rows:
            model_stats[row[2]] += 1  # model_name
            provider_stats[row[3]] += 1  # provider
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / total_requests,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_response_time": avg_response_time,
            "model_distribution": dict(model_stats),
            "provider_distribution": dict(provider_stats),
            "time_period_hours": hours
        }
    
    def get_system_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get system statistics for the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM system_metrics 
                WHERE datetime(timestamp) > datetime(?)
                ORDER BY timestamp DESC
            """, (cutoff.isoformat(),))
            
            rows = cursor.fetchall()
        
        if not rows:
            return {"data_points": 0}
        
        # Calculate averages and peaks
        cpu_values = [row[1] for row in rows]
        memory_values = [row[2] for row in rows]
        
        return {
            "data_points": len(rows),
            "avg_cpu_usage": sum(cpu_values) / len(cpu_values),
            "peak_cpu_usage": max(cpu_values),
            "avg_memory_usage": sum(memory_values) / len(memory_values),
            "peak_memory_usage": max(memory_values),
            "current_active_requests": len(self.active_requests),
            "time_period_hours": hours
        }

class RateLimiter:
    """
    Token bucket rate limiter for API requests
    """
    
    def __init__(self, requests_per_minute: int = 60, burst_size: Optional[int] = None):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or requests_per_minute
        self.tokens = self.burst_size
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens. Returns True if successful."""
        with self.lock:
            now = time.time()
            
            # Refill tokens based on time passed
            time_passed = now - self.last_refill
            tokens_to_add = int(time_passed * (self.requests_per_minute / 60.0))
            
            if tokens_to_add > 0:
                self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
                self.last_refill = now
            
            # Check if we can fulfill the request
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def acquire_async(self, tokens: int = 1, timeout: float = 30.0) -> bool:
        """Async version that waits for tokens to become available"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.acquire(tokens):
                return True
            
            # Wait a bit before retrying
            await asyncio.sleep(0.1)
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status"""
        with self.lock:
            return {
                "available_tokens": self.tokens,
                "max_tokens": self.burst_size,
                "requests_per_minute": self.requests_per_minute,
                "utilization": 1.0 - (self.tokens / self.burst_size)
            }

class LoadBalancer:
    """
    Simple load balancer for distributing requests across multiple providers/models
    """
    
    def __init__(self):
        self.providers: Dict[str, Dict[str, Any]] = {}
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxsize=100))
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
    
    def register_provider(self, name: str, weight: float = 1.0, max_requests_per_minute: int = 60):
        """Register a provider with load balancing"""
        with self.lock:
            self.providers[name] = {
                "weight": weight,
                "rate_limiter": RateLimiter(max_requests_per_minute),
                "enabled": True
            }
    
    def select_provider(self, exclude: Optional[List[str]] = None) -> Optional[str]:
        """Select the best provider based on current load and performance"""
        exclude = exclude or []
        
        with self.lock:
            available_providers = [
                name for name, config in self.providers.items()
                if config["enabled"] and name not in exclude
            ]
            
            if not available_providers:
                return None
            
            # Calculate scores for each provider
            scores = {}
            for name in available_providers:
                # Base score from weight
                score = self.providers[name]["weight"]
                
                # Adjust for current load
                current_load = self.request_counts[name]
                score *= (1.0 / (1.0 + current_load * 0.1))
                
                # Adjust for recent response times
                recent_times = list(self.response_times[name])
                if recent_times:
                    avg_time = sum(recent_times) / len(recent_times)
                    score *= (1.0 / (1.0 + avg_time * 0.1))
                
                # Adjust for error rate
                error_rate = self.error_counts[name] / max(1, self.request_counts[name])
                score *= (1.0 - error_rate * 0.5)
                
                scores[name] = score
            
            # Select provider with highest score
            return max(scores.items(), key=lambda x: x[1])[0]
    
    def record_request_start(self, provider: str):
        """Record the start of a request"""
        with self.lock:
            self.request_counts[provider] += 1
    
    def record_request_end(self, provider: str, response_time: float, success: bool):
        """Record the completion of a request"""
        with self.lock:
            self.response_times[provider].append(response_time)
            if not success:
                self.error_counts[provider] += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get load balancer status"""
        with self.lock:
            status = {}
            for name, config in self.providers.items():
                recent_times = list(self.response_times[name])
                avg_time = sum(recent_times) / len(recent_times) if recent_times else 0
                error_rate = self.error_counts[name] / max(1, self.request_counts[name])
                
                status[name] = {
                    "enabled": config["enabled"],
                    "weight": config["weight"],
                    "total_requests": self.request_counts[name],
                    "avg_response_time": avg_time,
                    "error_rate": error_rate,
                    "rate_limiter": config["rate_limiter"].get_status()
                }
            
            return status

class SessionManager:
    """
    User session management with persistent storage
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Path.home() / ".candyllm" / "sessions.db")
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
        self._init_database()
    
    def _init_database(self):
        """Initialize session database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TEXT,
                    last_active TEXT,
                    data TEXT,
                    metadata TEXT
                )
            """)
    
    def create_session(self, user_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """Create a new session"""
        import uuid
        
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": now,
            "last_active": now,
            "conversation_history": [],
            "preferences": {},
            "metadata": metadata or {}
        }
        
        with self.lock:
            self.active_sessions[session_id] = session_data
        
        # Persist to database
        self._persist_session(session_data)
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        with self.lock:
            if session_id in self.active_sessions:
                return self.active_sessions[session_id].copy()
        
        # Try to load from database
        return self._load_session(session_id)
    
    def update_session(self, session_id: str, data: Dict[str, Any]):
        """Update session data"""
        with self.lock:
            if session_id not in self.active_sessions:
                # Load from database if not in memory
                session_data = self._load_session(session_id)
                if not session_data:
                    return
                self.active_sessions[session_id] = session_data
            
            # Update data
            self.active_sessions[session_id].update(data)
            self.active_sessions[session_id]["last_active"] = datetime.now()
        
        # Persist changes
        self._persist_session(self.active_sessions[session_id])
    
    def add_to_conversation(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        session_data = self.get_session(session_id)
        if session_data:
            if "conversation_history" not in session_data:
                session_data["conversation_history"] = []
            
            session_data["conversation_history"].append(message)
            self.update_session(session_id, session_data)
    
    def _persist_session(self, session_data: Dict[str, Any]):
        """Persist session to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_data["session_id"],
                    session_data["user_id"],
                    session_data["created_at"].isoformat(),
                    session_data["last_active"].isoformat(),
                    json.dumps(session_data),
                    json.dumps(session_data.get("metadata", {}))
                ))
        except Exception as e:
            logging.error(f"Error persisting session: {e}")
    
    def _load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT data FROM sessions WHERE session_id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                if row:
                    session_data = json.loads(row[0])
                    # Convert timestamp strings back to datetime objects
                    session_data["created_at"] = datetime.fromisoformat(session_data["created_at"])
                    session_data["last_active"] = datetime.fromisoformat(session_data["last_active"])
                    return session_data
        except Exception as e:
            logging.error(f"Error loading session: {e}")
        
        return None
    
    def cleanup_old_sessions(self, days: int = 30):
        """Remove sessions older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    DELETE FROM sessions 
                    WHERE datetime(last_active) < datetime(?)
                """, (cutoff.isoformat(),))
        except Exception as e:
            logging.error(f"Error cleaning up sessions: {e}")

# Global instances
_metrics_collector = None
_load_balancer = None
_session_manager = None

def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

def get_load_balancer() -> LoadBalancer:
    """Get global load balancer instance"""
    global _load_balancer
    if _load_balancer is None:
        _load_balancer = LoadBalancer()
    return _load_balancer

def get_session_manager() -> SessionManager:
    """Get global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
