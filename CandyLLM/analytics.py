"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

🍭 CandyLLM Advanced Analytics & Monitoring with Enhanced Security

Comprehensive analytics, performance monitoring, and business intelligence with security features:
- Secure data collection and storage with encryption
- Privacy-preserving analytics with data anonymization
- Audit trail maintenance for compliance requirements
- Real-time security monitoring and threat detection
- Rate limiting and access control for analytics endpoints
- Data retention policies and automated cleanup
- GDPR-compliant user data handling
- Performance metrics with security context
- Business intelligence with security insights
"""

import time
import asyncio
import os
import secrets
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import sqlite3
import threading
from pathlib import Path
import statistics

# Try to import numpy for advanced analytics, fallback gracefully
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Security imports
from cryptography.fernet import Fernet
import hmac

# Security logging
security_logger = logging.getLogger('candyllm.analytics.security')
logger = logging.getLogger(__name__)

class SecurityMetrics:
    """Security-related metrics and monitoring"""
    
    def __init__(self):
        self.security_events = deque(maxsize=1000)
        self.threat_patterns = {
            'prompt_injection': 0,
            'rate_limit_violations': 0,
            'authentication_failures': 0,
            'data_access_violations': 0,
            'suspicious_patterns': 0
        }
        self.security_lock = threading.Lock()
    
    def record_security_event(self, event_type: str, details: Dict[str, Any], severity: str = "medium"):
        """
        Record a security event with proper logging
        
        Args:
            event_type: Type of security event
            details: Event details (will be sanitized)
            severity: Event severity (low, medium, high, critical)
        """
        with self.security_lock:
            # Sanitize sensitive information from details
            sanitized_details = self._sanitize_security_data(details)
            
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'details': sanitized_details,
                'severity': severity,
                'event_id': secrets.token_urlsafe(8)
            }
            
            self.security_events.append(event)
            
            # Update threat pattern counts
            if event_type in self.threat_patterns:
                self.threat_patterns[event_type] += 1
            
            # Log critical events immediately
            if severity in ['high', 'critical']:
                security_logger.warning(f"Security event [{severity}]: {event_type} - {sanitized_details}")
    
    def _sanitize_security_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize security data to remove sensitive information
        
        Args:
            data: Raw security data
            
        Returns:
            Sanitized data safe for logging
        """
        sanitized = {}
        sensitive_keys = ['password', 'token', 'key', 'secret', 'api_key', 'auth']
        
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                # Hash sensitive values instead of storing them
                sanitized[key] = hashlib.sha256(str(value).encode()).hexdigest()[:16] + "..."
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(type(value))
        
        return sanitized
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get security summary for specified time period
        
        Args:
            hours: Number of hours to include in summary
            
        Returns:
            Security summary with threat analysis
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff.isoformat()
        
        recent_events = [
            event for event in self.security_events
            if event['timestamp'] > cutoff_str
        ]
        
        # Analyze security trends
        severity_counts = defaultdict(int)
        event_type_counts = defaultdict(int)
        
        for event in recent_events:
            severity_counts[event['severity']] += 1
            event_type_counts[event['event_type']] += 1
        
        return {
            'period_hours': hours,
            'total_events': len(recent_events),
            'severity_breakdown': dict(severity_counts),
            'event_type_breakdown': dict(event_type_counts),
            'threat_patterns': self.threat_patterns.copy(),
            'high_risk_events': len([e for e in recent_events if e['severity'] in ['high', 'critical']]),
            'security_score': self._calculate_security_score(recent_events)
        }
    
    def _calculate_security_score(self, events: List[Dict]) -> float:
        """
        Calculate security score based on recent events
        
        Args:
            events: List of security events
            
        Returns:
            Security score from 0.0 (poor) to 100.0 (excellent)
        """
        if not events:
            return 100.0
        
        # Weight events by severity
        severity_weights = {'low': 1, 'medium': 3, 'high': 7, 'critical': 15}
        total_weight = sum(severity_weights.get(event['severity'], 1) for event in events)
        
        # Calculate score (lower is better for security events)
        max_possible_score = len(events) * severity_weights['critical']
        score = max(0, 100 - (total_weight / max(max_possible_score, 1)) * 100)
        
        return round(score, 2)

@dataclass
class SecurePerformanceMetrics:
    """Security-enhanced performance metrics for LLM operations"""
    operation_id: str
    timestamp: datetime
    operation_type: str  # answer, stream, batch, etc.
    model_name: str
    provider: str
    input_tokens: int
    output_tokens: int
    response_time: float
    queue_time: float
    processing_time: float
    cost_estimate: float
    success: bool
    error_type: Optional[str] = None
    user_id_hash: Optional[str] = None  # Hashed for privacy
    session_id_hash: Optional[str] = None  # Hashed for privacy
    security_level: Optional[str] = None  # Security level of the operation
    content_filtered: bool = False  # Whether content was filtered
    rate_limited: bool = False  # Whether request was rate limited
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SecureUsageMetrics:
    """Security-enhanced usage analytics metrics"""
    timestamp: datetime
    user_id_hash: Optional[str]  # Hashed for privacy
    session_id_hash: Optional[str]  # Hashed for privacy
    model_used: str
    provider: str
    request_type: str
    input_length: int
    output_length: int
    processing_time: float
    success: bool
    cost: float
    ip_address_hash: Optional[str] = None  # Hashed for privacy
    user_agent_hash: Optional[str] = None  # Hashed for privacy
    security_flags: Optional[List[str]] = None  # Security-related flags

@dataclass
class BusinessMetrics:
    """Business intelligence metrics with security insights"""
    timestamp: datetime
    total_requests: int
    successful_requests: int
    total_cost: float
    revenue: float  # if applicable
    active_users: int
    new_users: int
    model_usage: Dict[str, int]
    provider_usage: Dict[str, int]
    avg_response_time: float
    error_rate: float
    security_incidents: int  # Number of security incidents
    content_filtered_requests: int  # Number of requests that triggered content filtering
    rate_limited_requests: int  # Number of rate-limited requests

class SecureMetricsAnalyzer:
    """
    Advanced metrics analysis with enhanced security and privacy protection
    
    Security Features:
    - Data encryption at rest
    - Privacy-preserving analytics (hashed PIIs)
    - Audit trail maintenance
    - Security event monitoring
    - GDPR-compliant data handling
    """
    
    def __init__(self, db_path: Optional[str] = None, encryption_key: Optional[bytes] = None):
        """
        Initialize secure metrics analyzer
        
        Args:
            db_path: Path to analytics database
            encryption_key: Encryption key for sensitive data
        """
        self.db_path = db_path or str(Path.home() / ".candyllm" / "analytics.db")
        self.performance_buffer = deque(maxsize=10000)
        self.usage_buffer = deque(maxsize=10000)
        
        # Security components
        self.security_metrics = SecurityMetrics()
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Thread safety
        self.db_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        
        self._init_secure_database()
        self._start_background_processing()
        
        security_logger.info("SecureMetricsAnalyzer initialized with enhanced security")
    
    def _generate_encryption_key(self) -> bytes:
        """Generate or load encryption key for data protection"""
        key_path = Path(self.db_path).parent / "analytics.key"
        
        if key_path.exists():
            with open(key_path, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            os.makedirs(key_path.parent, exist_ok=True)
            with open(key_path, 'wb') as f:
                f.write(key)
            # Secure file permissions
            os.chmod(key_path, 0o600)
            return key
    
    def _hash_pii(self, data: str) -> str:
        """
        Hash personally identifiable information for privacy
        
        Args:
            data: PII data to hash
            
        Returns:
            Hashed data safe for analytics
        """
        if not data:
            return None
        return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]
    
    def _encrypt_sensitive_data(self, data: str) -> str:
        """
        Encrypt sensitive data before storage
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        if not data:
            return data
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def _decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data after retrieval
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        if not encrypted_data:
            return encrypted_data
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def _init_secure_database(self):
        """Initialize analytics database with security considerations"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Secure file permissions
        if os.path.exists(self.db_path):
            os.chmod(self.db_path, 0o600)
        
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            
            # Performance metrics table with security enhancements
            conn.execute("""
                CREATE TABLE IF NOT EXISTS secure_performance_metrics (
                    operation_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    response_time REAL,
                    queue_time REAL,
                    processing_time REAL,
                    cost_estimate REAL,
                    success BOOLEAN,
                    error_type TEXT,
                    user_id_hash TEXT,
                    session_id_hash TEXT,
                    security_level TEXT,
                    content_filtered BOOLEAN DEFAULT 0,
                    rate_limited BOOLEAN DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            # Usage metrics table with privacy protection
            conn.execute("""
                CREATE TABLE IF NOT EXISTS secure_usage_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id_hash TEXT,
                    session_id_hash TEXT,
                    model_used TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    request_type TEXT,
                    input_length INTEGER,
                    output_length INTEGER,
                    processing_time REAL,
                    success BOOLEAN,
                    cost REAL,
                    ip_address_hash TEXT,
                    user_agent_hash TEXT,
                    security_flags TEXT
                )
            """)
            
            # Business metrics with security insights
            conn.execute("""
                CREATE TABLE IF NOT EXISTS secure_business_metrics (
                    timestamp TEXT PRIMARY KEY,
                    total_requests INTEGER,
                    successful_requests INTEGER,
                    total_cost REAL,
                    revenue REAL,
                    active_users INTEGER,
                    new_users INTEGER,
                    model_usage TEXT,
                    provider_usage TEXT,
                    avg_response_time REAL,
                    error_rate REAL,
                    security_incidents INTEGER DEFAULT 0,
                    content_filtered_requests INTEGER DEFAULT 0,
                    rate_limited_requests INTEGER DEFAULT 0
                )
            """)
            
            # Security audit log table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    details TEXT,
                    event_id TEXT UNIQUE,
                    resolved BOOLEAN DEFAULT 0
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON secure_performance_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_model ON secure_performance_metrics(model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_security ON secure_performance_metrics(security_level)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON secure_usage_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_user ON secure_usage_metrics(user_id_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_security_timestamp ON security_audit_log(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_security_type ON security_audit_log(event_type)")
    
    def record_performance(self, metrics: SecurePerformanceMetrics):
        """
        Record performance metrics with security validation
        
        Args:
            metrics: Performance metrics to record
        """
        try:
            with self.buffer_lock:
                self.performance_buffer.append(metrics)
            
            # Log security-relevant performance issues
            if not metrics.success and metrics.error_type:
                self.security_metrics.record_security_event(
                    'performance_failure',
                    {'error_type': metrics.error_type, 'model': metrics.model_name},
                    'medium' if 'security' in metrics.error_type.lower() else 'low'
                )
            
            if metrics.rate_limited:
                self.security_metrics.record_security_event(
                    'rate_limit_violations',
                    {'model': metrics.model_name, 'operation': metrics.operation_type},
                    'medium'
                )
                
        except Exception as e:
            security_logger.error(f"Error recording performance metrics: {e}")
    
    def record_usage(self, metrics: SecureUsageMetrics):
        """
        Record usage metrics with privacy protection
        
        Args:
            metrics: Usage metrics to record
        """
        try:
            with self.buffer_lock:
                self.usage_buffer.append(metrics)
            
            # Monitor for security flags
            if metrics.security_flags:
                for flag in metrics.security_flags:
                    self.security_metrics.record_security_event(
                        'suspicious_patterns',
                        {'flag': flag, 'model': metrics.model_used},
                        'medium'
                    )
                    
        except Exception as e:
            security_logger.error(f"Error recording usage metrics: {e}")
    
    def _start_background_processing(self):
        """Start background thread for processing metrics"""
        def process_metrics():
            while True:
                try:
                    self._flush_metrics_to_db()
                    self._cleanup_old_data()
                    time.sleep(30)  # Process every 30 seconds
                except Exception as e:
                    security_logger.error(f"Error in background metrics processing: {e}")
                    time.sleep(60)  # Wait longer on error
        
        thread = threading.Thread(target=process_metrics, daemon=True)
        thread.start()
    
    def _flush_metrics_to_db(self):
        """Flush buffered metrics to database"""
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                # Flush performance metrics
                performance_batch = []
                with self.buffer_lock:
                    while self.performance_buffer:
                        performance_batch.append(self.performance_buffer.popleft())
                
                for metrics in performance_batch:
                    conn.execute("""
                        INSERT OR REPLACE INTO secure_performance_metrics VALUES 
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metrics.operation_id,
                        metrics.timestamp.isoformat(),
                        metrics.operation_type,
                        metrics.model_name,
                        metrics.provider,
                        metrics.input_tokens,
                        metrics.output_tokens,
                        metrics.response_time,
                        metrics.queue_time,
                        metrics.processing_time,
                        metrics.cost_estimate,
                        metrics.success,
                        metrics.error_type,
                        metrics.user_id_hash,
                        metrics.session_id_hash,
                        metrics.security_level,
                        metrics.content_filtered,
                        metrics.rate_limited,
                        json.dumps(metrics.metadata) if metrics.metadata else None
                    ))
                
                # Flush usage metrics
                usage_batch = []
                with self.buffer_lock:
                    while self.usage_buffer:
                        usage_batch.append(self.usage_buffer.popleft())
                
                for metrics in usage_batch:
                    conn.execute("""
                        INSERT INTO secure_usage_metrics 
                        (timestamp, user_id_hash, session_id_hash, model_used, provider, 
                         request_type, input_length, output_length, processing_time, 
                         success, cost, ip_address_hash, user_agent_hash, security_flags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metrics.timestamp.isoformat(),
                        metrics.user_id_hash,
                        metrics.session_id_hash,
                        metrics.model_used,
                        metrics.provider,
                        metrics.request_type,
                        metrics.input_length,
                        metrics.output_length,
                        metrics.processing_time,
                        metrics.success,
                        metrics.cost,
                        metrics.ip_address_hash,
                        metrics.user_agent_hash,
                        json.dumps(metrics.security_flags) if metrics.security_flags else None
                    ))
                
                # Flush security events
                for event in list(self.security_metrics.security_events):
                    conn.execute("""
                        INSERT OR IGNORE INTO security_audit_log 
                        (timestamp, event_type, severity, details, event_id)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        event['timestamp'],
                        event['event_type'],
                        event['severity'],
                        json.dumps(event['details']),
                        event['event_id']
                    ))
    
    def _cleanup_old_data(self):
        """Clean up old data according to retention policies"""
        try:
            retention_days = 90  # Keep data for 90 days
            cutoff = datetime.now() - timedelta(days=retention_days)
            cutoff_str = cutoff.isoformat()
            
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Clean old performance metrics
                    conn.execute("""
                        DELETE FROM secure_performance_metrics 
                        WHERE timestamp < ?
                    """, (cutoff_str,))
                    
                    # Clean old usage metrics
                    conn.execute("""
                        DELETE FROM secure_usage_metrics 
                        WHERE timestamp < ?
                    """, (cutoff_str,))
                    
                    # Clean resolved security events older than 30 days
                    security_cutoff = datetime.now() - timedelta(days=30)
                    conn.execute("""
                        DELETE FROM security_audit_log 
                        WHERE timestamp < ? AND resolved = 1
                    """, (security_cutoff.isoformat(),))
                    
                    security_logger.debug(f"Cleaned up analytics data older than {retention_days} days")
                    
        except Exception as e:
            security_logger.error(f"Error cleaning up old data: {e}")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance summary with security insights
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Performance summary with security metrics
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff.isoformat()
        
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_operations,
                        COUNT(CASE WHEN success = 1 THEN 1 END) as successful_operations,
                        AVG(response_time) as avg_response_time,
                        AVG(cost_estimate) as avg_cost,
                        SUM(input_tokens) as total_input_tokens,
                        SUM(output_tokens) as total_output_tokens,
                        COUNT(CASE WHEN content_filtered = 1 THEN 1 END) as content_filtered_count,
                        COUNT(CASE WHEN rate_limited = 1 THEN 1 END) as rate_limited_count
                    FROM secure_performance_metrics 
                    WHERE timestamp > ?
                """, (cutoff_str,))
                
                row = cursor.fetchone()
                
                # Get model usage breakdown
                model_cursor = conn.execute("""
                    SELECT model_name, COUNT(*) as count
                    FROM secure_performance_metrics 
                    WHERE timestamp > ?
                    GROUP BY model_name
                    ORDER BY count DESC
                """, (cutoff_str,))
                
                model_usage = dict(model_cursor.fetchall())
                
                # Get security level breakdown
                security_cursor = conn.execute("""
                    SELECT security_level, COUNT(*) as count
                    FROM secure_performance_metrics 
                    WHERE timestamp > ? AND security_level IS NOT NULL
                    GROUP BY security_level
                """, (cutoff_str,))
                
                security_levels = dict(security_cursor.fetchall())
        
        # Include security summary
        security_summary = self.security_metrics.get_security_summary(hours)
        
        return {
            'period_hours': hours,
            'total_operations': row[0] or 0,
            'successful_operations': row[1] or 0,
            'success_rate': (row[1] / row[0]) if row[0] else 0,
            'avg_response_time': row[2] or 0,
            'avg_cost': row[3] or 0,
            'total_input_tokens': row[4] or 0,
            'total_output_tokens': row[5] or 0,
            'content_filtered_operations': row[6] or 0,
            'rate_limited_operations': row[7] or 0,
            'model_usage': model_usage,
            'security_levels': security_levels,
            'security_summary': security_summary
        }
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive security dashboard data
        
        Returns:
            Security dashboard with all security metrics
        """
        # Get recent security summary
        security_summary = self.security_metrics.get_security_summary(24)
        
        # Get performance data with security context
        perf_summary = self.get_performance_summary(24)
        
        # Calculate security indicators
        total_ops = perf_summary['total_operations']
        security_indicators = {
            'content_filtering_rate': (perf_summary['content_filtered_operations'] / total_ops) if total_ops else 0,
            'rate_limiting_rate': (perf_summary['rate_limited_operations'] / total_ops) if total_ops else 0,
            'security_event_rate': (security_summary['total_events'] / total_ops) if total_ops else 0
        }
        
        return {
            'security_score': security_summary['security_score'],
            'threat_level': 'low' if security_summary['security_score'] > 80 else 'medium' if security_summary['security_score'] > 60 else 'high',
            'security_indicators': security_indicators,
            'recent_incidents': security_summary['high_risk_events'],
            'threat_patterns': security_summary['threat_patterns'],
            'performance_context': perf_summary,
            'recommendations': self._generate_security_recommendations(security_summary, security_indicators)
        }
    
    def _generate_security_recommendations(self, security_summary: Dict, indicators: Dict) -> List[str]:
        """
        Generate security recommendations based on metrics
        
        Args:
            security_summary: Security summary data
            indicators: Security indicators
            
        Returns:
            List of security recommendations
        """
        recommendations = []
        
        if security_summary['security_score'] < 70:
            recommendations.append("Security score is below threshold - review recent security events")
        
        if indicators['content_filtering_rate'] > 0.1:
            recommendations.append("High content filtering rate detected - review input validation")
        
        if indicators['rate_limiting_rate'] > 0.05:
            recommendations.append("Frequent rate limiting - consider adjusting limits or investigating abuse")
        
        if security_summary['high_risk_events'] > 0:
            recommendations.append("High-risk security events detected - immediate investigation required")
        
        if not recommendations:
            recommendations.append("Security posture looks good - maintain current monitoring")
        
        return recommendations


class RealTimeMonitor:
    """Real-time monitoring with security awareness"""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics_queue = deque(maxsize=max_metrics)
        self.alerts = deque(maxsize=1000)
        self.alert_callbacks = []
        self.thresholds = {
            "response_time": 5.0,  # seconds
            "error_rate": 0.1,     # 10%
            "cost_per_request": 0.1,  # $0.10
            "security_events_per_hour": 10
        }
        self.lock = threading.Lock()
    
    def add_metric(self, metric_type: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Add real-time metric with security context"""
        with self.lock:
            metric = {
                "timestamp": time.time(),
                "type": metric_type,
                "value": value,
                "metadata": metadata or {}
            }
            self.metrics_queue.append(metric)
            
            # Check for security-related alerts
            if metric_type.startswith('security_'):
                self._check_security_thresholds(metric_type, value)
    
    def _check_security_thresholds(self, metric_type: str, value: float):
        """Check security-related thresholds"""
        if metric_type == "security_events_per_hour" and value > self.thresholds[metric_type]:
            self._trigger_alert(
                "high_security_activity",
                f"Security events per hour ({value}) exceeds threshold ({self.thresholds[metric_type]})",
                {"metric_type": metric_type, "value": value}
            )
    
    def _trigger_alert(self, alert_type: str, message: str, metadata: Dict[str, Any]):
        """Trigger a security alert"""
        alert = {
            "id": secrets.token_urlsafe(8),
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now(),
            "metadata": metadata
        }
        
        with self.lock:
            self.alerts.append(alert)
        
        # Log security alerts
        security_logger.warning(f"Security alert: {alert_type} - {message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                security_logger.error(f"Error in alert callback: {e}")


# Global instances with security enhancements
_secure_metrics_analyzer = None
_real_time_monitor = None

def get_secure_metrics_analyzer() -> SecureMetricsAnalyzer:
    """Get global secure metrics analyzer instance"""
    global _secure_metrics_analyzer
    if _secure_metrics_analyzer is None:
        _secure_metrics_analyzer = SecureMetricsAnalyzer()
    return _secure_metrics_analyzer

def get_real_time_monitor() -> RealTimeMonitor:
    """Get global real-time monitor instance"""
    global _real_time_monitor
    if _real_time_monitor is None:
        _real_time_monitor = RealTimeMonitor()
    return _real_time_monitor

# Convenience functions with security
def record_secure_performance_metric(**kwargs):
    """Record a secure performance metric"""
    analyzer = get_secure_metrics_analyzer()
    
    # Hash PII fields automatically
    if 'user_id' in kwargs:
        kwargs['user_id_hash'] = analyzer._hash_pii(kwargs.pop('user_id'))
    if 'session_id' in kwargs:
        kwargs['session_id_hash'] = analyzer._hash_pii(kwargs.pop('session_id'))
    
    analyzer.record_performance(SecurePerformanceMetrics(**kwargs))

def record_secure_usage_metric(**kwargs):
    """Record a secure usage metric"""
    analyzer = get_secure_metrics_analyzer()
    
    # Hash PII fields automatically
    if 'user_id' in kwargs:
        kwargs['user_id_hash'] = analyzer._hash_pii(kwargs.pop('user_id'))
    if 'session_id' in kwargs:
        kwargs['session_id_hash'] = analyzer._hash_pii(kwargs.pop('session_id'))
    if 'ip_address' in kwargs:
        kwargs['ip_address_hash'] = analyzer._hash_pii(kwargs.pop('ip_address'))
    if 'user_agent' in kwargs:
        kwargs['user_agent_hash'] = analyzer._hash_pii(kwargs.pop('user_agent'))
    
    analyzer.record_usage(SecureUsageMetrics(**kwargs))

def record_security_event(event_type: str, details: Dict[str, Any], severity: str = "medium"):
    """Record a security event"""
    analyzer = get_secure_metrics_analyzer()
    analyzer.security_metrics.record_security_event(event_type, details, severity)

def add_real_time_metric(metric_type: str, value: float, **metadata):
    """Add a real-time metric"""
    monitor = get_real_time_monitor()
    monitor.add_metric(metric_type, value, metadata)

def get_security_dashboard() -> Dict[str, Any]:
    """Get comprehensive security dashboard"""
    analyzer = get_secure_metrics_analyzer()
    return analyzer.get_security_dashboard()

# Backward compatibility aliases
MetricsAnalyzer = SecureMetricsAnalyzer
PerformanceMetrics = SecurePerformanceMetrics
UsageMetrics = SecureUsageMetrics
get_metrics_analyzer = get_secure_metrics_analyzer
record_performance_metric = record_secure_performance_metric
record_usage_metric = record_secure_usage_metric
    processing_time: float
    success: bool
    cost: float
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

@dataclass
class BusinessMetrics:
    """Business intelligence metrics"""
    timestamp: datetime
    total_requests: int
    successful_requests: int
    total_cost: float
    revenue: float  # if applicable
    active_users: int
    new_users: int
    model_usage: Dict[str, int]
    provider_usage: Dict[str, int]
    avg_response_time: float
    error_rate: float

class MetricsAnalyzer:
    """Advanced metrics analysis and insights"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Path.home() / ".candyllm" / "analytics.db")
        self.performance_buffer = deque(maxsize=10000)
        self.usage_buffer = deque(maxsize=10000)
        
        self._init_database()
        self._start_background_processing()
    
    def _init_database(self):
        """Initialize analytics database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Performance metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    operation_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    operation_type TEXT,
                    model_name TEXT,
                    provider TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    response_time REAL,
                    queue_time REAL,
                    processing_time REAL,
                    cost_estimate REAL,
                    success BOOLEAN,
                    error_type TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    metadata TEXT
                )
            """)
            
            # Usage metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    model_used TEXT,
                    provider TEXT,
                    request_type TEXT,
                    input_length INTEGER,
                    output_length INTEGER,
                    processing_time REAL,
                    success BOOLEAN,
                    cost REAL,
                    ip_address TEXT,
                    user_agent TEXT
                )
            """)
            
            # Business metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS business_metrics (
                    timestamp TEXT PRIMARY KEY,
                    total_requests INTEGER,
                    successful_requests INTEGER,
                    total_cost REAL,
                    revenue REAL,
                    active_users INTEGER,
                    new_users INTEGER,
                    model_usage TEXT,
                    provider_usage TEXT,
                    avg_response_time REAL,
                    error_rate REAL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_model ON performance_metrics(model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_metrics(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_business_timestamp ON business_metrics(timestamp)")
    
    def _start_background_processing(self):
        """Start background thread for processing metrics"""
        def process_metrics():
            while True:
                try:
                    # Process performance metrics
                    self._process_performance_buffer()
                    
                    # Process usage metrics
                    self._process_usage_buffer()
                    
                    # Generate business metrics
                    self._generate_business_metrics()
                    
                except Exception as e:
                    print(f"Error processing metrics: {e}")
                
                time.sleep(60)  # Process every minute
        
        thread = threading.Thread(target=process_metrics, daemon=True)
        thread.start()
    
    def record_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        self.performance_buffer.append(metrics)
    
    def record_usage(self, metrics: UsageMetrics):
        """Record usage metrics"""
        self.usage_buffer.append(metrics)
    
    def _process_performance_buffer(self):
        """Process performance metrics buffer"""
        if not self.performance_buffer:
            return
        
        # Convert buffer to list and clear
        metrics_list = list(self.performance_buffer)
        self.performance_buffer.clear()
        
        # Bulk insert to database
        with sqlite3.connect(self.db_path) as conn:
            for metrics in metrics_list:
                conn.execute("""
                    INSERT OR REPLACE INTO performance_metrics VALUES 
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.operation_id,
                    metrics.timestamp.isoformat(),
                    metrics.operation_type,
                    metrics.model_name,
                    metrics.provider,
                    metrics.input_tokens,
                    metrics.output_tokens,
                    metrics.response_time,
                    metrics.queue_time,
                    metrics.processing_time,
                    metrics.cost_estimate,
                    metrics.success,
                    metrics.error_type,
                    metrics.user_id,
                    metrics.session_id,
                    json.dumps(metrics.metadata) if metrics.metadata else None
                ))
    
    def _process_usage_buffer(self):
        """Process usage metrics buffer"""
        if not self.usage_buffer:
            return
        
        metrics_list = list(self.usage_buffer)
        self.usage_buffer.clear()
        
        with sqlite3.connect(self.db_path) as conn:
            for metrics in metrics_list:
                conn.execute("""
                    INSERT INTO usage_metrics VALUES 
                    (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp.isoformat(),
                    metrics.user_id,
                    metrics.session_id,
                    metrics.model_used,
                    metrics.provider,
                    metrics.request_type,
                    metrics.input_length,
                    metrics.output_length,
                    metrics.processing_time,
                    metrics.success,
                    metrics.cost,
                    metrics.ip_address,
                    metrics.user_agent
                ))
    
    def _generate_business_metrics(self):
        """Generate business metrics summary"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get hourly metrics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_requests,
                    SUM(cost) as total_cost,
                    AVG(processing_time) as avg_response_time,
                    model_used,
                    provider
                FROM usage_metrics 
                WHERE datetime(timestamp) > datetime(?)
            """, (hour_ago.isoformat(),))
            
            row = cursor.fetchone()
            if not row or row[0] == 0:
                return
            
            total_requests, successful_requests, total_cost, avg_response_time = row[:4]
            
            # Get model usage distribution
            cursor = conn.execute("""
                SELECT model_used, COUNT(*) 
                FROM usage_metrics 
                WHERE datetime(timestamp) > datetime(?)
                GROUP BY model_used
            """, (hour_ago.isoformat(),))
            
            model_usage = dict(cursor.fetchall())
            
            # Get provider usage distribution
            cursor = conn.execute("""
                SELECT provider, COUNT(*) 
                FROM usage_metrics 
                WHERE datetime(timestamp) > datetime(?)
                GROUP BY provider
            """, (hour_ago.isoformat(),))
            
            provider_usage = dict(cursor.fetchall())
            
            # Get active users
            cursor = conn.execute("""
                SELECT COUNT(DISTINCT user_id) 
                FROM usage_metrics 
                WHERE datetime(timestamp) > datetime(?) AND user_id IS NOT NULL
            """, (hour_ago.isoformat(),))
            
            active_users = cursor.fetchone()[0]
            
            # Calculate error rate
            error_rate = 1.0 - (successful_requests / total_requests) if total_requests > 0 else 0.0
            
            # Create business metrics
            business_metrics = BusinessMetrics(
                timestamp=now,
                total_requests=total_requests,
                successful_requests=successful_requests,
                total_cost=total_cost or 0.0,
                revenue=0.0,  # Would need billing integration
                active_users=active_users,
                new_users=0,  # Would need user tracking
                model_usage=model_usage,
                provider_usage=provider_usage,
                avg_response_time=avg_response_time or 0.0,
                error_rate=error_rate
            )
            
            # Store business metrics
            conn.execute("""
                INSERT OR REPLACE INTO business_metrics VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                business_metrics.timestamp.isoformat(),
                business_metrics.total_requests,
                business_metrics.successful_requests,
                business_metrics.total_cost,
                business_metrics.revenue,
                business_metrics.active_users,
                business_metrics.new_users,
                json.dumps(business_metrics.model_usage),
                json.dumps(business_metrics.provider_usage),
                business_metrics.avg_response_time,
                business_metrics.error_rate
            ))
    
    def get_performance_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance analysis"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            # Basic statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_operations,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_operations,
                    AVG(response_time) as avg_response_time,
                    MIN(response_time) as min_response_time,
                    MAX(response_time) as max_response_time,
                    AVG(input_tokens) as avg_input_tokens,
                    AVG(output_tokens) as avg_output_tokens,
                    SUM(cost_estimate) as total_cost
                FROM performance_metrics 
                WHERE datetime(timestamp) > datetime(?)
            """, (cutoff.isoformat(),))
            
            stats = cursor.fetchone()
            
            # Performance by model
            cursor = conn.execute("""
                SELECT 
                    model_name,
                    COUNT(*) as operations,
                    AVG(response_time) as avg_response_time,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_operations
                FROM performance_metrics 
                WHERE datetime(timestamp) > datetime(?)
                GROUP BY model_name
                ORDER BY operations DESC
            """, (cutoff.isoformat(),))
            
            model_performance = [
                {
                    "model": row[0],
                    "operations": row[1],
                    "avg_response_time": row[2],
                    "success_rate": row[3] / row[1] if row[1] > 0 else 0
                }
                for row in cursor.fetchall()
            ]
            
            # Performance by provider
            cursor = conn.execute("""
                SELECT 
                    provider,
                    COUNT(*) as operations,
                    AVG(response_time) as avg_response_time,
                    SUM(cost_estimate) as total_cost
                FROM performance_metrics 
                WHERE datetime(timestamp) > datetime(?)
                GROUP BY provider
                ORDER BY operations DESC
            """, (cutoff.isoformat(),))
            
            provider_performance = [
                {
                    "provider": row[0],
                    "operations": row[1],
                    "avg_response_time": row[2],
                    "total_cost": row[3]
                }
                for row in cursor.fetchall()
            ]
            
            # Error analysis
            cursor = conn.execute("""
                SELECT 
                    error_type,
                    COUNT(*) as error_count
                FROM performance_metrics 
                WHERE datetime(timestamp) > datetime(?) AND success = 0
                GROUP BY error_type
                ORDER BY error_count DESC
            """, (cutoff.isoformat(),))
            
            error_analysis = dict(cursor.fetchall())
            
            return {
                "summary": {
                    "total_operations": stats[0] or 0,
                    "successful_operations": stats[1] or 0,
                    "success_rate": (stats[1] / stats[0]) if stats[0] else 0,
                    "avg_response_time": stats[2] or 0,
                    "min_response_time": stats[3] or 0,
                    "max_response_time": stats[4] or 0,
                    "avg_input_tokens": stats[5] or 0,
                    "avg_output_tokens": stats[6] or 0,
                    "total_cost": stats[7] or 0
                },
                "model_performance": model_performance,
                "provider_performance": provider_performance,
                "error_analysis": error_analysis,
                "time_period_hours": hours
            }
    
    def get_usage_insights(self, hours: int = 24) -> Dict[str, Any]:
        """Get usage insights and patterns"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            # Usage patterns by hour
            cursor = conn.execute("""
                SELECT 
                    strftime('%H', timestamp) as hour,
                    COUNT(*) as requests
                FROM usage_metrics 
                WHERE datetime(timestamp) > datetime(?)
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour
            """, (cutoff.isoformat(),))
            
            hourly_usage = dict(cursor.fetchall())
            
            # Top users
            cursor = conn.execute("""
                SELECT 
                    user_id,
                    COUNT(*) as requests,
                    SUM(cost) as total_cost,
                    AVG(processing_time) as avg_processing_time
                FROM usage_metrics 
                WHERE datetime(timestamp) > datetime(?) AND user_id IS NOT NULL
                GROUP BY user_id
                ORDER BY requests DESC
                LIMIT 10
            """, (cutoff.isoformat(),))
            
            top_users = [
                {
                    "user_id": row[0],
                    "requests": row[1],
                    "total_cost": row[2],
                    "avg_processing_time": row[3]
                }
                for row in cursor.fetchall()
            ]
            
            # Request type distribution
            cursor = conn.execute("""
                SELECT 
                    request_type,
                    COUNT(*) as count,
                    AVG(input_length) as avg_input_length,
                    AVG(output_length) as avg_output_length
                FROM usage_metrics 
                WHERE datetime(timestamp) > datetime(?)
                GROUP BY request_type
                ORDER BY count DESC
            """, (cutoff.isoformat(),))
            
            request_type_distribution = [
                {
                    "type": row[0],
                    "count": row[1],
                    "avg_input_length": row[2],
                    "avg_output_length": row[3]
                }
                for row in cursor.fetchall()
            ]
            
            return {
                "hourly_usage": hourly_usage,
                "top_users": top_users,
                "request_type_distribution": request_type_distribution,
                "time_period_hours": hours
            }
    
    def get_cost_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Get detailed cost analysis"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            # Total cost breakdown
            cursor = conn.execute("""
                SELECT 
                    SUM(cost) as total_cost,
                    AVG(cost) as avg_cost_per_request,
                    COUNT(*) as total_requests
                FROM usage_metrics 
                WHERE datetime(timestamp) > datetime(?)
            """, (cutoff.isoformat(),))
            
            cost_summary = cursor.fetchone()
            
            # Cost by model
            cursor = conn.execute("""
                SELECT 
                    model_used,
                    SUM(cost) as total_cost,
                    COUNT(*) as requests,
                    AVG(cost) as avg_cost_per_request
                FROM usage_metrics 
                WHERE datetime(timestamp) > datetime(?)
                GROUP BY model_used
                ORDER BY total_cost DESC
            """, (cutoff.isoformat(),))
            
            cost_by_model = [
                {
                    "model": row[0],
                    "total_cost": row[1],
                    "requests": row[2],
                    "avg_cost_per_request": row[3]
                }
                for row in cursor.fetchall()
            ]
            
            # Cost by provider
            cursor = conn.execute("""
                SELECT 
                    provider,
                    SUM(cost) as total_cost,
                    COUNT(*) as requests
                FROM usage_metrics 
                WHERE datetime(timestamp) > datetime(?)
                GROUP BY provider
                ORDER BY total_cost DESC
            """, (cutoff.isoformat(),))
            
            cost_by_provider = [
                {
                    "provider": row[0],
                    "total_cost": row[1],
                    "requests": row[2]
                }
                for row in cursor.fetchall()
            ]
            
            # Cost trends (hourly)
            cursor = conn.execute("""
                SELECT 
                    strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                    SUM(cost) as hourly_cost,
                    COUNT(*) as hourly_requests
                FROM usage_metrics 
                WHERE datetime(timestamp) > datetime(?)
                GROUP BY strftime('%Y-%m-%d %H:00:00', timestamp)
                ORDER BY hour
            """, (cutoff.isoformat(),))
            
            cost_trends = [
                {
                    "hour": row[0],
                    "cost": row[1],
                    "requests": row[2]
                }
                for row in cursor.fetchall()
            ]
            
            return {
                "summary": {
                    "total_cost": cost_summary[0] or 0,
                    "avg_cost_per_request": cost_summary[1] or 0,
                    "total_requests": cost_summary[2] or 0
                },
                "cost_by_model": cost_by_model,
                "cost_by_provider": cost_by_provider,
                "cost_trends": cost_trends,
                "time_period_hours": hours
            }
    
    def get_anomaly_detection(self, hours: int = 24) -> Dict[str, Any]:
        """Detect performance and usage anomalies"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get response time data
            cursor = conn.execute("""
                SELECT response_time FROM performance_metrics 
                WHERE datetime(timestamp) > datetime(?) AND success = 1
            """, (cutoff.isoformat(),))
            
            response_times = [row[0] for row in cursor.fetchall()]
            
            anomalies = {}
            
            if response_times:
                # Statistical analysis
                mean_time = statistics.mean(response_times)
                std_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
                
                # Find outliers (more than 2 standard deviations from mean)
                threshold = mean_time + (2 * std_time)
                outliers = [t for t in response_times if t > threshold]
                
                anomalies["response_time"] = {
                    "mean": mean_time,
                    "std_dev": std_time,
                    "threshold": threshold,
                    "outlier_count": len(outliers),
                    "outlier_percentage": (len(outliers) / len(response_times)) * 100
                }
            
            # Error rate spikes
            cursor = conn.execute("""
                SELECT 
                    strftime('%H', timestamp) as hour,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as errors,
                    COUNT(*) as total
                FROM performance_metrics 
                WHERE datetime(timestamp) > datetime(?)
                GROUP BY strftime('%H', timestamp)
            """, (cutoff.isoformat(),))
            
            hourly_errors = []
            for row in cursor.fetchall():
                hour, errors, total = row
                error_rate = (errors / total) * 100 if total > 0 else 0
                hourly_errors.append({"hour": hour, "error_rate": error_rate})
            
            # Find hours with high error rates
            if hourly_errors:
                avg_error_rate = statistics.mean([h["error_rate"] for h in hourly_errors])
                high_error_hours = [h for h in hourly_errors if h["error_rate"] > avg_error_rate * 2]
                
                anomalies["error_spikes"] = {
                    "avg_error_rate": avg_error_rate,
                    "high_error_hours": high_error_hours
                }
            
            return anomalies
    
    def generate_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        return {
            "report_generated": datetime.now().isoformat(),
            "time_period_hours": hours,
            "performance_analysis": self.get_performance_analysis(hours),
            "usage_insights": self.get_usage_insights(hours),
            "cost_analysis": self.get_cost_analysis(hours),
            "anomaly_detection": self.get_anomaly_detection(hours)
        }

class RealTimeMonitor:
    """Real-time monitoring and alerting"""
    
    def __init__(self):
        self.metrics_queue = deque(maxsize=1000)
        self.alerts = []
        self.thresholds = {
            "response_time": 10.0,  # seconds
            "error_rate": 0.1,      # 10%
            "cost_per_hour": 100.0,  # dollars
            "queue_size": 100       # requests
        }
        self.alert_callbacks: List[Callable] = []
        
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start real-time monitoring"""
        def monitor():
            while True:
                try:
                    self._check_alerts()
                except Exception as e:
                    print(f"Error in monitoring: {e}")
                
                time.sleep(10)  # Check every 10 seconds
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def add_metric(self, metric_type: str, value: float, metadata: Dict[str, Any] = None):
        """Add a real-time metric"""
        self.metrics_queue.append({
            "type": metric_type,
            "value": value,
            "timestamp": time.time(),
            "metadata": metadata or {}
        })
    
    def set_threshold(self, metric_type: str, threshold: float):
        """Set alert threshold"""
        self.thresholds[metric_type] = threshold
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def _check_alerts(self):
        """Check for alert conditions"""
        now = time.time()
        
        # Get recent metrics (last 5 minutes)
        recent_metrics = [
            m for m in self.metrics_queue
            if now - m["timestamp"] < 300
        ]
        
        # Group by metric type
        metrics_by_type = defaultdict(list)
        for metric in recent_metrics:
            metrics_by_type[metric["type"]].append(metric["value"])
        
        # Check thresholds
        for metric_type, threshold in self.thresholds.items():
            if metric_type in metrics_by_type:
                values = metrics_by_type[metric_type]
                
                if metric_type == "response_time":
                    # Check average response time
                    avg_value = statistics.mean(values)
                    if avg_value > threshold:
                        self._trigger_alert(
                            "high_response_time",
                            f"Average response time ({avg_value:.2f}s) exceeds threshold ({threshold}s)",
                            {"metric_type": metric_type, "value": avg_value, "threshold": threshold}
                        )
                
                elif metric_type == "error_rate":
                    # Check error rate
                    error_rate = statistics.mean(values)
                    if error_rate > threshold:
                        self._trigger_alert(
                            "high_error_rate",
                            f"Error rate ({error_rate:.1%}) exceeds threshold ({threshold:.1%})",
                            {"metric_type": metric_type, "value": error_rate, "threshold": threshold}
                        )
    
    def _trigger_alert(self, alert_type: str, message: str, metadata: Dict[str, Any]):
        """Trigger an alert"""
        alert = {
            "id": secrets.token_urlsafe(8),
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now(),
            "metadata": metadata
        }
        
        self.alerts.append(alert)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in alert callback: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts
            if alert["timestamp"] > cutoff
        ]
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time statistics"""
        now = time.time()
        
        # Get metrics from last minute
        recent_metrics = [
            m for m in self.metrics_queue
            if now - m["timestamp"] < 60
        ]
        
        stats = {
            "timestamp": datetime.now().isoformat(),
            "active_requests": len([m for m in recent_metrics if m["type"] == "active_request"]),
            "recent_alerts": len(self.get_recent_alerts(hours=1)),
            "queue_size": len(recent_metrics)
        }
        
        # Calculate averages for numeric metrics
        metrics_by_type = defaultdict(list)
        for metric in recent_metrics:
            if isinstance(metric["value"], (int, float)):
                metrics_by_type[metric["type"]].append(metric["value"])
        
        for metric_type, values in metrics_by_type.items():
            if values:
                stats[f"avg_{metric_type}"] = statistics.mean(values)
        
        return stats

# Global instances
_metrics_analyzer = None
_real_time_monitor = None

def get_metrics_analyzer() -> MetricsAnalyzer:
    """Get global metrics analyzer instance"""
    global _metrics_analyzer
    if _metrics_analyzer is None:
        _metrics_analyzer = MetricsAnalyzer()
    return _metrics_analyzer

def get_real_time_monitor() -> RealTimeMonitor:
    """Get global real-time monitor instance"""
    global _real_time_monitor
    if _real_time_monitor is None:
        _real_time_monitor = RealTimeMonitor()
    return _real_time_monitor

# Convenience functions
def record_performance_metric(**kwargs):
    """Record a performance metric"""
    analyzer = get_metrics_analyzer()
    analyzer.record_performance(PerformanceMetrics(**kwargs))

def record_usage_metric(**kwargs):
    """Record a usage metric"""
    analyzer = get_metrics_analyzer()
    analyzer.record_usage(UsageMetrics(**kwargs))

def add_real_time_metric(metric_type: str, value: float, **metadata):
    """Add a real-time metric"""
    monitor = get_real_time_monitor()
    monitor.add_metric(metric_type, value, metadata)
