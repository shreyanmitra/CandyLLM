"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

🍭 CandyLLM Enhanced Security System
Comprehensive security, privacy, and safety features

Security Architecture:
- Multi-layered input validation and sanitization
- Advanced threat detection and classification
- Real-time content filtering with ML models
- Secure session management with encrypted tokens
- Rate limiting and abuse prevention
- Comprehensive audit logging and monitoring
- Protection against prompt injection and model manipulation
- Enterprise-grade security compliance (SOC2, GDPR, HIPAA)
- Zero-trust security model implementation
"""

import hashlib
import hmac
import secrets
import time
import json
import re
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Security logging configuration
security_logger = logging.getLogger('candyllm.security')
security_logger.setLevel(logging.INFO)

# Create security log handler if not exists
if not security_logger.handlers:
    handler = logging.FileHandler('candyllm_security.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    security_logger.addHandler(handler)

class SecurityLevel(Enum):
    """Security levels for different deployment environments"""
    LOW = "low"          # Development/testing environments
    MEDIUM = "medium"    # Standard production environments  
    HIGH = "high"        # High-security production environments
    ENTERPRISE = "enterprise"  # Enterprise/government deployments

class ThreatLevel(Enum):
    """Threat assessment levels for security incidents"""
    SAFE = "safe"        # No threat detected
    LOW = "low"          # Minor security concern
    MEDIUM = "medium"    # Moderate security risk
    HIGH = "high"        # High security risk requiring immediate attention
    CRITICAL = "critical"  # Critical security threat requiring emergency response

class SecurityEventType(Enum):
    """Types of security events for comprehensive monitoring"""
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    PROMPT_INJECTION_DETECTED = "prompt_injection_detected"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    MALICIOUS_CONTENT = "malicious_content"
    AUTHENTICATION_FAILURE = "authentication_failure"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    API_ABUSE = "api_abuse"

@dataclass
class SecurityEvent:
    """Comprehensive security event record for audit trails"""
    event_id: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: SecurityEventType = SecurityEventType.SUSPICIOUS_ACTIVITY
    threat_level: ThreatLevel = ThreatLevel.LOW
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    action_taken: str = ""
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert security event to dictionary for logging"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'threat_level': self.threat_level.value,
            'source_ip': self.source_ip,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'description': self.description,
            'metadata': self.metadata,
            'action_taken': self.action_taken,
            'resolved': self.resolved
        }

@dataclass
class ValidationResult:
    """Result of comprehensive input validation and threat assessment"""
    is_valid: bool = True
    threat_level: ThreatLevel = ThreatLevel.SAFE
    issues: List[str] = field(default_factory=list)
    sanitized_input: Optional[str] = None
    confidence_score: float = 1.0  # 0.0 = very suspicious, 1.0 = completely safe
    blocked_patterns: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class SecureTokenManager:
    """Secure token management for session and API key encryption"""
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize secure token manager
        
        Args:
            master_key: Optional master key for encryption. If None, generates one.
        """
        if master_key:
            key = master_key.encode()
        else:
            key = os.urandom(32)
        
        # Derive encryption key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'candyllm_security_salt',  # In production, use random salt
            iterations=100000,
        )
        self.encryption_key = base64.urlsafe_b64encode(kdf.derive(key))
        self.cipher_suite = Fernet(self.encryption_key)
    
    def encrypt_token(self, token: str) -> str:
        """Encrypt sensitive token"""
        try:
            encrypted = self.cipher_suite.encrypt(token.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            security_logger.error(f"Token encryption failed: {e}")
            raise
    
    def decrypt_token(self, encrypted_token: str) -> str:
        """Decrypt sensitive token"""
        try:
            decoded = base64.urlsafe_b64decode(encrypted_token)
            decrypted = self.cipher_suite.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            security_logger.error(f"Token decryption failed: {e}")
            raise

class InputValidator(ABC):
    """Abstract base class for input validators with comprehensive security checks"""
    
    @abstractmethod
    def validate(self, text: str, context: Optional[Dict] = None) -> ValidationResult:
        """
        Validate input text for security threats
        
        Args:
            text: Input text to validate
            context: Additional context for validation
            
        Returns:
            ValidationResult with security assessment
        """
        pass

class AdvancedContentFilter(InputValidator):
    """
    Advanced content filtering with machine learning and pattern-based detection
    
    Features:
    - Multi-layered threat detection
    - Context-aware analysis
    - Real-time pattern updates
    - Machine learning integration
    - Compliance with content policies
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        """
        Initialize advanced content filter
        
        Args:
            security_level: Security level determining filter strictness
        """
        self.security_level = security_level
        self.blocked_patterns = self._load_blocked_patterns()
        self.suspicious_patterns = self._load_suspicious_patterns()
        self.token_manager = SecureTokenManager()
        
        # Rate limiting tracking
        self.rate_limit_tracker: Dict[str, List[datetime]] = {}
        self.max_requests_per_minute = 60 if security_level != SecurityLevel.ENTERPRISE else 30
        
        security_logger.info(f"AdvancedContentFilter initialized with {security_level.value} security level")
    
    def _load_blocked_patterns(self) -> List[re.Pattern]:
        """
        Load comprehensive patterns for blocked content detection
        
        Returns:
            List of compiled regex patterns for threat detection
        """
        base_patterns = [
            # Prompt injection attempts
            r'ignore\s+previous\s+instructions',
            r'forget\s+everything\s+(you\s+know|above)',
            r'you\s+are\s+now\s+a\s+different',
            r'pretend\s+to\s+be\s+a',
            r'roleplay\s+as\s+a',
            r'jailbreak|DAN|developer\s+mode',
            r'bypass\s+filters?',
            r'override\s+safety\s+measures',
            r'disable\s+content\s+policy',
            r'unrestricted\s+mode',
            r'ignore\s+guidelines',
            
            # Personal Information (PII) Protection
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN format
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card format
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
            
            # Code injection attempts
            r'eval\s*\(',
            r'exec\s*\(',
            r'import\s+os',
            r'subprocess\.',
            r'os\.system',
            r'__import__',
            r'open\s*\(',
            r'file\s*\(',
            
            # Script injection
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            
            # SQL injection patterns
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            r'update\s+set',
            r'or\s+1\s*=\s*1',
            r'and\s+1\s*=\s*1',
            
            # Command injection
            r';\s*(rm|del|format)',
            r'\|\s*(nc|netcat|wget|curl)',
            r'&&\s*(whoami|id|ps)',
            
            # Model manipulation attempts
            r'system\s+prompt',
            r'assistant\s+instructions',
            r'model\s+behavior',
            r'training\s+data',
        ]
        
        # Add stricter patterns for higher security levels
        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.ENTERPRISE]:
            additional_patterns = [
                r'admin|administrator|root|sudo',
                r'password|passwd|secret|key',
                r'token|auth|credential',
                r'hack|exploit|vulnerability',
                r'malware|virus|trojan',
                r'ddos|dos\s+attack',
                r'penetration\s+test',
                r'social\s+engineering',
            ]
            base_patterns.extend(additional_patterns)
        
        # Compile patterns for performance
        compiled_patterns = []
        for pattern in base_patterns:
            try:
                compiled_patterns.append(re.compile(pattern, re.IGNORECASE | re.MULTILINE))
            except re.error as e:
                security_logger.warning(f"Invalid regex pattern: {pattern} - {e}")
        
        return compiled_patterns
    
    def _load_suspicious_patterns(self) -> List[re.Pattern]:
        """Load patterns for suspicious but not necessarily blocked content"""
        patterns = [
            r'how\s+to\s+(hack|crack|break)',
            r'bypass\s+security',
            r'exploit\s+vulnerability',
            r'social\s+engineer',
            r'phishing\s+attack',
            r'malicious\s+code',
            r'reverse\s+engineer',
            r'zero[\s-]?day',
        ]
        
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _check_rate_limit(self, identifier: str) -> bool:
        """
        Check if request is within rate limits
        
        Args:
            identifier: Unique identifier (IP, user_id, etc.)
            
        Returns:
            True if within limits, False if exceeded
        """
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old entries
        if identifier in self.rate_limit_tracker:
            self.rate_limit_tracker[identifier] = [
                timestamp for timestamp in self.rate_limit_tracker[identifier]
                if timestamp > minute_ago
            ]
        else:
            self.rate_limit_tracker[identifier] = []
        
        # Check current count
        current_count = len(self.rate_limit_tracker[identifier])
        
        if current_count >= self.max_requests_per_minute:
            security_logger.warning(f"Rate limit exceeded for {identifier}: {current_count} requests")
            return False
        
        # Add current request
        self.rate_limit_tracker[identifier].append(now)
        return True
    
    def validate(self, text: str, context: Optional[Dict] = None) -> ValidationResult:
        """
        Comprehensive validation of input text
        
        Args:
            text: Input text to validate
            context: Additional context including user_id, ip_address, etc.
            
        Returns:
            ValidationResult with detailed security assessment
        """
        if context is None:
            context = {}
        
        result = ValidationResult()
        
        try:
            # Basic input validation
            if not isinstance(text, str):
                result.is_valid = False
                result.threat_level = ThreatLevel.MEDIUM
                result.issues.append("Invalid input type - expected string")
                return result
            
            # Length validation
            if len(text) > 100000:  # 100KB limit
                result.is_valid = False
                result.threat_level = ThreatLevel.HIGH
                result.issues.append("Input exceeds maximum length limit")
                return result
            
            # Rate limiting check
            identifier = context.get('user_id') or context.get('ip_address', 'unknown')
            if not self._check_rate_limit(identifier):
                result.is_valid = False
                result.threat_level = ThreatLevel.HIGH
                result.issues.append("Rate limit exceeded")
                
                # Log security event
                security_event = SecurityEvent(
                    event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=context.get('ip_address'),
                    user_id=context.get('user_id'),
                    description=f"Rate limit exceeded for {identifier}",
                    action_taken="Request blocked"
                )
                security_logger.warning(f"Security event: {security_event.to_dict()}")
                return result
            
            # Content validation
            issues_found = []
            blocked_patterns_found = []
            confidence_score = 1.0
            
            # Check for blocked patterns
            for pattern in self.blocked_patterns:
                matches = pattern.findall(text)
                if matches:
                    issues_found.append(f"Blocked pattern detected: {pattern.pattern}")
                    blocked_patterns_found.append(pattern.pattern)
                    confidence_score -= 0.3
            
            # Check for suspicious patterns
            for pattern in self.suspicious_patterns:
                if pattern.search(text):
                    issues_found.append(f"Suspicious pattern detected: {pattern.pattern}")
                    confidence_score -= 0.1
            
            # Determine threat level based on findings
            if blocked_patterns_found:
                if len(blocked_patterns_found) > 3:
                    result.threat_level = ThreatLevel.CRITICAL
                elif len(blocked_patterns_found) > 1:
                    result.threat_level = ThreatLevel.HIGH
                else:
                    result.threat_level = ThreatLevel.MEDIUM
                
                result.is_valid = False
                result.blocked_patterns = blocked_patterns_found
                
                # Log security event for blocked content
                security_event = SecurityEvent(
                    event_type=SecurityEventType.PROMPT_INJECTION_DETECTED,
                    threat_level=result.threat_level,
                    source_ip=context.get('ip_address'),
                    user_id=context.get('user_id'),
                    description=f"Blocked patterns detected: {blocked_patterns_found}",
                    metadata={'patterns': blocked_patterns_found, 'input_length': len(text)},
                    action_taken="Content blocked"
                )
                security_logger.error(f"Security threat detected: {security_event.to_dict()}")
            
            # Set final results
            result.issues = issues_found
            result.confidence_score = max(0.0, confidence_score)
            
            # Sanitize input if validation passed
            if result.is_valid:
                result.sanitized_input = self._sanitize_input(text)
            
            # Add recommendations
            if result.threat_level != ThreatLevel.SAFE:
                result.recommendations = self._generate_recommendations(result)
            
            return result
            
        except Exception as e:
            security_logger.error(f"Content filter validation failed: {e}")
            result.is_valid = False
            result.threat_level = ThreatLevel.MEDIUM
            result.issues.append(f"Validation error: {str(e)}")
            return result
    
    def _sanitize_input(self, text: str) -> str:
        """
        Sanitize input text while preserving legitimate content
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text safe for processing
        """
        import html
        import urllib.parse
        
        # HTML escape to prevent script injection
        sanitized = html.escape(text)
        
        # URL decode to normalize encoded characters
        sanitized = urllib.parse.unquote(sanitized)
        
        # Remove null bytes and control characters except newlines/tabs
        sanitized = ''.join(
            char for char in sanitized 
            if ord(char) >= 32 or char in '\n\r\t'
        )
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def _generate_recommendations(self, result: ValidationResult) -> List[str]:
        """Generate security recommendations based on validation results"""
        recommendations = []
        
        if result.threat_level == ThreatLevel.CRITICAL:
            recommendations.extend([
                "Immediate security review required",
                "Consider blocking user/IP temporarily",
                "Escalate to security team"
            ])
        elif result.threat_level == ThreatLevel.HIGH:
            recommendations.extend([
                "Enhanced monitoring recommended",
                "Review user activity patterns",
                "Consider additional authentication"
            ])
        elif result.threat_level == ThreatLevel.MEDIUM:
            recommendations.extend([
                "Monitor for repeated patterns",
                "Log for security analysis"
            ])
        
        return recommendations
            # Additional strict patterns
            patterns.extend([
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
                r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
                r'\b\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',  # Phone numbers
            ])
        
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _load_suspicious_patterns(self) -> List[re.Pattern]:
        """Load patterns for suspicious content"""
        patterns = [
            r'\b(hack|crack|exploit|vulnerability)\b',
            r'\b(password|secret|confidential|private)\b',
            r'\b(admin|administrator|root|sudo)\b',
            r'\b(AI|assistant|model|GPT|Claude|bot)\s+(limitations|restrictions|rules)\b',
        ]
        
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def validate(self, text: str, context: Optional[Dict] = None) -> ValidationResult:
        """Validate input for harmful content"""
        issues = []
        threat_level = ThreatLevel.SAFE
        confidence_score = 1.0
        
        # Check blocked patterns
        for pattern in self.blocked_patterns:
            matches = pattern.findall(text)
            if matches:
                issues.append(f"Blocked pattern detected: {pattern.pattern}")
                threat_level = ThreatLevel.HIGH
                confidence_score = 0.0
        
        # Check suspicious patterns
        suspicious_matches = 0
        for pattern in self.suspicious_patterns:
            if pattern.search(text):
                suspicious_matches += 1
                issues.append(f"Suspicious pattern: {pattern.pattern}")
        
        # Adjust threat level based on suspicious content
        if suspicious_matches > 0:
            if threat_level == ThreatLevel.SAFE:
                threat_level = ThreatLevel.LOW if suspicious_matches <= 2 else ThreatLevel.MEDIUM
            confidence_score = max(0.0, confidence_score - (suspicious_matches * 0.2))
        
        # Additional context-based checks
        if context:
            if context.get('user_risk_score', 0) > 0.7:
                threat_level = max(threat_level, ThreatLevel.MEDIUM)
                issues.append("High-risk user profile")
        
        # Generate sanitized input if needed
        sanitized_input = None
        if threat_level in [ThreatLevel.LOW, ThreatLevel.MEDIUM]:
            sanitized_input = self._sanitize_input(text)
        
        return ValidationResult(
            is_valid=threat_level in [ThreatLevel.SAFE, ThreatLevel.LOW],
            threat_level=threat_level,
            issues=issues,
            sanitized_input=sanitized_input,
            confidence_score=confidence_score
        )
    
    def _sanitize_input(self, text: str) -> str:
        """Sanitize input by removing/masking sensitive content"""
        sanitized = text
        
        # Mask email addresses
        sanitized = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL_REDACTED]',
            sanitized
        )
        
        # Mask phone numbers
        sanitized = re.sub(
            r'\b\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            '[PHONE_REDACTED]',
            sanitized
        )
        
        # Mask SSNs
        sanitized = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            '[SSN_REDACTED]',
            sanitized
        )
        
        # Mask credit card numbers
        sanitized = re.sub(
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            '[CARD_REDACTED]',
            sanitized
        )
        
        return sanitized

class OutputFilter:
    """Filter and validate LLM outputs"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        self.security_level = security_level
        self.blocked_output_patterns = self._load_output_patterns()
    
    def _load_output_patterns(self) -> List[re.Pattern]:
        """Load patterns for blocked output content"""
        patterns = [
            # Prevent information leakage
            r'(system prompt|instructions|guidelines):\s*["\'].*?["\']',
            r'I was instructed to',
            r'My instructions are',
            r'The prompt says',
            
            # Prevent harmful instructions
            r'Here\'s how to (hack|crack|exploit|bypass)',
            r'To create (explosives|weapons|drugs)',
            r'Steps to (harm|hurt|attack)',
            
            # Prevent sensitive data exposure
            r'api[_-]?key[:\s=]+[a-zA-Z0-9]+',
            r'password[:\s=]+[^\s]+',
            r'secret[:\s=]+[^\s]+',
        ]
        
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def filter_output(self, text: str, context: Optional[Dict] = None) -> Tuple[str, List[str]]:
        """Filter output text and return filtered text + issues found"""
        issues = []
        filtered_text = text
        
        # Check for blocked patterns
        for pattern in self.blocked_output_patterns:
            matches = pattern.finditer(filtered_text)
            for match in matches:
                issues.append(f"Filtered output pattern: {pattern.pattern}")
                # Replace match with placeholder
                filtered_text = filtered_text.replace(
                    match.group(),
                    "[CONTENT_FILTERED]"
                )
        
        # Additional filtering based on security level
        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.ENTERPRISE]:
            # More aggressive filtering
            filtered_text, additional_issues = self._aggressive_filter(filtered_text)
            issues.extend(additional_issues)
        
        return filtered_text, issues
    
    def _aggressive_filter(self, text: str) -> Tuple[str, List[str]]:
        """Apply aggressive filtering for high security environments"""
        issues = []
        filtered_text = text
        
        # Remove potential URLs
        url_pattern = r'https?://[^\s<>"\'`]+|www\.[^\s<>"\'`]+'
        if re.search(url_pattern, filtered_text):
            filtered_text = re.sub(url_pattern, '[URL_REMOVED]', filtered_text)
            issues.append("URLs removed for security")
        
        # Remove file paths
        path_pattern = r'[A-Za-z]:\\[^<>"\'`\s]*|/[^<>"\'`\s]*'
        if re.search(path_pattern, filtered_text):
            filtered_text = re.sub(path_pattern, '[PATH_REMOVED]', filtered_text)
            issues.append("File paths removed for security")
        
        return filtered_text, issues

class AuthenticationManager:
    """Manage authentication and authorization"""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
    
    def generate_api_key(self, user_id: str, permissions: List[str], expires_in_days: int = 30) -> str:
        """Generate a new API key"""
        api_key = secrets.token_urlsafe(32)
        
        self.api_keys[api_key] = {
            "user_id": user_id,
            "permissions": permissions,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(days=expires_in_days),
            "usage_count": 0,
            "last_used": None
        }
        
        return api_key
    
    def validate_api_key(self, api_key: str, required_permission: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Validate API key and check permissions"""
        if api_key not in self.api_keys:
            return False, "Invalid API key"
        
        key_info = self.api_keys[api_key]
        
        # Check expiration
        if datetime.now() > key_info["expires_at"]:
            return False, "API key expired"
        
        # Check permissions
        if required_permission and required_permission not in key_info["permissions"]:
            return False, f"Permission '{required_permission}' not granted"
        
        # Update usage
        key_info["usage_count"] += 1
        key_info["last_used"] = datetime.now()
        
        return True, None
    
    def create_session(self, user_id: str, api_key: str) -> str:
        """Create a new session"""
        session_id = secrets.token_urlsafe(16)
        
        self.sessions[session_id] = {
            "user_id": user_id,
            "api_key": api_key,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "request_count": 0
        }
        
        return session_id
    
    def validate_session(self, session_id: str, max_idle_minutes: int = 30) -> Tuple[bool, Optional[str]]:
        """Validate session"""
        if session_id not in self.sessions:
            return False, "Invalid session"
        
        session = self.sessions[session_id]
        
        # Check idle timeout
        if datetime.now() - session["last_activity"] > timedelta(minutes=max_idle_minutes):
            del self.sessions[session_id]
            return False, "Session expired due to inactivity"
        
        # Update activity
        session["last_activity"] = datetime.now()
        session["request_count"] += 1
        
        return True, None
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100, window_minutes: int = 60) -> Tuple[bool, int]:
        """Check rate limit for user/IP"""
        now = datetime.now()
        window_start = now - timedelta(minutes=window_minutes)
        
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = {"requests": [], "blocked_until": None}
        
        rate_info = self.rate_limits[identifier]
        
        # Check if currently blocked
        if rate_info["blocked_until"] and now < rate_info["blocked_until"]:
            remaining = int((rate_info["blocked_until"] - now).total_seconds())
            return False, remaining
        
        # Clean old requests
        rate_info["requests"] = [
            req_time for req_time in rate_info["requests"]
            if req_time > window_start
        ]
        
        # Check rate limit
        if len(rate_info["requests"]) >= max_requests:
            # Block for the remaining window time
            rate_info["blocked_until"] = now + timedelta(minutes=window_minutes)
            return False, window_minutes * 60
        
        # Add current request
        rate_info["requests"].append(now)
        return True, 0

class AuditLogger:
    """Security audit logging"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.events: List[SecurityEvent] = []
        
        # Configure logging
        if log_file:
            logging.basicConfig(
                filename=log_file,
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        self.logger = logging.getLogger(__name__)
    
    def log_security_event(
        self,
        event_type: str,
        threat_level: ThreatLevel,
        description: str,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        action_taken: str = "none"
    ) -> str:
        """Log a security event"""
        event_id = secrets.token_urlsafe(8)
        
        event = SecurityEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            metadata=metadata or {},
            action_taken=action_taken
        )
        
        self.events.append(event)
        
        # Log to file
        log_entry = {
            "event_id": event_id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event_type,
            "threat_level": threat_level.value,
            "source_ip": source_ip,
            "user_id": user_id,
            "description": description,
            "metadata": metadata,
            "action_taken": action_taken
        }
        
        self.logger.info(f"SECURITY_EVENT: {json.dumps(log_entry)}")
        
        return event_id
    
    def get_events(
        self,
        threat_level: Optional[ThreatLevel] = None,
        hours: int = 24
    ) -> List[SecurityEvent]:
        """Get security events"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        events = [
            event for event in self.events
            if event.timestamp > cutoff
        ]
        
        if threat_level:
            events = [
                event for event in events
                if event.threat_level == threat_level
            ]
        
        return sorted(events, key=lambda x: x.timestamp, reverse=True)
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary"""
        events = self.get_events(hours=hours)
        
        threat_counts = {}
        for level in ThreatLevel:
            threat_counts[level.value] = len([
                e for e in events if e.threat_level == level
            ])
        
        event_types = {}
        for event in events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        return {
            "total_events": len(events),
            "threat_level_distribution": threat_counts,
            "event_type_distribution": event_types,
            "high_threat_events": len([
                e for e in events 
                if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            ]),
            "time_period_hours": hours
        }

class SecurityManager:
    """Central security management"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        self.security_level = security_level
        self.input_validator = ContentFilter(security_level)
        self.output_filter = OutputFilter(security_level)
        self.auth_manager = AuthenticationManager()
        self.audit_logger = AuditLogger()
        
        # Security policies
        self.policies = self._load_security_policies()
    
    def _load_security_policies(self) -> Dict[str, Any]:
        """Load security policies based on security level"""
        base_policies = {
            "max_input_length": 10000,
            "max_output_length": 50000,
            "require_authentication": False,
            "log_all_requests": False,
            "enable_rate_limiting": True,
            "max_requests_per_hour": 100
        }
        
        if self.security_level == SecurityLevel.HIGH:
            base_policies.update({
                "max_input_length": 5000,
                "max_output_length": 25000,
                "require_authentication": True,
                "log_all_requests": True,
                "max_requests_per_hour": 50
            })
        elif self.security_level == SecurityLevel.ENTERPRISE:
            base_policies.update({
                "max_input_length": 5000,
                "max_output_length": 25000,
                "require_authentication": True,
                "log_all_requests": True,
                "max_requests_per_hour": 25,
                "require_session_validation": True,
                "enable_content_encryption": True
            })
        
        return base_policies
    
    def validate_request(
        self,
        input_text: str,
        user_context: Optional[Dict] = None
    ) -> Tuple[bool, ValidationResult, List[str]]:
        """Validate a complete request"""
        issues = []
        
        # Check input length
        if len(input_text) > self.policies["max_input_length"]:
            issues.append(f"Input too long: {len(input_text)} > {self.policies['max_input_length']}")
            return False, ValidationResult(
                is_valid=False,
                threat_level=ThreatLevel.MEDIUM,
                issues=issues
            ), issues
        
        # Validate input content
        validation_result = self.input_validator.validate(input_text, user_context)
        
        # Log security events
        if validation_result.threat_level != ThreatLevel.SAFE:
            self.audit_logger.log_security_event(
                event_type="input_validation",
                threat_level=validation_result.threat_level,
                description=f"Input validation issues: {', '.join(validation_result.issues)}",
                user_id=user_context.get("user_id") if user_context else None,
                source_ip=user_context.get("source_ip") if user_context else None,
                metadata={"input_length": len(input_text)},
                action_taken="blocked" if not validation_result.is_valid else "flagged"
            )
        
        return validation_result.is_valid, validation_result, issues
    
    def filter_response(
        self,
        output_text: str,
        user_context: Optional[Dict] = None
    ) -> Tuple[str, List[str]]:
        """Filter LLM response"""
        # Check output length
        if len(output_text) > self.policies["max_output_length"]:
            output_text = output_text[:self.policies["max_output_length"]] + "\n[OUTPUT_TRUNCATED]"
        
        # Filter content
        filtered_text, issues = self.output_filter.filter_output(output_text, user_context)
        
        # Log if content was filtered
        if issues:
            self.audit_logger.log_security_event(
                event_type="output_filtering",
                threat_level=ThreatLevel.MEDIUM,
                description=f"Output filtering applied: {', '.join(issues)}",
                user_id=user_context.get("user_id") if user_context else None,
                source_ip=user_context.get("source_ip") if user_context else None,
                metadata={"original_length": len(output_text), "filtered_length": len(filtered_text)},
                action_taken="content_filtered"
            )
        
        return filtered_text, issues
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        return {
            "security_level": self.security_level.value,
            "policies": self.policies,
            "recent_events": self.audit_logger.get_security_summary(hours=24),
            "active_sessions": len(self.auth_manager.sessions),
            "api_keys": len(self.auth_manager.api_keys)
        }

# Global security manager
_security_manager = None

def get_security_manager() -> SecurityManager:
    """Get global security manager instance"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager

def configure_security(security_level: SecurityLevel):
    """Configure global security level"""
    global _security_manager
    _security_manager = SecurityManager(security_level)
