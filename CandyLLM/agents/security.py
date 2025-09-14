"""
Agent Security Manager

Security wrapper system to apply CandyLLM security controls to all external
agentic frameworks, ensuring consistent security policies across providers.
"""

import uuid
import hashlib
import re
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from .base import ToolSpec, AgentSecurityLevel, AgentCapability


class SecurityAction(Enum):
    """Actions that can be taken by security policies"""
    ALLOW = "allow"
    DENY = "deny"
    QUARANTINE = "quarantine"
    SANITIZE = "sanitize"
    AUDIT = "audit"
    REQUIRE_APPROVAL = "require_approval"


class ThreatLevel(Enum):
    """Threat levels for security assessment"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityViolation:
    """Record of a security violation"""
    violation_id: str
    agent_id: str
    violation_type: str
    description: str
    threat_level: ThreatLevel
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class SecurityPolicy:
    """Security policy definition"""
    name: str
    description: str
    rules: List[Dict[str, Any]]
    applies_to: List[str] = field(default_factory=list)  # Provider names or agent types
    priority: int = 0
    enabled: bool = True


@dataclass
class ToolSecurityProfile:
    """Security profile for a tool"""
    tool_name: str
    risk_level: ThreatLevel
    allowed_agents: Set[str] = field(default_factory=set)
    blocked_agents: Set[str] = field(default_factory=set)
    parameter_validation: Dict[str, Any] = field(default_factory=dict)
    rate_limits: Dict[str, int] = field(default_factory=dict)  # calls per time period
    audit_level: str = "basic"
    requires_approval: bool = False


class ContentFilter:
    """Content filtering and sanitization"""
    
    def __init__(self):
        # Dangerous patterns to detect
        self.dangerous_patterns = [
            r'(?i)(?:rm|del|delete)\s+(?:-rf?\s+)?[\/\\]',  # File deletion
            r'(?i)(?:drop|truncate)\s+table',  # SQL injection
            r'(?i)eval\s*\(',  # Code evaluation
            r'(?i)exec\s*\(',  # Code execution
            r'(?i)__import__\s*\(',  # Python imports
            r'(?i)subprocess\.',  # Process execution
            r'(?i)os\.system',  # System commands
            r'(?i)shell\s*=\s*True',  # Shell execution
            r'(?i)password\s*[:=]\s*["\'][^"\']+["\']',  # Password exposure
            r'(?i)(?:api[_-]?key|token)\s*[:=]\s*["\'][^"\']+["\']',  # API key exposure
        ]
        
        # Sensitive data patterns
        self.sensitive_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',  # Phone
        ]
    
    def detect_threats(self, content: str) -> List[Dict[str, Any]]:
        """Detect potential security threats in content"""
        threats = []
        
        for pattern in self.dangerous_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                threats.append({
                    "type": "dangerous_pattern",
                    "pattern": pattern,
                    "match": match.group(),
                    "position": match.span(),
                    "severity": "high"
                })
        
        return threats
    
    def detect_sensitive_data(self, content: str) -> List[Dict[str, Any]]:
        """Detect sensitive data in content"""
        sensitive_data = []
        
        for pattern in self.sensitive_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                sensitive_data.append({
                    "type": "sensitive_data",
                    "pattern": pattern,
                    "match": match.group(),
                    "position": match.span()
                })
        
        return sensitive_data
    
    def sanitize_content(self, content: str) -> str:
        """Sanitize content by removing or masking sensitive information"""
        sanitized = content
        
        # Mask sensitive data
        for pattern in self.sensitive_patterns:
            if 'card' in pattern or r'\d{4}' in pattern:
                # Credit card - mask middle digits
                sanitized = re.sub(pattern, lambda m: f"{m.group()[:4]}****{m.group()[-4:]}", sanitized)
            elif 'ssn' in pattern.lower() or r'\d{3}-\d{2}-\d{4}' in pattern:
                # SSN - mask all but last 4
                sanitized = re.sub(pattern, "***-**-####", sanitized)
            elif '@' in pattern:
                # Email - mask domain
                sanitized = re.sub(pattern, lambda m: f"{m.group().split('@')[0]}@***", sanitized)
            else:
                # Generic masking
                sanitized = re.sub(pattern, "***", sanitized)
        
        return sanitized


class RateLimiter:
    """Rate limiting for agent and tool operations"""
    
    def __init__(self):
        self.call_counts: Dict[str, Dict[str, int]] = {}  # {agent_id: {operation: count}}
        self.last_reset: Dict[str, datetime] = {}
        self.limits: Dict[str, Dict[str, int]] = {}  # {agent_id: {operation: limit}}
    
    def set_limit(self, agent_id: str, operation: str, limit: int, window_minutes: int = 60):
        """Set rate limit for agent operation"""
        if agent_id not in self.limits:
            self.limits[agent_id] = {}
        self.limits[agent_id][operation] = limit
        
        # Store window info (simplified - would need more sophisticated tracking)
        self.limits[agent_id][f"{operation}_window"] = window_minutes
    
    def check_rate_limit(self, agent_id: str, operation: str) -> bool:
        """Check if operation is within rate limits"""
        if agent_id not in self.limits or operation not in self.limits[agent_id]:
            return True  # No limit set
        
        now = datetime.now()
        
        # Reset counters if window has passed
        if agent_id in self.last_reset:
            window_minutes = self.limits[agent_id].get(f"{operation}_window", 60)
            if now - self.last_reset[agent_id] > timedelta(minutes=window_minutes):
                self.call_counts[agent_id] = {}
                self.last_reset[agent_id] = now
        else:
            self.last_reset[agent_id] = now
        
        # Check current count
        if agent_id not in self.call_counts:
            self.call_counts[agent_id] = {}
        
        current_count = self.call_counts[agent_id].get(operation, 0)
        limit = self.limits[agent_id][operation]
        
        if current_count >= limit:
            return False
        
        # Increment counter
        self.call_counts[agent_id][operation] = current_count + 1
        return True


class AuditLogger:
    """Audit logging for agent operations"""
    
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("AgentSecurityAudit")
    
    def log_operation(self, agent_id: str, operation: str, details: Dict[str, Any]):
        """Log an agent operation"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "operation": operation,
            "details": details,
            "log_id": str(uuid.uuid4())
        }
        
        self.logs.append(log_entry)
        self.logger.info(f"Agent {agent_id} performed {operation}: {details}")
    
    def log_security_event(self, agent_id: str, event_type: str, 
                          severity: str, details: Dict[str, Any]):
        """Log a security event"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "event_type": "security_event",
            "security_event_type": event_type,
            "severity": severity,
            "details": details,
            "log_id": str(uuid.uuid4())
        }
        
        self.logs.append(log_entry)
        self.logger.warning(f"Security event for agent {agent_id}: {event_type} ({severity})")
    
    def get_agent_logs(self, agent_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get logs for a specific agent"""
        agent_logs = [log for log in self.logs if log.get("agent_id") == agent_id]
        return sorted(agent_logs, key=lambda x: x["timestamp"], reverse=True)[:limit]


class AgentSecurityManager:
    """
    Main security manager for the agentic provider system.
    
    Provides unified security controls across all external frameworks,
    ensuring consistent security policies and audit trails.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Core security components
        self.content_filter = ContentFilter()
        self.rate_limiter = RateLimiter()
        self.audit_logger = AuditLogger()
        
        # Security state
        self.violations: List[SecurityViolation] = []
        self.policies: Dict[str, SecurityPolicy] = {}
        self.tool_profiles: Dict[str, ToolSecurityProfile] = {}
        self.agent_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Default security policies
        self._initialize_default_policies()
    
    def register_agent(self, agent_id: str, provider: str, 
                      security_level: AgentSecurityLevel, 
                      capabilities: List[AgentCapability]):
        """Register an agent with security controls"""
        self.agent_profiles[agent_id] = {
            "provider": provider,
            "security_level": security_level,
            "capabilities": capabilities,
            "created_at": datetime.now(),
            "violations": [],
            "status": "active"
        }
        
        # Set default rate limits based on security level
        if security_level == AgentSecurityLevel.ENTERPRISE:
            self.rate_limiter.set_limit(agent_id, "tool_calls", 1000, 60)
            self.rate_limiter.set_limit(agent_id, "synthesis", 10, 60)
        elif security_level == AgentSecurityLevel.MONITORED:
            self.rate_limiter.set_limit(agent_id, "tool_calls", 100, 60)
            self.rate_limiter.set_limit(agent_id, "synthesis", 5, 60)
        else:
            self.rate_limiter.set_limit(agent_id, "tool_calls", 50, 60)
            self.rate_limiter.set_limit(agent_id, "synthesis", 2, 60)
        
        self.audit_logger.log_operation(agent_id, "agent_registered", {
            "provider": provider,
            "security_level": security_level.value,
            "capabilities": [cap.value for cap in capabilities]
        })
    
    def register_tool(self, agent_id: str, tool_spec: ToolSpec):
        """Register a tool with security validation"""
        # Create security profile for tool
        risk_level = self._assess_tool_risk(tool_spec)
        
        profile = ToolSecurityProfile(
            tool_name=tool_spec.name,
            risk_level=risk_level,
            parameter_validation=self._create_parameter_validation(tool_spec),
            requires_approval=risk_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        )
        
        self.tool_profiles[tool_spec.name] = profile
        
        # Log tool registration
        self.audit_logger.log_operation(agent_id, "tool_registered", {
            "tool_name": tool_spec.name,
            "risk_level": risk_level.value,
            "requires_approval": profile.requires_approval
        })
    
    def validate_tool_access(self, agent_id: str, tool_name: str) -> bool:
        """Validate if agent can access a tool"""
        if agent_id not in self.agent_profiles:
            return False
        
        profile = self.tool_profiles.get(tool_name)
        if not profile:
            return True  # No profile means no restrictions
        
        agent_profile = self.agent_profiles[agent_id]
        
        # Check blocked agents
        if agent_id in profile.blocked_agents:
            self._record_violation(agent_id, "blocked_tool_access", 
                                 f"Agent attempted to access blocked tool: {tool_name}")
            return False
        
        # Check allowed agents (if list exists)
        if profile.allowed_agents and agent_id not in profile.allowed_agents:
            self._record_violation(agent_id, "unauthorized_tool_access",
                                 f"Agent not authorized for tool: {tool_name}")
            return False
        
        # Check rate limits
        if not self.rate_limiter.check_rate_limit(agent_id, "tool_calls"):
            self._record_violation(agent_id, "rate_limit_exceeded",
                                 f"Rate limit exceeded for tool calls")
            return False
        
        # Check security level requirements
        if profile.risk_level == ThreatLevel.CRITICAL:
            if agent_profile["security_level"] != AgentSecurityLevel.ENTERPRISE:
                self._record_violation(agent_id, "insufficient_security_level",
                                     f"Critical tool requires enterprise security level")
                return False
        
        return True
    
    def validate_tool_inputs(self, tool_name: str, inputs: Dict[str, Any]) -> bool:
        """Validate tool inputs for security"""
        profile = self.tool_profiles.get(tool_name)
        if not profile:
            return True
        
        # Check parameter validation rules
        for param_name, validation_rules in profile.parameter_validation.items():
            if param_name in inputs:
                value = inputs[param_name]
                
                # Check data type
                expected_type = validation_rules.get("type")
                if expected_type and not isinstance(value, expected_type):
                    return False
                
                # Check string content
                if isinstance(value, str):
                    threats = self.content_filter.detect_threats(value)
                    if threats:
                        return False
                
                # Check value constraints
                min_val = validation_rules.get("min")
                max_val = validation_rules.get("max")
                if min_val is not None and value < min_val:
                    return False
                if max_val is not None and value > max_val:
                    return False
        
        return True
    
    def sanitize_output(self, content: str) -> str:
        """Sanitize agent output for security"""
        # Detect and mask sensitive data
        sanitized = self.content_filter.sanitize_content(content)
        
        # Remove dangerous patterns
        threats = self.content_filter.detect_threats(content)
        if threats:
            # For now, just log threats - could block or sanitize
            self.audit_logger.log_security_event(
                "system", "threat_detected_in_output", "medium",
                {"threats": threats, "content_length": len(content)}
            )
        
        return sanitized
    
    def validate_agent_prompt(self, agent_id: str, prompt: str) -> tuple[bool, str]:
        """Validate agent prompt for security"""
        # Check for threats in prompt
        threats = self.content_filter.detect_threats(prompt)
        if threats:
            self._record_violation(agent_id, "malicious_prompt",
                                 f"Dangerous patterns detected in prompt: {len(threats)} threats")
            return False, "Prompt contains potentially dangerous content"
        
        # Check prompt length
        max_length = self.config.get("max_prompt_length", 50000)
        if len(prompt) > max_length:
            return False, f"Prompt too long: {len(prompt)} > {max_length}"
        
        # Sanitize and return
        sanitized_prompt = self.content_filter.sanitize_content(prompt)
        return True, sanitized_prompt
    
    def get_security_status(self, agent_id: str) -> Dict[str, Any]:
        """Get security status for an agent"""
        if agent_id not in self.agent_profiles:
            return {"error": "Agent not found"}
        
        profile = self.agent_profiles[agent_id]
        recent_logs = self.audit_logger.get_agent_logs(agent_id, 10)
        
        return {
            "agent_id": agent_id,
            "security_level": profile["security_level"].value,
            "status": profile["status"],
            "violation_count": len(profile["violations"]),
            "recent_violations": profile["violations"][-5:],
            "recent_activity": len(recent_logs),
            "last_activity": recent_logs[0]["timestamp"] if recent_logs else None
        }
    
    def cleanup_agent(self, agent_id: str):
        """Clean up security data for an agent"""
        if agent_id in self.agent_profiles:
            del self.agent_profiles[agent_id]
        
        # Clean up rate limiter data
        if agent_id in self.rate_limiter.call_counts:
            del self.rate_limiter.call_counts[agent_id]
        if agent_id in self.rate_limiter.last_reset:
            del self.rate_limiter.last_reset[agent_id]
        if agent_id in self.rate_limiter.limits:
            del self.rate_limiter.limits[agent_id]
        
        self.audit_logger.log_operation(agent_id, "agent_cleanup", {})
    
    def _initialize_default_policies(self):
        """Initialize default security policies"""
        # High-risk tool policy
        high_risk_policy = SecurityPolicy(
            name="high_risk_tools",
            description="Policy for high-risk tools requiring approval",
            rules=[
                {"condition": "tool_risk_level", "value": "high", "action": "require_approval"},
                {"condition": "tool_risk_level", "value": "critical", "action": "require_approval"}
            ],
            priority=10
        )
        self.policies["high_risk_tools"] = high_risk_policy
        
        # Rate limiting policy
        rate_limit_policy = SecurityPolicy(
            name="rate_limiting",
            description="Standard rate limiting for all agents",
            rules=[
                {"condition": "security_level", "value": "unrestricted", "rate_limit": 1000},
                {"condition": "security_level", "value": "monitored", "rate_limit": 100},
                {"condition": "security_level", "value": "sandboxed", "rate_limit": 50}
            ],
            priority=5
        )
        self.policies["rate_limiting"] = rate_limit_policy
    
    def _assess_tool_risk(self, tool_spec: ToolSpec) -> ThreatLevel:
        """Assess risk level of a tool"""
        risk_score = 0
        
        # Check tool name for dangerous keywords
        dangerous_keywords = ["delete", "remove", "exec", "eval", "system", "shell", "subprocess"]
        if any(keyword in tool_spec.name.lower() for keyword in dangerous_keywords):
            risk_score += 3
        
        # Check description
        if any(keyword in tool_spec.description.lower() for keyword in dangerous_keywords):
            risk_score += 2
        
        # Check security policy
        policy_risk = tool_spec.security_policy.get("risk_level", "low")
        if policy_risk == "high":
            risk_score += 2
        elif policy_risk == "critical":
            risk_score += 3
        
        # Convert score to threat level
        if risk_score >= 5:
            return ThreatLevel.CRITICAL
        elif risk_score >= 3:
            return ThreatLevel.HIGH
        elif risk_score >= 1:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _create_parameter_validation(self, tool_spec: ToolSpec) -> Dict[str, Any]:
        """Create parameter validation rules for a tool"""
        validation = {}
        
        for param_name, param_def in tool_spec.parameters.items():
            if isinstance(param_def, dict):
                validation[param_name] = {
                    "type": param_def.get("type", str),
                    "required": param_def.get("required", False)
                }
                
                # Add constraints based on type
                if param_def.get("type") == "string":
                    validation[param_name]["max_length"] = 10000
                elif param_def.get("type") in ["integer", "number"]:
                    validation[param_name]["min"] = -1000000
                    validation[param_name]["max"] = 1000000
        
        return validation
    
    def _record_violation(self, agent_id: str, violation_type: str, description: str):
        """Record a security violation"""
        violation = SecurityViolation(
            violation_id=str(uuid.uuid4()),
            agent_id=agent_id,
            violation_type=violation_type,
            description=description,
            threat_level=ThreatLevel.MEDIUM,  # Default level
            timestamp=datetime.now()
        )
        
        self.violations.append(violation)
        
        # Add to agent profile
        if agent_id in self.agent_profiles:
            self.agent_profiles[agent_id]["violations"].append(violation.violation_id)
        
        # Log security event
        self.audit_logger.log_security_event(agent_id, violation_type, "medium", {
            "description": description,
            "violation_id": violation.violation_id
        })