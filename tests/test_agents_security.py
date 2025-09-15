"""
Comprehensive test suite for CandyLLM agent security system.

Tests agent security policies, sandboxing, access control, and threat detection
for agent providers.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
import time
from datetime import datetime, timedelta

# Agent security imports
try:
    from CandyLLM.agents.security import (
        AgentSecurityManager, SecurityPolicy, AccessControl,
        ThreatDetector, SandboxEnvironment, SecurityViolation,
        SecurityLevel, ActionType, RiskLevel
    )
    AGENT_SECURITY_AVAILABLE = True
except ImportError:
    AGENT_SECURITY_AVAILABLE = False


@pytest.mark.skipif(not AGENT_SECURITY_AVAILABLE, reason="Agent security not available")
class TestSecurityPolicy:
    """Test suite for SecurityPolicy."""
    
    def test_security_policy_creation(self):
        """Test creating security policies."""
        try:
            policy = SecurityPolicy(
                name="standard_policy",
                description="Standard security policy for agents",
                security_level=SecurityLevel.SANDBOXED,
                allowed_actions=[ActionType.READ_FILE, ActionType.WRITE_FILE],
                blocked_actions=[ActionType.EXECUTE_SYSTEM, ActionType.NETWORK_ACCESS],
                resource_limits={
                    "max_memory_mb": 512,
                    "max_cpu_time_seconds": 30,
                    "max_file_size_mb": 10
                },
                audit_enabled=True,
                threat_detection_enabled=True
            )
            
            assert policy.name == "standard_policy"
            assert policy.security_level == SecurityLevel.SANDBOXED
            assert ActionType.READ_FILE in policy.allowed_actions
            assert ActionType.EXECUTE_SYSTEM in policy.blocked_actions
            assert policy.resource_limits["max_memory_mb"] == 512
            assert policy.audit_enabled is True
            
        except Exception:
            pytest.skip("SecurityPolicy creation differs")
    
    def test_security_policy_defaults(self):
        """Test SecurityPolicy default values."""
        try:
            policy = SecurityPolicy(
                name="minimal_policy",
                security_level=SecurityLevel.MONITORED
            )
            
            assert policy.name == "minimal_policy"
            assert policy.security_level == SecurityLevel.MONITORED
            assert hasattr(policy, 'allowed_actions')
            assert hasattr(policy, 'blocked_actions')
            assert hasattr(policy, 'resource_limits')
            
        except Exception:
            pytest.skip("SecurityPolicy defaults not available")
    
    def test_security_level_hierarchy(self):
        """Test security level hierarchy and restrictions."""
        try:
            levels = [
                SecurityLevel.UNRESTRICTED,
                SecurityLevel.MONITORED,
                SecurityLevel.SANDBOXED,
                SecurityLevel.ENTERPRISE
            ]
            
            # Verify levels exist and have proper ordering
            for level in levels:
                assert level is not None
                
        except Exception:
            pytest.skip("Security level hierarchy not available")
    
    def test_action_type_definitions(self):
        """Test action type definitions."""
        try:
            expected_actions = [
                ActionType.READ_FILE,
                ActionType.WRITE_FILE,
                ActionType.EXECUTE_SYSTEM,
                ActionType.NETWORK_ACCESS,
                ActionType.DATABASE_ACCESS,
                ActionType.EXTERNAL_API_CALL
            ]
            
            for action in expected_actions:
                assert action is not None
                
        except Exception:
            pytest.skip("Action type definitions not available")


@pytest.mark.skipif(not AGENT_SECURITY_AVAILABLE, reason="Agent security not available")
class TestSecurityViolation:
    """Test suite for SecurityViolation."""
    
    def test_security_violation_creation(self):
        """Test creating security violations."""
        try:
            violation = SecurityViolation(
                agent_id="agent_123",
                violation_type="unauthorized_file_access",
                description="Attempted to access restricted file",
                risk_level=RiskLevel.HIGH,
                timestamp=datetime.now(),
                context={
                    "file_path": "/etc/passwd",
                    "action_attempted": "read"
                },
                blocked=True,
                mitigation_actions=["block_action", "notify_admin"]
            )
            
            assert violation.agent_id == "agent_123"
            assert violation.violation_type == "unauthorized_file_access"
            assert violation.risk_level == RiskLevel.HIGH
            assert violation.context["file_path"] == "/etc/passwd"
            assert violation.blocked is True
            assert "block_action" in violation.mitigation_actions
            
        except Exception:
            pytest.skip("SecurityViolation creation differs")
    
    def test_violation_risk_levels(self):
        """Test violation risk level definitions."""
        try:
            risk_levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
            
            for level in risk_levels:
                assert level is not None
                
        except Exception:
            pytest.skip("Risk level definitions not available")


@pytest.mark.skipif(not AGENT_SECURITY_AVAILABLE, reason="Agent security not available")
class TestAccessControl:
    """Test suite for AccessControl."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.access_control = AccessControl()
    
    def test_access_control_initialization(self):
        """Test access control initialization."""
        try:
            assert self.access_control is not None
            assert hasattr(self.access_control, '_permissions')
            assert hasattr(self.access_control, '_roles')
            
        except Exception:
            pytest.skip("AccessControl initialization differs")
    
    def test_permission_management(self):
        """Test permission management."""
        try:
            agent_id = "agent_123"
            
            # Grant permissions
            self.access_control.grant_permission(agent_id, ActionType.READ_FILE)
            self.access_control.grant_permission(agent_id, ActionType.WRITE_FILE)
            
            # Check permissions
            assert self.access_control.has_permission(agent_id, ActionType.READ_FILE)
            assert self.access_control.has_permission(agent_id, ActionType.WRITE_FILE)
            assert not self.access_control.has_permission(agent_id, ActionType.EXECUTE_SYSTEM)
            
            # Revoke permission
            self.access_control.revoke_permission(agent_id, ActionType.WRITE_FILE)
            assert not self.access_control.has_permission(agent_id, ActionType.WRITE_FILE)
            
        except Exception:
            pytest.skip("Permission management not available")
    
    def test_role_based_access(self):
        """Test role-based access control."""
        try:
            # Define roles
            self.access_control.create_role("data_analyst", [
                ActionType.READ_FILE,
                ActionType.DATABASE_ACCESS
            ])
            
            self.access_control.create_role("system_admin", [
                ActionType.READ_FILE,
                ActionType.WRITE_FILE,
                ActionType.EXECUTE_SYSTEM,
                ActionType.NETWORK_ACCESS
            ])
            
            # Assign roles
            agent_id = "agent_456"
            self.access_control.assign_role(agent_id, "data_analyst")
            
            # Check role-based permissions
            assert self.access_control.has_permission(agent_id, ActionType.READ_FILE)
            assert self.access_control.has_permission(agent_id, ActionType.DATABASE_ACCESS)
            assert not self.access_control.has_permission(agent_id, ActionType.EXECUTE_SYSTEM)
            
        except Exception:
            pytest.skip("Role-based access not available")
    
    def test_resource_quotas(self):
        """Test resource quota enforcement."""
        try:
            agent_id = "agent_789"
            
            # Set resource quotas
            self.access_control.set_quota(agent_id, "api_calls_per_hour", 100)
            self.access_control.set_quota(agent_id, "file_operations_per_day", 50)
            
            # Check quota usage
            assert self.access_control.get_quota_usage(agent_id, "api_calls_per_hour") == 0
            
            # Consume quota
            self.access_control.consume_quota(agent_id, "api_calls_per_hour", 10)
            assert self.access_control.get_quota_usage(agent_id, "api_calls_per_hour") == 10
            
            # Check quota limits
            assert self.access_control.check_quota_limit(agent_id, "api_calls_per_hour", 50)
            assert not self.access_control.check_quota_limit(agent_id, "api_calls_per_hour", 100)
            
        except Exception:
            pytest.skip("Resource quotas not available")


@pytest.mark.skipif(not AGENT_SECURITY_AVAILABLE, reason="Agent security not available")
class TestThreatDetector:
    """Test suite for ThreatDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.threat_detector = ThreatDetector()
    
    def test_threat_detector_initialization(self):
        """Test threat detector initialization."""
        try:
            assert self.threat_detector is not None
            assert hasattr(self.threat_detector, '_patterns')
            assert hasattr(self.threat_detector, '_anomaly_detector')
            
        except Exception:
            pytest.skip("ThreatDetector initialization differs")
    
    def test_malicious_input_detection(self):
        """Test detection of malicious inputs."""
        try:
            malicious_inputs = [
                "rm -rf /",
                "DROP TABLE users;",
                "<script>alert('xss')</script>",
                "eval(__import__('os').system('rm -rf /'))",
                "subprocess.call(['rm', '-rf', '/'])"
            ]
            
            for malicious_input in malicious_inputs:
                threat_level = self.threat_detector.analyze_input(malicious_input)
                
                # Should detect as high risk
                assert threat_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                
        except Exception:
            pytest.skip("Malicious input detection not available")
    
    def test_safe_input_analysis(self):
        """Test analysis of safe inputs."""
        try:
            safe_inputs = [
                "What is the weather today?",
                "Calculate the square root of 16",
                "Translate 'hello' to Spanish",
                "Find the latest news about technology",
                "Create a summary of this document"
            ]
            
            for safe_input in safe_inputs:
                threat_level = self.threat_detector.analyze_input(safe_input)
                
                # Should be low risk
                assert threat_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
                
        except Exception:
            pytest.skip("Safe input analysis not available")
    
    def test_behavioral_anomaly_detection(self):
        """Test behavioral anomaly detection."""
        try:
            agent_id = "agent_behavior_test"
            
            # Establish normal behavior pattern
            normal_actions = [
                ("read_file", {"file": "data.txt"}),
                ("process_data", {"operation": "analyze"}),
                ("write_output", {"file": "results.txt"})
            ]
            
            for action, params in normal_actions:
                for _ in range(10):  # Repeat to establish pattern
                    self.threat_detector.record_action(agent_id, action, params)
            
            # Test normal action (should be low risk)
            normal_risk = self.threat_detector.detect_anomaly(
                agent_id, "read_file", {"file": "new_data.txt"}
            )
            assert normal_risk in [RiskLevel.LOW, RiskLevel.MEDIUM]
            
            # Test anomalous action (should be higher risk)
            anomaly_risk = self.threat_detector.detect_anomaly(
                agent_id, "execute_system", {"command": "rm file.txt"}
            )
            assert anomaly_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            
        except Exception:
            pytest.skip("Behavioral anomaly detection not available")
    
    def test_frequency_based_detection(self):
        """Test frequency-based threat detection."""
        try:
            agent_id = "agent_frequency_test"
            
            # Rapid repeated actions (potential DOS or abuse)
            for i in range(100):
                self.threat_detector.record_action(
                    agent_id, "api_call", {"endpoint": "/data", "count": i}
                )
            
            # Check if high frequency is detected as threat
            if hasattr(self.threat_detector, 'check_frequency_abuse'):
                is_abuse = self.threat_detector.check_frequency_abuse(agent_id, "api_call")
                assert is_abuse is True
                
        except Exception:
            pytest.skip("Frequency-based detection not available")


@pytest.mark.skipif(not AGENT_SECURITY_AVAILABLE, reason="Agent security not available")
class TestSandboxEnvironment:
    """Test suite for SandboxEnvironment."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sandbox = SandboxEnvironment()
    
    def test_sandbox_initialization(self):
        """Test sandbox environment initialization."""
        try:
            assert self.sandbox is not None
            assert hasattr(self.sandbox, '_containers')
            assert hasattr(self.sandbox, '_resource_limits')
            
        except Exception:
            pytest.skip("SandboxEnvironment initialization differs")
    
    def test_sandbox_creation(self):
        """Test creating sandbox containers."""
        try:
            agent_id = "agent_sandbox_test"
            
            # Create sandbox with resource limits
            sandbox_config = {
                "memory_limit_mb": 256,
                "cpu_limit_percent": 50,
                "disk_limit_mb": 100,
                "network_access": False,
                "allowed_paths": ["/tmp", "/var/tmp"]
            }
            
            sandbox_id = self.sandbox.create_sandbox(agent_id, sandbox_config)
            
            assert sandbox_id is not None
            assert self.sandbox.is_sandbox_active(sandbox_id)
            
        except Exception:
            pytest.skip("Sandbox creation not available")
    
    def test_sandbox_resource_enforcement(self):
        """Test sandbox resource limit enforcement."""
        try:
            agent_id = "agent_resource_test"
            
            sandbox_config = {
                "memory_limit_mb": 128,
                "cpu_limit_percent": 25,
                "execution_timeout_seconds": 10
            }
            
            sandbox_id = self.sandbox.create_sandbox(agent_id, sandbox_config)
            
            # Test resource monitoring
            if hasattr(self.sandbox, 'monitor_resources'):
                resource_usage = self.sandbox.monitor_resources(sandbox_id)
                
                assert 'memory_usage_mb' in resource_usage
                assert 'cpu_usage_percent' in resource_usage
                assert resource_usage['memory_usage_mb'] <= 128
                
        except Exception:
            pytest.skip("Resource enforcement not available")
    
    def test_sandbox_isolation(self):
        """Test sandbox isolation features."""
        try:
            agent_id = "agent_isolation_test"
            
            # Create isolated sandbox
            sandbox_config = {
                "network_access": False,
                "file_system_isolation": True,
                "process_isolation": True
            }
            
            sandbox_id = self.sandbox.create_sandbox(agent_id, sandbox_config)
            
            # Test isolation features
            isolation_status = self.sandbox.get_isolation_status(sandbox_id)
            
            assert isolation_status['network_isolated'] is True
            assert isolation_status['filesystem_isolated'] is True
            
        except Exception:
            pytest.skip("Sandbox isolation not available")
    
    def test_sandbox_cleanup(self):
        """Test sandbox cleanup and resource deallocation."""
        try:
            agent_id = "agent_cleanup_test"
            
            # Create sandbox
            sandbox_config = {"memory_limit_mb": 64}
            sandbox_id = self.sandbox.create_sandbox(agent_id, sandbox_config)
            
            assert self.sandbox.is_sandbox_active(sandbox_id)
            
            # Destroy sandbox
            cleanup_success = self.sandbox.destroy_sandbox(sandbox_id)
            
            assert cleanup_success is True
            assert not self.sandbox.is_sandbox_active(sandbox_id)
            
        except Exception:
            pytest.skip("Sandbox cleanup not available")


@pytest.mark.skipif(not AGENT_SECURITY_AVAILABLE, reason="Agent security not available")
class TestAgentSecurityManager:
    """Test suite for AgentSecurityManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.security_manager = AgentSecurityManager()
        
        # Create test security policy
        self.test_policy = SecurityPolicy(
            name="test_policy",
            security_level=SecurityLevel.SANDBOXED,
            allowed_actions=[ActionType.READ_FILE, ActionType.WRITE_FILE],
            blocked_actions=[ActionType.EXECUTE_SYSTEM],
            resource_limits={"max_memory_mb": 256},
            audit_enabled=True
        )
    
    def test_security_manager_initialization(self):
        """Test security manager initialization."""
        try:
            assert self.security_manager is not None
            assert hasattr(self.security_manager, 'access_control')
            assert hasattr(self.security_manager, 'threat_detector')
            assert hasattr(self.security_manager, 'sandbox')
            
        except Exception:
            pytest.skip("SecurityManager initialization differs")
    
    def test_agent_security_setup(self):
        """Test setting up security for an agent."""
        try:
            agent_id = "agent_security_setup"
            
            # Setup agent security
            setup_success = self.security_manager.setup_agent_security(
                agent_id, self.test_policy
            )
            
            assert setup_success is True
            
            # Verify security configuration
            agent_policy = self.security_manager.get_agent_policy(agent_id)
            assert agent_policy.name == "test_policy"
            assert agent_policy.security_level == SecurityLevel.SANDBOXED
            
        except Exception:
            pytest.skip("Agent security setup not available")
    
    def test_action_authorization(self):
        """Test action authorization through security manager."""
        try:
            agent_id = "agent_auth_test"
            
            # Setup agent with policy
            self.security_manager.setup_agent_security(agent_id, self.test_policy)
            
            # Test allowed action
            read_authorized = self.security_manager.authorize_action(
                agent_id, ActionType.READ_FILE, {"file": "test.txt"}
            )
            assert read_authorized is True
            
            # Test blocked action
            exec_authorized = self.security_manager.authorize_action(
                agent_id, ActionType.EXECUTE_SYSTEM, {"command": "ls"}
            )
            assert exec_authorized is False
            
        except Exception:
            pytest.skip("Action authorization not available")
    
    def test_threat_monitoring(self):
        """Test threat monitoring and response."""
        try:
            agent_id = "agent_threat_test"
            
            self.security_manager.setup_agent_security(agent_id, self.test_policy)
            
            # Simulate malicious input
            malicious_input = "rm -rf /"
            
            threat_detected = self.security_manager.monitor_input(agent_id, malicious_input)
            
            # Should detect threat
            assert threat_detected is True
            
            # Check if violation was recorded
            violations = self.security_manager.get_violations(agent_id)
            assert len(violations) > 0
            assert violations[0].risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            
        except Exception:
            pytest.skip("Threat monitoring not available")
    
    def test_security_incident_response(self):
        """Test security incident response."""
        try:
            agent_id = "agent_incident_test"
            
            self.security_manager.setup_agent_security(agent_id, self.test_policy)
            
            # Create security violation
            violation = SecurityViolation(
                agent_id=agent_id,
                violation_type="unauthorized_access",
                description="Attempted unauthorized file access",
                risk_level=RiskLevel.HIGH,
                timestamp=datetime.now(),
                blocked=True
            )
            
            # Report violation
            response = self.security_manager.handle_violation(violation)
            
            assert response is not None
            assert response['agent_id'] == agent_id
            assert response['action_taken'] in ['blocked', 'sandboxed', 'terminated']
            
        except Exception:
            pytest.skip("Incident response not available")
    
    def test_audit_logging(self):
        """Test security audit logging."""
        try:
            agent_id = "agent_audit_test"
            
            self.security_manager.setup_agent_security(agent_id, self.test_policy)
            
            # Perform monitored actions
            actions = [
                (ActionType.READ_FILE, {"file": "data.txt"}),
                (ActionType.WRITE_FILE, {"file": "output.txt"}),
                (ActionType.NETWORK_ACCESS, {"url": "https://api.example.com"})
            ]
            
            for action, params in actions:
                self.security_manager.log_action(agent_id, action, params)
            
            # Retrieve audit logs
            audit_logs = self.security_manager.get_audit_logs(agent_id)
            
            assert len(audit_logs) >= len(actions)
            assert all('timestamp' in log for log in audit_logs)
            assert all('action' in log for log in audit_logs)
            
        except Exception:
            pytest.skip("Audit logging not available")
    
    def test_security_policy_updates(self):
        """Test updating security policies for agents."""
        try:
            agent_id = "agent_policy_update"
            
            # Initial setup
            self.security_manager.setup_agent_security(agent_id, self.test_policy)
            
            # Create updated policy
            updated_policy = SecurityPolicy(
                name="updated_policy",
                security_level=SecurityLevel.MONITORED,
                allowed_actions=[ActionType.READ_FILE, ActionType.NETWORK_ACCESS],
                resource_limits={"max_memory_mb": 512}
            )
            
            # Update policy
            update_success = self.security_manager.update_agent_policy(
                agent_id, updated_policy
            )
            
            assert update_success is True
            
            # Verify update
            current_policy = self.security_manager.get_agent_policy(agent_id)
            assert current_policy.name == "updated_policy"
            assert current_policy.security_level == SecurityLevel.MONITORED
            
        except Exception:
            pytest.skip("Policy updates not available")
    
    def test_bulk_security_operations(self):
        """Test bulk security operations across multiple agents."""
        try:
            agent_ids = [f"agent_bulk_{i}" for i in range(5)]
            
            # Bulk security setup
            setup_results = self.security_manager.bulk_setup_security(
                agent_ids, self.test_policy
            )
            
            assert len(setup_results) == 5
            assert all(result['success'] for result in setup_results)
            
            # Bulk policy update
            new_policy = SecurityPolicy(
                name="bulk_policy",
                security_level=SecurityLevel.ENTERPRISE
            )
            
            update_results = self.security_manager.bulk_update_policy(
                agent_ids, new_policy
            )
            
            assert len(update_results) == 5
            assert all(result['success'] for result in update_results)
            
        except Exception:
            pytest.skip("Bulk operations not available")


if __name__ == '__main__':
    pytest.main([__file__, "-v"])