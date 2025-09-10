"""
🍭 CandyLLM Configuration Management System
Secure, persistent configuration management with encryption for sensitive data
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from cryptography.fernet import Fernet
import base64
import hashlib

class ConfigManager:
    """
    Comprehensive configuration management for CandyLLM
    
    Features:
    - Secure storage of API keys with encryption
    - Multiple configuration sources (file, env, defaults)
    - Provider-specific configurations
    - Configuration validation
    - Backup and restore capabilities
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager"""
        self.config_dir = Path(config_dir) if config_dir else Path.home() / ".candyllm"
        self.config_file = self.config_dir / "config.yaml"
        self.secrets_file = self.config_dir / "secrets.enc"
        self.key_file = self.config_dir / ".key"
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True, mode=0o700)
        
        # Initialize encryption
        self._init_encryption()
        
        # Load configuration
        self._config = self._load_config()
        self._secrets = self._load_secrets()
    
    def _init_encryption(self):
        """Initialize encryption for sensitive data"""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Secure the key file
            os.chmod(self.key_file, 0o600)
        
        self.cipher = Fernet(key)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_file.exists():
            return self._get_default_config()
        
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
            return {**self._get_default_config(), **config}
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")
            return self._get_default_config()
    
    def _load_secrets(self) -> Dict[str, str]:
        """Load encrypted secrets"""
        if not self.secrets_file.exists():
            return {}
        
        try:
            with open(self.secrets_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            print(f"⚠️  Error loading secrets: {e}")
            return {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "version": "2.0.0",
            "ui": {
                "default_type": "legacy",
                "port": 7860,
                "host": "127.0.0.1",
                "theme": "soft",
                "auto_launch": True
            },
            "providers": {
                "openai": {
                    "default_model": "gpt-3.5-turbo",
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "base_url": "https://api.openai.com/v1"
                },
                "anthropic": {
                    "default_model": "claude-3-sonnet-20240229",
                    "max_tokens": 2048,
                    "temperature": 0.7
                },
                "huggingface": {
                    "default_model": "microsoft/DialoGPT-large",
                    "use_auth_token": False
                },
                "litellm": {
                    "default_model": "gpt-3.5-turbo",
                    "drop_params": True
                }
            },
            "tools": {
                "cache_size": 1000,
                "auto_register": True,
                "execution_timeout": 30
            },
            "safety": {
                "enable_guards": True,
                "max_input_length": 10000,
                "max_output_length": 50000,
                "content_filters": ["violence", "hate", "sexual"]
            },
            "logging": {
                "level": "INFO",
                "file": "candyllm.log",
                "max_size_mb": 100,
                "backup_count": 5
            }
        }
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def _save_secrets(self):
        """Save encrypted secrets"""
        try:
            data = json.dumps(self._secrets).encode()
            encrypted_data = self.cipher.encrypt(data)
            
            with open(self.secrets_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Secure the secrets file
            os.chmod(self.secrets_file, 0o600)
        except Exception as e:
            print(f"❌ Error saving secrets: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        # Check environment variables first
        env_key = f"CANDYLLM_{key.upper().replace('.', '_')}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value
        
        # Check secrets for sensitive keys
        if self._is_sensitive_key(key):
            return self._secrets.get(key, default)
        
        # Navigate nested config
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any, encrypt: bool = None):
        """Set configuration value"""
        if encrypt is None:
            encrypt = self._is_sensitive_key(key)
        
        if encrypt:
            # Store in encrypted secrets
            self._secrets[key] = str(value)
            self._save_secrets()
        else:
            # Store in regular config
            keys = key.split('.')
            config = self._config
            
            # Navigate to parent
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set value
            config[keys[-1]] = value
            self._save_config()
    
    def delete(self, key: str):
        """Delete configuration key"""
        # Try secrets first
        if key in self._secrets:
            del self._secrets[key]
            self._save_secrets()
            return
        
        # Try regular config
        keys = key.split('.')
        config = self._config
        
        try:
            # Navigate to parent
            for k in keys[:-1]:
                config = config[k]
            
            # Delete key
            if keys[-1] in config:
                del config[keys[-1]]
                self._save_config()
        except (KeyError, TypeError):
            pass  # Key doesn't exist
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration (excluding secrets)"""
        result = {}
        
        # Add non-sensitive config
        def flatten_dict(d, prefix=''):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    flatten_dict(v, key)
                else:
                    result[key] = v
        
        flatten_dict(self._config)
        
        # Add masked secrets
        for key in self._secrets.keys():
            result[key] = "***ENCRYPTED***"
        
        return result
    
    def reset_all(self):
        """Reset all configuration to defaults"""
        self._config = self._get_default_config()
        self._secrets = {}
        self._save_config()
        self._save_secrets()
    
    def backup(self, backup_path: str):
        """Create configuration backup"""
        backup_data = {
            "config": self._config,
            "secrets_keys": list(self._secrets.keys()),  # Don't backup actual secrets
            "version": "2.0.0",
            "timestamp": str(Path().ctime())
        }
        
        with open(backup_path, 'w') as f:
            yaml.dump(backup_data, f)
        
        print(f"✅ Configuration backed up to {backup_path}")
        print("⚠️  Note: Encrypted secrets not included in backup for security")
    
    def restore(self, backup_path: str):
        """Restore configuration from backup"""
        try:
            with open(backup_path, 'r') as f:
                backup_data = yaml.safe_load(f)
            
            if 'config' in backup_data:
                self._config = backup_data['config']
                self._save_config()
                print("✅ Configuration restored")
            
            if 'secrets_keys' in backup_data:
                print("⚠️  Secret keys found in backup but values not restored:")
                for key in backup_data['secrets_keys']:
                    print(f"   - {key}")
                print("💡 You'll need to set these values again")
        
        except Exception as e:
            print(f"❌ Error restoring backup: {e}")
    
    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a key contains sensitive information"""
        sensitive_patterns = [
            'key', 'token', 'secret', 'password', 'auth',
            'api_key', 'access_token', 'bearer_token'
        ]
        
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in sensitive_patterns)
    
    def validate(self) -> Dict[str, Any]:
        """Validate current configuration"""
        issues = []
        warnings = []
        
        # Check required API keys
        required_keys = {
            'openai_api_key': 'OpenAI functionality',
            'anthropic_api_key': 'Anthropic/Claude functionality',
            'huggingface_token': 'HuggingFace private models'
        }
        
        for key, purpose in required_keys.items():
            if not self.get(key):
                warnings.append(f"Missing {key} - {purpose} will not work")
        
        # Validate provider configurations
        providers = self.get('providers', {})
        for provider, config in providers.items():
            if not isinstance(config, dict):
                issues.append(f"Invalid {provider} configuration: must be a dictionary")
        
        # Validate UI configuration
        ui_config = self.get('ui', {})
        if ui_config.get('port', 0) < 1 or ui_config.get('port', 0) > 65535:
            issues.append("Invalid UI port: must be between 1 and 65535")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider"""
        base_config = self.get(f'providers.{provider}', {})
        
        # Add API key if available
        api_key = self.get(f'{provider}_api_key')
        if api_key:
            base_config['api_key'] = api_key
        
        return base_config
    
    def configure_provider(self, provider: str, **kwargs):
        """Configure a provider with validation"""
        provider_config = self.get(f'providers.{provider}', {})
        
        # Update configuration
        for key, value in kwargs.items():
            if self._is_sensitive_key(key):
                self.set(f'{provider}_{key}', value, encrypt=True)
            else:
                provider_config[key] = value
        
        # Save provider config
        self.set(f'providers.{provider}', provider_config)
        
        print(f"✅ {provider.title()} provider configured")

# Global configuration instance
_config_manager = None

def get_config() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def configure_provider(provider: str, **kwargs):
    """Convenience function to configure a provider"""
    config = get_config()
    config.configure_provider(provider, **kwargs)

def get_provider_config(provider: str) -> Dict[str, Any]:
    """Convenience function to get provider configuration"""
    config = get_config()
    return config.get_provider_config(provider)
