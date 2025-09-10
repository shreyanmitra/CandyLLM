"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

Unit tests for CandyLLM Enterprise Setup Wizard.

Tests cover:
- Configuration management functionality
- Setup wizard initialization and flow
- User input handling and validation
- File and directory creation
- Cloud provider configuration
- Security and validation features
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the setup wizard components
from CandyLLM.setup import CandyLLMEnterpriseSetupWizard, ConfigManager


class TestConfigManager:
    """Test suite for ConfigManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary directory for test config
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
        self.config_manager = ConfigManager(config_file=self.config_file)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initializes correctly."""
        assert self.config_manager.config_file == self.config_file
        assert isinstance(self.config_manager.data, dict)
    
    def test_set_and_get_config(self):
        """Test setting and getting configuration values."""
        # Test basic set/get
        self.config_manager.set("test_key", "test_value")
        assert self.config_manager.get("test_key") == "test_value"
        
        # Test nested configuration
        self.config_manager.set("cloud", {"provider": "aws", "region": "us-east-1"})
        cloud_config = self.config_manager.get("cloud")
        assert cloud_config["provider"] == "aws"
        assert cloud_config["region"] == "us-east-1"
    
    def test_get_with_default(self):
        """Test getting configuration with default values."""
        # Non-existent key should return default
        assert self.config_manager.get("nonexistent", "default") == "default"
        assert self.config_manager.get("nonexistent") is None
    
    def test_reset_all(self):
        """Test resetting all configuration."""
        self.config_manager.set("test_key", "test_value")
        assert len(self.config_manager.data) > 0
        
        self.config_manager.reset_all()
        assert len(self.config_manager.data) == 0
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration from file."""
        # Set some test data
        test_data = {
            "deployment": {"type": "cloud"},
            "api_keys": {"openai": "test_key"},
            "handlers": {"directory": "/test/path"}
        }
        
        for key, value in test_data.items():
            self.config_manager.set(key, value)
        
        # Save configuration
        assert self.config_manager.save_config() is True
        assert self.config_file.exists()
        
        # Create new config manager and verify data loads
        new_config = ConfigManager(config_file=self.config_file)
        assert new_config.get("deployment")["type"] == "cloud"
        assert new_config.get("api_keys")["openai"] == "test_key"
        assert new_config.get("handlers")["directory"] == "/test/path"


class TestCandyLLMEnterpriseSetupWizard:
    """Test suite for CandyLLMEnterpriseSetupWizard class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock Rich library to avoid dependency issues in tests
        with patch('CandyLLM.setup.RICH_AVAILABLE', False):
            self.wizard = CandyLLMEnterpriseSetupWizard()
    
    def test_wizard_initialization(self):
        """Test wizard initializes correctly."""
        assert self.wizard.console is None  # Rich disabled in test
        assert isinstance(self.wizard.config_manager, ConfigManager)
        assert isinstance(self.wizard.setup_data, dict)
        assert isinstance(self.wizard.cloud_config, dict)
        assert isinstance(self.wizard.deployment_config, dict)
    
    def test_print_method(self):
        """Test print method works without Rich."""
        # Should not raise any exceptions
        self.wizard.print("Test message")
        self.wizard.print("Test styled message", style="red")
    
    @patch('builtins.input')
    def test_input_method(self, mock_input):
        """Test input method functionality."""
        # Test basic input
        mock_input.return_value = "user_input"
        result = self.wizard.input("Test prompt")
        assert result == "user_input"
        
        # Test input with default
        mock_input.return_value = ""
        result = self.wizard.input("Test prompt", default="default_value")
        assert result == "default_value"
        
        # Test non-empty input overrides default
        mock_input.return_value = "custom_input"
        result = self.wizard.input("Test prompt", default="default_value")
        assert result == "custom_input"
    
    @patch('getpass.getpass')
    def test_input_method_password(self, mock_getpass):
        """Test input method with password masking."""
        mock_getpass.return_value = "secret_password"
        result = self.wizard.input("Password prompt", password=True)
        assert result == "secret_password"
        mock_getpass.assert_called_once_with("Password prompt: ")
    
    @patch('builtins.input')
    def test_input_int_method(self, mock_input):
        """Test integer input method."""
        # Test valid integer input
        mock_input.return_value = "42"
        result = self.wizard.input_int("Number prompt")
        assert result == 42
        
        # Test input with default
        mock_input.return_value = ""
        result = self.wizard.input_int("Number prompt", default=10)
        assert result == 10
    
    @patch('builtins.input')
    def test_input_int_method_validation(self, mock_input):
        """Test integer input validation."""
        # Test invalid input followed by valid input
        mock_input.side_effect = ["invalid", "42"]
        result = self.wizard.input_int("Number prompt")
        assert result == 42
        assert mock_input.call_count == 2
    
    @patch('builtins.input')
    def test_confirm_method(self, mock_input):
        """Test confirmation method."""
        # Test 'y' response
        mock_input.return_value = "y"
        assert self.wizard.confirm("Test confirmation") is True
        
        # Test 'n' response
        mock_input.return_value = "n"
        assert self.wizard.confirm("Test confirmation") is False
        
        # Test empty response with default True
        mock_input.return_value = ""
        assert self.wizard.confirm("Test confirmation", default=True) is True
        
        # Test empty response with default False
        mock_input.return_value = ""
        assert self.wizard.confirm("Test confirmation", default=False) is False
    
    @patch('builtins.input')
    def test_select_from_list(self, mock_input):
        """Test list selection method."""
        options = ["option1", "option2", "option3"]
        
        # Test valid selection
        mock_input.return_value = "2"
        result = self.wizard.select_from_list("Select option:", options)
        assert result == "option2"
        
        # Test selection with default
        mock_input.return_value = ""
        result = self.wizard.select_from_list("Select option:", options, default="option1")
        assert result == "option1"
    
    @patch('builtins.input')
    def test_select_from_list_validation(self, mock_input):
        """Test list selection validation."""
        options = ["option1", "option2", "option3"]
        
        # Test invalid input followed by valid input
        mock_input.side_effect = ["0", "4", "invalid", "2"]
        result = self.wizard.select_from_list("Select option:", options)
        assert result == "option2"
        assert mock_input.call_count == 4
    
    @patch.object(CandyLLMEnterpriseSetupWizard, '_show_welcome')
    @patch.object(CandyLLMEnterpriseSetupWizard, '_collect_deployment_architecture')
    @patch.object(CandyLLMEnterpriseSetupWizard, '_setup_cloud_providers')
    @patch.object(CandyLLMEnterpriseSetupWizard, '_configure_api_keys')
    @patch.object(CandyLLMEnterpriseSetupWizard, '_configure_handler_functions')
    @patch.object(CandyLLMEnterpriseSetupWizard, '_setup_storage_backends')
    @patch.object(CandyLLMEnterpriseSetupWizard, '_configure_compute_resources')
    @patch.object(CandyLLMEnterpriseSetupWizard, '_configure_knowledge_base')
    @patch.object(CandyLLMEnterpriseSetupWizard, '_configure_monitoring')
    @patch.object(CandyLLMEnterpriseSetupWizard, '_save_configuration')
    @patch.object(CandyLLMEnterpriseSetupWizard, '_show_completion')
    def test_run_setup_success(self, mock_completion, mock_save, mock_monitoring,
                              mock_kb, mock_compute, mock_storage, mock_handlers,
                              mock_api, mock_cloud, mock_deployment, mock_welcome):
        """Test successful setup wizard run."""
        result = self.wizard.run_setup()
        
        assert result is True
        mock_welcome.assert_called_once()
        mock_deployment.assert_called_once()
        mock_cloud.assert_called_once()
        mock_api.assert_called_once()
        mock_handlers.assert_called_once()
        mock_storage.assert_called_once()
        mock_compute.assert_called_once()
        mock_kb.assert_called_once()
        mock_monitoring.assert_called_once()
        mock_save.assert_called_once()
        mock_completion.assert_called_once()
    
    @patch.object(CandyLLMEnterpriseSetupWizard, '_show_welcome')
    def test_run_setup_keyboard_interrupt(self, mock_welcome):
        """Test setup wizard handles keyboard interrupt."""
        mock_welcome.side_effect = KeyboardInterrupt()
        
        result = self.wizard.run_setup()
        assert result is False
    
    @patch.object(CandyLLMEnterpriseSetupWizard, '_show_welcome')
    def test_run_setup_exception(self, mock_welcome):
        """Test setup wizard handles general exceptions."""
        mock_welcome.side_effect = Exception("Test error")
        
        result = self.wizard.run_setup()
        assert result is False


class TestSetupWizardConfiguration:
    """Test suite for setup wizard configuration methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('CandyLLM.setup.RICH_AVAILABLE', False):
            self.wizard = CandyLLMEnterpriseSetupWizard()
    
    @patch.object(CandyLLMEnterpriseSetupWizard, 'select_from_list')
    def test_collect_deployment_architecture(self, mock_select):
        """Test deployment architecture configuration."""
        mock_select.return_value = "cloud"
        
        self.wizard._collect_deployment_architecture()
        
        assert self.wizard.deployment_config['type'] == "cloud"
        assert self.wizard.setup_data['deployment'] == self.wizard.deployment_config
        mock_select.assert_called_once()
    
    @patch.object(CandyLLMEnterpriseSetupWizard, 'select_from_list')
    @patch.object(CandyLLMEnterpriseSetupWizard, 'confirm')
    @patch.object(CandyLLMEnterpriseSetupWizard, 'input')
    def test_setup_cloud_providers(self, mock_input, mock_confirm, mock_select):
        """Test cloud provider configuration."""
        mock_select.return_value = "aws"
        mock_confirm.return_value = True
        mock_input.return_value = "production"
        
        self.wizard._setup_cloud_providers()
        
        assert self.wizard.cloud_config['primary_provider'] == "aws"
        assert self.wizard.cloud_config['use_managed_services'] is True
        assert self.wizard.cloud_config['aws_profile'] == "production"
        assert self.wizard.setup_data['cloud'] == self.wizard.cloud_config
    
    @patch.object(CandyLLMEnterpriseSetupWizard, 'confirm')
    @patch.object(CandyLLMEnterpriseSetupWizard, 'input')
    def test_configure_api_keys(self, mock_input, mock_confirm):
        """Test API keys configuration."""
        # Mock user choosing to configure OpenAI and Anthropic
        mock_confirm.side_effect = [True, True, False, False]  # OpenAI: Yes, Anthropic: Yes, Google: No, Pinecone: No
        mock_input.side_effect = ["openai_key_123", "anthropic_key_456"]
        
        self.wizard._configure_api_keys()
        
        api_keys = self.wizard.setup_data['api_keys']
        assert api_keys['openai'] == "openai_key_123"
        assert api_keys['anthropic'] == "anthropic_key_456"
        assert 'google_ai' not in api_keys
        assert 'pinecone' not in api_keys


class TestSetupWizardFileOperations:
    """Test suite for setup wizard file operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        with patch('CandyLLM.setup.RICH_AVAILABLE', False):
            self.wizard = CandyLLMEnterpriseSetupWizard()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch.object(CandyLLMEnterpriseSetupWizard, 'input')
    @patch.object(CandyLLMEnterpriseSetupWizard, 'confirm')
    def test_configure_handler_functions(self, mock_confirm, mock_input):
        """Test handler functions configuration."""
        test_dir = Path(self.temp_dir) / "handlers"
        mock_input.return_value = str(test_dir)
        mock_confirm.side_effect = [True, True, True]  # Create dir, create examples, auto-load
        
        self.wizard._configure_handler_functions()
        
        # Verify directory was created
        assert test_dir.exists()
        assert test_dir.is_dir()
        
        # Verify configuration was saved
        assert self.wizard.handlers_config['directory'] == str(test_dir.absolute())
        assert self.wizard.handlers_config['auto_load'] is True
        assert self.wizard.setup_data['handlers'] == self.wizard.handlers_config
    
    def test_create_example_handlers(self):
        """Test creation of example handler files."""
        test_dir = Path(self.temp_dir) / "handlers"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        self.wizard._create_example_handlers(test_dir)
        
        # Verify example files were created
        expected_files = ["web_search.py", "data_analysis.py", "email_handler.py"]
        for filename in expected_files:
            file_path = test_dir / filename
            assert file_path.exists()
            assert file_path.is_file()
            
            # Verify file has content
            content = file_path.read_text()
            assert len(content) > 0
            assert "def " in content  # Should contain function definitions


if __name__ == '__main__':
    pytest.main([__file__])
