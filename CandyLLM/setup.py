"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

CandyLLM Enterprise Setup Wizard - Comprehensive Configuration System

This module provides a complete setup wizard for CandyLLM enterprise deployments,
covering all aspects from basic API configuration to advanced cloud infrastructure,
handler functions, deployment architecture, and security compliance.

Configuration Areas:
- API Keys and Provider Setup
- Handler Functions and Custom Tools Directory
- Cloud Provider Selection (AWS, Azure, GCP)
- Deployment Architecture (Local, Cloud, Hybrid)
- Storage and Database Configuration  
- Compute Resources and Scaling
- Network and Security Configuration
- Monitoring, Logging, and Compliance
- Knowledge Base and Vector Database Setup
"""

import os
import sys
import getpass
from pathlib import Path
from typing import List, Dict, Any
import logging
import json
import yaml

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Simple configuration manager for setup wizard."""
    
    def __init__(self, config_file: Path = None):
        self.config_file = config_file or Path.home() / ".candyllm" / "config.yaml"
        self.data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        return {}
    
    def save_config(self) -> bool:
        """Save configuration to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                yaml.dump(self.data, f, default_flow_style=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.data.get(key, default)
    
    def reset_all(self):
        """Reset all configuration."""
        self.data = {}

class CandyLLMEnterpriseSetupWizard:
    """Comprehensive enterprise setup wizard for CandyLLM."""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.config_manager = ConfigManager()
        self.setup_data = {}
        self.cloud_config = {}
        self.deployment_config = {}
        self.handlers_config = {}
        self.storage_config = {}
        self.compute_config = {}

    def print(self, message: str, style: str = None):
        """Print message with optional styling."""
        if self.console:
            if style:
                self.console.print(message, style=style)
            else:
                self.console.print(message)
        else:
            print(message)

    def input(self, prompt: str, default: str = None, password: bool = False) -> str:
        """Get user input with optional default and password masking."""
        if RICH_AVAILABLE and not password:
            return Prompt.ask(prompt, default=default)
        elif password:
            return getpass.getpass(f"{prompt}: ")
        else:
            if default:
                response = input(f"{prompt} [{default}]: ").strip()
                return response if response else default
            return input(f"{prompt}: ").strip()

    def input_int(self, prompt: str, default: int = None) -> int:
        """Get integer input from user."""
        if RICH_AVAILABLE:
            return IntPrompt.ask(prompt, default=default)
        else:
            while True:
                try:
                    if default:
                        response = input(f"{prompt} [{default}]: ").strip()
                        return int(response) if response else default
                    return int(input(f"{prompt}: "))
                except ValueError:
                    print("Please enter a valid number.")

    def confirm(self, prompt: str, default: bool = True) -> bool:
        """Get yes/no confirmation from user."""
        if RICH_AVAILABLE:
            return Confirm.ask(prompt, default=default)
        else:
            default_text = "Y/n" if default else "y/N"
            response = input(f"{prompt} [{default_text}]: ").strip().lower()
            if not response:
                return default
            return response.startswith('y')

    def select_from_list(self, prompt: str, options: List[str], default: str = None) -> str:
        """Allow user to select from a list of options."""
        self.print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            marker = " (default)" if option == default else ""
            self.print(f"  {i}. {option}{marker}")
        
        while True:
            try:
                choice = input("\nEnter choice number: ").strip()
                if not choice and default:
                    return default
                choice_num = int(choice)
                if 1 <= choice_num <= len(options):
                    return options[choice_num - 1]
                else:
                    print(f"Please enter a number between 1 and {len(options)}")
            except ValueError:
                print("Please enter a valid number")

    def run_setup(self) -> bool:
        """Run the complete setup wizard."""
        try:
            self._show_welcome()
            
            # Core setup steps
            self._collect_deployment_architecture()
            self._setup_cloud_providers()
            self._configure_api_keys()
            self._configure_handler_functions()
            self._setup_storage_backends()
            self._configure_compute_resources()
            self._configure_knowledge_base()
            self._configure_monitoring()
            
            # Save configuration and show completion
            self._save_configuration()
            self._show_completion()
            return True
            
        except KeyboardInterrupt:
            self.print("\n❌ Setup cancelled by user.", style="yellow")
            return False
        except Exception as e:
            self.print(f"❌ Setup failed: {e}", style="red")
            logger.error(f"Setup wizard error: {e}", exc_info=True)
            return False

    def _show_welcome(self):
        """Display welcome screen."""
        if self.console:
            self.console.print(Panel.fit(
                "🍭 CandyLLM Enterprise Setup Wizard\n\n"
                "This comprehensive wizard will configure:\n"
                "• Deployment Architecture (Local/Cloud/Hybrid)\n"
                "• Cloud Provider Setup (AWS/Azure/GCP)\n"
                "• API Keys and Authentication\n"
                "• Handler Functions Directory\n"
                "• Storage & Database Configuration\n"
                "• Compute Resources & Scaling\n"
                "• Knowledge Base & Vector DBs\n"
                "• Monitoring & Compliance",
                title="Enterprise Setup",
                border_style="blue"
            ))
        else:
            self.print("=" * 70)
            self.print("🍭 CandyLLM Enterprise Setup Wizard")
            self.print("=" * 70)
            self.print("This wizard will configure your complete CandyLLM enterprise deployment.")

    def _collect_deployment_architecture(self):
        """Configure deployment architecture."""
        self.print("\n🏗️  Deployment Architecture Configuration")
        self.print("-" * 50)
        
        deployment_options = [
            "local", "cloud", "hybrid", "edge", "enterprise"
        ]
        
        deployment_type = self.select_from_list(
            "Select deployment architecture:",
            deployment_options,
            default="local"
        )
        
        self.deployment_config['type'] = deployment_type
        
        if deployment_type in ['cloud', 'hybrid', 'enterprise']:
            regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
            region = self.select_from_list(
                "Select primary region:",
                regions,
                default="us-east-1"
            )
            self.deployment_config['primary_region'] = region
        
        self.setup_data['deployment'] = self.deployment_config

    def _setup_cloud_providers(self):
        """Configure cloud provider settings."""
        self.print("\n☁️  Cloud Provider Configuration")
        self.print("-" * 50)
        
        cloud_providers = ["aws", "azure", "gcp", "none"]
        primary_cloud = self.select_from_list(
            "Select primary cloud provider:",
            cloud_providers,
            default="aws"
        )
        
        self.cloud_config['primary_provider'] = primary_cloud
        
        if primary_cloud != "none":
            # Configure cloud-specific settings
            self.cloud_config['use_managed_services'] = self.confirm(
                "Use managed services (databases, storage, etc.)?"
            )
            
            if primary_cloud == "aws":
                self.cloud_config['aws_profile'] = self.input(
                    "AWS profile name", default="default"
                )
            elif primary_cloud == "azure":
                self.cloud_config['azure_subscription'] = self.input(
                    "Azure subscription ID"
                )
            elif primary_cloud == "gcp":
                self.cloud_config['gcp_project'] = self.input(
                    "GCP project ID"
                )
        
        self.setup_data['cloud'] = self.cloud_config

    def _configure_api_keys(self):
        """Configure API keys for various services."""
        self.print("\n🔑 API Keys Configuration")
        self.print("-" * 50)
        
        api_keys = {}
        
        # LLM Provider APIs
        if self.confirm("Configure OpenAI API key?"):
            api_keys['openai'] = self.input("OpenAI API key", password=True)
        
        if self.confirm("Configure Anthropic API key?"):
            api_keys['anthropic'] = self.input("Anthropic API key", password=True)
        
        if self.confirm("Configure Google AI API key?"):
            api_keys['google_ai'] = self.input("Google AI API key", password=True)
        
        # Vector Database APIs
        if self.confirm("Configure Pinecone API key?"):
            api_keys['pinecone'] = self.input("Pinecone API key", password=True)
        
        self.setup_data['api_keys'] = api_keys

    def _configure_handler_functions(self):
        """Configure handler functions directory and examples."""
        self.print("\n🔧 Handler Functions Configuration")
        self.print("-" * 50)
        
        default_handlers_dir = str(Path.home() / ".candyllm" / "handlers")
        handlers_dir = self.input(
            "Handler functions directory",
            default=default_handlers_dir
        )
        
        handlers_path = Path(handlers_dir)
        
        if not handlers_path.exists():
            if self.confirm(f"Create directory {handlers_dir}?"):
                handlers_path.mkdir(parents=True, exist_ok=True)
                self.print(f"✅ Created directory: {handlers_dir}")
        
        # Create example handler files
        if self.confirm("Create example handler functions?"):
            self._create_example_handlers(handlers_path)
        
        self.handlers_config['directory'] = str(handlers_path.absolute())
        self.handlers_config['auto_load'] = self.confirm(
            "Auto-load handlers on startup?", default=True
        )
        
        self.setup_data['handlers'] = self.handlers_config

    def _create_example_handlers(self, handlers_dir: Path):
        """Create example handler function files."""
        examples = {
            "web_search.py": '''
def web_search(query: str) -> str:
    """Example web search handler."""
    # Implementation would use real search API
    return f"Search results for: {query}"
''',
            "data_analysis.py": '''
import pandas as pd

def analyze_csv(file_path: str) -> str:
    """Example data analysis handler."""
    # Implementation would analyze real CSV
    return f"Analysis of {file_path} completed"
''',
            "email_handler.py": '''
def send_email(to: str, subject: str, body: str) -> str:
    """Example email handler."""
    # Implementation would send real email
    return f"Email sent to {to}"
'''
        }
        
        for filename, content in examples.items():
            file_path = handlers_dir / filename
            if not file_path.exists():
                file_path.write_text(content.strip())
                self.print(f"✅ Created example: {filename}")

    def _setup_storage_backends(self):
        """Configure storage backend options."""
        self.print("\n💾 Storage Backend Configuration")
        self.print("-" * 50)
        
        storage_options = [
            "filesystem", "cloud_storage", "database", "hybrid"
        ]
        
        storage_type = self.select_from_list(
            "Select primary storage backend:",
            storage_options,
            default="filesystem"
        )
        
        self.storage_config['type'] = storage_type
        
        if storage_type == "filesystem":
            data_dir = self.input(
                "Data directory", 
                default=str(Path.home() / ".candyllm" / "data")
            )
            self.storage_config['data_directory'] = data_dir
            
        elif storage_type == "cloud_storage":
            if self.cloud_config.get('primary_provider') == 'aws':
                bucket = self.input("S3 bucket name")
                self.storage_config['s3_bucket'] = bucket
            elif self.cloud_config.get('primary_provider') == 'azure':
                container = self.input("Azure container name")
                self.storage_config['azure_container'] = container
                
        elif storage_type == "database":
            db_options = ["postgresql", "mysql", "mongodb", "sqlite"]
            db_type = self.select_from_list(
                "Select database type:",
                db_options,
                default="postgresql"
            )
            self.storage_config['database_type'] = db_type
            
            if db_type != "sqlite":
                self.storage_config['database_host'] = self.input("Database host")
                self.storage_config['database_port'] = self.input_int("Database port")
                self.storage_config['database_name'] = self.input("Database name")
                self.storage_config['database_user'] = self.input("Database user")
                self.storage_config['database_password'] = self.input(
                    "Database password", password=True
                )
        
        self.setup_data['storage'] = self.storage_config

    def _configure_compute_resources(self):
        """Configure compute resources and scaling."""
        self.print("\n🖥️  Compute Resources Configuration")
        self.print("-" * 50)
        
        # Basic resource allocation
        cpu_cores = self.input_int("CPU cores per instance", default=2)
        memory_gb = self.input_int("Memory (GB) per instance", default=4)
        
        self.compute_config.update({
            'cpu_cores': cpu_cores,
            'memory_gb': memory_gb
        })
        
        # Container orchestration
        if self.deployment_config.get('type') in ['cloud', 'hybrid', 'enterprise']:
            orchestration_options = ["docker", "kubernetes", "ecs", "aks", "gke"]
            orchestration = self.select_from_list(
                "Select container orchestration:",
                orchestration_options,
                default="docker"
            )
            self.compute_config['orchestration'] = orchestration
            
            # Auto-scaling configuration
            if orchestration in ['kubernetes', 'ecs', 'aks', 'gke']:
                self.compute_config['auto_scaling'] = self.confirm(
                    "Enable auto-scaling?", default=True
                )
                if self.compute_config['auto_scaling']:
                    self.compute_config['min_instances'] = self.input_int(
                        "Minimum instances", default=1
                    )
                    self.compute_config['max_instances'] = self.input_int(
                        "Maximum instances", default=10
                    )
        
        self.setup_data['compute'] = self.compute_config

    def _configure_knowledge_base(self):
        """Configure knowledge base and vector database settings."""
        self.print("\n📚 Knowledge Base Configuration")
        self.print("-" * 50)
        
        if not self.confirm("Configure knowledge base?"):
            return
        
        kb_config = {}
        
        # Vector database selection
        vector_db_options = ["faiss", "chromadb", "pinecone", "qdrant", "weaviate"]
        vector_db = self.select_from_list(
            "Select vector database:",
            vector_db_options,
            default="faiss"
        )
        kb_config['vector_database'] = vector_db
        
        # Embedding model configuration
        embedding_options = ["openai", "sentence-transformers", "huggingface"]
        embedding_provider = self.select_from_list(
            "Select embedding provider:",
            embedding_options,
            default="sentence-transformers"
        )
        kb_config['embedding_provider'] = embedding_provider
        
        if embedding_provider == "sentence-transformers":
            model_name = self.input(
                "Sentence transformer model",
                default="all-MiniLM-L6-v2"
            )
            kb_config['embedding_model'] = model_name
        
        # Document processing
        kb_config['chunk_size'] = self.input_int("Document chunk size", default=1000)
        kb_config['chunk_overlap'] = self.input_int("Chunk overlap", default=200)
        
        self.setup_data['knowledge_base'] = kb_config

    def _configure_monitoring(self):
        """Configure monitoring and logging options."""
        self.print("\n📊 Monitoring & Logging Configuration")
        self.print("-" * 50)
        
        monitoring_config = {}
        
        # Logging level
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        log_level = self.select_from_list(
            "Select logging level:",
            log_levels,
            default="INFO"
        )
        monitoring_config['log_level'] = log_level
        
        # Metrics collection
        monitoring_config['enable_metrics'] = self.confirm(
            "Enable metrics collection?", default=True
        )
        
        if monitoring_config['enable_metrics']:
            if self.cloud_config.get('primary_provider') == 'aws':
                monitoring_config['use_cloudwatch'] = self.confirm(
                    "Use AWS CloudWatch?", default=True
                )
            elif self.cloud_config.get('primary_provider') == 'azure':
                monitoring_config['use_azure_monitor'] = self.confirm(
                    "Use Azure Monitor?", default=True
                )
            elif self.cloud_config.get('primary_provider') == 'gcp':
                monitoring_config['use_stackdriver'] = self.confirm(
                    "Use Google Cloud Monitoring?", default=True
                )
        
        # Security and compliance
        monitoring_config['audit_logging'] = self.confirm(
            "Enable audit logging?", default=True
        )
        
        self.setup_data['monitoring'] = monitoring_config

    def _save_configuration(self):
        """Save all configuration to files."""
        self.print("\n💾 Saving Configuration...")
        
        # Save to config manager
        for key, value in self.setup_data.items():
            self.config_manager.set(key, value)
        
        # Save configuration file
        if self.config_manager.save_config():
            self.print(f"✅ Configuration saved to: {self.config_manager.config_file}")
        else:
            self.print("❌ Failed to save configuration", style="red")
        
        # Generate environment file
        self._generate_env_file()

    def _generate_env_file(self):
        """Generate environment configuration file."""
        env_file = Path.home() / ".candyllm" / ".env"
        
        env_vars = []
        
        # API keys
        api_keys = self.setup_data.get('api_keys', {})
        for provider, key in api_keys.items():
            env_vars.append(f"{provider.upper()}_API_KEY={key}")
        
        # Cloud configuration
        cloud_config = self.setup_data.get('cloud', {})
        if cloud_config.get('primary_provider'):
            env_vars.append(f"CLOUD_PROVIDER={cloud_config['primary_provider']}")
        
        # Save environment file
        try:
            env_file.parent.mkdir(parents=True, exist_ok=True)
            env_file.write_text('\n'.join(env_vars))
            self.print(f"✅ Environment file saved to: {env_file}")
        except Exception as e:
            self.print(f"❌ Failed to save environment file: {e}", style="red")

    def _show_completion(self):
        """Display setup completion summary."""
        if self.console:
            # Create summary table
            table = Table(title="Setup Summary")
            table.add_column("Component", style="cyan")
            table.add_column("Configuration", style="green")
            
            for key, value in self.setup_data.items():
                if isinstance(value, dict):
                    config_str = ", ".join([f"{k}: {v}" for k, v in value.items()][:3])
                    if len(value) > 3:
                        config_str += "..."
                else:
                    config_str = str(value)
                table.add_row(key.title(), config_str)
            
            self.console.print("\n")
            self.console.print(table)
            
            self.console.print(Panel.fit(
                "🎉 CandyLLM Enterprise Setup Complete!\n\n"
                "Your enterprise deployment is now configured.\n\n"
                "Next steps:\n"
                "• Review configuration files\n"
                "• Start CandyLLM services\n"
                "• Test your deployment\n\n"
                "For support: https://github.com/shreyanmitra/CandyLLM",
                title="Setup Complete",
                border_style="green"
            ))
        else:
            self.print("=" * 70)
            self.print("🎉 CandyLLM Enterprise Setup Complete!")
            self.print("=" * 70)
            self.print("Your CandyLLM enterprise deployment is now configured!")
            self.print(f"Configuration saved to: {self.config_manager.config_file}")
            self.print("\nNext steps:")
            self.print("• Review configuration files")
            self.print("• Start CandyLLM services")
            self.print("• Test your deployment")


def main():
    """Main entry point for the setup wizard."""
    wizard = CandyLLMEnterpriseSetupWizard()
    success = wizard.run_setup()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()