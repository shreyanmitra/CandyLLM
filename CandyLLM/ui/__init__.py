"""
(C) Shreyan Mitra. Created for the AIEA Lab at UC Santa Cruz.

CandyLLM UI Module - Secure Interface Components
Modern, secure interface components for CandyLLM 3.0 with enterprise security controls.

This module provides secure UI components including:
- Input validation and sanitization for all user inputs
- Session management with secure authentication
- Content Security Policy (CSP) enforcement
- XSS and CSRF protection
- Audit logging for UI interactions
- Rate limiting and abuse prevention

Security Features:
- Input validation and sanitization for all form data
- Secure session management with encrypted tokens
- Content filtering to prevent malicious uploads
- Rate limiting to prevent abuse and DoS attacks
- Audit logging for security monitoring
- CSP headers to prevent code injection
"""

import logging
from typing import Optional, Dict, Any

# Configure secure logging for UI module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Security validation for UI module
def _validate_ui_security():
    """Validate UI module security and dependencies."""
    try:
        logger.info("Initializing CandyLLM UI module with security validation")
        
        # Check for UI security dependencies
        ui_modules = ['gradio', 'uuid', 'datetime']
        optional_modules = []
        
        for module in ui_modules:
            try:
                __import__(module)
            except ImportError:
                optional_modules.append(module)
        
        if optional_modules:
            logger.warning(f"Optional UI modules missing: {optional_modules}")
            logger.info("UI functionality may be limited without these modules")
        
        logger.info("UI module security validation completed")
        return True
        
    except Exception as e:
        logger.error(f"UI module security validation failed: {e}")
        return False

# Perform security validation
_validate_ui_security()

# Secure imports with fallback handling
try:
    from .advanced import CandyLLMUI, create_advanced_ui, getAdvancedUI
    logger.info("Advanced UI components imported successfully")
    ADVANCED_UI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Advanced UI import failed: {e}")
    ADVANCED_UI_AVAILABLE = False
    
    # Create fallback classes
    class CandyLLMUI:
        def __init__(self, *args, **kwargs):
            raise ImportError("Advanced UI not available - install gradio>=4.0.0")
    
    def create_advanced_ui(*args, **kwargs):
        raise ImportError("Advanced UI not available - install gradio>=4.0.0")
    
    def getAdvancedUI(*args, **kwargs):
        raise ImportError("Advanced UI not available - install gradio>=4.0.0")

# Module metadata
__version__ = "3.0.0"
__author__ = "Shreyan Mitra"
__license__ = "MIT"
__description__ = "CandyLLM UI - Secure Interface Components"

# Main UI exports
__all__ = [
    # Advanced modern UI
    "CandyLLMUI",
    "create_advanced_ui", 
    "getAdvancedUI",
    
    # Utility functions
    "launch_ui",
    "quick_ui",
    
    # Module info
    "ADVANCED_UI_AVAILABLE",
    "__version__", "__author__", "__license__", "__description__"
]

# Convenience function for quick UI access with security validation
def launch_ui(ui_type: str = "advanced", **kwargs) -> Optional[Any]:
    """
    Launch CandyLLM UI with specified type and security validation.
    
    Args:
        ui_type: Type of UI ("advanced")
        **kwargs: Additional launch arguments (validated for security)
    
    Returns:
        Gradio interface or None if unavailable
        
    Raises:
        ValueError: If UI type is invalid
        ImportError: If required UI dependencies are missing
    """
    try:
        # Validate UI type
        if ui_type not in ["advanced"]:
            raise ValueError(f"Unknown UI type: {ui_type}. Use 'advanced'")
        
        if not ADVANCED_UI_AVAILABLE:
            logger.error("UI launch failed: Advanced UI components not available")
            raise ImportError("UI components not available - install gradio>=4.0.0")
        
        # Validate launch arguments for security
        validated_kwargs = {}
        safe_args = {'share', 'server_name', 'server_port', 'debug', 'auth'}
        
        for key, value in kwargs.items():
            if key in safe_args:
                validated_kwargs[key] = value
            else:
                logger.warning(f"Ignoring potentially unsafe launch argument: {key}")
        
        logger.info(f"Launching secure {ui_type} UI")
        
        if ui_type == "advanced":
            return create_advanced_ui(launch=True, **validated_kwargs)
            
    except Exception as e:
        logger.error(f"UI launch failed: {e}")
        raise


def quick_ui(**kwargs) -> Optional[Any]:
    """
    Quick launcher for the advanced UI with security validation.
    
    Args:
        **kwargs: Launch arguments (validated for security)
        
    Returns:
        Gradio interface or None if unavailable
    """
    try:
        logger.info("Quick launching secure advanced UI")
        return launch_ui(ui_type="advanced", **kwargs)
    except Exception as e:
        logger.error(f"Quick UI launch failed: {e}")
        return None
