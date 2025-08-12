"""Security utilities for the ML pipeline."""

import os
import re
import hashlib
import secrets
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
from cryptography.fernet import Fernet
import base64

from .logger import get_logger

logger = get_logger(__name__)


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class InputValidator:
    """Validates and sanitizes input data to prevent injection attacks."""
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\s|^)(union|select|insert|update|delete|drop|create|alter|exec|execute)\s",
        r"(\s|^)(script|javascript|vbscript|onload|onerror)\s",
        r"['\";]",
        r"--",
        r"/\*.*?\*/",
        r"xp_cmdshell",
        r"sp_executesql"
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script.*?>.*?</script>",
        r"javascript:",
        r"onload\s*=",
        r"onerror\s*=",
        r"<iframe.*?>",
        r"<object.*?>",
        r"<embed.*?>"
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"~",
        r"\$\{.*\}",  # Environment variable injection
        r"%.*%"       # Windows environment variables
    ]
    
    @classmethod
    def validate_sql_input(cls, input_string: str) -> str:
        """Validate input against SQL injection patterns."""
        if not isinstance(input_string, str):
            raise SecurityError("Input must be a string")
        
        # Check for suspicious patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, input_string, re.IGNORECASE):
                logger.warning(f"Potential SQL injection detected: {pattern}")
                raise SecurityError(f"Invalid input detected: potential SQL injection")
        
        return input_string
    
    @classmethod
    def validate_file_path(cls, file_path: Union[str, Path]) -> Path:
        """Validate file path against path traversal attacks."""
        path_str = str(file_path)
        
        # Check for path traversal patterns
        for pattern in cls.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, path_str):
                logger.warning(f"Potential path traversal detected: {pattern}")
                raise SecurityError("Invalid file path: potential path traversal")
        
        # Resolve to absolute path and check if it's within allowed boundaries
        try:
            resolved_path = Path(file_path).resolve()
            
            # Ensure path is within project directory (basic sandbox)
            project_root = Path(__file__).parent.parent.parent.resolve()
            if not str(resolved_path).startswith(str(project_root)):
                logger.warning(f"File path outside project boundaries: {resolved_path}")
                raise SecurityError("File path outside allowed boundaries")
                
            return resolved_path
        except Exception as e:
            logger.error(f"Error validating file path: {e}")
            raise SecurityError(f"Invalid file path: {e}")
    
    @classmethod
    def sanitize_column_name(cls, column_name: str) -> str:
        """Sanitize database column names."""
        if not isinstance(column_name, str):
            raise SecurityError("Column name must be a string")
        
        # Allow only alphanumeric characters, underscores, and dots
        if not re.match(r'^[a-zA-Z0-9_\.]+$', column_name):
            raise SecurityError(f"Invalid column name: {column_name}")
        
        # Check length
        if len(column_name) > 128:
            raise SecurityError("Column name too long")
        
        return column_name
    
    @classmethod
    def validate_config_value(cls, key: str, value: Any) -> Any:
        """Validate configuration values."""
        # Convert key to string for validation
        cls.sanitize_column_name(key)
        
        # Check for suspicious content in string values
        if isinstance(value, str):
            # Check for XSS patterns
            for pattern in cls.XSS_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    logger.warning(f"Potential XSS in config value: {key}")
                    raise SecurityError(f"Invalid config value for {key}")
        
        return value


class SecureCredentialManager:
    """Manages secure storage and retrieval of credentials."""
    
    def __init__(self, key_file: Optional[Path] = None):
        """Initialize with optional key file path."""
        self.key_file = key_file or Path.home() / ".stimatrix" / "key.key"
        self._fernet = None
    
    def _get_or_create_key(self) -> bytes:
        """Get existing key or create new one."""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            # Create directory if it doesn't exist
            self.key_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate new key
            key = Fernet.generate_key()
            
            # Save with restricted permissions
            with open(self.key_file, 'wb') as f:
                f.write(key)
            
            # Set restrictive permissions (owner only)
            os.chmod(self.key_file, 0o600)
            
            logger.info(f"Created new encryption key at {self.key_file}")
            return key
    
    @property
    def fernet(self) -> Fernet:
        """Get Fernet instance for encryption/decryption."""
        if self._fernet is None:
            key = self._get_or_create_key()
            self._fernet = Fernet(key)
        return self._fernet
    
    def encrypt_credential(self, credential: str) -> str:
        """Encrypt a credential string."""
        if not isinstance(credential, str):
            raise SecurityError("Credential must be a string")
        
        encrypted = self.fernet.encrypt(credential.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_credential(self, encrypted_credential: str) -> str:
        """Decrypt a credential string."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_credential.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error("Failed to decrypt credential")
            raise SecurityError("Failed to decrypt credential")
    
    def get_secure_env_var(self, var_name: str, encrypted: bool = False) -> Optional[str]:
        """Get environment variable with optional decryption."""
        # Validate variable name
        InputValidator.sanitize_column_name(var_name)
        
        value = os.getenv(var_name)
        if value is None:
            return None
        
        if encrypted:
            return self.decrypt_credential(value)
        else:
            return value
    
    def hash_credential(self, credential: str, salt: Optional[bytes] = None) -> tuple[str, str]:
        """Hash a credential with salt for secure storage."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 for key derivation
        key = hashlib.pbkdf2_hmac('sha256', credential.encode(), salt, 100000)
        
        # Return base64 encoded hash and salt
        hash_b64 = base64.urlsafe_b64encode(key).decode()
        salt_b64 = base64.urlsafe_b64encode(salt).decode()
        
        return hash_b64, salt_b64
    
    def verify_credential(self, credential: str, stored_hash: str, stored_salt: str) -> bool:
        """Verify a credential against stored hash and salt."""
        try:
            salt = base64.urlsafe_b64decode(stored_salt.encode())
            hash_to_verify, _ = self.hash_credential(credential, salt)
            
            # Use constant-time comparison to prevent timing attacks
            return secrets.compare_digest(hash_to_verify, stored_hash)
        except Exception:
            return False


class SecureLogger:
    """Logger wrapper that prevents logging of sensitive information."""
    
    SENSITIVE_PATTERNS = [
        r'password["\s]*[:=]["\s]*[^"\s]+',
        r'token["\s]*[:=]["\s]*[^"\s]+',
        r'api[_\s]*key["\s]*[:=]["\s]*[^"\s]+',
        r'secret["\s]*[:=]["\s]*[^"\s]+',
        r'credential["\s]*[:=]["\s]*[^"\s]+',
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email addresses
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',      # Credit card numbers
    ]
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def _sanitize_message(self, message: str) -> str:
        """Remove sensitive information from log messages."""
        sanitized = message
        
        for pattern in self.SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def info(self, message: str, *args, **kwargs):
        """Log info message with sanitization."""
        sanitized = self._sanitize_message(message)
        self.logger.info(sanitized, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message with sanitization."""
        sanitized = self._sanitize_message(message)
        self.logger.warning(sanitized, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message with sanitization."""
        sanitized = self._sanitize_message(message)
        self.logger.error(sanitized, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message with sanitization."""
        sanitized = self._sanitize_message(message)
        self.logger.debug(sanitized, *args, **kwargs)


def get_secure_logger(name: Optional[str] = None) -> SecureLogger:
    """Get a secure logger instance."""
    base_logger = get_logger(name)
    return SecureLogger(base_logger)


class ConfigValidator:
    """Validates configuration objects for security issues."""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate entire configuration dictionary."""
        validated_config = {}
        
        for key, value in config.items():
            try:
                validated_key = InputValidator.sanitize_column_name(key)
                validated_value = InputValidator.validate_config_value(key, value)
                
                # Recursively validate nested dictionaries
                if isinstance(validated_value, dict):
                    validated_value = ConfigValidator.validate_config(validated_value)
                elif isinstance(validated_value, list):
                    validated_value = [
                        ConfigValidator.validate_config(item) if isinstance(item, dict)
                        else InputValidator.validate_config_value(f"{key}_item", item)
                        for item in validated_value
                    ]
                
                validated_config[validated_key] = validated_value
                
            except SecurityError as e:
                logger.error(f"Security validation failed for config key '{key}': {e}")
                raise
        
        return validated_config
    
    @staticmethod
    def validate_paths(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all path configurations."""
        paths_config = config.get('paths', {})
        
        for path_key, path_value in paths_config.items():
            if isinstance(path_value, (str, Path)):
                try:
                    validated_path = InputValidator.validate_file_path(path_value)
                    paths_config[path_key] = str(validated_path)
                except SecurityError as e:
                    logger.error(f"Path validation failed for '{path_key}': {e}")
                    raise
        
        return config


def audit_log(action: str, details: Dict[str, Any], user: str = "system") -> None:
    """Create audit log entry for security-sensitive actions."""
    audit_logger = get_logger("audit")
    
    # Sanitize details to remove sensitive information
    secure_logger = SecureLogger(audit_logger)
    
    audit_entry = {
        "timestamp": None,  # Will be added by logger
        "user": user,
        "action": action,
        "details": details
    }
    
    secure_logger.info(f"AUDIT: {audit_entry}")


# Utility functions for common security tasks
def generate_session_token() -> str:
    """Generate a secure session token."""
    return secrets.token_urlsafe(32)


def constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks."""
    return secrets.compare_digest(a, b)