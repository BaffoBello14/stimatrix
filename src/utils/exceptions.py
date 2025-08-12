"""Custom exceptions and error handling utilities for the ML pipeline."""

import sys
import traceback
import functools
from typing import Any, Callable, Dict, Optional, Type, Union
from pathlib import Path
import logging
from datetime import datetime
import json

from .logger import get_logger

logger = get_logger(__name__)


class PipelineError(Exception):
    """Base exception for all pipeline-related errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        self.timestamp = datetime.now()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class DataValidationError(PipelineError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, column: str = None, value: Any = None, expected: str = None):
        details = {}
        if column:
            details["column"] = column
        if value is not None:
            details["invalid_value"] = str(value)
        if expected:
            details["expected"] = expected
        
        super().__init__(message, "DATA_VALIDATION_ERROR", details)


class ConfigurationError(PipelineError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: str = None, config_value: Any = None):
        details = {}
        if config_key:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = str(config_value)
        
        super().__init__(message, "CONFIGURATION_ERROR", details)


class ModelTrainingError(PipelineError):
    """Raised when model training fails."""
    
    def __init__(self, message: str, model_type: str = None, hyperparams: Dict[str, Any] = None):
        details = {}
        if model_type:
            details["model_type"] = model_type
        if hyperparams:
            details["hyperparams"] = hyperparams
        
        super().__init__(message, "MODEL_TRAINING_ERROR", details)


class FeatureExtractionError(PipelineError):
    """Raised when feature extraction fails."""
    
    def __init__(self, message: str, feature_type: str = None, column: str = None):
        details = {}
        if feature_type:
            details["feature_type"] = feature_type
        if column:
            details["column"] = column
        
        super().__init__(message, "FEATURE_EXTRACTION_ERROR", details)


class DatabaseError(PipelineError):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: str = None, query: str = None):
        details = {}
        if operation:
            details["operation"] = operation
        if query:
            # Don't log full query for security, just first 100 chars
            details["query_preview"] = query[:100] + "..." if len(query) > 100 else query
        
        super().__init__(message, "DATABASE_ERROR", details)


class FileOperationError(PipelineError):
    """Raised when file operations fail."""
    
    def __init__(self, message: str, file_path: Union[str, Path] = None, operation: str = None):
        details = {}
        if file_path:
            details["file_path"] = str(file_path)
        if operation:
            details["operation"] = operation
        
        super().__init__(message, "FILE_OPERATION_ERROR", details)


class MemoryError(PipelineError):
    """Raised when memory-related issues occur."""
    
    def __init__(self, message: str, memory_usage: str = None, dataset_size: str = None):
        details = {}
        if memory_usage:
            details["memory_usage"] = memory_usage
        if dataset_size:
            details["dataset_size"] = dataset_size
        
        super().__init__(message, "MEMORY_ERROR", details)


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.logger = logger
        self.error_log_file = log_file or Path("logs") / "errors.json"
        self.error_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Error recovery strategies
        self.recovery_strategies = {
            DataValidationError: self._handle_data_validation_error,
            ConfigurationError: self._handle_configuration_error,
            ModelTrainingError: self._handle_model_training_error,
            FeatureExtractionError: self._handle_feature_extraction_error,
            DatabaseError: self._handle_database_error,
            FileOperationError: self._handle_file_operation_error,
            MemoryError: self._handle_memory_error,
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle an error with logging and potential recovery."""
        context = context or {}
        
        # Create error report
        error_report = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
        # Add pipeline-specific details if it's our custom exception
        if isinstance(error, PipelineError):
            error_report.update(error.to_dict())
        
        # Log error
        self._log_error(error_report)
        
        # Attempt recovery
        recovery_result = self._attempt_recovery(error, context)
        error_report["recovery_attempted"] = recovery_result is not None
        error_report["recovery_result"] = recovery_result
        
        return error_report
    
    def _log_error(self, error_report: Dict[str, Any]) -> None:
        """Log error to both standard logger and error log file."""
        # Log to standard logger
        self.logger.error(f"Pipeline error: {error_report['error_type']} - {error_report['message']}")
        
        # Append to error log file (JSON format for structured logging)
        try:
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                json.dump(error_report, f, ensure_ascii=False)
                f.write('\n')
        except Exception as log_error:
            self.logger.error(f"Failed to write to error log file: {log_error}")
    
    def _attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Attempt to recover from error using registered strategies."""
        error_type = type(error)
        
        # Try exact type match first
        if error_type in self.recovery_strategies:
            strategy = self.recovery_strategies[error_type]
            return strategy(error, context)
        
        # Try parent class matches
        for registered_type, strategy in self.recovery_strategies.items():
            if isinstance(error, registered_type):
                return strategy(error, context)
        
        return None
    
    def _handle_data_validation_error(self, error: DataValidationError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data validation errors."""
        self.logger.warning(f"Data validation failed: {error.message}")
        
        recovery_actions = []
        
        # If it's a column-specific issue, suggest dropping the column
        if "column" in error.details:
            recovery_actions.append({
                "action": "drop_column",
                "column": error.details["column"],
                "reason": "validation_failed"
            })
        
        # Suggest data cleaning
        recovery_actions.append({
            "action": "apply_data_cleaning",
            "reason": "validation_failed"
        })
        
        return {"strategy": "data_validation_recovery", "actions": recovery_actions}
    
    def _handle_configuration_error(self, error: ConfigurationError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle configuration errors."""
        self.logger.warning(f"Configuration error: {error.message}")
        
        # Suggest using default values
        return {
            "strategy": "use_default_config",
            "config_key": error.details.get("config_key"),
            "suggested_action": "revert_to_defaults"
        }
    
    def _handle_model_training_error(self, error: ModelTrainingError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model training errors."""
        self.logger.warning(f"Model training failed: {error.message}")
        
        recovery_actions = []
        
        # Suggest simpler hyperparameters
        if "hyperparams" in error.details:
            recovery_actions.append({
                "action": "simplify_hyperparams",
                "model_type": error.details.get("model_type")
            })
        
        # Suggest fallback model
        recovery_actions.append({
            "action": "use_fallback_model",
            "fallback": "linear_regression"
        })
        
        return {"strategy": "model_training_recovery", "actions": recovery_actions}
    
    def _handle_feature_extraction_error(self, error: FeatureExtractionError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle feature extraction errors."""
        self.logger.warning(f"Feature extraction failed: {error.message}")
        
        return {
            "strategy": "skip_feature_extraction",
            "feature_type": error.details.get("feature_type"),
            "column": error.details.get("column")
        }
    
    def _handle_database_error(self, error: DatabaseError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle database errors."""
        self.logger.error(f"Database error: {error.message}")
        
        return {
            "strategy": "use_cached_data",
            "operation": error.details.get("operation"),
            "retry_recommended": True
        }
    
    def _handle_file_operation_error(self, error: FileOperationError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file operation errors."""
        self.logger.error(f"File operation failed: {error.message}")
        
        recovery_actions = []
        
        # Create directory if it doesn't exist
        if "file_path" in error.details:
            file_path = Path(error.details["file_path"])
            if not file_path.parent.exists():
                recovery_actions.append({
                    "action": "create_directory",
                    "path": str(file_path.parent)
                })
        
        return {"strategy": "file_operation_recovery", "actions": recovery_actions}
    
    def _handle_memory_error(self, error: MemoryError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory errors."""
        self.logger.error(f"Memory error: {error.message}")
        
        return {
            "strategy": "reduce_memory_usage",
            "actions": [
                {"action": "enable_chunked_processing"},
                {"action": "reduce_batch_size"},
                {"action": "clear_cache"}
            ]
        }


# Global error handler instance
_global_error_handler = ErrorHandler()


def with_error_handling(
    error_types: Optional[Union[Type[Exception], tuple]] = None,
    reraise: bool = True,
    context_provider: Optional[Callable[[], Dict[str, Any]]] = None
):
    """Decorator to add error handling to functions."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if we should handle this error type
                if error_types is not None:
                    if isinstance(error_types, tuple):
                        if not isinstance(e, error_types):
                            raise
                    else:
                        if not isinstance(e, error_types):
                            raise
                
                # Get context if provider is available
                context = {}
                if context_provider:
                    try:
                        context = context_provider()
                    except Exception:
                        pass  # Don't fail if context provider fails
                
                # Add function context
                context.update({
                    "function": func.__name__,
                    "module": func.__module__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                })
                
                # Handle the error
                error_report = _global_error_handler.handle_error(e, context)
                
                # Log function-specific information
                logger.error(f"Error in function {func.__name__}: {error_report['error_type']}")
                
                if reraise:
                    raise
                else:
                    return error_report
        
        return wrapper
    return decorator


def raise_for_data_validation(condition: bool, message: str, **kwargs) -> None:
    """Raise DataValidationError if condition is False."""
    if not condition:
        raise DataValidationError(message, **kwargs)


def raise_for_configuration(condition: bool, message: str, **kwargs) -> None:
    """Raise ConfigurationError if condition is False."""
    if not condition:
        raise ConfigurationError(message, **kwargs)


def handle_pipeline_error(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function to handle any pipeline error."""
    return _global_error_handler.handle_error(error, context)


# Context manager for error handling
class ErrorContext:
    """Context manager for handling errors in code blocks."""
    
    def __init__(self, context: Dict[str, Any] = None, suppress: bool = False):
        self.context = context or {}
        self.suppress = suppress
        self.error_report = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.error_report = _global_error_handler.handle_error(exc_val, self.context)
            return self.suppress  # Suppress exception if requested
        return False