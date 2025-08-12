"""Abstract interfaces for pipeline components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
from dataclasses import dataclass


@dataclass
class ProcessingResult:
    """Result of a processing operation."""
    data: pd.DataFrame
    metadata: Dict[str, Any]
    success: bool
    warnings: List[str]
    errors: List[str]


class IDataLoader(ABC):
    """Interface for data loading components."""
    
    @abstractmethod
    def load(self, source: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data from a source."""
        pass
    
    @abstractmethod
    def validate_source(self, source: Union[str, Path]) -> bool:
        """Validate that the data source is accessible and valid."""
        pass


class IDataSaver(ABC):
    """Interface for data saving components."""
    
    @abstractmethod
    def save(self, data: pd.DataFrame, destination: Union[str, Path], **kwargs) -> bool:
        """Save data to a destination."""
        pass
    
    @abstractmethod
    def validate_destination(self, destination: Union[str, Path]) -> bool:
        """Validate that the destination is writable."""
        pass


class IFeatureExtractor(ABC):
    """Interface for feature extraction components."""
    
    @abstractmethod
    def extract(self, data: pd.DataFrame) -> ProcessingResult:
        """Extract features from data."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get names of features that will be extracted."""
        pass
    
    @abstractmethod
    def supports_column_type(self, column: str, dtype: str) -> bool:
        """Check if this extractor supports the given column type."""
        pass


class IDataTransformer(ABC):
    """Interface for data transformation components."""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'IDataTransformer':
        """Fit the transformer to the data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> ProcessingResult:
        """Transform the data."""
        pass
    
    @abstractmethod
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> ProcessingResult:
        """Fit and transform the data in one step."""
        pass
    
    @abstractmethod
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform the data if possible."""
        pass


class IOutlierDetector(ABC):
    """Interface for outlier detection components."""
    
    @abstractmethod
    def detect(self, data: pd.DataFrame, target_column: str) -> pd.Series:
        """Detect outliers and return a boolean mask."""
        pass
    
    @abstractmethod
    def get_outlier_score(self, data: pd.DataFrame, target_column: str) -> pd.Series:
        """Get outlier scores for each sample."""
        pass


class IImputer(ABC):
    """Interface for missing value imputation components."""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'IImputer':
        """Fit the imputer to the data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values."""
        pass
    
    @abstractmethod
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        pass


class IEncoder(ABC):
    """Interface for categorical encoding components."""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'IEncoder':
        """Fit the encoder to the data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        pass
    
    @abstractmethod
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        pass
    
    @abstractmethod
    def get_encoded_columns(self) -> List[str]:
        """Get names of encoded columns."""
        pass


class IModelTrainer(ABC):
    """Interface for model training components."""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Any:
        """Train a model."""
        pass
    
    @abstractmethod
    def predict(self, model: Any, X: pd.DataFrame) -> pd.Series:
        """Make predictions with a trained model."""
        pass
    
    @abstractmethod
    def evaluate(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        pass
    
    @abstractmethod
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[Dict[str, Any], float]:
        """Tune hyperparameters and return best params and score."""
        pass


class IHyperparameterTuner(ABC):
    """Interface for hyperparameter tuning components."""
    
    @abstractmethod
    def optimize(self, 
                 objective_function: callable,
                 search_space: Dict[str, Any],
                 n_trials: int,
                 **kwargs) -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters."""
        pass
    
    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found."""
        pass
    
    @abstractmethod
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        pass


class IMetricsCalculator(ABC):
    """Interface for metrics calculation components."""
    
    @abstractmethod
    def calculate(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Calculate metrics."""
        pass
    
    @abstractmethod
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics."""
        pass


class IConfigurationProvider(ABC):
    """Interface for configuration providers."""
    
    @abstractmethod
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from a source."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure and values."""
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        pass


class IDatabaseConnector(ABC):
    """Interface for database connection components."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish database connection."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Close database connection."""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a database query."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test database connection."""
        pass


class ILogger(ABC):
    """Interface for logging components."""
    
    @abstractmethod
    def log(self, level: str, message: str, **kwargs) -> None:
        """Log a message."""
        pass
    
    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        pass
    
    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        pass
    
    @abstractmethod
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        pass
    
    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        pass


class IPipelineOrchestrator(ABC):
    """Interface for pipeline orchestration components."""
    
    @abstractmethod
    def run_step(self, step_name: str, **kwargs) -> ProcessingResult:
        """Run a single pipeline step."""
        pass
    
    @abstractmethod
    def run_pipeline(self, steps: List[str], **kwargs) -> List[ProcessingResult]:
        """Run multiple pipeline steps."""
        pass
    
    @abstractmethod
    def get_available_steps(self) -> List[str]:
        """Get list of available pipeline steps."""
        pass
    
    @abstractmethod
    def validate_pipeline(self, steps: List[str]) -> bool:
        """Validate that a pipeline configuration is valid."""
        pass


class IModelRegistry(ABC):
    """Interface for model registry components."""
    
    @abstractmethod
    def save_model(self, model: Any, model_name: str, metadata: Dict[str, Any]) -> str:
        """Save a model and return its ID."""
        pass
    
    @abstractmethod
    def load_model(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        """Load a model and its metadata."""
        pass
    
    @abstractmethod
    def list_models(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List available models."""
        pass
    
    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        pass


class ICacheManager(ABC):
    """Interface for cache management components."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass