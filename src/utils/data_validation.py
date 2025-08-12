"""Data validation utilities for the ML pipeline."""

from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
from dataclasses import dataclass, field
from enum import Enum

from utils.logger import get_logger
from utils.exceptions import DataValidationError, raise_for_data_validation

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a data validation issue."""
    severity: ValidationSeverity
    message: str
    column: Optional[str] = None
    row_indices: Optional[List[int]] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "severity": self.severity.value,
            "message": self.message,
            "column": self.column,
            "row_indices": self.row_indices,
            "expected_value": str(self.expected_value) if self.expected_value is not None else None,
            "actual_value": str(self.actual_value) if self.actual_value is not None else None,
            "suggestion": self.suggestion
        }


@dataclass
class ValidationReport:
    """Report containing all validation results."""
    dataset_name: str
    timestamp: datetime
    total_rows: int
    total_columns: int
    issues: List[ValidationIssue] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
    
    def add_passed_check(self, check_name: str) -> None:
        """Add a passed validation check."""
        self.passed_checks.append(check_name)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        severity_counts = {severity.value: 0 for severity in ValidationSeverity}
        for issue in self.issues:
            severity_counts[issue.severity.value] += 1
        
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp.isoformat(),
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "total_issues": len(self.issues),
            "severity_counts": severity_counts,
            "passed_checks": len(self.passed_checks),
            "validation_passed": not self.has_critical_issues() and not self.has_errors()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "summary": self.get_summary(),
            "issues": [issue.to_dict() for issue in self.issues],
            "passed_checks": self.passed_checks
        }
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save report to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class DataValidator:
    """Comprehensive data validation system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional configuration."""
        self.config = config or {}
        self.validators: Dict[str, Callable] = {
            "schema": self._validate_schema,
            "missing_values": self._validate_missing_values,
            "data_types": self._validate_data_types,
            "ranges": self._validate_ranges,
            "uniqueness": self._validate_uniqueness,
            "patterns": self._validate_patterns,
            "consistency": self._validate_consistency,
            "distributions": self._validate_distributions,
            "outliers": self._validate_outliers,
            "temporal": self._validate_temporal_data,
            "referential_integrity": self._validate_referential_integrity
        }
    
    def validate_dataset(self, 
                        df: pd.DataFrame, 
                        dataset_name: str = "unknown",
                        schema: Optional[Dict[str, Any]] = None,
                        custom_validators: Optional[List[Callable]] = None) -> ValidationReport:
        """Perform comprehensive dataset validation."""
        logger.info(f"Starting validation for dataset: {dataset_name}")
        
        report = ValidationReport(
            dataset_name=dataset_name,
            timestamp=datetime.now(),
            total_rows=len(df),
            total_columns=len(df.columns)
        )
        
        # Run standard validators
        for validator_name, validator_func in self.validators.items():
            if self._should_run_validator(validator_name):
                try:
                    logger.debug(f"Running validator: {validator_name}")
                    validator_func(df, report, schema)
                    report.add_passed_check(validator_name)
                except Exception as e:
                    logger.error(f"Validator {validator_name} failed: {e}")
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Validator {validator_name} failed: {e}"
                    ))
        
        # Run custom validators if provided
        if custom_validators:
            for custom_validator in custom_validators:
                try:
                    custom_validator(df, report)
                except Exception as e:
                    logger.error(f"Custom validator failed: {e}")
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Custom validator failed: {e}"
                    ))
        
        logger.info(f"Validation completed for {dataset_name}: {len(report.issues)} issues found")
        return report
    
    def _should_run_validator(self, validator_name: str) -> bool:
        """Check if a validator should be run based on configuration."""
        validator_config = self.config.get("validators", {})
        return validator_config.get(validator_name, {}).get("enabled", True)
    
    def _validate_schema(self, df: pd.DataFrame, report: ValidationReport, schema: Optional[Dict[str, Any]]) -> None:
        """Validate DataFrame schema."""
        if not schema:
            schema = self.config.get("schema", {})
        
        if not schema:
            return  # No schema to validate against
        
        expected_columns = set(schema.get("required_columns", []))
        actual_columns = set(df.columns)
        
        # Check for missing required columns
        missing_columns = expected_columns - actual_columns
        if missing_columns:
            report.add_issue(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message=f"Missing required columns: {list(missing_columns)}",
                suggestion="Add missing columns or update schema definition"
            ))
        
        # Check for unexpected columns
        unexpected_columns = actual_columns - expected_columns
        if unexpected_columns and schema.get("strict", False):
            report.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Unexpected columns found: {list(unexpected_columns)}",
                suggestion="Remove unexpected columns or update schema definition"
            ))
        
        # Validate column data types
        expected_types = schema.get("column_types", {})
        for column, expected_type in expected_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if not self._is_compatible_dtype(actual_type, expected_type):
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Column {column} has incorrect type",
                        column=column,
                        expected_value=expected_type,
                        actual_value=actual_type,
                        suggestion=f"Convert column {column} to {expected_type}"
                    ))
    
    def _validate_missing_values(self, df: pd.DataFrame, report: ValidationReport, schema: Optional[Dict[str, Any]]) -> None:
        """Validate missing value patterns."""
        missing_config = self.config.get("missing_values", {})
        max_missing_percent = missing_config.get("max_missing_percent", 0.5)
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percent = missing_count / len(df)
            
            if missing_percent > max_missing_percent:
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Column {column} has {missing_percent:.1%} missing values",
                    column=column,
                    actual_value=f"{missing_percent:.1%}",
                    suggestion=f"Consider imputation or removal of column {column}"
                ))
            
            # Check for columns that should not have missing values
            no_null_columns = missing_config.get("no_null_columns", [])
            if column in no_null_columns and missing_count > 0:
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Column {column} should not have missing values",
                    column=column,
                    actual_value=f"{missing_count} missing values",
                    suggestion=f"Impute or remove rows with missing {column}"
                ))
    
    def _validate_data_types(self, df: pd.DataFrame, report: ValidationReport, schema: Optional[Dict[str, Any]]) -> None:
        """Validate data types and detect type inconsistencies."""
        for column in df.columns:
            if df[column].dtype == 'object':
                # Check for mixed types in object columns
                sample_values = df[column].dropna().head(1000)
                types_found = set(type(val).__name__ for val in sample_values)
                
                if len(types_found) > 1:
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Column {column} contains mixed data types: {types_found}",
                        column=column,
                        actual_value=str(types_found),
                        suggestion=f"Consider type conversion or data cleaning for {column}"
                    ))
        
        # Check for numeric columns that should be integers
        numeric_config = self.config.get("numeric_validation", {})
        integer_columns = numeric_config.get("should_be_integer", [])
        
        for column in integer_columns:
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                non_integer_mask = df[column].notna() & (df[column] != df[column].astype(int))
                if non_integer_mask.any():
                    non_integer_indices = df[non_integer_mask].index.tolist()
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Column {column} contains non-integer values",
                        column=column,
                        row_indices=non_integer_indices[:10],  # Show first 10
                        suggestion=f"Consider rounding or type conversion for {column}"
                    ))
    
    def _validate_ranges(self, df: pd.DataFrame, report: ValidationReport, schema: Optional[Dict[str, Any]]) -> None:
        """Validate numeric ranges and constraints."""
        range_config = self.config.get("ranges", {})
        
        for column, constraints in range_config.items():
            if column not in df.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue
            
            col_data = df[column].dropna()
            
            # Check minimum value
            if "min" in constraints:
                min_val = constraints["min"]
                below_min = col_data < min_val
                if below_min.any():
                    violating_indices = df[df[column] < min_val].index.tolist()
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Column {column} has values below minimum {min_val}",
                        column=column,
                        row_indices=violating_indices[:10],
                        expected_value=f">= {min_val}",
                        suggestion=f"Remove or correct values below {min_val} in {column}"
                    ))
            
            # Check maximum value
            if "max" in constraints:
                max_val = constraints["max"]
                above_max = col_data > max_val
                if above_max.any():
                    violating_indices = df[df[column] > max_val].index.tolist()
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Column {column} has values above maximum {max_val}",
                        column=column,
                        row_indices=violating_indices[:10],
                        expected_value=f"<= {max_val}",
                        suggestion=f"Remove or correct values above {max_val} in {column}"
                    ))
    
    def _validate_uniqueness(self, df: pd.DataFrame, report: ValidationReport, schema: Optional[Dict[str, Any]]) -> None:
        """Validate uniqueness constraints."""
        uniqueness_config = self.config.get("uniqueness", {})
        unique_columns = uniqueness_config.get("unique_columns", [])
        
        for column in unique_columns:
            if column not in df.columns:
                continue
            
            duplicates = df[column].duplicated()
            if duplicates.any():
                duplicate_indices = df[duplicates].index.tolist()
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Column {column} contains duplicate values",
                    column=column,
                    row_indices=duplicate_indices[:10],
                    suggestion=f"Remove duplicate values from {column}"
                ))
        
        # Check composite uniqueness
        composite_unique = uniqueness_config.get("composite_unique", [])
        for column_group in composite_unique:
            if all(col in df.columns for col in column_group):
                duplicates = df.duplicated(subset=column_group)
                if duplicates.any():
                    duplicate_indices = df[duplicates].index.tolist()
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Composite key {column_group} contains duplicates",
                        row_indices=duplicate_indices[:10],
                        suggestion=f"Remove duplicate combinations of {column_group}"
                    ))
    
    def _validate_patterns(self, df: pd.DataFrame, report: ValidationReport, schema: Optional[Dict[str, Any]]) -> None:
        """Validate string patterns using regex."""
        pattern_config = self.config.get("patterns", {})
        
        for column, pattern_info in pattern_config.items():
            if column not in df.columns:
                continue
            
            pattern = pattern_info.get("pattern")
            if not pattern:
                continue
            
            col_data = df[column].dropna().astype(str)
            invalid_mask = ~col_data.str.match(pattern)
            
            if invalid_mask.any():
                invalid_indices = df[df[column].notna()][invalid_mask].index.tolist()
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Column {column} contains values not matching pattern {pattern}",
                    column=column,
                    row_indices=invalid_indices[:10],
                    expected_value=pattern,
                    suggestion=pattern_info.get("suggestion", f"Fix format in {column}")
                ))
    
    def _validate_consistency(self, df: pd.DataFrame, report: ValidationReport, schema: Optional[Dict[str, Any]]) -> None:
        """Validate data consistency across columns."""
        consistency_config = self.config.get("consistency", {})
        
        # Check conditional constraints
        conditions = consistency_config.get("conditions", [])
        for condition in conditions:
            try:
                condition_expr = condition["condition"]
                violated_mask = ~df.eval(condition_expr)
                
                if violated_mask.any():
                    violating_indices = df[violated_mask].index.tolist()
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Consistency condition violated: {condition_expr}",
                        row_indices=violating_indices[:10],
                        suggestion=condition.get("suggestion", "Fix inconsistent data")
                    ))
            except Exception as e:
                logger.warning(f"Failed to evaluate consistency condition {condition}: {e}")
    
    def _validate_distributions(self, df: pd.DataFrame, report: ValidationReport, schema: Optional[Dict[str, Any]]) -> None:
        """Validate statistical distributions."""
        distribution_config = self.config.get("distributions", {})
        
        for column, dist_info in distribution_config.items():
            if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
                continue
            
            col_data = df[column].dropna()
            
            # Check skewness
            if "max_skewness" in dist_info:
                skewness = col_data.skew()
                if abs(skewness) > dist_info["max_skewness"]:
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Column {column} has high skewness: {skewness:.2f}",
                        column=column,
                        actual_value=f"{skewness:.2f}",
                        suggestion=f"Consider transformation for {column} to reduce skewness"
                    ))
            
            # Check kurtosis
            if "max_kurtosis" in dist_info:
                kurtosis = col_data.kurtosis()
                if abs(kurtosis) > dist_info["max_kurtosis"]:
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Column {column} has extreme kurtosis: {kurtosis:.2f}",
                        column=column,
                        actual_value=f"{kurtosis:.2f}",
                        suggestion=f"Consider checking for outliers in {column}"
                    ))
    
    def _validate_outliers(self, df: pd.DataFrame, report: ValidationReport, schema: Optional[Dict[str, Any]]) -> None:
        """Validate and detect outliers."""
        outlier_config = self.config.get("outliers", {})
        
        for column in df.select_dtypes(include=[np.number]).columns:
            col_data = df[column].dropna()
            
            if len(col_data) < 10:  # Skip if too few values
                continue
            
            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_factor = outlier_config.get("iqr_factor", 1.5)
            lower_bound = Q1 - outlier_factor * IQR
            upper_bound = Q3 + outlier_factor * IQR
            
            outliers = (col_data < lower_bound) | (col_data > upper_bound)
            outlier_count = outliers.sum()
            outlier_percent = outlier_count / len(col_data)
            
            max_outlier_percent = outlier_config.get("max_outlier_percent", 0.05)
            
            if outlier_percent > max_outlier_percent:
                outlier_indices = df[df[column].notna() & 
                                  ((df[column] < lower_bound) | (df[column] > upper_bound))].index.tolist()
                
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Column {column} has {outlier_percent:.1%} outliers",
                    column=column,
                    row_indices=outlier_indices[:10],
                    actual_value=f"{outlier_percent:.1%}",
                    suggestion=f"Review outliers in {column} for data quality issues"
                ))
    
    def _validate_temporal_data(self, df: pd.DataFrame, report: ValidationReport, schema: Optional[Dict[str, Any]]) -> None:
        """Validate temporal/date columns."""
        temporal_config = self.config.get("temporal", {})
        date_columns = temporal_config.get("date_columns", [])
        
        for column in date_columns:
            if column not in df.columns:
                continue
            
            # Try to convert to datetime if not already
            try:
                if not pd.api.types.is_datetime64_any_dtype(df[column]):
                    date_series = pd.to_datetime(df[column], errors='coerce')
                else:
                    date_series = df[column]
                
                # Check for invalid dates
                invalid_dates = date_series.isna() & df[column].notna()
                if invalid_dates.any():
                    invalid_indices = df[invalid_dates].index.tolist()
                    report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Column {column} contains invalid date values",
                        column=column,
                        row_indices=invalid_indices[:10],
                        suggestion=f"Fix or remove invalid dates in {column}"
                    ))
                
                # Check date ranges
                valid_dates = date_series.dropna()
                if len(valid_dates) > 0:
                    min_date = valid_dates.min()
                    max_date = valid_dates.max()
                    
                    # Check if dates are in reasonable range
                    current_year = datetime.now().year
                    if min_date.year < 1900 or max_date.year > current_year + 10:
                        report.add_issue(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Column {column} has dates outside reasonable range",
                            column=column,
                            actual_value=f"{min_date.date()} to {max_date.date()}",
                            suggestion=f"Verify date values in {column}"
                        ))
            
            except Exception as e:
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Failed to validate temporal column {column}: {e}",
                    column=column,
                    suggestion=f"Check date format in {column}"
                ))
    
    def _validate_referential_integrity(self, df: pd.DataFrame, report: ValidationReport, schema: Optional[Dict[str, Any]]) -> None:
        """Validate referential integrity constraints."""
        ref_config = self.config.get("referential_integrity", {})
        
        for constraint in ref_config.get("foreign_keys", []):
            child_column = constraint.get("child_column")
            parent_values = constraint.get("parent_values")
            
            if child_column not in df.columns:
                continue
            
            # Check if all child values exist in parent values
            child_data = df[child_column].dropna()
            invalid_refs = ~child_data.isin(parent_values)
            
            if invalid_refs.any():
                invalid_indices = df[df[child_column].notna()][invalid_refs].index.tolist()
                report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Column {child_column} contains invalid foreign key references",
                    column=child_column,
                    row_indices=invalid_indices[:10],
                    suggestion=f"Fix or remove invalid references in {child_column}"
                ))
    
    def _is_compatible_dtype(self, actual_type: str, expected_type: str) -> bool:
        """Check if actual data type is compatible with expected type."""
        # Simple compatibility mapping
        compatibility_map = {
            "int": ["int64", "int32", "int16", "int8"],
            "float": ["float64", "float32", "int64", "int32", "int16", "int8"],
            "string": ["object", "string"],
            "datetime": ["datetime64", "object"],
            "bool": ["bool", "object"]
        }
        
        for expected, compatible_types in compatibility_map.items():
            if expected_type.lower().startswith(expected):
                return any(actual_type.startswith(ct) for ct in compatible_types)
        
        return actual_type == expected_type


def create_validation_config_template() -> Dict[str, Any]:
    """Create a template configuration for data validation."""
    return {
        "validators": {
            "schema": {"enabled": True},
            "missing_values": {"enabled": True},
            "data_types": {"enabled": True},
            "ranges": {"enabled": True},
            "uniqueness": {"enabled": True},
            "patterns": {"enabled": True},
            "consistency": {"enabled": True},
            "distributions": {"enabled": False},
            "outliers": {"enabled": True},
            "temporal": {"enabled": True},
            "referential_integrity": {"enabled": False}
        },
        "schema": {
            "required_columns": [],
            "column_types": {},
            "strict": False
        },
        "missing_values": {
            "max_missing_percent": 0.5,
            "no_null_columns": []
        },
        "ranges": {},
        "uniqueness": {
            "unique_columns": [],
            "composite_unique": []
        },
        "patterns": {},
        "consistency": {
            "conditions": []
        },
        "distributions": {},
        "outliers": {
            "iqr_factor": 1.5,
            "max_outlier_percent": 0.05
        },
        "temporal": {
            "date_columns": []
        },
        "referential_integrity": {
            "foreign_keys": []
        }
    }


# Convenience functions
def validate_dataframe(df: pd.DataFrame, 
                      config: Optional[Dict[str, Any]] = None,
                      dataset_name: str = "unknown") -> ValidationReport:
    """Validate a DataFrame with optional configuration."""
    validator = DataValidator(config)
    return validator.validate_dataset(df, dataset_name)


def quick_validate(df: pd.DataFrame) -> bool:
    """Quick validation that returns True if data passes basic checks."""
    basic_config = {
        "validators": {
            "schema": {"enabled": False},
            "missing_values": {"enabled": True},
            "data_types": {"enabled": True},
            "ranges": {"enabled": False},
            "uniqueness": {"enabled": False},
            "patterns": {"enabled": False},
            "consistency": {"enabled": False},
            "distributions": {"enabled": False},
            "outliers": {"enabled": False},
            "temporal": {"enabled": False},
            "referential_integrity": {"enabled": False}
        },
        "missing_values": {
            "max_missing_percent": 0.9  # Very lenient for quick check
        }
    }
    
    validator = DataValidator(basic_config)
    report = validator.validate_dataset(df, "quick_check")
    
    return not report.has_critical_issues() and not report.has_errors()