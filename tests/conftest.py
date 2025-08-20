"""Pytest configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any
import yaml


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "paths": {
            "raw_data": "test_data/raw",
            "preprocessed_data": "test_data/preprocessed",
            "schema": "test_schema.json",
            "models_dir": "test_models"
        },
        "target": {
            "column_candidates": ["AI_Prezzo_Ridistribuito"],
            "log_transform": False
        },
        "outliers": {
            "method": "iqr",
            "iqr_factor": 1.5,
            "min_group_size": 10
        },
        "imputation": {
            "numeric_strategy": "median",
            "categorical_strategy": "most_frequent"
        },
        "encoding": {
            "max_ohe_cardinality": 10
        },
        "temporal_split": {
            "year_col": "A_AnnoStipula",
            "month_col": "A_MeseStipula",
            "mode": "fraction",
            "fraction": {"train": 0.8, "valid": 0.1}
        },
        "numeric_coercion": {
            "enabled": True,
            "threshold": 0.95,
            "blacklist_globs": ["*ID*", "*COD*"]
        }
    }


@pytest.fixture
def sample_real_estate_data() -> pd.DataFrame:
    """Generate sample real estate data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        # Target variable
        "AI_Prezzo_Ridistribuito": np.random.lognormal(12, 0.5, n_samples),
        
        # Temporal columns
        "A_AnnoStipula": np.random.choice([2020, 2021, 2022, 2023], n_samples),
        "A_MeseStipula": np.random.randint(1, 13, n_samples),
        
        # Numeric features
        "AI_Superficie": np.random.normal(100, 30, n_samples).clip(20, 500),
        "AI_Piano": np.random.choice(["P1", "P2", "P3", "S1", "T"], n_samples),
        "AI_Civico": [f"{np.random.randint(1, 200)}{np.random.choice(['', 'A', 'B'])}" for _ in range(n_samples)],
        
        # Categorical features
        "AI_IdCategoriaCatastale": np.random.choice([1, 2, 3, 4, 5], n_samples),
        "AI_IdTipologia": np.random.choice(["RES", "COM", "IND"], n_samples),
        
        # Geometry-like columns
        "PC_PoligonoGeometrico": [
            f"POINT ({np.random.uniform(8, 18)} {np.random.uniform(35, 47)})" 
            for _ in range(n_samples)
        ],
        
        # JSON-like columns
        "PC_PoligonoGeoJson": [
            f'{"{"}"type": "Feature", "geometry": {"{"}"type": "Polygon"{"}"}, "properties": {"{"}"areaMq": {np.random.uniform(50, 200):.1f}{"}"}{"}"}}'
            for _ in range(n_samples)
        ],
        
        # Columns with missing values
        "AI_Optional": np.random.choice([1, 2, np.nan], n_samples, p=[0.4, 0.4, 0.2]),
        
        # Constant column (to be dropped)
        "ConstantCol": ["SAME_VALUE"] * n_samples,
        
        # High cardinality categorical
        "HighCardCat": [f"CAT_{i % 50}" for i in range(n_samples)]
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Create and cleanup temporary directory."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_wkt_data() -> pd.DataFrame:
    """Sample data with WKT geometries."""
    return pd.DataFrame({
        "point_col": [
            "POINT (12.4924 41.8902)",
            "POINT (9.1900 45.4642)",
            "INVALID WKT",
            None
        ],
        "polygon_col": [
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)))",
            None
        ]
    })


@pytest.fixture
def sample_training_data(temp_dir):
    """Create sample training data files."""
    data_dir = temp_dir / "preprocessed"
    data_dir.mkdir(exist_ok=True)
    
    # Create sample training data
    X_train = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 100),
        "feature2": np.random.normal(0, 1, 100),
        "feature3": np.random.randint(0, 5, 100)
    })
    y_train = pd.DataFrame({
        "AI_Prezzo_Ridistribuito": np.random.lognormal(12, 0.5, 100)
    })
    
    X_test = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 30),
        "feature2": np.random.normal(0, 1, 30),
        "feature3": np.random.randint(0, 5, 30)
    })
    y_test = pd.DataFrame({
        "AI_Prezzo_Ridistribuito": np.random.lognormal(12, 0.5, 30)
    })
    
    # Save files
    X_train.to_parquet(data_dir / "X_train_tree.parquet")
    y_train.to_parquet(data_dir / "y_train_tree.parquet")
    X_test.to_parquet(data_dir / "X_test_tree.parquet")
    y_test.to_parquet(data_dir / "y_test_tree.parquet")
    
    return data_dir