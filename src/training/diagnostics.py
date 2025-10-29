"""Advanced diagnostics: residual analysis, drift detection, prediction intervals."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import RegressorMixin

from utils.logger import get_logger
from utils.io import save_json

logger = get_logger(__name__)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute residuals (prediction errors)."""
    return y_true - y_pred


def residual_analysis(
    model_key: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: pd.DataFrame | None,
    config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Perform comprehensive residual analysis.
    
    Analyzes:
    - Overall residual distribution
    - Residuals by group
    - Worst predictions
    - Residual plots
    
    Args:
        model_key: Model name
        y_true: True target values
        y_pred: Predicted values
        X: Feature DataFrame (for grouping)
        config: Configuration dict
        output_dir: Output directory for plots
    
    Returns:
        Dictionary with residual analysis results
    """
    res_cfg = config.get("diagnostics", {}).get("residual_analysis", {})
    if not res_cfg.get("enabled", True):
        return {}
    
    logger.info(f"[{model_key}] Performing residual analysis")
    
    residuals = compute_residuals(y_true, y_pred)
    
    results = {
        "overall": {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "median": float(np.median(residuals)),
            "q25": float(np.percentile(residuals, 25)),
            "q75": float(np.percentile(residuals, 75)),
            "skewness": float(stats.skew(residuals)),
            "kurtosis": float(stats.kurtosis(residuals))
        },
        "by_group": {}
    }
    
    # Residuals by group
    by_groups = res_cfg.get("by_groups", [])
    if X is not None and by_groups:
        for group_col in by_groups:
            # Handle price_quartile specially
            if group_col == "price_quartile":
                try:
                    quartiles = pd.qcut(y_true, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates='drop')
                    group_data = pd.DataFrame({"residual": residuals, "group": quartiles})
                except Exception as e:
                    logger.warning(f"  Failed to create price quartiles: {e}")
                    continue
            else:
                if group_col not in X.columns:
                    continue
                group_data = pd.DataFrame({"residual": residuals, "group": X[group_col]})
            
            group_stats = group_data.groupby("group")["residual"].agg([
                "mean", "std", "median", "count"
            ]).to_dict(orient="index")
            
            results["by_group"][group_col] = {str(k): v for k, v in group_stats.items()}
    
    # Worst predictions
    if res_cfg.get("save_worst_predictions", True):
        top_n = int(res_cfg.get("top_n_worst", 50))
        abs_residuals = np.abs(residuals)
        worst_indices = np.argsort(abs_residuals)[-top_n:][::-1]
        
        worst_df = pd.DataFrame({
            "true": y_true[worst_indices],
            "predicted": y_pred[worst_indices],
            "residual": residuals[worst_indices],
            "abs_residual": abs_residuals[worst_indices]
        })
        
        worst_file = output_dir / f"{model_key}_worst_predictions.csv"
        worst_df.to_csv(worst_file, index=False)
        logger.info(f"  Saved worst predictions: {worst_file}")
    
    # Plots
    plots = res_cfg.get("plots", [])
    if plots:
        plot_dir = ensure_dir(output_dir / f"{model_key}_residual_plots")
        
        if "residual_vs_predicted" in plots:
            try:
                plt.figure(figsize=(10, 6))
                plt.scatter(y_pred, residuals, alpha=0.5, s=10)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel("Predicted")
                plt.ylabel("Residual")
                plt.title(f"{model_key}: Residuals vs Predicted")
                plt.tight_layout()
                plt.savefig(plot_dir / "residual_vs_predicted.png", dpi=150)
                plt.close()
            except Exception as e:
                logger.warning(f"  Failed to create residual_vs_predicted plot: {e}")
        
        if "residual_vs_actual" in plots:
            try:
                plt.figure(figsize=(10, 6))
                plt.scatter(y_true, residuals, alpha=0.5, s=10)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel("Actual")
                plt.ylabel("Residual")
                plt.title(f"{model_key}: Residuals vs Actual")
                plt.tight_layout()
                plt.savefig(plot_dir / "residual_vs_actual.png", dpi=150)
                plt.close()
            except Exception as e:
                logger.warning(f"  Failed to create residual_vs_actual plot: {e}")
        
        if "residual_distribution" in plots:
            try:
                plt.figure(figsize=(10, 6))
                plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
                plt.axvline(x=0, color='r', linestyle='--')
                plt.xlabel("Residual")
                plt.ylabel("Frequency")
                plt.title(f"{model_key}: Residual Distribution")
                plt.tight_layout()
                plt.savefig(plot_dir / "residual_distribution.png", dpi=150)
                plt.close()
            except Exception as e:
                logger.warning(f"  Failed to create residual_distribution plot: {e}")
        
        logger.info(f"  Saved residual plots: {plot_dir}")
    
    return results


def drift_detection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Detect distribution drift between train and test sets.
    
    Uses:
    - PSI (Population Stability Index)
    - Kolmogorov-Smirnov test
    
    Args:
        X_train: Training features
        X_test: Test features
        config: Configuration dict
        output_dir: Output directory
    
    Returns:
        Dictionary with drift analysis results
    """
    drift_cfg = config.get("diagnostics", {}).get("drift_detection", {})
    if not drift_cfg.get("enabled", True):
        return {}
    
    logger.info("Performing drift detection (train vs test)")
    
    methods = drift_cfg.get("methods", ["psi", "ks_test"])
    alert_threshold = float(drift_cfg.get("alert_threshold", 0.15))
    
    results = {"features": {}, "alerts": []}
    
    # Only check numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    common_cols = [c for c in numeric_cols if c in X_test.columns]
    
    for col in common_cols:
        train_vals = X_train[col].dropna().values
        test_vals = X_test[col].dropna().values
        
        if len(train_vals) == 0 or len(test_vals) == 0:
            continue
        
        col_results = {}
        
        # PSI (Population Stability Index)
        if "psi" in methods:
            try:
                psi = calculate_psi(train_vals, test_vals)
                col_results["psi"] = float(psi)
                
                if psi > alert_threshold:
                    results["alerts"].append({
                        "feature": col,
                        "method": "psi",
                        "value": float(psi),
                        "threshold": alert_threshold
                    })
            except Exception as e:
                logger.debug(f"  PSI calculation failed for {col}: {e}")
                col_results["psi"] = None
        
        # Kolmogorov-Smirnov test
        if "ks_test" in methods:
            try:
                ks_stat, ks_pval = stats.ks_2samp(train_vals, test_vals)
                col_results["ks_statistic"] = float(ks_stat)
                col_results["ks_pvalue"] = float(ks_pval)
                
                if ks_pval < 0.05:  # Significant drift
                    results["alerts"].append({
                        "feature": col,
                        "method": "ks_test",
                        "statistic": float(ks_stat),
                        "pvalue": float(ks_pval)
                    })
            except Exception as e:
                logger.debug(f"  KS test failed for {col}: {e}")
                col_results["ks_statistic"] = None
                col_results["ks_pvalue"] = None
        
        results["features"][col] = col_results
    
    # Save report
    if drift_cfg.get("save_report", True):
        output_file = drift_cfg.get("output_file", "models/drift_report.json")
        save_json(results, output_file)
        logger.info(f"Drift detection: {len(results['alerts'])} alerts, saved to {output_file}")
    
    return results


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    PSI interpretation:
    - < 0.1: No significant shift
    - 0.1 - 0.15: Moderate shift
    - > 0.15: Significant shift
    
    Args:
        expected: Expected distribution (train)
        actual: Actual distribution (test)
        bins: Number of bins for discretization
    
    Returns:
        PSI value
    """
    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates
    
    if len(breakpoints) < 2:
        return 0.0
    
    # Bin the data
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # Calculate PSI
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    
    return psi


def prediction_intervals(
    model: RegressorMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute prediction intervals using residual bootstrap.
    
    Args:
        model: Fitted model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        config: Configuration dict
    
    Returns:
        Dictionary with prediction intervals
    """
    pi_cfg = config.get("uncertainty", {}).get("prediction_intervals", {})
    if not pi_cfg.get("enabled", True):
        return {}
    
    logger.info("Computing prediction intervals (residual bootstrap)")
    
    method = pi_cfg.get("method", "residual_bootstrap")
    n_bootstraps = int(pi_cfg.get("n_bootstraps", 100))
    confidence_levels = pi_cfg.get("confidence_levels", [0.8, 0.9])
    
    # Get predictions and residuals on training set
    y_train_pred = model.predict(X_train)
    train_residuals = y_train - y_train_pred
    
    # Get test predictions
    y_test_pred = model.predict(X_test)
    
    # Bootstrap residuals
    intervals = {}
    for conf_level in confidence_levels:
        alpha = 1 - conf_level
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        
        # Resample residuals
        bootstrap_preds = []
        for _ in range(n_bootstraps):
            sampled_residuals = np.random.choice(train_residuals, size=len(y_test), replace=True)
            bootstrap_pred = y_test_pred + sampled_residuals
            bootstrap_preds.append(bootstrap_pred)
        
        bootstrap_preds = np.array(bootstrap_preds)
        
        # Compute percentiles
        lower_bound = np.percentile(bootstrap_preds, lower_q * 100, axis=0)
        upper_bound = np.percentile(bootstrap_preds, upper_q * 100, axis=0)
        
        # Coverage (how many true values fall within interval)
        coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
        
        # Average interval width
        avg_width = np.mean(upper_bound - lower_bound)
        
        intervals[f"{int(conf_level*100)}%"] = {
            "lower_bound": lower_bound.tolist(),
            "upper_bound": upper_bound.tolist(),
            "coverage": float(coverage),
            "average_width": float(avg_width),
            "target_coverage": float(conf_level)
        }
        
        logger.info(f"  {int(conf_level*100)}% PI: coverage={coverage:.3f}, avg_width={avg_width:.1f}")
    
    return intervals
