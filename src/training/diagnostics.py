"""Advanced diagnostics: residual analysis, drift detection, prediction intervals."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from utils.logger import get_logger
from utils.io import save_json

logger = get_logger(__name__)


def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute residuals (prediction errors)."""
    return y_true - y_pred


def residual_analysis(
    model_key: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: Optional[pd.DataFrame],
    config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Perform comprehensive residual analysis.
    
    Analyzes:
    - Overall residual distribution
    - Residuals by group (ZonaOmi, Categoria, price quartile)
    - Worst predictions identification
    - Residual plots (vs predicted, vs actual, distribution)
    
    Args:
        model_key: Model name
        y_true: True target values (original scale)
        y_pred: Predicted values (original scale)
        X: Feature DataFrame (for grouping)
        config: Configuration dict
        output_dir: Output directory for plots and results
    
    Returns:
        Dictionary with residual analysis results
    """
    res_cfg = config.get("diagnostics", {}).get("residual_analysis", {})
    if not res_cfg.get("enabled", False):
        logger.info(f"[{model_key}] Residual analysis disabled")
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
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
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
                    logger.warning(f"  Could not create price quartiles: {e}")
                    continue
            else:
                if group_col not in X.columns:
                    logger.debug(f"  Group column {group_col} not found in X")
                    continue
                group_data = pd.DataFrame({"residual": residuals, "group": X[group_col].values})
            
            try:
                group_stats = group_data.groupby("group")["residual"].agg([
                    "mean", "std", "median", "count"
                ]).to_dict(orient="index")
                
                # Convert to serializable format
                results["by_group"][group_col] = {
                    str(k): {
                        "mean": float(v["mean"]),
                        "std": float(v["std"]),
                        "median": float(v["median"]),
                        "count": int(v["count"])
                    }
                    for k, v in group_stats.items()
                }
                logger.info(f"  Grouped by {group_col}: {len(group_stats)} groups")
            except Exception as e:
                logger.warning(f"  Failed to compute group stats for {group_col}: {e}")
    
    # Worst predictions
    if res_cfg.get("save_worst_predictions", True):
        top_n = int(res_cfg.get("top_n_worst", 50))
        abs_residuals = np.abs(residuals)
        worst_indices = np.argsort(abs_residuals)[-top_n:][::-1]
        
        worst_df = pd.DataFrame({
            "true": y_true[worst_indices],
            "predicted": y_pred[worst_indices],
            "residual": residuals[worst_indices],
            "abs_residual": abs_residuals[worst_indices],
            "pct_error": 100 * abs_residuals[worst_indices] / (y_true[worst_indices] + 1e-8)
        })
        
        worst_file = output_dir / f"{model_key}_worst_predictions.csv"
        worst_df.to_csv(worst_file, index=False)
        logger.info(f"  Saved worst {top_n} predictions: {worst_file.name}")
    
    # Residual plots
    plots = res_cfg.get("plots", [])
    if plots:
        plot_dir = output_dir / f"{model_key}_residual_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        if "residual_vs_predicted" in plots:
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.5, s=10, edgecolors='none')
            plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
            plt.xlabel("Predicted Value")
            plt.ylabel("Residual (True - Predicted)")
            plt.title(f"{model_key}: Residuals vs Predicted")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_dir / "residual_vs_predicted.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        if "residual_vs_actual" in plots:
            plt.figure(figsize=(10, 6))
            plt.scatter(y_true, residuals, alpha=0.5, s=10, edgecolors='none')
            plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
            plt.xlabel("Actual Value")
            plt.ylabel("Residual (True - Predicted)")
            plt.title(f"{model_key}: Residuals vs Actual")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_dir / "residual_vs_actual.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        if "residual_distribution" in plots:
            plt.figure(figsize=(10, 6))
            plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
            plt.axvline(x=np.mean(residuals), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(residuals):.1f}')
            plt.axvline(x=np.median(residuals), color='b', linestyle='--', linewidth=2, label=f'Median: {np.median(residuals):.1f}')
            plt.xlabel("Residual")
            plt.ylabel("Frequency")
            plt.title(f"{model_key}: Residual Distribution")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_dir / "residual_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        if "qq_plot" in plots:
            plt.figure(figsize=(8, 8))
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title(f"{model_key}: Q-Q Plot (Normality Check)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_dir / "qq_plot.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        logger.info(f"  Saved residual plots: {plot_dir.name}")
    
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
    - PSI (Population Stability Index): PSI > 0.15 indicates significant drift
    - Kolmogorov-Smirnov test: p < 0.05 indicates significant difference
    
    Args:
        X_train: Training features
        X_test: Test features
        config: Configuration dict
        output_dir: Output directory for drift report
    
    Returns:
        Dictionary with drift analysis results
    """
    drift_cfg = config.get("diagnostics", {}).get("drift_detection", {})
    if not drift_cfg.get("enabled", False):
        logger.info("Drift detection disabled")
        return {}
    
    logger.info("Performing drift detection (train vs test)")
    
    methods = drift_cfg.get("methods", ["psi", "ks_test"])
    alert_threshold = float(drift_cfg.get("alert_threshold", 0.15))
    
    results = {"features": {}, "alerts": [], "summary": {}}
    
    # Only check numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    common_cols = [c for c in numeric_cols if c in X_test.columns]
    
    psi_alerts = 0
    ks_alerts = 0
    
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
                        "threshold": alert_threshold,
                        "severity": "high" if psi > 0.25 else "moderate"
                    })
                    psi_alerts += 1
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
                        "pvalue": float(ks_pval),
                        "severity": "high" if ks_pval < 0.01 else "moderate"
                    })
                    ks_alerts += 1
            except Exception as e:
                logger.debug(f"  KS test failed for {col}: {e}")
                col_results["ks_statistic"] = None
                col_results["ks_pvalue"] = None
        
        if col_results:
            results["features"][col] = col_results
    
    # Summary
    results["summary"] = {
        "total_features_checked": len(common_cols),
        "psi_alerts": psi_alerts,
        "ks_alerts": ks_alerts,
        "total_alerts": len(results["alerts"])
    }
    
    # Save report
    if drift_cfg.get("save_report", True):
        output_file = output_dir / "drift_report.json"
        save_json(results, str(output_file))
        logger.info(
            f"Drift detection: {len(results['alerts'])} alerts "
            f"({psi_alerts} PSI, {ks_alerts} KS) on {len(common_cols)} features"
        )
        logger.info(f"  Saved: {output_file.name}")
    
    return results


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    PSI interpretation:
    - < 0.1: No significant shift
    - 0.1 - 0.15: Moderate shift
    - > 0.15: Significant shift (model may need retraining)
    
    Args:
        expected: Expected distribution (train)
        actual: Actual distribution (test)
        bins: Number of bins for discretization
    
    Returns:
        PSI value
    """
    # Create bins based on expected distribution percentiles
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
    
    return abs(psi)


def prediction_intervals(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
    model_key: str,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Compute prediction intervals using residual bootstrap.
    
    Method: Residual Bootstrap
    1. Fit model on training data
    2. Compute training residuals
    3. For each test prediction, resample residuals and add to prediction
    4. Compute percentiles to form confidence intervals
    
    Args:
        model: Fitted model (must have .predict() method)
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target (for coverage evaluation)
        config: Configuration dict
        model_key: Model name
        output_dir: Output directory
    
    Returns:
        Dictionary with prediction intervals and coverage metrics
    """
    pi_cfg = config.get("uncertainty", {}).get("prediction_intervals", {})
    if not pi_cfg.get("enabled", False):
        logger.info(f"[{model_key}] Prediction intervals disabled")
        return {}
    
    logger.info(f"[{model_key}] Computing prediction intervals (residual bootstrap)")
    
    n_bootstraps = int(pi_cfg.get("n_bootstraps", 100))
    confidence_levels = pi_cfg.get("confidence_levels", [0.8, 0.9])
    
    # Get predictions and residuals on training set
    try:
        y_train_pred = model.predict(X_train)
        train_residuals = y_train - y_train_pred
    except Exception as e:
        logger.warning(f"  Failed to compute training residuals: {e}")
        return {}
    
    # Get test predictions
    try:
        y_test_pred = model.predict(X_test)
    except Exception as e:
        logger.warning(f"  Failed to get test predictions: {e}")
        return {}
    
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
        
        # Interval width as % of actual price (not prediction)
        # Use y_test instead of y_test_pred to avoid explosion when predictions are very small
        avg_width_pct = 100.0 * avg_width / np.mean(y_test)
        
        intervals[f"{int(conf_level*100)}%"] = {
            "coverage": float(coverage),
            "average_width": float(avg_width),
            "average_width_pct": float(avg_width_pct),
            "target_coverage": float(conf_level)
        }
        
        logger.info(
            f"  {int(conf_level*100)}% PI: coverage={coverage:.3f} "
            f"(target={conf_level:.2f}), avg_width={avg_width:.1f}"
        )
    
    # Save to file
    output_file = output_dir / f"{model_key}_prediction_intervals.json"
    save_json(intervals, str(output_file))
    logger.info(f"  Saved: {output_file.name}")
    
    return intervals
