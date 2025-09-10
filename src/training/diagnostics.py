"""
Modulo per diagnostica avanzata e controlli di stabilità dei modelli.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import learning_curve
from sklearn.base import BaseEstimator
from scipy import stats
import warnings

from utils.logger import get_logger

logger = get_logger(__name__)


class ModelDiagnostics:
    """Classe per diagnostica completa dei modelli."""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.diagnostics_dir = output_dir / "diagnostics"
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurazione diagnostica
        self.stability_config = config.get("training", {}).get("stability_checks", {})
        self.enabled = self.stability_config.get("enabled", False)
        
    def analyze_learning_curves(self, 
                               model: BaseEstimator, 
                               X: np.ndarray, 
                               y: np.ndarray, 
                               model_name: str,
                               cv=None) -> Dict[str, Any]:
        """Analizza learning curves per detecting overfitting."""
        
        if not self.stability_config.get("learning_curves", {}).get("enabled", False):
            return {}
        
        try:
            lc_config = self.stability_config.get("learning_curves", {})
            train_sizes = lc_config.get("train_sizes", [0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
            cv_folds = lc_config.get("cv_folds", 5)
            
            # Calcola learning curves
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y, 
                train_sizes=train_sizes,
                cv=cv_folds,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                random_state=42
            )
            
            # Statistiche
            train_mean = -train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            val_mean = -val_scores.mean(axis=1)
            val_std = val_scores.std(axis=1)
            
            # Plot learning curves
            if lc_config.get("save_plots", True):
                plt.figure(figsize=(10, 6))
                plt.plot(train_sizes_abs, train_mean, 'o-', label='Training RMSE')
                plt.fill_between(train_sizes_abs, train_mean - train_std, 
                               train_mean + train_std, alpha=0.1)
                
                plt.plot(train_sizes_abs, val_mean, 'o-', label='Validation RMSE')
                plt.fill_between(train_sizes_abs, val_mean - val_std, 
                               val_mean + val_std, alpha=0.1)
                
                plt.xlabel('Training Set Size')
                plt.ylabel('RMSE')
                plt.title(f'Learning Curves - {model_name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plot_path = self.diagnostics_dir / f"{model_name}_learning_curves.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Learning curves salvate: {plot_path}")
            
            # Analisi gap overfitting
            final_train_rmse = train_mean[-1]
            final_val_rmse = val_mean[-1]
            overfitting_gap = final_val_rmse - final_train_rmse
            overfitting_ratio = final_val_rmse / final_train_rmse if final_train_rmse > 0 else np.inf
            
            return {
                "train_sizes": train_sizes_abs.tolist(),
                "train_rmse_mean": train_mean.tolist(),
                "train_rmse_std": train_std.tolist(),
                "val_rmse_mean": val_mean.tolist(),
                "val_rmse_std": val_std.tolist(),
                "final_overfitting_gap": float(overfitting_gap),
                "final_overfitting_ratio": float(overfitting_ratio),
                "convergence_score": self._calculate_convergence_score(val_mean)
            }
            
        except Exception as e:
            logger.error(f"Errore nell'analisi learning curves per {model_name}: {e}")
            return {}
    
    def analyze_residuals(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         model_name: str) -> Dict[str, Any]:
        """Analisi dei residui per identificare pattern e problemi."""
        
        if not self.stability_config.get("residual_analysis", {}).get("enabled", False):
            return {}
        
        try:
            residuals = y_true - y_pred
            ra_config = self.stability_config.get("residual_analysis", {})
            
            # Statistiche residui
            residual_stats = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "skewness": float(stats.skew(residuals)),
                "kurtosis": float(stats.kurtosis(residuals)),
                "jarque_bera_pvalue": float(stats.jarque_bera(residuals)[1])
            }
            
            # Test normalità residui
            _, shapiro_p = stats.shapiro(residuals[:min(5000, len(residuals))])  # Shapiro max 5000
            residual_stats["shapiro_pvalue"] = float(shapiro_p)
            
            # Plot residui
            if ra_config.get("save_plots", True):
                plot_types = ra_config.get("plot_types", ["scatter", "histogram", "qq"])
                
                if len(plot_types) > 1:
                    fig, axes = plt.subplots(1, len(plot_types), figsize=(5*len(plot_types), 5))
                    if len(plot_types) == 1:
                        axes = [axes]
                else:
                    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
                    axes = [axes]
                
                for i, plot_type in enumerate(plot_types):
                    ax = axes[i] if len(plot_types) > 1 else axes[0]
                    
                    if plot_type == "scatter":
                        ax.scatter(y_pred, residuals, alpha=0.6)
                        ax.axhline(y=0, color='red', linestyle='--')
                        ax.set_xlabel('Predicted Values')
                        ax.set_ylabel('Residuals')
                        ax.set_title('Residuals vs Predicted')
                        
                    elif plot_type == "histogram":
                        ax.hist(residuals, bins=50, alpha=0.7, density=True)
                        ax.axvline(x=0, color='red', linestyle='--')
                        ax.set_xlabel('Residuals')
                        ax.set_ylabel('Density')
                        ax.set_title('Residuals Distribution')
                        
                    elif plot_type == "qq":
                        stats.probplot(residuals, dist="norm", plot=ax)
                        ax.set_title('Q-Q Plot (Normal)')
                
                plt.tight_layout()
                plot_path = self.diagnostics_dir / f"{model_name}_residuals.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Analisi residui salvata: {plot_path}")
            
            return residual_stats
            
        except Exception as e:
            logger.error(f"Errore nell'analisi residui per {model_name}: {e}")
            return {}
    
    def check_overfitting_thresholds(self, 
                                   train_metrics: Dict[str, float], 
                                   test_metrics: Dict[str, float], 
                                   model_name: str) -> Dict[str, Any]:
        """Controlla soglie di overfitting e genera alert."""
        
        thresholds = self.stability_config.get("overfitting_thresholds", {})
        
        # Gap R²
        r2_gap = train_metrics.get("r2", 0) - test_metrics.get("r2", 0)
        r2_warning = thresholds.get("r2_gap_warning", 0.15)
        r2_critical = thresholds.get("r2_gap_critical", 0.25)
        
        # Ratio RMSE
        train_rmse = train_metrics.get("rmse", 1)
        test_rmse = test_metrics.get("rmse", 1)
        rmse_ratio = test_rmse / train_rmse if train_rmse > 0 else np.inf
        rmse_warning = thresholds.get("rmse_ratio_warning", 1.5)
        rmse_critical = thresholds.get("rmse_ratio_critical", 2.0)
        
        # Classificazione overfitting
        overfitting_level = "none"
        alerts = []
        
        if r2_gap > r2_critical or rmse_ratio > rmse_critical:
            overfitting_level = "critical"
            alerts.append(f"CRITICAL: Severe overfitting detected")
        elif r2_gap > r2_warning or rmse_ratio > rmse_warning:
            overfitting_level = "warning"
            alerts.append(f"WARNING: Moderate overfitting detected")
        
        if r2_gap > r2_warning:
            alerts.append(f"R² gap: {r2_gap:.3f} (threshold: {r2_warning:.3f})")
        if rmse_ratio > rmse_warning:
            alerts.append(f"RMSE ratio: {rmse_ratio:.3f} (threshold: {rmse_warning:.3f})")
        
        return {
            "overfitting_level": overfitting_level,
            "r2_gap": float(r2_gap),
            "rmse_ratio": float(rmse_ratio),
            "alerts": alerts,
            "recommendations": self._get_overfitting_recommendations(overfitting_level)
        }
    
    def _calculate_convergence_score(self, val_scores: np.ndarray) -> float:
        """Calcola score di convergenza basato su stabilità delle validation scores."""
        if len(val_scores) < 3:
            return 0.0
        
        # Variazione negli ultimi 3 punti
        last_3 = val_scores[-3:]
        variation = np.std(last_3) / np.mean(last_3) if np.mean(last_3) > 0 else 1.0
        
        # Score: più basso è meglio (meno variazione = più convergenza)
        convergence_score = 1.0 / (1.0 + variation)
        return float(convergence_score)
    
    def _get_overfitting_recommendations(self, level: str) -> List[str]:
        """Genera raccomandazioni basate sul livello di overfitting."""
        if level == "critical":
            return [
                "Reduce model complexity (max_depth, num_leaves)",
                "Increase regularization (reg_alpha, reg_lambda)",
                "Reduce training data or use early stopping",
                "Consider ensemble methods with diversity"
            ]
        elif level == "warning":
            return [
                "Tune regularization parameters",
                "Consider cross-validation for hyperparameter selection",
                "Monitor learning curves during training"
            ]
        else:
            return ["Model shows good generalization"]


def run_comprehensive_diagnostics(models_results: Dict[str, Any], 
                                config: Dict[str, Any], 
                                models_dir: Path) -> Dict[str, Any]:
    """Esegue diagnostica completa su tutti i modelli."""
    
    diagnostics = ModelDiagnostics(config, models_dir)
    
    if not diagnostics.enabled:
        logger.info("Diagnostica avanzata disabilitata")
        return {}
    
    logger.info("Avvio diagnostica completa modelli")
    
    comprehensive_results = {
        "overfitting_analysis": {},
        "model_rankings": {},
        "stability_scores": {},
        "recommendations": {}
    }
    
    # Analisi overfitting per ogni modello
    for model_name, results in models_results.get("models", {}).items():
        train_metrics = results.get("metrics_train", {})
        test_metrics = results.get("metrics_test", {})
        
        overfitting_analysis = diagnostics.check_overfitting_thresholds(
            train_metrics, test_metrics, model_name
        )
        comprehensive_results["overfitting_analysis"][model_name] = overfitting_analysis
        
        # Log alerts critici
        if overfitting_analysis["overfitting_level"] == "critical":
            logger.warning(f"CRITICAL OVERFITTING: {model_name}")
            for alert in overfitting_analysis["alerts"]:
                logger.warning(f"  - {alert}")
    
    # Ranking modelli per stabilità
    stability_ranking = []
    for model_name, analysis in comprehensive_results["overfitting_analysis"].items():
        stability_score = 1.0 if analysis["overfitting_level"] == "none" else \
                         0.5 if analysis["overfitting_level"] == "warning" else 0.0
        
        stability_ranking.append({
            "model": model_name,
            "stability_score": stability_score,
            "overfitting_level": analysis["overfitting_level"]
        })
    
    stability_ranking.sort(key=lambda x: x["stability_score"], reverse=True)
    comprehensive_results["stability_scores"] = stability_ranking
    
    # Raccomandazioni generali
    critical_models = [r for r in stability_ranking if r["overfitting_level"] == "critical"]
    warning_models = [r for r in stability_ranking if r["overfitting_level"] == "warning"]
    
    general_recommendations = []
    if critical_models:
        general_recommendations.append(f"{len(critical_models)} models show critical overfitting")
        general_recommendations.append("Consider reducing model complexity globally")
    if warning_models:
        general_recommendations.append(f"{len(warning_models)} models show moderate overfitting")
        general_recommendations.append("Fine-tune regularization parameters")
    
    if not critical_models and not warning_models:
        general_recommendations.append("Models show good generalization overall")
    
    comprehensive_results["recommendations"]["general"] = general_recommendations
    
    logger.info(f"Diagnostica completata: {len(critical_models)} critical, {len(warning_models)} warning")
    
    return comprehensive_results