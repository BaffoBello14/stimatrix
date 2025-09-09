from __future__ import annotations

from typing import Dict, Any, Union, List
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score
)


def select_primary_value(primary_metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcola il valore della metrica primaria per l'ottimizzazione.
    
    Le metriche negative sono usate per massimizzazione in Optuna.
    """
    pm = primary_metric.lower()
    if pm == "r2":
        return r2_score(y_true, y_pred)
    elif pm == "neg_mean_squared_error":
        return -mean_squared_error(y_true, y_pred)
    elif pm == "neg_root_mean_squared_error":
        return -np.sqrt(mean_squared_error(y_true, y_pred))
    elif pm == "neg_mean_absolute_error":
        return -mean_absolute_error(y_true, y_pred)
    elif pm == "neg_mean_absolute_percentage_error":
        return -mean_absolute_percentage_error(y_true, y_pred)
    else:
        raise ValueError(f"Unknown primary metric: {primary_metric}")


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcola tutte le metriche di regressione standard.
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": r2_score(y_true, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "explained_variance": explained_variance_score(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred)
    }


def calculate_overfit_metrics(
    metrics_train: Dict[str, float], 
    metrics_test: Dict[str, float]
) -> Dict[str, float]:
    """
    Calcola metriche che quantificano l'overfitting tra train e test set.
    
    Args:
        metrics_train: Metriche sul training set
        metrics_test: Metriche sul test set
    
    Returns:
        Dict con varie misure di overfitting
    """
    overfit_metrics = {}
    
    # Gap assoluto per R2 e explained variance (metriche da massimizzare)
    overfit_metrics["gap_r2"] = metrics_train["r2"] - metrics_test["r2"]
    overfit_metrics["gap_explained_variance"] = (
        metrics_train["explained_variance"] - metrics_test["explained_variance"]
    )
    
    # Ratio per metriche di errore (dovrebbero essere simili tra train e test)
    # Un ratio > 1 indica che il test error è maggiore del train error
    for metric in ["rmse", "mse", "mae", "mape", "medae"]:
        if metrics_train[metric] > 0:  # Evita divisione per zero
            overfit_metrics[f"ratio_{metric}"] = metrics_test[metric] / metrics_train[metric]
            overfit_metrics[f"delta_{metric}"] = metrics_test[metric] - metrics_train[metric]
        else:
            overfit_metrics[f"ratio_{metric}"] = float('inf') if metrics_test[metric] > 0 else 1.0
            overfit_metrics[f"delta_{metric}"] = metrics_test[metric]
    
    return overfit_metrics


def calculate_weighted_score(
    metrics: Dict[str, float],
    weights: Dict[str, float],
    normalize: bool = True
) -> float:
    """
    Calcola uno score pesato combinando multiple metriche.
    
    Args:
        metrics: Dizionario con le metriche
        weights: Pesi per ogni metrica (devono sommare a 1.0)
        normalize: Se True, normalizza le metriche prima di combinarle
    
    Returns:
        Score combinato
    """
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("I pesi devono sommare a 1.0")
    
    score = 0.0
    
    for metric_name, weight in weights.items():
        if metric_name not in metrics:
            raise ValueError(f"Metrica {metric_name} non trovata")
        
        value = metrics[metric_name]
        
        # Normalizzazione opzionale basata sui range tipici
        if normalize:
            if metric_name == "r2":
                # R2 è già tra -inf e 1, mappiamo a [0, 1]
                normalized_value = max(0, value)
            elif metric_name in ["rmse", "mse", "mae"]:
                # Per metriche di errore, invertiamo e normalizziamo
                # Assumiamo un range tipico basato sui dati
                if metric_name == "rmse":
                    # Basato sui risultati visti, RMSE tipico è tra 40k e 60k
                    normalized_value = 1.0 - min(value / 100000, 1.0)
                elif metric_name == "mae":
                    # MAE tipico tra 20k e 40k
                    normalized_value = 1.0 - min(value / 80000, 1.0)
                else:
                    # MSE molto variabile, usiamo scala logaritmica
                    normalized_value = 1.0 / (1.0 + np.log1p(value / 1e9))
            elif metric_name == "mape":
                # MAPE in percentuale, invertiamo
                normalized_value = 1.0 - min(value, 1.0)
            else:
                normalized_value = value
        else:
            normalized_value = value
        
        score += weight * normalized_value
    
    return score


def calculate_robust_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    primary_metric: str = "neg_mean_absolute_error",
    secondary_metrics: List[str] = None,
    outlier_robust: bool = True
) -> float:
    """
    Calcola uno score robusto che considera multiple metriche e outlier.
    
    Args:
        y_true: Valori veri
        y_pred: Valori predetti
        primary_metric: Metrica principale
        secondary_metrics: Metriche secondarie da considerare
        outlier_robust: Se True, applica tecniche per ridurre l'impatto degli outlier
    
    Returns:
        Score robusto
    """
    if outlier_robust:
        # Identifica potenziali outlier usando IQR sui residui
        residuals = np.abs(y_true - y_pred)
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        outlier_mask = residuals > (q3 + 1.5 * iqr)
        
        # Se ci sono troppi outlier (>10%), usa un approccio più conservativo
        if np.sum(outlier_mask) > 0.1 * len(y_true):
            # Usa winsorization invece di rimozione
            residuals_winsorized = np.clip(residuals, 0, q3 + 1.5 * iqr)
            y_pred_adjusted = np.where(
                y_pred > y_true,
                y_true + residuals_winsorized,
                y_true - residuals_winsorized
            )
            y_pred = y_pred_adjusted
        else:
            # Rimuovi outlier estremi
            clean_mask = ~outlier_mask
            y_true = y_true[clean_mask]
            y_pred = y_pred[clean_mask]
    
    # Calcola metrica primaria
    primary_score = select_primary_value(primary_metric, y_true, y_pred)
    
    # Se non ci sono metriche secondarie, ritorna il primary score
    if not secondary_metrics:
        return primary_score
    
    # Calcola tutte le metriche
    all_metrics = calculate_all_metrics(y_true, y_pred)
    
    # Combina primary e secondary metrics con pesi decrescenti
    weights = [0.6]  # Peso per metrica primaria
    remaining_weight = 0.4
    n_secondary = len(secondary_metrics)
    
    if n_secondary > 0:
        # Distribuisci il peso rimanente tra le metriche secondarie
        secondary_weights = [remaining_weight / n_secondary] * n_secondary
        weights.extend(secondary_weights)
    
    # Calcola score combinato
    combined_score = primary_score * weights[0]
    
    for i, metric_name in enumerate(secondary_metrics):
        metric_value = select_primary_value(metric_name, y_true, y_pred)
        combined_score += metric_value * weights[i + 1]
    
    return combined_score


def evaluate_model_stability(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bootstrap: int = 10,
    sample_size: float = 0.8,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Valuta la stabilità del modello usando bootstrap sampling.
    
    Un modello stabile dovrebbe avere bassa varianza nelle predizioni
    su diversi sottocampioni del test set.
    """
    np.random.seed(seed)
    n_samples = len(X_test)
    sample_size_n = int(n_samples * sample_size)
    
    metrics_list = []
    predictions_std = []
    
    for i in range(n_bootstrap):
        # Bootstrap sampling
        indices = np.random.choice(n_samples, sample_size_n, replace=True)
        
        if hasattr(X_test, 'iloc'):
            X_sample = X_test.iloc[indices]
            y_sample = y_test.iloc[indices]
        else:
            X_sample = X_test[indices]
            y_sample = y_test[indices]
        
        # Predizioni
        y_pred = model.predict(X_sample)
        
        # Calcola metriche
        metrics = calculate_all_metrics(y_sample, y_pred)
        metrics_list.append(metrics)
        
        # Traccia variabilità delle predizioni per campioni comuni
        if i == 0:
            common_indices = indices[:100]  # Primi 100 campioni
            predictions_matrix = [y_pred[:100]]
        else:
            # Trova indici comuni
            common_mask = np.isin(indices[:100], common_indices)
            if np.any(common_mask):
                predictions_matrix.append(y_pred[:100][common_mask])
    
    # Calcola statistiche di stabilità
    stability_metrics = {}
    
    # Media e deviazione standard per ogni metrica
    for metric in metrics_list[0].keys():
        values = [m[metric] for m in metrics_list]
        stability_metrics[f"{metric}_mean"] = np.mean(values)
        stability_metrics[f"{metric}_std"] = np.std(values)
        stability_metrics[f"{metric}_cv"] = np.std(values) / (np.mean(values) + 1e-10)  # Coefficient of variation
    
    # Stabilità delle predizioni
    if predictions_matrix:
        pred_std = np.mean([np.std(preds) for preds in predictions_matrix if len(preds) > 1])
        stability_metrics["prediction_stability"] = 1.0 / (1.0 + pred_std)  # Normalizzato tra 0 e 1
    
    return stability_metrics