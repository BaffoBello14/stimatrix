"""
Sistema di valutazione avanzato con supporto multi-scala.
Ispirato al sistema robusto di evaluation di RealEstatePricePrediction.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, 
    mean_absolute_percentage_error, explained_variance_score
)
from utils.logger import get_logger
from utils.detailed_logging import DetailedLogger

logger = get_logger(__name__)


class MultiScaleEvaluation:
    """Sistema di valutazione multi-scala per modelli con trasformazioni target."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza evaluator multi-scala.
        
        Args:
            config: Configurazione evaluation
        """
        self.config = config.get('evaluation', {})
        self.metrics_config = config.get('training', {})
        self.target_config = config.get('target', {})
        
        # Metriche da calcolare
        self.metrics_to_calculate = self.config.get('metrics', [
            'mae', 'mse', 'rmse', 'r2', 'mape', 'explained_variance'
        ])
        
        logger.info(f"Multi-Scale Evaluator configurato - Metriche: {self.metrics_to_calculate}")
    
    def evaluate_model_dual_scale(
        self,
        model: Any,
        model_name: str,
        X_test: pd.DataFrame,
        y_test_transformed: pd.Series,
        y_test_original: pd.Series,
        transform_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Valuta modello sia su scala trasformata che originale.
        
        Args:
            model: Modello addestrato
            model_name: Nome modello
            X_test: Features test
            y_test_transformed: Target test trasformato
            y_test_original: Target test scala originale
            transform_info: Info trasformazioni applicate
            
        Returns:
            Dict con risultati dual-scale
        """
        logger.info(f"🎯 Valutazione dual-scale per {model_name}...")
        
        results = {
            'model_name': model_name,
            'transform_info': transform_info,
            'transformed_scale': {},
            'original_scale': {},
            'scale_comparison': {},
            'predictions': {}
        }
        
        try:
            # Predizioni in scala trasformata
            y_pred_transformed = model.predict(X_test)
            results['predictions']['transformed'] = y_pred_transformed
            
            # Metriche scala trasformata
            results['transformed_scale'] = self._calculate_metrics(
                y_test_transformed, y_pred_transformed, scale_name="transformed"
            )
            
            # Inverse transform per scala originale
            y_pred_original = self._inverse_transform_predictions(
                y_pred_transformed, transform_info
            )
            results['predictions']['original'] = y_pred_original
            
            # Metriche scala originale
            results['original_scale'] = self._calculate_metrics(
                y_test_original, y_pred_original, scale_name="original"
            )
            
            # Confronto scale
            results['scale_comparison'] = self._compare_scales(
                results['transformed_scale'], results['original_scale']
            )
            
            # Log risultati principali
            self._log_dual_scale_results(model_name, results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Errore valutazione dual-scale per {model_name}: {e}")
            results['error'] = str(e)
            return results
    
    def evaluate_multiple_models_dual_scale(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test_transformed: pd.Series,
        y_test_original: pd.Series,
        transform_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Valuta modelli multipli con dual-scale evaluation.
        
        Args:
            models: Dict con modelli addestrati
            X_test: Features test
            y_test_transformed: Target test trasformato
            y_test_original: Target test originale
            transform_info: Info trasformazioni
            
        Returns:
            Dict con risultati per tutti i modelli
        """
        logger.info("=== VALUTAZIONE MULTI-SCALA MODELLI MULTIPLI ===")
        
        all_results = {
            'models': {},
            'rankings': {},
            'best_models': {},
            'summary_statistics': {}
        }
        
        # Valuta ogni modello
        for model_key, model_data in models.items():
            model = model_data['model']
            model_name = model_data.get('name', model_key)
            
            model_results = self.evaluate_model_dual_scale(
                model, model_name, X_test, y_test_transformed, y_test_original, transform_info
            )
            
            all_results['models'][model_key] = model_results
        
        # Crea rankings per scala
        all_results['rankings'] = self._create_multi_scale_rankings(all_results['models'])
        
        # Identifica migliori modelli per scala
        all_results['best_models'] = self._identify_best_models(all_results['rankings'])
        
        # Statistiche summary
        all_results['summary_statistics'] = self._calculate_summary_statistics(all_results['models'])
        
        # Log risultati finali
        self._log_multi_model_results(all_results)
        
        return all_results
    
    def _calculate_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        scale_name: str = ""
    ) -> Dict[str, float]:
        """Calcola tutte le metriche configurate."""
        metrics = {}
        
        try:
            # Rimuovi valori NaN per calcoli sicuri
            mask = ~(pd.isna(y_true) | pd.isna(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                logger.warning(f"Nessun valore valido per calcolo metriche {scale_name}")
                return {'error': 'no_valid_values'}
            
            # Calcola metriche richieste
            if 'mae' in self.metrics_to_calculate:
                metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
            
            if 'mse' in self.metrics_to_calculate:
                metrics['mse'] = mean_squared_error(y_true_clean, y_pred_clean)
            
            if 'rmse' in self.metrics_to_calculate:
                metrics['rmse'] = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
            
            if 'r2' in self.metrics_to_calculate:
                metrics['r2'] = r2_score(y_true_clean, y_pred_clean)
            
            if 'mape' in self.metrics_to_calculate:
                # MAPE solo per valori positivi
                positive_mask = y_true_clean > 0
                if positive_mask.sum() > 0:
                    metrics['mape'] = mean_absolute_percentage_error(
                        y_true_clean[positive_mask], y_pred_clean[positive_mask]
                    )
                else:
                    metrics['mape'] = np.nan
            
            if 'explained_variance' in self.metrics_to_calculate:
                metrics['explained_variance'] = explained_variance_score(y_true_clean, y_pred_clean)
            
            # Metriche aggiuntive
            metrics['n_samples'] = len(y_true_clean)
            metrics['prediction_range'] = {
                'min': float(y_pred_clean.min()),
                'max': float(y_pred_clean.max()),
                'mean': float(y_pred_clean.mean()),
                'std': float(y_pred_clean.std())
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Errore calcolo metriche {scale_name}: {e}")
            return {'error': str(e)}
    
    def _inverse_transform_predictions(
        self,
        y_pred_transformed: np.ndarray,
        transform_info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Applica inverse transform alle predizioni.
        
        Args:
            y_pred_transformed: Predizioni scala trasformata
            transform_info: Info trasformazioni applicate
            
        Returns:
            Predizioni scala originale
        """
        y_pred_original = y_pred_transformed.copy()
        
        # Inverse log transform
        if transform_info.get('log', False):
            y_pred_original = np.expm1(y_pred_original)
            logger.debug("Applicato inverse log transform (expm1)")
        
        # Altri inverse transforms possibili
        if transform_info.get('sqrt', False):
            y_pred_original = np.square(y_pred_original)
            logger.debug("Applicato inverse sqrt transform")
        
        if transform_info.get('box_cox', False):
            lambda_param = transform_info.get('box_cox_lambda', 1.0)
            if lambda_param != 0:
                y_pred_original = np.power(y_pred_original * lambda_param + 1, 1/lambda_param)
            else:
                y_pred_original = np.exp(y_pred_original)
            logger.debug(f"Applicato inverse Box-Cox transform (lambda={lambda_param})")
        
        return y_pred_original
    
    def _compare_scales(
        self,
        transformed_metrics: Dict[str, Any],
        original_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Confronta metriche tra scale diverse."""
        comparison = {
            'metrics_comparison': {},
            'scale_preference': {},
            'consistency_analysis': {}
        }
        
        # Confronta metriche comuni
        common_metrics = set(transformed_metrics.keys()) & set(original_metrics.keys())
        common_metrics.discard('prediction_range')  # Escludi range prediction
        common_metrics.discard('n_samples')  # Escludi n_samples
        
        for metric in common_metrics:
            if metric in ['error']:  # Skip error keys
                continue
                
            trans_val = transformed_metrics.get(metric)
            orig_val = original_metrics.get(metric)
            
            if trans_val is not None and orig_val is not None and not pd.isna(trans_val) and not pd.isna(orig_val):
                comparison['metrics_comparison'][metric] = {
                    'transformed': trans_val,
                    'original': orig_val,
                    'difference': orig_val - trans_val,
                    'relative_difference': (orig_val - trans_val) / abs(trans_val) if trans_val != 0 else 0
                }
        
        return comparison
    
    def _create_multi_scale_rankings(self, models_results: Dict[str, Any]) -> Dict[str, Any]:
        """Crea rankings per entrambe le scale."""
        rankings = {
            'transformed_scale': {},
            'original_scale': {},
            'combined_ranking': []
        }
        
        # Rankings per scala trasformata
        for metric in self.metrics_to_calculate:
            metric_values = []
            for model_key, model_results in models_results.items():
                if 'error' not in model_results:
                    trans_metrics = model_results.get('transformed_scale', {})
                    if metric in trans_metrics and not pd.isna(trans_metrics[metric]):
                        metric_values.append((model_key, trans_metrics[metric]))
            
            # Ordina (crescente per errori, decrescente per score)
            ascending = metric in ['mae', 'mse', 'rmse', 'mape']
            metric_values.sort(key=lambda x: x[1], reverse=not ascending)
            rankings['transformed_scale'][metric] = metric_values
        
        # Rankings per scala originale
        for metric in self.metrics_to_calculate:
            metric_values = []
            for model_key, model_results in models_results.items():
                if 'error' not in model_results:
                    orig_metrics = model_results.get('original_scale', {})
                    if metric in orig_metrics and not pd.isna(orig_metrics[metric]):
                        metric_values.append((model_key, orig_metrics[metric]))
            
            ascending = metric in ['mae', 'mse', 'rmse', 'mape']
            metric_values.sort(key=lambda x: x[1], reverse=not ascending)
            rankings['original_scale'][metric] = metric_values
        
        return rankings
    
    def _identify_best_models(self, rankings: Dict[str, Any]) -> Dict[str, Any]:
        """Identifica migliori modelli per scala e metrica."""
        best_models = {
            'by_scale': {},
            'overall_best': {},
            'consensus': {}
        }
        
        # Migliori per scala trasformata
        transformed_rankings = rankings.get('transformed_scale', {})
        for metric, ranking in transformed_rankings.items():
            if ranking:
                best_models['by_scale'][f'transformed_{metric}'] = {
                    'model': ranking[0][0],
                    'value': ranking[0][1]
                }
        
        # Migliori per scala originale
        original_rankings = rankings.get('original_scale', {})
        for metric, ranking in original_rankings.items():
            if ranking:
                best_models['by_scale'][f'original_{metric}'] = {
                    'model': ranking[0][0],
                    'value': ranking[0][1]
                }
        
        # Consensus (modello che appare più spesso nei top 3)
        model_appearances = {}
        for scale_rankings in [transformed_rankings, original_rankings]:
            for metric, ranking in scale_rankings.items():
                for i, (model_key, _) in enumerate(ranking[:3]):  # Top 3
                    score = 3 - i  # 3 punti per 1°, 2 per 2°, 1 per 3°
                    model_appearances[model_key] = model_appearances.get(model_key, 0) + score
        
        if model_appearances:
            consensus_model = max(model_appearances, key=model_appearances.get)
            best_models['consensus'] = {
                'model': consensus_model,
                'total_score': model_appearances[consensus_model],
                'appearances': model_appearances
            }
        
        return best_models
    
    def _calculate_summary_statistics(self, models_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calcola statistiche summary per tutti i modelli."""
        summary = {
            'model_count': len(models_results),
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'metrics_summary': {}
        }
        
        # Conta successi/fallimenti
        for model_results in models_results.values():
            if 'error' in model_results:
                summary['failed_evaluations'] += 1
            else:
                summary['successful_evaluations'] += 1
        
        # Statistiche per metrica
        for metric in self.metrics_to_calculate:
            metric_summary = {
                'transformed_scale': {'values': [], 'stats': {}},
                'original_scale': {'values': [], 'stats': {}}
            }
            
            # Raccogli valori per scala
            for model_results in models_results.values():
                if 'error' not in model_results:
                    # Scala trasformata
                    trans_val = model_results.get('transformed_scale', {}).get(metric)
                    if trans_val is not None and not pd.isna(trans_val):
                        metric_summary['transformed_scale']['values'].append(trans_val)
                    
                    # Scala originale
                    orig_val = model_results.get('original_scale', {}).get(metric)
                    if orig_val is not None and not pd.isna(orig_val):
                        metric_summary['original_scale']['values'].append(orig_val)
            
            # Calcola statistiche
            for scale in ['transformed_scale', 'original_scale']:
                values = metric_summary[scale]['values']
                if values:
                    metric_summary[scale]['stats'] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'count': len(values)
                    }
            
            summary['metrics_summary'][metric] = metric_summary
        
        return summary
    
    def _log_dual_scale_results(self, model_name: str, results: Dict[str, Any]) -> None:
        """Log risultati dual-scale per modello."""
        logger.info(f"📊 Risultati {model_name}:")
        
        # Metriche principali
        main_metrics = ['rmse', 'r2', 'mae']
        
        for metric in main_metrics:
            trans_val = results['transformed_scale'].get(metric)
            orig_val = results['original_scale'].get(metric)
            
            if trans_val is not None and orig_val is not None:
                logger.info(f"  {metric.upper()}: Trasf={trans_val:.6f} | Orig={orig_val:.6f}")
        
        # MAPE solo per scala originale (più interpretabile)
        mape_orig = results['original_scale'].get('mape')
        if mape_orig is not None:
            logger.info(f"  MAPE (orig): {mape_orig:.4f} ({mape_orig*100:.2f}%)")
    
    def _log_multi_model_results(self, all_results: Dict[str, Any]) -> None:
        """Log risultati complessivi multi-model."""
        logger.info("=== SUMMARY VALUTAZIONE MULTI-SCALA ===")
        
        summary = all_results['summary_statistics']
        logger.info(f"  📊 Modelli valutati: {summary['model_count']}")
        logger.info(f"  ✅ Successi: {summary['successful_evaluations']}")
        logger.info(f"  ❌ Fallimenti: {summary['failed_evaluations']}")
        
        # Migliori modelli
        best_models = all_results['best_models']
        if 'consensus' in best_models:
            consensus = best_models['consensus']
            logger.info(f"  🏆 Consensus Best Model: {consensus['model']} (score: {consensus['total_score']})")
        
        # Top 3 per RMSE originale
        rankings = all_results['rankings']
        if 'original_scale' in rankings and 'rmse' in rankings['original_scale']:
            top_3_rmse = rankings['original_scale']['rmse'][:3]
            logger.info("  🥇 Top 3 RMSE (scala originale):")
            for i, (model_key, rmse_val) in enumerate(top_3_rmse, 1):
                logger.info(f"    {i}. {model_key}: {rmse_val:.2f}")


class ResidualAnalyzer:
    """Analizzatore residui per diagnostica modelli."""
    
    @staticmethod
    def analyze_residuals(
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizza residui del modello con diagnostiche avanzate.
        
        Args:
            y_true: Valori veri
            y_pred: Predizioni
            model_name: Nome modello
            output_dir: Directory output per plot
            
        Returns:
            Dict con analisi residui
        """
        logger.info(f"📈 Analisi residui per {model_name}...")
        
        # Calcola residui
        residuals = y_true - y_pred
        
        analysis = {
            'basic_stats': {
                'mean': float(residuals.mean()),
                'std': float(residuals.std()),
                'min': float(residuals.min()),
                'max': float(residuals.max()),
                'median': float(residuals.median()),
                'skewness': float(residuals.skew()),
                'kurtosis': float(residuals.kurtosis())
            },
            'distribution_tests': {},
            'heteroscedasticity_test': {},
            'outlier_analysis': {}
        }
        
        # Test normalità residui
        try:
            from scipy.stats import shapiro, jarque_bera
            
            # Shapiro-Wilk (campiona se troppi dati)
            if len(residuals) <= 5000:
                shapiro_stat, shapiro_p = shapiro(residuals.dropna())
                analysis['distribution_tests']['shapiro'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
            
            # Jarque-Bera
            jb_stat, jb_p = jarque_bera(residuals.dropna())
            analysis['distribution_tests']['jarque_bera'] = {
                'statistic': float(jb_stat),
                'p_value': float(jb_p),
                'is_normal': jb_p > 0.05
            }
            
        except Exception as e:
            logger.warning(f"Test normalità falliti: {e}")
        
        # Analisi outlier residui
        residuals_clean = residuals.dropna()
        if len(residuals_clean) > 0:
            q1, q3 = residuals_clean.quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_threshold = 1.5 * iqr
            
            outliers = residuals_clean[(residuals_clean < q1 - outlier_threshold) | 
                                     (residuals_clean > q3 + outlier_threshold)]
            
            analysis['outlier_analysis'] = {
                'n_outliers': len(outliers),
                'outlier_percentage': len(outliers) / len(residuals_clean) * 100,
                'outlier_threshold': float(outlier_threshold),
                'extreme_residuals': {
                    'min': float(outliers.min()) if len(outliers) > 0 else None,
                    'max': float(outliers.max()) if len(outliers) > 0 else None
                }
            }
        
        # Genera plot se richiesto
        if output_dir:
            ResidualAnalyzer._create_residual_plots(
                y_true, y_pred, residuals, model_name, output_dir
            )
        
        return analysis
    
    @staticmethod
    def _create_residual_plots(
        y_true: pd.Series,
        y_pred: np.ndarray,
        residuals: pd.Series,
        model_name: str,
        output_dir: str
    ) -> None:
        """Crea plot diagnostici residui."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Residuals vs Fitted
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title(f'{model_name} - Residuals vs Fitted')
        
        # Plot 2: Q-Q plot
        plt.subplot(1, 2, 2)
        from scipy.stats import probplot
        probplot(residuals.dropna(), dist="norm", plot=plt)
        plt.title(f'{model_name} - Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_residual_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Distribution residui
        plt.figure(figsize=(10, 6))
        plt.hist(residuals.dropna(), bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(residuals.mean(), color='red', linestyle='--', label=f'Mean: {residuals.mean():.2f}')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title(f'{model_name} - Residuals Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_residual_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Salvati plot residui per {model_name} in {output_dir}")


class ModelComparator:
    """Comparatore avanzato per modelli multipli."""
    
    @staticmethod
    def create_comparison_report(
        models_results: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Crea report comparativo dettagliato.
        
        Args:
            models_results: Risultati valutazione modelli
            output_file: File output per salvare report
            
        Returns:
            DataFrame con report comparativo
        """
        logger.info("📊 Creazione report comparativo modelli...")
        
        report_data = []
        
        for model_key, model_results in models_results.items():
            if 'error' in model_results:
                continue
            
            model_name = model_results.get('model_name', model_key)
            
            # Dati base
            row = {
                'Model_Key': model_key,
                'Model_Name': model_name,
                'Transform_Applied': model_results.get('transform_info', {}).get('log', False)
            }
            
            # Metriche scala trasformata
            trans_metrics = model_results.get('transformed_scale', {})
            for metric, value in trans_metrics.items():
                if metric not in ['prediction_range', 'n_samples'] and not pd.isna(value):
                    row[f'Trans_{metric.upper()}'] = value
            
            # Metriche scala originale
            orig_metrics = model_results.get('original_scale', {})
            for metric, value in orig_metrics.items():
                if metric not in ['prediction_range', 'n_samples'] and not pd.isna(value):
                    row[f'Orig_{metric.upper()}'] = value
            
            # Confronti scale
            scale_comp = model_results.get('scale_comparison', {}).get('metrics_comparison', {})
            for metric, comp_data in scale_comp.items():
                row[f'Scale_Diff_{metric.upper()}'] = comp_data.get('relative_difference', 0)
            
            report_data.append(row)
        
        # Crea DataFrame
        report_df = pd.DataFrame(report_data)
        
        if not report_df.empty:
            # Ordina per RMSE originale (se disponibile)
            if 'Orig_RMSE' in report_df.columns:
                report_df = report_df.sort_values('Orig_RMSE').reset_index(drop=True)
            
            # Aggiungi ranking
            if 'Orig_RMSE' in report_df.columns:
                report_df['RMSE_Rank'] = range(1, len(report_df) + 1)
            
            if 'Orig_R2' in report_df.columns:
                report_df['R2_Rank'] = report_df['Orig_R2'].rank(ascending=False).astype(int)
        
        # Salva se richiesto
        if output_file:
            report_df.to_csv(output_file, index=False)
            logger.info(f"Report salvato: {output_file}")
        
        logger.info(f"Report creato per {len(report_df)} modelli")
        return report_df
    
    @staticmethod
    def create_performance_visualization(
        models_results: Dict[str, Any],
        output_dir: str,
        metrics_to_plot: List[str] = ['rmse', 'r2', 'mae']
    ) -> None:
        """Crea visualizzazioni performance modelli."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepara dati per plotting
        plot_data = []
        for model_key, model_results in models_results.items():
            if 'error' in model_results:
                continue
            
            model_name = model_results.get('model_name', model_key)
            
            for scale in ['transformed_scale', 'original_scale']:
                scale_metrics = model_results.get(scale, {})
                for metric in metrics_to_plot:
                    if metric in scale_metrics and not pd.isna(scale_metrics[metric]):
                        plot_data.append({
                            'Model': model_name,
                            'Scale': scale.replace('_scale', '').title(),
                            'Metric': metric.upper(),
                            'Value': scale_metrics[metric]
                        })
        
        if not plot_data:
            logger.warning("Nessun dato per visualizzazioni")
            return
        
        plot_df = pd.DataFrame(plot_data)
        
        # Plot per metrica
        for metric in metrics_to_plot:
            metric_data = plot_df[plot_df['Metric'] == metric.upper()]
            
            if metric_data.empty:
                continue
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=metric_data, x='Model', y='Value', hue='Scale')
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Model Comparison - {metric.upper()}')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/models_comparison_{metric}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot combinato
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics_to_plot, 1):
            plt.subplot(2, 2, i)
            metric_data = plot_df[plot_df['Metric'] == metric.upper()]
            if not metric_data.empty:
                sns.barplot(data=metric_data, x='Model', y='Value', hue='Scale')
                plt.xticks(rotation=45, ha='right')
                plt.title(f'{metric.upper()}')
                if i == 1:
                    plt.legend()
                else:
                    plt.legend().set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/models_comparison_combined.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizzazioni salvate in {output_dir}")