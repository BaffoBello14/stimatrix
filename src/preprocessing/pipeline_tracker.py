"""
Sistema di tracking completo per monitoraggio evoluzione pipeline ML.

Questo modulo traccia l'intero lifecycle della pipeline registrando:
- Evoluzione shape dataset step-by-step con memoria e timing
- Feature engineering: colonne aggiunte/rimosse per ogni operazione
- Outlier detection: campioni rimossi per metodo e categoria
- Encoding results: trasformazioni categoriche applicate
- Model training: iperparametri, performance, tempo training
- Performance metrics: efficienza step, bottleneck identification

Il tracker genera report automatici (JSON/CSV/Excel) per analisi post-execution
e monitoring real-time con alerting per performance anomale.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from utils.logger import get_logger
from utils.detailed_logging import DetailedLogger

logger = get_logger(__name__)


class PipelineTracker:
    """Tracker completo per evoluzione pipeline con statistiche dettagliate."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza pipeline tracker.
        
        Args:
            config: Configurazione generale pipeline
        """
        self.config = config
        self.tracking_enabled = config.get('tracking', {}).get('enabled', True)
        self.save_intermediate = config.get('tracking', {}).get('save_intermediate', False)
        
        # Stato tracking
        self.steps_info = {}
        self.dataset_evolution = []
        self.performance_metrics = {}
        self.start_time = datetime.now()
        self.step_times = {}
        
        # Configurazione salvataggio
        self.output_dir = Path(config.get('paths', {}).get('preprocessed_data', 'data/preprocessed'))
        self.reports_dir = self.output_dir / 'tracking_reports'
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline Tracker inizializzato - Enabled: {self.tracking_enabled}")
    
    def track_step_start(self, step_name: str) -> None:
        """Registra inizio step."""
        if not self.tracking_enabled:
            return
        
        self.step_times[step_name] = {'start': datetime.now()}
        logger.info(f"🚀 Inizio step: {step_name}")
    
    def track_step_completion(
        self,
        step_name: str,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        step_info: Dict[str, Any],
        save_snapshot: bool = False
    ) -> Dict[str, Any]:
        """
        Registra completamento step con evoluzione dataset.
        
        Args:
            step_name: Nome dello step
            df_before: DataFrame prima dello step
            df_after: DataFrame dopo lo step
            step_info: Informazioni specifiche dello step
            save_snapshot: Se salvare snapshot del dataset
            
        Returns:
            Dict con informazioni evoluzione
        """
        if not self.tracking_enabled:
            return {}
        
        # Registra tempo completamento
        if step_name in self.step_times:
            self.step_times[step_name]['end'] = datetime.now()
            self.step_times[step_name]['duration'] = (
                self.step_times[step_name]['end'] - self.step_times[step_name]['start']
            ).total_seconds()
        
        # Traccia evoluzione dataset
        evolution_info = DetailedLogger.log_dataset_evolution(
            df_before, df_after, step_name, step_info
        )
        
        # Aggiungi timing
        evolution_info['timing'] = self.step_times.get(step_name, {})
        
        # Salva in tracking
        self.steps_info[step_name] = step_info
        self.dataset_evolution.append(evolution_info)
        
        # Salva snapshot se richiesto
        if save_snapshot and self.save_intermediate:
            snapshot_path = self.reports_dir / f"snapshot_{step_name}.parquet"
            df_after.to_parquet(snapshot_path)
            evolution_info['snapshot_path'] = str(snapshot_path)
        
        logger.info(f"✅ Completato step: {step_name} in {evolution_info['timing'].get('duration', 0):.1f}s")
        return evolution_info
    
    def track_feature_engineering(
        self,
        features_before: List[str],
        features_after: List[str],
        feature_extraction_results: Dict[str, Any]
    ) -> None:
        """
        Traccia risultati feature engineering.
        
        Args:
            features_before: Features prima dell'estrazione
            features_after: Features dopo l'estrazione
            feature_extraction_results: Risultati estrazione
        """
        if not self.tracking_enabled:
            return
        
        features_added = set(features_after) - set(features_before)
        features_removed = set(features_before) - set(features_after)
        
        fe_tracking = {
            'features_before_count': len(features_before),
            'features_after_count': len(features_after),
            'features_added': list(features_added),
            'features_removed': list(features_removed),
            'net_change': len(features_after) - len(features_before),
            'extraction_results': feature_extraction_results
        }
        
        self.steps_info['feature_engineering'] = fe_tracking
        
        # Log dettagliato
        DetailedLogger.log_feature_extraction_results(
            feature_extraction_results, "FEATURE ENGINEERING"
        )
        
        logger.info(f"🔧 Feature Engineering: {len(features_before)} → {len(features_after)} "
                   f"(+{len(features_added)}, -{len(features_removed)})")
    
    def track_outlier_detection(
        self,
        outliers_info: Dict[str, Any],
        total_samples: int,
        samples_removed: int
    ) -> None:
        """
        Traccia risultati outlier detection.
        
        Args:
            outliers_info: Informazioni outliers
            total_samples: Campioni totali
            samples_removed: Campioni rimossi
        """
        if not self.tracking_enabled:
            return
        
        outlier_tracking = {
            'total_samples': total_samples,
            'samples_removed': samples_removed,
            'removal_percentage': (samples_removed / total_samples) * 100 if total_samples > 0 else 0,
            'outliers_info': outliers_info
        }
        
        self.steps_info['outlier_detection'] = outlier_tracking
        
        # Log dettagliato
        DetailedLogger.log_outlier_detection_summary(outliers_info, total_samples)
    
    def track_encoding_results(
        self,
        encoding_info: Dict[str, Any],
        categorical_columns_before: List[str],
        categorical_columns_after: List[str]
    ) -> None:
        """
        Traccia risultati encoding categoriche.
        
        Args:
            encoding_info: Info encoding
            categorical_columns_before: Colonne categoriche prima
            categorical_columns_after: Colonne categoriche dopo
        """
        if not self.tracking_enabled:
            return
        
        encoding_tracking = {
            'categorical_before': categorical_columns_before,
            'categorical_after': categorical_columns_after,
            'categorical_before_count': len(categorical_columns_before),
            'categorical_after_count': len(categorical_columns_after),
            'encoding_methods_used': encoding_info.get('methods_used', []),
            'encoding_info': encoding_info
        }
        
        self.steps_info['encoding'] = encoding_tracking
        
        logger.info(f"🏷️  Encoding: {len(categorical_columns_before)} → {len(categorical_columns_after)} "
                   f"colonne categoriche")
    
    def track_model_training(
        self,
        model_name: str,
        training_results: Dict[str, Any],
        training_time: float,
        hyperparameters: Dict[str, Any]
    ) -> None:
        """
        Traccia risultati training modello.
        
        Args:
            model_name: Nome modello
            training_results: Risultati training
            training_time: Tempo training in secondi
            hyperparameters: Iperparametri utilizzati
        """
        if not self.tracking_enabled:
            return
        
        if 'model_training' not in self.performance_metrics:
            self.performance_metrics['model_training'] = {}
        
        self.performance_metrics['model_training'][model_name] = {
            'training_time_seconds': training_time,
            'hyperparameters': hyperparameters,
            'training_results': training_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log progresso
        DetailedLogger.log_model_training_progress(
            model_name, 1, 1, 
            training_results.get('best_score', 0),
            training_results.get('best_score', 0),
            training_time
        )
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Genera report completo della pipeline.
        
        Returns:
            Dict con report completo
        """
        if not self.tracking_enabled:
            return {'tracking_disabled': True}
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        comprehensive_report = {
            'pipeline_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration_seconds': total_time,
                'total_duration_formatted': self._format_duration(total_time),
                'steps_completed': len(self.dataset_evolution),
                'tracking_enabled': self.tracking_enabled
            },
            'dataset_evolution': self.dataset_evolution,
            'steps_detailed_info': self.steps_info,
            'performance_metrics': self.performance_metrics,
            'step_timings': self.step_times,
            'summary_statistics': self._calculate_summary_statistics()
        }
        
        # Salva report
        self._save_comprehensive_report(comprehensive_report)
        
        return comprehensive_report
    
    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calcola statistiche summary della pipeline."""
        summary = {
            'dataset_size_evolution': {},
            'timing_analysis': {},
            'feature_evolution': {},
            'memory_usage_evolution': {}
        }
        
        if not self.dataset_evolution:
            return summary
        
        # Evoluzione dimensioni dataset
        initial_shape = self.dataset_evolution[0]['shape_before']
        final_shape = self.dataset_evolution[-1]['shape_after']
        
        summary['dataset_size_evolution'] = {
            'initial_shape': initial_shape,
            'final_shape': final_shape,
            'total_rows_change': final_shape[0] - initial_shape[0],
            'total_cols_change': final_shape[1] - initial_shape[1],
            'size_reduction_factor': final_shape[0] / initial_shape[0] if initial_shape[0] > 0 else 0
        }
        
        # Analisi timing
        step_durations = []
        for step_name, timing in self.step_times.items():
            if 'duration' in timing:
                step_durations.append({
                    'step': step_name,
                    'duration': timing['duration']
                })
        
        if step_durations:
            durations = [s['duration'] for s in step_durations]
            summary['timing_analysis'] = {
                'total_steps': len(step_durations),
                'total_duration': sum(durations),
                'average_step_duration': np.mean(durations),
                'longest_step': max(step_durations, key=lambda x: x['duration']),
                'shortest_step': min(step_durations, key=lambda x: x['duration'])
            }
        
        # Evoluzione features
        total_features_added = sum(
            len(step['cols_added']) for step in self.dataset_evolution
        )
        total_features_removed = sum(
            len(step['cols_removed']) for step in self.dataset_evolution
        )
        
        summary['feature_evolution'] = {
            'total_features_added': total_features_added,
            'total_features_removed': total_features_removed,
            'net_feature_change': total_features_added - total_features_removed
        }
        
        # Evoluzione memoria
        memory_usage = [step['memory_after_mb'] for step in self.dataset_evolution]
        if memory_usage:
            summary['memory_usage_evolution'] = {
                'initial_memory_mb': self.dataset_evolution[0]['memory_before_mb'],
                'final_memory_mb': memory_usage[-1],
                'peak_memory_mb': max(memory_usage),
                'memory_efficiency': memory_usage[-1] / self.dataset_evolution[0]['memory_before_mb']
            }
        
        return summary
    
    def _save_comprehensive_report(self, report: Dict[str, Any]) -> None:
        """Salva report completo su file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salva JSON completo
        json_path = self.reports_dir / f"pipeline_report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Salva summary leggibile
        summary_path = self.reports_dir / f"pipeline_summary_{timestamp}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(self._format_summary_report(report))
        
        logger.info(f"📋 Report salvati: {json_path}, {summary_path}")
    
    def _format_summary_report(self, report: Dict[str, Any]) -> str:
        """Formatta report summary in formato leggibile."""
        lines = []
        lines.append("=" * 80)
        lines.append("STIMATRIX ML PIPELINE - EXECUTION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Summary generale
        summary = report['pipeline_summary']
        lines.append("🚀 PIPELINE SUMMARY")
        lines.append(f"Start Time: {summary['start_time']}")
        lines.append(f"End Time: {summary['end_time']}")
        lines.append(f"Total Duration: {summary['total_duration_formatted']}")
        lines.append(f"Steps Completed: {summary['steps_completed']}")
        lines.append("")
        
        # Dataset evolution
        if report['dataset_evolution']:
            lines.append("📊 DATASET EVOLUTION")
            for step in report['dataset_evolution']:
                lines.append(f"Step: {step['step_name']}")
                lines.append(f"  Shape: {step['shape_before']} → {step['shape_after']}")
                lines.append(f"  Memory: {step['memory_before_mb']:.1f}MB → {step['memory_after_mb']:.1f}MB")
                if step['cols_added']:
                    lines.append(f"  Added: {step['cols_added']}")
                if step['cols_removed']:
                    lines.append(f"  Removed: {step['cols_removed']}")
                lines.append("")
        
        # Timing analysis
        if 'timing_analysis' in report.get('summary_statistics', {}):
            timing = report['summary_statistics']['timing_analysis']
            lines.append("⏱️ TIMING ANALYSIS")
            lines.append(f"Total Duration: {timing['total_duration']:.1f}s")
            lines.append(f"Average Step: {timing['average_step_duration']:.1f}s")
            lines.append(f"Longest Step: {timing['longest_step']['step']} ({timing['longest_step']['duration']:.1f}s)")
            lines.append(f"Shortest Step: {timing['shortest_step']['step']} ({timing['shortest_step']['duration']:.1f}s)")
            lines.append("")
        
        # Performance metrics
        if report['performance_metrics']:
            lines.append("🎯 PERFORMANCE METRICS")
            for category, metrics in report['performance_metrics'].items():
                lines.append(f"{category.upper()}:")
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        lines.append(f"  {key}: {value}")
                lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def _format_duration(self, seconds: float) -> str:
        """Formatta durata in formato leggibile."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def get_step_performance_summary(self) -> Dict[str, Any]:
        """Ottiene summary performance per step."""
        if not self.dataset_evolution:
            return {}
        
        performance_summary = {}
        
        for step_info in self.dataset_evolution:
            step_name = step_info['step_name']
            
            performance_summary[step_name] = {
                'duration_seconds': step_info.get('timing', {}).get('duration', 0),
                'memory_change_mb': step_info['memory_change_mb'],
                'rows_change': step_info['rows_change'],
                'cols_change': step_info['cols_change'],
                'efficiency_score': self._calculate_step_efficiency(step_info)
            }
        
        return performance_summary
    
    def _calculate_step_efficiency(self, step_info: Dict[str, Any]) -> float:
        """Calcola score efficienza per step."""
        # Score basato su tempo, memoria e utilità
        duration = step_info.get('timing', {}).get('duration', 1)
        memory_change = abs(step_info.get('memory_change_mb', 0))
        cols_useful_change = max(0, step_info.get('cols_change', 0))  # Solo aggiunte positive
        
        # Normalizza e combina (più basso è meglio per tempo/memoria, più alto per features)
        time_score = 1 / (1 + duration / 60)  # Normalizza per minuto
        memory_score = 1 / (1 + memory_change / 100)  # Normalizza per 100MB
        utility_score = cols_useful_change / 10  # Normalizza per 10 colonne
        
        efficiency = (time_score + memory_score + utility_score) / 3
        return min(1.0, efficiency)  # Cap a 1.0
    
    def export_tracking_data(self, format: str = 'json') -> str:
        """
        Esporta dati tracking in formato specificato.
        
        Args:
            format: Formato export ('json', 'csv', 'excel')
            
        Returns:
            Path file esportato
        """
        if not self.tracking_enabled:
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            export_path = self.reports_dir / f"tracking_export_{timestamp}.json"
            export_data = {
                'steps_info': self.steps_info,
                'dataset_evolution': self.dataset_evolution,
                'performance_metrics': self.performance_metrics,
                'step_times': self.step_times
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format == 'csv':
            export_path = self.reports_dir / f"tracking_export_{timestamp}.csv"
            
            # Converti dataset evolution in DataFrame
            evolution_df = pd.DataFrame(self.dataset_evolution)
            evolution_df.to_csv(export_path, index=False)
        
        elif format == 'excel':
            export_path = self.reports_dir / f"tracking_export_{timestamp}.xlsx"
            
            with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                # Sheet evoluzione dataset
                evolution_df = pd.DataFrame(self.dataset_evolution)
                evolution_df.to_excel(writer, sheet_name='Dataset_Evolution', index=False)
                
                # Sheet timing
                timing_data = []
                for step_name, timing in self.step_times.items():
                    timing_data.append({
                        'step_name': step_name,
                        'duration_seconds': timing.get('duration', 0),
                        'start_time': timing.get('start', ''),
                        'end_time': timing.get('end', '')
                    })
                
                timing_df = pd.DataFrame(timing_data)
                timing_df.to_excel(writer, sheet_name='Step_Timings', index=False)
        
        else:
            raise ValueError(f"Formato non supportato: {format}")
        
        logger.info(f"📤 Dati tracking esportati: {export_path}")
        return str(export_path)


class PipelineMonitor:
    """Monitor real-time per pipeline execution."""
    
    def __init__(self, tracker: PipelineTracker):
        """
        Inizializza monitor pipeline.
        
        Args:
            tracker: Pipeline tracker da monitorare
        """
        self.tracker = tracker
        self.monitoring_enabled = tracker.tracking_enabled
        self.alerts_config = tracker.config.get('monitoring', {}).get('alerts', {})
        
        # Soglie alert
        self.max_step_duration = self.alerts_config.get('max_step_duration_minutes', 30) * 60
        self.max_memory_usage = self.alerts_config.get('max_memory_usage_mb', 2000)
        self.min_samples_threshold = self.alerts_config.get('min_samples_threshold', 1000)
        
    def check_pipeline_health(self) -> Dict[str, Any]:
        """
        Controlla salute pipeline e genera alert se necessario.
        
        Returns:
            Dict con stato salute pipeline
        """
        if not self.monitoring_enabled:
            return {'monitoring_disabled': True}
        
        health_status = {
            'overall_status': 'HEALTHY',
            'alerts': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Controllo timing
        for step_name, timing in self.tracker.step_times.items():
            duration = timing.get('duration', 0)
            if duration > self.max_step_duration:
                health_status['alerts'].append(
                    f"Step '{step_name}' durata eccessiva: {duration:.1f}s "
                    f"(max: {self.max_step_duration}s)"
                )
                health_status['overall_status'] = 'ALERT'
        
        # Controllo memoria
        for step_info in self.tracker.dataset_evolution:
            memory_after = step_info.get('memory_after_mb', 0)
            if memory_after > self.max_memory_usage:
                health_status['alerts'].append(
                    f"Step '{step_info['step_name']}' memoria eccessiva: {memory_after:.1f}MB "
                    f"(max: {self.max_memory_usage}MB)"
                )
                health_status['overall_status'] = 'ALERT'
        
        # Controllo campioni
        if self.tracker.dataset_evolution:
            final_samples = self.tracker.dataset_evolution[-1]['shape_after'][0]
            if final_samples < self.min_samples_threshold:
                health_status['warnings'].append(
                    f"Pochi campioni finali: {final_samples} "
                    f"(min raccomandato: {self.min_samples_threshold})"
                )
                if health_status['overall_status'] == 'HEALTHY':
                    health_status['overall_status'] = 'WARNING'
        
        # Raccomandazioni
        if health_status['overall_status'] != 'HEALTHY':
            health_status['recommendations'] = self._generate_recommendations(health_status)
        
        return health_status
    
    def _generate_recommendations(self, health_status: Dict[str, Any]) -> List[str]:
        """Genera raccomandazioni basate su alert/warning."""
        recommendations = []
        
        # Raccomandazioni per performance
        if any('durata eccessiva' in alert for alert in health_status['alerts']):
            recommendations.append("Considera di ridurre n_trials per Optuna o disabilitare modelli lenti")
            recommendations.append("Abilita timeout per training: training.timeout = 1800")
        
        # Raccomandazioni per memoria
        if any('memoria eccessiva' in alert for alert in health_status['alerts']):
            recommendations.append("Riduci sample_size per SHAP analysis")
            recommendations.append("Disabilita PCA se non necessario: profiles.*.pca.enabled = false")
            recommendations.append("Considera preprocessing incrementale per dataset grandi")
        
        # Raccomandazioni per campioni
        if any('Pochi campioni' in warning for warning in health_status['warnings']):
            recommendations.append("Rivedi parametri outlier detection per preservare più campioni")
            recommendations.append("Considera di ridurre soglie filtering correlazioni")
        
        return recommendations