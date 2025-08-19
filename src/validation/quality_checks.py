"""
Sistema completo di Quality Checks per validazione pipeline ML immobiliari.

Questo modulo implementa controlli automatici di qualità che verificano:
- Data leakage temporale: sovrapposizioni tra train/validation/test splits
- Target leakage: features che contengono informazioni del target
- Drift distribuzione: cambiamenti distribuzione categorie tra splits
- Stabilità features: consistenza durante preprocessing
- Integrità dati: validazione strutturale DataFrame

I quality checks sono progettati per essere eseguiti automaticamente durante
la pipeline e forniscono diagnostica dettagliata per identificare problemi
comuni che possono compromettere la validità dei modelli ML.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from utils.logger import get_logger
from utils.detailed_logging import DetailedLogger
from utils.temporal_advanced import AdvancedTemporalUtils

logger = get_logger(__name__)


class QualityChecker:
    """Sistema completo di quality checks per validazione pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza quality checker con configurazione.
        
        Args:
            config: Configurazione quality checks
        """
        self.config = config.get('quality_checks', {})
        self.enabled_checks = {
            'temporal_leakage': self.config.get('check_temporal_leakage', True),
            'target_leakage': self.config.get('check_target_leakage', True),
            'category_distribution': self.config.get('check_category_distribution', True),
            'feature_stability': self.config.get('check_feature_stability', True),
            'data_drift': self.config.get('check_data_drift', False),
            'outlier_consistency': self.config.get('check_outlier_consistency', False)
        }
        
        # Parametri configurabili
        self.max_category_drift = self.config.get('max_category_drift', 0.05)
        self.max_feature_drift = self.config.get('max_feature_drift', 0.1)
        self.min_temporal_gap_months = self.config.get('min_temporal_gap_months', 1)
        
        logger.info(f"Quality Checker inizializzato - Checks abilitati: {self.enabled_checks}")
    
    def run_all_checks(
        self,
        X_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame],
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: Optional[pd.Series],
        y_test: pd.Series,
        preprocessing_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Esegue tutti i quality checks abilitati.
        
        Args:
            X_train: Features training
            X_val: Features validation (opzionale)
            X_test: Features test
            y_train: Target training
            y_val: Target validation (opzionale)
            y_test: Target test
            preprocessing_info: Info preprocessing per checks avanzati
            
        Returns:
            Dict con risultati di tutti i checks
        """
        logger.info("=== ESECUZIONE QUALITY CHECKS ===")
        
        all_results = {
            'checks_executed': [],
            'checks_passed': [],
            'checks_failed': [],
            'warnings': [],
            'critical_errors': [],
            'overall_status': 'UNKNOWN'
        }
        
        # 1. Temporal Leakage Check
        if self.enabled_checks['temporal_leakage']:
            temporal_result = self.check_temporal_leakage(X_train, X_val, X_test)
            all_results['temporal_leakage'] = temporal_result
            all_results['checks_executed'].append('temporal_leakage')
            
            if temporal_result['is_valid']:
                all_results['checks_passed'].append('temporal_leakage')
            else:
                all_results['checks_failed'].append('temporal_leakage')
                all_results['critical_errors'].extend(temporal_result.get('errors', []))
        
        # 2. Target Leakage Check
        if self.enabled_checks['target_leakage']:
            target_result = self.check_target_leakage(X_train, y_train.name if hasattr(y_train, 'name') else 'target')
            all_results['target_leakage'] = target_result
            all_results['checks_executed'].append('target_leakage')
            
            if target_result['is_valid']:
                all_results['checks_passed'].append('target_leakage')
            else:
                all_results['checks_failed'].append('target_leakage')
                all_results['critical_errors'].extend(target_result.get('errors', []))
        
        # 3. Category Distribution Check
        if self.enabled_checks['category_distribution']:
            category_result = self.check_category_distribution(X_train, X_val, X_test)
            all_results['category_distribution'] = category_result
            all_results['checks_executed'].append('category_distribution')
            
            if category_result['is_valid']:
                all_results['checks_passed'].append('category_distribution')
            else:
                all_results['checks_failed'].append('category_distribution')
                all_results['warnings'].extend(category_result.get('warnings', []))
        
        # 4. Feature Stability Check
        if self.enabled_checks['feature_stability'] and preprocessing_info:
            stability_result = self.check_feature_stability(preprocessing_info)
            all_results['feature_stability'] = stability_result
            all_results['checks_executed'].append('feature_stability')
            
            if stability_result['is_valid']:
                all_results['checks_passed'].append('feature_stability')
            else:
                all_results['checks_failed'].append('feature_stability')
                all_results['warnings'].extend(stability_result.get('warnings', []))
        
        # Determina status generale
        if all_results['critical_errors']:
            all_results['overall_status'] = 'CRITICAL_ERRORS'
        elif all_results['checks_failed']:
            all_results['overall_status'] = 'FAILED'
        elif all_results['warnings']:
            all_results['overall_status'] = 'WARNINGS'
        else:
            all_results['overall_status'] = 'PASSED'
        
        # Log risultati finali
        self._log_overall_results(all_results)
        
        return all_results
    
    def check_temporal_leakage(
        self,
        X_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame],
        X_test: pd.DataFrame,
        year_col: str = 'A_AnnoStipula',
        month_col: str = 'A_MeseStipula'
    ) -> Dict[str, Any]:
        """
        Verifica sovrapposizioni temporali tra split.
        
        Args:
            X_train: Features training
            X_val: Features validation
            X_test: Features test
            year_col: Colonna anno
            month_col: Colonna mese
            
        Returns:
            Dict con risultati check
        """
        logger.info("🕐 Check temporal leakage...")
        
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'temporal_ranges': {}
        }
        
        try:
            # Verifica esistenza colonne temporali
            datasets = {'train': X_train, 'test': X_test}
            if X_val is not None:
                datasets['val'] = X_val
            
            # Controlla se le colonne temporali esistono
            temporal_cols_exist = all(
                year_col in dataset.columns and month_col in dataset.columns
                for dataset in datasets.values()
                if not dataset.empty
            )
            
            if not temporal_cols_exist:
                result['warnings'].append(f"Colonne temporali {year_col}/{month_col} non trovate in tutti i dataset")
                logger.warning("Colonne temporali mancanti - skip temporal leakage check")
                return result
            
            # Usa utility avanzate per validazione
            validation_results = AdvancedTemporalUtils.validate_temporal_split_integrity(
                X_train, X_val, X_test, year_col, month_col
            )
            
            result.update(validation_results)
            
            # Controlli aggiuntivi specifici
            if result['is_valid']:
                # Verifica gap temporale minimo
                temporal_ranges = result.get('temporal_ranges', {})
                if 'val' in temporal_ranges and 'train' in temporal_ranges:
                    train_max_key = temporal_ranges['train']['max_key']
                    val_min_key = temporal_ranges['val']['min_key']
                    gap_months = val_min_key - train_max_key
                    
                    if gap_months < self.min_temporal_gap_months:
                        result['warnings'].append(
                            f"Gap temporale train-val troppo piccolo: {gap_months} mesi "
                            f"(minimo: {self.min_temporal_gap_months})"
                        )
            
            if result['is_valid'] and not result['warnings']:
                logger.info("✅ Temporal leakage check: PASSED")
            else:
                logger.warning(f"⚠️ Temporal leakage check: WARNINGS - {result['warnings']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Errore temporal leakage check: {e}")
            result['is_valid'] = False
            result['errors'].append(str(e))
            return result
    
    def check_target_leakage(
        self,
        X: pd.DataFrame,
        target_col: str
    ) -> Dict[str, Any]:
        """
        Verifica se features contengono informazioni del target (data leakage).
        
        Args:
            X: Features dataset
            target_col: Nome colonna target
            
        Returns:
            Dict con risultati check
        """
        logger.info("🎯 Check target leakage...")
        
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suspicious_features': []
        }
        
        try:
            # Pattern sospetti per target leakage
            target_patterns = [
                target_col.lower(),
                'prezzo', 'price', 'valore', 'value',
                'target', 'label', 'y_',
                'ridistribuito', 'redistributed'
            ]
            
            # Cerca colonne sospette
            suspicious_features = []
            for col in X.columns:
                col_lower = col.lower()
                for pattern in target_patterns:
                    if pattern in col_lower and col != target_col:
                        suspicious_features.append({
                            'feature': col,
                            'pattern_matched': pattern,
                            'similarity_score': self._calculate_column_similarity(col_lower, pattern)
                        })
            
            # Cerca correlazioni sospette con ID o codici
            id_patterns = ['id', 'cod', 'key', 'index']
            for col in X.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in id_patterns):
                    # Se è un ID numerico, potrebbe essere problematico
                    if X[col].dtype in ['int64', 'float64'] and X[col].nunique() == len(X):
                        suspicious_features.append({
                            'feature': col,
                            'pattern_matched': 'unique_id',
                            'reason': 'ID univoco potenzialmente problematico'
                        })
            
            result['suspicious_features'] = suspicious_features
            
            # Determina risultato
            if suspicious_features:
                high_risk_features = [
                    f for f in suspicious_features 
                    if f.get('similarity_score', 0) > 0.8 or 'prezzo' in f.get('pattern_matched', '').lower()
                ]
                
                if high_risk_features:
                    result['errors'].append(
                        f"Features ad alto rischio target leakage: {[f['feature'] for f in high_risk_features]}"
                    )
                    result['is_valid'] = False
                else:
                    result['warnings'].append(
                        f"Features potenzialmente sospette: {[f['feature'] for f in suspicious_features]}"
                    )
            
            if result['is_valid'] and not result['warnings']:
                logger.info("✅ Target leakage check: PASSED")
            elif result['warnings']:
                logger.warning(f"⚠️ Target leakage check: WARNINGS - {len(suspicious_features)} features sospette")
            else:
                logger.error(f"❌ Target leakage check: FAILED - {len(high_risk_features)} features ad alto rischio")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Errore target leakage check: {e}")
            result['is_valid'] = False
            result['errors'].append(str(e))
            return result
    
    def check_category_distribution(
        self,
        X_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame],
        X_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Verifica drift distribuzione categorie tra split.
        
        Args:
            X_train: Features training
            X_val: Features validation
            X_test: Features test
            
        Returns:
            Dict con risultati check
        """
        logger.info("📊 Check category distribution...")
        
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'drift_analysis': {}
        }
        
        try:
            # Identifica colonne categoriche
            categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not categorical_cols:
                logger.info("Nessuna colonna categorica trovata - skip distribution check")
                return result
            
            datasets = {'train': X_train, 'test': X_test}
            if X_val is not None:
                datasets['val'] = X_val
            
            drift_analysis = {}
            max_drift_detected = 0
            
            for col in categorical_cols:
                col_analysis = {'distributions': {}, 'drift_scores': {}, 'max_drift': 0}
                
                # Calcola distribuzioni per ogni split
                distributions = {}
                for split_name, dataset in datasets.items():
                    if col in dataset.columns:
                        dist = dataset[col].value_counts(normalize=True).fillna(0)
                        distributions[split_name] = dist
                
                col_analysis['distributions'] = distributions
                
                # Calcola drift tra splits
                if len(distributions) >= 2:
                    splits = list(distributions.keys())
                    for i in range(len(splits)):
                        for j in range(i+1, len(splits)):
                            split1, split2 = splits[i], splits[j]
                            drift_score = self._calculate_distribution_drift(
                                distributions[split1], distributions[split2]
                            )
                            col_analysis['drift_scores'][f'{split1}_vs_{split2}'] = drift_score
                            col_analysis['max_drift'] = max(col_analysis['max_drift'], drift_score)
                
                drift_analysis[col] = col_analysis
                max_drift_detected = max(max_drift_detected, col_analysis['max_drift'])
            
            result['drift_analysis'] = drift_analysis
            
            # Valuta risultati
            high_drift_cols = [
                col for col, analysis in drift_analysis.items()
                if analysis['max_drift'] > self.max_category_drift
            ]
            
            if high_drift_cols:
                result['warnings'].append(
                    f"Drift categorie elevato (>{self.max_category_drift:.3f}): {high_drift_cols}"
                )
                result['warnings'].append(
                    f"Max drift rilevato: {max_drift_detected:.3f}"
                )
            
            if result['warnings']:
                logger.warning(f"⚠️ Category distribution check: WARNINGS - {len(high_drift_cols)} colonne con drift elevato")
            else:
                logger.info("✅ Category distribution check: PASSED")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Errore category distribution check: {e}")
            result['is_valid'] = False
            result['errors'].append(str(e))
            return result
    
    def check_feature_stability(
        self,
        preprocessing_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verifica stabilità features durante preprocessing.
        
        Args:
            preprocessing_info: Info dai passi di preprocessing
            
        Returns:
            Dict con risultati check
        """
        logger.info("🔧 Check feature stability...")
        
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stability_analysis': {}
        }
        
        try:
            # Analizza evoluzione dataset
            if 'dataset_evolution' in preprocessing_info:
                evolution = preprocessing_info['dataset_evolution']
                
                # Controlla rimozione massiva di colonne
                for step_info in evolution:
                    if 'cols_change' in step_info:
                        cols_removed = abs(min(0, step_info['cols_change']))
                        total_cols_before = step_info.get('shape_before', [0, 0])[1]
                        
                        if total_cols_before > 0:
                            removal_rate = cols_removed / total_cols_before
                            
                            if removal_rate > self.max_feature_drift:
                                result['warnings'].append(
                                    f"Step '{step_info['step_name']}': rimozione massiva colonne "
                                    f"({removal_rate:.1%} delle colonne)"
                                )
            
            # Controlla consistenza tipi dati
            if 'steps_info' in preprocessing_info:
                for step_name, step_info in preprocessing_info['steps_info'].items():
                    if 'type_changes' in step_info:
                        unexpected_changes = step_info['type_changes']
                        if unexpected_changes:
                            result['warnings'].append(
                                f"Step '{step_name}': cambiamenti tipo dati inattesi: {unexpected_changes}"
                            )
            
            if result['warnings']:
                logger.warning(f"⚠️ Feature stability check: WARNINGS - {len(result['warnings'])} problemi rilevati")
            else:
                logger.info("✅ Feature stability check: PASSED")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Errore feature stability check: {e}")
            result['is_valid'] = False
            result['errors'].append(str(e))
            return result
    
    def _calculate_column_similarity(self, col_name: str, pattern: str) -> float:
        """Calcola similarità tra nome colonna e pattern."""
        if pattern in col_name:
            return len(pattern) / len(col_name)
        return 0.0
    
    def _calculate_distribution_drift(self, dist1: pd.Series, dist2: pd.Series) -> float:
        """Calcola drift tra due distribuzioni usando Total Variation Distance."""
        # Allinea indici
        all_categories = set(dist1.index) | set(dist2.index)
        aligned_dist1 = dist1.reindex(all_categories, fill_value=0)
        aligned_dist2 = dist2.reindex(all_categories, fill_value=0)
        
        # Total Variation Distance
        tvd = 0.5 * np.sum(np.abs(aligned_dist1 - aligned_dist2))
        return tvd
    
    def _log_overall_results(self, results: Dict[str, Any]) -> None:
        """Log risultati complessivi quality checks."""
        logger.info("=== RISULTATI QUALITY CHECKS ===")
        
        total_checks = len(results['checks_executed'])
        passed_checks = len(results['checks_passed'])
        failed_checks = len(results['checks_failed'])
        
        logger.info(f"  📊 Checks eseguiti: {total_checks}")
        logger.info(f"  ✅ Checks superati: {passed_checks}")
        logger.info(f"  ❌ Checks falliti: {failed_checks}")
        logger.info(f"  ⚠️  Warnings: {len(results['warnings'])}")
        logger.info(f"  🚨 Errori critici: {len(results['critical_errors'])}")
        
        status_emoji = {
            'PASSED': '✅',
            'WARNINGS': '⚠️',
            'FAILED': '❌',
            'CRITICAL_ERRORS': '🚨'
        }
        
        status = results['overall_status']
        emoji = status_emoji.get(status, '❓')
        logger.info(f"  {emoji} STATUS GENERALE: {status}")
        
        if results['critical_errors']:
            logger.error("ERRORI CRITICI RILEVATI:")
            for error in results['critical_errors']:
                logger.error(f"  • {error}")
        
        if results['warnings']:
            logger.warning("WARNINGS RILEVATI:")
            for warning in results['warnings']:
                logger.warning(f"  • {warning}")


class DataQualityMetrics:
    """Metriche per valutazione qualità dati."""
    
    @staticmethod
    def calculate_data_quality_score(
        df: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calcola score complessivo qualità dati.
        
        Args:
            df: DataFrame da valutare
            target_col: Colonna target (opzionale)
            
        Returns:
            Dict con metriche qualità
        """
        metrics = {
            'completeness_score': 0.0,
            'consistency_score': 0.0,
            'uniqueness_score': 0.0,
            'validity_score': 0.0,
            'overall_score': 0.0,
            'details': {}
        }
        
        # Completeness (% valori non nulli)
        total_cells = df.size
        non_null_cells = df.count().sum()
        metrics['completeness_score'] = non_null_cells / total_cells if total_cells > 0 else 0
        
        # Consistency (% colonne con tipo dati consistente)
        consistent_cols = 0
        for col in df.columns:
            if df[col].dtype != 'object':
                consistent_cols += 1
            else:
                # Per object, controlla se tutti i valori hanno lo stesso tipo
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    types = set(type(val) for val in non_null_values)
                    if len(types) == 1:
                        consistent_cols += 1
        
        metrics['consistency_score'] = consistent_cols / len(df.columns) if len(df.columns) > 0 else 0
        
        # Uniqueness (media % valori unici per colonna)
        uniqueness_scores = []
        for col in df.columns:
            if col != target_col:  # Escludi target
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                uniqueness_scores.append(min(unique_ratio, 1.0))  # Cap a 1.0
        
        metrics['uniqueness_score'] = np.mean(uniqueness_scores) if uniqueness_scores else 0
        
        # Validity (% colonne senza valori anomali evidenti)
        valid_cols = 0
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Controlla outlier estremi (oltre 5 std)
                if len(df[col].dropna()) > 0:
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    extreme_outliers = (z_scores > 5).sum()
                    if extreme_outliers < len(df) * 0.01:  # <1% outlier estremi
                        valid_cols += 1
                else:
                    valid_cols += 1
            else:
                valid_cols += 1  # Assume valide le non-numeriche
        
        metrics['validity_score'] = valid_cols / len(df.columns) if len(df.columns) > 0 else 0
        
        # Overall score (media pesata)
        weights = {'completeness': 0.3, 'consistency': 0.2, 'uniqueness': 0.2, 'validity': 0.3}
        metrics['overall_score'] = (
            metrics['completeness_score'] * weights['completeness'] +
            metrics['consistency_score'] * weights['consistency'] +
            metrics['uniqueness_score'] * weights['uniqueness'] +
            metrics['validity_score'] * weights['validity']
        )
        
        # Dettagli aggiuntivi
        metrics['details'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_cells': total_cells - non_null_cells,
            'missing_percentage': ((total_cells - non_null_cells) / total_cells * 100) if total_cells > 0 else 0,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        return metrics