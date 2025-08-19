"""
Sistema di logging dettagliato per pipeline ML con statistiche operative complete.

Questo modulo fornisce funzionalità di logging avanzato che traccia:
- Evoluzione del dataset attraverso gli step di preprocessing
- Statistiche dettagliate per rimozione colonne (frequenze, correlazioni)
- Range temporali per split dataset con validazione integrità
- Risultati feature extraction con successi/fallimenti
- Progress training modelli con timing e performance
- Summary outlier detection con percentuali per metodo/categoria

Il sistema è progettato per debugging, monitoring e audit di pipeline ML complesse.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class DetailedLogger:
    """Sistema di logging avanzato per operazioni di preprocessing e training."""
    
    @staticmethod
    def log_column_removal(removed_cols: List[str], column_stats: Dict[str, Dict], operation_name: str) -> None:
        """
        Log dettagliato per rimozione colonne con statistiche complete.
        
        Args:
            removed_cols: Lista colonne rimosse
            column_stats: Statistiche per ogni colonna
            operation_name: Nome dell'operazione
        """
        if not removed_cols:
            logger.info(f"=== {operation_name} === - Nessuna colonna rimossa")
            return
            
        logger.info(f"=== {operation_name} === - Rimosse {len(removed_cols)} colonne:")
        for col in removed_cols:
            stats = column_stats.get(col, {})
            if 'max_frequency' in stats:
                logger.info(f"  ❌ {col}: {stats['max_frequency']:.3f} frequenza, "
                           f"{stats['unique_values']} valori unici, "
                           f"valore dominante: {stats['most_common_value']}")
            elif 'correlation' in stats:
                logger.info(f"  ❌ {col}: correlazione={stats['correlation']:.3f} "
                           f"con {stats['correlated_with']}")
            else:
                logger.info(f"  ❌ {col}: {stats}")
    
    @staticmethod
    def log_column_operation_results(
        requested_cols: List[str], 
        existing_cols: List[str], 
        missing_cols: List[str], 
        operation_name: str
    ) -> None:
        """
        Log risultati operazioni su colonne con gestione missing.
        
        Args:
            requested_cols: Colonne richieste
            existing_cols: Colonne esistenti
            missing_cols: Colonne mancanti
            operation_name: Nome operazione
        """
        logger.info(f"=== {operation_name} ===")
        logger.info(f"  📝 Richieste: {len(requested_cols)} colonne")
        logger.info(f"  ✅ Esistenti: {len(existing_cols)} colonne")
        
        if missing_cols:
            logger.warning(f"  ⚠️  Mancanti: {len(missing_cols)} colonne - {missing_cols}")
        
        if existing_cols:
            logger.info(f"  🔧 Processate: {existing_cols}")
    
    @staticmethod
    def log_split_temporal_ranges(
        df_sorted: pd.DataFrame, 
        year_col: str, 
        month_col: str, 
        val_idx: int, 
        test_idx: int
    ) -> None:
        """
        Log dettagliato dei range temporali per split.
        
        Args:
            df_sorted: DataFrame ordinato temporalmente
            year_col: Colonna anno
            month_col: Colonna mese
            val_idx: Indice inizio validation
            test_idx: Indice inizio test
        """
        logger.info("=== SPLIT TEMPORALE - RANGE DATE ===")
        
        # Train range
        train_start = f"{df_sorted[year_col].iloc[0]}/{df_sorted[month_col].iloc[0]:02d}"
        train_end = f"{df_sorted[year_col].iloc[val_idx-1]}/{df_sorted[month_col].iloc[val_idx-1]:02d}"
        logger.info(f"  📈 Train: {train_start} → {train_end} ({val_idx} samples)")
        
        # Validation range
        if val_idx < test_idx:
            val_start = f"{df_sorted[year_col].iloc[val_idx]}/{df_sorted[month_col].iloc[val_idx]:02d}"
            val_end = f"{df_sorted[year_col].iloc[test_idx-1]}/{df_sorted[month_col].iloc[test_idx-1]:02d}"
            logger.info(f"  📊 Val: {val_start} → {val_end} ({test_idx - val_idx} samples)")
        
        # Test range
        test_start = f"{df_sorted[year_col].iloc[test_idx]}/{df_sorted[month_col].iloc[test_idx]:02d}"
        test_end = f"{df_sorted[year_col].iloc[-1]}/{df_sorted[month_col].iloc[-1]:02d}"
        logger.info(f"  📋 Test: {test_start} → {test_end} ({len(df_sorted) - test_idx} samples)")
    
    @staticmethod
    def log_dataset_evolution(
        df_before: pd.DataFrame, 
        df_after: pd.DataFrame, 
        step_name: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log evoluzione dataset con statistiche complete.
        
        Args:
            df_before: DataFrame prima dell'operazione
            df_after: DataFrame dopo l'operazione
            step_name: Nome dello step
            additional_info: Informazioni aggiuntive
            
        Returns:
            Dict con statistiche evoluzione
        """
        # Calcola cambiamenti
        rows_before, cols_before = df_before.shape
        rows_after, cols_after = df_after.shape
        
        cols_added = set(df_after.columns) - set(df_before.columns)
        cols_removed = set(df_before.columns) - set(df_after.columns)
        
        memory_before = df_before.memory_usage(deep=True).sum() / 1024**2
        memory_after = df_after.memory_usage(deep=True).sum() / 1024**2
        
        # Log principale
        logger.info(f"=== {step_name.upper()} - EVOLUZIONE DATASET ===")
        logger.info(f"  📊 Shape: {rows_before:,} × {cols_before} → {rows_after:,} × {cols_after}")
        logger.info(f"  🔢 Rows: {rows_after - rows_before:+,} | Cols: {cols_after - cols_before:+}")
        logger.info(f"  💾 Memory: {memory_before:.1f}MB → {memory_after:.1f}MB ({memory_after - memory_before:+.1f}MB)")
        
        # Log colonne aggiunte/rimosse
        if cols_added:
            logger.info(f"  ➕ Colonne aggiunte ({len(cols_added)}): {list(cols_added)}")
        if cols_removed:
            logger.info(f"  ➖ Colonne rimosse ({len(cols_removed)}): {list(cols_removed)}")
        
        # Log info aggiuntive
        if additional_info:
            for key, value in additional_info.items():
                logger.info(f"  ℹ️  {key}: {value}")
        
        return {
            'step_name': step_name,
            'shape_before': (rows_before, cols_before),
            'shape_after': (rows_after, cols_after),
            'rows_change': rows_after - rows_before,
            'cols_change': cols_after - cols_before,
            'cols_added': list(cols_added),
            'cols_removed': list(cols_removed),
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_change_mb': memory_after - memory_before,
            'additional_info': additional_info or {}
        }
    
    @staticmethod
    def log_feature_extraction_results(
        extraction_results: Dict[str, Any], 
        operation_name: str
    ) -> None:
        """
        Log risultati feature extraction con dettagli.
        
        Args:
            extraction_results: Risultati estrazione
            operation_name: Nome operazione
        """
        logger.info(f"=== {operation_name.upper()} ===")
        
        for feature_type, results in extraction_results.items():
            if isinstance(results, dict):
                logger.info(f"  🔧 {feature_type}:")
                for key, value in results.items():
                    if isinstance(value, (list, tuple)):
                        logger.info(f"    • {key}: {len(value)} elementi")
                    else:
                        logger.info(f"    • {key}: {value}")
            else:
                logger.info(f"  🔧 {feature_type}: {results}")
    
    @staticmethod
    def log_outlier_detection_summary(
        outliers_info: Dict[str, Any], 
        total_samples: int
    ) -> None:
        """
        Log summary outlier detection.
        
        Args:
            outliers_info: Informazioni outliers
            total_samples: Numero totale campioni
        """
        logger.info("=== OUTLIER DETECTION SUMMARY ===")
        
        total_outliers = outliers_info.get('total_outliers', 0)
        outlier_percentage = (total_outliers / total_samples) * 100 if total_samples > 0 else 0
        
        logger.info(f"  🎯 Outliers: {total_outliers:,} / {total_samples:,} ({outlier_percentage:.2f}%)")
        
        if 'by_method' in outliers_info:
            logger.info("  📊 Per metodo:")
            for method, count in outliers_info['by_method'].items():
                method_percentage = (count / total_samples) * 100 if total_samples > 0 else 0
                logger.info(f"    • {method}: {count:,} ({method_percentage:.2f}%)")
        
        if 'by_category' in outliers_info:
            logger.info("  🏷️  Per categoria:")
            for category, count in outliers_info['by_category'].items():
                logger.info(f"    • {category}: {count:,}")
    
    @staticmethod
    def log_model_training_progress(
        model_name: str, 
        trial_num: int, 
        total_trials: int, 
        current_score: float, 
        best_score: float,
        elapsed_time: float
    ) -> None:
        """
        Log progresso training modello.
        
        Args:
            model_name: Nome modello
            trial_num: Numero trial corrente
            total_trials: Totale trials
            current_score: Score corrente
            best_score: Miglior score
            elapsed_time: Tempo trascorso
        """
        progress = (trial_num / total_trials) * 100
        logger.info(f"🤖 {model_name} | Trial {trial_num}/{total_trials} ({progress:.1f}%) | "
                   f"Score: {current_score:.6f} | Best: {best_score:.6f} | "
                   f"Time: {elapsed_time:.1f}s")


class StatisticsCalculator:
    """Calcolatore di statistiche per logging dettagliato."""
    
    @staticmethod
    def calculate_column_statistics(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Calcola statistiche dettagliate per colonne.
        
        Args:
            df: DataFrame
            columns: Lista colonne da analizzare
            
        Returns:
            Dict con statistiche per colonna
        """
        stats = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            col_stats = {}
            
            # Statistiche base
            col_stats['dtype'] = str(df[col].dtype)
            col_stats['null_count'] = df[col].isnull().sum()
            col_stats['null_percentage'] = (col_stats['null_count'] / len(df)) * 100
            
            # Statistiche specifiche per tipo
            if df[col].dtype in ['object', 'category']:
                value_counts = df[col].value_counts(normalize=True, dropna=False)
                col_stats['unique_values'] = len(value_counts)
                col_stats['max_frequency'] = value_counts.iloc[0] if len(value_counts) > 0 else 0
                col_stats['most_common_value'] = value_counts.index[0] if len(value_counts) > 0 else None
            else:
                col_stats['mean'] = df[col].mean()
                col_stats['std'] = df[col].std()
                col_stats['min'] = df[col].min()
                col_stats['max'] = df[col].max()
                col_stats['unique_values'] = df[col].nunique()
            
            stats[col] = col_stats
        
        return stats
    
    @staticmethod
    def calculate_correlation_statistics(
        df: pd.DataFrame, 
        numeric_cols: List[str], 
        threshold: float = 0.95
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calcola statistiche correlazioni.
        
        Args:
            df: DataFrame
            numeric_cols: Colonne numeriche
            threshold: Soglia correlazione
            
        Returns:
            Dict con statistiche correlazioni
        """
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Trova correlazioni alte
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                
                if corr_val >= threshold:
                    high_corr_pairs.append({
                        'col1': col1,
                        'col2': col2,
                        'correlation': corr_val
                    })
        
        return {
            'total_pairs': len(numeric_cols) * (len(numeric_cols) - 1) // 2,
            'high_correlation_pairs': len(high_corr_pairs),
            'threshold_used': threshold,
            'pairs_details': high_corr_pairs
        }