"""
Utilities temporali avanzate per gestione dati time-series.
Ispirato al sistema temporale robusto di RealEstatePricePrediction.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.logger import get_logger
from utils.detailed_logging import DetailedLogger

logger = get_logger(__name__)


class AdvancedTemporalUtils:
    """Utilities avanzate per gestione dati temporali."""
    
    @staticmethod
    def create_temporal_sort_key(
        df: pd.DataFrame, 
        year_col: str, 
        month_col: str, 
        day_col: Optional[str] = None,
        hour_col: Optional[str] = None
    ) -> pd.Series:
        """
        Crea chiave di sorting temporale composita.
        
        Args:
            df: DataFrame
            year_col: Colonna anno
            month_col: Colonna mese
            day_col: Colonna giorno (opzionale)
            hour_col: Colonna ora (opzionale)
            
        Returns:
            Serie con chiave di sorting
        """
        # Verifica esistenza colonne
        required_cols = [year_col, month_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Colonne temporali mancanti: {missing_cols}")
        
        # Crea chiave base (anno * 12 + mese)
        sort_key = df[year_col] * 12 + df[month_col]
        
        # Aggiungi giorno se disponibile
        if day_col and day_col in df.columns:
            sort_key = sort_key * 31 + df[day_col].fillna(1)
        
        # Aggiungi ora se disponibile
        if hour_col and hour_col in df.columns:
            sort_key = sort_key * 24 + df[hour_col].fillna(0)
        
        return sort_key
    
    @staticmethod
    def temporal_sort_dataframe(
        df: pd.DataFrame, 
        year_col: str, 
        month_col: str, 
        day_col: Optional[str] = None,
        hour_col: Optional[str] = None,
        ascending: bool = True
    ) -> pd.DataFrame:
        """
        Ordina DataFrame temporalmente con chiave composita.
        
        Args:
            df: DataFrame da ordinare
            year_col: Colonna anno
            month_col: Colonna mese
            day_col: Colonna giorno (opzionale)
            hour_col: Colonna ora (opzionale)
            ascending: Ordine crescente/decrescente
            
        Returns:
            DataFrame ordinato
        """
        logger.info(f"Ordinamento temporale: {year_col}/{month_col}" + 
                   (f"/{day_col}" if day_col else "") + 
                   (f"/{hour_col}" if hour_col else ""))
        
        # Crea chiave di sorting
        sort_key = AdvancedTemporalUtils.create_temporal_sort_key(
            df, year_col, month_col, day_col, hour_col
        )
        
        # Ordina DataFrame
        df_sorted = df.iloc[sort_key.argsort()].copy()
        
        if not ascending:
            df_sorted = df_sorted.iloc[::-1].copy()
        
        # Reset index
        df_sorted = df_sorted.reset_index(drop=True)
        
        logger.info(f"DataFrame ordinato: {len(df_sorted)} righe")
        return df_sorted
    
    @staticmethod
    def validate_temporal_split_integrity(
        X_train: pd.DataFrame, 
        X_val: Optional[pd.DataFrame], 
        X_test: pd.DataFrame, 
        year_col: str, 
        month_col: str,
        day_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Valida integrità split temporale (no overlap, ordine corretto).
        
        Args:
            X_train: Dataset training
            X_val: Dataset validation (opzionale)
            X_test: Dataset test
            year_col: Colonna anno
            month_col: Colonna mese
            day_col: Colonna giorno (opzionale)
            
        Returns:
            Dict con risultati validazione
        """
        logger.info("Validazione integrità split temporale...")
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'temporal_ranges': {}
        }
        
        try:
            # Calcola range temporali per ogni split
            datasets = {'train': X_train, 'test': X_test}
            if X_val is not None:
                datasets['val'] = X_val
            
            temporal_ranges = {}
            for split_name, dataset in datasets.items():
                if dataset.empty:
                    continue
                    
                sort_key = AdvancedTemporalUtils.create_temporal_sort_key(
                    dataset, year_col, month_col, day_col
                )
                
                temporal_ranges[split_name] = {
                    'min_key': sort_key.min(),
                    'max_key': sort_key.max(),
                    'min_date': f"{dataset[year_col].min()}/{dataset[month_col].min():02d}",
                    'max_date': f"{dataset[year_col].max()}/{dataset[month_col].max():02d}",
                    'samples': len(dataset)
                }
            
            validation_results['temporal_ranges'] = temporal_ranges
            
            # Controlli di integrità
            if 'val' in temporal_ranges:
                # Train < Val < Test
                if temporal_ranges['train']['max_key'] >= temporal_ranges['val']['min_key']:
                    validation_results['errors'].append(
                        "Overlap temporale train-validation"
                    )
                    validation_results['is_valid'] = False
                
                if temporal_ranges['val']['max_key'] >= temporal_ranges['test']['min_key']:
                    validation_results['errors'].append(
                        "Overlap temporale validation-test"
                    )
                    validation_results['is_valid'] = False
            else:
                # Train < Test
                if temporal_ranges['train']['max_key'] >= temporal_ranges['test']['min_key']:
                    validation_results['errors'].append(
                        "Overlap temporale train-test"
                    )
                    validation_results['is_valid'] = False
            
            # Controlli di warning
            for split_name, range_info in temporal_ranges.items():
                if range_info['samples'] < 100:
                    validation_results['warnings'].append(
                        f"Split {split_name} ha pochi campioni: {range_info['samples']}"
                    )
            
            # Log risultati
            if validation_results['errors']:
                logger.error(f"Errori integrità temporale: {validation_results['errors']}")
            elif validation_results['warnings']:
                logger.warning(f"Warning integrità temporale: {validation_results['warnings']}")
            else:
                logger.info("Integrità temporale validata ✅")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Errore validazione integrità temporale: {e}")
            validation_results['is_valid'] = False
            validation_results['errors'].append(str(e))
            return validation_results
    
    @staticmethod
    def log_temporal_split_statistics(
        df_splits: Dict[str, pd.DataFrame], 
        year_col: str, 
        month_col: str,
        day_col: Optional[str] = None
    ) -> None:
        """
        Log statistiche dettagliate sui split temporali.
        
        Args:
            df_splits: Dict con splits (train, val, test)
            year_col: Colonna anno
            month_col: Colonna mese
            day_col: Colonna giorno (opzionale)
        """
        logger.info("=== STATISTICHE SPLIT TEMPORALI ===")
        
        for split_name, df_split in df_splits.items():
            if df_split.empty:
                logger.info(f"  📊 {split_name.upper()}: Dataset vuoto")
                continue
            
            # Statistiche temporali
            year_range = f"{df_split[year_col].min()}-{df_split[year_col].max()}"
            month_range = f"{df_split[month_col].min():02d}-{df_split[month_col].max():02d}"
            
            # Calcola durata in mesi
            start_key = df_split[year_col].min() * 12 + df_split[month_col].min()
            end_key = df_split[year_col].max() * 12 + df_split[month_col].max()
            duration_months = end_key - start_key + 1
            
            logger.info(f"  📊 {split_name.upper()}:")
            logger.info(f"    📅 Range: {year_range} | Mesi: {month_range}")
            logger.info(f"    ⏱️  Durata: {duration_months} mesi")
            logger.info(f"    📈 Campioni: {len(df_split):,}")
            
            # Distribuzione per anno
            year_dist = df_split[year_col].value_counts().sort_index()
            logger.info(f"    🗓️  Per anno: {dict(year_dist)}")
    
    @staticmethod
    def create_temporal_features(
        df: pd.DataFrame, 
        year_col: str, 
        month_col: str,
        day_col: Optional[str] = None,
        prefix: str = "temporal_"
    ) -> pd.DataFrame:
        """
        Crea feature temporali derivate.
        
        Args:
            df: DataFrame
            year_col: Colonna anno
            month_col: Colonna mese
            day_col: Colonna giorno (opzionale)
            prefix: Prefisso per nuove colonne
            
        Returns:
            DataFrame con nuove feature temporali
        """
        logger.info("Creazione feature temporali derivate...")
        
        df_result = df.copy()
        
        # Feature base
        df_result[f'{prefix}year'] = df[year_col]
        df_result[f'{prefix}month'] = df[month_col]
        
        # Feature derivate dal mese
        df_result[f'{prefix}quarter'] = ((df[month_col] - 1) // 3) + 1
        df_result[f'{prefix}semester'] = ((df[month_col] - 1) // 6) + 1
        df_result[f'{prefix}season'] = df[month_col].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        # Feature ciclica per mese (sin/cos)
        df_result[f'{prefix}month_sin'] = np.sin(2 * np.pi * df[month_col] / 12)
        df_result[f'{prefix}month_cos'] = np.cos(2 * np.pi * df[month_col] / 12)
        
        # Feature dal giorno se disponibile
        if day_col and day_col in df.columns:
            df_result[f'{prefix}day'] = df[day_col]
            df_result[f'{prefix}day_of_month_normalized'] = df[day_col] / 31
            
            # Inizio/fine mese
            df_result[f'{prefix}is_month_start'] = (df[day_col] <= 5).astype(int)
            df_result[f'{prefix}is_month_end'] = (df[day_col] >= 26).astype(int)
        
        # Trend temporale (anni dalla data minima)
        min_year = df[year_col].min()
        df_result[f'{prefix}years_from_start'] = df[year_col] - min_year
        
        # Feature composita (mesi totali)
        df_result[f'{prefix}months_total'] = (df[year_col] - min_year) * 12 + df[month_col]
        
        new_features = [col for col in df_result.columns if col.startswith(prefix)]
        logger.info(f"Create {len(new_features)} feature temporali: {new_features}")
        
        return df_result
    
    @staticmethod
    def detect_temporal_anomalies(
        df: pd.DataFrame, 
        year_col: str, 
        month_col: str,
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Rileva anomalie temporali nei dati.
        
        Args:
            df: DataFrame
            year_col: Colonna anno
            month_col: Colonna mese
            target_col: Colonna target per analisi (opzionale)
            
        Returns:
            Dict con anomalie rilevate
        """
        logger.info("Rilevamento anomalie temporali...")
        
        anomalies = {
            'missing_periods': [],
            'duplicate_periods': [],
            'outlier_periods': [],
            'target_anomalies': []
        }
        
        try:
            # Crea serie temporale completa
            sort_key = AdvancedTemporalUtils.create_temporal_sort_key(
                df, year_col, month_col
            )
            df_sorted = df.iloc[sort_key.argsort()].copy()
            
            # Rileva periodi mancanti
            unique_periods = df_sorted.groupby([year_col, month_col]).size()
            min_year, max_year = df[year_col].min(), df[year_col].max()
            
            expected_periods = []
            for year in range(min_year, max_year + 1):
                for month in range(1, 13):
                    expected_periods.append((year, month))
            
            missing_periods = []
            for period in expected_periods:
                if period not in unique_periods.index:
                    missing_periods.append(f"{period[0]}/{period[1]:02d}")
            
            anomalies['missing_periods'] = missing_periods
            
            # Rileva periodi con troppi dati
            period_counts = unique_periods.values
            if len(period_counts) > 0:
                q75 = np.percentile(period_counts, 75)
                q25 = np.percentile(period_counts, 25)
                iqr = q75 - q25
                outlier_threshold = q75 + 1.5 * iqr
                
                outlier_periods = []
                for (year, month), count in unique_periods.items():
                    if count > outlier_threshold:
                        outlier_periods.append({
                            'period': f"{year}/{month:02d}",
                            'count': count,
                            'threshold': outlier_threshold
                        })
                
                anomalies['outlier_periods'] = outlier_periods
            
            # Analisi target se disponibile
            if target_col and target_col in df.columns:
                target_by_period = df_sorted.groupby([year_col, month_col])[target_col].agg(['mean', 'std', 'count'])
                
                # Rileva periodi con target anomalo
                target_means = target_by_period['mean'].dropna()
                if len(target_means) > 0:
                    target_q75 = target_means.quantile(0.75)
                    target_q25 = target_means.quantile(0.25)
                    target_iqr = target_q75 - target_q25
                    target_lower = target_q25 - 1.5 * target_iqr
                    target_upper = target_q75 + 1.5 * target_iqr
                    
                    target_anomalies = []
                    for (year, month), row in target_by_period.iterrows():
                        if pd.notna(row['mean']) and (row['mean'] < target_lower or row['mean'] > target_upper):
                            target_anomalies.append({
                                'period': f"{year}/{month:02d}",
                                'mean_value': row['mean'],
                                'expected_range': f"[{target_lower:.2f}, {target_upper:.2f}]",
                                'sample_count': row['count']
                            })
                    
                    anomalies['target_anomalies'] = target_anomalies
            
            # Log risultati
            total_anomalies = (len(anomalies['missing_periods']) + 
                             len(anomalies['outlier_periods']) + 
                             len(anomalies['target_anomalies']))
            
            if total_anomalies > 0:
                logger.warning(f"Rilevate {total_anomalies} anomalie temporali")
                if anomalies['missing_periods']:
                    logger.warning(f"Periodi mancanti: {len(anomalies['missing_periods'])}")
                if anomalies['outlier_periods']:
                    logger.warning(f"Periodi con troppi dati: {len(anomalies['outlier_periods'])}")
                if anomalies['target_anomalies']:
                    logger.warning(f"Periodi con target anomalo: {len(anomalies['target_anomalies'])}")
            else:
                logger.info("Nessuna anomalia temporale rilevata ✅")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Errore rilevamento anomalie temporali: {e}")
            return {'error': str(e)}


class TemporalSplitter:
    """Splitter avanzato per dati temporali."""
    
    @staticmethod
    def split_temporal_with_validation(
        df: pd.DataFrame,
        year_col: str,
        month_col: str,
        train_fraction: float = 0.7,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        min_samples_per_split: int = 100
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Split temporale con validazione e statistiche.
        
        Args:
            df: DataFrame da splittare
            year_col: Colonna anno
            month_col: Colonna mese
            train_fraction: Frazione training
            val_fraction: Frazione validation
            test_fraction: Frazione test
            min_samples_per_split: Minimo campioni per split
            
        Returns:
            Tuple con train, val, test DataFrame e info split
        """
        logger.info(f"Split temporale: {train_fraction:.2f}/{val_fraction:.2f}/{test_fraction:.2f}")
        
        # Verifica frazioni
        total_fraction = train_fraction + val_fraction + test_fraction
        if abs(total_fraction - 1.0) > 0.001:
            raise ValueError(f"Frazioni non sommano a 1.0: {total_fraction}")
        
        # Ordina DataFrame temporalmente
        df_sorted = AdvancedTemporalUtils.temporal_sort_dataframe(df, year_col, month_col)
        
        # Calcola indici split
        n_samples = len(df_sorted)
        train_end = int(n_samples * train_fraction)
        val_end = int(n_samples * (train_fraction + val_fraction))
        
        # Esegui split
        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()
        
        # Verifica dimensioni minime
        splits_info = {
            'train': {'samples': len(train_df), 'valid': len(train_df) >= min_samples_per_split},
            'val': {'samples': len(val_df), 'valid': len(val_df) >= min_samples_per_split},
            'test': {'samples': len(test_df), 'valid': len(test_df) >= min_samples_per_split}
        }
        
        # Log dettagliato range temporali
        DetailedLogger.log_split_temporal_ranges(df_sorted, year_col, month_col, train_end, val_end)
        
        # Valida integrità
        validation_results = AdvancedTemporalUtils.validate_temporal_split_integrity(
            train_df, val_df, test_df, year_col, month_col
        )
        
        split_info = {
            'splits_info': splits_info,
            'validation_results': validation_results,
            'total_samples': n_samples,
            'fractions_used': {
                'train': len(train_df) / n_samples,
                'val': len(val_df) / n_samples,
                'test': len(test_df) / n_samples
            }
        }
        
        return train_df, val_df, test_df, split_info