"""
Smart Configuration Manager con validazione intelligente e fallback robusti.
Ispirato al sistema di configurazione robusto di RealEstatePricePrediction.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)


class SmartConfigurationManager:
    """Manager intelligente per configurazioni con validazione e fallback."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inizializza smart configuration manager.
        
        Args:
            config_path: Path al file di configurazione
        """
        self.config_path = config_path
        self.config = {}
        self.validation_errors = []
        self.validation_warnings = []
        self.applied_fallbacks = []
        
        if config_path:
            self.load_and_validate_config(config_path)
    
    def load_and_validate_config(self, config_path: str) -> Dict[str, Any]:
        """
        Carica e valida configurazione con fallback intelligenti.
        
        Args:
            config_path: Path al file di configurazione
            
        Returns:
            Configurazione validata e arricchita
        """
        logger.info(f"🔧 Caricamento configurazione da: {config_path}")
        
        try:
            # Carica configurazione base
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # Applica default robusti
            self._apply_robust_defaults()
            
            # Valida configurazione
            self._validate_configuration()
            
            # Risolvi dipendenze
            self._resolve_configuration_dependencies()
            
            # Log risultati
            self._log_configuration_status()
            
            return self.config
            
        except Exception as e:
            logger.error(f"❌ Errore caricamento configurazione: {e}")
            logger.info("🔄 Applicazione configurazione di emergenza...")
            self.config = self._get_emergency_config()
            return self.config
    
    def resolve_target_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Risoluzione intelligente colonne target con fallback automatici.
        
        Args:
            df: DataFrame per risoluzione colonne
            
        Returns:
            Dict con info colonne target risolte
        """
        logger.info("🎯 Risoluzione intelligente colonne target...")
        
        target_config = self.config.get('target', {})
        target_candidates = target_config.get('column_candidates', ['AI_Prezzo_Ridistribuito'])
        
        resolution_result = {
            'target_column': None,
            'original_column': None,
            'needs_inverse_transform': False,
            'transform_info': {},
            'resolution_method': 'unknown'
        }
        
        # 1. Cerca colonna target principale
        target_col = None
        for candidate in target_candidates:
            if candidate in df.columns:
                target_col = candidate
                resolution_result['target_column'] = target_col
                resolution_result['resolution_method'] = 'direct_match'
                break
        
        # 2. Se non trovata, cerca pattern simili
        if target_col is None:
            target_col = self._find_target_by_pattern(df, target_candidates)
            if target_col:
                resolution_result['target_column'] = target_col
                resolution_result['resolution_method'] = 'pattern_match'
                self.applied_fallbacks.append(f"Target column fallback: {target_col}")
        
        # 3. Cerca colonna originale corrispondente
        if target_col:
            original_col = self._find_original_target_column(df, target_col)
            resolution_result['original_column'] = original_col
            
            # 4. Determina se serve inverse transform
            if '_log' in target_col.lower() and original_col is None:
                resolution_result['needs_inverse_transform'] = True
                resolution_result['transform_info'] = {'log': True, 'method': 'log1p'}
            elif original_col:
                resolution_result['transform_info'] = {'log': False, 'method': 'none'}
        
        # Log risultati
        self._log_target_resolution(resolution_result)
        
        return resolution_result
    
    def resolve_temporal_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Risoluzione intelligente colonne temporali.
        
        Args:
            df: DataFrame per risoluzione
            
        Returns:
            Dict con info colonne temporali
        """
        logger.info("📅 Risoluzione colonne temporali...")
        
        temporal_config = self.config.get('temporal_split', {})
        
        resolution_result = {
            'year_column': None,
            'month_column': None,
            'day_column': None,
            'datetime_column': None,
            'resolution_method': 'unknown',
            'temporal_available': False
        }
        
        # Pattern per colonne temporali
        year_patterns = [
            temporal_config.get('year_col', 'A_AnnoStipula'),
            'year', 'anno', 'Year', 'Anno', 'YEAR', 'ANNO'
        ]
        
        month_patterns = [
            temporal_config.get('month_col', 'A_MeseStipula'),
            'month', 'mese', 'Month', 'Mese', 'MONTH', 'MESE'
        ]
        
        day_patterns = ['day', 'giorno', 'Day', 'Giorno', 'DAY', 'GIORNO']
        
        # Cerca colonne anno
        year_col = self._find_column_by_patterns(df, year_patterns)
        if year_col:
            resolution_result['year_column'] = year_col
            resolution_result['temporal_available'] = True
        
        # Cerca colonne mese
        month_col = self._find_column_by_patterns(df, month_patterns)
        if month_col:
            resolution_result['month_column'] = month_col
        
        # Cerca colonne giorno (opzionale)
        day_col = self._find_column_by_patterns(df, day_patterns)
        if day_col:
            resolution_result['day_column'] = day_col
        
        # Cerca colonne datetime
        datetime_col = self._find_datetime_column(df)
        if datetime_col:
            resolution_result['datetime_column'] = datetime_col
            if not resolution_result['temporal_available']:
                resolution_result['temporal_available'] = True
                resolution_result['resolution_method'] = 'datetime_extraction'
        
        if resolution_result['year_column'] and resolution_result['month_column']:
            resolution_result['resolution_method'] = 'year_month_columns'
        elif resolution_result['temporal_available']:
            resolution_result['resolution_method'] = 'partial_temporal'
        
        # Log risultati
        self._log_temporal_resolution(resolution_result)
        
        return resolution_result
    
    def resolve_categorical_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Risoluzione intelligente colonne categoriche per strategie specifiche.
        
        Args:
            df: DataFrame per analisi
            
        Returns:
            Dict con info colonne categoriche
        """
        logger.info("🏷️  Risoluzione colonne categoriche...")
        
        resolution_result = {
            'category_column_for_outliers': None,
            'high_cardinality_columns': [],
            'low_cardinality_columns': [],
            'categorical_columns': [],
            'id_columns': [],
            'resolution_summary': {}
        }
        
        # Identifica tutte le colonne categoriche
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        resolution_result['categorical_columns'] = categorical_cols
        
        # Analizza cardinalità
        encoding_config = self.config.get('encoding', {})
        max_ohe_cardinality = encoding_config.get('max_ohe_cardinality', 12)
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            
            if unique_count <= max_ohe_cardinality:
                resolution_result['low_cardinality_columns'].append(col)
            else:
                resolution_result['high_cardinality_columns'].append(col)
            
            # Identifica colonne ID (cardinalità molto alta)
            if unique_count > len(df) * 0.8:  # >80% valori unici
                resolution_result['id_columns'].append(col)
        
        # Trova colonna per stratificazione outlier
        outlier_config = self.config.get('outliers', {})
        category_candidates = [
            outlier_config.get('group_by_col', 'AI_IdCategoriaCatastale'),
            'categoria', 'category', 'type', 'tipo', 'class', 'classe'
        ]
        
        category_col = self._find_column_by_patterns(df, category_candidates)
        if category_col and category_col in categorical_cols:
            # Verifica che abbia cardinalità ragionevole per stratificazione
            unique_count = df[category_col].nunique()
            if 2 <= unique_count <= 50:  # Range ragionevole per stratificazione
                resolution_result['category_column_for_outliers'] = category_col
            else:
                logger.warning(f"Colonna categoria {category_col} ha cardinalità {unique_count} "
                              f"non ottimale per stratificazione outlier")
        
        # Summary
        resolution_result['resolution_summary'] = {
            'total_categorical': len(categorical_cols),
            'low_cardinality': len(resolution_result['low_cardinality_columns']),
            'high_cardinality': len(resolution_result['high_cardinality_columns']),
            'id_columns': len(resolution_result['id_columns']),
            'outlier_category_available': resolution_result['category_column_for_outliers'] is not None
        }
        
        # Log risultati
        self._log_categorical_resolution(resolution_result)
        
        return resolution_result
    
    def optimize_config_for_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Ottimizza configurazione basata sulle caratteristiche del dataset.
        
        Args:
            df: DataFrame per ottimizzazione
            
        Returns:
            Configurazione ottimizzata
        """
        logger.info("⚙️ Ottimizzazione configurazione per dataset...")
        
        optimized_config = self.config.copy()
        optimizations_applied = []
        
        # Ottimizzazioni basate su dimensioni dataset
        n_rows, n_cols = df.shape
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        
        # Dataset grande - ottimizzazioni performance
        if n_rows > 100000 or memory_mb > 1000:
            logger.info(f"📊 Dataset grande rilevato: {n_rows:,} righe, {memory_mb:.1f}MB")
            
            # Riduci sample size per SHAP
            if 'training' in optimized_config and 'shap' in optimized_config['training']:
                original_sample = optimized_config['training']['shap'].get('sample_size', 500)
                optimized_sample = min(200, original_sample)
                optimized_config['training']['shap']['sample_size'] = optimized_sample
                optimizations_applied.append(f"SHAP sample_size: {original_sample} → {optimized_sample}")
            
            # Disabilita PCA se troppo costoso
            if n_cols > 200:
                for profile_name, profile_config in optimized_config.get('profiles', {}).items():
                    if profile_config.get('pca', {}).get('enabled', False):
                        optimized_config['profiles'][profile_name]['pca']['enabled'] = False
                        optimizations_applied.append(f"Disabilitato PCA per profilo {profile_name}")
        
        # Dataset piccolo - ottimizzazioni robustezza
        elif n_rows < 5000:
            logger.info(f"📊 Dataset piccolo rilevato: {n_rows:,} righe")
            
            # Riduci cross-validation folds
            if 'training' in optimized_config:
                original_folds = optimized_config['training'].get('cv_when_no_val', {}).get('n_splits', 5)
                optimized_folds = max(3, min(original_folds, n_rows // 1000))
                optimized_config['training']['cv_when_no_val']['n_splits'] = optimized_folds
                if optimized_folds != original_folds:
                    optimizations_applied.append(f"CV folds: {original_folds} → {optimized_folds}")
            
            # Riduci soglie outlier per preservare campioni
            if 'outliers' in optimized_config:
                original_contamination = optimized_config['outliers'].get('iso_forest_contamination', 0.02)
                optimized_contamination = max(0.01, original_contamination * 0.5)
                optimized_config['outliers']['iso_forest_contamination'] = optimized_contamination
                optimizations_applied.append(f"Outlier contamination: {original_contamination} → {optimized_contamination}")
        
        # Ottimizzazioni basate su colonne categoriche
        categorical_info = self.resolve_categorical_columns(df)
        high_card_cols = len(categorical_info['high_cardinality_columns'])
        
        if high_card_cols > 10:
            logger.info(f"🏷️  Molte colonne alta cardinalità: {high_card_cols}")
            
            # Riduci soglia per target encoding
            if 'encoding' in optimized_config:
                original_threshold = optimized_config['encoding'].get('max_ohe_cardinality', 12)
                optimized_threshold = max(5, original_threshold - 2)
                optimized_config['encoding']['max_ohe_cardinality'] = optimized_threshold
                optimizations_applied.append(f"OHE threshold: {original_threshold} → {optimized_threshold}")
        
        # Ottimizzazioni temporali
        temporal_info = self.resolve_temporal_columns(df)
        if not temporal_info['temporal_available']:
            logger.warning("⚠️ Colonne temporali non disponibili - disabilito split temporale")
            if 'temporal_split' in optimized_config:
                optimized_config['temporal_split']['mode'] = 'random'
                optimizations_applied.append("Split temporale → split random")
        
        # Log ottimizzazioni applicate
        if optimizations_applied:
            logger.info("🎯 Ottimizzazioni applicate:")
            for opt in optimizations_applied:
                logger.info(f"  • {opt}")
        else:
            logger.info("✅ Configurazione già ottimale per questo dataset")
        
        return optimized_config
    
    def _apply_robust_defaults(self) -> None:
        """Applica default robusti per configurazione mancante."""
        defaults = {
            'paths': {
                'raw_data': 'data/raw',
                'preprocessed_data': 'data/preprocessed',
                'models_dir': 'models',
                'raw_filename': 'raw.parquet',
                'preprocessed_filename': 'preprocessed.parquet'
            },
            'target': {
                'column_candidates': ['AI_Prezzo_Ridistribuito', 'AI_Prezzo', 'prezzo', 'price'],
                'log_transform': False
            },
            'temporal_split': {
                'year_col': 'A_AnnoStipula',
                'month_col': 'A_MeseStipula',
                'mode': 'fraction',
                'train_fraction': 0.7,
                'valid_fraction': 0.15
            },
            'outliers': {
                'method': 'ensemble',
                'z_thresh': 3.0,
                'iqr_factor': 1.5,
                'iso_forest_contamination': 0.02,
                'group_by_col': 'AI_IdCategoriaCatastale',
                'min_group_size': 30
            },
            'encoding': {
                'max_ohe_cardinality': 12
            },
            'training': {
                'primary_metric': 'r2',
                'report_metrics': ['r2', 'rmse', 'mse', 'mae', 'mape'],
                'seed': 42,
                'shap': {
                    'enabled': True,
                    'sample_size': 500,
                    'max_display': 20
                }
            },
            'quality_checks': {
                'check_temporal_leakage': True,
                'check_target_leakage': True,
                'check_category_distribution': True,
                'max_category_drift': 0.05
            }
        }
        
        # Applica default per sezioni mancanti
        for section, section_defaults in defaults.items():
            if section not in self.config:
                self.config[section] = section_defaults
                self.applied_fallbacks.append(f"Sezione mancante '{section}' - applicati default")
            else:
                # Applica default per chiavi mancanti nella sezione
                for key, default_value in section_defaults.items():
                    if key not in self.config[section]:
                        self.config[section][key] = default_value
                        self.applied_fallbacks.append(f"Chiave mancante '{section}.{key}' - applicato default")
    
    def _validate_configuration(self) -> None:
        """Valida configurazione e raccoglie errori/warning."""
        # Valida paths
        paths = self.config.get('paths', {})
        for path_name, path_value in paths.items():
            if not isinstance(path_value, str):
                self.validation_errors.append(f"Path '{path_name}' deve essere stringa")
        
        # Valida training
        training = self.config.get('training', {})
        if 'primary_metric' in training:
            valid_metrics = ['r2', 'rmse', 'mae', 'mse', 'mape']
            if training['primary_metric'] not in valid_metrics:
                self.validation_warnings.append(
                    f"Metrica primaria '{training['primary_metric']}' non standard. "
                    f"Valide: {valid_metrics}"
                )
        
        # Valida outliers
        outliers = self.config.get('outliers', {})
        if 'z_thresh' in outliers:
            z_thresh = outliers['z_thresh']
            if not isinstance(z_thresh, (int, float)) or z_thresh <= 0:
                self.validation_errors.append("outliers.z_thresh deve essere numero positivo")
            elif z_thresh < 1.0 or z_thresh > 5.0:
                self.validation_warnings.append(
                    f"outliers.z_thresh={z_thresh} fuori range raccomandato [1.0, 5.0]"
                )
        
        # Valida frazioni temporal split
        temporal = self.config.get('temporal_split', {})
        fractions = ['train_fraction', 'valid_fraction']
        total_fraction = 0
        for frac_name in fractions:
            if frac_name in temporal:
                frac_value = temporal[frac_name]
                if not isinstance(frac_value, (int, float)) or not 0 < frac_value < 1:
                    self.validation_errors.append(f"temporal_split.{frac_name} deve essere in (0, 1)")
                else:
                    total_fraction += frac_value
        
        if total_fraction >= 1.0:
            self.validation_errors.append(
                f"Somma frazioni temporal_split >= 1.0: {total_fraction}"
            )
    
    def _resolve_configuration_dependencies(self) -> None:
        """Risolve dipendenze tra configurazioni."""
        # Se PCA abilitato, assicura che scaling sia abilitato
        profiles = self.config.get('profiles', {})
        for profile_name, profile_config in profiles.items():
            if profile_config.get('pca', {}).get('enabled', False):
                if not profile_config.get('scaling', {}).get('scaler_type'):
                    self.config['profiles'][profile_name]['scaling'] = {'scaler_type': 'standard'}
                    self.applied_fallbacks.append(
                        f"Profilo {profile_name}: PCA abilitato → scaling automatico"
                    )
        
        # Se quality checks abilitati ma colonne temporali non specificate
        quality_checks = self.config.get('quality_checks', {})
        if quality_checks.get('check_temporal_leakage', False):
            temporal = self.config.get('temporal_split', {})
            if not temporal.get('year_col') or not temporal.get('month_col'):
                self.validation_warnings.append(
                    "Quality check temporal_leakage abilitato ma colonne temporali non specificate"
                )
    
    def _find_target_by_pattern(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Trova colonna target usando pattern matching."""
        # Pattern comuni per target
        target_patterns = ['prezzo', 'price', 'valore', 'value', 'target', 'y']
        
        for col in df.columns:
            col_lower = col.lower()
            for pattern in target_patterns:
                if pattern in col_lower and df[col].dtype in ['int64', 'float64']:
                    logger.info(f"Target trovato per pattern '{pattern}': {col}")
                    return col
        
        return None
    
    def _find_original_target_column(self, df: pd.DataFrame, target_col: str) -> Optional[str]:
        """Trova colonna target originale corrispondente."""
        # Varianti possibili
        original_variants = [
            target_col.replace('_log', '_original'),
            target_col.replace('_log', ''),
            target_col + '_original',
            target_col.replace('Ridistribuito', 'Originale'),
            target_col.replace('ridistribuito', 'originale')
        ]
        
        for variant in original_variants:
            if variant in df.columns:
                logger.info(f"Colonna originale trovata: {variant}")
                return variant
        
        return None
    
    def _find_column_by_patterns(self, df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
        """Trova colonna usando lista di pattern."""
        for pattern in patterns:
            if pattern in df.columns:
                return pattern
        
        # Pattern matching case-insensitive
        for pattern in patterns:
            pattern_lower = pattern.lower()
            for col in df.columns:
                if pattern_lower in col.lower():
                    return col
        
        return None
    
    def _find_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Trova colonna datetime."""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            return datetime_cols[0]
        
        # Cerca pattern datetime in object columns
        datetime_patterns = ['date', 'datetime', 'timestamp', 'time']
        for col in df.select_dtypes(include=['object']).columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in datetime_patterns):
                return col
        
        return None
    
    def _get_emergency_config(self) -> Dict[str, Any]:
        """Configurazione di emergenza minimale."""
        return {
            'paths': {
                'raw_data': 'data/raw',
                'preprocessed_data': 'data/preprocessed',
                'models_dir': 'models'
            },
            'target': {
                'column_candidates': ['AI_Prezzo_Ridistribuito', 'prezzo', 'target']
            },
            'training': {
                'primary_metric': 'r2',
                'models': {
                    'linear': {'enabled': True, 'trials': 1},
                    'rf': {'enabled': True, 'trials': 10}
                }
            },
            'profiles': {
                'scaled': {'enabled': True},
                'tree': {'enabled': True}
            }
        }
    
    def _log_configuration_status(self) -> None:
        """Log status configurazione."""
        logger.info("=== STATUS CONFIGURAZIONE ===")
        
        if self.validation_errors:
            logger.error(f"❌ Errori validazione: {len(self.validation_errors)}")
            for error in self.validation_errors:
                logger.error(f"  • {error}")
        
        if self.validation_warnings:
            logger.warning(f"⚠️ Warning validazione: {len(self.validation_warnings)}")
            for warning in self.validation_warnings:
                logger.warning(f"  • {warning}")
        
        if self.applied_fallbacks:
            logger.info(f"🔄 Fallback applicati: {len(self.applied_fallbacks)}")
            for fallback in self.applied_fallbacks:
                logger.info(f"  • {fallback}")
        
        if not self.validation_errors and not self.validation_warnings:
            logger.info("✅ Configurazione validata con successo")
    
    def _log_target_resolution(self, resolution: Dict[str, Any]) -> None:
        """Log risoluzione target."""
        logger.info("🎯 Risoluzione Target:")
        logger.info(f"  Target column: {resolution['target_column']}")
        logger.info(f"  Original column: {resolution['original_column']}")
        logger.info(f"  Inverse transform: {resolution['needs_inverse_transform']}")
        logger.info(f"  Method: {resolution['resolution_method']}")
    
    def _log_temporal_resolution(self, resolution: Dict[str, Any]) -> None:
        """Log risoluzione temporale."""
        logger.info("📅 Risoluzione Temporale:")
        logger.info(f"  Year column: {resolution['year_column']}")
        logger.info(f"  Month column: {resolution['month_column']}")
        logger.info(f"  Day column: {resolution['day_column']}")
        logger.info(f"  Available: {resolution['temporal_available']}")
        logger.info(f"  Method: {resolution['resolution_method']}")
    
    def _log_categorical_resolution(self, resolution: Dict[str, Any]) -> None:
        """Log risoluzione categoriche."""
        summary = resolution['resolution_summary']
        logger.info("🏷️  Risoluzione Categoriche:")
        logger.info(f"  Total: {summary['total_categorical']}")
        logger.info(f"  Low cardinality: {summary['low_cardinality']}")
        logger.info(f"  High cardinality: {summary['high_cardinality']}")
        logger.info(f"  ID columns: {summary['id_columns']}")
        logger.info(f"  Outlier category: {resolution['category_column_for_outliers']}")


class ConfigurationValidator:
    """Validatore avanzato per configurazioni complesse."""
    
    @staticmethod
    def validate_model_configuration(models_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida configurazione modelli.
        
        Args:
            models_config: Configurazione modelli
            
        Returns:
            Dict con risultati validazione
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'enabled_models': []
        }
        
        if not isinstance(models_config, dict):
            validation_result['errors'].append("Configurazione modelli deve essere dict")
            validation_result['is_valid'] = False
            return validation_result
        
        # Controlla modelli abilitati
        enabled_count = 0
        for model_name, model_config in models_config.items():
            if isinstance(model_config, dict) and model_config.get('enabled', False):
                enabled_count += 1
                validation_result['enabled_models'].append(model_name)
                
                # Valida trials
                trials = model_config.get('trials', 1)
                if not isinstance(trials, int) or trials <= 0:
                    validation_result['errors'].append(
                        f"Modello {model_name}: trials deve essere intero positivo"
                    )
                    validation_result['is_valid'] = False
                elif trials > 200:
                    validation_result['warnings'].append(
                        f"Modello {model_name}: trials={trials} potrebbe essere eccessivo"
                    )
        
        # Controlla che almeno un modello sia abilitato
        if enabled_count == 0:
            validation_result['errors'].append("Nessun modello abilitato")
            validation_result['is_valid'] = False
        elif enabled_count > 10:
            validation_result['warnings'].append(
                f"Molti modelli abilitati ({enabled_count}) - training potrebbe essere lento"
            )
        
        return validation_result
    
    @staticmethod
    def validate_profiles_configuration(profiles_config: Dict[str, Any]) -> Dict[str, Any]:
        """Valida configurazione profili."""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'enabled_profiles': []
        }
        
        required_profiles = ['scaled', 'tree']
        enabled_count = 0
        
        for profile_name, profile_config in profiles_config.items():
            if isinstance(profile_config, dict) and profile_config.get('enabled', False):
                enabled_count += 1
                validation_result['enabled_profiles'].append(profile_name)
        
        # Controlla profili richiesti
        for required_profile in required_profiles:
            if required_profile not in profiles_config:
                validation_result['warnings'].append(
                    f"Profilo raccomandato '{required_profile}' non presente"
                )
            elif not profiles_config[required_profile].get('enabled', False):
                validation_result['warnings'].append(
                    f"Profilo raccomandato '{required_profile}' non abilitato"
                )
        
        if enabled_count == 0:
            validation_result['errors'].append("Nessun profilo abilitato")
            validation_result['is_valid'] = False
        
        return validation_result