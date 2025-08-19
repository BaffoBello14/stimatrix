"""
Operazioni robuste sui dati con fallback intelligenti.
Ispirato al sistema robusto di RealEstatePricePrediction.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import pandas as pd
import numpy as np
from utils.logger import get_logger
from utils.detailed_logging import DetailedLogger

logger = get_logger(__name__)


class RobustDataOperations:
    """Operazioni robuste sui dati con gestione errori e fallback intelligenti."""
    
    @staticmethod
    def safe_column_operation(
        df: pd.DataFrame, 
        columns: List[str], 
        operation_name: str, 
        operation_func: Callable,
        fallback_func: Optional[Callable] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Esegue operazione sicura su colonne con gestione missing e fallback.
        
        Args:
            df: DataFrame su cui operare
            columns: Lista colonne richieste
            operation_name: Nome dell'operazione per logging
            operation_func: Funzione principale da eseguire
            fallback_func: Funzione fallback opzionale
            **kwargs: Parametri aggiuntivi per le funzioni
            
        Returns:
            Tuple con DataFrame risultante e info operazione
        """
        existing_cols = [col for col in columns if col in df.columns]
        missing_cols = [col for col in columns if col not in df.columns]
        
        # Log risultati ricerca colonne
        DetailedLogger.log_column_operation_results(
            columns, existing_cols, missing_cols, operation_name
        )
        
        operation_info = {
            'operation_name': operation_name,
            'requested_columns': columns,
            'existing_columns': existing_cols,
            'missing_columns': missing_cols,
            'columns_processed_count': len(existing_cols),
            'success': False,
            'fallback_used': False
        }
        
        # Nessuna colonna da processare
        if not existing_cols:
            logger.info(f"{operation_name} - Nessuna colonna da processare, dataset invariato")
            operation_info['skipped'] = True
            operation_info['success'] = True
            return df, operation_info
        
        try:
            # Esegui operazione principale
            result_df, additional_info = operation_func(df, existing_cols, **kwargs)
            operation_info.update(additional_info)
            operation_info['success'] = True
            
            logger.info(f"{operation_name} - Operazione completata con successo")
            return result_df, operation_info
            
        except Exception as e:
            logger.warning(f"{operation_name} - Operazione fallita: {e}")
            
            # Prova fallback se disponibile
            if fallback_func:
                try:
                    logger.info(f"{operation_name} - Tentativo fallback...")
                    result_df, fallback_info = fallback_func(df, existing_cols, **kwargs)
                    operation_info.update(fallback_info)
                    operation_info['fallback_used'] = True
                    operation_info['success'] = True
                    operation_info['original_error'] = str(e)
                    
                    logger.info(f"{operation_name} - Fallback completato con successo")
                    return result_df, operation_info
                    
                except Exception as fallback_error:
                    logger.error(f"{operation_name} - Anche il fallback è fallito: {fallback_error}")
                    operation_info['error'] = str(e)
                    operation_info['fallback_error'] = str(fallback_error)
            else:
                operation_info['error'] = str(e)
            
            # Ritorna dataset originale se tutto fallisce
            logger.warning(f"{operation_name} - Ritorno dataset originale")
            return df, operation_info
    
    @staticmethod
    def remove_columns_safe(
        df: pd.DataFrame, 
        columns_to_remove: List[str], 
        operation_name: str = "RIMOZIONE COLONNE"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Rimozione sicura di colonne con gestione missing.
        
        Args:
            df: DataFrame
            columns_to_remove: Colonne da rimuovere
            operation_name: Nome operazione
            
        Returns:
            Tuple con DataFrame e info rimozione
        """
        def remove_operation(df_inner, existing_cols, **kwargs):
            df_result = df_inner.drop(columns=existing_cols)
            return df_result, {'columns_removed': existing_cols}
        
        return RobustDataOperations.safe_column_operation(
            df, columns_to_remove, operation_name, remove_operation
        )
    
    @staticmethod
    def select_columns_safe(
        df: pd.DataFrame, 
        columns_to_select: List[str], 
        operation_name: str = "SELEZIONE COLONNE"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Selezione sicura di colonne con gestione missing.
        
        Args:
            df: DataFrame
            columns_to_select: Colonne da selezionare
            operation_name: Nome operazione
            
        Returns:
            Tuple con DataFrame selezionato e info
        """
        def select_operation(df_inner, existing_cols, **kwargs):
            df_result = df_inner[existing_cols]
            return df_result, {'columns_selected': existing_cols}
        
        return RobustDataOperations.safe_column_operation(
            df, columns_to_select, operation_name, select_operation
        )
    
    @staticmethod
    def apply_function_safe(
        df: pd.DataFrame, 
        columns: List[str], 
        func: Callable, 
        operation_name: str,
        fallback_value: Any = None,
        **func_kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Applica funzione sicura a colonne con fallback.
        
        Args:
            df: DataFrame
            columns: Colonne su cui applicare la funzione
            func: Funzione da applicare
            operation_name: Nome operazione
            fallback_value: Valore fallback in caso di errore
            **func_kwargs: Parametri per la funzione
            
        Returns:
            Tuple con DataFrame e info operazione
        """
        def apply_operation(df_inner, existing_cols, **kwargs):
            df_result = df_inner.copy()
            applied_cols = []
            failed_cols = []
            
            for col in existing_cols:
                try:
                    df_result[col] = func(df_result[col], **func_kwargs)
                    applied_cols.append(col)
                except Exception as e:
                    logger.warning(f"Funzione fallita per colonna {col}: {e}")
                    failed_cols.append(col)
                    if fallback_value is not None:
                        df_result[col] = fallback_value
            
            return df_result, {
                'applied_columns': applied_cols,
                'failed_columns': failed_cols,
                'success_rate': len(applied_cols) / len(existing_cols) if existing_cols else 0
            }
        
        return RobustDataOperations.safe_column_operation(
            df, columns, operation_name, apply_operation
        )


class RobustColumnAnalyzer:
    """Analizzatore robusto di colonne per operazioni di preprocessing."""
    
    @staticmethod
    def find_constant_columns(
        df: pd.DataFrame, 
        threshold: float = 0.95,
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Trova colonne quasi costanti con statistiche dettagliate.
        
        Args:
            df: DataFrame da analizzare
            threshold: Soglia per considerare colonna costante
            exclude_columns: Colonne da escludere dall'analisi
            
        Returns:
            Tuple con lista colonne costanti e statistiche
        """
        exclude_columns = exclude_columns or []
        constant_columns = []
        column_stats = {}
        
        for col in df.columns:
            if col in exclude_columns:
                continue
                
            try:
                # Calcola frequenza valore più comune
                value_counts = df[col].value_counts(normalize=True, dropna=False)
                
                if len(value_counts) == 0:
                    continue
                    
                max_frequency = value_counts.iloc[0]
                most_common_value = value_counts.index[0]
                unique_values = len(value_counts)
                
                column_stats[col] = {
                    'max_frequency': max_frequency,
                    'unique_values': unique_values,
                    'most_common_value': most_common_value,
                    'dtype': str(df[col].dtype),
                    'null_count': df[col].isnull().sum()
                }
                
                if max_frequency >= threshold:
                    constant_columns.append(col)
                    
            except Exception as e:
                logger.warning(f"Errore analisi colonna {col}: {e}")
                column_stats[col] = {'error': str(e)}
        
        return constant_columns, column_stats
    
    @staticmethod
    def find_highly_correlated_columns(
        df: pd.DataFrame, 
        threshold: float = 0.95,
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Trova colonne altamente correlate con statistiche.
        
        Args:
            df: DataFrame da analizzare
            threshold: Soglia correlazione
            exclude_columns: Colonne da escludere
            
        Returns:
            Tuple con colonne da rimuovere e statistiche correlazioni
        """
        exclude_columns = exclude_columns or []
        
        # Seleziona solo colonne numeriche
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
        
        if len(numeric_cols) < 2:
            return [], {}
        
        try:
            corr_matrix = df[numeric_cols].corr().abs()
            
            # Trova coppie altamente correlate
            high_corr_pairs = []
            columns_to_remove = set()
            correlation_stats = {}
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    
                    if pd.notna(corr_val) and corr_val >= threshold:
                        high_corr_pairs.append({
                            'col1': col1,
                            'col2': col2,
                            'correlation': corr_val
                        })
                        
                        # Rimuovi la seconda colonna (strategia semplice)
                        columns_to_remove.add(col2)
                        correlation_stats[col2] = {
                            'correlation': corr_val,
                            'correlated_with': col1
                        }
            
            return list(columns_to_remove), correlation_stats
            
        except Exception as e:
            logger.error(f"Errore calcolo correlazioni: {e}")
            return [], {}
    
    @staticmethod
    def find_columns_by_pattern(
        df: pd.DataFrame, 
        patterns: List[str], 
        case_sensitive: bool = False
    ) -> Dict[str, List[str]]:
        """
        Trova colonne che matchano pattern specifici.
        
        Args:
            df: DataFrame
            patterns: Lista pattern da cercare
            case_sensitive: Se la ricerca è case-sensitive
            
        Returns:
            Dict con pattern e colonne matchate
        """
        import fnmatch
        
        results = {}
        columns = df.columns.tolist()
        
        if not case_sensitive:
            columns_lower = [col.lower() for col in columns]
        
        for pattern in patterns:
            matched_cols = []
            
            if case_sensitive:
                matched_cols = [col for col in columns if fnmatch.fnmatch(col, pattern)]
            else:
                pattern_lower = pattern.lower()
                for i, col_lower in enumerate(columns_lower):
                    if fnmatch.fnmatch(col_lower, pattern_lower):
                        matched_cols.append(columns[i])
            
            results[pattern] = matched_cols
        
        return results
    
    @staticmethod
    def analyze_missing_values(
        df: pd.DataFrame, 
        threshold: float = 0.95
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Analizza valori mancanti e identifica colonne con troppi NA.
        
        Args:
            df: DataFrame da analizzare
            threshold: Soglia percentuale NA per rimozione
            
        Returns:
            Tuple con colonne da rimuovere e statistiche NA
        """
        columns_to_remove = []
        missing_stats = {}
        
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_percentage = null_count / len(df)
            
            missing_stats[col] = {
                'null_count': null_count,
                'null_percentage': null_percentage,
                'dtype': str(df[col].dtype),
                'total_values': len(df)
            }
            
            if null_percentage >= threshold:
                columns_to_remove.append(col)
        
        return columns_to_remove, missing_stats


class RobustDataValidator:
    """Validatore robusto per operazioni sui dati."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, operation_name: str = "VALIDAZIONE") -> Dict[str, Any]:
        """
        Valida DataFrame per operazioni robuste.
        
        Args:
            df: DataFrame da validare
            operation_name: Nome operazione per logging
            
        Returns:
            Dict con risultati validazione
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Controlli base
        if df.empty:
            validation_results['errors'].append("DataFrame vuoto")
            validation_results['is_valid'] = False
        
        if len(df.columns) == 0:
            validation_results['errors'].append("Nessuna colonna presente")
            validation_results['is_valid'] = False
        
        # Controlli duplicati
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            validation_results['warnings'].append(f"Colonne duplicate: {duplicate_cols}")
        
        # Controlli tipi dati
        mixed_types = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Controlla se ci sono tipi misti
                types = df[col].dropna().apply(type).unique()
                if len(types) > 1:
                    mixed_types.append(col)
        
        if mixed_types:
            validation_results['warnings'].append(f"Colonne con tipi misti: {mixed_types}")
        
        # Controlli memoria
        if validation_results['memory_usage_mb'] > 1000:  # 1GB
            validation_results['warnings'].append(f"Dataset grande: {validation_results['memory_usage_mb']:.1f}MB")
        
        # Log risultati
        if validation_results['errors']:
            logger.error(f"{operation_name} - Errori validazione: {validation_results['errors']}")
        if validation_results['warnings']:
            logger.warning(f"{operation_name} - Warning validazione: {validation_results['warnings']}")
        else:
            logger.info(f"{operation_name} - Validazione superata ✅")
        
        return validation_results
    
    @staticmethod
    def validate_columns_exist(
        df: pd.DataFrame, 
        required_columns: List[str], 
        operation_name: str = "VALIDAZIONE COLONNE"
    ) -> Dict[str, Any]:
        """
        Valida esistenza colonne richieste.
        
        Args:
            df: DataFrame
            required_columns: Colonne richieste
            operation_name: Nome operazione
            
        Returns:
            Dict con risultati validazione
        """
        existing_cols = [col for col in required_columns if col in df.columns]
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        validation_results = {
            'is_valid': len(missing_cols) == 0,
            'required_columns': required_columns,
            'existing_columns': existing_cols,
            'missing_columns': missing_cols,
            'success_rate': len(existing_cols) / len(required_columns) if required_columns else 1.0
        }
        
        if missing_cols:
            logger.warning(f"{operation_name} - Colonne mancanti: {missing_cols}")
        else:
            logger.info(f"{operation_name} - Tutte le colonne richieste presenti ✅")
        
        return validation_results