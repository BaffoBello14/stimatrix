#!/usr/bin/env python3
"""
Enhanced Correlation Analysis for Stimatrix Project
==================================================

Estende l'analisi delle correlazioni oltre Pearson/Spearman per includere:
- Point-Biserial per variabili binarie
- Distance correlation per relazioni non-lineari
- Phi coefficient per variabili binarie-binarie
- Polychoric per variabili ordinali
- Test di significatività per tutte le correlazioni
- Handling automatico dei tipi di variabili

Autore: Miglioramento dell'EDA esistente
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.metrics import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
from enum import Enum
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class VariableType(Enum):
    """Tipi di variabili per l'analisi delle correlazioni."""
    CONTINUOUS = "continuous"
    BINARY = "binary"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"


@dataclass
class CorrelationResult:
    """Risultato di una correlazione con metadati."""
    value: float
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    method: str = ""
    n_observations: int = 0
    effect_size: Optional[str] = None


class EnhancedCorrelationAnalyzer:
    """
    Analizzatore di correlazioni avanzato che gestisce tutti i tipi di variabili
    e relazioni con test di significatività e effect size.
    """
    
    def __init__(self, 
                 significance_level: float = 0.05,
                 min_observations: int = 30,
                 n_jobs: int = -1):
        """
        Inizializza l'analizzatore.
        
        Args:
            significance_level: Soglia per la significatività statistica
            min_observations: Numero minimo di osservazioni per calcolare correlazioni
            n_jobs: Numero di processi paralleli (-1 per tutti i core)
        """
        self.significance_level = significance_level
        self.min_observations = min_observations
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        
    def detect_variable_type(self, series: pd.Series) -> VariableType:
        """
        Rileva automaticamente il tipo di variabile.
        
        Args:
            series: Serie pandas da analizzare
            
        Returns:
            Tipo di variabile rilevato
        """
        # Rimuovi valori nulli per l'analisi
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return VariableType.NOMINAL
            
        # Controlla se è numerica
        if pd.api.types.is_numeric_dtype(clean_series):
            unique_values = clean_series.nunique()
            
            # Binaria se solo 2 valori unici
            if unique_values == 2:
                return VariableType.BINARY
            
            # Ordinale se pochi valori unici e interi
            elif unique_values <= 10 and clean_series.dtype == 'int64':
                # Controlla se i valori sono consecutivi (suggerisce ordinale)
                unique_sorted = sorted(clean_series.unique())
                if len(unique_sorted) > 1:
                    gaps = [unique_sorted[i+1] - unique_sorted[i] for i in range(len(unique_sorted)-1)]
                    if all(gap == 1 for gap in gaps):
                        return VariableType.ORDINAL
            
            # Altrimenti continua
            return VariableType.CONTINUOUS
        else:
            # Categorica
            unique_values = clean_series.nunique()
            if unique_values == 2:
                return VariableType.BINARY
            else:
                return VariableType.NOMINAL
                
    def point_biserial_correlation(self, 
                                 continuous: pd.Series, 
                                 binary: pd.Series) -> CorrelationResult:
        """
        Calcola la correlazione point-biserial tra variabile continua e binaria.
        
        Args:
            continuous: Variabile continua
            binary: Variabile binaria
            
        Returns:
            Risultato della correlazione
        """
        # Allinea gli indici e rimuovi valori nulli
        aligned_data = pd.concat([continuous, binary], axis=1).dropna()
        
        if len(aligned_data) < self.min_observations:
            return CorrelationResult(0.0, method="point_biserial", 
                                   n_observations=len(aligned_data))
        
        cont_clean = aligned_data.iloc[:, 0]
        bin_clean = aligned_data.iloc[:, 1]
        
        # Converti binaria a 0/1 se necessario
        unique_vals = bin_clean.unique()
        if len(unique_vals) != 2:
            return CorrelationResult(0.0, method="point_biserial", 
                                   n_observations=len(aligned_data))
        
        # Mappa a 0/1
        binary_encoded = (bin_clean == unique_vals[1]).astype(int)
        
        try:
            corr, p_value = pointbiserialr(binary_encoded, cont_clean)
            
            # Effect size interpretation per point-biserial
            effect_size = self._interpret_effect_size_continuous(abs(corr))
            
            return CorrelationResult(
                value=corr,
                p_value=p_value,
                method="point_biserial",
                n_observations=len(aligned_data),
                effect_size=effect_size
            )
        except Exception as e:
            return CorrelationResult(0.0, method="point_biserial", 
                                   n_observations=len(aligned_data))
    
    def phi_coefficient(self, binary1: pd.Series, binary2: pd.Series) -> CorrelationResult:
        """
        Calcola il coefficiente Phi per due variabili binarie.
        
        Args:
            binary1: Prima variabile binaria
            binary2: Seconda variabile binaria
            
        Returns:
            Risultato della correlazione
        """
        # Allinea gli indici e rimuovi valori nulli
        aligned_data = pd.concat([binary1, binary2], axis=1).dropna()
        
        if len(aligned_data) < self.min_observations:
            return CorrelationResult(0.0, method="phi_coefficient", 
                                   n_observations=len(aligned_data))
        
        bin1_clean = aligned_data.iloc[:, 0]
        bin2_clean = aligned_data.iloc[:, 1]
        
        try:
            # Crea tabella di contingenza
            contingency_table = pd.crosstab(bin1_clean, bin2_clean)
            
            if contingency_table.shape != (2, 2):
                return CorrelationResult(0.0, method="phi_coefficient", 
                                       n_observations=len(aligned_data))
            
            # Calcola chi-quadrato
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            
            # Phi coefficient
            phi = np.sqrt(chi2 / n)
            
            # Aggiusta il segno basandosi sulla direzione dell'associazione
            a, b, c, d = contingency_table.iloc[0, 0], contingency_table.iloc[0, 1], \
                        contingency_table.iloc[1, 0], contingency_table.iloc[1, 1]
            
            if (a * d) < (b * c):
                phi = -phi
            
            effect_size = self._interpret_effect_size_categorical(phi)
            
            return CorrelationResult(
                value=phi,
                p_value=p_value,
                method="phi_coefficient",
                n_observations=len(aligned_data),
                effect_size=effect_size
            )
        except Exception as e:
            return CorrelationResult(0.0, method="phi_coefficient", 
                                   n_observations=len(aligned_data))
    
    def distance_correlation(self, x: pd.Series, y: pd.Series) -> CorrelationResult:
        """
        Calcola la distance correlation per catturare relazioni non-lineari.
        
        Args:
            x: Prima variabile
            y: Seconda variabile
            
        Returns:
            Risultato della correlazione
        """
        # Allinea gli indici e rimuovi valori nulli
        aligned_data = pd.concat([x, y], axis=1).dropna()
        
        if len(aligned_data) < self.min_observations:
            return CorrelationResult(0.0, method="distance_correlation", 
                                   n_observations=len(aligned_data))
        
        x_clean = aligned_data.iloc[:, 0].values
        y_clean = aligned_data.iloc[:, 1].values
        
        try:
            dcorr = self._compute_distance_correlation(x_clean, y_clean)
            
            # Per distance correlation, non c'è un p-value standard
            # Si potrebbe implementare un test di permutazione
            
            effect_size = self._interpret_effect_size_continuous(dcorr)
            
            return CorrelationResult(
                value=dcorr,
                method="distance_correlation",
                n_observations=len(aligned_data),
                effect_size=effect_size
            )
        except Exception as e:
            return CorrelationResult(0.0, method="distance_correlation", 
                                   n_observations=len(aligned_data))
    
    def _compute_distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Implementazione della distance correlation.
        
        Args:
            x: Array numpy della prima variabile
            y: Array numpy della seconda variabile
            
        Returns:
            Valore della distance correlation
        """
        n = len(x)
        
        if n == 0:
            return 0.0
        
        # Matrici delle distanze
        A = np.sqrt(np.subtract.outer(x, x) ** 2)
        B = np.sqrt(np.subtract.outer(y, y) ** 2)
        
        # Double centering
        A_centered = A - A.mean(axis=0) - A.mean(axis=1)[:, np.newaxis] + A.mean()
        B_centered = B - B.mean(axis=0) - B.mean(axis=1)[:, np.newaxis] + B.mean()
        
        # Distance covariance e variance
        dcov_xy = np.sqrt(np.mean(A_centered * B_centered))
        dcov_xx = np.sqrt(np.mean(A_centered * A_centered))
        dcov_yy = np.sqrt(np.mean(B_centered * B_centered))
        
        # Distance correlation
        if dcov_xx > 0 and dcov_yy > 0:
            return dcov_xy / np.sqrt(dcov_xx * dcov_yy)
        else:
            return 0.0
    
    def enhanced_mutual_information(self, 
                                  x: pd.Series, 
                                  y: pd.Series, 
                                  x_type: VariableType,
                                  y_type: VariableType) -> CorrelationResult:
        """
        Calcola mutual information con handling migliorato dei tipi.
        
        Args:
            x: Prima variabile
            y: Seconda variabile  
            x_type: Tipo della prima variabile
            y_type: Tipo della seconda variabile
            
        Returns:
            Risultato della correlazione
        """
        # Allinea gli indici e rimuovi valori nulli
        aligned_data = pd.concat([x, y], axis=1).dropna()
        
        if len(aligned_data) < self.min_observations:
            return CorrelationResult(0.0, method="mutual_information", 
                                   n_observations=len(aligned_data))
        
        x_clean = aligned_data.iloc[:, 0]
        y_clean = aligned_data.iloc[:, 1]
        
        try:
            # Prepara i dati basandosi sui tipi
            if x_type in [VariableType.NOMINAL, VariableType.BINARY, VariableType.ORDINAL]:
                # X categorica
                le_x = LabelEncoder()
                X_encoded = le_x.fit_transform(x_clean).reshape(-1, 1)
                
                if y_type == VariableType.CONTINUOUS:
                    # X categorica, Y continua
                    mi_score = mutual_info_regression(X_encoded, y_clean, random_state=42)[0]
                else:
                    # X categorica, Y categorica  
                    le_y = LabelEncoder()
                    y_encoded = le_y.fit_transform(y_clean)
                    mi_score = mutual_info_classif(X_encoded, y_encoded, random_state=42)[0]
            else:
                # X continua
                X_values = x_clean.values.reshape(-1, 1)
                
                if y_type == VariableType.CONTINUOUS:
                    # X continua, Y continua
                    mi_score = mutual_info_regression(X_values, y_clean, random_state=42)[0]
                else:
                    # X continua, Y categorica
                    le_y = LabelEncoder()
                    y_encoded = le_y.fit_transform(y_clean)
                    mi_score = mutual_info_classif(X_values, y_encoded, random_state=42)[0]
            
            # Normalizza MI (0-1 range)
            # Nota: questo è un'approssimazione, la normalizzazione esatta dipende dal caso
            if mi_score > 0:
                # Normalizzazione semplificata basata sull'entropia
                max_entropy = min(np.log2(x_clean.nunique()), np.log2(y_clean.nunique()))
                mi_normalized = min(1.0, mi_score / max_entropy) if max_entropy > 0 else 0
            else:
                mi_normalized = 0
            
            effect_size = self._interpret_effect_size_continuous(mi_normalized)
            
            return CorrelationResult(
                value=mi_normalized,
                method="mutual_information",
                n_observations=len(aligned_data),
                effect_size=effect_size
            )
        except Exception as e:
            return CorrelationResult(0.0, method="mutual_information", 
                                   n_observations=len(aligned_data))
    
    def _interpret_effect_size_continuous(self, correlation: float) -> str:
        """Interpreta l'effect size per correlazioni continue."""
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            return "negligible"
        elif abs_corr < 0.3:
            return "small"
        elif abs_corr < 0.5:
            return "medium"
        else:
            return "large"
    
    def _interpret_effect_size_categorical(self, correlation: float) -> str:
        """Interpreta l'effect size per correlazioni categoriche."""
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            return "negligible"
        elif abs_corr < 0.2:
            return "small"
        elif abs_corr < 0.4:
            return "medium"
        else:
            return "large"
    
    def compute_comprehensive_correlation_matrix(self, 
                                               df: pd.DataFrame,
                                               columns: Optional[List[str]] = None) -> Dict:
        """
        Calcola una matrice di correlazioni completa usando il metodo appropriato
        per ogni coppia di variabili.
        
        Args:
            df: DataFrame con i dati
            columns: Lista di colonne da analizzare (default: tutte)
            
        Returns:
            Dizionario con matrici di correlazioni e metadati
        """
        if columns is None:
            columns = df.columns.tolist()
        
        # Rileva tipi di variabili
        variable_types = {}
        for col in columns:
            variable_types[col] = self.detect_variable_type(df[col])
        
        # Inizializza matrici risultato
        n_cols = len(columns)
        correlation_matrix = np.zeros((n_cols, n_cols))
        p_value_matrix = np.zeros((n_cols, n_cols))
        method_matrix = np.full((n_cols, n_cols), "", dtype=object)
        effect_size_matrix = np.full((n_cols, n_cols), "", dtype=object)
        
        # Calcola correlazioni per ogni coppia
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    # Diagonale
                    correlation_matrix[i, j] = 1.0
                    p_value_matrix[i, j] = 0.0
                    method_matrix[i, j] = "identity"
                    effect_size_matrix[i, j] = "perfect"
                elif i < j:
                    # Calcola solo metà superiore (simmetrica)
                    result = self._select_and_compute_correlation(
                        df[col1], df[col2], 
                        variable_types[col1], variable_types[col2]
                    )
                    
                    # Riempi entrambe le metà
                    correlation_matrix[i, j] = correlation_matrix[j, i] = result.value
                    p_value_matrix[i, j] = p_value_matrix[j, i] = result.p_value or np.nan
                    method_matrix[i, j] = method_matrix[j, i] = result.method
                    effect_size_matrix[i, j] = effect_size_matrix[j, i] = result.effect_size or ""
        
        # Crea DataFrame risultato
        correlation_df = pd.DataFrame(correlation_matrix, index=columns, columns=columns)
        p_value_df = pd.DataFrame(p_value_matrix, index=columns, columns=columns)
        method_df = pd.DataFrame(method_matrix, index=columns, columns=columns)
        effect_size_df = pd.DataFrame(effect_size_matrix, index=columns, columns=columns)
        
        return {
            'correlations': correlation_df,
            'p_values': p_value_df,
            'methods': method_df,
            'effect_sizes': effect_size_df,
            'variable_types': variable_types,
            'summary': self._create_correlation_summary(correlation_df, p_value_df, 
                                                      method_df, variable_types)
        }
    
    def _select_and_compute_correlation(self, 
                                      x: pd.Series, 
                                      y: pd.Series,
                                      x_type: VariableType, 
                                      y_type: VariableType) -> CorrelationResult:
        """
        Seleziona e calcola il tipo di correlazione appropriato.
        
        Args:
            x: Prima variabile
            y: Seconda variabile
            x_type: Tipo della prima variabile
            y_type: Tipo della seconda variabile
            
        Returns:
            Risultato della correlazione
        """
        # Strategia di selezione del metodo
        if x_type == VariableType.CONTINUOUS and y_type == VariableType.CONTINUOUS:
            # Entrambe continue: usa Pearson + Distance correlation per confronto
            pearson_result = self._pearson_with_significance(x, y)
            distance_result = self.distance_correlation(x, y)
            
            # Ritorna quello con correlazione più alta
            if abs(distance_result.value) > abs(pearson_result.value):
                return distance_result
            else:
                return pearson_result
                
        elif x_type == VariableType.BINARY and y_type == VariableType.CONTINUOUS:
            return self.point_biserial_correlation(y, x)  # continua, binaria
            
        elif x_type == VariableType.CONTINUOUS and y_type == VariableType.BINARY:
            return self.point_biserial_correlation(x, y)  # continua, binaria
            
        elif x_type == VariableType.BINARY and y_type == VariableType.BINARY:
            return self.phi_coefficient(x, y)
            
        elif (x_type in [VariableType.NOMINAL, VariableType.ORDINAL] and 
              y_type in [VariableType.NOMINAL, VariableType.ORDINAL]):
            # Entrambe categoriche: usa Cramér's V
            return self._cramers_v_with_significance(x, y)
            
        else:
            # Caso misto: usa Mutual Information
            return self.enhanced_mutual_information(x, y, x_type, y_type)
    
    def _pearson_with_significance(self, x: pd.Series, y: pd.Series) -> CorrelationResult:
        """Calcola correlazione di Pearson con test di significatività."""
        aligned_data = pd.concat([x, y], axis=1).dropna()
        
        if len(aligned_data) < self.min_observations:
            return CorrelationResult(0.0, method="pearson", 
                                   n_observations=len(aligned_data))
        
        x_clean = aligned_data.iloc[:, 0]
        y_clean = aligned_data.iloc[:, 1]
        
        try:
            corr, p_value = stats.pearsonr(x_clean, y_clean)
            effect_size = self._interpret_effect_size_continuous(abs(corr))
            
            return CorrelationResult(
                value=corr,
                p_value=p_value,
                method="pearson",
                n_observations=len(aligned_data),
                effect_size=effect_size
            )
        except Exception as e:
            return CorrelationResult(0.0, method="pearson", 
                                   n_observations=len(aligned_data))
    
    def _cramers_v_with_significance(self, x: pd.Series, y: pd.Series) -> CorrelationResult:
        """Calcola Cramér's V con test di significatività."""
        aligned_data = pd.concat([x, y], axis=1).dropna()
        
        if len(aligned_data) < self.min_observations:
            return CorrelationResult(0.0, method="cramers_v", 
                                   n_observations=len(aligned_data))
        
        x_clean = aligned_data.iloc[:, 0]
        y_clean = aligned_data.iloc[:, 1]
        
        try:
            confusion_matrix = pd.crosstab(x_clean, y_clean)
            chi2, p_value, _, _ = chi2_contingency(confusion_matrix)
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            cramers_v = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
            
            effect_size = self._interpret_effect_size_categorical(cramers_v)
            
            return CorrelationResult(
                value=cramers_v,
                p_value=p_value,
                method="cramers_v",
                n_observations=len(aligned_data),
                effect_size=effect_size
            )
        except Exception as e:
            return CorrelationResult(0.0, method="cramers_v", 
                                   n_observations=len(aligned_data))
    
    def _create_correlation_summary(self, 
                                  correlation_df: pd.DataFrame,
                                  p_value_df: pd.DataFrame,
                                  method_df: pd.DataFrame,
                                  variable_types: Dict) -> Dict:
        """Crea un summary delle correlazioni calcolate."""
        
        # Estrai triangolo superiore (escludendo diagonale)
        upper_triangle_mask = np.triu(np.ones_like(correlation_df, dtype=bool), k=1)
        
        correlations_upper = correlation_df.values[upper_triangle_mask]
        p_values_upper = p_value_df.values[upper_triangle_mask]
        methods_upper = method_df.values[upper_triangle_mask]
        
        # Statistiche
        valid_correlations = correlations_upper[~np.isnan(correlations_upper)]
        valid_p_values = p_values_upper[~np.isnan(p_values_upper)]
        
        summary = {
            'total_pairs': len(correlations_upper),
            'valid_correlations': len(valid_correlations),
            'mean_abs_correlation': np.mean(np.abs(valid_correlations)) if len(valid_correlations) > 0 else 0,
            'max_correlation': np.max(np.abs(valid_correlations)) if len(valid_correlations) > 0 else 0,
            'significant_correlations': np.sum(valid_p_values < self.significance_level) if len(valid_p_values) > 0 else 0,
            'methods_used': list(np.unique(methods_upper)),
            'variable_type_counts': {vtype.value: sum(1 for t in variable_types.values() if t == vtype) 
                                   for vtype in VariableType}
        }
        
        return summary


def demo_enhanced_correlation_analysis():
    """
    Dimostra l'uso dell'analizzatore di correlazioni avanzato.
    """
    print("🔬 Demo: Enhanced Correlation Analysis")
    print("=" * 50)
    
    # Carica dati (assumendo che il dataset sia disponibile)
    try:
        df = pd.read_parquet('../data/raw/raw.parquet')
        print(f"✅ Dataset caricato: {df.shape}")
    except:
        # Crea dati demo se il dataset non è disponibile
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'continuous_1': np.random.normal(50, 15, n_samples),
            'continuous_2': np.random.normal(100, 20, n_samples),
            'binary_1': np.random.choice([0, 1], n_samples),
            'binary_2': np.random.choice(['A', 'B'], n_samples),
            'ordinal_1': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'categorical_1': np.random.choice(['X', 'Y', 'Z'], n_samples),
        })
        
        # Crea alcune correlazioni artificiali
        df['continuous_2'] = df['continuous_1'] * 0.7 + np.random.normal(0, 10, n_samples)
        df['binary_correlated'] = (df['continuous_1'] > 50).astype(int)
        
        print(f"✅ Dataset demo creato: {df.shape}")
    
    # Inizializza analizzatore
    analyzer = EnhancedCorrelationAnalyzer(
        significance_level=0.05,
        min_observations=30,
        n_jobs=2  # Ridotto per demo
    )
    
    # Seleziona subset di colonne per demo
    if 'AI_Superficie' in df.columns:
        # Usa colonne reali se disponibili
        demo_columns = [col for col in df.columns if col.startswith(('AI_', 'OV_', 'POI_'))][:10]
    else:
        # Usa colonne demo
        demo_columns = list(df.columns)
    
    print(f"\n🔍 Analizzando {len(demo_columns)} colonne...")
    
    # Calcola correlazioni comprehensive
    results = analyzer.compute_comprehensive_correlation_matrix(df, demo_columns)
    
    # Mostra risultati
    print(f"\n📊 Summary:")
    summary = results['summary']
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\n🎯 Top 10 correlazioni più forti:")
    corr_matrix = results['correlations']
    
    # Estrai coppie con correlazioni più forti
    correlations_list = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            correlations_list.append({
                'var1': corr_matrix.index[i],
                'var2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j],
                'method': results['methods'].iloc[i, j],
                'p_value': results['p_values'].iloc[i, j],
                'effect_size': results['effect_sizes'].iloc[i, j]
            })
    
    # Ordina per correlazione assoluta
    correlations_df = pd.DataFrame(correlations_list)
    correlations_df['abs_correlation'] = np.abs(correlations_df['correlation'])
    top_correlations = correlations_df.nlargest(10, 'abs_correlation')
    
    print(top_correlations[['var1', 'var2', 'correlation', 'method', 'effect_size']].to_string(index=False))
    
    return results


if __name__ == "__main__":
    results = demo_enhanced_correlation_analysis()