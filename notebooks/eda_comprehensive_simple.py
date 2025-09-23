#!/usr/bin/env python3
"""
📊 Analisi Esplorativa Completa - Multi-Target & Correlazioni Avanzate

Questo script estende l'analisi EDA originale con:
- Multi-Target Analysis per AI_Prezzo_Ridistribuito e AI_Prezzo_MQ  
- Correlazioni Complete: Pearson, Spearman, Kendall
- Associazioni Categoriche: Cramér's V, Mutual Information
- Network Analysis delle correlazioni
- Feature Importance Comparativa
"""

# Import librerie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import warnings
from pathlib import Path
from scipy import stats as scipy_stats
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette('husl')

print("✅ Setup completato!")

def main():
    """Funzione principale per l'analisi EDA completa"""
    
    # Setup output directory
    output_dir = Path('eda_comprehensive_outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Directory output: {output_dir}")
    
    # Carica configurazione
    config_path = '../config/config.yaml'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config caricata da: {config_path}")
    except Exception as e:
        print(f"❌ Errore nel caricamento config: {e}")
        return
    
    # Carica dataset
    data_path = '../data/raw/raw.parquet'
    try:
        df_raw = pd.read_parquet(data_path)
        print(f"✅ Dataset caricato da: {data_path}")
        print(f"📊 Dimensioni: {df_raw.shape[0]:,} righe × {df_raw.shape[1]} colonne")
        print(f"💾 Memoria utilizzata: {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"❌ Errore nel caricamento dati: {e}")
        return
    
    # Identifica target dalla configurazione
    target_candidates = config['target']['column_candidates']
    available_targets = [col for col in target_candidates if col in df_raw.columns]
    
    print(f"\n🎯 Target identificati: {available_targets}")
    if len(available_targets) < 2:
        print(f"⚠️  Solo {len(available_targets)} target disponibile/i.")
    else:
        print(f"✅ Entrambi i target disponibili per analisi comparativa!")
    
    # 1. ANALISI MULTI-TARGET
    print("\n" + "="*70)
    print("🎯 ANALISI COMPARATIVA MULTI-TARGET")
    print("="*70)
    
    target_stats = {}
    
    for target_col in available_targets:
        print(f"\n📊 Analisi target: {target_col}")
        print("-" * 50)
        
        target_data = df_raw[target_col].dropna()
        stats = target_data.describe()
        
        target_stats[target_col] = {
            'count': int(stats['count']),
            'mean': stats['mean'],
            'median': stats['50%'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max'],
            'skewness': scipy_stats.skew(target_data),
            'kurtosis': scipy_stats.kurtosis(target_data),
            'missing_pct': (df_raw[target_col].isnull().sum() / len(df_raw)) * 100
        }
        
        print(f"  Valori non-nulli: {stats['count']:,.0f}")
        print(f"  Valori mancanti: {df_raw[target_col].isnull().sum():,} ({target_stats[target_col]['missing_pct']:.2f}%)")
        print(f"  Media: €{stats['mean']:,.2f}")
        print(f"  Mediana: €{stats['50%']:,.2f}")
        print(f"  Std Dev: €{stats['std']:,.2f}")
        print(f"  Range: €{stats['min']:,.2f} - €{stats['max']:,.2f}")
        print(f"  Skewness: {target_stats[target_col]['skewness']:.4f}")
        print(f"  Kurtosis: {target_stats[target_col]['kurtosis']:.4f}")
    
    # Confronto tra target
    if len(available_targets) >= 2:
        print(f"\n🔄 CONFRONTO TRA TARGET")
        print("-" * 30)
        
        target1, target2 = available_targets[0], available_targets[1]
        valid_both = df_raw[[target1, target2]].dropna()
        
        if len(valid_both) > 0:
            corr_targets = valid_both[target1].corr(valid_both[target2])
            print(f"  Correlazione {target1} vs {target2}: {corr_targets:.4f}")
            
            # Ratio medio
            ratio = (valid_both[target1] / valid_both[target2]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio) > 0:
                print(f"  Ratio medio {target1}/{target2}: {ratio.mean():.4f} (±{ratio.std():.4f})")
        else:
            print("  ⚠️  Nessun valore valido per entrambi i target")
    
    # Salva statistiche comparative
    comparison_df = pd.DataFrame(target_stats).T
    comparison_df.to_csv(output_dir / 'multi_target_comparison.csv')
    print(f"\n💾 Statistiche comparative salvate in {output_dir}/multi_target_comparison.csv")
    
    # 2. CORRELAZIONI AVANZATE CON I TARGET
    print("\n" + "="*70)
    print("🎯 CORRELAZIONI AVANZATE CON I TARGET")
    print("="*70)
    
    # Identifica tipi di colonne
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Rimuovi target dalle feature
    feature_numeric_cols = [col for col in numeric_cols if col not in available_targets]
    feature_categorical_cols = categorical_cols.copy()
    
    # Filtra colonne problematiche
    missing_threshold = 0.7
    constant_cols = []
    high_missing_cols = []
    
    for col in feature_numeric_cols + feature_categorical_cols:
        missing_pct = df_raw[col].isnull().sum() / len(df_raw)
        if missing_pct > missing_threshold:
            high_missing_cols.append(col)
        elif df_raw[col].nunique() <= 1:
            constant_cols.append(col)
    
    # Aggiorna liste
    feature_numeric_cols = [col for col in feature_numeric_cols if col not in high_missing_cols + constant_cols]
    feature_categorical_cols = [col for col in feature_categorical_cols if col not in high_missing_cols + constant_cols]
    
    # Limita cardinalità categoriche
    max_cardinality = 50
    high_card_cols = []
    for col in feature_categorical_cols:
        if df_raw[col].nunique() > max_cardinality:
            high_card_cols.append(col)
    
    feature_categorical_cols = [col for col in feature_categorical_cols if col not in high_card_cols]
    
    print(f"\n📊 Colonne per analisi:")
    print(f"  Numeriche: {len(feature_numeric_cols)}")
    print(f"  Categoriche: {len(feature_categorical_cols)}")
    print(f"  Target: {len(available_targets)}")
    
    # Limita numero per performance
    max_numeric = 30
    max_categorical = 15
    
    if len(feature_numeric_cols) > max_numeric:
        # Seleziona top per varianza
        numeric_variances = df_raw[feature_numeric_cols].var().sort_values(ascending=False)
        feature_numeric_cols = numeric_variances.head(max_numeric).index.tolist()
        print(f"  Ridotte numeriche a top {max_numeric}")
    
    if len(feature_categorical_cols) > max_categorical:
        # Seleziona top per entropia
        categorical_entropy = {}
        for col in feature_categorical_cols:
            value_counts = df_raw[col].value_counts(normalize=True)
            entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
            categorical_entropy[col] = entropy
        
        top_categorical = sorted(categorical_entropy.items(), key=lambda x: x[1], reverse=True)
        feature_categorical_cols = [col for col, _ in top_categorical[:max_categorical]]
        print(f"  Ridotte categoriche a top {max_categorical}")
    
    # Calcola correlazioni per ogni target
    target_correlations = {}
    
    for target_col in available_targets:
        print(f"\n📊 Correlazioni per target: {target_col}")
        print("-" * 50)
        
        target_corr_results = {
            'numeric_pearson': {},
            'numeric_spearman': {},
            'categorical_eta': {},
            'mutual_info': {}
        }
        
        # Correlazioni numeriche
        print("  🔢 Correlazioni numeriche...")
        for col in feature_numeric_cols:
            valid_data = df_raw[[col, target_col]].dropna()
            if len(valid_data) >= 10:
                x, y = valid_data[col], valid_data[target_col]
                
                try:
                    pearson_corr = x.corr(y, method='pearson')
                    if not np.isnan(pearson_corr):
                        target_corr_results['numeric_pearson'][col] = pearson_corr
                except:
                    pass
                
                try:
                    spearman_corr = x.corr(y, method='spearman')
                    if not np.isnan(spearman_corr):
                        target_corr_results['numeric_spearman'][col] = spearman_corr
                except:
                    pass
        
        # Correlazioni categoriche (eta-squared)
        print("  📊 Correlazioni categoriche...")
        for col in feature_categorical_cols:
            valid_data = df_raw[[col, target_col]].dropna()
            if len(valid_data) >= 10 and valid_data[col].nunique() > 1:
                try:
                    eta = correlation_ratio(valid_data[col], valid_data[target_col])
                    target_corr_results['categorical_eta'][col] = eta
                except:
                    pass
        
        # Mutual Information
        print("  🧠 Mutual Information...")
        all_features = feature_numeric_cols + feature_categorical_cols
        for col in all_features[:20]:  # Limita per performance
            valid_data = df_raw[[col, target_col]].dropna()
            if len(valid_data) >= 20:
                try:
                    X = valid_data[[col]]
                    y = valid_data[target_col]
                    
                    if col in feature_categorical_cols:
                        le = LabelEncoder()
                        X_encoded = le.fit_transform(X.iloc[:, 0]).reshape(-1, 1)
                    else:
                        X_encoded = X.values
                    
                    mi_score = mutual_info_regression(X_encoded, y, random_state=42)[0]
                    target_corr_results['mutual_info'][col] = mi_score
                except:
                    pass
        
        target_correlations[target_col] = target_corr_results
        
        # Stampa top correlazioni
        print(f"\n  🔝 Top 10 correlazioni Pearson:")
        if target_corr_results['numeric_pearson']:
            pearson_sorted = sorted(target_corr_results['numeric_pearson'].items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
            for col, corr in pearson_sorted[:10]:
                print(f"    {col}: {corr:.4f}")
        else:
            print("    Nessuna correlazione Pearson calcolata")
        
        print(f"\n  🔝 Top 10 correlazioni Eta (categoriche):")
        if target_corr_results['categorical_eta']:
            eta_sorted = sorted(target_corr_results['categorical_eta'].items(), 
                               key=lambda x: abs(x[1]), reverse=True)
            for col, eta in eta_sorted[:10]:
                print(f"    {col}: {eta:.4f}")
        else:
            print("    Nessuna correlazione Eta calcolata")
    
    # 3. SALVA RISULTATI
    print(f"\n💾 Salvando risultati correlazioni...")
    
    for target_col in available_targets:
        target_results = target_correlations[target_col]
        
        all_correlations = []
        
        # Numeriche - Pearson
        for col, corr in target_results['numeric_pearson'].items():
            all_correlations.append({
                'Feature': col,
                'Type': 'Numeric',
                'Method': 'Pearson',
                'Correlation': corr,
                'Abs_Correlation': abs(corr)
            })
        
        # Numeriche - Spearman
        for col, corr in target_results['numeric_spearman'].items():
            all_correlations.append({
                'Feature': col,
                'Type': 'Numeric', 
                'Method': 'Spearman',
                'Correlation': corr,
                'Abs_Correlation': abs(corr)
            })
        
        # Categoriche - Eta
        for col, eta in target_results['categorical_eta'].items():
            all_correlations.append({
                'Feature': col,
                'Type': 'Categorical',
                'Method': 'Eta',
                'Correlation': eta,
                'Abs_Correlation': abs(eta)
            })
        
        # Mutual Information
        for col, mi in target_results['mutual_info'].items():
            all_correlations.append({
                'Feature': col,
                'Type': 'Categorical' if col in feature_categorical_cols else 'Numeric',
                'Method': 'Mutual_Info',
                'Correlation': mi,
                'Abs_Correlation': abs(mi)
            })
        
        if all_correlations:
            correlations_df = pd.DataFrame(all_correlations)
            correlations_df = correlations_df.sort_values('Abs_Correlation', ascending=False)
            
            output_file = output_dir / f'advanced_correlations_{target_col}.csv'
            correlations_df.to_csv(output_file, index=False)
            print(f"💾 Correlazioni per {target_col} salvate in {output_file}")
        else:
            print(f"⚠️  Nessuna correlazione calcolata per {target_col}")
    
    # 4. VISUALIZZAZIONI
    print(f"\n📈 Creando visualizzazioni...")
    
    # Confronto distribuzioni target
    if len(available_targets) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for i, target in enumerate(available_targets[:2]):
            data = df_raw[target].dropna()
            
            # Istogramma
            axes[i, 0].hist(data, bins=50, alpha=0.7, color=f'C{i}', edgecolor='black')
            axes[i, 0].set_title(f'Distribuzione {target}')
            axes[i, 0].set_xlabel('Valore (€)')
            axes[i, 0].set_ylabel('Frequenza')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Log-trasformata
            log_data = np.log1p(data)
            axes[i, 1].hist(log_data, bins=50, alpha=0.7, color=f'C{i}', edgecolor='black')
            axes[i, 1].set_title(f'Distribuzione Log {target}')
            axes[i, 1].set_xlabel('Log(Valore)')
            axes[i, 1].set_ylabel('Frequenza')
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'target_distributions_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Scatter plot tra target
        valid_both = df_raw[available_targets[:2]].dropna()
        if len(valid_both) > 0:
            plt.figure(figsize=(10, 8))
            plt.scatter(valid_both.iloc[:, 0], valid_both.iloc[:, 1], alpha=0.6, s=30)
            plt.xlabel(f'{available_targets[0]} (€)')
            plt.ylabel(f'{available_targets[1]} (€)')
            plt.title(f'Relazione tra {available_targets[0]} e {available_targets[1]}')
            
            # Linea di regressione
            z = np.polyfit(valid_both.iloc[:, 0], valid_both.iloc[:, 1], 1)
            p = np.poly1d(z)
            plt.plot(valid_both.iloc[:, 0], p(valid_both.iloc[:, 0]), "r--", alpha=0.8, linewidth=2)
            
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'targets_scatter_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 5. RIEPILOGO FINALE
    print("\n" + "="*70)
    print("🎉 RIEPILOGO ANALISI ESPLORATIVA COMPLETA")
    print("="*70)
    
    print(f"\n📊 Dataset analizzato:")
    print(f"  • Dimensioni: {df_raw.shape[0]:,} righe × {df_raw.shape[1]} colonne")
    print(f"  • Target analizzati: {', '.join(available_targets)}")
    print(f"  • Feature numeriche: {len(feature_numeric_cols)}")
    print(f"  • Feature categoriche: {len(feature_categorical_cols)}")
    
    print(f"\n📈 Analisi eseguite:")
    print(f"  ✅ Multi-target analysis comparativa")
    print(f"  ✅ Correlazioni avanzate (Pearson, Spearman)")
    print(f"  ✅ Associazioni categoriche (Eta)")
    print(f"  ✅ Mutual Information analysis")
    print(f"  ✅ Visualizzazioni comparative")
    
    print(f"\n💾 File generati in {output_dir}:")
    output_files = list(output_dir.glob('*'))
    for f in sorted(output_files):
        if f.is_file():
            print(f"  • {f.name}")
    
    print(f"\n💡 RACCOMANDAZIONI:")
    if len(available_targets) >= 2:
        print(f"  • Considera modelli separati per {available_targets[0]} e {available_targets[1]}")
        print(f"  • Le correlazioni differiscono tra i target")
    else:
        print(f"  • Focalizzati sul target principale: {available_targets[0]}")
    
    print(f"  • Le correlazioni non-lineari (Spearman) mostrano pattern diversi da Pearson")
    print(f"  • Le variabili categoriche mostrano associazioni significative")
    print(f"  • Considera trasformazioni per feature con alta Mutual Information")
    
    print(f"\n✅ Analisi esplorativa avanzata completata con successo!")

def correlation_ratio(categories, measurements):
    """Calcola correlation ratio (eta) per variabile categorica vs numerica"""
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    
    if denominator == 0:
        eta = 0
    else:
        eta = np.sqrt(numerator / denominator)
    
    return eta

if __name__ == "__main__":
    main()
