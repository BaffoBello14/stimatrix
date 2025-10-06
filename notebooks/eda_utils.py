#!/usr/bin/env python3
"""
EDA Utilities - Funzioni comuni per Exploratory Data Analysis
Modulo condiviso tra i notebook EDA per evitare duplicazione di codice.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_plotting_style():
    """Configura lo stile dei plot per grafici leggibili e professionali"""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    sns.set_palette("husl")
    logger.info("✅ Stile plotting configurato")


def setup_output_dir(output_dir_name: str = 'eda_outputs') -> Path:
    """
    Crea la directory di output per salvare i risultati
    
    Args:
        output_dir_name: Nome della directory di output
        
    Returns:
        Path object della directory creata
    """
    notebook_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    output_dir = notebook_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Directory output: {output_dir}")
    return output_dir


def load_config_and_data(
    config_path: str = '../config/config.yaml',
    data_path: str = '../data/raw/raw.parquet'
) -> Tuple[Dict, pd.DataFrame]:
    """
    Carica configurazione e dataset con gestione robusta degli errori
    
    Args:
        config_path: Path al file di configurazione YAML
        data_path: Path al file parquet dei dati
        
    Returns:
        Tuple (config_dict, dataframe)
        
    Raises:
        FileNotFoundError: Se i file non esistono
        Exception: Per altri errori di caricamento
    """
    logger.info("🔧 Caricamento configurazione e dati...")
    
    # Carica configurazione
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Config caricata da: {config_path}")
    except FileNotFoundError:
        logger.error(f"❌ File di configurazione non trovato: {config_path}")
        raise
    except Exception as e:
        logger.error(f"❌ Errore nel caricamento config: {e}")
        raise
    
    # Carica dataset
    try:
        df = pd.read_parquet(data_path)
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"✅ Dataset caricato da: {data_path}")
        logger.info(f"📊 Dimensioni: {df.shape[0]:,} righe × {df.shape[1]} colonne")
        logger.info(f"💾 Memoria utilizzata: {memory_mb:.2f} MB")
        return config, df
    except FileNotFoundError:
        logger.error(f"❌ File dati non trovato: {data_path}")
        raise
    except Exception as e:
        logger.error(f"❌ Errore nel caricamento dati: {e}")
        raise


def get_target_column(config: Dict) -> str:
    """
    Estrae il nome della colonna target dalla configurazione
    
    Args:
        config: Dictionary di configurazione
        
    Returns:
        Nome della colonna target
    """
    target_col = config['target']['column_candidates'][0]
    logger.info(f"🎯 Target identificato: {target_col}")
    return target_col


def analyze_missingness(df: pd.DataFrame, output_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Analizza la missingness nel dataset
    
    Args:
        df: DataFrame da analizzare
        output_dir: Directory dove salvare i risultati (opzionale)
        
    Returns:
        DataFrame con statistiche di missingness
    """
    logger.info("\n" + "="*60)
    logger.info("🔍 ANALISI MISSINGNESS")
    logger.info("="*60)

    missing_stats = pd.DataFrame({
        'Totale_Null': df.isnull().sum(),
        'Percentuale_Null': (df.isnull().mean() * 100).round(2),
        'Tipo': df.dtypes
    })

    high_missing = missing_stats[
        missing_stats['Percentuale_Null'] > 50
    ].sort_values('Percentuale_Null', ascending=False)

    logger.info(f"\n⚠️  Colonne con >50% valori mancanti: {len(high_missing)}")
    if len(high_missing) > 0:
        logger.info(f"\n{high_missing.head(10)}")
    else:
        logger.info("  ✅ Nessuna colonna con >50% valori mancanti")

    # Salva risultati
    if output_dir:
        output_file = output_dir / 'missingness_analysis.csv'
        missing_stats.to_csv(output_file)
        logger.info(f"\n💾 Risultati salvati in {output_file}")
    
    return missing_stats


def analyze_target_distribution(
    df: pd.DataFrame, 
    target_col: str, 
    output_dir: Optional[Path] = None
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Analizza la distribuzione del target
    
    Args:
        df: DataFrame contenente i dati
        target_col: Nome della colonna target
        output_dir: Directory dove salvare i risultati (opzionale)
        
    Returns:
        Tuple (statistiche_descrittive, price_bands)
    """
    logger.info("\n" + "="*60)
    logger.info(f"🎯 ANALISI DISTRIBUZIONE TARGET: {target_col}")
    logger.info("="*60)

    if target_col not in df.columns:
        logger.error(f"❌ Target '{target_col}' non trovato nelle colonne!")
        return None, None

    logger.info(f"\n📊 Info target:")
    logger.info(f"  Tipo: {df[target_col].dtype}")
    logger.info(f"  Valori non-nulli: {df[target_col].count():,}")
    logger.info(f"  Valori mancanti: {df[target_col].isnull().sum():,} ({df[target_col].isnull().mean()*100:.2f}%)")

    target_data = df[target_col].dropna()
    if len(target_data) == 0:
        logger.warning("❌ Nessun dato valido per il target")
        return None, None
        
    stats = target_data.describe()

    logger.info(f"\n📈 Statistiche descrittive:")
    logger.info(f"  Conteggio: {stats['count']:,.0f}")
    logger.info(f"  Media: €{stats['mean']:,.2f}")
    logger.info(f"  Mediana: €{stats['50%']:,.2f}")
    logger.info(f"  Std Dev: €{stats['std']:,.2f}")
    logger.info(f"  Min: €{stats['min']:,.2f}")
    logger.info(f"  Max: €{stats['max']:,.2f}")

    # Calcola fasce di prezzo a quantili
    quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    price_bands = target_data.quantile(quantiles)

    logger.info(f"\n🏷️  Fasce di prezzo (quantili):")
    for i in range(len(quantiles)-1):
        q_low = quantiles[i]
        q_high = quantiles[i+1]
        price_low = price_bands.iloc[i]
        price_high = price_bands.iloc[i+1]
        logger.info(f"  Q{int(q_low*100):02d}-Q{int(q_high*100):02d}: €{price_low:,.0f} - €{price_high:,.0f}")

    # Salva statistiche target
    if output_dir:
        stats_df = pd.DataFrame(stats).reset_index()
        stats_df.columns = ['Statistic', 'Value']
        output_file = output_dir / 'target_statistics.csv'
        stats_df.to_csv(output_file, index=False)
        logger.info(f"\n💾 Statistiche target salvate in {output_file}")

    return stats, price_bands


def analyze_correlations(
    df: pd.DataFrame, 
    target_col: str, 
    output_dir: Optional[Path] = None,
    threshold: float = 0.1
) -> pd.DataFrame:
    """
    Calcola correlazioni con il target
    
    Args:
        df: DataFrame contenente i dati
        target_col: Nome della colonna target
        output_dir: Directory dove salvare i risultati (opzionale)
        threshold: Soglia per correlazioni significative
        
    Returns:
        DataFrame con correlazioni ordinate
    """
    logger.info("\n" + "="*60)
    logger.info(f"📊 ANALISI CORRELAZIONI CON {target_col}")
    logger.info("="*60)

    # Identifica colonne numeriche escludendo il target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    logger.info(f"\n🔢 Colonne numeriche identificate: {len(numeric_cols)}")

    # Identifica colonne costanti
    constant_cols = [col for col in numeric_cols if df[col].nunique() <= 1]
    logger.info(f"⚠️  Colonne costanti (rimosse): {len(constant_cols)}")

    # Rimuovi colonne costanti
    analysis_cols = [c for c in numeric_cols if c not in constant_cols]
    
    corr_data = df[analysis_cols + [target_col]].dropna()
    if len(corr_data) == 0:
        logger.warning("❌ Nessun dato valido per correlazioni")
        return pd.DataFrame()

    correlations = corr_data.corr(numeric_only=True)[target_col].drop(target_col)

    corr_df = pd.DataFrame({
        'Colonna': correlations.index,
        'Correlazione': correlations.values,
        'Correlazione_Assoluta': np.abs(correlations.values)
    }).round(4).sort_values('Correlazione_Assoluta', ascending=False)

    significant_corr = corr_df[corr_df['Correlazione_Assoluta'] >= threshold]
    logger.info(f"\n📈 Correlazioni significative (|r| >= {threshold}): {len(significant_corr)}")
    if len(significant_corr) > 0:
        logger.info(f"\n🔝 Top 10 correlazioni:\n{significant_corr.head(10)}")

    logger.info(f"\n📊 Statistiche correlazioni:")
    logger.info(f"  Max correlazione positiva: {corr_df['Correlazione'].max():.4f}")
    logger.info(f"  Max correlazione negativa: {corr_df['Correlazione'].min():.4f}")
    logger.info(f"  Media correlazione assoluta: {corr_df['Correlazione_Assoluta'].mean():.4f}")

    # Salva risultati
    if output_dir:
        output_file = output_dir / 'correlations_with_target.csv'
        corr_df.to_csv(output_file, index=False)
        logger.info(f"\n💾 Correlazioni salvate in {output_file}")

    return corr_df


def save_plot(
    filename: str, 
    output_dir: Path, 
    dpi: int = 100, 
    format: str = 'png',
    bbox_inches: str = 'tight'
):
    """
    Salva un plot ottimizzato senza mostrarlo
    
    Args:
        filename: Nome del file (con o senza estensione)
        output_dir: Directory dove salvare
        dpi: Risoluzione (default 100 per dimensioni ragionevoli)
        format: Formato file (png, jpg, pdf)
        bbox_inches: Gestione dei bordi
    """
    if not filename.endswith(f'.{format}'):
        filename = f'{filename}.{format}'
    
    filepath = output_dir / filename
    
    # Salva il plot - matplotlib non supporta 'optimize' come parametro
    plt.savefig(
        filepath,
        dpi=dpi,
        format=format,
        bbox_inches=bbox_inches
    )
    plt.close()  # Chiude la figura per liberare memoria
    
    file_size = filepath.stat().st_size / 1024  # KB
    logger.info(f"💾 Plot salvato: {filename} ({file_size:.1f} KB)")


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    output_dir: Optional[Path] = None,
    save_filename: Optional[str] = None
):
    """
    Crea un plot della distribuzione del target
    
    Args:
        df: DataFrame contenente i dati
        target_col: Nome della colonna target
        output_dir: Directory dove salvare (opzionale)
        save_filename: Nome del file di output (opzionale)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    target_data = df[target_col].dropna()
    
    # Histogram
    axes[0].hist(target_data, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Frequenza')
    axes[0].set_title(f'Distribuzione {target_col}')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(target_data, vert=True)
    axes[1].set_ylabel(target_col)
    axes[1].set_title(f'Box Plot {target_col}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir and save_filename:
        save_plot(save_filename, output_dir)
    elif output_dir:
        save_plot(f'target_distribution_{target_col}', output_dir)


def create_correlation_heatmap(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 20,
    output_dir: Optional[Path] = None,
    save_filename: Optional[str] = None
):
    """
    Crea una heatmap delle correlazioni delle top N feature con il target
    
    Args:
        df: DataFrame contenente i dati
        target_col: Nome della colonna target
        top_n: Numero di top feature da visualizzare
        output_dir: Directory dove salvare (opzionale)
        save_filename: Nome del file di output (opzionale)
    """
    # Calcola correlazioni
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[target_col].drop(target_col)
    
    # Prendi top N per valore assoluto
    top_features = correlations.abs().nlargest(top_n).index.tolist()
    
    # Crea subset per heatmap
    subset = df[top_features + [target_col]].corr()
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(subset, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(f'Heatmap Correlazioni - Top {top_n} Feature vs {target_col}')
    plt.tight_layout()
    
    if output_dir and save_filename:
        save_plot(save_filename, output_dir, dpi=100)
    elif output_dir:
        save_plot(f'correlation_heatmap_top{top_n}', output_dir, dpi=100)


def get_dataset_summary(df: pd.DataFrame) -> Dict:
    """
    Restituisce un summary completo del dataset
    
    Args:
        df: DataFrame da analizzare
        
    Returns:
        Dictionary con metriche del dataset
    """
    summary = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'n_numeric': len(df.select_dtypes(include=[np.number]).columns),
        'n_categorical': len(df.select_dtypes(include=['object', 'category']).columns),
        'n_datetime': len(df.select_dtypes(include=['datetime64']).columns),
        'total_missing': df.isnull().sum().sum(),
        'missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
    }
    return summary


def print_dataset_summary(df: pd.DataFrame):
    """
    Stampa un summary leggibile del dataset
    
    Args:
        df: DataFrame da analizzare
    """
    summary = get_dataset_summary(df)
    
    logger.info("\n" + "="*60)
    logger.info("📋 DATASET SUMMARY")
    logger.info("="*60)
    logger.info(f"📊 Dimensioni: {summary['n_rows']:,} righe × {summary['n_columns']} colonne")
    logger.info(f"💾 Memoria: {summary['memory_mb']:.2f} MB")
    logger.info(f"🔢 Colonne numeriche: {summary['n_numeric']}")
    logger.info(f"📝 Colonne categoriche: {summary['n_categorical']}")
    logger.info(f"📅 Colonne datetime: {summary['n_datetime']}")
    logger.info(f"❓ Valori mancanti: {summary['total_missing']:,} ({summary['missing_pct']:.2f}%)")
    logger.info("="*60)


# ============================================================================
# FUNZIONI AVANZATE PER EDA COMPREHENSIVE
# ============================================================================

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Calcola Cramér's V per associazione tra variabili categoriche
    
    Args:
        x, y: Serie categoriche da confrontare
        
    Returns:
        Valore di Cramér's V (0-1)
    """
    from scipy.stats import chi2_contingency
    
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    """
    Calcola correlation ratio (eta) per variabile categorica vs numerica
    
    Args:
        categories: Serie categorica
        measurements: Serie numerica
        
    Returns:
        Correlation ratio (0-1)
    """
    from scipy.stats import f_oneway
    
    categories = categories.astype('category')
    measurements = measurements.astype(float)
    
    # Raggruppa measurements per categoria
    groups = [measurements[categories == cat].dropna().values 
              for cat in categories.cat.categories]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return 0.0
    
    # Calcola varianza totale e within-group
    all_data = measurements.dropna()
    overall_mean = all_data.mean()
    
    # Between-group variance
    ssb = sum(len(g) * (np.mean(g) - overall_mean)**2 for g in groups)
    
    # Total variance
    sst = sum((all_data - overall_mean)**2)
    
    if sst == 0:
        return 0.0
    
    return np.sqrt(ssb / sst)


if __name__ == "__main__":
    # Test delle funzioni
    logger.info("🧪 Test del modulo eda_utils")
    
    try:
        config, df = load_config_and_data()
        logger.info("✅ Caricamento dati: OK")
        
        target_col = get_target_column(config)
        logger.info("✅ Identificazione target: OK")
        
        print_dataset_summary(df)
        logger.info("✅ Summary dataset: OK")
        
        logger.info("\n✅ Tutti i test passati!")
        
    except Exception as e:
        logger.error(f"❌ Test fallito: {e}")
