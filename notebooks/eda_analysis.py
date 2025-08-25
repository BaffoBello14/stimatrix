#!/usr/bin/env python3
"""
Exploratory Data Analysis - Raw Dataset (FIXED VERSION)
Questo script esegue un'analisi esplorativa completa del dataset raw per il progetto StiMatrix.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import re
from pathlib import Path
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# Configurazione matplotlib per grafici più leggibili
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def setup_output_dir():
    """Crea la directory di output per salvare i risultati"""
    output_dir = Path('eda_outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_config_and_data():
    """Carica configurazione e dataset con gestione robusta degli errori"""
    print("🔧 Caricamento configurazione e dati...")
    
    # Carica configurazione
    config_path = '../config/config.yaml'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config caricata da: {config_path}")
    except Exception as e:
        print(f"❌ Errore nel caricamento config: {e}")
        raise
    
    # Carica dataset
    data_path = '../data/raw/raw.parquet'
    try:
        df = pd.read_parquet(data_path)
        print(f"✅ Dataset caricato da: {data_path}")
        print(f"📊 Dimensioni: {df.shape[0]:,} righe × {df.shape[1]} colonne")
        print(f"💾 Memoria utilizzata: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return config, df
    except Exception as e:
        print(f"❌ Errore nel caricamento dati: {e}")
        raise

def dataset_overview(df, config):
    """Fornisce overview completo del dataset"""
    print("\n" + "="*60)
    print("📋 OVERVIEW DATASET")
    print("="*60)
    
    print(f"\n📊 Dimensioni: {df.shape[0]:,} righe × {df.shape[1]} colonne")
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"💾 Memoria utilizzata: {memory_mb:.2f} MB")
    
    print(f"\n📋 Tipi di dati:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} colonne")
    
    target_col = config['target']['column_candidates'][0]
    print(f"\n🎯 Target identificato: {target_col}")
    
    # Prime righe del dataset
    print(f"\n🔍 Prime 5 righe del dataset:")
    print(df.head())
    
    return dtype_counts

def analyze_missingness(df, output_dir):
    """Analizza la missingness nel dataset"""
    print("\n" + "="*60)
    print("🔍 ANALISI MISSINGNESS")
    print("="*60)

    missing_stats = pd.DataFrame({
        'Totale_Null': df.isnull().sum(),
        'Percentuale_Null': (df.isnull().mean() * 100).round(2),
        'Tipo': df.dtypes
    })

    high_missing = missing_stats[missing_stats['Percentuale_Null'] > 50].sort_values('Percentuale_Null', ascending=False)

    print(f"\n⚠️  Colonne con >50% valori mancanti ({len(high_missing)} colonne):")
    if len(high_missing) > 0:
        print(high_missing.head(10))
    else:
        print("  Nessuna colonna con >50% valori mancanti")

    # Salva risultati
    missing_stats.to_csv(output_dir / 'missingness_analysis.csv')
    print(f"\n💾 Risultati salvati in {output_dir}/missingness_analysis.csv")
    
    return missing_stats

def analyze_target_distribution(df, target_col, output_dir):
    """Analizza la distribuzione del target"""
    print("\n" + "="*60)
    print(f"🎯 ANALISI DISTRIBUZIONE TARGET: {target_col}")
    print("="*60)

    if target_col not in df.columns:
        print(f"❌ Target '{target_col}' non trovato nelle colonne!")
        return None, None

    print(f"\n📊 Info target:")
    print(f"  Tipo: {df[target_col].dtype}")
    print(f"  Valori non-nulli: {df[target_col].count():,}")
    print(f"  Valori mancanti: {df[target_col].isnull().sum():,} ({df[target_col].isnull().mean()*100:.2f}%)")

    target_data = df[target_col].dropna()
    if len(target_data) == 0:
        print("❌ Nessun dato valido per il target")
        return None, None
        
    stats = target_data.describe()

    print(f"\n📈 Statistiche descrittive:")
    print(f"  Conteggio: {stats['count']:,.0f}")
    print(f"  Media: €{stats['mean']:,.2f}")
    print(f"  Mediana: €{stats['50%']:,.2f}")
    print(f"  Std Dev: €{stats['std']:,.2f}")
    print(f"  Min: €{stats['min']:,.2f}")
    print(f"  Max: €{stats['max']:,.2f}")

    # Calcola fasce di prezzo a quantili
    quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    price_bands = target_data.quantile(quantiles)

    print(f"\n🏷️  Fasce di prezzo (quantili):")
    for i in range(len(quantiles)-1):
        q_low = quantiles[i]
        q_high = quantiles[i+1]
        price_low = price_bands.iloc[i]
        price_high = price_bands.iloc[i+1]
        print(f"  Q{int(q_low*100):02d}-Q{int(q_high*100):02d}: €{price_low:,.0f} - €{price_high:,.0f}")

    # Salva statistiche target
    stats_df = pd.DataFrame(stats).reset_index()
    stats_df.columns = ['Statistic', 'Value']
    stats_df.to_csv(output_dir / 'target_statistics.csv', index=False)
    print(f"\n💾 Statistiche target salvate in {output_dir}/target_statistics.csv")

    return stats, price_bands

def analyze_correlations(df, target_col, output_dir):
    """Calcola correlazioni con il target"""
    print("\n" + "="*60)
    print(f"📊 ANALISI CORRELAZIONI CON {target_col}")
    print("="*60)

    # Identifica colonne numeriche escludendo il target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    print(f"\n🔢 Colonne numeriche identificate: {len(numeric_cols)}")

    # Identifica colonne costanti
    constant_cols = [col for col in numeric_cols if df[col].nunique() <= 1]
    print(f"⚠️  Colonne costanti identificate: {len(constant_cols)}")
    if constant_cols:
        print("  Colonne:", constant_cols[:10])

    # Rimuovi colonne costanti
    analysis_cols = [c for c in numeric_cols if c not in constant_cols]
    
    corr_data = df[analysis_cols + [target_col]].dropna()
    if len(corr_data) == 0:
        print("❌ Nessun dato valido per correlazioni")
        return pd.DataFrame()

    correlations = corr_data.corr(numeric_only=True)[target_col].drop(target_col)

    corr_df = pd.DataFrame({
        'Colonna': correlations.index,
        'Correlazione': correlations.values,
        'Correlazione_Assoluta': np.abs(correlations.values)
    }).round(4).sort_values('Correlazione_Assoluta', ascending=False)

    threshold = 0.1
    significant_corr = corr_df[corr_df['Correlazione_Assoluta'] >= threshold]
    print(f"\n📈 Correlazioni significative (|r| >= {threshold}): {len(significant_corr)}")
    if len(significant_corr) > 0:
        print("\n🔝 Top 15 correlazioni:")
        print(significant_corr.head(15))

    print(f"\n📊 Statistiche correlazioni:")
    print(f"  Max correlazione positiva: {corr_df['Correlazione'].max():.4f}")
    print(f"  Max correlazione negativa: {corr_df['Correlazione'].min():.4f}")
    print(f"  Media correlazione assoluta: {corr_df['Correlazione_Assoluta'].mean():.4f}")

    # Salva risultati
    corr_df.to_csv(output_dir / 'correlations_with_target.csv', index=False)
    print(f"\n💾 Correlazioni salvate in {output_dir}/correlations_with_target.csv")

    return corr_df

def analyze_group_summaries(df, target_col, output_dir):
    """Crea summary per gruppi specifici"""
    print("\n" + "="*60)
    print("👥 ANALISI SUMMARY PER GRUPPI")
    print("="*60)
    
    group_cols = ['AI_ZonaOmi', 'AI_IdCategoriaCatastale']
    
    # Verifica che le colonne esistano
    missing_group_cols = [col for col in group_cols if col not in df.columns]
    if missing_group_cols:
        print(f"❌ Colonne di grouping mancanti: {missing_group_cols}")
        print(f"🔍 Colonne disponibili con 'AI_': {[c for c in df.columns if 'AI_' in c][:10]}")
    else:
        print(f"✅ Colonne di grouping trovate: {group_cols}")
    
    group_summaries = {}
    
    for group_col in group_cols:
        if group_col in df.columns:
            print(f"\n📊 Summary per gruppo: {group_col}")
            print("-" * 50)

            valid_data = df[[group_col, target_col]].dropna()
            if len(valid_data) == 0:
                print(f"❌ Nessun dato valido per {group_col}")
                continue

            group_stats = valid_data.groupby(group_col)[target_col].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(2)

            group_stats = group_stats.sort_values('count', ascending=False)
            group_stats['cv'] = (group_stats['std'] / group_stats['mean'] * 100).round(2)

            print(f"Gruppi trovati: {len(group_stats)}")
            print(f"\n🔝 Top 10 gruppi per dimensione:")
            print(group_stats.head(10))

            group_summaries[group_col] = group_stats
            
            # Salva su CSV
            output_file = output_dir / f'group_summary_{group_col}.csv'
            group_stats.to_csv(output_file)
            print(f"💾 Salvato: {output_file}")
        else:
            print(f"⚠️  Colonna {group_col} non trovata, skip")
    
    return group_summaries

def identify_geospatial_columns(df, output_dir):
    """Identifica colonne che potrebbero contenere dati geospaziali"""
    print("\n" + "="*60)
    print("🗺️  CHECK GEOSPAZIALE RAPIDO")
    print("="*60)

    candidates = []
    all_cols = df.columns.tolist()

    # Pattern di interesse
    geo_patterns = [
        'wkt', 'geometry', 'geom', 'geojson', 'the_geom', 'shape', 'polygon', 'point', 'linestring',
        'lat', 'latitude', 'lon', 'lng', 'longitude', 'coord', 'x_', 'y_', 'easting', 'northing',
        'geo', 'spatial', 'location', 'posizione', 'indirizzo', 'address'
    ]

    for col in all_cols:
        col_l = col.lower()
        for pat in geo_patterns:
            if pat in col_l:
                sample_vals = df[col].dropna().head(3).tolist()
                candidates.append({
                    'Colonna': col,
                    'Tipo': str(df[col].dtype),
                    'Pattern': pat,
                    'NonNull_Count': int(df[col].count()),
                    'Esempi': str(sample_vals)[:100]  # Limita lunghezza esempi
                })
                break

    if not candidates:
        print("\n❌ Nessuna colonna geospaziale candidata individuata")
        return pd.DataFrame()

    geo_df = pd.DataFrame(candidates).drop_duplicates(subset=['Colonna', 'Pattern'])
    print(f"\n🗺️  Colonne candidate geospaziali: {len(geo_df)}")
    print(geo_df)
    
    # Salva risultati
    geo_df.to_csv(output_dir / 'geospatial_columns_check.csv', index=False)
    print(f"\n💾 Check geospaziale salvato in {output_dir}/geospatial_columns_check.csv")
    
    return geo_df

def main():
    """Funzione principale che esegue l'analisi completa"""
    print("🚀 Avvio Analisi Esplorativa Dataset Raw")
    print("=" * 60)
    
    # Setup
    output_dir = setup_output_dir()
    print(f"📁 Directory output: {output_dir}")
    
    # Caricamento dati
    config, df = load_config_and_data()
    target_col = config['target']['column_candidates'][0]
    
    # Analisi 1: Overview
    print("\n🔍 FASE 1: Overview Dataset")
    dataset_overview(df, config)
    
    # Analisi 2: Missingness
    print("\n🔍 FASE 2: Analisi Missingness")
    missing_stats = analyze_missingness(df, output_dir)
    
    # Analisi 3: Target Distribution
    print("\n🔍 FASE 3: Distribuzione Target")
    target_stats, price_bands = analyze_target_distribution(df, target_col, output_dir)
    
    # Analisi 4: Correlazioni
    print("\n🔍 FASE 4: Correlazioni")
    correlations = analyze_correlations(df, target_col, output_dir)
    
    # Analisi 5: Summary per gruppi
    print("\n🔍 FASE 5: Summary per Gruppi")
    group_summaries = analyze_group_summaries(df, target_col, output_dir)
    
    # Analisi 6: Check geospaziale
    print("\n🔍 FASE 6: Check Geospaziale")
    geo_cols = identify_geospatial_columns(df, output_dir)
    
    # Riepilogo finale
    print("\n" + "="*60)
    print("🎉 ANALISI ESPLORATIVA COMPLETATA")
    print("="*60)
    
    print(f"\n📊 Dataset analizzato:")
    print(f"  • Dimensioni: {df.shape[0]:,} righe × {df.shape[1]} colonne")
    print(f"  • Target: {target_col}")
    print(f"  • Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n💾 File generati in {output_dir}/:")
    output_files = list(output_dir.glob('*.csv'))
    for f in output_files:
        print(f"  • {f.name}")
    
    print(f"\n✅ Analisi completata con successo!")
    print(f"Puoi ora procedere con il preprocessing e il training del modello.")

if __name__ == "__main__":
    main()