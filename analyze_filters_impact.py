#!/usr/bin/env python3
"""
Script di analisi rapida per verificare l'impatto dei filtri configurati.

Esegue:
1. Analisi distribuzione temporale
2. Calcolo righe rimosse per ciascun filtro
3. Statistiche finali del dataset filtrato
4. Confronto distribuzioni pre/post filtri

Usage:
    python analyze_filters_impact.py
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def analyze_temporal_distribution(df: pd.DataFrame) -> None:
    """Analizza distribuzione temporale del dataset."""
    print("\n" + "=" * 80)
    print("📅 DISTRIBUZIONE TEMPORALE")
    print("=" * 80)
    
    if 'A_AnnoStipula' not in df.columns:
        print("⚠️  Colonna A_AnnoStipula non trovata")
        return
    
    # Distribuzione per anno
    year_dist = df['A_AnnoStipula'].value_counts().sort_index()
    print("\nTransazioni per anno:")
    for year, count in year_dist.items():
        pct = count / len(df) * 100
        print(f"  {year}: {count:>6,} ({pct:>5.1f}%)")
    
    # Range temporale
    print(f"\nPeriodo: {df['A_AnnoStipula'].min()} - {df['A_AnnoStipula'].max()}")
    
    # Distribuzione per anno-mese (ultimi 12 mesi)
    if 'A_MeseStipula' in df.columns:
        df_temp = df.copy()
        df_temp['year_month'] = df_temp['A_AnnoStipula'].astype(int) * 100 + df_temp['A_MeseStipula'].astype(int)
        ym_dist = df_temp['year_month'].value_counts().sort_index().tail(12)
        print("\nUltimi 12 mesi (anno-mese):")
        for ym, count in ym_dist.items():
            year = ym // 100
            month = ym % 100
            print(f"  {year}-{month:02d}: {count:>5,}")


def simulate_filters(df: pd.DataFrame, config: dict) -> dict:
    """Simula applicazione filtri e restituisce statistiche."""
    filters = config.get('data_filters', {})
    
    if not filters or not any(v is not None for k, v in filters.items() if k not in ['description', 'experiment_name']):
        print("\n⚠️  Nessun filtro attivo nella configurazione")
        return {}
    
    initial_rows = len(df)
    results = {
        'initial': initial_rows,
        'steps': []
    }
    
    print("\n" + "=" * 80)
    print("🔍 SIMULAZIONE FILTRI")
    print("=" * 80)
    print(f"\n📊 Dataset iniziale: {initial_rows:,} righe")
    
    df_filtered = df.copy()
    
    # Filtro 1: Anno minimo
    if filters.get('anno_min') and 'A_AnnoStipula' in df.columns:
        anno_min = filters['anno_min']
        before = len(df_filtered)
        df_filtered = df_filtered[df_filtered['A_AnnoStipula'] >= anno_min]
        removed = before - len(df_filtered)
        pct = removed / initial_rows * 100
        
        print(f"\n1️⃣  Filtro anno_min >= {anno_min}:")
        print(f"   Rimossi: {removed:,} righe ({pct:.1f}%)")
        print(f"   Rimanenti: {len(df_filtered):,} righe ({len(df_filtered)/initial_rows*100:.1f}%)")
        
        results['steps'].append({
            'filter': f'anno_min >= {anno_min}',
            'removed': removed,
            'remaining': len(df_filtered),
            'pct_removed': pct
        })
    
    # Filtro 2: Zone escluse
    if filters.get('zone_escluse') and 'AI_ZonaOmi' in df.columns:
        zone_escluse = filters['zone_escluse']
        before = len(df_filtered)
        
        # Mostra distribuzione zone da escludere
        print(f"\n2️⃣  Filtro zone_escluse: {zone_escluse}")
        for zone in zone_escluse:
            count = (df_filtered['AI_ZonaOmi'] == zone).sum()
            if count > 0:
                print(f"   - {zone}: {count:,} righe ({count/before*100:.1f}%)")
        
        df_filtered = df_filtered[~df_filtered['AI_ZonaOmi'].isin(zone_escluse)]
        removed = before - len(df_filtered)
        pct = removed / initial_rows * 100
        
        print(f"   Rimossi totale: {removed:,} righe ({pct:.1f}%)")
        print(f"   Rimanenti: {len(df_filtered):,} righe ({len(df_filtered)/initial_rows*100:.1f}%)")
        
        results['steps'].append({
            'filter': f'zone_escluse: {zone_escluse}',
            'removed': removed,
            'remaining': len(df_filtered),
            'pct_removed': pct
        })
    
    # Filtro 3: Tipologie escluse
    if filters.get('tipologie_escluse') and 'AI_IdTipologiaEdilizia' in df.columns:
        tipo_escluse = filters['tipologie_escluse']
        before = len(df_filtered)
        
        # Mostra distribuzione tipologie da escludere
        print(f"\n3️⃣  Filtro tipologie_escluse: {tipo_escluse}")
        for tipo in tipo_escluse:
            count = (df_filtered['AI_IdTipologiaEdilizia'].astype(str) == str(tipo)).sum()
            if count > 0:
                print(f"   - Tipologia {tipo}: {count:,} righe ({count/before*100:.1f}%)")
        
        df_filtered = df_filtered[~df_filtered['AI_IdTipologiaEdilizia'].astype(str).isin(tipo_escluse)]
        removed = before - len(df_filtered)
        pct = removed / initial_rows * 100
        
        print(f"   Rimossi totale: {removed:,} righe ({pct:.1f}%)")
        print(f"   Rimanenti: {len(df_filtered):,} righe ({len(df_filtered)/initial_rows*100:.1f}%)")
        
        results['steps'].append({
            'filter': f'tipologie_escluse: {tipo_escluse}',
            'removed': removed,
            'remaining': len(df_filtered),
            'pct_removed': pct
        })
    
    # Riepilogo finale
    final_rows = len(df_filtered)
    total_removed = initial_rows - final_rows
    total_pct = total_removed / initial_rows * 100
    
    print("\n" + "=" * 80)
    print("📊 RIEPILOGO FINALE")
    print("=" * 80)
    print(f"Dataset iniziale:    {initial_rows:>8,} righe (100.0%)")
    print(f"Dataset finale:      {final_rows:>8,} righe ({final_rows/initial_rows*100:>5.1f}%)")
    print(f"Rimossi totale:      {total_removed:>8,} righe ({total_pct:>5.1f}%)")
    
    results['final'] = final_rows
    results['total_removed'] = total_removed
    results['total_pct_removed'] = total_pct
    
    # Stima split train/val/test
    temporal_cfg = config.get('temporal_split', {})
    if temporal_cfg.get('mode') == 'fraction':
        frac = temporal_cfg.get('fraction', {})
        train_frac = frac.get('train', 0.7)
        valid_frac = frac.get('valid', 0.2)
        test_frac = 1.0 - train_frac - valid_frac
        
        train_size = int(final_rows * train_frac)
        valid_size = int(final_rows * valid_frac)
        test_size = final_rows - train_size - valid_size
        
        print(f"\n📊 Stima split temporale (mode=fraction):")
        print(f"   Train ({train_frac*100:.0f}%):      {train_size:>8,} righe")
        print(f"   Validation ({valid_frac*100:.0f}%): {valid_size:>8,} righe")
        print(f"   Test ({test_frac*100:.0f}%):        {test_size:>8,} righe")
        
        results['split'] = {
            'train': train_size,
            'valid': valid_size,
            'test': test_size
        }
    
    # Warnings
    print("\n⚠️  WARNINGS:")
    if final_rows < 2000:
        print(f"   🚨 Dataset molto piccolo ({final_rows} righe)! Rischio overfitting.")
    elif final_rows < 3000:
        print(f"   ⚠️  Dataset ridotto ({final_rows} righe). Considera di ridurre complessità modelli.")
    
    if total_pct > 70:
        print(f"   🚨 Rimossi {total_pct:.1f}% dei dati! Dataset molto ristretto.")
    elif total_pct > 50:
        print(f"   ⚠️  Rimossi {total_pct:.1f}% dei dati. Verifica rappresentatività.")
    
    # Salva dataset filtrato per verifica (opzionale)
    # df_filtered.to_parquet('data/raw/raw_filtered_preview.parquet', index=False)
    
    return results


def compare_distributions(df_original: pd.DataFrame, df_filtered: pd.DataFrame) -> None:
    """Confronta distribuzioni prima e dopo filtri."""
    print("\n" + "=" * 80)
    print("📊 CONFRONTO DISTRIBUZIONI PRE/POST FILTRI")
    print("=" * 80)
    
    target_col = 'AI_Prezzo_Ridistribuito'
    
    if target_col not in df_original.columns:
        print(f"⚠️  Colonna {target_col} non trovata")
        return
    
    # Target statistics
    print(f"\n💰 TARGET: {target_col}")
    print("\n" + "-" * 80)
    print(f"{'Statistica':<20} {'Originale':>15} {'Filtrato':>15} {'Delta':>15}")
    print("-" * 80)
    
    stats_map = {
        'Count': lambda s: len(s),
        'Mean': lambda s: s.mean(),
        'Median': lambda s: s.median(),
        'Std': lambda s: s.std(),
        'Min': lambda s: s.min(),
        'Max': lambda s: s.max(),
        'Q25': lambda s: s.quantile(0.25),
        'Q75': lambda s: s.quantile(0.75),
        'Skewness': lambda s: s.skew(),
        'Kurtosis': lambda s: s.kurtosis(),
    }
    
    orig = df_original[target_col].dropna()
    filt = df_filtered[target_col].dropna()
    
    for stat_name, stat_func in stats_map.items():
        try:
            val_orig = stat_func(orig)
            val_filt = stat_func(filt)
            
            if stat_name == 'Count':
                delta = val_filt - val_orig
                print(f"{stat_name:<20} {val_orig:>15,.0f} {val_filt:>15,.0f} {delta:>15,.0f}")
            else:
                delta = ((val_filt - val_orig) / val_orig * 100) if val_orig != 0 else 0
                print(f"{stat_name:<20} {val_orig:>15,.2f} {val_filt:>15,.2f} {delta:>14.1f}%")
        except Exception as e:
            print(f"{stat_name:<20} {'ERROR':<15} {'ERROR':<15} {'N/A':<15}")
    
    print("-" * 80)
    
    # Zone distribution
    if 'AI_ZonaOmi' in df_original.columns and 'AI_ZonaOmi' in df_filtered.columns:
        print("\n🗺️  ZONE OMI - Top 10")
        print("-" * 80)
        print(f"{'Zona':<8} {'Originale':>12} {'Filtrato':>12} {'Delta':>12}")
        print("-" * 80)
        
        zones_orig = df_original['AI_ZonaOmi'].value_counts()
        zones_filt = df_filtered['AI_ZonaOmi'].value_counts()
        
        for zone in zones_orig.head(10).index:
            count_orig = zones_orig.get(zone, 0)
            count_filt = zones_filt.get(zone, 0)
            delta = count_filt - count_orig
            print(f"{zone:<8} {count_orig:>12,} {count_filt:>12,} {delta:>12,}")
        
        print("-" * 80)


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("🔍 ANALISI IMPATTO FILTRI - STIMATRIX")
    print("=" * 80)
    
    # Load config
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print(f"❌ Config file non trovato: {config_path}")
        return
    
    config = load_config(str(config_path))
    print(f"✅ Configurazione caricata da: {config_path}")
    
    # Load raw data
    raw_path = Path("data/raw/raw.parquet")
    if not raw_path.exists():
        print(f"❌ Dataset raw non trovato: {raw_path}")
        return
    
    print(f"✅ Dataset raw caricato da: {raw_path}")
    df = pd.read_parquet(raw_path)
    print(f"   Dimensioni: {len(df):,} righe × {len(df.columns)} colonne")
    
    # Analyze temporal distribution
    analyze_temporal_distribution(df)
    
    # Simulate filters and get filtered dataset
    print("\n" + "=" * 80)
    print("🎯 APPLICAZIONE FILTRI DA CONFIG")
    print("=" * 80)
    
    filters = config.get('data_filters', {})
    print("\nFiltri configurati:")
    for key, value in filters.items():
        if key not in ['description', 'experiment_name'] and value is not None:
            print(f"  - {key}: {value}")
    
    # Simulate
    df_filtered = df.copy()
    if filters.get('anno_min') and 'A_AnnoStipula' in df.columns:
        df_filtered = df_filtered[df_filtered['A_AnnoStipula'] >= filters['anno_min']]
    if filters.get('zone_escluse') and 'AI_ZonaOmi' in df.columns:
        df_filtered = df_filtered[~df_filtered['AI_ZonaOmi'].isin(filters['zone_escluse'])]
    if filters.get('tipologie_escluse') and 'AI_IdTipologiaEdilizia' in df.columns:
        df_filtered = df_filtered[~df_filtered['AI_IdTipologiaEdilizia'].astype(str).isin(filters['tipologie_escluse'])]
    
    results = simulate_filters(df, config)
    
    # Compare distributions
    if len(df_filtered) > 0:
        compare_distributions(df, df_filtered)
    
    print("\n" + "=" * 80)
    print("✅ ANALISI COMPLETATA")
    print("=" * 80)
    print("\n💡 Prossimi passi:")
    print("   1. Verifica che dataset finale abbia dimensione adeguata (>2,000 righe)")
    print("   2. Controlla se distribuzione target è cambiata significativamente")
    print("   3. Se OK, esegui training con: python main.py --config fast")
    print("   4. Confronta metriche con baseline (dataset completo)")
    print()


if __name__ == "__main__":
    main()
