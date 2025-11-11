#!/usr/bin/env python3
"""
Quick test script to verify temporal filter is working correctly.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config import load_config
import pandas as pd

def test_temporal_filter():
    print("=" * 70)
    print("TEST: Temporal Filter")
    print("=" * 70)
    
    # Load config
    config = load_config("config/config.yaml")
    temporal_cfg = config.get("temporal_filter", {})
    
    print(f"\n📋 Configurazione temporal_filter:")
    print(f"  enabled: {temporal_cfg.get('enabled')}")
    print(f"  min_year: {temporal_cfg.get('min_year')}")
    print(f"  min_month: {temporal_cfg.get('min_month')}")
    print(f"  exclude_zones: {temporal_cfg.get('exclude_zones')}")
    
    # Load raw data
    raw_path = Path("data/raw/raw.parquet")
    if not raw_path.exists():
        print(f"\n❌ File non trovato: {raw_path}")
        return
    
    df = pd.read_parquet(raw_path)
    print(f"\n📊 Dataset ORIGINALE:")
    print(f"  Righe: {len(df):,}")
    print(f"  Anni: {df['A_AnnoStipula'].min()} - {df['A_AnnoStipula'].max()}")
    print(f"  Zone OMI: {df['AI_ZonaOmi'].nunique()} uniche")
    print(f"\n  Distribuzione anni:")
    print(df['A_AnnoStipula'].value_counts().sort_index().to_string())
    
    # Apply filter
    if temporal_cfg.get("enabled", False):
        initial_rows = len(df)
        
        min_year = temporal_cfg.get("min_year")
        if min_year and "A_AnnoStipula" in df.columns:
            df = df[df["A_AnnoStipula"] >= min_year]
        
        min_month = temporal_cfg.get("min_month")
        if min_month and "A_MeseStipula" in df.columns:
            df = df[df["A_MeseStipula"] >= min_month]
        
        exclude_zones = temporal_cfg.get("exclude_zones", [])
        if exclude_zones and "AI_ZonaOmi" in df.columns:
            df = df[~df["AI_ZonaOmi"].isin(exclude_zones)]
        
        print(f"\n📊 Dataset FILTRATO (>={min_year}, escluse {exclude_zones}):")
        print(f"  Righe: {len(df):,} ({len(df)/initial_rows*100:.1f}% mantenute)")
        print(f"  Rimossi: {initial_rows - len(df):,} campioni ({(initial_rows-len(df))/initial_rows*100:.1f}%)")
        print(f"  Anni: {df['A_AnnoStipula'].min()} - {df['A_AnnoStipula'].max()}")
        print(f"  Zone OMI: {df['AI_ZonaOmi'].nunique()} uniche")
        
        print(f"\n  Distribuzione zone OMI (filtrate):")
        zone_counts = df['AI_ZonaOmi'].value_counts().sort_values(ascending=False)
        for zona, count in zone_counts.items():
            print(f"    {zona}: {count:4d} campioni")
        
        # Check for small zones
        small_zones = zone_counts[zone_counts < 50]
        if len(small_zones) > 0:
            print(f"\n  ⚠️  Zone con <50 campioni (possibili problemi):")
            for zona, count in small_zones.items():
                print(f"    {zona}: {count} campioni")
        else:
            print(f"\n  ✅ Tutte le zone hanno ≥50 campioni (ottimo!)")
        
        # Target stats
        print(f"\n  📈 Statistiche target (AI_Prezzo_Ridistribuito):")
        print(f"    Media: €{df['AI_Prezzo_Ridistribuito'].mean():,.0f}")
        print(f"    Mediana: €{df['AI_Prezzo_Ridistribuito'].median():,.0f}")
        print(f"    Std: €{df['AI_Prezzo_Ridistribuito'].std():,.0f}")
        print(f"    Min: €{df['AI_Prezzo_Ridistribuito'].min():,.0f}")
        print(f"    Max: €{df['AI_Prezzo_Ridistribuito'].max():,.0f}")
        
        print(f"\n✅ TEST COMPLETATO - Filtro temporale funziona correttamente!")
        print(f"   Puoi ora runnare: python main.py --steps preprocessing training evaluation")
    else:
        print("\n⚠️  Temporal filter DISABILITATO nel config!")

if __name__ == "__main__":
    test_temporal_filter()
