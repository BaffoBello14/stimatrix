#!/usr/bin/env python3
"""
Script per verificare la presenza delle colonne CENED nel dataset raw.parquet
"""

import pandas as pd
import sys

def check_cened_columns(parquet_path='data/raw/raw.parquet'):
    """Verifica presenza colonne CENED nel dataset."""
    
    print('=' * 70)
    print('📊 ANALISI COLONNE CENED IN RAW.PARQUET')
    print('=' * 70)
    print()
    
    try:
        # Carica il parquet
        print(f'⏳ Caricamento {parquet_path}...')
        df = pd.read_parquet(parquet_path)
        print(f'✅ File caricato correttamente')
        print()
        
        # Info generali
        print(f'📏 Dimensioni dataset: {df.shape[0]:,} righe × {df.shape[1]:,} colonne')
        print()
        
        # Cerca colonne CENED
        cened1_cols = sorted([c for c in df.columns if c.startswith('C1_')])
        cened2_cols = sorted([c for c in df.columns if c.startswith('C2_')])
        
        # Report CENED1
        print('🔍 COLONNE VIEW attiimmobili_cened1 (alias C1):')
        if cened1_cols:
            print(f'   ✅ Trovate {len(cened1_cols)} colonne')
            print(f'   Colonne:')
            for col in cened1_cols:
                non_null = df[col].notna().sum()
                pct = (non_null / len(df)) * 100
                print(f'      - {col:<40} ({non_null:>6,} valori non-null, {pct:>5.1f}%)')
        else:
            print('   ❌ NESSUNA COLONNA TROVATA!')
            print('   ⚠️  Le view CENED non sono state incluse nella query.')
        print()
        
        # Report CENED2
        print('🔍 COLONNE VIEW attiimmobili_cened2 (alias C2):')
        if cened2_cols:
            print(f'   ✅ Trovate {len(cened2_cols)} colonne')
            print(f'   Colonne:')
            for col in cened2_cols:
                non_null = df[col].notna().sum()
                pct = (non_null / len(df)) * 100
                print(f'      - {col:<40} ({non_null:>6,} valori non-null, {pct:>5.1f}%)')
        else:
            print('   ❌ NESSUNA COLONNA TROVATA!')
            print('   ⚠️  Le view CENED non sono state incluse nella query.')
        print()
        
        # Riepilogo
        total_cened = len(cened1_cols) + len(cened2_cols)
        print('=' * 70)
        print('📊 RIEPILOGO')
        print('=' * 70)
        print(f'Totale colonne dataset: {len(df.columns)}')
        print(f'Colonne CENED (C1 + C2): {total_cened}')
        print(f'Percentuale CENED: {(total_cened / len(df.columns) * 100):.1f}%')
        print()
        
        if total_cened > 0:
            print('✅ LE VIEW CENED SONO STATE INTEGRATE CORRETTAMENTE!')
            print()
            
            # Mostra sample
            print('📋 CAMPIONE PRIMI 5 RECORD (prime colonne CENED):')
            sample_cols = (cened1_cols[:3] + cened2_cols[:3])[:5]
            if sample_cols:
                print()
                print(df[sample_cols].head(5).to_string())
                print()
        else:
            print('❌ LE VIEW CENED NON SONO PRESENTI NEL DATASET')
            print()
            print('🔧 POSSIBILI CAUSE:')
            print('   1. Il file raw.parquet è stato generato PRIMA del refactoring')
            print('   2. Le view non esistono nel database')
            print('   3. Lo schema JSON non contiene le view CENED')
            print()
            print('🚀 SOLUZIONE: Rigenera il dataset')
            print('   python main.py --steps schema,dataset')
            print()
        
        # Info sul file size
        import os
        file_size = os.path.getsize(parquet_path) / (1024 * 1024)  # MB
        print(f'💾 Dimensione file: {file_size:.2f} MB')
        if file_size > 50:
            print('   ℹ️  Il file supera i 50MB - normale per dataset ML con molte features')
        print()
        
        return total_cened > 0
        
    except FileNotFoundError:
        print(f'❌ ERRORE: File non trovato: {parquet_path}')
        print('   Assicurati di aver eseguito: python main.py --steps dataset')
        return False
    except Exception as e:
        print(f'❌ ERRORE: {e}')
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    parquet_file = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/raw.parquet'
    success = check_cened_columns(parquet_file)
    sys.exit(0 if success else 1)
