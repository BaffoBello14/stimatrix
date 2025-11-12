#!/usr/bin/env python3
"""Analisi data-driven del dataset per identificare colonne inutili."""
import sys
from pathlib import Path

# Add src to path (same as main.py)
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import pandas as pd
    import numpy as np
    
    # Load raw data
    df = pd.read_parquet('data/raw/raw.parquet')
    
    print("="*100)
    print("📊 ANALISI DATASET RAW - DATA DRIVEN")
    print("="*100)
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Analisi per colonna
    results = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique(dropna=True)
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        
        # Identifica se è constant
        is_constant = nunique <= 1
        
        # Identifica se è quasi-costante (>98% un solo valore)
        if nunique > 0 and len(df) > 0:
            top_freq = df[col].value_counts().iloc[0] / len(df) * 100
            is_quasi_constant = top_freq > 98
        else:
            top_freq = 100
            is_quasi_constant = True
        
        # Identifica se è ID-like (molti unique)
        is_id_like = nunique == len(df) or (nunique > len(df) * 0.95 and dtype == 'object')
        
        # Identifica se è quasi tutto missing
        is_mostly_missing = missing_pct > 95
        
        # Check se sembra codice (pattern numerico in stringa che dovrebbe rimanere stringa)
        is_code_like = False
        if dtype == 'object' and nunique > 0:
            sample = df[col].dropna().head(100).astype(str)
            if len(sample) > 0:
                # Codici: solo numeri con leading zeros o pattern fisso
                all_numeric = sample.str.match(r'^\d+$').mean() > 0.8
                has_leading_zeros = sample.str.match(r'^0\d+').mean() > 0.3
                is_code_like = all_numeric and has_leading_zeros
        
        results.append({
            'column': col,
            'dtype': dtype,
            'nunique': nunique,
            'missing_%': missing_pct,
            'constant': is_constant,
            'quasi_constant': is_quasi_constant,
            'id_like': is_id_like,
            'mostly_missing': is_mostly_missing,
            'code_like': is_code_like,
            'top_freq_%': top_freq
        })
    
    df_analysis = pd.DataFrame(results)
    
    # Categorie di colonne da droppare
    print("\n" + "="*100)
    print("🗑️  COLONNE DA DROPPARE (SICURAMENTE INUTILI)")
    print("="*100)
    
    # 1. Costanti
    constants = df_analysis[df_analysis['constant']].sort_values('column')
    if len(constants) > 0:
        print(f"\n1️⃣  COSTANTI ({len(constants)} colonne):")
        print("   Hanno 0-1 valori unici → Non portano informazione")
        for _, row in constants.iterrows():
            val = df[row['column']].dropna().unique()
            val_str = str(val[0]) if len(val) > 0 else 'NaN'
            print(f"   • {row['column']:<50} = '{val_str}' (sempre)")
    
    # 2. Quasi-costanti
    quasi_constants = df_analysis[(df_analysis['quasi_constant']) & (~df_analysis['constant'])].sort_values('top_freq_%', ascending=False).head(20)
    if len(quasi_constants) > 0:
        print(f"\n2️⃣  QUASI-COSTANTI ({len(quasi_constants)} colonne mostrate, top 20):")
        print("   >98% dei valori è lo stesso → Quasi zero variabilità")
        for _, row in quasi_constants.iterrows():
            print(f"   • {row['column']:<50} top_freq={row['top_freq_%']:.1f}%, missing={row['missing_%']:.1f}%")
    
    # 3. ID-like
    id_like = df_analysis[df_analysis['id_like']].sort_values('column')
    if len(id_like) > 0:
        print(f"\n3️⃣  ID-LIKE ({len(id_like)} colonne):")
        print("   Quasi tutti valori unici → Identificatori, non feature")
        for _, row in id_like.iterrows():
            print(f"   • {row['column']:<50} nunique={row['nunique']:,}/{len(df):,} ({row['nunique']/len(df)*100:.1f}%)")
    
    # 4. Quasi tutto missing
    mostly_missing = df_analysis[df_analysis['mostly_missing']].sort_values('missing_%', ascending=False)
    if len(mostly_missing) > 0:
        print(f"\n4️⃣  QUASI TUTTO MISSING ({len(mostly_missing)} colonne):")
        print("   >95% missing → Troppo pochi dati per essere utili")
        for _, row in mostly_missing.iterrows():
            print(f"   • {row['column']:<50} missing={row['missing_%']:.1f}%")
    
    # 5. Colonne CODE-LIKE (per numeric_coercion blacklist)
    code_like = df_analysis[df_analysis['code_like']].sort_values('column')
    print(f"\n5️⃣  CODE-LIKE ({len(code_like)} colonne):")
    print("   Stringhe numeriche con leading zeros → NON devono essere convertite in float")
    print("   (da aggiungere a numeric_coercion.blacklist_globs)")
    for _, row in code_like.iterrows():
        sample_val = df[row['column']].dropna().iloc[0] if len(df[row['column']].dropna()) > 0 else 'N/A'
        print(f"   • {row['column']:<50} es: '{sample_val}'")
    
    # Summary colonne da droppare
    to_drop = df_analysis[
        df_analysis['constant'] | 
        df_analysis['quasi_constant'] | 
        df_analysis['id_like'] | 
        df_analysis['mostly_missing']
    ]['column'].tolist()
    
    print(f"\n" + "="*100)
    print(f"📝 SUMMARY: {len(to_drop)} colonne DA DROPPARE")
    print("="*100)
    
    # Group by reason
    drop_by_reason = {}
    for col in to_drop:
        row = df_analysis[df_analysis['column'] == col].iloc[0]
        reasons = []
        if row['constant']: reasons.append('COSTANTE')
        elif row['quasi_constant']: reasons.append('QUASI-COSTANTE')
        if row['id_like']: reasons.append('ID-LIKE')
        if row['mostly_missing']: reasons.append('MOSTLY-MISSING')
        
        reason_key = ', '.join(reasons)
        if reason_key not in drop_by_reason:
            drop_by_reason[reason_key] = []
        drop_by_reason[reason_key].append(col)
    
    for reason, cols in sorted(drop_by_reason.items()):
        print(f"\n{reason} ({len(cols)} colonne):")
        for col in sorted(cols):
            print(f"   • {col}")
    
    # Salva lista per config
    print(f"\n" + "="*100)
    print("📋 YAML PER CONFIG (feature_pruning.drop_columns):")
    print("="*100)
    print("feature_pruning:")
    print("  drop_columns:")
    for col in sorted(to_drop):
        print(f"    - '{col}'")
    
    # Analisi numeric_coercion
    print(f"\n" + "="*100)
    print("🔧 ANALISI NUMERIC_COERCION")
    print("="*100)
    print("\nCosa fa: Converte colonne 'object' che sembrano numeriche (es. '123.45') in float")
    print("Blacklist: Colonne da NON convertire (es. codici catastali '00020' → devono rimanere stringhe)")
    
    # Identifica colonne che dovrebbero essere in blacklist
    blacklist_candidates = []
    
    for col in df.select_dtypes(include=['object']).columns:
        if col in to_drop:
            continue  # Skip colonne già da droppare
        
        sample = df[col].dropna().head(200).astype(str)
        if len(sample) == 0:
            continue
        
        # Pattern che indicano "codice" (non numero puro)
        has_leading_zeros = sample.str.match(r'^0\d+').mean() > 0.2
        has_fixed_length = sample.str.len().std() < 1.0  # Lunghezza molto fissa
        all_digits = sample.str.match(r'^\d+$').mean() > 0.8
        
        # Se ha leading zeros o pattern fisso → è un codice
        if all_digits and (has_leading_zeros or has_fixed_length):
            blacklist_candidates.append({
                'column': col,
                'reason': 'leading_zeros' if has_leading_zeros else 'fixed_length',
                'sample': sample.iloc[0] if len(sample) > 0 else '',
                'nunique': df[col].nunique()
            })
        
        # Pattern speciali
        col_lower = col.lower()
        if any(x in col_lower for x in ['id', 'cod', 'foglio', 'particella', 'subalterno', 'sezione', 'zona']):
            if col not in [x['column'] for x in blacklist_candidates]:
                blacklist_candidates.append({
                    'column': col,
                    'reason': 'name_pattern',
                    'sample': sample.iloc[0] if len(sample) > 0 else '',
                    'nunique': df[col].nunique()
                })
    
    if blacklist_candidates:
        print(f"\n✅ COLONNE DA BLACKLIST ({len(blacklist_candidates)} trovate):")
        print("   Queste colonne SEMBRANO numeriche ma NON devono essere convertite:")
        for item in sorted(blacklist_candidates, key=lambda x: x['column']):
            print(f"   • {item['column']:<50} reason={item['reason']:<15} sample='{item['sample']}' nunique={item['nunique']}")
        
        print(f"\n📋 YAML PER numeric_coercion.blacklist_globs:")
        print("numeric_coercion:")
        print("  enabled: true")
        print("  threshold: 0.95")
        print("  blacklist_globs:")
        
        # Create glob patterns
        patterns = set()
        for item in blacklist_candidates:
            col = item['column']
            # Try to create glob pattern
            if col.startswith('II_'):
                patterns.add("'II_*'")
            elif col.startswith('AI_Id'):
                patterns.add("'AI_Id*'")
            elif col.startswith('PC_'):
                patterns.add("'PC_*'")
            elif 'Foglio' in col:
                patterns.add("'*Foglio*'")
            elif 'Particella' in col:
                patterns.add("'*Particella*'")
            elif 'Subalterno' in col:
                patterns.add("'*Subalterno*'")
            elif 'Sezione' in col:
                patterns.add("'*Sezione*'")
            elif 'Zona' in col:
                patterns.add("'*Zona*'")
            elif 'COD' in col.upper():
                patterns.add("'*COD*'")
            elif 'Id' in col:
                patterns.add("'*Id*'")
            else:
                patterns.add(f"'{col}'")
        
        for pattern in sorted(patterns):
            print(f"    - {pattern}")
    
    # Keep list
    keep_cols = [c for c in df.columns if c not in to_drop]
    print(f"\n" + "="*100)
    print(f"✅ COLONNE DA MANTENERE: {len(keep_cols)}/{len(df.columns)}")
    print("="*100)
    
    # Sample alcune colonne interessanti
    interesting = df_analysis[
        (~df_analysis['constant']) & 
        (~df_analysis['quasi_constant']) & 
        (~df_analysis['id_like']) & 
        (~df_analysis['mostly_missing']) &
        (df_analysis['missing_%'] < 50)
    ].sort_values('nunique', ascending=False).head(30)
    
    print("\nTop 30 colonne potenzialmente utili (alta variabilità, pochi missing):")
    print(f"{'Column':<50} {'Type':<15} {'Unique':>10} {'Missing%':>10}")
    print("-" * 100)
    for _, row in interesting.iterrows():
        print(f"{row['column']:<50} {row['dtype']:<15} {row['nunique']:>10,} {row['missing_%']:>9.1f}%")
    
    print("\n" + "="*100)
    print("✅ Analisi completata!")
    print("="*100)
    
except Exception as e:
    print(f"❌ Errore: {e}")
    import traceback
    traceback.print_exc()
