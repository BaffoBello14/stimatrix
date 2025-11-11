#!/usr/bin/env python3
"""
Script per testare le ottimizzazioni implementate.
Confronta le performance tra configurazione standard e ottimizzata.
"""
import sys
import subprocess
import json
import pandas as pd
from pathlib import Path
import shutil
import time

def run_pipeline(config_file: str, output_suffix: str) -> dict:
    """Esegue la pipeline con un dato config file."""
    print(f"\n{'='*60}")
    print(f"Eseguendo pipeline con {config_file}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Backup della cartella models esistente se presente
    models_dir = Path("models")
    if models_dir.exists():
        backup_dir = Path(f"models_backup_{output_suffix}_{int(time.time())}")
        shutil.move(str(models_dir), str(backup_dir))
        print(f"Backup dei modelli esistenti in: {backup_dir}")
    
    # Esegui la pipeline
    cmd = [
        sys.executable,
        "main.py" if "original" in output_suffix else "main_optimized.py",
        "--config", config_file,
        "--steps", "training", "evaluation"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Pipeline completata con successo!")
        
        # Tempo di esecuzione
        execution_time = time.time() - start_time
        print(f"Tempo di esecuzione: {execution_time:.2f} secondi")
        
        # Carica i risultati
        results_file = models_dir / f"validation_results.csv"
        if results_file.exists():
            df_results = pd.read_csv(results_file)
            
            # Rinomina il file per preservarlo
            new_name = models_dir / f"validation_results_{output_suffix}.csv"
            shutil.copy(str(results_file), str(new_name))
            
            return {
                "success": True,
                "execution_time": execution_time,
                "results": df_results,
                "results_file": str(new_name)
            }
        else:
            print("ATTENZIONE: File dei risultati non trovato!")
            return {
                "success": False,
                "execution_time": execution_time,
                "error": "Results file not found"
            }
            
    except subprocess.CalledProcessError as e:
        print(f"ERRORE nell'esecuzione della pipeline:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return {
            "success": False,
            "execution_time": time.time() - start_time,
            "error": str(e)
        }


def compare_results(original_results: pd.DataFrame, optimized_results: pd.DataFrame) -> pd.DataFrame:
    """Confronta i risultati tra versione originale e ottimizzata."""
    # Prepara i dataframe per il merge
    original = original_results.copy()
    original['Version'] = 'Original'
    
    optimized = optimized_results.copy()
    optimized['Version'] = 'Optimized'
    
    # Combina i risultati
    combined = pd.concat([original, optimized], ignore_index=True)
    
    # Crea un confronto side-by-side per i migliori modelli
    comparison = []
    
    # Trova i top 10 modelli per versione
    top_original = original.nsmallest(10, 'Test_RMSE')[['Model', 'Test_RMSE', 'Test_R2']]
    top_optimized = optimized.nsmallest(10, 'Test_RMSE')[['Model', 'Test_RMSE', 'Test_R2']]
    
    print("\n" + "="*80)
    print("CONFRONTO TOP 10 MODELLI")
    print("="*80)
    
    print("\nORIGINALE:")
    print(top_original.to_string(index=False))
    
    print("\n\nOTTIMIZZATO:")
    print(top_optimized.to_string(index=False))
    
    # Calcola miglioramenti
    print("\n" + "="*80)
    print("ANALISI MIGLIORAMENTI")
    print("="*80)
    
    # Confronta i migliori modelli overall
    best_original = original.loc[original['Test_RMSE'].idxmin()]
    best_optimized = optimized.loc[optimized['Test_RMSE'].idxmin()]
    
    print(f"\nMiglior modello ORIGINALE: {best_original['Model']}")
    print(f"  - R²: {best_original['Test_R2']:.4f}")
    print(f"  - RMSE: {best_original['Test_RMSE']:.2f}")
    
    print(f"\nMiglior modello OTTIMIZZATO: {best_optimized['Model']}")
    print(f"  - R²: {best_optimized['Test_R2']:.4f}")
    print(f"  - RMSE: {best_optimized['Test_RMSE']:.2f}")
    
    # Calcola miglioramento percentuale
    rmse_improvement = (best_original['Test_RMSE'] - best_optimized['Test_RMSE']) / best_original['Test_RMSE'] * 100
    r2_improvement = (best_optimized['Test_R2'] - best_original['Test_R2']) / (1 - best_original['Test_R2']) * 100
    
    print(f"\nMIGLIORAMENTO:")
    print(f"  - RMSE: {rmse_improvement:.2f}% {'(migliorato)' if rmse_improvement > 0 else '(peggiorato)'}")
    print(f"  - R²: {r2_improvement:.2f}% del gap verso R²=1")
    
    return combined


def main():
    """Funzione principale per testare le ottimizzazioni."""
    print("STIMATRIX - Test delle Ottimizzazioni")
    print("="*60)
    
    # Verifica che esistano i file necessari
    config_original = Path("config/config.yaml")
    config_optimized = Path("config/config_optimized.yaml")
    
    if not config_original.exists():
        print(f"ERRORE: File {config_original} non trovato!")
        return
    
    if not config_optimized.exists():
        print(f"ERRORE: File {config_optimized} non trovato!")
        print("Assicurati di aver eseguito le modifiche per creare config_optimized.yaml")
        return
    
    # Verifica che i dati preprocessati esistano
    preprocessed_dir = Path("data/preprocessed")
    if not preprocessed_dir.exists() or not any(preprocessed_dir.glob("X_*.parquet")):
        print("ERRORE: Dati preprocessati non trovati!")
        print("Esegui prima: python main.py --steps preprocessing")
        return
    
    results = {}
    
    # Test 1: Pipeline originale
    print("\n1. Esecuzione pipeline ORIGINALE...")
    results['original'] = run_pipeline(str(config_original), "original")
    
    # Test 2: Pipeline ottimizzata
    print("\n2. Esecuzione pipeline OTTIMIZZATA...")
    results['optimized'] = run_pipeline(str(config_optimized), "optimized")
    
    # Confronta i risultati
    if results['original']['success'] and results['optimized']['success']:
        print("\n3. Confronto dei risultati...")
        comparison = compare_results(
            results['original']['results'],
            results['optimized']['results']
        )
        
        # Salva il confronto
        comparison_file = Path("models/optimization_comparison.csv")
        comparison.to_csv(comparison_file, index=False)
        print(f"\nConfronto salvato in: {comparison_file}")
        
        # Report finale
        print("\n" + "="*80)
        print("REPORT FINALE")
        print("="*80)
        print(f"Tempo esecuzione ORIGINALE: {results['original']['execution_time']:.2f} secondi")
        print(f"Tempo esecuzione OTTIMIZZATO: {results['optimized']['execution_time']:.2f} secondi")
        print(f"Speedup: {results['original']['execution_time'] / results['optimized']['execution_time']:.2f}x")
        
    else:
        print("\nERRORE: Una o entrambe le pipeline hanno fallito!")
        if not results['original']['success']:
            print(f"Pipeline originale fallita: {results['original'].get('error', 'Unknown error')}")
        if not results['optimized']['success']:
            print(f"Pipeline ottimizzata fallita: {results['optimized'].get('error', 'Unknown error')}")
    
    print("\nTest completato!")


if __name__ == "__main__":
    main()