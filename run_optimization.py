#!/usr/bin/env python3
"""
Script per eseguire training ottimizzato e confrontare risultati.

Esegue:
1. Preprocessing con feature contestuali
2. Training con regularizzazione aggressiva
3. Confronto metriche baseline vs ottimizzato
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config import load_config
from utils.logger import setup_logger
from preprocessing.pipeline import run_preprocessing
from training.train import run_training
from training.evaluation import run_evaluation


def load_baseline_results() -> Dict[str, Any]:
    """Carica risultati baseline da models/summary.json."""
    baseline_path = Path("models/summary.json")
    if not baseline_path.exists():
        print("⚠️  Nessun baseline trovato. Verrà usato come baseline il primo run.")
        return {}
    
    with open(baseline_path, 'r') as f:
        return json.load(f)


def compare_results(baseline: Dict[str, Any], optimized: Dict[str, Any]):
    """Confronta risultati baseline vs ottimizzato."""
    print("\n" + "="*80)
    print("📊 CONFRONTO RISULTATI: BASELINE vs OTTIMIZZATO")
    print("="*80)
    
    # Best model comparison
    baseline_models = baseline.get('models', {})
    optimized_models = optimized.get('models', {})
    
    if not baseline_models or not optimized_models:
        print("⚠️  Dati insufficienti per confronto")
        return
    
    # Find best models (CatBoost)
    catboost_baseline = baseline_models.get('catboost', {})
    catboost_optimized = optimized_models.get('catboost', {})
    
    print("\n🎯 CATBOOST - Miglior Modello")
    print("-" * 80)
    
    # Test metrics (transformed scale)
    print("\n📈 METRICHE TEST (scala trasformata):")
    print(f"{'Metric':<15} {'Baseline':>15} {'Ottimizzato':>15} {'Delta':>15} {'Δ%':>10}")
    print("-" * 80)
    
    test_baseline = catboost_baseline.get('metrics_test', {})
    test_optimized = catboost_optimized.get('metrics_test', {})
    
    for metric in ['r2', 'rmse', 'mae', 'mape']:
        baseline_val = test_baseline.get(metric, 0)
        optimized_val = test_optimized.get(metric, 0)
        delta = optimized_val - baseline_val
        delta_pct = (delta / baseline_val * 100) if baseline_val != 0 else 0
        
        # R2: higher is better, others: lower is better
        improvement_icon = "✅" if (delta > 0 and metric == 'r2') or (delta < 0 and metric != 'r2') else "⚠️"
        
        print(f"{metric:<15} {baseline_val:>15.6f} {optimized_val:>15.6f} {delta:>15.6f} {delta_pct:>9.2f}% {improvement_icon}")
    
    # Test metrics (original scale - EURO)
    print("\n💰 METRICHE TEST (scala originale - EURO):")
    print(f"{'Metric':<15} {'Baseline':>15} {'Ottimizzato':>15} {'Delta':>15} {'Δ%':>10}")
    print("-" * 80)
    
    test_orig_baseline = catboost_baseline.get('metrics_test_original', {})
    test_orig_optimized = catboost_optimized.get('metrics_test_original', {})
    
    for metric in ['r2', 'rmse', 'mae', 'mape', 'mape_floor']:
        baseline_val = test_orig_baseline.get(metric, 0)
        optimized_val = test_orig_optimized.get(metric, 0)
        delta = optimized_val - baseline_val
        delta_pct = (delta / baseline_val * 100) if baseline_val != 0 else 0
        
        # Format based on metric
        if metric == 'r2':
            improvement_icon = "✅" if delta > 0 else "⚠️"
            print(f"{metric:<15} {baseline_val:>15.6f} {optimized_val:>15.6f} {delta:>15.6f} {delta_pct:>9.2f}% {improvement_icon}")
        elif metric in ['mape', 'mape_floor']:
            improvement_icon = "✅" if delta < 0 else "⚠️"
            print(f"{metric:<15} {baseline_val:>14.2%} {optimized_val:>14.2%} {delta:>14.2%} {delta_pct:>9.2f}% {improvement_icon}")
        else:  # rmse, mae
            improvement_icon = "✅" if delta < 0 else "⚠️"
            print(f"{metric:<15} {baseline_val:>15,.0f}€ {optimized_val:>15,.0f}€ {delta:>15,.0f}€ {delta_pct:>9.2f}% {improvement_icon}")
    
    # Overfitting metrics
    print("\n🔍 OVERFITTING (Train-Test Gap):")
    print(f"{'Metric':<20} {'Baseline':>15} {'Ottimizzato':>15} {'Delta':>15}")
    print("-" * 80)
    
    overfit_baseline = catboost_baseline.get('overfit', {})
    overfit_optimized = catboost_optimized.get('overfit', {})
    
    for metric in ['gap_r2', 'ratio_rmse', 'ratio_mae', 'ratio_mape']:
        baseline_val = overfit_baseline.get(metric, 0)
        optimized_val = overfit_optimized.get(metric, 0)
        delta = optimized_val - baseline_val
        
        # Lower is better for all overfitting metrics
        improvement_icon = "✅" if delta < 0 else "⚠️"
        
        print(f"{metric:<20} {baseline_val:>15.6f} {optimized_val:>15.6f} {delta:>15.6f} {improvement_icon}")
    
    # Summary
    print("\n" + "="*80)
    print("📝 SUMMARY")
    print("="*80)
    
    rmse_improvement = (test_orig_baseline.get('rmse', 1) - test_orig_optimized.get('rmse', 1)) / test_orig_baseline.get('rmse', 1) * 100
    mape_improvement = (test_orig_baseline.get('mape_floor', 1) - test_orig_optimized.get('mape_floor', 1)) / test_orig_baseline.get('mape_floor', 1) * 100
    r2_improvement = (test_orig_optimized.get('r2', 0) - test_orig_baseline.get('r2', 0)) / test_orig_baseline.get('r2', 1) * 100
    overfit_improvement = (overfit_baseline.get('gap_r2', 1) - overfit_optimized.get('gap_r2', 0)) / overfit_baseline.get('gap_r2', 1) * 100
    
    print(f"\n🎯 Target Metrics:")
    print(f"   • RMSE ridotto:     {rmse_improvement:>6.2f}% {'✅' if rmse_improvement > 0 else '⚠️'}")
    print(f"   • MAPE ridotto:     {mape_improvement:>6.2f}% {'✅' if mape_improvement > 0 else '⚠️'}")
    print(f"   • R² migliorato:    {r2_improvement:>6.2f}% {'✅' if r2_improvement > 0 else '⚠️'}")
    print(f"   • Overfitting ridotto: {overfit_improvement:>6.2f}% {'✅' if overfit_improvement > 0 else '⚠️'}")
    
    print("\n💡 Prossimi Passi:")
    if mape_improvement < 15:
        print("   • MAPE ancora alto: considera modelli specializzati per fascia prezzo")
    if overfit_improvement < 30:
        print("   • Overfitting ancora presente: aumenta regularizzazione o riduci feature")
    if rmse_improvement < 20:
        print("   • RMSE migliorabile: prova feature engineering avanzato o ensemble")
    
    print("\n" + "="*80)


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("🚀 OTTIMIZZAZIONE STIMATRIX")
    print("="*80)
    print("\nObjectives:")
    print("  ✅ Ridurre overfitting (gap R² < 0.10)")
    print("  ✅ Migliorare MAPE (target < 35%)")
    print("  ✅ Migliorare RMSE (target < 26k€)")
    print("  ✅ Performance uniforme tra gruppi")
    print("\nModifiche applicate:")
    print("  • Feature contestuali (zona, tipologia, interazioni)")
    print("  • Regularizzazione aggressiva su tutti i modelli")
    print("  • Outlier detection più stringente")
    print("  • Early stopping abilitato")
    print("  • Hyperparameter ranges ottimizzati")
    print("="*80)
    
    # Load config
    config_path = "config/config_optimized.yaml"
    print(f"\n📄 Loading config: {config_path}")
    config = load_config(config_path)
    
    # Setup logger
    setup_logger(config_path)
    
    # Load baseline results
    print("\n📊 Loading baseline results...")
    baseline_results = load_baseline_results()
    
    # Step 1: Preprocessing (with contextual features)
    print("\n" + "-"*80)
    print("STEP 1: PREPROCESSING (con feature contestuali)")
    print("-"*80)
    run_preprocessing(config)
    
    # Step 2: Training (with aggressive regularization)
    print("\n" + "-"*80)
    print("STEP 2: TRAINING (con regularizzazione aggressiva)")
    print("-"*80)
    optimized_results = run_training(config)
    
    # Step 3: Evaluation
    print("\n" + "-"*80)
    print("STEP 3: EVALUATION")
    print("-"*80)
    run_evaluation(config)
    
    # Step 4: Comparison
    if baseline_results:
        compare_results(baseline_results, optimized_results)
    else:
        print("\n⚠️  Nessun baseline per confronto. Questi risultati sono il nuovo baseline.")
        print("\n📊 RISULTATI OTTIMIZZATI:")
        print(json.dumps(optimized_results.get('models', {}).get('catboost', {}).get('metrics_test_original', {}), indent=2))
    
    print("\n✅ Ottimizzazione completata!")
    print("\n💾 Risultati salvati in:")
    print(f"   • models/catboost/metrics.json")
    print(f"   • models/summary.json")
    print(f"   • models/validation_results.csv")


if __name__ == "__main__":
    main()
