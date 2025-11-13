#!/usr/bin/env python3
"""
Script per re-training con fix al data leakage.

Questo script esegue preprocessing e training con le feature contestuali
calcolate CORRETTAMENTE (fit su train, transform su val/test).
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from utils.logger import setup_logger, get_logger
from utils.config import load_config
from utils.io import save_json
from preprocessing.pipeline import run_preprocessing
from training.train import run_training


def main():
    print("=" * 80)
    print("🔧 RE-TRAINING CON FIX DATA LEAKAGE")
    print("=" * 80)
    print()
    print("Modifiche applicate:")
    print("  ✅ Feature contestuali calcolate DOPO temporal split")
    print("  ✅ Fit SOLO su train, transform su val/test")
    print("  ✅ NO leakage di informazioni dal test al train")
    print()
    print("Aspettative:")
    print("  • R² più basso (~0.75-0.85 invece di 0.99)")
    print("  • RMSE più alto (~15-25k€ invece di 7k€)")
    print("  • MAPE più alto (~25-35% invece di 2%)")
    print("  • Overfit VISIBILE (train migliore di test)")
    print("=" * 80)
    print()
    
    # Load config
    config_path = "config/config_optimized.yaml"
    
    # Setup logger
    setup_logger(config_path)
    logger = get_logger(__name__)
    
    logger.info(f"📄 Loading config: {config_path}")
    config = load_config(config_path)
    
    # ==========================================
    # STEP 1: PREPROCESSING (with LEAK-FREE contextual features)
    # ==========================================
    print()
    print("-" * 80)
    print("STEP 1: PREPROCESSING (LEAK-FREE)")
    print("-" * 80)
    
    try:
        run_preprocessing(config)
        logger.info("✅ Preprocessing completed (LEAK-FREE)")
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        raise
    
    # ==========================================
    # STEP 2: TRAINING
    # ==========================================
    print()
    print("-" * 80)
    print("STEP 2: TRAINING")
    print("-" * 80)
    
    try:
        results = run_training(config)
        logger.info("✅ Training completed")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    # ==========================================
    # STEP 3: SUMMARY
    # ==========================================
    print()
    print("=" * 80)
    print("📊 RISULTATI REALISTICI (senza leakage)")
    print("=" * 80)
    
    # Extract best model metrics
    if "models" in results:
        best_model = None
        best_mape = float('inf')
        
        for model_name, model_results in results["models"].items():
            if "metrics_test_original" in model_results:
                mape = model_results["metrics_test_original"].get("mape_floor", float('inf'))
                if mape < best_mape:
                    best_mape = mape
                    best_model = model_name
        
        if best_model:
            metrics = results["models"][best_model]["metrics_test_original"]
            overfit = results["models"][best_model]["overfit"]
            
            print(f"\n🏆 Best Model: {best_model.upper()}")
            print(f"   R²:          {metrics['r2']:.4f}")
            print(f"   RMSE:        {metrics['rmse']:,.0f} €")
            print(f"   MAE:         {metrics['mae']:,.0f} €")
            print(f"   MAPE:        {metrics['mape']*100:.2f}%")
            print(f"   MAPE floor:  {metrics['mape_floor']*100:.2f}%")
            print()
            print(f"   Overfit (gap R²):   {overfit['gap_r2']:.4f}")
            print(f"   Overfit (ratio RMSE): {overfit['ratio_rmse']:.2f}x")
            
            # Valutazione risultati
            print()
            print("📈 Valutazione:")
            
            # R²
            if metrics['r2'] >= 0.85:
                print("   ✅ R² eccellente (≥0.85)")
            elif metrics['r2'] >= 0.75:
                print("   ✅ R² buono (≥0.75)")
            elif metrics['r2'] >= 0.65:
                print("   ⚠️  R² accettabile (≥0.65)")
            else:
                print("   ❌ R² basso (<0.65) - necessario miglioramento")
            
            # MAPE
            mape_pct = metrics['mape_floor'] * 100
            if mape_pct <= 20:
                print(f"   ✅ MAPE eccellente (≤20%): {mape_pct:.1f}%")
            elif mape_pct <= 35:
                print(f"   ✅ MAPE buono (≤35%): {mape_pct:.1f}%")
            elif mape_pct <= 50:
                print(f"   ⚠️  MAPE accettabile (≤50%): {mape_pct:.1f}%")
            else:
                print(f"   ❌ MAPE alto (>{mape_pct:.1f}%) - necessario miglioramento")
            
            # Overfit
            if overfit['gap_r2'] <= 0.05:
                print(f"   ✅ Overfit minimo (gap R²={overfit['gap_r2']:.4f})")
            elif overfit['gap_r2'] <= 0.10:
                print(f"   ✅ Overfit accettabile (gap R²={overfit['gap_r2']:.4f})")
            elif overfit['gap_r2'] <= 0.15:
                print(f"   ⚠️  Overfit moderato (gap R²={overfit['gap_r2']:.4f})")
            else:
                print(f"   ❌ Overfit alto (gap R²={overfit['gap_r2']:.4f})")
            
            print()
            print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
