# 🚀 Stimatrix ML Pipeline - Funzionalità Avanzate

Questa documentazione descrive le funzionalità enterprise-grade integrate nella pipeline per renderla robusta e production-ready.

## 📋 Panoramica Integrazione

La pipeline è stata arricchita con sistemi avanzati che forniscono:
- **Robustezza operativa** con fallback automatici
- **Quality assurance** con validazione data leakage  
- **Monitoring completo** con tracking evoluzione
- **Interpretabilità avanzata** con feature importance multi-metodo
- **Evaluation sofisticata** per target trasformati

## 🔍 Quality Checks System

### Funzionalità
Il sistema di quality checks valida automaticamente:

1. **Temporal Leakage Detection**
   - Verifica sovrapposizioni temporali tra train/validation/test
   - Controlla gap temporali minimi tra splits
   - Valida integrità ordinamento cronologico

2. **Target Leakage Detection**  
   - Rileva features che contengono informazioni del target
   - Pattern matching per colonne sospette (prezzo, valore, etc.)
   - Identificazione ID univoci potenzialmente problematici

3. **Category Distribution Monitoring**
   - Calcola drift distribuzione categorie tra splits
   - Total Variation Distance per quantificare cambiamenti
   - Alert per drift superiori a soglie configurabili

4. **Feature Stability Validation**
   - Monitora stabilità features durante preprocessing
   - Rileva rimozioni massive di colonne
   - Verifica consistenza tipi dati

### Configurazione
```yaml
quality_checks:
  check_temporal_leakage: true
  check_target_leakage: true
  check_category_distribution: true
  max_category_drift: 0.05          # 5% max drift
  min_temporal_gap_months: 1        # 1 mese gap minimo
```

### Utilizzo
```python
from validation.quality_checks import QualityChecker

checker = QualityChecker(config)
results = checker.run_all_checks(X_train, X_val, X_test, y_train, y_val, y_test)

if results['overall_status'] == 'CRITICAL_ERRORS':
    print("⚠️ Errori critici rilevati - verifica data leakage")
```

## 📊 Pipeline Tracking System

### Funzionalità
Il tracker monitora l'intera evoluzione della pipeline:

1. **Dataset Evolution Tracking**
   - Shape (righe × colonne) per ogni step
   - Memoria utilizzata e variazioni
   - Colonne aggiunte/rimosse con dettagli

2. **Performance Monitoring**
   - Timing di ogni step con breakdown
   - Identificazione bottleneck automatica
   - Score efficienza per step

3. **Feature Engineering Tracking**
   - Risultati feature extraction per tipo
   - Outlier detection con statistiche per categoria
   - Encoding results con metodi utilizzati

4. **Automated Reporting**
   - Report JSON/CSV/Excel automatici
   - Summary leggibili con raccomandazioni
   - Export configurabile per analisi esterne

### Configurazione
```yaml
tracking:
  enabled: true
  save_intermediate: false          # Snapshot intermedi

monitoring:
  alerts:
    max_step_duration_minutes: 30
    max_memory_usage_mb: 2000
    min_samples_threshold: 1000
```

### Utilizzo
```python
from preprocessing.pipeline_tracker import PipelineTracker

tracker = PipelineTracker(config)
tracker.track_step_start('preprocessing')

# ... operazioni preprocessing ...

tracker.track_step_completion('preprocessing', df_before, df_after, step_info)
final_report = tracker.generate_comprehensive_report()
```

## 🛡️ Robust Operations System

### Funzionalità
Operazioni fail-safe che non interrompono mai la pipeline:

1. **Safe Column Operations**
   - Gestione automatica colonne mancanti
   - Fallback functions per operazioni fallite
   - Logging dettagliato successi/fallimenti

2. **Intelligent Column Analysis**
   - Detection colonne costanti con statistiche
   - Analisi correlazioni con gestione errori
   - Pattern matching per identificazione colonne

3. **DataFrame Validation**
   - Controlli strutturali automatici
   - Rilevamento tipi dati misti
   - Validazione memoria e dimensioni

### Esempi Utilizzo
```python
from utils.robust_operations import RobustDataOperations, RobustColumnAnalyzer

# Rimozione sicura colonne
df_clean, info = RobustDataOperations.remove_columns_safe(
    df, ['col1', 'missing_col'], "PULIZIA COLONNE"
)

# Analisi colonne costanti
constant_cols, stats = RobustColumnAnalyzer.find_constant_columns(
    df, threshold=0.95, exclude_columns=['target']
)
```

## ⏰ Temporal Utilities Advanced

### Funzionalità
Gestione sofisticata dati temporali immobiliari:

1. **Smart Temporal Sorting**
   - Chiavi composite anno/mese/giorno/ora
   - Gestione periodi mancanti
   - Ordinamento con validazione integrità

2. **Advanced Temporal Splitting**
   - Split con validazione anti-leakage
   - Gap temporali configurabili
   - Logging dettagliato range date

3. **Temporal Feature Engineering**
   - Quarter, semester, season automatici
   - Feature cicliche (sin/cos) per mese
   - Trend temporali e indicatori periodo

4. **Anomaly Detection**
   - Rilevamento gap temporali
   - Periodi con troppi/pochi dati
   - Target anomali per periodo

### Esempi Utilizzo
```python
from utils.temporal_advanced import AdvancedTemporalUtils, TemporalSplitter

# Ordinamento temporale
df_sorted = AdvancedTemporalUtils.temporal_sort_dataframe(
    df, 'A_AnnoStipula', 'A_MeseStipula'
)

# Split con validazione
train, val, test, info = TemporalSplitter.split_temporal_with_validation(
    df, 'A_AnnoStipula', 'A_MeseStipula', 0.7, 0.15, 0.15
)

# Feature temporali
df_enhanced = AdvancedTemporalUtils.create_temporal_features(
    df, 'A_AnnoStipula', 'A_MeseStipula'
)
```

## 🧠 Feature Importance Multi-Method

### Funzionalità
Sistema completo per interpretabilità modelli:

1. **Multiple Methods Integration**
   - Built-in importance (feature_importances_, coef_)
   - Permutation importance con performance degradation
   - SHAP values con explainer ottimizzati per tipo modello

2. **Consensus Analysis**
   - Combinazione weighted di tutti i metodi
   - Stability metrics tra modelli
   - Top features consistency analysis

3. **Performance Optimization**
   - Campionamento intelligente per SHAP
   - Gestione ensemble models con metodi velocizzati
   - Fallback automatici per metodi non supportati

### Configurazione
```yaml
training:
  shap:
    enabled: true
    sample_size: 500                # Ridotto per performance
    max_display: 20
    save_plots: true

feature_importance:
  enable_shap: true
  enable_permutation: true
  max_features_plot: 20
```

### Utilizzo
```python
from training.feature_importance_advanced import AdvancedFeatureImportance

fi_system = AdvancedFeatureImportance(config)
comprehensive_results = fi_system.calculate_comprehensive_importance(
    trained_models, X_train, X_test, y_test, feature_names
)

# Consensus importance
consensus = comprehensive_results['consensus']['overall_consensus']
top_features = sorted(consensus.items(), key=lambda x: x[1], reverse=True)[:10]
```

## 📈 Evaluation Dual-Scale

### Funzionalità
Valutazione sofisticata per modelli con target trasformati:

1. **Dual-Scale Metrics**
   - Metriche su scala trasformata (log, box-cox)
   - Metriche su scala originale (interpretabili business)
   - Inverse transform automatico con parametri

2. **Advanced Model Comparison**
   - Rankings multi-metrica
   - Consensus best model identification
   - Performance visualization automatica

3. **Residual Analysis**
   - Test normalità residui (Shapiro-Wilk, Jarque-Bera)
   - Heteroscedasticity detection
   - Outlier analysis con soglie IQR

### Configurazione
```yaml
evaluation:
  metrics: ['mae', 'mse', 'rmse', 'r2', 'mape']
  dual_scale_evaluation: true
  residual_analysis: true
  save_plots: true
```

### Utilizzo
```python
from training.evaluation_advanced import MultiScaleEvaluation

evaluator = MultiScaleEvaluation(config)
results = evaluator.evaluate_multiple_models_dual_scale(
    models, X_test, y_test_transformed, y_test_original, transform_info
)

# Best models per scala
best_rmse_original = results['rankings']['original_scale']['rmse'][0]
best_r2_transformed = results['rankings']['transformed_scale']['r2'][0]
```

## ⚙️ Smart Configuration Manager

### Funzionalità
Gestione intelligente configurazioni con adattamento automatico:

1. **Automatic Column Resolution**
   - Target columns con pattern matching
   - Temporal columns detection
   - Categorical columns analysis per strategie

2. **Dataset-Adaptive Optimization**
   - Configurazione ottimizzata per dimensioni dataset
   - Riduzione automatica parametri per dataset grandi
   - Fallback per dataset piccoli

3. **Robust Validation**
   - Validazione configurazione con error recovery
   - Default intelligenti per sezioni mancanti
   - Dependency resolution automatica

### Esempi Utilizzo
```python
from utils.smart_config import SmartConfigurationManager

# Inizializzazione con validazione
smart_config = SmartConfigurationManager('config/config.yaml')

# Risoluzione automatica colonne
target_info = smart_config.resolve_target_columns(df)
temporal_info = smart_config.resolve_temporal_columns(df)

# Ottimizzazione per dataset
optimized_config = smart_config.optimize_config_for_dataset(df)
```

## 🚀 Comandi Makefile

Il Makefile fornisce automazione completa per sviluppo:

```bash
# Setup e installazione
make install              # Dipendenze base
make install-dev          # Dipendenze sviluppo + tools
make setup-dev           # Setup completo con pre-commit

# Testing
make test                # Test completi con coverage
make test-quality        # Test quality checks specifici
make test-fast           # Solo test veloci
make test-integration    # Test integrazione

# Code quality
make lint                # Linting con flake8
make format              # Formattazione con black/isort
make type-check          # Type checking con mypy

# Pipeline execution
make run-basic           # Pipeline base
make run-full            # Pipeline completa
make run-with-quality-checks # Con quality checks forzati

# Diagnostica
make diagnose            # Diagnostica completa sistema
make check-imports       # Verifica import moduli
make validate-config     # Validazione configurazione
```

## 📋 Migration Guide

### Da Configurazione Standard a Enhanced

1. **Copia configurazione**
   ```bash
   cp config/config.yaml config/config_backup.yaml
   cp config/config_enhanced.yaml config/config.yaml
   ```

2. **Abilita funzionalità gradualmente**
   ```yaml
   # Inizia con quality checks
   quality_checks:
     check_temporal_leakage: true
   
   # Aggiungi tracking
   tracking:
     enabled: true
   
   # Abilita feature importance avanzata
   training:
     shap:
       enabled: true
       sample_size: 200  # Inizia con valore basso
   ```

3. **Testa integrazione**
   ```bash
   make diagnose
   make test-quality
   python main.py --validate-config
   ```

### Backward Compatibility

Tutte le funzionalità sono **backward compatible**:
- Configurazioni esistenti continuano a funzionare
- Nuove funzionalità sono opt-in (disabilitate di default)
- Fallback automatici per configurazioni incomplete
- Output format invariato per compatibilità downstream

## 🎯 Best Practices

### Per Produzione
1. **Abilita sempre quality checks**
   ```yaml
   quality_checks:
     check_temporal_leakage: true
     check_target_leakage: true
   ```

2. **Usa tracking per monitoring**
   ```yaml
   tracking:
     enabled: true
   monitoring:
     alerts:
       max_memory_usage_mb: 4000  # Adatta al tuo sistema
   ```

3. **Ottimizza SHAP per performance**
   ```yaml
   training:
     shap:
       sample_size: 200           # Riduci per dataset grandi
       save_values: false         # Non salvare raw values
   ```

### Per Development
1. **Usa configurazione enhanced per test completi**
2. **Abilita debug mode per troubleshooting**
3. **Controlla tracking reports per ottimizzazioni**
4. **Usa Makefile per automazione**

## 🔧 Troubleshooting Avanzato

### Quality Checks Failures
- **Temporal leakage**: Verifica configurazione split temporale
- **Target leakage**: Rimuovi colonne sospette da surface.drop_columns
- **Category drift**: Controlla bilanciamento dataset

### Performance Issues  
- **Memory errors**: Riduci shap.sample_size e disabilita PCA
- **Slow training**: Disabilita modelli lenti (catboost) e riduci trials
- **Large datasets**: Abilita tracking.save_intermediate: false

### Configuration Issues
- **Missing columns**: Smart config risolve automaticamente
- **Invalid parameters**: Fallback automatici applicati
- **Dependency conflicts**: Validation automatica con warning

---

**Le funzionalità avanzate rendono Stimatrix ML Pipeline una soluzione enterprise-ready per predizione prezzi immobiliari con robustezza operativa e quality assurance automatica.**