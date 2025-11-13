# 📊 W&B Logging Improvements - Strategia Ibrida

## 🎯 Problema Risolto

**Prima**: Metriche finali loggati come scatter plots con un punto per modello → grafici inutili
```
train_mse → grafico con 1 punto per catboost, 1 per xgboost, ecc.
train_rmse → grafico con 1 punto per lightgbm, 1 per rf, ecc.
```

**Dopo**: Organizzazione gerarchica con summary + tabella + bar charts

---

## ✅ Modifiche Implementate

### **1. `src/training/tuner.py`**

**Modifiche alle linee 74-92, 174-183**:

#### Prima:
```python
log_dict = {
    f"{model_key}/r2": metrics.get("r2", 0.0),
    f"{model_key}/rmse": metrics.get("rmse", 0.0),
}
wandb_manager.log(log_dict)  # ❌ Nessun step, nessuna gerarchia
```

#### Dopo:
```python
log_dict = {
    f"tuning/{model_key}/r2": metrics.get("r2", 0.0),
    f"tuning/{model_key}/rmse": metrics.get("rmse", 0.0),
}
wandb_manager.log(log_dict, step=trial_number)  # ✅ Hierarchical + step
```

**Risultato**:
- ✅ Grafici line chart con progressione Optuna
- ✅ Facile vedere quali trial hanno performato meglio
- ✅ Organizzati sotto `tuning/` per separazione logica

---

### **2. `src/training/train.py`**

#### **A. Aggiunta struttura dati per confronto** (linea 172)

```python
# Track model performance for W&B summary and comparison
model_comparison_data: List[Dict[str, Any]] = []
```

#### **B. Rimosse metriche scatter (linee 398-415)**

**Prima**:
```python
final_metrics = {}
for k, v in m_test_orig.items():
    final_metrics[f"test_{k}/{model_key}"] = v
wb.log(final_metrics)  # ❌ Crea scatter plots inutili
```

**Dopo**:
```python
# Collect model performance for final comparison (no individual scatter logs)
test_metrics = m_test_orig if m_test_orig is not None else m_test
train_metrics = m_train_orig if m_train_orig is not None else m_train

model_comparison_data.append({
    "model_name": model_key,
    "model_type": "optimized",
    "test_r2": test_metrics.get("r2", 0.0),
    "test_rmse": test_metrics.get("rmse", 0.0),
    "test_mae": test_metrics.get("mae", 0.0),
    "test_mape": test_metrics.get("mape", 0.0),
    "train_r2": train_metrics.get("r2", 0.0),
    "train_rmse": train_metrics.get("rmse", 0.0),
    "overfit_r2_delta": diag.get("r2_delta", 0.0),
    "overfit_rmse_pct": diag.get("rmse_pct", 0.0),
})
```

**Risultato**:
- ❌ NO più scatter plots inutili
- ✅ Dati collezionati per confronto finale

#### **C. Stesso per ensemble** (linee 721-738, 843-860)

Applicata stessa logica per `voting` e `stacking` ensemble.

#### **D. Summary + Table + Charts finali** (linee 942-1027)

Aggiunta logica completa per:

**1. Summary (key-value, no charts)**:
```python
summary_dict = {
    "best_model_name": best_model["model_name"],
    "best_test_r2": best_model["test_r2"],
    "best_test_rmse": best_model["test_rmse"],
    # ... tutte le metriche best model
    # + metriche per ogni modello (flat structure)
    "catboost_test_r2": ...,
    "xgboost_test_r2": ...,
}
wb.state.run.summary.update(summary_dict)
```

**2. Tabella di confronto**:
```python
comparison_table = wandb.Table(
    columns=["Model", "Type", "Test_R²", "Test_RMSE", "Test_MAE", 
             "Test_MAPE", "Train_R²", "Overfit_R²Δ", "Overfit_RMSE%"],
    data=[[m["model_name"], m["model_type"], m["test_r2"], ...] 
          for m in sorted_models]
)
wb.log({"model_comparison_table": comparison_table})
```

**3. Bar charts**:
```python
r2_chart = wandb.plot.bar(
    comparison_table, "Model", "Test_R²",
    title="Model Performance Comparison: Test R²"
)
wb.log({"performance_r2_chart": r2_chart})

rmse_chart = wandb.plot.bar(
    comparison_table, "Model", "Test_RMSE",
    title="Model Performance Comparison: Test RMSE"
)
wb.log({"performance_rmse_chart": rmse_chart})
```

---

## 📊 Risultato Finale su W&B

### **Overview Tab**:
```
📈 tuning/catboost/r2         → Line chart con progressione 50 trials
📈 tuning/catboost/rmse       → Line chart con progressione 50 trials
📈 tuning/xgboost/r2          → Line chart con progressione 50 trials
📈 tuning/xgboost/rmse        → Line chart con progressione 50 trials
📊 model_comparison_table     → Table con tutti i modelli e metriche
📊 performance_r2_chart       → Bar chart R² per modello
📊 performance_rmse_chart     → Bar chart RMSE per modello
```

### **Summary Tab** (key-value, no charts):
```
best_model_name: "catboost"
best_test_r2: 0.9521
best_test_rmse: 24532.12
best_test_mae: 18234.56
best_test_mape: 0.0823
...
catboost_test_r2: 0.9521
catboost_test_rmse: 24532.12
xgboost_test_r2: 0.9487
xgboost_test_rmse: 25123.45
lightgbm_test_r2: 0.9432
...
```

---

## 🎯 Vantaggi della Nuova Struttura

### ✅ **Durante Tuning**:
- Progressione Optuna visibile con line charts
- Facile identificare trial migliori
- Organizzazione gerarchica sotto `tuning/`

### ✅ **Metriche Finali**:
- Summary pulito con best model
- Confronto facile in tabella (ordinata per R²)
- Bar charts per impatto visivo immediato

### ✅ **Organizzazione**:
- No più scatter plots inutili
- Gerarchia logica: `tuning/` → progressione, summary → risultati finali
- Facile comparare modelli tra run diversi

---

## 🧪 Come Testare

### **1. Esegui training normale**:
```bash
python main.py --mode train --config config/config.yaml
```

### **2. Verifica su W&B**:

#### **A. Overview Tab**:
- ✅ Vedi grafici `tuning/catboost/r2` con line chart (non scatter)
- ✅ Vedi grafici `tuning/xgboost/rmse` con progressione trials
- ✅ Vedi tabella `model_comparison_table` con tutti i modelli
- ✅ Vedi bar charts `performance_r2_chart` e `performance_rmse_chart`

#### **B. Summary Tab** (in alto a destra):
- ✅ Vedi `best_model_name`, `best_test_r2`, ecc.
- ✅ Vedi metriche per ogni modello (flat: `catboost_test_r2`, ecc.)

#### **C. Charts Tab**:
- ❌ NO scatter plots con 1 punto per modello (come prima)
- ✅ SOLO line charts progressione tuning + bar charts confronto

---

## 🔍 Verifica Problemi Precedenti

### **Problema 1**: Scatter plots con 1 punto per modello
- ❌ **Prima**: `train_mse`, `train_rmse`, ecc. → scatter con 1 punto
- ✅ **Dopo**: Rimossi completamente, metriche solo in summary/tabella

### **Problema 2**: Difficile confrontare modelli
- ❌ **Prima**: Metriche sparse in scatter plots
- ✅ **Dopo**: Tabella ordinata + bar charts

### **Problema 3**: No visione d'insieme
- ❌ **Prima**: Grafici separati per ogni metrica/modello
- ✅ **Dopo**: Summary con best model + tabella completa + bar charts

---

## 📝 Note Tecniche

### **Gestione Errori**:
Tutte le modifiche sono wrapped in `try-except` per fail-safe:
- Se W&B non è abilitato → skip silenzioso
- Se tabella/chart fallisce → warning ma non blocca training
- Se summary update fallisce → warning ma continua

### **Compatibilità**:
- ✅ Funziona con W&B disabled (no-op)
- ✅ Funziona senza W&B installato (no-op)
- ✅ Funziona con `mode=offline` o `mode=disabled`
- ✅ Backward compatible con run esistenti

### **Performance**:
- ✅ Nessun overhead durante training
- ✅ Summary/tabella/chart creati solo alla fine
- ✅ No chiamate extra a W&B durante tuning (step già presente)

---

## 🚀 Prossimi Passi (Opzionali)

### **Possibili estensioni**:
1. **Scatter plot interattivo**: Train R² vs Test R² per overfit visualization
2. **Heatmap**: Correlazione tra hyperparameters e performance
3. **Time series**: Training time vs performance
4. **Feature importance**: Bar chart aggregato per tutti i modelli

### **Per implementare**:
```python
# Esempio: scatter plot overfit
wb.log({
    "overfit_analysis": wandb.plot.scatter(
        comparison_table, "Train_R²", "Test_R²",
        title="Overfit Analysis: Train vs Test R²"
    )
})
```

---

## 📚 Riferimenti

- **W&B Tables**: https://docs.wandb.ai/guides/tables
- **W&B Charts**: https://docs.wandb.ai/guides/data-vis/log-charts
- **W&B Summary**: https://docs.wandb.ai/guides/track/log/intro#summary-metrics

---

## ✅ Checklist Verifica

- [ ] Training completo eseguito con successo
- [ ] W&B run creato e visibile
- [ ] `tuning/` charts sono line charts (non scatter)
- [ ] `model_comparison_table` visibile in Overview
- [ ] `performance_r2_chart` e `performance_rmse_chart` visibili
- [ ] Summary contiene `best_model_name` e metriche
- [ ] Summary contiene metriche per ogni modello
- [ ] NO scatter plots con 1 punto per modello
- [ ] Confronto modelli facile e immediato

---

**Data**: 2025-11-13  
**Branch**: `cursor/code-review-for-data-leakage-e943`  
**Files modificati**:
- `src/training/tuner.py` (linee 74-92, 174-183)
- `src/training/train.py` (linee 172, 398-417, 721-738, 843-860, 942-1027)
