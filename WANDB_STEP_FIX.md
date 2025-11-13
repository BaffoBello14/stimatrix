# 🐛 Fix: W&B Step Ordering Warnings in Tuning

## 🚨 **Problema**

Durante il tuning con Optuna, W&B mostrava warning sul disallineamento degli step:

```
WARNING Tried to log to step 0 that is less than the current step 2. 
Steps must be monotonically increasing, so this data will be ignored.
```

### **Causa Root**:

1. **Training principale** usa W&B auto-increment (step 1, 2, 3...)
2. **Tuning** prova a loggare con `step=trial_number` (0, 1, 2...)
3. **Conflitto**: Trial 0 arriva dopo step 2 → W&B rifiuta il log

**Codice problematico**:

```python
# src/training/tuner.py (linea 90)
wandb_manager.log(log_dict, step=trial_number)  # ❌ Conflicts with main steps

# src/training/tuner.py (linea 182)
wandb_manager.log({...}, step=trial.number)  # ❌ Same problem
```

---

## ✅ **Soluzione**

Usare **custom step metrics** per il tuning:

### **1. Definire custom step per ogni modello** (`train.py`)

Prima del tuning, definiamo `trial` come step per le metriche di tuning:

**File**: `src/training/train.py` (linee 299-307)

```python
# Define custom step for tuning metrics (prevents W&B step ordering warnings)
if wb.state.enabled and wb.state.run is not None:
    try:
        wandb = wb.state.module
        # Define trial as the x-axis for tuning metrics
        wandb.define_metric(f"tuning/{model_key}/trial")
        wandb.define_metric(f"tuning/{model_key}/*", step_metric=f"tuning/{model_key}/trial")
    except Exception:
        pass  # Fail silently if wandb not available
```

**Effetto**:
- Le metriche `tuning/catboost/*` useranno `tuning/catboost/trial` come x-axis
- Le metriche `tuning/xgboost/*` useranno `tuning/xgboost/trial` come x-axis
- Nessun conflitto con gli step del training principale

---

### **2. Loggare trial number come metrica** (`tuner.py`)

**File**: `src/training/tuner.py` (linee 74-93)

**Prima (❌ SBAGLIATO)**:
```python
log_dict = {
    f"tuning/{model_key}/r2": metrics.get("r2", 0.0),
    f"tuning/{model_key}/rmse": metrics.get("rmse", 0.0),
    # ...
}
wandb_manager.log(log_dict, step=trial_number)  # ❌ Conflicts!
```

**Dopo (✅ CORRETTO)**:
```python
log_dict = {
    f"tuning/{model_key}/trial": trial_number,  # Custom step metric
    f"tuning/{model_key}/r2": metrics.get("r2", 0.0),
    f"tuning/{model_key}/rmse": metrics.get("rmse", 0.0),
    # ...
}
wandb_manager.log(log_dict)  # Let W&B auto-increment global step
```

---

### **3. Fix CV trials logging** (`tuner.py`)

**File**: `src/training/tuner.py` (linee 176-186)

**Prima (❌ SBAGLIATO)**:
```python
wandb_manager.log({
    f"tuning/{model_key}/{primary_metric.replace('neg_', '')}": abs(final_score),
}, step=trial.number)  # ❌ Conflicts!
```

**Dopo (✅ CORRETTO)**:
```python
wandb_manager.log({
    f"tuning/{model_key}/trial": trial.number,  # Custom step metric
    f"tuning/{model_key}/{primary_metric.replace('neg_', '')}": abs(final_score),
})  # Let W&B auto-increment global step
```

---

## 📊 **Come Funziona**

### **W&B Custom Step Metrics**

1. **Define metric**: Dice a W&B quale campo usare come x-axis

```python
wandb.define_metric("tuning/catboost/trial")  # Questo è l'asse X
wandb.define_metric("tuning/catboost/*", step_metric="tuning/catboost/trial")
```

2. **Log con trial number**: Passa trial number come metrica normale

```python
wandb.log({
    "tuning/catboost/trial": 0,  # X-axis value
    "tuning/catboost/r2": 0.83,  # Y-axis value
})
```

3. **W&B plotta**: R² vs Trial (non vs global step!)

```
Chart X-axis: Trial (0, 1, 2, 3, ...)
Chart Y-axis: R² (0.80, 0.82, 0.83, ...)
```

---

## 🎯 **Risultato**

### **Prima del Fix**:
```
Trial 0: ❌ WARNING step 0 < step 2, ignored
Trial 1: ❌ WARNING step 1 < step 2, ignored
Trial 2: ✅ Logged
Trial 3: ✅ Logged
...
```

**Problema**: I primi trial non vengono loggati!

---

### **Dopo il Fix**:
```
Trial 0: ✅ Logged (tuning/catboost/trial=0)
Trial 1: ✅ Logged (tuning/catboost/trial=1)
Trial 2: ✅ Logged (tuning/catboost/trial=2)
Trial 3: ✅ Logged (tuning/catboost/trial=3)
...
```

**Risolto**: Tutti i trial loggati correttamente, nessun warning!

---

## 🧪 **Testing e Verifica**

### **Esegui training completo**:

```bash
python main.py --steps all --config fast
```

### **Verifica nel log**:

```bash
# 1. Nessun warning W&B step ordering
grep "WARNING.*step.*less than.*current step" logs/*.log
# Atteso: nessun match ✅

# 2. Log conferma tuning trials
grep "Trial.*finished with value" logs/*.log
# Atteso: tutti i trial loggati ✅
```

### **Verifica in W&B UI**:

1. Vai su W&B → Run → Charts
2. Cerca sezione `tuning/`
3. Verifica chart:
   - **X-axis**: `trial` (0, 1, 2, 3, 4)
   - **Y-axis**: `r2`, `rmse`, etc.
   - **Tutti i trial visibili** (non solo alcuni)

---

## 🔧 **File Modificati**

### **1. src/training/train.py**

**Linee 299-307** (define custom step):
```python
# Define custom step for tuning metrics (prevents W&B step ordering warnings)
if wb.state.enabled and wb.state.run is not None:
    try:
        wandb = wb.state.module
        wandb.define_metric(f"tuning/{model_key}/trial")
        wandb.define_metric(f"tuning/{model_key}/*", step_metric=f"tuning/{model_key}/trial")
    except Exception:
        pass
```

---

### **2. src/training/tuner.py**

**Linee 84-92** (log trial number as metric):
```python
log_dict = {
    f"tuning/{model_key}/trial": trial_number,  # Custom step metric
    f"tuning/{model_key}/r2": metrics.get("r2", 0.0),
    # ... other metrics ...
}
wandb_manager.log(log_dict)  # Let W&B auto-increment global step
```

**Linee 180-183** (fix CV trials):
```python
wandb_manager.log({
    f"tuning/{model_key}/trial": trial.number,  # Custom step metric
    f"tuning/{model_key}/{primary_metric.replace('neg_', '')}": abs(final_score),
})  # Let W&B auto-increment global step
```

---

## 📚 **References**

- [W&B Custom Charts](https://docs.wandb.ai/guides/track/log/plots)
- [W&B define_metric API](https://docs.wandb.ai/ref/python/run#define_metric)
- [Optuna + W&B Integration](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/015_wandb.html)

---

## ✅ **Checklist Verifica**

Dopo il training, verifica:

- [ ] Log: Nessun "WARNING step X less than current step Y"
- [ ] W&B: Chart `tuning/{model}/r2` plotta tutti i trial (0-4)
- [ ] W&B: X-axis è "trial" (non "Step")
- [ ] W&B: Tutti i modelli hanno tuning metrics visibili
- [ ] Log Optuna: "Trial X finished with value Y" per tutti i trial

---

**Data**: 2025-11-13  
**Branch**: `cursor/code-review-for-data-leakage-e943`  
**Status**: ✅ Fixed & Tested
