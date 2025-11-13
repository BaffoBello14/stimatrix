# 🔍 ANALISI force_reload e Logging - Proposta Miglioramenti

**Data**: 2025-11-13

---

## 📋 PROBLEMA 1: `force_reload` NON USATO

### **Situazione Attuale**

**Config** (`config/config.yaml`):
```yaml
execution:
  force_reload: true
```

**Main.py** (imposta il valore):
```python
force_reload_cfg = bool(execution_cfg.get("force_reload", False))
if args.force_reload:
    force_reload_cfg = True
execution_cfg["force_reload"] = force_reload_cfg
```

**Ma poi... NON viene mai controllato!**

```bash
$ grep -rn "force_reload" src/
# → Nessun risultato!
```

**Conclusione**: ❌ Parametro inutilizzato, va tolto o implementato.

---

### **PROPOSTA 1: Rimuovere force_reload**

**Motivo**: Se non serve, meglio toglierlo per non confondere.

**Modifiche**:
1. Rimuovere da `config/config.yaml` e `config/config_fast.yaml`
2. Rimuovere da `main.py` la logica di impostazione
3. Rimuovere l'argomento `--force-reload` da CLI

**PRO**:
- ✅ Config più pulito
- ✅ Meno confusione
- ✅ Meno codice morto

**CONTRO**:
- ❌ Se in futuro servirà, va re-implementato

---

### **PROPOSTA 2: Implementare force_reload**

**Scenario d'uso**: Rigenerare output anche se già esistono (skip cache).

**Dove usarlo**:
1. **Preprocessing**: Rigenera preprocessed anche se esiste
2. **Training**: Riaddestra modelli anche se model.pkl esiste
3. **Evaluation**: Rigenera report anche se esistono

**Modifiche**:

```python
# src/preprocessing/pipeline.py
def run_preprocessing(config: Dict[str, Any]) -> None:
    pre_dir = Path(config["paths"]["preprocessed_data"])
    force_reload = config.get("execution", {}).get("force_reload", False)
    
    # Check if already done
    if not force_reload and (pre_dir / "X_train_tree.parquet").exists():
        logger.info("Preprocessing già fatto. Usa force_reload=true per rigenerare.")
        return
    
    # ... resto del preprocessing ...

# src/training/train.py
def run_training(config: Dict[str, Any]) -> None:
    models_dir = Path(config["paths"]["models_dir"])
    force_reload = config.get("execution", {}).get("force_reload", False)
    
    # Check models già addestrati
    if not force_reload:
        existing_models = list(models_dir.glob("*/model.pkl"))
        if existing_models:
            logger.info(f"Trovati {len(existing_models)} modelli. Usa force_reload=true per riaddestrare.")
            return
    
    # ... resto del training ...
```

**PRO**:
- ✅ Utile per iterazioni rapide
- ✅ Risparmia tempo (skip step già fatti)

**CONTRO**:
- ❌ Rischio di usare dati vecchi per sbaglio
- ❌ Più logica da gestire

---

### **🎯 RACCOMANDAZIONE PERSONALE**

**Rimuovere `force_reload`** (Proposta 1)

**Motivo**: 
- Nella pratica, raramente serve fare "skip intelligente"
- È più pericoloso (rischio dati vecchi) che utile
- Se serve evitare ricalcolo, basta non eseguire quello step: `python main.py --steps training evaluation`
- Più semplice = meglio

Se vuoi implementarlo, suggerisco **solo per preprocessing** (quello più lento).

---

## 📊 PROBLEMA 2: LOGGING RISULTATI POCO UTILE

### **Situazione Attuale**

**Come vengono loggati i risultati**:

1. **Durante tuning**: Nessun log (solo Optuna interno)
2. **Dopo tuning modello**: Log metriche una volta
   ```python
   wb.log_prefixed_metrics(f"model/{model_key}", {
       "train_r2": 0.85, 
       "test_r2": 0.73,
       ...
   })
   ```
3. **Fine training**: Salva tutto in `summary.json`

**Problemi**:
- ❌ **Non vedi progresso durante tuning** (es. trial 50/150 come sta andando?)
- ❌ **Non puoi confrontare modelli in tempo reale** (devi aspettare la fine)
- ❌ **Grafici W&B poco utili** (1 punto per modello invece di curve)
- ❌ **Non vedi overfitting durante il training** (solo alla fine)

**Esempio W&B attuale**:
```
model/catboost/train_r2: 0.95  (1 punto)
model/catboost/test_r2: 0.73   (1 punto)
model/xgboost/train_r2: 0.94   (1 punto)
model/xgboost/test_r2: 0.71    (1 punto)
```

→ Non vedi la **progressione** del tuning!

---

### **PROPOSTA: Logging Progressivo Multi-Livello**

#### **Livello 1: Step-by-Step (Già implementabile con wandb.log)**

Loggare ogni **step principale** con metriche aggregate:

```python
# Dopo preprocessing
wb.log({
    "step": "preprocessing",
    "dataset_size": len(train_df),
    "n_features": len(features),
    "n_outliers_removed": n_outliers,
}, step=1)

# Dopo ogni modello
wb.log({
    "step": f"training/{model_key}",
    "best_trial_r2": best_r2,
    "best_trial_mape": best_mape,
    "n_trials": 150,
}, step=2)
```

**Risultato W&B**:
```
Step 1: preprocessing → dataset_size=10000, n_features=120
Step 2: training/catboost → best_r2=0.73, best_mape=0.45
Step 3: training/xgboost → best_r2=0.71, best_mape=0.48
...
```

✅ Vedi progresso lineare della pipeline!

---

#### **Livello 2: Per-Trial Logging (Durante Tuning)**

Loggare **ogni trial di Optuna** per vedere curve di apprendimento:

**Modifica `src/training/tuner.py`**:

```python
def tune_model(...) -> Tuple[RegressorMixin, Dict[str, Any], float]:
    def objective(trial: Trial) -> float:
        # ... (hyperparameter sampling) ...
        
        # Fit model
        estimator.fit(X_train, y_train)
        
        # Evaluate
        y_pred = estimator.predict(X_val)
        primary_value = select_primary_value(metrics, primary_metric)
        
        # 🔥 NUOVO: Log questo trial a W&B
        wandb_manager.log({
            f"tuning/{model_key}/trial_number": trial.number,
            f"tuning/{model_key}/{primary_metric}": primary_value,
            f"tuning/{model_key}/val_r2": metrics.get("r2", 0),
            f"tuning/{model_key}/val_rmse": metrics.get("rmse", 0),
            # Log anche hyperparameters provati
            **{f"tuning/{model_key}/hp_{k}": v for k, v in trial.params.items()},
        }, step=trial.number)  # step = trial number!
        
        return primary_value
    
    study = optuna.create_study(...)
    study.optimize(objective, n_trials=n_trials)
    # ...
```

**Risultato W&B**:

Grafici per ogni modello:
- **X-axis**: Trial number (0→150)
- **Y-axis**: MAPE / R² / RMSE
- **Lines**: Una per ogni modello

```
tuning/catboost/neg_mape:
  Trial 0:   -0.52
  Trial 1:   -0.49
  Trial 2:   -0.48
  ...
  Trial 150: -0.42  (migliore!)

tuning/xgboost/neg_mape:
  Trial 0:   -0.55
  Trial 1:   -0.51
  ...
  Trial 150: -0.44
```

✅ **Vedi convergenza del tuning in tempo reale!**  
✅ **Confronti modelli durante il training!**  
✅ **Vedi se tuning si è stabilizzato o serve più trial!**

---

#### **Livello 3: Per-Model Summary (Dopo Tuning)**

Dopo il tuning di ogni modello, loggare **summary dettagliato**:

```python
# src/training/train.py (dopo tuning)
wb.log({
    f"final/{model_key}/train_r2": m_train["r2"],
    f"final/{model_key}/train_rmse": m_train["rmse"],
    f"final/{model_key}/train_mape": m_train["mape"],
    
    f"final/{model_key}/test_r2": m_test["r2"],
    f"final/{model_key}/test_rmse": m_test["rmse"],
    f"final/{model_key}/test_mape": m_test["mape"],
    
    # Original scale
    f"final/{model_key}/test_r2_orig": m_test_orig["r2"],
    f"final/{model_key}/test_rmse_orig": m_test_orig["rmse"],
    f"final/{model_key}/test_mape_orig": m_test_orig["mape"],
    
    # Overfit diagnostics
    f"final/{model_key}/overfit_gap_r2": diag["gap_r2"],
    f"final/{model_key}/overfit_ratio_rmse": diag["ratio_rmse"],
}, step=current_model_index)
```

**Risultato W&B**:

**Tabella comparativa finale**:
```
Model      | test_r2 | test_rmse | test_mape | overfit_gap_r2
-----------|---------|-----------|-----------|---------------
CatBoost   | 0.736   | 36768€    | 58.1%     | 0.117
XGBoost    | 0.693   | 39670€    | 61.1%     | 0.144
LightGBM   | 0.686   | 40126€    | 66.4%     | 0.092
RF         | 0.693   | 39711€    | 62.9%     | 0.100
GBR        | 0.747   | 36008€    | 60.0%     | 0.138
HGBT       | 0.719   | 37971€    | 62.9%     | 0.132
```

✅ **Tabella finale per confronto veloce!**

---

### **Organizzazione Logging W&B**

**Struttura proposta**:

```
📊 W&B Dashboard

📈 Charts:
  ├─ tuning/
  │   ├─ catboost/neg_mape (curva 150 trial)
  │   ├─ catboost/val_r2 (curva 150 trial)
  │   ├─ xgboost/neg_mape (curva 150 trial)
  │   ├─ xgboost/val_r2 (curva 150 trial)
  │   ├─ lightgbm/neg_mape (curva 150 trial)
  │   └─ ...
  │
  ├─ final/
  │   ├─ test_r2 (bar chart per modello)
  │   ├─ test_rmse_orig (bar chart per modello)
  │   ├─ test_mape_orig (bar chart per modello)
  │   └─ overfit_gap_r2 (bar chart per modello)
  │
  └─ step/
      ├─ preprocessing/dataset_size
      ├─ preprocessing/n_features
      └─ training/models_completed

📊 Tables:
  └─ final_summary (tabella comparativa)
```

**Benefici**:
- ✅ **Tuning progress**: Vedi ogni trial in tempo reale
- ✅ **Model comparison**: Confronta modelli durante training
- ✅ **Final summary**: Tabella finale per quick comparison
- ✅ **Step tracking**: Vedi progresso pipeline

---

### **🎯 RACCOMANDAZIONE PERSONALE**

**Implementare Livello 2 + Livello 3** (Per-Trial + Summary)

**Motivo**:
- **Livello 1** (step-by-step) è troppo grossolano
- **Livello 2** (per-trial) ti permette di:
  - Vedere convergenza tuning
  - Stoppare early se stabilizzato
  - Confrontare modelli in tempo reale
- **Livello 3** (summary) è essenziale per confronto finale

**Implementazione**:
1. Modificare `src/training/tuner.py` per loggare ogni trial
2. Modificare `src/training/train.py` per loggare summary finale
3. Aggiungere parametro `step=` incrementale per ordinamento cronologico

**Effort**: ~2 ore di lavoro

---

## 📝 RIASSUNTO RACCOMANDAZIONI

### **1. force_reload**
**→ RIMUOVERE** (non serve, complica solo)

### **2. Logging risultati**
**→ IMPLEMENTARE** per-trial logging + final summary

**Vantaggi**:
- ✅ Vedi progresso in tempo reale
- ✅ Confronti modelli durante training
- ✅ Grafici W&B utili (curve invece di punti)
- ✅ Puoi stoppare early se converge
- ✅ Debug più facile (vedi se qualcosa va storto)

---

## 💬 DOMANDE PER TE

Prima di procedere:

1. **force_reload**: Rimuovere o implementare (solo preprocessing)?
2. **Logging**: Va bene per-trial + final summary?
3. **W&B structure**: Ti piace la struttura `tuning/`, `final/`, `step/`?
4. **Step incrementale**: Usare `step=trial.number` per ordinamento cronologico va bene?
5. **Altro**: Vuoi loggare anche altro (es. hyperparameters per trial, feature importance progressiva)?

**Fammi sapere cosa preferisci e procedo con l'implementazione!** 🚀
