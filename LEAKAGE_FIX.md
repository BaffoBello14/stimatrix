# 🚨 DATA LEAKAGE: Analisi e Correzione

## 📋 SOMMARIO

**Status**: ⚠️ **LEAKAGE CRITICO IDENTIFICATO E CORRETTO**

I risultati precedenti (R²=0.99, MAPE=2%) erano **troppo ottimistici** a causa di data leakage nelle feature contestuali.

---

## 🔴 IL PROBLEMA

### **Cosa È Successo:**

Nel file `src/preprocessing/pipeline.py`, le feature contestuali erano calcolate **PRIMA** del temporal split:

```python
# ❌ ERRORE: Calcolo statistiche su TUTTO il dataset
df = add_all_contextual_features(df, target_col=target_col, config=config)

# Split DOPO
train_df, val_df, test_df = temporal_split_3way(df, ...)
```

### **Feature Affette da Leakage:**

Tutte le statistiche aggregate che usano il **target** (`AI_Prezzo_Ridistribuito`):

1. **`zone_price_mean`**, **`zone_price_std`**, `zone_price_median`, `zone_price_min`, `zone_price_max`, `zone_price_q25`, `zone_price_q75`
   - Calcolate su train+val+test → il modello vede informazioni sul target del test!

2. **`type_zone_price_mean`**, `type_zone_price_std`, `type_zone_price_median`
   - Idem, per combinazioni tipologia×zona

3. **`temporal_price_mean`**, `temporal_price_median`
   - Trend temporali calcolati includendo il test set

4. **Feature derivate** che dipendono dalle precedenti:
   - `price_vs_zone_mean_ratio`
   - `price_zone_zscore`
   - `price_vs_type_zone_mean`
   - `price_vs_temporal_mean`
   - `prezzo_mq_vs_zone`

### **Impatto:**

- **R² artificialmente alto** (~0.99): Il modello "indovina" perché ha visto statistiche del test
- **MAPE artificialmente basso** (~2%): Idem
- **Overfit mascherato**: Train e test sembrano simili perché entrambi hanno visto le stesse statistiche globali
- **Generalizzazione impossibile**: Su dati nuovi (produzione) le performance crollerebbero

---

## ✅ LA SOLUZIONE

### **Pattern Corretto: Fit/Transform Separato**

```python
# 1. Temporal split PRIMA
train_df, val_df, test_df = temporal_split_3way(df, ...)

# 2. Fit statistiche SOLO su train
stats = fit_contextual_features(train_df, target_col=target_col)

# 3. Transform train/val/test con statistiche del train (frozen)
train_df = transform_contextual_features(train_df, stats)
val_df = transform_contextual_features(val_df, stats)
test_df = transform_contextual_features(test_df, stats)
```

### **Implementazione:**

Ho creato `src/preprocessing/contextual_features_fixed.py` con:

- **`fit_contextual_features(train_df)`**: Calcola statistiche SOLO su train, ritorna dict
- **`transform_contextual_features(df, stats)`**: Applica statistiche pre-calcolate
- **`fit_transform_contextual_features(train, val, test)`**: Wrapper per fit+transform in un colpo

Ho modificato `src/preprocessing/pipeline.py` per:

1. Importare la nuova funzione
2. Spostare la chiamata **DOPO** il temporal split
3. Usare fit/transform separato

---

## 🎯 COSA ASPETTARSI ORA (Risultati Realistici)

### **Aspettative Corrette:**

| Metrica | Prima (con leakage) | Dopo (corretto) | Realistico |
|---------|---------------------|-----------------|------------|
| **R²** | 0.9887 | ??? | **0.75-0.85** |
| **RMSE** | 7,610€ | ??? | **15-25k€** |
| **MAPE** | 2.07% | ??? | **25-35%** |
| **Overfit gap R²** | 0.001 | ??? | **0.05-0.10** |

### **Perché Sarà Diverso:**

- **Test set**: Non ha mai visto statistiche del train
  - `zone_price_mean` per zona X nel test può essere molto diversa dal train
  - Se zona X nel test ha prezzi sistematicamente più alti/bassi, il modello sbaglierà
  
- **Performance più variabile tra gruppi**: Zone/tipologie poco rappresentate nel train avranno errori più alti nel test

- **Overfit visibile**: Train avrà metriche migliori del test (come deve essere!)

---

## 📝 PASSI SUCCESSIVI

### **1. Re-Training con Fix Applicato**

```bash
# Usa la config ottimizzata con il fix
python run_optimization.py
```

**Nota**: Il preprocessing sarà più lento perché deve:
- Split temporale
- Fit contextual features su train
- Transform train/val/test separatamente

### **2. Analisi Risultati Realistici**

Dopo il re-training, analizza:

```python
# Confronto train/test per vedere overfit reale
train_r2 vs test_r2  # Gap realistico: 0.05-0.10
train_mape vs test_mape  # Ratio realistico: 1.5-2.0x

# Performance per gruppo (zone, tipologie)
# Zone con pochi samples nel train avranno errori più alti nel test
```

### **3. Iterazione (se necessario)**

Se i risultati realistici sono insoddisfacenti:

**A. Più dati per feature contestuali:**
- Usa dati storici più lunghi (2020-2021 se disponibili)
- Più samples per zona/tipologia → statistiche più stabili

**B. Smoothing/Regularizzazione delle statistiche:**
- Bayesian averaging per zone con pochi samples
- Fallback a statistiche globali se gruppo troppo piccolo

**C. Feature alternative non-target-based:**
- Indicatori ISTAT (già presenti)
- Distanze da POI (già presenti)
- Feature geometriche

**D. Time-series cross-validation:**
- Multiple temporal splits per validare stabilità
- Walk-forward validation

---

## 🔧 MODIFICHE AI FILE

### **File Creati:**
- `src/preprocessing/contextual_features_fixed.py`: Implementazione leak-free

### **File Modificati:**
- `src/preprocessing/pipeline.py`:
  - Import cambiato: `contextual_features_fixed`
  - Chiamata spostata DOPO split
  - Usa fit_transform pattern

### **File Da Eliminare (opzionale):**
- `src/preprocessing/contextual_features.py`: Vecchia versione con leakage

---

## 📊 VALIDAZIONE DEL FIX

### **Come Verificare Che il Fix Funzioni:**

1. **Statistiche diverse tra split:**
   ```python
   # Dopo preprocessing, confronta:
   train_df['zone_price_mean'].mean()  # Basata su train
   # vs statistiche reali del test
   test_df.groupby('AI_ZonaOmi')['AI_Prezzo_Ridistribuito'].mean()
   # Dovrebbero essere DIVERSE!
   ```

2. **Overfit visibile:**
   ```python
   # Train dovrebbe essere migliore del test
   assert train_r2 > test_r2
   assert train_mape < test_mape
   # Gap realistico: 0.05-0.10 per R², 1.5-2.0x per MAPE
   ```

3. **Log del preprocessing:**
   ```
   ✅ Cerca nel log:
   "Fit contextual features (TRAINING only)"
   "Transform: +XX features"
   
   ❌ NON deve esserci:
   "Aggiunta feature contestuali (zona, tipologia, interazioni)..." 
   PRIMA del temporal split
   ```

---

## 🎓 LEZIONE APPRESA

### **Regola d'Oro per Feature Engineering:**

> **QUALSIASI feature che aggrega informazioni dal target deve essere calcolata SOLO sul train set, poi applicata a val/test come trasformazione frozen.**

### **Altri Casi di Leakage Comuni:**

1. **Target encoding**: Encoding di categoriche usando statistiche del target
   - ✅ Fit su train, transform su val/test
   
2. **Scaling/Normalization**: Min-max, standard scaler, ecc.
   - ✅ Fit mean/std su train, applica a val/test

3. **Imputation**: Riempimento missing con mean/median
   - ✅ Calcola mean/median su train, usa per val/test

4. **Feature selection**: Selezione basata su correlazione/importanza
   - ✅ Seleziona su train, applica mask a val/test

5. **Outlier detection**: Identificazione outliers basata su soglie
   - ✅ Calcola soglie su train, applica a val/test (ma NON rimuovere da test!)

---

## ✅ CHECKLIST ANTI-LEAKAGE

Per ogni nuova feature/trasformazione, chiediti:

- [ ] Usa informazioni dal **target**?
- [ ] Usa statistiche **aggregate** (mean, std, count, ecc.)?
- [ ] Dipende dalla **distribuzione globale** dei dati?

Se **SÌ** a qualcuna:
- [ ] È calcolata **DOPO** il temporal split?
- [ ] È fittata **SOLO su train**?
- [ ] È trasformata **separatamente** su train/val/test?

---

## 📞 CONTATTO

Se hai dubbi o domande sul fix:
- Controlla i log del preprocessing
- Verifica che gli overfit metrics siano realistici
- Analizza le distribuzioni delle feature contestuali per split

**Remember**: Risultati "troppo belli per essere veri" di solito lo sono! 🚨
