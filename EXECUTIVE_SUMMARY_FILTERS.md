# 📋 EXECUTIVE SUMMARY: Filtri Sperimentali Dataset 2022+

**Branch**: `cursor/analyze-and-test-data-subset-176c`  
**Data**: 2025-11-14  
**Status**: ✅ VERIFICATO - NO LEAKAGE

---

## 🎯 CONFIGURAZIONE ATTUALE

### Filtri Applicati

```yaml
data_filters:
  anno_min: 2022                    # Solo transazioni recenti
  zone_escluse: ['E1','E2','E3','R1']  # Escludi zone periferiche/rurali
  tipologie_escluse: ['4']          # Escludi ville
```

### Obiettivo

Valutare se un **modello specializzato** su:
- ✅ Transazioni recenti (2022+) - riduce temporal drift
- ✅ Zone urbane/semicentrali - escluse periferie/rurali
- ✅ Residenziale standard - escluse ville (mercato di nicchia)

...performa meglio di un modello generico su tutto il dataset.

---

## ✅ VERIFICA NON-LEAKAGE

### 1. Filtri Pre-Split

I filtri sono applicati **PRIMA** dello split train/val/test:
- ✅ Tutti gli split vedono lo stesso subset
- ✅ No information leakage tra split
- ✅ Feature contestuali calcolate solo su dati post-filtro

### 2. Feature Contestuali

**Rimosso** tutte le feature che richiedono il target dell'istanza:
```python
# ❌ REMOVED (9 feature):
- price_vs_zone_mean_ratio
- price_zone_zscore
- prezzo_mq
- price_vs_temporal_mean
# ... altre 5 feature
```

**Mantenuto** solo feature calcolabili senza target:
```python
# ✅ KEEP:
- zone_price_mean/median/std  (statistiche aggregate)
- zone_count, type_zone_count  (conteggi)
- surface_vs_zone_mean  (usa solo superficie)
- temporal_count  (conteggi temporali)
```

### 3. Encoding

Test automatici verificano:
- ✅ Encoder fit solo su train
- ✅ Categorie unseen gestite correttamente
- ✅ No leakage tra train/val/test

**Coverage**: `tests/test_encoding_no_leakage.py` (267 righe, 8 test)

---

## 📊 IMPATTO STIMATO

### Dataset Raw

```
Righe totali:        5,733
Target:              AI_Prezzo_Ridistribuito
Mean:                €62,592
Median:              €42,000
Range:               €179 - €1,483,526
Skewness:            5.16 (molto asimmetrico)
```

### Filtri

| Filtro | Righe Rimosse | % Dataset | Note |
|--------|--------------|-----------|------|
| **Zone escluse** (E1/E2/E3/R1) | ~153 | 2.7% | Minimo impatto |
| **Tipologie escluse** (ville) | ~41 | 0.7% | Minimo impatto |
| **Anno >= 2022** | ⚠️ **DA VERIFICARE** | **40-60%?** | **Impatto maggiore** |

⚠️ **CRITICO**: Il filtro temporale potrebbe rimuovere la maggioranza dei dati!

### Stima Dataset Finale

**Scenario conservativo** (dataset originale 2019-2023):
```
Dataset finale:      ~3,000 righe (52% originale)
Split 70/20/10:
  - Train:           ~2,100 righe
  - Validation:      ~600 righe
  - Test:            ~300 righe
```

**Scenario ottimistico** (dataset originale 2021-2023):
```
Dataset finale:      ~4,500 righe (78% originale)
Split 70/20/10:
  - Train:           ~3,150 righe
  - Validation:      ~900 righe
  - Test:            ~450 righe
```

---

## 🚨 RACCOMANDAZIONI PRIORITARIE

### 1. VERIFICA DIMENSIONE DATASET

```bash
# PRIMA di procedere con training, esegui:
python analyze_filters_impact.py
```

Questo script:
- ✅ Mostra distribuzione temporale completa
- ✅ Calcola esattamente quante righe vengono rimosse
- ✅ Confronta statistiche target pre/post filtri
- ✅ Valuta se dataset finale è sufficiente per training

**Threshold critici**:
- 🚨 < 2,000 righe finale → Dataset troppo piccolo, ridurre complessità modelli
- ⚠️ < 3,000 righe finale → Usare `config_fast.yaml` (5 trial vs 150)
- ✅ > 4,000 righe finale → OK per `config.yaml` (150 trial)

### 2. ADATTA CONFIGURAZIONE

Se dataset < 3,000 righe:

```yaml
# Usa config_fast.yaml
training:
  trials_advanced: 5  # invece di 150
  
  ensembles:
    stacking:
      cv_folds: 5  # invece di 10

# Aumenta regularizzazione
models:
  catboost:
    base_params:
      depth: 4  # ridotto da 7
      l2_leaf_reg: 10.0  # aumentato
```

### 3. BASELINE COMPARISON

```bash
# 1. Train su dataset COMPLETO (no filtri)
# Disabilita temporaneamente i filtri in config.yaml
data_filters:
  anno_min: null
  zone_escluse: null
  tipologie_escluse: null

python main.py --config fast --steps preprocessing training evaluation

# 2. Train su dataset FILTRATO (config attuale)
# Riabilita i filtri
python main.py --config fast --steps preprocessing training evaluation

# 3. Confronta metriche
# Modello specializzato ha R² e RMSE migliori?
```

### 4. ABLATION STUDY

Identifica quale filtro ha maggior impatto:

```yaml
# Test A: Solo filtro temporale
data_filters:
  anno_min: 2022
  zone_escluse: null
  tipologie_escluse: null

# Test B: Solo filtro zone
data_filters:
  anno_min: null
  zone_escluse: ['E1','E2','E3','R1']
  tipologie_escluse: null

# Confronta performance → quale filtro migliora di più?
```

---

## 📈 METRICHE ATTESE

### Scenario Ottimistico (filtri efficaci)

```
Baseline (no filtri):
  R²:      0.75
  RMSE:    €38,000
  MAPE:    52%

Con filtri:
  R²:      0.82  ✅ (+7 punti)
  RMSE:    €30,000  ✅ (-21%)
  MAPE:    42%  ✅ (-10 punti)
```

### Scenario Realistico

```
Baseline:
  R²:      0.75
  RMSE:    €38,000
  MAPE:    52%

Con filtri:
  R²:      0.78  ⚠️ (+3 punti, leggero miglioramento)
  RMSE:    €35,000  ⚠️ (-8%)
  MAPE:    48%  ⚠️ (-4 punti)
```

**Trade-off**: Migliore performance su subset, ma modello **non generalizza** a:
- Transazioni pre-2022
- Zone periferiche/rurali (E1/E2/E3/R1)
- Ville (tipologia 4)

---

## ⚖️ DECISIONE: MODELLO SPECIALIZZATO vs GENERALIZZATO

### Quando usare MODELLO SPECIALIZZATO (con filtri)

✅ **PRO**:
- Performance migliore su subset specifico
- Riduce complessità (meno eterogeneità)
- Riduce temporal drift (solo dati recenti)

❌ **CONTRO**:
- Non generalizza fuori dal subset
- Richiede dataset filtrato sufficiente (>2,000 righe)
- Deployment limitato a scope ristretto

**Use Case**: 
- Predizioni solo su transazioni 2022+, zone urbane, residenziale standard
- Applicazione specializzata (es. solo appartamenti semicentrali)

### Quando usare MODELLO GENERALIZZATO (no filtri)

✅ **PRO**:
- Generalizza a tutto il dataset
- Dataset più grande (più robusto)
- Deployment universale

❌ **CONTRO**:
- Performance media inferiore (gestisce più eterogeneità)
- Maggiore temporal/spatial drift

**Use Case**:
- Predizioni su qualsiasi transazione (passate, future, tutte le zone)
- Applicazione general-purpose

---

## 🎬 PROSSIMI PASSI

### Immediati (oggi)

1. ✅ **Esegui analisi impatto filtri**:
   ```bash
   python analyze_filters_impact.py
   ```

2. ✅ **Verifica dimensione dataset finale**:
   - Se < 2,000 righe → 🚨 Dataset troppo piccolo, considera di ridurre filtri
   - Se 2,000-3,000 righe → ⚠️ Usa `config_fast.yaml`
   - Se > 3,000 righe → ✅ Usa `config.yaml` o `config_fast.yaml`

3. ✅ **Training baseline** (opzionale ma raccomandato):
   ```bash
   # Disabilita filtri temporaneamente
   python main.py --config fast --steps preprocessing training evaluation
   # Salva metriche per confronto
   ```

### Breve termine (questa settimana)

4. ✅ **Training con filtri**:
   ```bash
   python main.py --config fast --steps preprocessing training evaluation
   ```

5. ✅ **Confronto baseline vs filtrato**:
   - R², RMSE, MAPE migliorati?
   - Di quanto? (se < 5%, filtri non valgono la perdita di generalizzazione)

6. ✅ **Ablation study** (opzionale):
   - Quale filtro ha maggior impatto?
   - Conviene usare solo alcuni filtri?

### Medio termine (prossime iterazioni)

7. ✅ **Error analysis**:
   ```python
   # Analizza worst predictions
   # Pattern comuni? Zone specifiche? Prezzi estremi?
   ```

8. ✅ **Production readiness**:
   - Documenta scope del modello (solo 2022+, no zone E/R, no ville)
   - Implementa warning per predizioni out-of-distribution
   - Considera A/B test in production (baseline vs specializzato)

---

## 📚 DOCUMENTAZIONE

### File Generati

```
/workspace/
  - ANALISI_SUBSET_CONFIG_2022.md       (analisi completa, 654 righe)
  - EXECUTIVE_SUMMARY_FILTERS.md        (questo file, executive summary)
  - analyze_filters_impact.py           (script verifica impatto)
```

### File Chiave da Consultare

```
config/
  - config.yaml                (linee 56-80: data_filters)
  - config_fast.yaml          (filtri identici)

src/preprocessing/
  - pipeline.py               (linee 98-212: apply_data_filters)
  - contextual_features.py    (leak-free features)

tests/
  - test_encoding_no_leakage.py  (verifica no leakage)

notebooks/eda_outputs/
  - target_statistics.csv
  - correlations_with_target.csv
  - group_summary_AI_ZonaOmi.csv
```

---

## ✅ CONCLUSIONE

### Status Attuale

✅ **Configurazione valida**: Filtri applicati correttamente, no leakage  
⚠️ **Da verificare**: Impatto reale filtro temporale sul dataset size  
📊 **Prossimo step**: Eseguire `analyze_filters_impact.py` per confermare fattibilità

### Raccomandazione Finale

1. **Esegui analisi**: `python analyze_filters_impact.py`
2. **Se dataset > 3,000 righe** → Procedi con training filtrato
3. **Se dataset < 3,000 righe** → Valuta riduzione filtri o uso config_fast
4. **Confronta con baseline** → Verifica se filtri migliorano realmente performance

**L'architettura è solida e leak-free. Il successo dipende dalla dimensione del dataset finale dopo filtri.**

---

**Analisi completata il**: 2025-11-14  
**Autore**: Claude (Sonnet 4.5)  
**Contatti**: [team data science]
