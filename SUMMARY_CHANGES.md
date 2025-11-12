# 📋 Summary Modifiche - Data-Driven Optimization

## 🎯 Cosa È Stato Fatto

Ho implementato **3 modifiche chiave** basate su **analisi data-driven** del dataset:

---

## 1️⃣ **FEATURE PRUNING** (-56 colonne inutili) 🗑️

### Analisi Eseguita
- ✅ Letta correlation matrix (`notebooks/eda_comprehensive_outputs/correlation_matrix_pearson.csv`)
- ✅ Analizzata SQL query (`sql/base_query.sql`)
- ✅ Identificate colonne ridondanti, costanti, ID, poco predittive

### Colonne Rimosse (Data-Driven)

| Categoria | # Colonne | Esempi | Ragione |
|-----------|-----------|---------|---------|
| **ID e Foreign Keys** | 12 | `A_Id`, `AI_Id`, `PC_Id`, `OZ_Id` | Identificatori univoci (non feature) |
| **Superficie Ridondanti** | 5 | `AI_SuperficieVisuraTotale` (r=1.0 con `AI_Superficie`) | Correlazione > 0.98 |
| **Indicatori Istat Ridondanti** | 7 | `II_ST2_B`, `II_ST21`, `II_ST29` (r > 0.98) | Cluster ad alta correlazione |
| **OmiValori Ridondanti** | 4 | `OV_ValoreMercatoMax_normale` (r=0.98 con Min) | Max vs Min quasi identici |
| **Metadata/Tecnici** | 13 | `A_Semestre`, `A_DataStipula`, `PC_PoligonoMetrico` | Non feature predittive |
| **Codici Catastali** | 8 | `PC_Foglio`, `PC_Particella`, `PC_Subalterno` | Troppi unique, poco predittivi |
| **Privacy/Poco Predittivi** | 7 | `A_EtaMediaAcquirenti`, `A_VenditoriCount` | Scarsa utilità predittiva |
| **TOTALE** | **56** | - | **~40% feature in meno** |

### Benefici Attesi
- ✅ **Meno noise**: Modello più robusto e generalizzabile
- ✅ **Meno multicollinearità**: Coefficienti più stabili
- ✅ **Training più veloce**: ~40% feature in meno → ~30% tempo training
- ✅ **Meno overfitting**: Feature ridondanti causano memorizzazione

---

## 2️⃣ **NUMERIC_COERCION CORRETTO** 🔧

### Problema Identificato
```yaml
# PRIMA (ERRATO):
blacklist_globs:
  - 'II_*'  # ❌ Blocca TUTTI gli indicatori Istat!
```

**Effetto**: Metriche numeriche come `II_ST1`, `II_P98` (popolazione, densità) rimanevano stringhe invece di essere convertite in float → modelli lineari/tree non le usavano correttamente!

### Soluzione (Data-Driven)
```yaml
# DOPO (CORRETTO):
blacklist_globs:
  - 'II_IdIstatZonaCensuaria'  # ✅ Solo ID Istat (codice)
  # II_ST1, II_ST2, II_P98, ... → CONVERTITI in float (corretto!)
```

**Analisi**: 
- `II_ST*` e `II_P*` sono **metriche numeriche** (es. `II_ST1 = 3245.7` = popolazione)
- `II_IdIstatZonaCensuaria` è **codice ID** (es. `"123456789"` → deve rimanere string)

### Altre Correzioni Blacklist

**Aggiunti pattern più specifici**:
```yaml
- '*Id'               # Tutti gli ID (A_Id, AI_Id, ...)
- '*_Id*'             # Varianti (IdAtto, IdParticella, ...)
- 'AI_ZonaOmi'        # Zona OMI ("D2", "C4") - CATEGORICO
- '*IdCategoriaCatastale*'  # "00210", "00020" - con leading zeros
- '*IdTipologiaEdilizia*'   # "2", "3", "8" - codici categorici
```

**Beneficio**: Le metriche Istat ora sono convertite correttamente in float → modelli possono usarle!

---

## 3️⃣ **FEATURE CONTESTUALI** (+44 feature) 🎯

### Cosa Fanno
Aggiungono **contesto di mercato locale** che prima mancava:

**Zone Statistics** (13 feature):
- `zone_price_mean`, `zone_price_median`, `zone_price_std`: Prezzo medio/mediano/std per zona
- `zone_price_q25`, `zone_price_q75`: Quartili
- `price_vs_zone_mean_ratio`: Posizione immobile rispetto a media zona
- `price_zone_zscore`: Z-score nella zona
- ...

**Typology×Zone Statistics** (8 feature):
- `type_zone_price_mean`: Prezzo per tipologia × zona (nicchie di mercato)
- `type_zone_rarity`: Quanto è rara questa combinazione
- ...

**Surface Context** (5 feature):
- `surface_vs_zone_mean`: Superficie relativa alla zona
- `surface_vs_type_zone_mean`: Superficie relativa a tipologia×zona
- ...

**Interaction Features** (4+ feature):
- `prezzo_mq`: Prezzo al metro quadro
- `prezzo_mq_vs_zone`: Prezzo/mq relativo alla zona
- `log_superficie`: Effetti scala
- ...

**Temporal Context** (7 feature):
- `temporal_price_mean`: Trend temporale prezzi
- `quarter`: Stagionalità
- `months_from_start`: Trend lineare
- ...

### Perché Aiutano

**PRIMA**: 
```
Modello vede: Immobile 150k€ in zona "D2"
Modello NON sa: 150k€ è tanto? Poco? Nella media?
```

**DOPO**:
```
Modello vede: 
- Immobile 150k€ in zona "D2"
- Zona D2: prezzo medio 160k€, mediano 155k€
- Questo immobile: 6% sotto media → "normale, leggermente economico"
- Prezzo/mq: 2500€/mq vs zona media 2600€/mq → "in linea"
```

**Risultato**: Il modello capisce il **contesto** e fa previsioni più accurate!

---

## 4️⃣ **REGULARIZZAZIONE AGGRESSIVA** 🛡️

### Problema
**Overfitting MASSICCIO**:
- Gap R² train-test: 0.214 (21%!)
- RMSE ratio: 2.67x (train 13k€ vs test 37k€)

### Soluzione
Ridotti tutti gli hyperparameter ranges per prevenire overfitting:

**CatBoost** (esempio):
```yaml
depth: 4-10 → 4-7                 ✅ RIDOTTO
learning_rate: 0.001-0.3 → 0.01-0.12  ✅ RIDOTTO
l2_leaf_reg: 10-100 → 3-30        ✅ RIDOTTO
+ early_stopping_rounds: 50       ✅ NUOVO
+ min_data_in_leaf: 20-80         ✅ NUOVO
```

Stesso principio per **XGBoost**, **LightGBM**, **GBR**, **HGBT**, **RF**.

---

## 📊 Risultati Attesi (Baseline → Target)

### Metriche Test (Scala Originale - EURO)

| Metrica | Baseline | Target | Miglioramento |
|---------|----------|--------|---------------|
| **RMSE** | 36,767€ | 22-26k€ | **-30% a -40%** ✅✅ |
| **MAE** | 19,811€ | 12-15k€ | **-35% a -40%** ✅✅ |
| **MAPE** | 58.1% | 25-35% | **-40% a -55%** ✅✅✅ |
| **R²** | 0.736 | 0.82-0.87 | **+10% a +18%** ✅ |

### Overfitting

| Metrica | Baseline | Target | Miglioramento |
|---------|----------|--------|---------------|
| **Gap R²** | 0.214 | <0.10 | **-50% a -70%** ✅✅ |
| **RMSE Ratio** | 2.67x | <1.8x | **-30% a -40%** ✅ |

### Performance Gruppi

**Baseline**: R² NEGATIVI per fasce prezzo basse, MAPE 134% per zona C4

**Target**: 
- ✅ Tutte zone con R² > 0.60
- ✅ Tutte fasce prezzo con R² > 0.40
- ✅ Nessuna zona/fascia con MAPE > 60%

---

## 📁 File Modificati/Creati

### Creati
```
✅ src/preprocessing/contextual_features.py    (44 feature contestuali)
✅ config/config_optimized.yaml                (config completa ottimizzata)
✅ run_optimization.py                         (script esecuzione automatica)
✅ OPTIMIZATION_GUIDE.md                       (guida dettagliata)
✅ DATA_DRIVEN_ANALYSIS.md                     (analisi data-driven)
✅ QUICK_START_OPTIMIZATION.md                 (quick start)
✅ SUMMARY_CHANGES.md                          (questo file)
```

### Modificati
```
✅ src/preprocessing/pipeline.py  → Integrata chiamata a add_all_contextual_features()
```

---

## 🚀 Come Eseguire

### Opzione 1: Script Automatico (CONSIGLIATO)
```bash
python run_optimization.py
```

Fa tutto:
1. ✅ Preprocessing con feature contestuali
2. ✅ Training con regularizzazione aggressiva
3. ✅ Evaluation
4. ✅ Confronto baseline vs ottimizzato

**Tempo**: ~30-45 min (solo CatBoost) o ~2 ore (tutti i modelli)

### Opzione 2: Manuale
```bash
python main.py --config config/config_optimized.yaml --steps preprocessing
python main.py --config config/config_optimized.yaml --steps training
python main.py --config config/config_optimized.yaml --steps evaluation
```

---

## 📋 Checklist Modifiche

### Feature Pruning
- [x] Identificate 56 colonne da droppare (data-driven)
- [x] Aggiunte a `config_optimized.yaml` → `feature_pruning.drop_columns`
- [x] Categorizzate per ragione (ID, ridondanti, metadata, ecc.)

### Numeric Coercion
- [x] Corretto errore `'II_*'` che bloccava metriche numeriche
- [x] Aggiunti pattern specifici per ID e codici categorici
- [x] Verificato che `II_ST*`, `II_P*` siano convertiti in float

### Feature Contestuali
- [x] Creato modulo `contextual_features.py`
- [x] Implementate 5 funzioni (zone, typology, surface, interactions, temporal)
- [x] Integrato in `pipeline.py` PRIMA dello split temporale
- [x] Testate feature aggiunte (44 totali)

### Regularizzazione
- [x] Ridotti hyperparameter ranges per tutti i modelli tree-based
- [x] Aggiunto early_stopping per CatBoost/XGBoost
- [x] Aumentati constraint min (min_samples_leaf, min_child_weight, ecc.)
- [x] Aumentata CV folds da 5 a 10

### Documentazione
- [x] `OPTIMIZATION_GUIDE.md`: Guida completa
- [x] `DATA_DRIVEN_ANALYSIS.md`: Analisi data-driven feature pruning
- [x] `QUICK_START_OPTIMIZATION.md`: Quick start
- [x] `SUMMARY_CHANGES.md`: Questo summary

---

## ✅ Pronto per Esecuzione

**TUTTO è pronto**. Ora puoi eseguire:

```bash
# Backup baseline (opzionale)
cp -r models/ models_baseline/

# Run ottimizzazione
python run_optimization.py

# Verifica risultati
cat models/catboost/metrics.json | grep -A 15 metrics_test_original
```

**Atteso**: MAPE da 58% a 25-35%, RMSE da 37k€ a 22-26k€ 🎯

---

**Domande?** Leggi:
- `QUICK_START_OPTIMIZATION.md` per esecuzione rapida
- `DATA_DRIVEN_ANALYSIS.md` per dettagli feature pruning
- `OPTIMIZATION_GUIDE.md` per guida completa e fasi successive
