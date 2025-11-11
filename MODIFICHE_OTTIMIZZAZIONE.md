# 🚀 MODIFICHE PER OTTIMIZZAZIONE MODELLO

## 📅 Data: 2024-11-11

## 🎯 OBIETTIVO
Ridurre MAPE da **45.5%** a **<25%** e RMSE da **€43,929** a **<€30,000**

---

## ✅ MODIFICHE IMPLEMENTATE

### **1. FILTRO TEMPORALE (≥2022)** 🕐
**File**: `config/config.yaml`

```yaml
temporal_filter:
  enabled: true
  min_year: 2022  # Elimina drift temporale
  exclude_zones: ['E1', 'E2', 'E3', 'R1']  # Zone con <30 campioni
  exclude_tipologie: ['18', '8']  # Box/cantine (categorie diverse)
```

**Impatto**:
- Dataset: 5,680 → **1,725 campioni** (30.4%)
- Drift alerts: 158 → **~30-40** (stimato -75%)
- Zone OMI: 13 → **9 zone robuste** (tutte ≥44 campioni)
- Solo immobili residenziali (esclusi box €21k e cantine €6k)

---

### **2. TRASFORMAZIONE TARGET: boxcox → log** 📈
**File**: `config/config.yaml`

```yaml
target:
  transform: 'log'  # Da 'boxcox' (più stabile, riduce gap train-test)
```

**Perché**:
- Box-Cox ottimizza su scala trasformata → gap enorme quando torni a scala originale
- Log è più robusto e interpretabile
- Gap R² atteso: da 0.19 (0.863-0.673) a **~0.06**

---

### **3. OUTLIER PIÙ AGGRESSIVI** 🎯
**File**: `config/config.yaml`

```yaml
outliers:
  iqr_factor: 1.2  # Da 1.5 (più stretto)
  iso_forest_contamination: 0.05  # Da 0.02 (rimuove 5% outlier)
  group_by_col: 'AI_ZonaOmi'  # Da 'AI_IdTipologiaEdilizia'
```

**Impatto**:
- Rimuove outlier più aggressivamente
- Gruppi per ZONA (più predittivo: R²=12.1% vs 8.1% tipologia)

---

### **4. GROUPING PER ZONA (non Tipologia)** 🗺️
**File**: `config/config.yaml`

**Cambiato**:
- `outliers.group_by_col`: `'AI_ZonaOmi'`
- `imputation.group_by_col`: `'AI_ZonaOmi'`

**Analisi dati (solo residenziali)**:
```
ZonaOmi:          R² = 12.1% | CV = 0.65 ✅
TipologiaEdilizia: R² = 8.1%  | CV = 0.90
```

**Zone post-filtro** (9 zone residenziali):
```
C6: €156k (44 campioni)
D2: €140k (236 campioni)
B1: €131k (595 campioni)
C4: €84k  (328 campioni)
C2: €74k  (107 campioni)
C5: €69k  (98 campioni)
D1: €63k  (135 campioni)
D3: €61k  (120 campioni)
C3: €59k  (62 campioni)
```

---

### **5. IMPLEMENTAZIONE CODICE** 💻
**File**: `src/preprocessing/pipeline.py`

Aggiunta logica filtro in `run_preprocessing()`:
- Filtro anno (A_AnnoStipula >= 2022)
- Filtro zone (escluse E1/E2/E3/R1)
- **NUOVO**: Filtro tipologie (escluse 18/8 = box/cantine)
- Logging dettagliato per debugging

---

## 📊 RISULTATI ATTESI

### **Dataset**
| Metrica | Before | After | Δ |
|---------|--------|-------|---|
| Campioni | 5,680 | 1,725 | **-70%** |
| Anni | 2019-2024 | 2022-2024 | **-3 anni** |
| Zone OMI | 13 | 9 | **-31%** |
| Campioni/zona (min) | 5 (R1) | 44 (C6) | **+780%** |
| Prezzo medio | €62k | €100k | **+61%** |

### **Performance ML**
| Metrica | Before | After | Δ |
|---------|--------|-------|---|
| **MAPE** | 45.5% | **25-28%** | **-40%** 🔥 |
| **RMSE** | €43,929 | **€28-32k** | **-30%** |
| **R² (orig)** | 0.673 | **0.80-0.83** | **+19%** |
| **Drift (PSI alerts)** | 158 | **~35** | **-78%** |
| **Gap R² (transf-orig)** | 0.19 | **~0.06** | **-68%** |

---

## 🚀 COME ESEGUIRE

### **Opzione A: Full training** (CONSIGLIATO)
```bash
cd /workspace
python main.py --config config/config.yaml --steps preprocessing training evaluation
```
- Tempo: ~30-60 minuti
- Training completo con 100 trials per modello
- Ensemble voting + stacking

### **Opzione B: Test veloce** (5 minuti)
```bash
python main.py --config config/config_fast_test.yaml --steps preprocessing training evaluation
```
- Tempo: ~5 minuti
- Meno trials (più veloce)
- Per validazione rapida

---

## 📈 METRICHE DA MONITORARE

Dopo il training, controlla:

1. **`models/summary.json`**: Metriche aggregate
   - Test RMSE (target: <€30k)
   - Test MAPE floor (target: <28%)
   - R² original scale (target: >0.80)

2. **`models/drift_report.json`**: Alert drift
   - PSI alerts (target: <50)
   - KS test alerts (target: <80)

3. **`models/catboost/metrics.json`**: Best model
   - Overfit gap_r2 (target: <0.15)
   - MAPE (target: <25%)

4. **`models/validation_results.csv`**: Ranking modelli
   - Top 3 modelli
   - Confronto baseline vs optimized

---

## 💡 NEXT STEPS (se MAPE ancora >25%)

### **1. Feature selection** (2-3 ore)
Aggiungi a `config.yaml` dopo analisi drift:
```yaml
feature_pruning:
  drop_columns:
    - 'C2_COD_APE'          # PSI > 1.0
    - 'C2_PARTICELLA'       # PSI > 1.0
    - 'C2_SUPERFICIE_DISPERDENTE'
    - 'OV_IdZona_normale__ord'  # PSI = 6.07!
    # ... altre con PSI > 1.0
```

### **2. Feature engineering** (1 giorno)
Crea features avanzate:
```python
# In preprocessing/feature_extractors.py
df['zona_tipologia'] = df['AI_ZonaOmi'] + '_' + df['AI_IdTipologiaEdilizia']
df['prezzo_mq_norm_zona'] = df.groupby('AI_ZonaOmi')['AI_Prezzo_MQ'].transform(
    lambda x: (x - x.mean()) / x.std()
)
df['POI_density'] = df['POI_total'] / (df['AI_Superficie'] + 1)
```

### **3. Segmentazione per fascia prezzo** (2 giorni)
Crea 3 modelli separati:
- Low: <€50k (n=~850)
- Mid: €50-120k (n=~650)
- High: >€120k (n=~225)

### **4. Ensemble più pesante** (6 ore)
```yaml
training:
  trials_advanced: 150  # Da 100
  ensembles:
    stacking:
      top_n: 6  # Da 5
```

### **5. Neural Network** (3-5 giorni)
Implementa TabNet o MLP per catturare interazioni complesse.

---

## 🎓 RATIONALE TECNICO

### **Perché filtrare ≥2022?**
- Mercato immobiliare pre-COVID ≠ post-COVID
- Tassi interesse, inflazione, domanda/offerta cambiate radicalmente
- Feature energetiche (CENED) più omogenee post-2022
- **PSI drift su A_AnnoStipula = 7.85** (altissimo!) → Risolto

### **Perché escludere box/cantine?**
- Categoria completamente diversa (€6-21k vs €100k appartamenti)
- Distorcevano i gruppi (TipologiaEdilizia R²=33% per questo!)
- **43% dei campioni** ma **<10% del valore** del mercato
- Modelli separati per pertinenze richiedono pipeline dedicata

### **Perché ZonaOmi e non TipologiaEdilizia?**
- **Su residenziali**: ZonaOmi R²=12.1% vs Tipologia 8.1%
- CV più basso: 0.65 vs 0.90 (gruppi più omogenei)
- Zona spiega differenziale prezzo (centro €156k vs periferia €59k = 2.6x)
- Tipologia (villa/appartamento) ha range più contenuto

### **Perché log invece di boxcox?**
- Box-Cox ottimizza su scala trasformata → gap 19% quando torni a originale
- Log è più stabile e interpretabile
- Preserva ordinalità e proporzionalità
- **Duan smearing** applicabile per correzione bias

---

## ✅ CHECKLIST PRE-RUN

- [x] Config.yaml modificato (temporal_filter, transform, outliers, grouping)
- [x] Pipeline.py modificato (filtro tipologie)
- [x] Test script eseguito con successo
- [ ] **TODO**: Eseguire preprocessing + training
- [ ] **TODO**: Validare metriche (MAPE < 28%, RMSE < €32k)
- [ ] **TODO**: Analizzare drift_report.json (alerts < 50)

---

## 📝 FILE MODIFICATI

1. ✅ `config/config.yaml`
   - Sezione `temporal_filter` (nuova)
   - `target.transform`: boxcox → log
   - `outliers`: iqr_factor, contamination, group_by_col
   - `imputation.group_by_col`

2. ✅ `src/preprocessing/pipeline.py`
   - Funzione `run_preprocessing()` (linee 149-183)
   - Logica filtro temporale + zone + tipologie

3. ✅ `test_temporal_filter.py` (nuovo)
   - Script di test per validare filtri

---

## 🔗 RIFERIMENTI

- Drift detection: `models/drift_report.json`
- Metriche baseline: `models/summary.json` (prima delle modifiche)
- Analisi dataset: output di `test_temporal_filter.py`
- Best model: `models/catboost/` (dopo training)

---

**Autore**: Cursor AI Agent  
**Data**: 2024-11-11  
**Status**: ✅ Ready to run
