# 📊 ANALISI APPROFONDITA: SUBSET CONFIGURATION 2022+

**Data analisi**: 2025-11-14  
**Branch**: `cursor/analyze-and-test-data-subset-176c`  
**Configurazione**: Filtri sperimentali per testing su subset specifico

---

## 🎯 EXECUTIVE SUMMARY

Questa analisi esamina la configurazione attuale che applica **filtri sperimentali** al dataset per valutare modelli specializzati su un subset temporale e geografico specifico.

### Filtri Applicati (config.yaml e config_fast.yaml)

```yaml
data_filters:
  anno_min: 2022              # ✅ Solo transazioni dal 2022 in poi
  zone_escluse:               # ✅ Esclusione zone periferiche/rurali
    - 'E1'                    #    (Periferia 1)
    - 'E2'                    #    (Periferia 2)
    - 'E3'                    #    (Periferia 3)
    - 'R1'                    #    (Rurale 1)
  tipologie_escluse:          # ✅ Esclusione ville
    - '4'                     #    (Tipologia 4 = Ville)
```

### Motivazione

L'utente specifica che questi filtri sono applicati **per test** e **NON costituiscono leakage**. I filtri servono per:

1. **Focus temporale**: Concentrarsi su dati recenti (2022+) riducendo il drift temporale
2. **Focus geografico**: Escludere zone periferiche/rurali con pattern di pricing diversi
3. **Focus tipologico**: Escludere ville (mercato di nicchia con dinamiche diverse)

---

## 📈 DATASET OVERVIEW

### Dimensioni

```
Dataset raw:                      5,733 righe × 265 colonne
Memoria:                          ~4.0 MB (compresso parquet)
Periodo temporale completo:       [Da verificare con A_AnnoStipula]
```

### Target: AI_Prezzo_Ridistribuito

| Statistica | Valore |
|------------|--------|
| **Count** | 5,733 (100% non-null) |
| **Mean** | €62,592 |
| **Median** | €42,000 |
| **Std Dev** | €79,533 (127% del mean!) |
| **Min** | €179 |
| **Max** | €1,483,526 |
| **Q1 (25%)** | €13,690 |
| **Q3 (75%)** | €82,070 |
| **Skewness** | **5.16** (molto asimmetrico a destra) |
| **Kurtosis** | **54.18** (code estremamente pesanti) |

⚠️ **Osservazione critica**: Il target ha una distribuzione **fortemente skewed** con code pesanti, giustificando l'uso della trasformazione Yeo-Johnson configurata.

### Distribuzione per Categorie Catastali

| Categoria | Count | Mean Price | Median Price | CV (%) |
|-----------|-------|------------|--------------|--------|
| **00275** (Pertinenze/box) | 1,812 | €20,650 | €15,787 | 89% |
| **00020** (Abitazioni civili) | 1,789 | €113,673 | €90,000 | 78% |
| **00030** (Abitazioni economiche) | 1,131 | €66,942 | €58,000 | 83% |
| **00210** (Magazzini/depositi) | 651 | €8,067 | €2,596 | **264%** (!) |
| **00200** (Uffici/studi) | 151 | €90,608 | €45,000 | 158% |
| **00100** (Negozi/botteghe) | 95 | €143,986 | €79,851 | 124% |
| **00070** (Ville/villini) | 41 | €149,706 | €145,512 | 45% |

**Insight**: Le pertinenze (00275) e magazzini (00210) hanno alta varianza, mentre le ville (00070) sono più omogenee ma saranno **escluse** dal filtro tipologie.

### Distribuzione per Zone OMI

| Zona | Count | Mean Price | Median Price | Note |
|------|-------|------------|--------------|------|
| **B1** (Semicentrale) | 1,797 | €86,727 | €59,602 | **Maggioranza** |
| **C4** (Periferica buona) | 1,105 | €50,457 | €44,974 | Inclusa |
| **D2** (Popolare) | 718 | €75,142 | €48,677 | Inclusa |
| **D1** (Popolare economica) | 553 | €36,768 | €25,000 | Inclusa |
| **C2** (Periferica media) | 386 | €38,052 | €17,930 | Inclusa |
| **E3** (Periferia 3) | 59 | €86,365 | €36,537 | ❌ **ESCLUSA** |
| **E2** (Periferia 2) | 49 | €55,730 | €33,500 | ❌ **ESCLUSA** |
| **E1** (Periferia 1) | 37 | €51,578 | €18,505 | ❌ **ESCLUSA** |
| **R1** (Rurale 1) | 8 | €61,551 | €26,250 | ❌ **ESCLUSA** |

---

## 🔍 IMPATTO DEI FILTRI

### Calcolo Teorico delle Righe Rimosse

#### 1. Filtro Zone Escluse (E1, E2, E3, R1)

```
Zone E1:  37 transazioni  (0.6%)
Zone E2:  49 transazioni  (0.9%)
Zone E3:  59 transazioni  (1.0%)
Zone R1:   8 transazioni  (0.1%)
-------------------------------------------
Totale:  153 transazioni  (2.7% del dataset)
```

**Impatto**: Minimo. Solo 2.7% dei dati rimossi.

#### 2. Filtro Tipologie Escluse (Ville = categoria '4')

Dalla distribuzione per categorie, le ville (00070) sono **41 transazioni** (~0.7%).

**Nota**: Il mapping esatto tra `AI_IdTipologiaEdilizia` e categorie catastali va verificato, ma l'impatto è comunque minimo.

#### 3. Filtro Temporale (anno_min: 2022)

⚠️ **DATO MANCANTE**: Non è stato possibile analizzare la distribuzione temporale di `A_AnnoStipula` dal file raw.

**Raccomandazione**: Eseguire l'analisi con:
```python
df = pd.read_parquet('data/raw/raw.parquet')
print(df['A_AnnoStipula'].value_counts().sort_index())
```

**Stima conservativa**: Se il dataset copre 3-5 anni (es. 2019-2023), il filtro 2022+ potrebbe rimuovere **40-60% dei dati**.

### Stima Totale Dataset Finale

```
Scenario conservativo (dataset 2019-2023):
  Iniziale:           5,733 righe
  Filtro temporale:  -2,500 righe (45%)
  Filtro zone:         -153 righe (2.7%)
  Filtro tipologie:     -41 righe (0.7%)
  ----------------------------------------
  Dataset finale:    ~3,000 righe (52% dell'originale)

Scenario ottimistico (dataset 2021-2023):
  Dataset finale:    ~4,500 righe (78% dell'originale)
```

⚠️ **ATTENZIONE**: Con ~3,000 righe e split 70/20/10:
- **Train**: ~2,100 righe
- **Validation**: ~600 righe
- **Test**: ~300 righe

Questo è **al limite minimo** per training robusto di modelli complessi (XGBoost, CatBoost con 150 trial Optuna).

---

## ✅ VERIFICA NON-LEAKAGE

### 1. Applicazione Filtri PRE-SPLIT ✅

Dal codice `src/preprocessing/pipeline.py` (linee 98-212):

```python
def apply_data_filters(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Applica filtri al dataset INTERO (prima dello split).
    """
    # ... filtri applicati al dataset raw completo ...
    return df
```

**✅ CORRETTO**: I filtri sono applicati al dataset raw **PRIMA** dello split temporale, quindi:
- Non c'è leakage tra train/val/test
- Tutti gli split vedono lo stesso subset filtrato
- Le feature contestuali sono calcolate solo su dati post-filtro

### 2. Feature Contestuali LEAK-FREE ✅

Dal codice `src/preprocessing/contextual_features.py`:

#### Feature RIMOSSE (richiedevano target dell'istanza):

```python
# ❌ REMOVED: price_vs_zone_mean_ratio         (linea 161)
# ❌ REMOVED: price_vs_zone_median_ratio       (linea 162)
# ❌ REMOVED: price_zone_zscore                (linea 163)
# ❌ REMOVED: price_zone_iqr_position          (linea 164)
# ❌ REMOVED: price_zone_range_position        (linea 165)
# ❌ REMOVED: price_vs_type_zone_mean          (linea 186)
# ❌ REMOVED: prezzo_mq                        (linea 209)
# ❌ REMOVED: prezzo_mq_vs_zone                (linea 210)
# ❌ REMOVED: price_vs_temporal_mean           (linea 226)
```

#### Feature MANTENUTE (calcolabili senza target):

```python
# ✅ KEEP: zone_price_mean, zone_price_median, zone_price_std
# ✅ KEEP: zone_surface_mean, surface_vs_zone_mean
# ✅ KEEP: type_zone_count, type_zone_rarity
# ✅ KEEP: temporal_count, months_from_start, quarter
# ✅ KEEP: log_superficie, superficie_x_categoria
```

**Processo**:
1. **Fit** su train: calcola statistiche aggregate (media prezzo zona, count, ecc.)
2. **Transform** su train/val/test: applica le stesse statistiche

**✅ VERIFICATO**: Nessun leakage, tutte le feature sono production-ready.

### 3. Encoding Multi-Strategy ✅

Dal file `tests/test_encoding_no_leakage.py` (267 righe di test):

```python
def test_encoding_fit_only_on_train(self):
    """Test che gli encoder siano fit solo su training set."""
    # Test con categorie unseen in validation
    # Verifica che encoder non "veda" categorie del test
```

**Test Coverage**:
- ✅ OneHot encoding: categorie unseen gestite correttamente
- ✅ Target encoding: categorie unseen → valore globale/neutro
- ✅ Frequency encoding: categorie unseen → frequenza 0
- ✅ Ordinal encoding: categorie unseen → sentinel -1
- ✅ Riproducibilità: stesso input → stesso output

**✅ VERIFICATO**: Encoding leak-free con test automatici.

### 4. Split Temporale ✅

Dal codice `src/preprocessing/pipeline.py` (linee 376-394):

```python
# Temporal split FIRST to avoid leakage (contextual features AFTER split!)
split_cfg = TemporalSplitConfig(
    mode='fraction',  # o 'date'
    train_fraction=0.7,
    valid_fraction=0.2,
)
train_df, val_df, test_df = temporal_split_3way(Xy_full, split_cfg)
```

**✅ CORRETTO**: Split temporale basato su `A_AnnoStipula` e `A_MeseStipula`, preserva ordinamento cronologico.

---

## 📊 TOP CORRELAZIONI CON TARGET

### Correlazioni Pearson (da EDA)

| Feature | Pearson | Spearman | Tipo | Note |
|---------|---------|----------|------|------|
| **AI_Rendita** | 0.688 | 0.675 | Numerica | Rendita catastale (forte predittore) |
| **AI_Superficie** | 0.669 | 0.671 | Numerica | Superficie immobile |
| **AI_SuperficieVisuraTotaleAttuale** | 0.672 | 0.686 | Numerica | Superficie da visura catastale |
| **OV_ValoreMercatoMin_normale** | 0.336 | 0.363 | Numerica | Valori OMI (Osservatorio Mercato Immobiliare) |
| **OV_ValoreMercatoMax_normale** | 0.335 | 0.368 | Numerica | Valori OMI max |
| **POI_shopping_mall_count** | 0.293 | - | Numerica | Presenza centri commerciali (feature geospaziale) |
| **II_ST1** | -0.271 | -0.363 | Numerica | Indicatore ISTAT (correlazione negativa) |
| **II_ST19** | -0.270 | -0.367 | Numerica | Indicatore ISTAT (correlazione negativa) |

**Insight**:
- Le feature catastali (rendita, superficie) sono i predittori più forti
- I valori OMI aggiungono informazione contestuale sul mercato
- Le feature geospaziali (POI) hanno correlazione moderata
- Gli indicatori ISTAT catturano contesto socio-economico (negativo = zone disagiate)

### Feature Droppate nella Config

Dal `config.yaml` (linee 181-253), ~56 colonne vengono rimosse:

**Categorie**:
1. **ID e chiavi esterne** (12 colonne): `A_Id`, `AI_IdAtto`, `PC_Id`, ecc.
2. **Superfici ridondanti** (5 colonne): `AI_SuperficieCalcolata`, `AI_SuperficieVisuraTotale`, ecc.
3. **Indicatori ISTAT ridondanti** (7 colonne): `II_ST2_B`, `II_ST21`, ecc.
4. **OmiValori ridondanti** (4 colonne): `OV_ValoreMercatoMax_scadente`, ecc.
5. **Metadata e colonne tecniche** (13 colonne): `A_Semestre`, `A_DataStipula`, ecc.
6. **Codici catastali** (8 colonne): `PC_Foglio`, `PC_Particella`, ecc.
7. **Colonne poco predittive** (7 colonne): `AI_Piano`, `AI_Civico`, `AI_Rendita`, ecc.

**Motivazione**: Feature pruning data-driven basato su:
- Correlazione > 0.98 (ridondanza)
- Missing > 80% (non informatività)
- ID/codici (non predittivi)

---

## 🧪 CONFIGURAZIONE TRAINING

### Modelli Abilitati

#### Config Completo (`config.yaml`)

```yaml
trials_advanced: 150  # Optuna hyperparameter tuning

models:
  catboost:   enabled: true   (150 trial, 800 iterations)
  xgboost:    enabled: true   (150 trial)
  lightgbm:   enabled: true   (150 trial)
  gbr:        enabled: true   (150 trial, sklearn GBR)
  hgbt:       enabled: true   (150 trial, sklearn HistGBT)
  rf:         enabled: true   (150 trial, Random Forest)

ensembles:
  voting:     enabled: true   (top 5 modelli)
  stacking:   enabled: true   (top 7, Ridge meta-learner, CV 10-fold)
```

**Tempo stimato**: ~2-3 ore con dataset completo

#### Config Fast (`config_fast.yaml`)

```yaml
trials_advanced: 5  # Ridotto per test rapidi

models:
  catboost:   enabled: true   (5 trial, 200 iterations)
  xgboost:    enabled: true   (5 trial)
  lightgbm:   enabled: true   (5 trial)
  rf:         enabled: true   (5 trial)
  gbr:        enabled: false  # Disabilitato
  hgbt:       enabled: false  # Disabilitato

ensembles:
  voting:     enabled: false  # Disabilitato
  stacking:   enabled: true   (top 4, Ridge, CV 5-fold)
```

**Tempo stimato**: ~20 minuti

### Target Transformation

```yaml
target:
  column_candidates: ['AI_Prezzo_Ridistribuito']
  transform: 'yeojohnson'  # ✅ OTTIMO per distribuzione skewed
```

**Motivazione Yeo-Johnson**:
- Gestisce **valori positivi e negativi** (a differenza di Box-Cox)
- **Normalizza** distribuzione con skewness 5.16
- **Riduce impatto outlier** (kurtosis 54.18)
- **Lambda fitted su train**, applicato a val/test (no leakage)

### Outlier Detection

```yaml
outliers:
  method: 'ensemble'  # IQR + Z-score + Isolation Forest
  z_thresh: 2.5
  iqr_factor: 1.0
  iso_forest_contamination: 0.08  # 8% outlier attesi
  group_by_col: 'AI_ZonaOmi'      # Per-zone detection
  min_group_size: 20
  fallback_strategy: 'global'     # Se zona troppo piccola
```

**✅ CORRETTO**: Outlier detection solo su train, no leakage.

---

## ⚠️ POTENZIALI PROBLEMI E RACCOMANDAZIONI

### 1. 🚨 Dataset Size Dopo Filtri

**Problema**: Se il filtro `anno_min: 2022` rimuove 40-60% dei dati, il dataset finale potrebbe essere troppo piccolo per:
- Hyperparameter tuning con 150 trial
- Ensemble stacking con CV 10-fold
- Generalizzazione robusta

**Raccomandazione**:

```bash
# 1. Verifica dimensione dataset post-filtri
python -c "
import pandas as pd
df = pd.read_parquet('data/raw/raw.parquet')
print('Original:', len(df))
df = df[df['A_AnnoStipula'] >= 2022]
print('After anno_min=2022:', len(df))
df = df[~df['AI_ZonaOmi'].isin(['E1','E2','E3','R1'])]
print('After zone filter:', len(df))
"

# 2. Se < 3,000 righe → riduci complessità
#    - Usa config_fast.yaml (5 trial invece di 150)
#    - Riduci cv_folds da 10 a 5
#    - Disabilita alcuni modelli pesanti
```

### 2. 🔍 Verifica Distribuzione Temporale

**Problema**: Non sappiamo se il filtro 2022+ rimuove il 20% o l'80% dei dati.

**Raccomandazione**:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet('data/raw/raw.parquet')

# Distribuzione temporale
temporal_dist = df['A_AnnoStipula'].value_counts().sort_index()
print(temporal_dist)

# Visualizza
temporal_dist.plot(kind='bar', title='Distribuzione Temporale')
plt.xlabel('Anno Stipula')
plt.ylabel('Numero Transazioni')
plt.savefig('temporal_distribution.png')

# Calcola impatto filtro
pct_removed = (df['A_AnnoStipula'] < 2022).sum() / len(df) * 100
print(f"Filtro anno_min=2022 rimuove: {pct_removed:.1f}%")
```

### 3. 📊 Validazione Cross-Zone

**Problema**: Se escludi zone E1/E2/E3/R1, il modello non sarà testato su zone periferiche/rurali.

**Implicazioni**:
- ✅ Modello specializzato su zone urbane/semicentrali
- ❌ Non generalizza a zone periferiche
- ⚠️ Deployment limitato a zone incluse nel training

**Raccomandazione**:

Se il goal è un modello **production generalizzato**:
1. **Non filtrare** le zone, ma usa `group_by_col: 'AI_ZonaOmi'` per valutare performance per-zona
2. **Valuta separatamente** performance su zone centrali vs periferiche
3. **Segnala** all'utente finale se la predizione è su zona out-of-distribution

Se il goal è **test/sperimentazione**:
- ✅ Filtri OK, ma **documenta** che il modello è zone-specific

### 4. 🎯 Tipologie Escluse

**Problema**: Ville (tipologia 4) hanno pattern di pricing diversi (€150k median vs €42k generale).

**Raccomandazione**:

```yaml
# Opzione A: Modelli specializzati
data_filters:
  tipologie_incluse: ['1', '2', '3']  # Solo residenziale standard
  description: "Modello specializzato residenziale (no ville)"

# Opzione B: Mantieni ville, valuta separatamente
evaluation:
  group_metrics:
    group_by_columns: ['AI_IdTipologiaEdilizia']
    # Valuta MAPE per tipologia → identifica se ville performano male
```

### 5. 🧮 Numerical Stability

**Problema**: Prezzi in range €179 - €1,483,526 (ordine di grandezza 10^3 - 10^6).

**Verifica**: La Yeo-Johnson transform **normalizza** questa varianza.

**Raccomandazione**: Monitora se:
- Predizioni su prezzi estremi (< €5k o > €500k) hanno MAPE alto
- In caso affermativo, considera **price band stratification**:

```yaml
evaluation:
  price_band:
    method: 'quantile'
    quantiles: [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    # Valuta MAPE per fascia di prezzo
```

### 6. 🔧 Regularization per Dataset Ridotto

Se il dataset post-filtri è piccolo (<3,000 righe), aumenta regularizzazione:

```yaml
# config.yaml - Modelli più conservativi
training:
  models:
    catboost:
      base_params:
        depth: 4              # Ridotto da 7
        l2_leaf_reg: 10.0     # Aumentato da 6.0
        min_data_in_leaf: 40  # Aumentato
    
    xgboost:
      search_space:
        max_depth: {low: 2, high: 4}     # Ridotto da 3-6
        reg_lambda: {low: 10.0, high: 200.0}  # Aumentato
        min_child_weight: {low: 10.0, high: 40.0}  # Aumentato
```

---

## 🧪 PIANO DI TEST RACCOMANDATO

### Test 1: Baseline su Dataset Completo

```bash
# 1. Disabilita tutti i filtri
# config_test_baseline.yaml
data_filters:
  anno_min: null
  zone_escluse: null
  tipologie_escluse: null

# 2. Train veloce
python main.py --config fast --steps preprocessing training evaluation

# 3. Salva metriche baseline
# R²: ?, RMSE: ?, MAPE: ?
```

### Test 2: Filtri Sperimentali (Config Attuale)

```bash
# 1. Con filtri attuali (anno>=2022, zone escluse, no ville)
python main.py --config fast --steps preprocessing training evaluation

# 2. Confronta con baseline
# - R² migliorato? (atteso: sì, dataset più omogeneo)
# - RMSE ridotto? (atteso: sì, meno outlier)
# - Dataset size? (attenzione se < 3,000)
```

### Test 3: Ablation Studies

```bash
# Test 3a: Solo filtro temporale
data_filters:
  anno_min: 2022
  zone_escluse: null
  tipologie_escluse: null

# Test 3b: Solo filtro zone
data_filters:
  anno_min: null
  zone_escluse: ['E1', 'E2', 'E3', 'R1']
  tipologie_escluse: null

# Test 3c: Solo filtro tipologie
data_filters:
  anno_min: null
  zone_escluse: null
  tipologie_escluse: ['4']

# Confronta: quale filtro ha maggior impatto su performance?
```

### Test 4: Verifica Generalizzazione

```bash
# 1. Train su dati filtrati
# 2. Valuta su dati esclusi (se possibile)

# Esempio: Train su zone centrali, test su E1/E2/E3
# Quanto degrada la performance? → Misura specializzazione
```

---

## 📋 CHECKLIST DEPLOYMENT

Prima di usare il modello in production:

- [ ] **Dataset size**: ≥ 2,000 righe post-filtri (min per training robusto)
- [ ] **Zone coverage**: Documentare quali zone sono supportate
- [ ] **Tipologie coverage**: Documentare quali tipologie sono supportate
- [ ] **Temporal validity**: Modello valido solo per transazioni 2022+
- [ ] **Outlier range**: Definire range prezzo accettabile (es. €5k - €500k)
- [ ] **Monitoring**: Traccia predizioni out-of-distribution
  - Zona non vista in training
  - Tipologia non vista in training
  - Prezzo fuori range training
- [ ] **A/B test**: Confronta con modello baseline (nessun filtro)
- [ ] **Error analysis**: Analizza worst predictions per pattern comuni
- [ ] **Documentation**: README con limitazioni e scope del modello

---

## 🎓 CONCLUSIONI

### ✅ Punti di Forza

1. **No Data Leakage**: Architettura robusta con fit/transform pattern corretto
2. **Feature Engineering**: Feature contestuali leak-free e production-ready
3. **Configurazione Flessibile**: Facile testare diversi subset via YAML
4. **Test Coverage**: Test automatici per encoding, split, target transform
5. **Documentazione**: Codice ben commentato e configurazione chiara

### ⚠️ Aree di Attenzione

1. **Dataset Size**: Verifica che post-filtri ci siano abbastanza dati (≥2,000 righe)
2. **Generalizzazione**: Modello specializzato, non generalizza a zone/tipologie escluse
3. **Temporal Drift**: Modello valido solo per periodo 2022+
4. **Hyperparameter Tuning**: Con dataset ridotto, 150 trial potrebbero essere eccessivi

### 🚀 Prossimi Passi

1. **Analisi Temporale**: Eseguire script per verificare distribuzione `A_AnnoStipula`
2. **Baseline Comparison**: Train su dataset completo vs filtrato
3. **Ablation Studies**: Identificare quale filtro ha maggior impatto
4. **Error Analysis**: Analizzare worst predictions per pattern comuni
5. **Production Readiness**: Decidere se modello è zone-specific o generalizzato

---

## 📚 RIFERIMENTI

### File Chiave Analizzati

```
config/
  - config.yaml (linee 56-80: data_filters)
  - config_fast.yaml (linee 54-72: data_filters identici)

src/preprocessing/
  - pipeline.py (linee 98-212: apply_data_filters)
  - contextual_features.py (linee 161-240: leak-free features)
  - encoders.py (encoding multi-strategy)

src/training/
  - train.py (orchestrator training)
  - tuner.py (Optuna hyperparameter tuning)

tests/
  - test_encoding_no_leakage.py (267 linee, coverage completo)
  - test_temporal_split_fix.py
  - test_target_transforms.py

notebooks/eda_outputs/
  - target_statistics.csv (5,733 righe, skewness 5.16)
  - correlations_with_target.csv (AI_Rendita 0.68, Superficie 0.67)
  - group_summary_AI_ZonaOmi.csv (13 zone, E1/E2/E3/R1 da escludere)
```

### Metriche Target (Pre-Filtri)

```
N:          5,733
Mean:       €62,592
Median:     €42,000
Std:        €79,533
Range:      €179 - €1,483,526
Skewness:   5.16 (molto asimmetrico)
Kurtosis:   54.18 (code pesanti)
```

---

**Analisi completata il**: 2025-11-14  
**Autore**: Claude (Sonnet 4.5)  
**Versione**: 1.0
