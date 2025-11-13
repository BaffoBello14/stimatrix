# 📊 ANALISI RISULTATI POST-CLEANUP - Diagnosi e Piano d'Azione

**Data**: 2025-11-12  
**Context**: Primi risultati realistici dopo rimozione data leakage e feature production-ready

---

## 🎯 EXECUTIVE SUMMARY

### **Stato Attuale:**
- ✅ **Codebase pulito** → No data leakage, production-ready
- ⚠️ **Performance accettabile ma bassa** → R²~0.73, MAPE~58%
- ⚠️ **Overfitting moderato** → Gap R²~0.12, Ratio RMSE~2.5x
- ❌ **MAPE troppo alto** → Errore medio del 58% sul prezzo (target <20%)

### **Best Model: CatBoost (non RF come riportato inizialmente)**
```
Scala Trasformata (log):          Scala Originale (€):
  R²:          0.8624                R²:          0.7364
  RMSE:        0.5083                RMSE:        36,768 €
  MAE:         0.3607                MAE:         19,812 €
  MAPE:        0.0364 (3.64%)        MAPE:        58.10%
                                     MAPE floor:  57.52%

Overfit:
  Gap R²:      0.1166
  Ratio RMSE:  2.53x
```

### **Confronto con Risultati Precedenti (Con Leakage):**
| Metrica | Con Leakage ❌ | Senza Leakage ✅ | Delta |
|---------|----------------|------------------|-------|
| R² | ~0.9845 | **0.7364** | -0.2481 |
| RMSE | ~8,911€ | **36,768€** | +27,857€ |
| MAPE | ~2.68% | **58.10%** | +55.42% |
| Gap R² | ~0.02 | **0.1166** | +0.0966 |

**Interpretazione:**
- ✅ **Risultati ora realistici** → Riflettono vera capacità predittiva
- ❌ **Performance insufficiente per produzione** → MAPE target dovrebbe essere <20%

---

## 🔍 ANALISI DETTAGLIATA

### **1. WORST PREDICTIONS - Pattern di Errore**

#### **Top 10 Errori Assoluti (RF):**

| True | Predicted | Errore % | Pattern Identificato |
|------|-----------|----------|----------------------|
| 570,000€ | 251,480€ | **55.9%** | Sottostima immobili di lusso |
| 417,221€ | 224,238€ | **46.3%** | Sottostima fascia alta |
| 379,569€ | 186,625€ | **50.8%** | Sottostima fascia alta |
| **35,128€** | **140,526€** | **300%** | **SOVRASTIMA ENORME fascia bassa** ⚠️ |
| **36,999€** | **142,285€** | **285%** | **SOVRASTIMA ENORME fascia bassa** ⚠️ |
| **1,531€** | **89,838€** | **5,768%** | **OUTLIER - dovrebbe essere filtrato** ❌ |

#### **Diagnosi Pattern:**
1. ✅ **Sottostima sistematica immobili di lusso** (>300k€)
   - Probabile: pochi esempi nel training set
   - Soluzione: Stratified sampling, feature per prezzo/mq zona

2. ❌ **SOVRASTIMA GRAVISSIMA prezzi bassi** (<50k€)
   - **CRITICO**: Errori fino al **5,768%**!
   - Causa: Trasformazione log + outlier detection insufficiente
   - Soluzione: Filtro prezzo_min più alto, Yeo-Johnson transform

3. ❌ **Outlier estremi non filtrati** (prezzo=1,531€)
   - Causa: Outlier detection non abbastanza aggressivo
   - Soluzione: Aumentare contamination, aggiungere filtro min/max price

---

### **2. GROUP METRICS - Segmenti Problematici**

#### **A) Performance per Zona OMI:**

| Zona | N | R² | RMSE | MAPE | Valutazione |
|------|---|-----|------|------|-------------|
| **D2** | 29 | **0.86** | 25,306€ | **39.2%** | ✅ **OTTIMA** |
| **C2** | 26 | **0.81** | 18,800€ | **25.7%** | ✅ **BUONA** |
| C5 | 24 | 0.68 | 21,146€ | 35.3% | ⚠️ Accettabile |
| B1 | 97 | 0.65 | 54,829€ | 39.8% | ⚠️ Accettabile |
| C4 | 54 | 0.65 | 34,746€ | **147%** | ❌ **CRITICO** |
| **C3** | 18 | **-0.32** | 36,805€ | **69%** | ❌ **FALLIMENTO** |

**Key Insights:**
- ✅ **Zone D2, C2**: Modello funziona bene (MAPE <40%)
- ❌ **Zona C3**: R² NEGATIVO (-0.32) → Modello fa **peggio della media**!
- ❌ **Zona C4**: MAPE **147%** → Errori enormi (3 volte il valore!)
- ⚠️ **Zona B1**: Alta RMSE (54k€) ma MAPE ok → Prezzi alti, errori in €

**Azioni:**
1. ✅ Analizzare caratteristiche uniche zone C3 e C4
2. ✅ Considerare modelli specializzati per zona
3. ✅ Verificare se C3/C4 hanno pochi dati (18 e 54 esempi)

#### **B) Performance per Tipologia Edilizia:**

| Tipo | N | R² | RMSE | MAPE | Valutazione |
|------|---|-----|------|------|-------------|
| **8** | 40 | **0.62** | 12,564€ | **33.7%** | ✅ **BUONA** |
| 2 | 92 | 0.50 | 63,483€ | 33.0% | ⚠️ Accettabile |
| 18 | 87 | 0.36 | 10,841€ | 54.4% | ⚠️ Bassa |
| 7 | 13 | 0.29 | 34,386€ | 63.0% | ⚠️ Bassa |
| **3** | 62 | **-0.02** | 29,312€ | 32.2% | ❌ **FALLIMENTO** |

**Key Insights:**
- ✅ **Tipo 8**: Best performance (R²=0.62, MAPE=33.7%)
- ❌ **Tipo 3**: R² quasi negativo (-0.02) → Modello inutile
- ⚠️ **Tipo 7**: Solo 13 esempi → Pochi dati per apprendere

**Azioni:**
1. ✅ Feature engineering specifica per tipo 3
2. ✅ Raggruppare tipologie rare (tipo 7: 13 esempi)
3. ✅ Analizzare se tipo 3 ha caratteristiche uniche

#### **C) Performance per Price Band:**

| Price Band | N | R² | MAPE | Valutazione |
|------------|---|-----|------|-------------|
| 137k-570k (alto) | 31 | **0.02** | 29.5% | ❌ Pessimo R² |
| 97k-138k | 30 | **-13.01** | 29.8% | ❌ **FALLIMENTO** |
| 78k-97k | 30 | **-22.41** | 27.1% | ❌ **FALLIMENTO** |
| 64k-78k | 30 | **-57.34** | 28.2% | ❌ **FALLIMENTO** |
| 48k-64k | 30 | **-18.70** | 28.7% | ❌ **FALLIMENTO** |
| 31k-48k | 31 | **-66.70** | **74.5%** | ❌ **CRITICO** |
| **18k-31k** | 30 | **-3.56** | 31.1% | ❌ **FALLIMENTO** |
| **10k-18k** | 30 | **-10.24** | 35.9% | ❌ **FALLIMENTO** |
| **5k-10k** | 30 | **-70.33** | **79.0%** | ❌ **CRITICO** |
| **NaN** | 31 | **-178.63** | **232%** | ❌ **DISASTROSO** |

**Key Insights:**
- ❌ **TUTTI i price band hanno R² NEGATIVO** (tranne il più alto con R²=0.02)
- ❌ **Fascia bassa (5k-10k)**: R²=-70, MAPE=79% → **CRITICO**
- ❌ **Price band NaN**: R²=-178 → Possibile problema preprocessing
- ⚠️ MAPE relativamente buoni (28-35%) tranne fasce estreme

**Diagnosi:**
- **R² negativi significano che il modello fa PEGGIO della media semplice**
- Il problema è **sistematico** su quasi tutti i price band
- La trasformazione log + feature attuali **NON catturano pattern prezzo**

**Azioni CRITICHE:**
1. ❌ **Rivedere completamente strategia trasformazione target**
2. ❌ **Aggiungere feature esplicite per price band/quantili**
3. ❌ **Rimuovere outlier con price NaN o fuori range** (5k-10k, 500k+)
4. ❌ **Considerare modelli stratificati per fascia di prezzo**

---

### **3. OVERFIT ANALYSIS - Tutti i Modelli**

| Modello | Gap R² | Ratio RMSE | Valutazione |
|---------|--------|------------|-------------|
| LightGBM | 0.092 | 1.55x | ✅ Basso |
| **RF** | **0.100** | **1.67x** | ✅ **Basso** |
| **CatBoost** | **0.117** | **2.53x** | ⚠️ **Moderato** |
| HGBT | 0.132 | 2.29x | ⚠️ Moderato |
| Voting | 0.135 | 2.61x | ⚠️ Moderato |
| GBR | 0.138 | 2.55x | ⚠️ Moderato |
| Stacking | 0.141 | 3.37x | ❌ **Alto** |
| **XGBoost** | **0.144** | **2.82x** | ❌ **Alto** |

**Key Insights:**
- ✅ **RF e LightGBM**: Overfit più basso (gap R²<0.10)
- ⚠️ **Maggioranza modelli**: Overfit moderato (gap R²~0.12-0.14)
- ❌ **Stacking**: Overfit PEGGIORE (gap R²=0.14, ratio=3.37x) → Ensemble non aiuta!
- ⚠️ Tutti hanno ratio RMSE > 1.5x → Train accuracy molto superiore a test

**Cause Possibili:**
1. ✅ **Dataset relativamente piccolo** → Pochi esempi per zone/tipologie rare
2. ✅ **Feature troppo specifiche** → 28 feature production-ready potrebbero non bastare
3. ✅ **Regularization insufficiente** → Hyperparameter tuning non ha trovato ottimo
4. ✅ **Mancanza di data augmentation** → Pochi esempi fasce estreme

---

### **4. FEATURE IMPORTANCE - Analysis**

*Nota: Feature importance plots non sono stati generati in questa run, ma possiamo dedurre da group metrics:*

**Feature Probabilmente Importanti:**
1. ✅ **AI_ZonaOmi** → Performance varia moltissimo per zona (0.86 vs -0.32)
2. ✅ **AI_IdTipologiaEdilizia** → Performance varia per tipo (0.62 vs -0.02)
3. ✅ **AI_Superficie** → (se non droppata) Base per prezzo/mq
4. ✅ **Contextual features zona** → zone_prezzo_mean, zone_count, etc.

**Feature Probabilmente Mancanti:**
1. ❌ **Prezzo/mq zona** → Rimossa perché richiedeva target istanza
2. ❌ **Percentile prezzo** → Feature derivata dal target
3. ❌ **Interazioni prezzo*zona** → Rimosse per production-readiness
4. ❌ **Target-encoded features** → Rimosse per evitare leakage

---

## 🎯 DIAGNOSI COMPLESSIVA

### **Problemi Identificati (Priorità):**

#### **🔥 CRITICO (Risolvere Subito):**

1. **Outlier Detection Insufficiente**
   - Prezzo=1,531€ con errore 5,768% non dovrebbe esistere
   - Prezzo NaN con R²=-178 indica preprocessing fallito
   - **Azione**: Aumentare contamination da 0.08 a 0.15, filtro min_price

2. **Trasformazione Target Inadeguata per Range Estremi**
   - Log transform non gestisce bene fascia 5k-500k€
   - R² negativi su TUTTI i price band
   - **Azione**: Provare Yeo-Johnson, stratified modeling

3. **Overfitting Moderato-Alto**
   - Gap R²~0.12, Ratio RMSE~2.5x
   - Modello impara pattern specifici train che non generalizzano
   - **Azione**: Aumentare regularization, early stopping, dropout

#### **⚠️ IMPORTANTE (Risolvere Presto):**

4. **Zone/Tipologie Problematiche**
   - Zona C3: R²=-0.32, Zona C4: MAPE=147%
   - Tipo 3: R²=-0.02
   - **Azione**: Modelli specializzati, feature engineering mirata

5. **Feature Production-Ready Insufficienti**
   - Rimosse 9 feature che usavano target → Performance drop
   - 28 feature potrebbero non catturare tutta la complessità
   - **Azione**: Feature engineering nuovo, interactions, polynomial

6. **Dataset Size Limitato per Segmenti**
   - Zona C3: 18 esempi, Tipo 7: 13 esempi
   - Troppo pochi per apprendere pattern robusti
   - **Azione**: Data augmentation, transfer learning, grouping

#### **🔹 MIGLIORAMENTO (Nice to Have):**

7. **Ensemble Non Aiutano**
   - Stacking: gap R²=0.14 (peggio di CatBoost 0.12)
   - Voting: simile a singoli modelli
   - **Azione**: Tuning ensemble, diversità modelli base

8. **Fascia Alta Sottostimata**
   - Immobili >300k€ sistematicamente sottostimati
   - **Azione**: Feature luxuryscore, weight sampling

---

## 🚀 PIANO D'AZIONE - Step by Step

### **FASE 1: Quick Wins (1-2 giorni)**

#### **1.1 Filtro Outlier Più Aggressivo**

**File**: `config/config_optimized.yaml`

```yaml
# Opzione A: Aumentare contamination outlier detection
outliers:
  iso_forest_contamination: 0.15  # ✅ Da 0.08 → 0.15

# Opzione B: Aggiungere filtri prezzo espliciti (data_filters già presente!)
data_filters:
  prezzo_min: 20000   # ✅ Rimuovi prezzi <20k (outlier/errori)
  prezzo_max: 500000  # ✅ Rimuovi prezzi >500k (outlier/lusso)
  superficie_min: 10  # ✅ Rimuovi superficie <10mq (errori)
  superficie_max: 300 # ✅ Rimuovi superficie >300mq (outlier)
```

**Impatto Atteso**: RMSE -5-10%, MAPE -10-15%, R² +0.05-0.10

---

#### **1.2 Trasformazione Target: Log → Yeo-Johnson**

**File**: `config/config_optimized.yaml`

```yaml
target:
  transform: 'yeojohnson'  # ✅ Da 'log' → 'yeojohnson'
  # Yeo-Johnson gestisce meglio range estremi e non richiede y>0
```

**Impatto Atteso**: R² +0.02-0.05, MAPE -5-10%

---

#### **1.3 Aumentare Regularization CatBoost**

**File**: `config/config_optimized.yaml`

```yaml
catboost:
  base_params:
    l2_leaf_reg: 5.0      # ✅ Da 3.0 → 5.0
  search_space:
    l2_leaf_reg:
      low: 3.0            # ✅ Da 1.0 → 3.0
      high: 10.0          # ✅ Da 5.0 → 10.0
```

**Impatto Atteso**: Gap R² -0.02-0.05, Ratio RMSE -0.2-0.5x

---

### **FASE 2: Feature Engineering (3-5 giorni)**

#### **2.1 Aggiungere Feature Price Band (Production-Ready)**

**Nuovo File**: `src/preprocessing/price_features.py`

```python
def add_price_band_features(df: pd.DataFrame, train_stats: Dict = None) -> pd.DataFrame:
    """
    Aggiunge feature price band production-ready (NO target leak).
    
    Features calcolate SOLO da zona/tipo/superficie (no prezzo istanza):
    - prezzo_mq_zona_median (da train)
    - prezzo_stimato = superficie * prezzo_mq_zona_median
    - price_quantile_zona (calcolato su train, applicato a test)
    """
    # ... implementazione ...
    return df
```

**Features da Aggiungere (10 nuove):**
1. `prezzo_mq_zona_median` (mediana zona su train)
2. `prezzo_mq_tipo_median` (mediana tipo su train)
3. `prezzo_stimato` (superficie * prezzo_mq_zona)
4. `superficie_vs_zona_median_ratio`
5. `zona_price_volatility` (std/mean prezzo zona)
6. `tipo_price_volatility`
7. `is_high_price_zone` (zona in top 20% prezzi)
8. `is_low_price_zone` (zona in bottom 20% prezzi)
9. `zona_tipo_interaction` (combinazione zona+tipo)
10. `prezzo_norm_zona` (prezzo_stimato normalizzato per zona)

**Impatto Atteso**: R² +0.05-0.10, MAPE -10-15%

---

#### **2.2 Feature Interactions Polynomial**

**File**: `config/config_optimized.yaml`

```yaml
feature_extraction:
  polynomial:
    enabled: true
    degree: 2  # Interazioni ordine 2
    include_bias: false
    interaction_only: true  # Solo interactions (no quadrati)
    features:  # Solo feature chiave
      - AI_Superficie
      - AI_Locali
      - A_AnnoStipula
      - zone_prezzo_mean
      - type_prezzo_mean
```

**Impatto Atteso**: R² +0.02-0.05 (rischio overfit)

---

### **FASE 3: Stratified Modeling (5-7 giorni)**

#### **3.1 Modelli Specializzati per Fascia Prezzo**

**Strategia**: Dividere dataset in 3 fasce, addestrare modello dedicato

```python
# Pseudocode
fasce = {
    'low': prezzo < 50k,
    'mid': 50k <= prezzo <= 150k,
    'high': prezzo > 150k
}

for fascia, subset in fasce.items():
    model = train_catboost(subset)
    save_model(f"catboost_{fascia}")
```

**Impatto Atteso**: R² +0.05-0.10 per fascia, MAPE -10-20%

---

### **FASE 4: Advanced Techniques (Opzionale, 7-14 giorni)**

#### **4.1 Transfer Learning da Modello Pre-training**

- Pre-train su dataset nazionale (se disponibile)
- Fine-tune su dataset locale

#### **4.2 Attention Mechanism per Zone**

- Implementare attention layer per imparare importanza zone dinamicamente

#### **4.3 Ensemble Diversificato**

- Random Forest + CatBoost + XGBoost + MLP
- Stacking con meta-features

---

## 📈 ROADMAP RACCOMANDAZIONI

### **Settimana 1 (FASE 1 - Quick Wins):**
1. ✅ Implementare filtri outlier (prezzo_min=20k, prezzo_max=500k)
2. ✅ Cambiare trasformazione target → Yeo-Johnson
3. ✅ Aumentare regularization CatBoost
4. ✅ Eseguire training e confrontare risultati
5. ✅ **Target**: MAPE < 50%, R² > 0.75

### **Settimana 2 (FASE 2 - Feature Engineering):**
1. ✅ Implementare price_features.py (10 nuove feature)
2. ✅ Aggiungere polynomial interactions
3. ✅ Testare feature importance → rimuovere feature inutili
4. ✅ Eseguire training e confrontare
5. ✅ **Target**: MAPE < 40%, R² > 0.80

### **Settimana 3-4 (FASE 3 - Stratified Modeling):**
1. ✅ Implementare pipeline stratified per fascia prezzo
2. ✅ Addestrare 3 modelli specializzati (low/mid/high)
3. ✅ Testare ensemble predizioni
4. ✅ **Target**: MAPE < 30%, R² > 0.85

### **Mese 2+ (FASE 4 - Advanced, Opzionale):**
1. ✅ Transfer learning
2. ✅ Attention mechanism
3. ✅ **Target**: MAPE < 20%, R² > 0.90 (production-ready)

---

## 🎯 METRICHE TARGET FINALI

| Metrica | Attuale | Target Q1 (1 mese) | Target Q2 (3 mesi) | Production-Ready |
|---------|---------|-------------------|-------------------|------------------|
| **R²** | 0.736 | **0.80** | **0.85** | **≥0.90** |
| **RMSE** | 36,768€ | **<30,000€** | **<25,000€** | **<20,000€** |
| **MAPE** | 58.10% | **<40%** | **<30%** | **<20%** |
| **Gap R²** | 0.117 | **<0.08** | **<0.05** | **<0.03** |
| **Ratio RMSE** | 2.53x | **<2.0x** | **<1.5x** | **<1.3x** |

---

## ✅ CONCLUSIONI

### **Stato Attuale: ACCETTABILE MA INSUFFICIENTE PER PRODUZIONE**

**Positivo:**
- ✅ Codebase pulito, no data leakage, production-ready
- ✅ R² discreto (0.74) → Modello cattura pattern generali
- ✅ Alcune zone/tipologie funzionano bene (D2, C2, tipo 8)

**Critico:**
- ❌ MAPE troppo alto (58%) → Errori inaccettabili per business
- ❌ R² negativi su price band → Modello non generalizza per fasce prezzo
- ❌ Outlier non filtrati → Errori fino al 5,768%
- ❌ Zone problematiche (C3, C4) → Performance disastrosa

### **Prossimi Step Immediati:**

**1. Quick Fix (Oggi):**
```yaml
# config/config_optimized.yaml
data_filters:
  prezzo_min: 20000
  prezzo_max: 500000

outliers:
  iso_forest_contamination: 0.15

target:
  transform: 'yeojohnson'
```

**2. Eseguire Training:**
```bash
python run_fixed_training.py
```

**3. Analizzare Risultati:**
- Verificare MAPE < 50%
- Verificare R² > 0.75
- Verificare no più outlier estremi (1,531€)

**4. Se OK → FASE 2 Feature Engineering**

---

**Remember**: La strada per un modello production-ready è lunga, ma abbiamo una base solida e un piano chiaro! 🚀
