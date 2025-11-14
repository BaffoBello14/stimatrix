# 📚 INDICE COMPLETO: Analisi Subset Configuration 2022+

**Branch**: `cursor/analyze-and-test-data-subset-176c`  
**Data**: 2025-11-14  
**Status**: ✅ Analisi completata, architettura leak-free verificata

---

## 🎯 INIZIA QUI

### Per chi ha fretta (5 min)

📊 **[QUICKSTART_ANALYSIS.md](./QUICKSTART_ANALYSIS.md)**  
→ Guida rapida per eseguire analisi impatto filtri

### Per manager/stakeholder (10 min)

📋 **[EXECUTIVE_SUMMARY_FILTERS.md](./EXECUTIVE_SUMMARY_FILTERS.md)**  
→ Sintesi esecutiva: cosa sono i filtri, perché, impatto stimato, raccomandazioni

### Per sviluppatori/data scientist (30 min)

📖 **[ANALISI_SUBSET_CONFIG_2022.md](./ANALISI_SUBSET_CONFIG_2022.md)**  
→ Analisi approfondita: 654 righe di analisi tecnica completa

---

## 📄 DOCUMENTI GENERATI

### 1. EXECUTIVE_SUMMARY_FILTERS.md (9.3 KB)

**Contenuto**:
- Configurazione attuale filtri
- Verifica non-leakage (3 livelli)
- Impatto stimato sui dati
- Raccomandazioni prioritarie
- Metriche attese
- Decisione: modello specializzato vs generalizzato
- Piano prossimi passi

**Audience**: Manager, Product Owner, Stakeholder tecnici  
**Tempo lettura**: 10 minuti

### 2. ANALISI_SUBSET_CONFIG_2022.md (21 KB)

**Contenuto**:
- Executive summary
- Dataset overview completo
- Impatto dettagliato filtri (con calcoli)
- Verifica non-leakage approfondita (con codice)
- Top correlazioni con target
- Configurazione training dettagliata
- 6 potenziali problemi + raccomandazioni
- Piano di test completo (4 test suite)
- Checklist deployment
- Conclusioni e riferimenti

**Audience**: Data Scientist, ML Engineer, Developer  
**Tempo lettura**: 30 minuti

### 3. QUICKSTART_ANALYSIS.md (3.5 KB)

**Contenuto**:
- Comandi rapidi per eseguire analisi
- Cosa aspettarsi dall'output
- Filtri configurati
- Threshold critici
- Troubleshooting comune

**Audience**: Tutti (quick reference)  
**Tempo lettura**: 5 minuti

### 4. analyze_filters_impact.py (13 KB)

**Script Python** per analisi automatica:
- Analisi distribuzione temporale
- Simulazione filtri con statistiche dettagliate
- Confronto distribuzioni pre/post filtri
- Warning automatici se dataset troppo piccolo
- Stima split train/val/test

**Usage**:
```bash
python analyze_filters_impact.py
```

**Output**: Report completo su console + verifiche automatiche

---

## 🔍 ARGOMENTI TRATTATI

### Configurazione

- ✅ Filtri applicati: anno >= 2022, zone escluse (E1/E2/E3/R1), no ville
- ✅ Motivazione: ridurre temporal drift, focus urbano, escludi nicchie
- ✅ Modalità applicazione: pre-split (no leakage)

### Verifica Non-Leakage

- ✅ Filtri applicati pre-split temporale
- ✅ Feature contestuali calcolate solo su train (9 feature rimosse)
- ✅ Encoding multi-strategy fit solo su train (test coverage completo)
- ✅ Split temporale preserva ordinamento cronologico

### Dataset

- ✅ 5,733 righe × 265 colonne
- ✅ Target: €62,592 mean, €42,000 median, skewness 5.16
- ✅ Top feature: AI_Rendita (0.68), AI_Superficie (0.67)
- ✅ 13 zone OMI, 8 categorie catastali

### Impatto Filtri

- ✅ Zone escluse: ~153 righe (2.7%)
- ✅ Tipologie escluse: ~41 righe (0.7%)
- ⚠️ Anno >= 2022: **DA VERIFICARE** (probabile 40-60%)
- ⚠️ Dataset finale stimato: 2,500-4,500 righe

### Raccomandazioni

1. **Immediata**: Eseguire `analyze_filters_impact.py` per conferma dimensioni
2. **Prioritaria**: Se < 3,000 righe → usare `config_fast.yaml`
3. **Consigliata**: Baseline comparison (train con e senza filtri)
4. **Opzionale**: Ablation study (quale filtro impatta di più?)

### Training

- ✅ 6 modelli (CatBoost, XGBoost, LightGBM, RF, GBR, HGBT)
- ✅ 150 trial Optuna (config completo) o 5 trial (config fast)
- ✅ Ensemble: Voting (top 5) + Stacking (top 7, Ridge, CV 10-fold)
- ✅ Target transform: Yeo-Johnson (ottimo per skewness 5.16)
- ✅ Outlier detection: Ensemble (IQR + Z-score + Isolation Forest)

---

## 🚀 QUICK ACTIONS

### Analizza Impatto Filtri

```bash
python analyze_filters_impact.py
```

### Training Veloce con Filtri

```bash
python main.py --config fast --steps preprocessing training evaluation
```

### Training Baseline (No Filtri)

```bash
# 1. Disabilita filtri in config.yaml
data_filters:
  anno_min: null
  zone_escluse: null
  tipologie_escluse: null

# 2. Esegui training
python main.py --config fast --steps preprocessing training evaluation
```

### Confronto Metriche

```bash
# Dopo aver eseguito entrambi i training, confronta:
# - R² (higher is better)
# - RMSE (lower is better)
# - MAPE (lower is better)

# Verifica se filtri migliorano abbastanza da giustificare perdita di generalizzazione
```

---

## 📊 METRICHE & THRESHOLD

### Dimensione Dataset

| Righe Finali | Status | Azione |
|--------------|--------|--------|
| **< 2,000** | 🚨 Critico | Ridurre filtri o complessità |
| **2,000-3,000** | ⚠️ Attenzione | Usare `config_fast.yaml` |
| **3,000-4,000** | ✅ Accettabile | Config normale OK |
| **> 4,000** | ✅ Ottimo | Tutti i config OK |

### Miglioramento Atteso

| Scenario | R² Improvement | RMSE Reduction | Decisione |
|----------|----------------|----------------|-----------|
| **Forte** | > +7 punti | > -20% | ✅ Usa filtri |
| **Medio** | +3-7 punti | -10-20% | ⚠️ Valuta trade-off |
| **Debole** | < +3 punti | < -10% | ❌ No filtri |

### Test Coverage

- ✅ `test_encoding_no_leakage.py`: 8 test, 267 righe
- ✅ `test_temporal_split_fix.py`: Split corretto
- ✅ `test_target_transforms.py`: Transform leak-free
- ✅ `test_preprocessing_pipeline.py`: Pipeline completa

---

## 🎓 CONCLUSIONI CHIAVE

### ✅ Punti di Forza

1. **Architettura robusta**: Fit/transform pattern corretto, no leakage
2. **Test coverage**: Suite completa di test automatici
3. **Configurazione flessibile**: Facile testare diversi subset
4. **Documentazione**: 40+ KB di documentazione tecnica

### ⚠️ Aree di Attenzione

1. **Dataset size**: Verifica che post-filtri sia ≥ 2,000 righe
2. **Generalizzazione**: Modello non generalizza a zone/tipologie escluse
3. **Temporal drift**: Modello valido solo per periodo 2022+
4. **Hyperparameter tuning**: Con dataset ridotto, 150 trial eccessivi

### 🎯 Next Steps

1. ✅ Esegui `analyze_filters_impact.py` → conferma fattibilità
2. ✅ Train baseline + filtrato → confronta metriche
3. ✅ Ablation study → identifica filtro più efficace
4. ✅ Production readiness → documenta scope e limitazioni

---

## 📞 SUPPORTO

### Domande Frequenti

**Q: I filtri causano data leakage?**  
A: ❌ NO. Verificato a 3 livelli (filtri pre-split, feature leak-free, encoding corretto).

**Q: Quanto dataset rimane dopo filtri?**  
A: ⚠️ DA VERIFICARE con `analyze_filters_impact.py`. Stima: 2,500-4,500 righe.

**Q: Posso usare config.yaml con dataset ridotto?**  
A: ⚠️ Se < 3,000 righe, meglio usare `config_fast.yaml` (5 trial vs 150).

**Q: Il modello generalizza a tutte le zone?**  
A: ❌ NO. Modello specializzato, non generalizza a zone E1/E2/E3/R1 escluse.

**Q: Quali metriche confrontare con baseline?**  
A: ✅ R², RMSE, MAPE su test set (scala originale).

### Contatti

- **Issues**: [GitHub Issues]
- **Team**: [Data Science Team]
- **Docs**: Questa cartella (`/workspace/`)

---

## 📁 FILE TREE

```
/workspace/
├── INDICE_ANALISI.md                  # ← Questo file
├── QUICKSTART_ANALYSIS.md             # Quick start guide
├── EXECUTIVE_SUMMARY_FILTERS.md       # Executive summary
├── ANALISI_SUBSET_CONFIG_2022.md      # Analisi approfondita
├── analyze_filters_impact.py          # Script analisi automatica
├── README.md                          # README principale (aggiornato)
├── config/
│   ├── config.yaml                    # Config completo (filtri attivi)
│   └── config_fast.yaml               # Config veloce (filtri attivi)
├── src/preprocessing/
│   ├── pipeline.py                    # apply_data_filters (linee 98-212)
│   └── contextual_features.py         # Leak-free features
├── tests/
│   └── test_encoding_no_leakage.py    # 8 test, 267 righe
├── notebooks/eda_outputs/
│   ├── target_statistics.csv          # 5,733 righe, skew 5.16
│   ├── correlations_with_target.csv   # Top: Rendita 0.68
│   └── group_summary_AI_ZonaOmi.csv   # 13 zone, E1/E2/E3/R1 da escludere
└── data/raw/
    └── raw.parquet                    # 5,733 righe × 265 colonne
```

---

**Analisi completata il**: 2025-11-14  
**Autore**: Claude (Sonnet 4.5)  
**Versione**: 1.0

**Pronto per iniziare?**

```bash
python analyze_filters_impact.py
```
