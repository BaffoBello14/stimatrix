# 📚 PROPOSTA: Notebook Aggiuntivi Utili

Basato sull'analisi del progetto e le best practices ML, ecco i notebook più utili da implementare.

---

## 🎯 PRIORITÀ ALTA

### 1. **`target_transformations_comparison.ipynb`** ⭐⭐⭐

**Obiettivo**: Confrontare TUTTE le trasformazioni del target per scegliere la migliore.

**Contenuto**:

```python
Trasformazioni da confrontare:
1. None (originale)
2. Log (log(x + offset))
3. Log10 (log10(x + offset))
4. Sqrt (√x)
5. Box-Cox (parametrico, solo x > 0)
6. Yeo-Johnson (parametrico, anche x ≤ 0) ← Attualmente usato
7. Quantile (QuantileTransformer)
8. PowerTransformer
9. RobustScaler (per outlier)

Per ogni trasformazione:
- Distribuzione (histogram + KDE)
- Q-Q plot (normalità)
- Skewness & Kurtosis
- Statistiche descrittive
- Shapiro-Wilk test (normalità)
- Anderson-Darling test
- Jarque-Bera test

Output:
- Tabella comparativa completa
- Griglia di plot (3×3) con tutte le distribuzioni
- Q-Q plots affiancati
- Metriche normalità
- Raccomandazione automatica: "Migliore trasformazione: Yeo-Johnson"
```

**Perché è utile**:
- ✅ Valida se Yeo-Johnson è davvero la scelta migliore
- ✅ Mostra visivamente le differenze
- ✅ Test statistici per oggettività
- ✅ Riutilizzabile: cambia target e riesegui

**Stima tempo**: 2-3 ore implementazione, 5 min esecuzione

---

### 2. **`outlier_detection_analysis.ipynb`** ⭐⭐⭐

**Obiettivo**: Analizzare outlier con diversi metodi e validare scelta ensemble.

**Contenuto**:

```python
Metodi da confrontare:
1. IQR (Interquartile Range)
2. Z-Score (μ ± k·σ)
3. Modified Z-Score (median-based)
4. Isolation Forest ← Usato nel progetto
5. LOF (Local Outlier Factor)
6. DBSCAN
7. Elliptic Envelope
8. One-Class SVM

Per ogni metodo:
- Numero outlier rilevati
- Outlier rilevati sovrapposti tra metodi (Venn diagram)
- Scatter plot con outlier evidenziati
- Distribuzione dopo rimozione
- Impatto su statistiche (mean, median, std)

Analisi per gruppo:
- Outlier per zona OMI
- Outlier per categoria catastale
- Outlier per price band

Ensemble method (config attuale):
- Combinazione IQR + Z-Score + Isolation Forest
- Soglie: IQR factor=1.0, Z=2.5, ISO contamination=0.08
- Validazione: quanti outlier identificati?
- Confronto con singoli metodi
```

**Perché è utile**:
- ✅ Valida configurazione outlier attuale
- ✅ Identifica pattern negli outlier (zone problematiche?)
- ✅ Test sensibilità parametri
- ✅ Mostra trade-off rimozione vs conservazione

**Stima tempo**: 3-4 ore implementazione, 5-10 min esecuzione

---

### 3. **`encoding_strategies_comparison.ipynb`** ⭐⭐

**Obiettivo**: Confrontare strategie encoding per feature categoriche.

**Contenuto**:

```python
Per feature categorica del dataset:

1. Analisi cardinalità:
   - OneHot: ≤8 categorie
   - Target Encoding: 9-20 categorie ← Potenziale leakage!
   - Frequency: 21-100 categorie
   - Ordinal: 101-200 categorie
   - Drop: >200 categorie

2. Per ogni feature categorica:
   - Cardinalità (numero categorie uniche)
   - Strategia scelta (secondo config)
   - Distribuzione categorie (top 10 + altre)
   - Correlazione con target (per categoria)
   - Missing rate

3. Target Encoding analysis (CRITICO):
   - Cross-validation per evitare leakage
   - Smoothing effect (min_samples_leaf=10)
   - Confronto media globale vs per categoria
   - Verifica no leakage: encoding calcolato solo su train

4. Comparison matrix:
   Feature | Cardinalità | Strategy | Leakage Risk | Performance
   AI_ZonaOmi | 13 | Target | ✅ OK (CV) | +5% R²

5. Alternative encoding:
   - Leave-One-Out encoding
   - James-Stein encoding
   - M-estimator
   - CatBoost encoding (CTR)
```

**Perché è utile**:
- ✅ Valida che encoding sia leak-free
- ✅ Confronta strategie alternative
- ✅ Identifica feature categoriche problematiche
- ✅ Ottimizza soglie cardinalità

**Stima tempo**: 3-4 ore implementazione, 10 min esecuzione

---

## 🎯 PRIORITÀ MEDIA

### 4. **`feature_importance_deep_dive.ipynb`** ⭐⭐

**Obiettivo**: Analisi feature importance con più metodi.

**Contenuto**:

```python
Metodi di feature importance:

1. Tree-based importance (Gini, split count)
2. Permutation importance
3. SHAP values (già nel progetto)
4. Partial Dependence Plots
5. Individual Conditional Expectation
6. Correlation-based importance
7. Recursive Feature Elimination

Per ogni metodo:
- Ranking top 30 feature
- Confronto ranking tra metodi
- Agreement score (quante feature comuni?)

Feature groups analysis:
- Feature catastali (superficie, rendita, ...)
- Feature ISTAT (indicatori socio-economici)
- Feature OMI (valori mercato)
- Feature POI (punti di interesse)
- Feature contestuali (zona_price_mean, type_zone_count, ...)

Actionable insights:
- Feature ridondanti da droppare
- Feature mancanti ma potenzialmente utili
- Feature interaction da esplorare
```

**Perché è utile**:
- ✅ Valida feature engineering
- ✅ Identifica feature da rimuovere
- ✅ Suggerisce nuove feature
- ✅ Confronta metodi (robustezza)

**Stima tempo**: 4-5 ore implementazione, 15-20 min esecuzione

---

### 5. **`temporal_drift_analysis.ipynb`** ⭐⭐

**Obiettivo**: Analizzare drift temporale tra train/val/test.

**Contenuto**:

```python
Analisi per split:

1. Distribuzione target per split:
   - Train: 2019-2021 (o fino a soglia)
   - Val: 2021-2022
   - Test: 2022-2023
   - Confronto statistiche (mean, std, skewness)

2. Feature drift detection:
   - PSI (Population Stability Index) per feature
   - KS test (Kolmogorov-Smirnov)
   - Chi-square per categoriche
   - Drift score per feature

3. Concept drift:
   - Cambio relazione feature-target nel tempo?
   - Correlazioni temporali (rolling window)
   - Zone con drift maggiore

4. Covariate shift:
   - Distribuzione feature cambia tra split?
   - Quale feature ha maggior shift?

5. Raccomandazioni:
   - Se PSI > 0.25: feature instabile, considera rimozione
   - Se KS p-value < 0.05: distribuzione diversa
   - Strategie: retrain periodico, time-based features, ...
```

**Perché è utile**:
- ✅ Valida split temporale
- ✅ Identifica feature instabili
- ✅ Spiega performance drop su test
- ✅ Suggerisce strategie mitigazione

**Stima tempo**: 3-4 ore implementazione, 10 min esecuzione

---

### 6. **`model_results_deep_analysis.ipynb`** ⭐⭐

**Obiettivo**: Analisi approfondita risultati modelli trained.

**Contenuto**:

```python
1. Performance comparison:
   - Tabella modelli (R², RMSE, MAE, MAPE)
   - Ranking per metrica
   - Overfitting check (train vs val vs test)
   - Ensemble performance

2. Error analysis:
   - Scatter plot predicted vs actual
   - Residual plot (eteroschedasticità?)
   - Residual distribution (normalità?)
   - Q-Q plot residui

3. Error by group:
   - MAPE per zona OMI (heatmap)
   - MAPE per tipologia
   - MAPE per price band (quantili)
   - Worst zones/types identificati

4. Worst predictions:
   - Top 50 worst predictions
   - Pattern comuni? (sempre stessa zona? prezzo estremo?)
   - Feature values per worst cases
   - Suggerimenti miglioramento

5. Prediction intervals (se abilitati):
   - Coverage analysis (80%, 90%)
   - Calibration plot
   - Width vs price (heteroscedastic?)
   - Under/over-confident predictions

6. Ensemble analysis:
   - Diversità tra modelli (correlation predictions)
   - Peso ottimale per voting
   - Stacking meta-learner coefficients
   - Quando ensemble batte singoli?
```

**Perché è utile**:
- ✅ Capire perché modello sbaglia
- ✅ Identificare gruppi problematici
- ✅ Validare ensemble
- ✅ Iterare su feature engineering

**Stima tempo**: 4-5 ore implementazione, 10-15 min esecuzione

---

## 🎯 PRIORITÀ BASSA (Nice-to-have)

### 7. **`correlation_analysis_advanced.ipynb`** ⭐

**Obiettivo**: Analisi correlazioni approfondita.

**Contenuto**:

```python
1. Correlation matrix completa (heatmap)
2. Correlazioni multiple:
   - Pearson (lineare)
   - Spearman (monotonica)
   - Kendall (ranking)
3. Partial correlation (controlling for confounders)
4. Multicollinearity detection (VIF)
5. Feature clusters (correlazione alta tra feature)
6. Correlation networks (graph visualization)
```

---

### 8. **`data_quality_report.ipynb`** ⭐

**Obiettivo**: Report qualità dati.

**Contenuto**:

```python
1. Missing values:
   - Pattern (MCAR, MAR, MNAR?)
   - Imputation validation
2. Duplicates detection
3. Data types consistency
4. Range validation (min/max plausibili?)
5. Categorical consistency
6. Temporal consistency
7. Cross-field validation
```

---

### 9. **`shap_explainability.ipynb`** ⭐

**Obiettivo**: Interpretabilità modello con SHAP.

**Contenuto**:

```python
1. SHAP summary plot
2. SHAP dependence plots (top 10 feature)
3. SHAP waterfall (singole predizioni)
4. SHAP force plot
5. SHAP interaction values
6. Feature clustering based on SHAP
```

---

## 📊 RIEPILOGO RACCOMANDAZIONI

### Implementa SUBITO (questa settimana):

1. **`target_transformations_comparison.ipynb`** ⭐⭐⭐
   - Risponde alla tua domanda specifica
   - Valida Yeo-Johnson
   - 2-3 ore implementazione

2. **`outlier_detection_analysis.ipynb`** ⭐⭐⭐
   - Critico per qualità dati
   - Valida config attuale
   - 3-4 ore implementazione

### Implementa DOPO (prossime iterazioni):

3. **`encoding_strategies_comparison.ipynb`** ⭐⭐
   - Valida no leakage
   - 3-4 ore implementazione

4. **`model_results_deep_analysis.ipynb`** ⭐⭐
   - Dopo primo training completo
   - 4-5 ore implementazione

5. **`temporal_drift_analysis.ipynb`** ⭐⭐
   - Se noti performance drop su test
   - 3-4 ore implementazione

### Opzionali (se hai tempo):

6. Feature importance deep dive
7. Correlation analysis advanced
8. Data quality report
9. SHAP explainability

---

## 🎯 LA MIA RACCOMANDAZIONE TOP

**Implementa PRIMA**: `target_transformations_comparison.ipynb`

**Perché**:
1. Hai chiesto specificamente questo
2. È veloce da implementare (2-3 ore)
3. È veloce da eseguire (5 min)
4. Output immediato e visivamente chiaro
5. Valida scelta cruciale (Yeo-Johnson vs alternative)
6. Riutilizzabile per altri target (AI_Prezzo_MQ, etc.)

**Struttura proposta**:

```
1. Load data & filtri
2. Per ogni trasformazione (9 totali):
   - Applica trasformazione
   - Calcola statistiche (skew, kurtosis, ...)
   - Test normalità (Shapiro, Anderson, Jarque-Bera)
   - Plot (histogram, KDE, Q-Q)
3. Comparison table con tutte le metriche
4. Grid di plot (3×3) per confronto visivo
5. Raccomandazione automatica: "Best: Yeo-Johnson (lowest skew, highest normality test p-value)"
6. Export report JSON + plots
```

**Output atteso**:

```
Transformation Comparison Table:
                       Skew  Kurtosis  Shapiro_p  Best_Score
None (Original)        5.16     54.18      0.000       0.12
Log                    2.34     12.45      0.003       0.45
Log10                  2.34     12.45      0.003       0.45
Sqrt                   3.21     22.34      0.001       0.32
Box-Cox (λ=-0.23)      0.87      2.34      0.082       0.78
Yeo-Johnson (λ=0.12)   0.45      1.23      0.124       0.92  ← BEST!
Quantile               0.02      0.01      0.856       0.95  ← Very good but loses interpretability
PowerTransformer       0.51      1.45      0.098       0.88
```

---

## 🚀 VUOI CHE IMPLEMENTI?

Dimmi quale notebook vuoi che implementi per primo e procedo!

Opzioni:
- A) `target_transformations_comparison.ipynb` (tua richiesta)
- B) `outlier_detection_analysis.ipynb` (alta priorità)
- C) Altro dalla lista
- D) Tutti e 2 (A + B)

Quanto tempo hai? Se hai 2-3 ore, faccio il notebook transformations completo.
