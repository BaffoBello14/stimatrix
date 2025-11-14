# 🎯 PREDICTION INTERVALS - Spiegazione Completa

## 📖 COS'È UN PREDICTION INTERVAL?

Un **prediction interval** (intervallo di previsione) è un **range di valori** entro cui ci aspettiamo che cada il valore reale con una certa probabilità.

### Differenza con Confidence Interval

| Tipo | Cosa Stima | Esempio |
|------|------------|---------|
| **Confidence Interval** | Incertezza sul **parametro del modello** (media) | "Il prezzo medio è €50k ± €2k con 95% confidence" |
| **Prediction Interval** | Incertezza sulla **singola predizione** | "Questo appartamento vale €50k, ma potrebbe essere tra €30k e €70k con 90% probability" |

**Prediction interval è SEMPRE più largo** perché include:
1. Incertezza del modello (quanto è preciso?)
2. Variabilità intrinseca dei dati (case simili hanno prezzi diversi)

---

## 🔬 METODO: RESIDUAL BOOTSTRAP

Il progetto usa il metodo **Residual Bootstrap** per calcolare i prediction intervals.

### Come Funziona (Step-by-Step)

```python
# 1. TRAINING: Fit del modello
model.fit(X_train, y_train)

# 2. TRAINING: Calcola residui (errori) sul training set
y_train_pred = model.predict(X_train)
train_residuals = y_train - y_train_pred
# Esempio residui: [-5000, +3000, -8000, +2000, ...]

# 3. TEST: Fai predizione base
y_test_pred = model.predict(X_test)
# Esempio: previsione = €50,000

# 4. BOOTSTRAP: Simula incertezza
for i in range(n_bootstraps):  # Default: 100 volte
    # Campiona casualmente dai residui del training
    bootstrapped_residuals = np.random.choice(train_residuals, size=len(y_test))
    
    # Aggiungi il residuo campionato alla predizione
    y_test_with_uncertainty[i] = y_test_pred + bootstrapped_residuals

# 5. INTERVALLO: Calcola percentili
for confidence_level in [0.8, 0.9]:
    lower_percentile = (1 - confidence_level) / 2  # 10% per 80%, 5% per 90%
    upper_percentile = 1 - lower_percentile        # 90% per 80%, 95% per 90%
    
    lower_bound = np.percentile(y_test_with_uncertainty, lower_percentile * 100, axis=0)
    upper_bound = np.percentile(y_test_with_uncertainty, upper_percentile * 100, axis=0)
```

### Esempio Concreto

```
Appartamento da predire:
  - Features: 80 mq, zona B1, 3 locali, ...

1. Predizione base:
   y_pred = €50,000

2. Residui training (esempi):
   [-8000, -3000, -1000, +2000, +5000, +12000, ...]

3. Bootstrap (100 simulazioni):
   Simulazione 1: €50,000 + (-3000) = €47,000
   Simulazione 2: €50,000 + (+5000) = €55,000
   Simulazione 3: €50,000 + (-8000) = €42,000
   ...
   Simulazione 100: €50,000 + (+2000) = €52,000

4. Ordina i 100 valori simulati:
   [€38k, €40k, €42k, ..., €48k, €50k, €52k, ..., €65k, €68k, €72k]

5. Intervallo 80% (percentili 10-90):
   Lower: €43,000 (10° percentile)
   Upper: €58,000 (90° percentile)
   Width: €15,000
   
   Interpretazione: "C'è l'80% di probabilità che il prezzo reale sia tra €43k e €58k"

6. Intervallo 90% (percentili 5-95):
   Lower: €38,000 (5° percentile)
   Upper: €65,000 (95° percentile)
   Width: €27,000
   
   Interpretazione: "C'è il 90% di probabilità che il prezzo reale sia tra €38k e €65k"
```

---

## 📊 INTERPRETAZIONE DEI RISULTATI

### I Tuoi Risultati

```json
{
  "80%": {
    "coverage": 0.7887788778877888,      // ← 78.9% effettivo
    "average_width": 124204.26349175065, // ← €124k di larghezza media
    "average_width_pct": 211103.6828245542, // ← ⚠️ ERRORE (vedi sotto)
    "target_coverage": 0.8               // ← 80% atteso
  },
  "90%": {
    "coverage": 0.8778877887788779,      // ← 87.8% effettivo
    "average_width": 162661.94656711374, // ← €162k di larghezza media
    "average_width_pct": 275776.4360946532, // ← ⚠️ ERRORE
    "target_coverage": 0.9               // ← 90% atteso
  }
}
```

### Metrica 1: **Coverage** ✅

**Definizione**: Percentuale di volte che l'intervallo contiene il valore reale.

```python
coverage = (y_test >= lower_bound) & (y_test <= upper_bound).mean()
```

**I Tuoi Valori**:
- **80% interval**: coverage = 78.9% → Leggermente **sotto-calibrato** (atteso 80%)
- **90% interval**: coverage = 87.8% → Leggermente **sotto-calibrato** (atteso 90%)

**Interpretazione**:
- ✅ **78.9%** è vicino a 80% → Accettabile (margine ±5%)
- ✅ **87.8%** è vicino a 90% → Accettabile
- ⚠️ Sotto-calibrazione indica che gli intervalli sono leggermente **troppo stretti**
- 💡 Causa possibile: Residui del training non rappresentano bene variabilità del test

**È un problema?**
- ❌ **NO** se differenza < 5% (come nel tuo caso)
- ⚠️ **SÌ** se differenza > 10% (es. coverage 70% per target 90%)

### Metrica 2: **Average Width** ✅

**Definizione**: Larghezza media degli intervalli in euro.

```python
average_width = np.mean(upper_bound - lower_bound)
```

**I Tuoi Valori**:
- **80% interval**: €124,204 di larghezza media
- **90% interval**: €162,662 di larghezza media

**Interpretazione**:

Per contestualizzare, confrontiamo con il target:
- Target mean: €62,592 (dal tuo dataset)
- Target median: €42,000

```
80% Interval Width / Target Mean = €124k / €62k = 198%
80% Interval Width / Target Median = €124k / €42k = 296%

Esempio concreto:
  Predizione: €50,000
  Intervallo 80%: €50,000 ± €62,000 → [circa €25k - €75k]
  Intervallo 90%: €50,000 ± €81,000 → [circa €20k - €90k]
```

**È troppo largo?**
- ⚠️ **SÌ**, gli intervalli sono **molto larghi** (2-3x il prezzo medio!)
- 💡 Significa alta incertezza nelle predizioni
- 📊 Tipico quando:
  - Dataset piccolo (hai ~5,733 righe, post-filtri forse 3,000)
  - Target molto skewed (skewness 5.16 nel tuo caso)
  - Alta varianza residui (coefficiente variazione 127%)

**Come ridurli?**
1. ✅ Migliorare performance del modello (R² più alto → residui più piccoli)
2. ✅ Dataset più grande (più training data → stima migliore variabilità)
3. ✅ Feature engineering migliori (catturare pattern mancanti)
4. ⚠️ Usare metodi alternativi (quantile regression, conformal prediction)

### Metrica 3: **Average Width PCT** ❌ (BUG!)

**I Tuoi Valori**:
- **80% interval**: 211,103% (!)
- **90% interval**: 275,776% (!)

**⚠️ ERRORE NEL CALCOLO**: Questi valori sono assurdi (>2000%).

**Probabile causa** nel codice:

```python
# Codice attuale (ERRATO):
average_width_pct = np.mean(100 * (upper - lower) / y_pred)  # ← Divisione per y_pred!

# Problema: se y_pred è piccolo (es. €500), width_pct esplode!
# Esempio: (€124k / €500) * 100 = 24,800% !!

# Codice corretto (dovrebbe essere):
average_width_pct = np.mean(100 * (upper - lower) / y_test)  # ← Divisione per y_test reale
# Oppure:
average_width_pct = 100 * average_width / np.mean(y_test)
```

**Valore atteso**:
```python
# Calcolo manuale corretto
average_width_pct = 100 * (124204 / 62592) = 198%  # Per 80%
average_width_pct = 100 * (162662 / 62592) = 260%  # Per 90%
```

Questi valori hanno senso: l'intervallo è ~2x il prezzo medio.

---

## 🐛 FIX DEL BUG

### Localizza il Bug

```bash
grep -n "average_width_pct" src/training/diagnostics.py
```

### Fix Proposto

```python
# PRIMA (ERRATO):
avg_width_pct = float(np.mean(100.0 * (upper - lower) / (y_test_pred + 1e-8)))

# DOPO (CORRETTO):
avg_width_pct = float(100.0 * np.mean(upper - lower) / np.mean(y_test))
# O in alternativa (percentuale per ogni predizione, poi media):
avg_width_pct = float(np.mean(100.0 * (upper - lower) / (y_test + 1e-8)))
```

**Razionale**:
- Larghezza percentuale dovrebbe essere rispetto al **valore reale** (y_test)
- Non rispetto alla **predizione** (y_test_pred) che può essere molto diversa

---

## 📈 COME MIGLIORARE I PREDICTION INTERVALS

### 1. Migliorare il Modello ✅ (Priorità ALTA)

Ridurre i residui → intervalli più stretti

```yaml
# Strategie:
- Migliorare feature engineering (più feature contestuali)
- Hyperparameter tuning più aggressivo (più trial)
- Ensemble più robusti (stacking)
- Dataset più grande (meno filtri se possibile)
```

### 2. Calibrazione Post-Hoc ⚠️ (Priorità MEDIA)

Aggiustare gli intervalli se sotto/sovra-calibrati

```python
# Se coverage < target (come nel tuo caso):
calibration_factor = target_coverage / observed_coverage
adjusted_lower = y_pred - calibration_factor * (y_pred - lower)
adjusted_upper = y_pred + calibration_factor * (upper - y_pred)
```

### 3. Metodi Alternativi 🔬 (Priorità BASSA)

#### a) Quantile Regression

Predice direttamente i quantili invece di bootstrap:

```python
from sklearn.ensemble import GradientBoostingRegressor

# Modello per lower bound (10° percentile)
model_lower = GradientBoostingRegressor(loss='quantile', alpha=0.10)
model_lower.fit(X_train, y_train)

# Modello per upper bound (90° percentile)
model_upper = GradientBoostingRegressor(loss='quantile', alpha=0.90)
model_upper.fit(X_train, y_train)

# Predizioni
y_pred = model.predict(X_test)  # Predizione centrale
lower = model_lower.predict(X_test)
upper = model_upper.predict(X_test)
```

#### b) Conformal Prediction

Metodo più moderno, garantisce coverage esatto:

```python
# Pseudocodice
from mapie.regression import MapieRegressor

mapie = MapieRegressor(estimator=model, cv=10)
mapie.fit(X_train, y_train)

y_pred, y_intervals = mapie.predict(X_test, alpha=0.1)  # 90% interval
```

### 4. Stratificazione 📊 (Priorità MEDIA)

Calcola intervalli separati per diversi gruppi:

```python
# Esempio: intervalli diversi per fascia di prezzo
price_bands = pd.qcut(y_train, q=4, labels=['low', 'med', 'high', 'very_high'])

for band in ['low', 'med', 'high', 'very_high']:
    mask_train = (price_bands == band)
    mask_test = (test_price_bands == band)
    
    # Bootstrap solo sui residui di quella fascia
    band_residuals = train_residuals[mask_train]
    # ... calcola intervalli per quel band ...
```

**Vantaggio**: Case economiche hanno residui piccoli, case costose hanno residui grandi → intervalli più appropriati.

---

## 🎯 RACCOMANDAZIONI PER IL TUO PROGETTO

### Immediate

1. **Fix del bug** `average_width_pct`:
   ```python
   # In src/training/diagnostics.py
   avg_width_pct = float(100.0 * np.mean(upper - lower) / np.mean(y_test))
   ```

2. **Verifica coverage per fascia di prezzo**:
   ```python
   # Aggiungere analisi stratificata
   for quantile in ['Q1', 'Q2', 'Q3', 'Q4']:
       mask = (price_quantiles == quantile)
       coverage_q = ((y_test[mask] >= lower[mask]) & 
                     (y_test[mask] <= upper[mask])).mean()
       print(f"{quantile}: coverage = {coverage_q:.2%}")
   ```

### Medio Termine

3. **Migliorare modello** → R² più alto → residui più piccoli → intervalli più stretti

4. **Aumentare `n_bootstraps`** da 100 a 500-1000:
   ```yaml
   # config.yaml
   uncertainty:
     prediction_intervals:
       n_bootstraps: 500  # Più stable estimates
   ```

5. **Salvare grafici diagnostici**:
   ```python
   # Aggiungere in diagnostics.py
   plt.figure(figsize=(12, 6))
   
   # Coverage per price band
   plt.subplot(1, 2, 1)
   # ... plot coverage ...
   
   # Width vs price
   plt.subplot(1, 2, 2)
   plt.scatter(y_test, upper - lower, alpha=0.3)
   plt.xlabel('True Price')
   plt.ylabel('Interval Width')
   # ...
   ```

### Lungo Termine

6. **Implementare Conformal Prediction** (garantisce coverage esatto)

7. **Quantile Regression** come alternativa

---

## 📚 RISORSE UTILI

### Paper & Riferimenti

- **Residual Bootstrap**: Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
- **Prediction Intervals for ML**: Khosravi et al. (2011)
- **Conformal Prediction**: Shafer & Vovk (2008)

### Librerie Python

```bash
pip install mapie  # Conformal prediction
pip install ngboost  # Natural gradient boosting con uncertainty
```

### Check Coverage Quality

```python
def evaluate_coverage_quality(y_test, lower, upper, target_coverage):
    """
    Valuta qualità del coverage.
    
    Returns:
        - coverage: coverage effettivo
        - calibration_error: |coverage - target|
        - rating: 'excellent', 'good', 'fair', 'poor'
    """
    coverage = ((y_test >= lower) & (y_test <= upper)).mean()
    calibration_error = abs(coverage - target_coverage)
    
    if calibration_error < 0.02:  # ±2%
        rating = 'excellent'
    elif calibration_error < 0.05:  # ±5%
        rating = 'good'
    elif calibration_error < 0.10:  # ±10%
        rating = 'fair'
    else:
        rating = 'poor'
    
    return {
        'coverage': coverage,
        'target': target_coverage,
        'calibration_error': calibration_error,
        'rating': rating
    }

# Esempio uso:
result_80 = evaluate_coverage_quality(y_test, lower_80, upper_80, 0.80)
print(f"80% interval: {result_80['rating']} (coverage={result_80['coverage']:.1%})")
# Output: "80% interval: good (coverage=78.9%)"
```

---

## ✅ SUMMARY

### Cosa Sono

- **Range di valori** entro cui cade il valore reale con probabilità nota
- **Più larghi di confidence intervals** (includono variabilità dati + incertezza modello)

### I Tuoi Risultati

| Metrica | 80% Interval | 90% Interval | Valutazione |
|---------|-------------|-------------|-------------|
| **Coverage** | 78.9% | 87.8% | ✅ **GOOD** (sotto-calibrato ~2%) |
| **Avg Width** | €124k | €163k | ⚠️ **LARGO** (2-3x prezzo medio) |
| **Avg Width %** | 211103% | 275776% | ❌ **BUG** (valore errato) |

### Next Steps

1. ✅ **Fix bug** `average_width_pct` (priorità: ora)
2. ✅ **Migliorare modello** → R² più alto → intervalli più stretti (priorità: alta)
3. ⚠️ **Analisi stratificata** per price band (priorità: media)
4. 🔬 **Conformal prediction** come alternativa (priorità: bassa)

**Gli intervalli funzionano correttamente (coverage OK), ma sono larghi perché i residui sono grandi (dataset skewed, alta varianza). Migliora il modello prima di tutto!**
