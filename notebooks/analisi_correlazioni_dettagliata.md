# 📊 Analisi Dettagliata delle Correlazioni - Dataset StiMatrix

## 🎯 Overview

Questa analisi esamina le correlazioni tra le variabili di input del dataset StiMatrix, con particolare focus sui target `AI_Prezzo_Ridistribuito` e `AI_Prezzo_MQ`. L'analisi è basata su 5,733 osservazioni e 262 variabili, utilizzando diversi metodi di correlazione.

## 📈 Principali Risultati

### 1. **Correlazioni più Forti con AI_Prezzo_Ridistribuito**

#### 🏢 **Variabili di Superficie (Correlazioni > 0.8)**
Le variabili di superficie mostrano le correlazioni più forti con il prezzo:

- **AI_SuperficieVisuraTotaleAttuale**: r = 0.851 (Spearman)
- **AI_Rendita**: r = 0.850 (Spearman)  
- **AI_SuperficieVisuraTotaleEAttuale**: r = 0.848 (Spearman)
- **AI_Superficie**: r = 0.845 (Spearman)
- **AI_SuperficieVisuraTotale**: r = 0.845 (Spearman)
- **AI_SuperficieCalcolata**: r = 0.837 (Spearman)

**💡 Insight**: Le variabili di superficie sono altamente intercorrelate (r > 0.97) e rappresentano essenzialmente la stessa informazione. Questo indica un forte pattern dimensionale nel dataset.

#### 💰 **Valori di Mercato (Correlazioni 0.6-0.7)**
- **OV_ValoreMercatoMin_ottimo**: r = 0.689 (Spearman)
- **OV_ValoreMercatoMax_ottimo**: r = 0.648 (Spearman)

#### 🏘️ **Variabili Categoriche (Correlazioni > 0.5)**
- **AI_IdCategoriaCatastale**: η = 0.565 (Eta correlation)
- **AI_IdTipologiaEdilizia**: η = 0.564 (Eta correlation)
- **AI_IdSettoreMercato**: η = 0.485 (Eta correlation)

### 2. **Correlazioni con AI_Prezzo_MQ**

Il prezzo al metro quadro mostra pattern diversi:

#### 💰 **Valori di Mercato (Correlazioni più forti)**
- **OV_ValoreMercatoMin_ottimo**: r = 0.720 (Spearman)
- **OV_ValoreMercatoMax_ottimo**: r = 0.671 (Spearman)
- **OV_ValoreMercatoMin_normale**: r = 0.640 (Spearman)

#### 🌍 **Variabili Geografiche**
- **AI_Longitudine**: MI = 0.503 (Mutual Information)
- **AI_Latitudine**: MI = 0.464 (Mutual Information)

**💡 Insight**: Il prezzo al metro quadro è più influenzato dalla localizzazione geografica e dalle valutazioni di mercato che dalle dimensioni assolute.

## 🔍 Analisi per Metodi di Correlazione

### **Spearman vs Pearson**
- **Spearman** cattura meglio le relazioni non-lineari
- **Pearson** mostra correlazioni più basse per le stesse variabili
- Differenze significative indicano relazioni non-lineari nei dati

### **Mutual Information**
- Rivela pattern complessi non catturati dalle correlazioni lineari
- Particolarmente utile per variabili geografiche e di zona

### **Kendall Tau**
- Conferma i risultati di Spearman ma con valori più conservativi
- Più robusto agli outliers

## 🏗️ Pattern di Intercorrelazione tra Variabili di Input

### **Cluster di Variabili Superficie**
Le variabili di superficie formano un cluster fortemente correlato:
- `AI_Superficie` ↔ `AI_SuperficieVisuraTotaleAttuale`: r = 0.993
- `AI_SuperficieVisuraTotaleAttuale` ↔ `AI_SuperficieVisuraTotaleEAttuale`: r = 0.999
- `AI_SuperficieCalcolata` ↔ altre superficie: r > 0.975

**⚠️ Rischio**: Multicollinearità estrema - queste variabili sono essenzialmente ridondanti.

### **Cluster Valori di Mercato**
- `OV_ValoreMercatoMin_ottimo` ↔ `OV_ValoreMercatoMax_ottimo`: r = 0.980
- Forte correlazione tra valori min/max nelle stesse condizioni

### **Cluster POI (Points of Interest)**
Le variabili POI mostrano correlazioni moderate-forti tra loro:
- `POI_tourist_attraction_count` ↔ `POI_cafe_count`: r = 0.863
- `POI_restaurant_count` ↔ `POI_cafe_count`: r = 0.955
- Indicano concentrazione urbana/turistica

## 📊 Implications per il Modeling

### **1. Feature Selection**
- **Ridurre ridondanza**: Selezionare una sola variabile di superficie rappresentativa
- **Combinare informazioni**: Creare feature composite da variabili correlate
- **Rimuovere outliers**: Le correlazioni Spearman molto superiori a Pearson indicano non-linearità

### **2. Engineering Recommendations**
- **Logarithmic transformation**: Il target mostra forte skewness (5.16 → -0.77 dopo log)
- **Geographic clustering**: Sfruttare le correlazioni geografiche per il prezzo al metro quadro
- **Categorical encoding**: Le variabili categoriche mostrano correlazioni significative

### **3. Validation Strategy**
- **Attenzione al data leakage**: Le variabili di superficie potrebbero essere derivate dal target
- **Cross-validation geografica**: Date le forti correlazioni spaziali
- **Stratified sampling**: Per categoria catastale e tipologia edilizia

## 🎯 Key Takeaways

1. **Le dimensioni dominano il prezzo totale** - correlazioni > 0.8
2. **La localizzazione determina il prezzo al metro quadro** - correlazioni geografiche forti
3. **Forte multicollinearità** tra variabili di superficie - necessaria feature selection
4. **Pattern non-lineari** evidenti dalla differenza Spearman/Pearson
5. **Variabili categoriche significative** - η > 0.5 per categoria catastale e tipologia

## 📈 Raccomandazioni per l'Analisi Successiva

1. **Principal Component Analysis** per ridurre la dimensionalità delle variabili di superficie
2. **Clustering geografico** per catturare pattern spaziali
3. **Feature interaction analysis** tra categoria catastale e superficie
4. **Outlier detection** basato sui residui delle correlazioni non-lineari
5. **Temporal analysis** per verificare trend temporali nelle correlazioni

---
*Analisi generata il: $(date)*
*Dataset: StiMatrix - 5,733 osservazioni, 262 variabili*
*Metodi: Pearson, Spearman, Kendall, Mutual Information, Eta*