# 📊 Analisi Data-Driven per Feature Pruning

## Fonte Dati
- Correlation Matrix: `notebooks/eda_comprehensive_outputs/correlation_matrix_pearson.csv`
- SQL Query: `sql/base_query.sql`
- Config attuale: `config/config.yaml`

## 🗑️ COLONNE DA DROPPARE

### 1. **SUPERFICIE RIDONDANTI** (Correlazione > 0.98)

Dalla correlation matrix:
- `AI_Superficie` e `AI_SuperficieVisuraTotale`: r=1.0 (IDENTICHE!)
- `AI_SuperficieVisuraTotaleAttuale` e `AI_SuperficieVisuraTotaleEAttuale`: r=0.995
- `AI_SuperficieVisuraTotale` e `AI_SuperficieVisuraTotaleE`: r=0.995

**Decision**:
- **KEEP**: `AI_Superficie` (è il target di superficie principale)
- **DROP**: Tutte le altre varianti (ridondanti)

```yaml
- 'AI_SuperficieCalcolata'          # r=0.91 con AI_Superficie
- 'AI_SuperficieVisuraTotale'       # r=1.0 con AI_Superficie  
- 'AI_SuperficieVisuraTotaleE'      # r=0.995 con Totale
- 'AI_SuperficieVisuraTotaleAttuale'  # r=0.98 con AI_Superficie
- 'AI_SuperficieVisuraTotaleEAttuale' # r=0.976 con Totale
```

### 2. **INDICATORI ISTAT RIDONDANTI** (Correlazione > 0.95)

Dalla correlation matrix, molti II_ST* sono quasi perfettamente correlati:
- `II_ST2_B`, `II_ST21`, `II_ST1`, `II_ST29`: r > 0.98 tra loro
- `II_ST19`, `II_ST23`, `II_ST31`, `II_ST4`: r > 0.97 tra loro
- `II_ST2`, `II_ST20`, `II_ST26`: r > 0.98 tra loro

**Decision**: Tenere solo 1 rappresentante per cluster di correlazione alta

**DROP** (ridondanti con altri II_ST*):
```yaml
- 'II_ST2_B'   # r=0.99 con II_ST21, II_ST29
- 'II_ST21'    # r=0.98 con II_ST1, II_ST29  
- 'II_ST29'    # r=0.98 con II_ST2_B, II_ST1
- 'II_ST31'    # r=0.99 con II_ST23
- 'II_ST22'    # r=0.94 con II_ST2, II_ST20
- 'II_ST26'    # r=0.99 con II_ST2, II_ST20
- 'II_ST32'    # r=0.98 con II_ST23, II_ST31
```

### 3. **OmiValori RIDONDANTI**

Dalla correlation matrix:
- `OV_ValoreMercatoMin_normale` e `OV_ValoreMercatoMax_normale`: r=0.98
- `OV_ValoreMercatoMin_ottimo` e `OV_ValoreMercatoMax_ottimo`: r=0.98

**Decision**: Tenere solo Min (range è quasi costante)

```yaml
- 'OV_ValoreMercatoMax_normale'  # r=0.98 con Min_normale
- 'OV_ValoreMercatoMax_ottimo'   # r=0.98 con Min_ottimo  
- 'OV_ValoreMercatoMin_scadente' # Pochi dati (scadente poco usato)
- 'OV_ValoreMercatoMax_scadente' # Pochi dati
```

### 4. **COLONNE TECNICHE/METADATA** (Non feature)

Dalla SQL query, ci sono colonne tecniche che non sono feature predittive:

```yaml
# ID e chiavi esterne (non predittivi)
- 'A_Id'                    # ID atto (univoco)
- 'AI_Id'                   # ID atto immobile (quasi univoco)
- 'AI_IdAtto'               # Foreign key a A_Id
- 'AI_IdParticellaCatastale' # Foreign key
- 'PC_Id'                   # ID particella
- 'PC_IdSezioneCensuaria'   # Foreign key
- 'ISC_Id'                  # ID sezione censuaria
- 'II_IdIstatZonaCensuaria' # Foreign key
- 'OZ_Id'                   # ID zona OMI
- 'PC_OZ_IdParticella'      # Foreign key
- 'PC_OZ_IdZona'            # Foreign key

# Semestri e date ridondanti (già usati per temporalKey)
- 'A_Semestre'              # Ridondante con AnnoStipula/MeseStipula
- 'OZ_IdAnnoSemestre'       # Ridondante
- 'A_DataStipula'           # Già estratti anno/mese
- 'A_DataRegistrazione'     # Metadata, non feature

# Colonne metadata non predittive
- 'A_TotaleFabbricati'      # Sempre uguale a TotaleImmobili (filtro WHERE)
- 'A_TotaleImmobili'        # Constant (sempre 1 per filtro)
- 'A_NumeroRepertorio'      # Metadata notaio
- 'A_IdNotaio'              # Identificativo, troppi unique
- 'PC_PoligonoMetricoSrid'  # Constant (sempre stesso SRID)

# Coordinate raw (già estratte in feature)
- 'PC_PoligonoMetrico'      # WKT grezzo (già processato)
- 'AI_IndirizzoGeometry'    # Geometry raw (già processato)
```

### 5. **COLONNE CON TROPPO MISSING O COSTANTI**

Da config attuale (già in drop_columns) + aggiunte:

```yaml
# Gia in config:
- 'A_ImmobiliPrincipaliConSuperficieValorizzata'
- 'AI_SuperficieCalcolata'           # Già coperto sopra
- 'AI_SuperficieVisuraTotale'        # Già coperto sopra
- 'AI_SuperficieVisuraTotaleE'       # Già coperto sopra
- 'AI_SuperficieVisuraTotaleAttuale' # Già coperto sopra
- 'AI_SuperficieVisuraTotaleEAttuale' # Già coperto sopra
- 'A_AcquirentiCount'                # Poco predittivo
- 'A_VenditoriCount'                 # Poco predittivo
- 'A_EtaMediaAcquirenti'             # Privacy + poco predittivo
- 'A_EtaMediaVenditori'              # Privacy + poco predittivo
- 'A_AcquirentiVenditoriStessoCognome'  # Poco predittivo
- 'A_VenditoriEredita'               # Poco predittivo
- 'AI_Subalterno'                    # Codice catastale (non feature)
- 'AI_Piano'                         # Raw (già estratte floor features)
- 'AI_Rendita'                       # Alta correlazione con Prezzo (risk leakage)

# NUOVE DA AGGIUNGERE:
- 'AI_IdImmobile'            # ID univoco
- 'AI_Civico'                # Raw (già estratto AI_Civico_num)
- 'PC_Foglio'                # Codice catastale (poco predittivo)
- 'PC_Particella'            # Codice catastale (troppi unique)
- 'PC_SezioneUrbana'         # Codice (molti missing)
- 'PC_SezioneAmministrativa' # Codice (molti missing)
- 'PC_SezioneAggraria'       # Quasi sempre missing
- 'OZ_CodiceZona'            # Codice (già abbiamo AI_ZonaOmi)
- 'OZ_DescrizioneZona'       # Testo (ridondante con CodiceZona)
- 'OZ_IdComune'              # Sempre costante (singolo comune)
```

### 6. **COLONNE CENED PROBLEMATICHE** (Troppi missing)

Da SQL query, C1 e C2 sono LEFT JOIN → molti missing

```yaml
# CENED spesso assente (>80% missing)
- 'C1_ClasseEnergetica'    # Troppi missing (keep se <50% missing)
- 'C1_EPGlNren'            # Troppi missing (metrica tecnica)
- 'C1_EPHTot'              # Troppi missing
- 'C2_ClasseEnergetica'    # Ridondante con C1
- 'C2_EPGlNren'            # Ridondante con C1
```

**NOTA**: Se CENED ha <50% missing, KEEP (è predittivo). Altrimenti DROP.

---

## 🔧 NUMERIC_COERCION BLACKLIST

### Cosa fa `numeric_coercion`
Converte colonne `object` che sembrano numeriche (es. "123.45") in `float`.

**Problema**: Alcuni codici SEMBRANO numeri ma NON lo sono (es. "00020" = codice catastale).
Se convertiti in float → perdono leading zeros → CatBoost non capisce pattern.

### Blacklist Attuale (da correggere)
```yaml
blacklist_globs:
  - 'II_*'              # ⚠️ TROPPO AGGRESSIVO - alcuni II_* sono veri numerici!
  - 'AI_Id*'            # ✅ OK (ID)
  - 'Foglio'            # ✅ OK (codice catastale)
  - 'Particella*'       # ✅ OK (codice catastale)
  - 'Subalterno'        # ✅ OK (codice catastale)
  - 'SezioneAmministrativa' # ✅ OK (codice)
  - 'ZonaOmi'           # ⚠️ TROPPO AGGRESSIVO - esiste AI_ZonaOmi che è categorico!
  - '*COD*'             # ✅ OK (codici vari)
```

### Blacklist CORRETTA (Data-Driven)

```yaml
numeric_coercion:
  enabled: true
  threshold: 0.95  # Converti solo se ≥95% sono convertibili
  
  blacklist_globs:
    # === ID e codici catastali === #
    - '*Id'            # Tutti gli ID (A_Id, AI_Id, PC_Id, OZ_Id, ...)
    - '*_Id*'          # Varianti (IdAtto, IdParticella, ...)
    - 'Foglio*'        # Foglio catastale ("0001")
    - '*Particella*'   # Particella catastale ("00350")  
    - '*Subalterno*'   # Subalterno ("0001")
    - '*Sezione*'      # Sezioni catastali ("01")
    
    # === Codici e zone === #
    - '*COD*'          # Tutti i codici (CodiceZona, ...)
    - '*Codice*'       # Varianti
    - 'AI_ZonaOmi'     # Zona OMI ("D2", "C4") - CATEGORICO
    - 'OZ_CodiceZona'  # Codice zona OMI
    - '*IdCategoriaCatastale*' # Categorie ("00210", "00020")
    - '*IdTipologiaEdilizia*'  # Tipologie ("2", "3", "8")
    
    # === Altri pattern === #
    - '*Repertorio*'   # Numero repertorio notaio
    - '*Anno*'         # Anno già numeric, ma se string serve keep
    - '*Mese*'         # Mese già numeric
    
    # === IMPORTANTE: NON blacklist II_* generico! === #
    # Gli indicatori Istat II_ST* e II_P* sono NUMERICI VERI
    # SOLO blacklist ID Istat:
    - 'II_IdIstatZonaCensuaria'  # ID (non metrica)
    - 'ISC_Id'                   # ID sezione
```

### Spiegazione Correzione

**PRIMA** (Errore):
```yaml
- 'II_*'  # ❌ BLOCCA TUTTO Istat (anche metriche numeriche valide!)
```

**DOPO** (Corretto):
```yaml
- 'II_IdIstatZonaCensuaria'  # ✅ Solo ID (non metriche)
# II_ST1, II_ST2, II_P98, ecc. → CONVERTITI in float (corretto!)
```

**Perché**: `II_ST1`, `II_P98` sono METRICHE NUMERICHE (popolazione, densità, ecc.), NON codici.
Devono essere convertite in float per essere usate dai modelli.

---

## 📋 CONFIGURAZIONE FINALE

### feature_pruning.drop_columns (Lista Completa)

```yaml
feature_pruning:
  drop_columns:
    # === ID e chiavi esterne === #
    - 'A_Id'
    - 'AI_Id'
    - 'AI_IdAtto'
    - 'AI_IdParticellaCatastale'
    - 'AI_IdImmobile'
    - 'PC_Id'
    - 'PC_IdSezioneCensuaria'
    - 'ISC_Id'
    - 'II_IdIstatZonaCensuaria'
    - 'OZ_Id'
    - 'PC_OZ_IdParticella'
    - 'PC_OZ_IdZona'
    
    # === Superficie ridondanti === #
    - 'AI_SuperficieCalcolata'
    - 'AI_SuperficieVisuraTotale'
    - 'AI_SuperficieVisuraTotaleE'
    - 'AI_SuperficieVisuraTotaleAttuale'
    - 'AI_SuperficieVisuraTotaleEAttuale'
    
    # === Indicatori Istat ridondanti === #
    - 'II_ST2_B'
    - 'II_ST21'
    - 'II_ST29'
    - 'II_ST31'
    - 'II_ST22'
    - 'II_ST26'
    - 'II_ST32'
    
    # === OmiValori ridondanti === #
    - 'OV_ValoreMercatoMax_normale'
    - 'OV_ValoreMercatoMax_ottimo'
    - 'OV_ValoreMercatoMin_scadente'
    - 'OV_ValoreMercatoMax_scadente'
    
    # === Metadata e colonne tecniche === #
    - 'A_Semestre'
    - 'OZ_IdAnnoSemestre'
    - 'A_DataStipula'
    - 'A_DataRegistrazione'
    - 'A_TotaleFabbricati'
    - 'A_TotaleImmobili'
    - 'A_NumeroRepertorio'
    - 'A_IdNotaio'
    - 'PC_PoligonoMetricoSrid'
    - 'PC_PoligonoMetrico'
    - 'AI_IndirizzoGeometry'
    
    # === Codici catastali (poco predittivi) === #
    - 'PC_Foglio'
    - 'PC_Particella'
    - 'PC_SezioneUrbana'
    - 'PC_SezioneAmministrativa'
    - 'PC_SezioneAggraria'
    - 'OZ_CodiceZona'
    - 'OZ_DescrizioneZona'
    - 'OZ_IdComune'
    
    # === Colonne già processate === #
    - 'AI_Subalterno'  # Codice catastale
    - 'AI_Piano'       # Raw (già estratte floor features)
    - 'AI_Civico'      # Raw (già estratto AI_Civico_num)
    - 'AI_Rendita'     # Alta correlazione con Prezzo (risk leakage)
    
    # === Privacy / poco predittivo === #
    - 'A_ImmobiliPrincipaliConSuperficieValorizzata'
    - 'A_AcquirentiCount'
    - 'A_VenditoriCount'
    - 'A_EtaMediaAcquirenti'
    - 'A_EtaMediaVenditori'
    - 'A_AcquirentiVenditoriStessoCognome'
    - 'A_VenditoriEredita'
    
    # === CENED con troppo missing (check prima!) === #
    # Uncomment se missing > 50%:
    # - 'C1_EPGlNren'
    # - 'C1_EPHTot'
    # - 'C2_ClasseEnergetica'
    # - 'C2_EPGlNren'
```

### numeric_coercion.blacklist_globs (Lista Corretta)

```yaml
numeric_coercion:
  enabled: true
  threshold: 0.95
  blacklist_globs:
    - '*Id'
    - '*_Id*'
    - 'Foglio*'
    - '*Particella*'
    - '*Subalterno*'
    - '*Sezione*'
    - '*COD*'
    - '*Codice*'
    - 'AI_ZonaOmi'
    - 'OZ_CodiceZona'
    - '*IdCategoriaCatastale*'
    - '*IdTipologiaEdilizia*'
    - '*Repertorio*'
    - 'II_IdIstatZonaCensuaria'  # Solo ID Istat, NON II_ST* !
    - 'ISC_Id'
```

---

## ✅ Riepilogo Modifiche

| Categoria | # Colonne Drop | Ragione |
|-----------|----------------|---------|
| ID e FK | 12 | Identificatori univoci |
| Superficie ridondanti | 5 | Correlazione > 0.98 |
| Istat ridondanti | 7 | Correlazione > 0.95 |
| OmiValori ridondanti | 4 | Correlazione > 0.98 |
| Metadata/Tecnici | 13 | Non feature predittive |
| Codici catastali | 8 | Poco predittivi |
| Privacy/Poco predittivi | 7 | Scarsa utilità |
| **TOTALE** | **~56 colonne** | **Da ~150 a ~94 feature** |

---

## 🎯 Prossimi Passi

1. ✅ Aggiorna `config_optimized.yaml` con liste complete
2. ✅ Test preprocessing con nuova config
3. ✅ Verifica che feature contestuali siano aggiunte correttamente
4. ✅ Run training e confronta risultati
