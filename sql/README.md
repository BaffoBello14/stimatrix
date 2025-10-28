# SQL Templates Documentation

Questa directory contiene i template SQL per la generazione delle query di data retrieval.

## 📁 Struttura

```
sql/
├── README.md                  # Questo file
├── base_query.sql            # Query base senza POI/ZTL
├── query_with_poi_ztl.sql    # Query completa con POI e ZTL
├── poi_counts_cte.sql        # CTE per conteggio POI
└── ztl_check_cte.sql         # CTE per verifica ZTL
```

## 🎯 Vantaggi del sistema a template

✅ Query SQL leggibili in file dedicati  
✅ Facile aggiungere JOIN (anche con view!)  
✅ DBA può modificare SQL senza toccare Python  
✅ Versionamento separato SQL/Python  
✅ Testing più semplice  

## 🔧 Come funziona

### 1. Template SQL

I file `.sql` contengono placeholder che vengono sostituiti a runtime:

```sql
-- Esempio: base_query.sql
SELECT
    {select_clause}  -- ← sostituito con colonne selezionate
FROM
    Atti A
    INNER JOIN AttiImmobili AI ON AI.IdAtto = A.Id
    ...
```

### 2. Loader Python

Il modulo `utils.sql_templates.SQLTemplateLoader` gestisce:
- Caricamento dei template
- Sostituzione dei placeholder
- Costruzione di query composite (POI, ZTL)

### 3. Integrazione in retrieval.py

```python
# In DatasetBuilder
self.sql_loader = SQLTemplateLoader("sql")

# Generazione query
query = self.sql_loader.build_query_with_poi_ztl(
    select_clause=select_clause,
    poi_categories=poi_categories,
    include_poi=True,
    include_ztl=True
)
```

## 📝 File Template Dettagli

### base_query.sql
Query base che include:
- Tutte le JOIN principali (Atti, AttiImmobili, ParticelleCatastali, ecc.)
- JOIN con OmiValori per stati: Normale, Ottimo, Scadente
- **LEFT JOIN con view CENED** (attiimmobili_cened1, attiimmobili_cened2)
- Filtri WHERE standard

**Placeholder:**
- `{select_clause}`: Colonne da selezionare

### query_with_poi_ztl.sql
Query completa con features geospaziali:
- Tutte le JOIN della base query
- CTE per POI (Points of Interest)
- CTE per ZTL (Zone a Traffico Limitato)
- LEFT JOIN dinamiche per categorie POI

**Placeholder:**
- `{poi_cte}`: CTE per conteggio POI
- `{ztl_cte}`: CTE per verifica ZTL
- `{select_clause}`: Colonne base da selezionare
- `{poi_selects}`: Colonne POI generate dinamicamente
- `{poi_joins}`: JOIN POI generate dinamicamente

### poi_counts_cte.sql
Common Table Expression per calcolare il conteggio di POI per categoria all'interno dell'isodistanza di ogni particella catastale.

Logica:
1. CROSS JOIN tra tutte le particelle e tutte le tipologie POI
2. LEFT JOIN con PuntiDiInteresse filtrando per contenimento spaziale
3. GROUP BY per contare i POI per categoria

### ztl_check_cte.sql
Common Table Expression per verificare se il centroide di una particella cade in una Zona a Traffico Limitato.

Logica:
- Usa `STContains` per controllo geometrico
- Restituisce 1 (in ZTL) o 0 (fuori ZTL)

## 🆕 Aggiungere nuove JOIN/View

### Scenario: Aggiungere una nuova view o tabella

**Opzione A: Modificare i template SQL**

1. Edita `sql/base_query.sql` e `sql/query_with_poi_ztl.sql`:

```sql
-- Aggiungi la nuova JOIN
LEFT JOIN nuova_view NV ON AI.Id = NV.IdRiferimento
```

2. Aggiungi l'alias nel config:

```yaml
# config/config.yaml
database:
  selected_aliases: ['A', 'AI', ..., 'NV']  # ← nuovo alias
```

3. Rigenera lo schema per includere la view:

```bash
python main.py --step schema
```

**Opzione B: Solo per query specifiche**

Crea un nuovo template SQL dedicato:

```sql
-- sql/query_custom.sql
SELECT ...
FROM ...
    LEFT JOIN my_special_view MSV ON ...
```

Poi usa in Python:

```python
custom_query = self.sql_loader.load_template('query_custom.sql')
custom_query = custom_query.format(select_clause=select_clause)
```

## 🔍 View CENED Example

Le view `attiimmobili_cened1` e `attiimmobili_cened2` sono state aggiunte così:

### 1. Template SQL
```sql
-- In base_query.sql e query_with_poi_ztl.sql
LEFT JOIN attiimmobili_cened1 C1 ON AI.Id = C1.IdAttoImmobile
LEFT JOIN attiimmobili_cened2 C2 ON AI.Id = C2.IdAttoImmobile
```

### 2. Schema extraction
`schema_extract.py` è stato esteso per rilevare view:

```python
# Estrae anche le view
for view_name in inspector.get_view_names(schema=schema_name):
    schema[view_name] = {"type": "view", ...}
```

### 3. Configurazione
```yaml
# config.yaml
selected_aliases: ['A', 'AI', ..., 'C1', 'C2']
```

## 🧪 Testing

Verifica che le query siano corrette:

```python
# Test query generation
from utils.sql_templates import SQLTemplateLoader

loader = SQLTemplateLoader()
query = loader.build_base_query("A.Id, AI.Superficie")
print(query)
```

## 🚀 Best Practices

1. **Commenti SQL**: Documenta le JOIN complesse
2. **Indentazione**: Usa 4 spazi per leggibilità
3. **Alias corti**: A, AI, PC invece di nomi lunghi
4. **LEFT vs INNER**: Usa LEFT JOIN per dati opzionali
5. **Performance**: Aggiungi hint SQL se necessario

## 📊 Performance Considerations

- Le CTE (POI_COUNTS, ZTL_CHECK) vengono eseguite una volta
- Le JOIN con view sono efficienti quanto le tabelle
- Considera di creare indici su:
  - `IdAttoImmobile` (per CENED views)
  - `IdParticella` (per POI joins)
  - Colonne usate in `STContains` (geometrie)

## 📞 Supporto

Per domande o problemi:
1. Controlla i log: la query generata viene sempre loggata
2. Verifica che i template esistano in `sql/`
3. Controlla che gli alias in config corrispondano allo schema
