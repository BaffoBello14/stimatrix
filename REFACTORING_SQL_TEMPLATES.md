# 🎯 Refactoring SQL Templates - Riepilogo

## 📋 Problema Iniziale

Il sistema di data retrieval aveva un'architettura ibrida problematica:
- **Colonne**: Configurabili tramite schema JSON ✅
- **JOIN**: Hardcoded nei metodi Python ❌
- **View**: Non supportate ❌

Questo impediva di aggiungere LEFT JOIN con view del database senza modificare il codice Python.

## 🔧 Soluzione Implementata

### **Approccio: SQL Templates con Query Builder**

Abbiamo refactorizzato il sistema separando:
1. **Query SQL** → File `.sql` versionabili e modificabili
2. **Logica Python** → Caricamento template e sostituzione parametri
3. **Schema extraction** → Esteso per supportare view

---

## 📁 Modifiche Apportate

### 1. **Nuovi file SQL** (directory `/workspace/sql/`)

```
sql/
├── base_query.sql            # Query base con tutte le JOIN (incluse CENED)
├── query_with_poi_ztl.sql    # Query completa con POI e ZTL
├── poi_counts_cte.sql        # CTE per conteggio POI
├── ztl_check_cte.sql         # CTE per verifica ZTL
└── README.md                 # Documentazione template
```

**✅ Le view CENED sono ora incluse nei template SQL:**

```sql
-- In base_query.sql e query_with_poi_ztl.sql
LEFT JOIN attiimmobili_cened1 C1 ON AI.Id = C1.IdAttoImmobile
LEFT JOIN attiimmobili_cened2 C2 ON AI.Id = C2.IdAttoImmobile
```

### 2. **Nuovo modulo Python** (`src/utils/sql_templates.py`)

Classe `SQLTemplateLoader` che gestisce:
- Caricamento dei template SQL
- Costruzione dinamica di query con POI/ZTL
- Sostituzione dei placeholder

**API principale:**

```python
loader = SQLTemplateLoader("sql")

# Query base
query = loader.build_base_query(select_clause)

# Query con POI e ZTL
query = loader.build_query_with_poi_ztl(
    select_clause=select_clause,
    poi_categories=['scuole', 'ospedali'],
    include_poi=True,
    include_ztl=True
)
```

### 3. **Schema extraction esteso** (`src/db/schema_extract.py`)

**Modifiche:**

✅ Estrae anche le **view** (non solo tabelle)
✅ Supporta **alias personalizzati** configurabili
✅ Distingue tra `"type": "table"` e `"type": "view"`

```python
def extract_schema(engine, schema_name=None, custom_aliases=None):
    # Extract tables
    for table_name in inspector.get_table_names():
        ...
    
    # ✅ NUOVO: Extract views
    for view_name in inspector.get_view_names():
        schema[view_name] = {
            "alias": generate_table_alias(view_name, custom_aliases),
            "type": "view",
            ...
        }
```

### 4. **Refactoring retrieval.py**

**Rimossi metodi hardcoded:**
- ❌ `generate_poi_counts_subquery()`
- ❌ `generate_ztl_subquery()`
- ❌ `generate_query_with_poi_and_ztl()`
- ❌ `generate_query_dual_omi()`

**Sostituiti con:**

```python
class DatasetBuilder:
    def __init__(self, engine=None, sql_templates_dir="sql"):
        self.sql_loader = SQLTemplateLoader(sql_templates_dir)
    
    def retrieve_data(...):
        # Carica template SQL invece di costruire query Python
        query = self.sql_loader.build_query_with_poi_ztl(...)
```

### 5. **Configurazione aggiornata** (`config/config.yaml`)

```yaml
database:
  # ✅ Aggiunti alias per le view CENED
  selected_aliases: ['A', 'AI', ..., 'C1', 'C2']
  
  # ✅ Mapping personalizzato alias → tabelle/view
  custom_aliases:
    attiimmobili_cened1: 'C1'
    attiimmobili_cened2: 'C2'
```

---

## 🎯 Vantaggi del Nuovo Sistema

### **Prima (Hardcoded):**
```python
# Query SQL distribuite in 3+ metodi Python
def generate_query_dual_omi(self, select_clause: str) -> str:
    return f"""
    SELECT {select_clause}
    FROM Atti A
    INNER JOIN AttiImmobili AI ON ...
    -- 50+ righe di SQL embedded in Python
    """
```

❌ Difficile leggere la query completa  
❌ Non supportava view  
❌ Modifiche richiedevano toccare Python  

### **Dopo (SQL Templates):**
```sql
-- sql/base_query.sql (file dedicato)
SELECT
    {select_clause}
FROM
    Atti A
    INNER JOIN AttiImmobili AI ON AI.IdAtto = A.Id
    LEFT JOIN attiimmobili_cened1 C1 ON AI.Id = C1.IdAttoImmobile
    ...
```

✅ Query SQL leggibile e manutenibile  
✅ Supporta view out-of-the-box  
✅ DBA può modificare SQL senza toccare Python  
✅ Versionamento separato SQL/Python  
✅ Testing più semplice  

---

## 🚀 Come Usare il Nuovo Sistema

### **Aggiungere una nuova View/Tabella**

#### 1. Modifica il template SQL

Edita `sql/base_query.sql` e `sql/query_with_poi_ztl.sql`:

```sql
-- Aggiungi la tua JOIN
LEFT JOIN nuova_view NV ON AI.Id = NV.IdRiferimento
```

#### 2. Configura l'alias

Edita `config/config.yaml`:

```yaml
database:
  selected_aliases: ['A', 'AI', ..., 'NV']  # ← nuovo
  custom_aliases:
    nuova_view: 'NV'  # ← opzionale se l'alias auto-generato non va bene
```

#### 3. Rigenera lo schema

```bash
python main.py --steps schema
```

✅ **Fatto!** Le colonne della view saranno automaticamente recuperate.

---

## 🧪 Testing

Verifica che il sistema funzioni:

```python
from src.utils.sql_templates import SQLTemplateLoader

loader = SQLTemplateLoader()

# Test caricamento template
base_query = loader.load_template('base_query.sql')
print("Template caricato:", len(base_query), "caratteri")

# Test generazione query
query = loader.build_base_query("A.Id, AI.Superficie")
print(query)

# Verifica presenza JOIN CENED
assert "LEFT JOIN attiimmobili_cened1 C1" in query
assert "LEFT JOIN attiimmobili_cened2 C2" in query
```

---

## 📊 Confronto Architetture

| Aspetto | Prima (Hardcoded) | Dopo (Templates) |
|---------|------------------|------------------|
| **Leggibilità SQL** | ❌ Distribuito in 3+ metodi | ✅ File SQL dedicati |
| **Supporto View** | ❌ Richiedeva modifica codice | ✅ Automatico |
| **Manutenibilità** | ❌ DBA deve conoscere Python | ✅ Modifica solo SQL |
| **Testing** | ❌ Difficile testare query | ✅ Template isolabili |
| **Versionamento** | ❌ SQL mescolato a Python | ✅ SQL versionato separatamente |
| **Aggiungere JOIN** | ❌ 3 passi (Python + config) | ✅ 1 passo (template) |
| **Performance** | ✅ Identica | ✅ Identica |

---

## 🔍 Esempio Pratico: View CENED

### Richiesta Originale
> "Ho bisogno di aggiungere due LEFT JOIN con le view `attiimmobili_cened1` e `attiimmobili_cened2`, ma il sistema non lo permetteva."

### Soluzione con il Vecchio Sistema
1. Modificare `generate_query_dual_omi()` aggiungendo le JOIN
2. Modificare `generate_query_with_poi_and_ztl()` aggiungendo le JOIN (duplicazione!)
3. Aggiungere logica per estrarre colonne delle view
4. Gestire manualmente gli alias
5. Testare che le modifiche non rompano altre parti

❌ **Complessità:** Alta  
❌ **Tempo stimato:** 2-3 ore  
❌ **Rischio:** Medio (SQL duplicato in 2+ posti)

### Soluzione con il Nuovo Sistema
1. Aprire `sql/base_query.sql`
2. Aggiungere 2 righe:
   ```sql
   LEFT JOIN attiimmobili_cened1 C1 ON AI.Id = C1.IdAttoImmobile
   LEFT JOIN attiimmobili_cened2 C2 ON AI.Id = C2.IdAttoImmobile
   ```
3. Stessa cosa in `sql/query_with_poi_ztl.sql`
4. Aggiungere alias in `config.yaml`
5. Rigenerare schema

✅ **Complessità:** Bassa  
✅ **Tempo stimato:** 15 minuti  
✅ **Rischio:** Minimo (modifiche isolate)

---

## 🔄 Backward Compatibility

- ✅ **API pubblica invariata**: `retrieve_data()` funziona come prima
- ✅ **Schema JSON compatibile**: Solo aggiunto campo `"type"`
- ✅ **Config retrocompatibile**: `custom_aliases` è opzionale
- ⚠️ **Metodi interni rimossi**: Se qualcuno usava direttamente `generate_query_dual_omi()`, dovrà aggiornare

---

## 📚 Documentazione

- **SQL Templates**: `/workspace/sql/README.md` - Guida completa ai template
- **Questo file**: Panoramica del refactoring
- **Commenti in-code**: Docstring aggiornate in tutti i moduli modificati

---

## 🎓 Lezioni Apprese

### **Cosa ha funzionato bene:**
✅ Separazione delle responsabilità (SQL vs Python)  
✅ Template leggibili e manutenibili  
✅ Sistema di alias personalizzabili  
✅ Backward compatibility preservata  

### **Alternative considerate (e perché non scelte):**

**1. ORM completo (SQLAlchemy Models)**
- ❌ Troppo overkill per query read-only
- ❌ Learning curve alta
- ❌ Overhead per query complesse

**2. Query Builder programmatico**
- ❌ Meno leggibile dei template SQL
- ❌ Ancora mixing SQL/Python

**3. Stored Procedures**
- ❌ Deploy più complesso
- ❌ Difficile versionare
- ❌ Debugging più difficile

---

## 🚧 Possibili Miglioramenti Futuri

1. **Validazione template**: Aggiungere parser SQL per validare sintassi prima dell'esecuzione
2. **Cache query**: Cachare query generate per performance
3. **Multiple database support**: Template diversi per SQL Server/PostgreSQL/MySQL
4. **Query profiling**: Log automatico dei tempi di esecuzione per ogni CTE
5. **Dynamic CTE selection**: Includere solo CTE necessarie basandosi sui parametri

---

## 📞 Supporto

Per domande o problemi:
1. Consulta `/workspace/sql/README.md`
2. Controlla i log: la query generata è sempre loggata
3. Verifica template esistano in `sql/`
4. Controlla che alias in config corrispondano allo schema

---

## ✅ Checklist Implementazione

- [x] Creati template SQL (base_query.sql, query_with_poi_ztl.sql, ecc.)
- [x] Implementato SQLTemplateLoader
- [x] Esteso schema_extract.py per view
- [x] Aggiunto supporto custom aliases
- [x] Refactorizzato retrieval.py
- [x] Aggiornato config.yaml
- [x] Scritto documentazione (README in sql/, questo file)
- [x] Aggiunte view CENED (attiimmobili_cened1, attiimmobili_cened2)
- [ ] Testato con database reale
- [ ] Aggiornati test unitari (se esistenti)

---

**Data implementazione**: 2025-10-28  
**Versione**: 2.0 (SQL Templates System)  
**Breaking changes**: Metodi interni query generator rimossi  
**Migration effort**: Nessuno (se si usa solo API pubblica `retrieve_data()`)
