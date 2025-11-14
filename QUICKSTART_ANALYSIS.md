# 🚀 QUICK START: Analisi Filtri Dataset

**Sei sul branch `cursor/analyze-and-test-data-subset-176c` e vuoi capire l'impatto dei filtri applicati?**

## ⚡ Esecuzione Rapida

```bash
# 1. Analizza impatto filtri (2-3 minuti)
python analyze_filters_impact.py

# 2. Se dataset > 3,000 righe → Procedi con training
python main.py --config fast --steps preprocessing training evaluation

# 3. Se dataset < 3,000 righe → Valuta riduzione filtri
```

## 📊 Cosa Aspettarsi

Lo script `analyze_filters_impact.py` mostrerà:

1. **Distribuzione temporale** completa del dataset
2. **Righe rimosse** per ciascun filtro:
   - Filtro temporale (`anno_min: 2022`)
   - Filtro zone (`zone_escluse: E1/E2/E3/R1`)
   - Filtro tipologie (`tipologie_escluse: '4'`)
3. **Dataset finale** post-filtri con stima split train/val/test
4. **Confronto distribuzioni** target pre/post filtri
5. **Warning automatici** se dataset troppo piccolo

## 📚 Documentazione Completa

Per analisi dettagliata, consulta:

- **[EXECUTIVE_SUMMARY_FILTERS.md](./EXECUTIVE_SUMMARY_FILTERS.md)** - Sintesi esecutiva (5 min lettura)
- **[ANALISI_SUBSET_CONFIG_2022.md](./ANALISI_SUBSET_CONFIG_2022.md)** - Analisi completa (20 min lettura)

## 🎯 Filtri Attualmente Configurati

```yaml
data_filters:
  anno_min: 2022                      # Solo transazioni dal 2022 in poi
  zone_escluse: ['E1','E2','E3','R1'] # Escludi zone periferiche/rurali
  tipologie_escluse: ['4']            # Escludi ville
```

## ✅ Verifiche Completate

- ✅ **No data leakage**: Filtri applicati pre-split, feature leak-free
- ✅ **Test coverage**: `test_encoding_no_leakage.py` (267 righe, 8 test)
- ✅ **Architettura solida**: Fit/transform pattern corretto

## 🚨 Threshold Critici

| Dataset Finale | Azione Raccomandata |
|---------------|---------------------|
| **< 2,000 righe** | 🚨 Dataset troppo piccolo, ridurre filtri o complessità modelli |
| **2,000-3,000 righe** | ⚠️ Usare `config_fast.yaml` (5 trial vs 150) |
| **> 3,000 righe** | ✅ Usare `config.yaml` o `config_fast.yaml` |

## 💡 Troubleshooting

### Problema: Script fallisce con "ModuleNotFoundError: pandas"

```bash
# Soluzione: Installa dipendenze
pip install -r requirements.txt
```

### Problema: "FileNotFoundError: data/raw/raw.parquet"

```bash
# Soluzione: Genera dataset o usa step dataset
python main.py --steps dataset
```

### Problema: Dataset finale < 2,000 righe

```bash
# Soluzione 1: Riduci filtri (esempio: rimuovi filtro zone)
# Modifica config.yaml:
data_filters:
  anno_min: 2022
  zone_escluse: null  # Disabilitato
  tipologie_escluse: ['4']

# Soluzione 2: Riduci complessità training
# Usa config_fast.yaml (5 trial invece di 150)
python main.py --config fast
```

---

**Pronto per iniziare?**

```bash
python analyze_filters_impact.py
```
