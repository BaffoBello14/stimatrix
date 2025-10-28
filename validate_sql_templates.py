#!/usr/bin/env python3
"""
Script di validazione per il nuovo sistema SQL Templates.
Verifica che tutti i componenti siano correttamente configurati.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_template_loader():
    """Test caricamento template SQL."""
    print("🔍 Test 1: Caricamento template SQL...")
    
    try:
        from utils.sql_templates import SQLTemplateLoader
        
        loader = SQLTemplateLoader("sql")
        
        # Test caricamento template base
        base_query = loader.load_template('base_query.sql')
        assert len(base_query) > 0, "Template base_query.sql vuoto"
        assert "{select_clause}" in base_query, "Placeholder mancante in base_query.sql"
        
        # Test presenza JOIN CENED
        assert "LEFT JOIN attiimmobili_cened1 C1" in base_query, "JOIN CENED1 mancante"
        assert "LEFT JOIN attiimmobili_cened2 C2" in base_query, "JOIN CENED2 mancante"
        
        print("   ✅ Template base_query.sql caricato correttamente")
        print(f"   📏 Dimensione: {len(base_query)} caratteri")
        
        # Test caricamento altri template
        templates = ['query_with_poi_ztl.sql', 'poi_counts_cte.sql', 'ztl_check_cte.sql']
        for template_name in templates:
            template = loader.load_template(template_name)
            assert len(template) > 0, f"Template {template_name} vuoto"
            print(f"   ✅ Template {template_name} caricato correttamente")
        
        print("✅ Test 1 PASSED\n")
        return True
        
    except Exception as e:
        print(f"   ❌ ERRORE: {e}\n")
        return False


def test_query_building():
    """Test costruzione query."""
    print("🔍 Test 2: Costruzione query...")
    
    try:
        from utils.sql_templates import SQLTemplateLoader
        
        loader = SQLTemplateLoader("sql")
        
        # Test query base
        select_clause = "A.Id, AI.Superficie"
        query = loader.build_base_query(select_clause)
        
        assert select_clause in query, "SELECT clause non presente nella query"
        assert "FROM" in query, "FROM mancante"
        assert "WHERE" in query, "WHERE mancante"
        assert "LEFT JOIN attiimmobili_cened1 C1" in query, "JOIN CENED1 mancante"
        
        print("   ✅ Query base costruita correttamente")
        
        # Test query con POI/ZTL
        poi_categories = ['scuole', 'ospedali']
        query_poi = loader.build_query_with_poi_ztl(
            select_clause=select_clause,
            poi_categories=poi_categories,
            include_poi=True,
            include_ztl=True
        )
        
        assert "POI_COUNTS" in query_poi, "CTE POI_COUNTS mancante"
        assert "ZTL_CHECK" in query_poi, "CTE ZTL_CHECK mancante"
        assert "POI_scuole_count" in query_poi, "Colonna POI scuole mancante"
        assert "LEFT JOIN attiimmobili_cened1 C1" in query_poi, "JOIN CENED1 mancante in query POI"
        
        print("   ✅ Query con POI/ZTL costruita correttamente")
        
        # Test POI selects
        poi_selects = loader.build_poi_selects(poi_categories)
        assert "POI_scuole_count" in poi_selects
        assert "POI_ospedali_count" in poi_selects
        print("   ✅ POI selects generati correttamente")
        
        # Test POI joins
        poi_joins = loader.build_poi_joins(poi_categories)
        assert "LEFT JOIN POI_COUNTS POI_scuole" in poi_joins
        print("   ✅ POI joins generati correttamente")
        
        print("✅ Test 2 PASSED\n")
        return True
        
    except Exception as e:
        print(f"   ❌ ERRORE: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_schema_extraction():
    """Test estrazione schema con view."""
    print("🔍 Test 3: Schema extraction con view...")
    
    try:
        from db.schema_extract import generate_table_alias
        
        # Test alias auto-generati
        assert generate_table_alias("Atti") == "A"
        assert generate_table_alias("AttiImmobili") == "AI"
        assert generate_table_alias("ParticelleCatastali") == "PC"
        
        print("   ✅ Alias auto-generati corretti")
        
        # Test alias personalizzati
        custom_aliases = {
            'attiimmobili_cened1': 'C1',
            'attiimmobili_cened2': 'C2'
        }
        
        assert generate_table_alias("attiimmobili_cened1", custom_aliases) == "C1"
        assert generate_table_alias("attiimmobili_cened2", custom_aliases) == "C2"
        
        print("   ✅ Alias personalizzati funzionanti")
        
        print("✅ Test 3 PASSED\n")
        return True
        
    except Exception as e:
        print(f"   ❌ ERRORE: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_config_compatibility():
    """Test compatibilità configurazione."""
    print("🔍 Test 4: Compatibilità configurazione...")
    
    try:
        from utils.config import load_config
        
        # Carica config principale
        config = load_config("config/config.yaml")
        
        # Verifica presenza alias CENED
        selected_aliases = config.get('database', {}).get('selected_aliases', [])
        assert 'C1' in selected_aliases, "Alias C1 mancante in config"
        assert 'C2' in selected_aliases, "Alias C2 mancante in config"
        
        print("   ✅ Alias C1, C2 presenti in selected_aliases")
        
        # Verifica custom_aliases
        custom_aliases = config.get('database', {}).get('custom_aliases', {})
        assert custom_aliases.get('attiimmobili_cened1') == 'C1'
        assert custom_aliases.get('attiimmobili_cened2') == 'C2'
        
        print("   ✅ Custom aliases configurati correttamente")
        
        print("✅ Test 4 PASSED\n")
        return True
        
    except Exception as e:
        print(f"   ❌ ERRORE: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """Test struttura file."""
    print("🔍 Test 5: Struttura file...")
    
    required_files = [
        'sql/base_query.sql',
        'sql/query_with_poi_ztl.sql',
        'sql/poi_counts_cte.sql',
        'sql/ztl_check_cte.sql',
        'sql/README.md',
        'src/utils/sql_templates.py',
        'REFACTORING_SQL_TEMPLATES.md'
    ]
    
    workspace = Path(__file__).parent
    
    all_exist = True
    for file_path in required_files:
        full_path = workspace / file_path
        if full_path.exists():
            print(f"   ✅ {file_path} presente")
        else:
            print(f"   ❌ {file_path} MANCANTE")
            all_exist = False
    
    if all_exist:
        print("✅ Test 5 PASSED\n")
    else:
        print("❌ Test 5 FAILED\n")
    
    return all_exist


def main():
    """Esegue tutti i test di validazione."""
    print("=" * 70)
    print("🚀 VALIDAZIONE SISTEMA SQL TEMPLATES")
    print("=" * 70)
    print()
    
    tests = [
        ("Template Loader", test_template_loader),
        ("Query Building", test_query_building),
        ("Schema Extraction", test_schema_extraction),
        ("Config Compatibility", test_config_compatibility),
        ("File Structure", test_file_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' FALLITO con eccezione: {e}\n")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("=" * 70)
    print("📊 RISULTATI FINALI")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}  {test_name}")
    
    print()
    print(f"Totale: {passed}/{total} test passati")
    
    if passed == total:
        print("\n🎉 TUTTI I TEST PASSATI! Il sistema è pronto all'uso.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test falliti. Verifica gli errori sopra.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
