from dataset_builder.data.retrieval import build_select_clause_dual_omi
from utils.io import load_json


def test_build_select_clause_dual_omi_example_schema(tmp_path):
    schema_path = tmp_path / "schema.json"
    schema_json = (
        '{\n'
        '  "Atti": {"alias": "A", "columns": [\n'
        '    {"name": "Id", "type": "int", "retrieve": true}\n'
        '  ]},\n'
        '  "OmiValori": {"alias": "OV", "columns": [\n'
        '    {"name": "ValoreMercatoMin", "type": "decimal", "retrieve": true},\n'
        '    {"name": "ValoreMercatoMax", "type": "decimal", "retrieve": true}\n'
        '  ]}\n'
        '}\n'
    )
    schema_path.write_text(schema_json)
    schema = load_json(str(schema_path))
    clause = build_select_clause_dual_omi(schema, selected_aliases=["A", "OV"])
    assert "A.Id AS A_Id" in clause
    assert "OVN.ValoreMercatoMin AS OV_ValoreMercatoMin_normale" in clause
    assert "OVO.ValoreMercatoMax AS OV_ValoreMercatoMax_ottimo" in clause
    assert "OVS.ValoreMercatoMax AS OV_ValoreMercatoMax_scadente" in clause