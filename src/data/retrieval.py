from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from ..db.connect import get_engine
from ..utils.io import load_json, save_dataframe
from ..utils.logger import get_logger

logger = get_logger(__name__)


def build_select_clause_dual_omi(
    schema: Dict[str, Any], selected_aliases: Optional[List[str]] = None
) -> str:
    """
    Costruisce la SELECT SQL includendo i valori OMI 'Normale', 'Ottimo' e 'Scadente'.
    """
    selects: List[str] = []

    for table_name, table_info in schema.items():
        alias = table_info.get("alias", table_name[:2].upper())

        if selected_aliases is not None and alias not in selected_aliases:
            continue

        for col in table_info["columns"]:
            if not col.get("retrieve", False):
                continue

            col_name = col["name"]
            col_type = str(col["type"]).lower()

            if alias == "OV":
                # Gestione speciale per OmiValori con stati diversi
                for stato_alias, stato_suffix in [("OVN", "normale"), ("OVO", "ottimo"), ("OVS", "scadente")]:
                    selects.append(f"{stato_alias}.{col_name} AS {alias}_{col_name}_{stato_suffix}")
            else:
                if col_type in ["geometry", "geography"]:
                    # Nota: verrà incluso solo se include_spatial=True in estrazione schema
                    selects.append(f"{alias}.{col_name}.STAsText() AS {alias}_{col_name}")
                else:
                    selects.append(f"{alias}.{col_name} AS {alias}_{col_name}")

    return ",\n        ".join(selects)


def get_poi_categories_query() -> str:
    return (
        """
    SELECT DISTINCT Id, Denominazione
    FROM PuntiDiInteresseTipologie
    ORDER BY Denominazione
    """
    ).strip()


def generate_poi_counts_subquery() -> str:
    return (
        """
    -- Subquery per conteggi POI per tipologia
    POI_COUNTS AS (
        SELECT 
            PC_MAIN.Id as IdParticella,
            PDIT.Id as TipologiaPOI,
            PDIT.Denominazione as DenominazionePOI,
            COUNT(PDI.Id) as ConteggioPOI
        FROM 
            ParticelleCatastali PC_MAIN
            CROSS JOIN PuntiDiInteresseTipologie PDIT
            LEFT JOIN (
                PuntiDiInteresse PDI 
                INNER JOIN PuntiDiInteresse_Tipologie PDI_T ON PDI.Id = PDI_T.IdPuntoDiInteresse
            ) ON PDI_T.IdTipologia = PDIT.Id 
                AND PC_MAIN.Isodistanza.STContains(PDI.Posizione) = 1
        GROUP BY PC_MAIN.Id, PDIT.Id, PDIT.Denominazione
    )
    """
    ).strip()


def generate_ztl_subquery() -> str:
    return (
        """
    -- Subquery per verifica ZTL
    ZTL_CHECK AS (
        SELECT 
            PC_MAIN.Id as IdParticella,
            CASE 
                WHEN EXISTS (
                    SELECT 1 
                    FROM ZoneTrafficoLimitato ZTL 
                    WHERE ZTL.Poligono.STContains(PC_MAIN.Centroide) = 1
                ) THEN 1 
                ELSE 0 
            END as InZTL
        FROM ParticelleCatastali PC_MAIN
    )
    """
    ).strip()


def _safe_poi_alias_from_category(category: Any) -> str:
    # Crea alias stabile per colonne POI anche con ID numerici
    try:
        int_id = int(category)
        return f"POI_ID_{int_id}"
    except Exception:
        safe = str(category)
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", safe).strip("_")
        return f"POI_{safe}"


def generate_query_with_poi_and_ztl(select_clause: str, poi_categories: List[Any]) -> str:
    poi_subquery = generate_poi_counts_subquery()
    ztl_subquery = generate_ztl_subquery()

    poi_joins: List[str] = []
    poi_selects: List[str] = []

    for category in poi_categories:
        alias = _safe_poi_alias_from_category(category)
        # Condizione: se category è numerico non quotare
        try:
            category_id = int(category)
            cond = f"{alias}.TipologiaPOI = {category_id}"
        except Exception:
            category_str = str(category).replace("'", "''")
            cond = f"{alias}.TipologiaPOI = '{category_str}'"

        poi_joins.append(
            f"\n        LEFT JOIN POI_COUNTS {alias} ON PC.Id = {alias}.IdParticella AND {cond}"
        )
        poi_selects.append(f"COALESCE({alias}.ConteggioPOI, 0) AS {alias}_count")

    poi_joins_str = "".join(poi_joins)
    poi_selects_str = ",\n        ".join(poi_selects) if poi_selects else "0 AS POI_none"

    return f"""
    WITH 
    {poi_subquery},
    {ztl_subquery}

    SELECT
        {select_clause},
        -- Conteggi POI per tipologia
        {poi_selects_str},
        -- Informazioni ZTL
        ZTL_INFO.InZTL as InZTL
    FROM
        Atti A
        INNER JOIN AttiImmobili AI ON AI.IdAtto = A.Id
        INNER JOIN ParticelleCatastali PC ON AI.IdParticellaCatastale = PC.Id
        INNER JOIN IstatSezioniCensuarie2021 ISC ON PC.IdSezioneCensuaria = ISC.Id
        INNER JOIN IstatIndicatori2021 II ON II.IdIstatZonaCensuaria = ISC.Id
        INNER JOIN ParticelleCatastali_OmiZone PC_OZ ON PC_OZ.IdParticella = PC.Id
        INNER JOIN OmiZone OZ ON PC_OZ.IdZona = OZ.Id
        -- Join su OmiValori per stato Normale (necessaria)
        INNER JOIN OmiValori OVN ON OZ.Id = OVN.IdZona
            AND OVN.Stato = 'Normale'
            AND AI.IdTipologiaEdilizia = OVN.IdTipologiaEdilizia
            AND A.Semestre = OZ.IdAnnoSemestre
        -- Join su OmiValori per stato Ottimo (opzionale)
        LEFT JOIN OmiValori OVO ON OZ.Id = OVO.IdZona
            AND OVO.Stato = 'Ottimo'
            AND AI.IdTipologiaEdilizia = OVO.IdTipologiaEdilizia
            AND A.Semestre = OZ.IdAnnoSemestre
        -- Join su OmiValori per stato Scadente (opzionale)
        LEFT JOIN OmiValori OVS ON OZ.Id = OVS.IdZona
            AND OVS.Stato = 'Scadente'
            AND AI.IdTipologiaEdilizia = OVS.IdTipologiaEdilizia
            AND A.Semestre = OZ.IdAnnoSemestre
        -- Join per informazioni ZTL
        LEFT JOIN ZTL_CHECK ZTL_INFO ON PC.Id = ZTL_INFO.IdParticella
        -- Join per conteggi POI{poi_joins_str}
    WHERE 
        A.TotaleFabbricati = A.TotaleImmobili
        AND AI.IdTipologiaEdilizia IS NOT NULL
        AND A.Id NOT IN (
            SELECT IdAtto
            FROM AttiImmobili
            WHERE Superficie IS NULL
            OR IdTipologiaEdilizia IS NULL
        )
    ORDER BY A.Id
    """.strip()


def generate_query_dual_omi(select_clause: str) -> str:
    return f"""
    SELECT
        {select_clause}
    FROM
        Atti A
        INNER JOIN AttiImmobili AI ON AI.IdAtto = A.Id
        INNER JOIN ParticelleCatastali PC ON AI.IdParticellaCatastale = PC.Id
        INNER JOIN IstatSezioniCensuarie2021 ISC ON PC.IdSezioneCensuaria = ISC.Id
        INNER JOIN IstatIndicatori2021 II ON II.IdIstatZonaCensuaria = ISC.Id
        INNER JOIN ParticelleCatastali_OmiZone PC_OZ ON PC_OZ.IdParticella = PC.Id
        INNER JOIN OmiZone OZ ON PC_OZ.IdZona = OZ.Id
        -- Join su OmiValori per stato Normale (necessaria)
        INNER JOIN OmiValori OVN ON OZ.Id = OVN.IdZona
            AND OVN.Stato = 'Normale'
            AND AI.IdTipologiaEdilizia = OVN.IdTipologiaEdilizia
            AND A.Semestre = OZ.IdAnnoSemestre
        -- Join su OmiValori per stato Ottimo (opzionale)
        LEFT JOIN OmiValori OVO ON OZ.Id = OVO.IdZona
            AND OVO.Stato = 'Ottimo'
            AND AI.IdTipologiaEdilizia = OVO.IdTipologiaEdilizia
            AND A.Semestre = OZ.IdAnnoSemestre
        -- Join su OmiValori per stato Scadente (opzionale)
        LEFT JOIN OmiValori OVS ON OZ.Id = OVS.IdZona
            AND OVS.Stato = 'Scadente'
            AND AI.IdTipologiaEdilizia = OVS.IdTipologiaEdilizia
            AND A.Semestre = OZ.IdAnnoSemestre
    WHERE 
        A.TotaleFabbricati = A.TotaleImmobili
        AND AI.IdTipologiaEdilizia IS NOT NULL
        AND A.Id NOT IN (
            SELECT IdAtto
            FROM AttiImmobili
            WHERE Superficie IS NULL
            OR IdTipologiaEdilizia IS NULL
        )
    ORDER BY A.Id
    """.strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Pulizia DataFrame: sostituzione stringhe vuote con NaN")
    return df.replace("", np.nan)


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Rimozione colonne duplicate...")

    cols = df.columns.tolist()
    to_drop: set[str] = set()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if cols[j] in to_drop:
                continue
            s1 = df[cols[i]].fillna("##nan##")
            s2 = df[cols[j]].fillna("##nan##")
            if s1.equals(s2):
                to_drop.add(cols[j])

    if to_drop:
        df = df.drop(columns=list(to_drop))
        logger.info(f"Colonne duplicate rimosse: {list(to_drop)}")
    else:
        logger.info("Nessuna colonna duplicata trovata")

    return df


def get_poi_categories(engine) -> List[Any]:
    try:
        query = get_poi_categories_query()
        with engine.connect() as connection:
            result = pd.read_sql_query(sql=text(query), con=connection)
            categories = result["Id"].tolist()
            logger.info(f"Trovate {len(categories)} categorie POI")
            return categories
    except Exception as e:
        logger.warning(f"Errore nel recupero categorie POI: {e}")
        return []


def retrieve_data(
    schema_path: str,
    selected_aliases: List[str],
    output_path: str,
    include_poi: bool = True,
    include_ztl: bool = True,
    poi_categories: Optional[List[Any]] = None,
) -> pd.DataFrame:
    logger.info("Avvio recupero dati dal database...")
    logger.info(f"Schema: {schema_path}")
    logger.info(f"Alias selezionati: {selected_aliases}")
    logger.info(f"Include POI: {include_poi}, Include ZTL: {include_ztl}")

    # Carica schema
    schema = load_json(schema_path)
    logger.info(f"Schema caricato con {len(schema)} tabelle")

    # Costruisce query
    select_clause = build_select_clause_dual_omi(schema, selected_aliases)

    # Ottieni engine
    engine = get_engine()

    # Determina se usare la query estesa con POI e ZTL
    if include_poi or include_ztl:
        if include_poi:
            if poi_categories is None:
                poi_categories = get_poi_categories(engine)
            logger.info(f"Usando {len(poi_categories)} categorie POI")
        else:
            poi_categories = []
        query = generate_query_with_poi_and_ztl(select_clause, poi_categories)
    else:
        query = generate_query_dual_omi(select_clause)

    logger.info("Query SQL generata")

    # Esegue query
    with engine.connect() as connection:
        logger.info("Esecuzione query in corso...")
        df = pd.read_sql_query(sql=text(query), con=connection)
        logger.info(f"Query completata: {df.shape[0]} righe, {df.shape[1]} colonne")

    # Pulizia dati
    df = clean_dataframe(df)
    df = drop_duplicate_columns(df)

    # Salva risultati
    save_dataframe(df, output_path, format="parquet")
    logger.info(f"Dati salvati in: {output_path}")

    return df


def get_poi_categories_info() -> pd.DataFrame:
    logger.info("Recupero informazioni categorie POI...")

    engine = get_engine()
    query = get_poi_categories_query()

    with engine.connect() as connection:
        df = pd.read_sql_query(sql=text(query), con=connection)
        logger.info(f"Trovate {len(df)} categorie POI")
        return df


def _inject_top_limit_for_cte_query(query_with_cte: str, limit: int) -> str:
    """Inserisce TOP {limit} nella SELECT principale di una query con CTE.

    Heuristica: sostituisce l'ultima occorrenza di una riga che inizia con 'SELECT' con 'SELECT TOP {limit}'.
    """
    pattern = re.compile(r"(^\s*SELECT\s*)", re.IGNORECASE | re.MULTILINE)
    matches = list(pattern.finditer(query_with_cte))
    if not matches:
        return query_with_cte
    # Prendi l'ultima occorrenza (la SELECT principale dovrebbe essere l'ultima definita dopo i CTE)
    last = matches[-1]
    start, end = last.span()
    return query_with_cte[:start] + f"SELECT TOP {limit} " + query_with_cte[end:]


def test_poi_and_ztl_sample(
    schema_path: str, selected_aliases: List[str], limit: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Test recupero POI e ZTL su campione di {limit} righe...")

    schema = load_json(schema_path)
    select_clause = build_select_clause_dual_omi(schema, selected_aliases)

    engine = get_engine()
    poi_categories = get_poi_categories(engine)[:5]

    base_query = generate_query_with_poi_and_ztl(select_clause, poi_categories)
    test_query = _inject_top_limit_for_cte_query(base_query, limit)

    with engine.connect() as connection:
        logger.info("Esecuzione query di test...")
        df = pd.read_sql_query(sql=text(test_query), con=connection)
        logger.info(f"Test completato: {df.shape[0]} righe, {df.shape[1]} colonne")

    poi_info = get_poi_categories_info()

    return df, poi_info