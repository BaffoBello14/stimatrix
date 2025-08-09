from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..db.connect import get_engine
from ..utils.io import load_json, save_dataframe
from ..utils.logger import get_logger

logger = get_logger(__name__)


def build_select_clause_dual_omi(
    schema: Dict[str, Any], selected_aliases: Optional[List[str]] = None
) -> str:
    """
    Costruisce la SELECT SQL includendo i valori OMI 'Normale', 'Ottimo' e 'Scadente'.

    Args:
        schema: Schema del database
        selected_aliases: Lista degli alias da includere

    Returns:
        Clausola SELECT SQL
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
            col_type = str(col.get("type", "")).lower()

            if alias == "OV":
                # Gestione speciale per OmiValori con stati diversi
                for stato_alias, stato_suffix in [
                    ("OVN", "normale"),
                    ("OVO", "ottimo"),
                    ("OVS", "scadente"),
                ]:
                    selects.append(
                        f"{stato_alias}.{col_name} AS {alias}_{col_name}_{stato_suffix}"
                    )
            else:
                if col_type in ["geometry", "geography"]:
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
    )


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
    )


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
    )


def generate_query_with_poi_and_ztl(select_clause: str, poi_categories: List[str]) -> str:
    poi_subquery = generate_poi_counts_subquery()
    ztl_subquery = generate_ztl_subquery()

    poi_joins: List[str] = []
    poi_selects: List[str] = []

    for category in poi_categories:
        safe_category = (
            str(category).replace("-", "_").replace(" ", "_").replace(".", "_")
        )
        alias = f"POI_{safe_category}"

        poi_joins.append(
            f"""
        LEFT JOIN POI_COUNTS {alias} ON PC.Id = {alias}.IdParticella 
            AND {alias}.TipologiaPOI = '{category}'"""
        )

        poi_selects.append(
            f"COALESCE({alias}.ConteggioPOI, 0) AS POI_{safe_category}_count"
        )

    poi_joins_str = "".join(poi_joins)
    poi_selects_str = ",\n        ".join(poi_selects) if poi_selects else "0 as no_poi"

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
            AND OVS.IdTipologiaEdilizia = AI.IdTipologiaEdilizia
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
    """


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
            AND OVS.IdTipologiaEdilizia = AI.IdTipologiaEdilizia
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
    """


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Pulizia DataFrame: sostituzione stringhe vuote con NaN")
    df.replace("", np.nan, inplace=True)
    df = df.infer_objects(copy=False)
    return df


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


def get_poi_categories(engine) -> List[str]:
    try:
        query = get_poi_categories_query()
        with engine.connect() as connection:
            result = pd.read_sql(query, connection)
            categories = result["Id"].tolist()
            logger.info(f"Trovate {len(categories)} categorie POI: {categories}")
            return categories
    except Exception as exc:
        logger.warning(f"Errore nel recupero categorie POI: {exc}")
        return []


def filter_coherent_acts(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    logger.info("Filtro atti coerenti...")

    initial_rows = len(df)

    if "A_Id" not in df.columns or "A_TotaleImmobili" not in df.columns:
        logger.warning(
            "Colonne A_Id o A_TotaleImmobili non trovate, skip filtro atti coerenti"
        )
        return df, {"skipped": True, "reason": "missing_columns"}

    grouped = (
        df.groupby("A_Id").agg(num_rows=("A_Id", "size"), expected=("A_TotaleImmobili", "first")).reset_index()
    )

    valid_ids = grouped[grouped["num_rows"] == grouped["expected"]]["A_Id"]
    df_filtered = df[df["A_Id"].isin(valid_ids)].copy()

    filtered_rows = len(df_filtered)
    percentage_kept = (filtered_rows / initial_rows * 100) if initial_rows > 0 else 0

    logger.info(
        f"Atti filtrati: {initial_rows} -> {filtered_rows} righe ({percentage_kept:.1f}% mantenute)"
    )

    stats = {
        "initial_rows": initial_rows,
        "filtered_rows": filtered_rows,
        "percentage_kept": percentage_kept,
        "acts_removed": len(grouped) - len(valid_ids),
    }

    return df_filtered, stats


def estimate_prices_dual(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    logger.info("Stima prezzi e ridistribuzione...")

    required_cols = [
        "A_Id",
        "A_Prezzo",
        "AI_Superficie",
        "OV_ValoreMercatoMin_normale",
        "OV_ValoreMercatoMax_normale",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logger.warning(
            f"Colonne mancanti per stima prezzi: {missing_cols}, skip stima prezzi"
        )
        if "AI_Prezzo_Ridistribuito" not in df.columns:
            df["AI_Prezzo_Ridistribuito"] = np.nan
        return df, {
            "skipped": True,
            "reason": "missing_columns",
            "missing_columns": missing_cols,
        }

    df["prezzo_m2"] = (
        df["OV_ValoreMercatoMin_normale"] + df["OV_ValoreMercatoMax_normale"]
    ) / 2
    df["prezzo_stimato_immobile"] = df["prezzo_m2"] * df["AI_Superficie"]

    prezzi = (
        df.groupby("A_Id")
        .agg(
            prezzo_stimato_totale=("prezzo_stimato_immobile", "sum"),
            A_Prezzo=("A_Prezzo", "first"),
        )
        .reset_index()
    )

    valid_mask = (prezzi["prezzo_stimato_totale"] > 0) & (prezzi["A_Prezzo"] > 0)
    invalid_count = (~valid_mask).sum()

    if invalid_count > 0:
        logger.warning(
            f"Trovati {invalid_count} atti con prezzi stimati o reali <= 0; coefficiente=1.0"
        )

    prezzi["coefficiente"] = 1.0
    prezzi.loc[valid_mask, "coefficiente"] = (
        prezzi.loc[valid_mask, "A_Prezzo"] / prezzi.loc[valid_mask, "prezzo_stimato_totale"]
    )

    valid_coefficients = prezzi.loc[valid_mask, "coefficiente"]
    if len(valid_coefficients) > 0:
        logger.info(
            f"Coefficienti: min={valid_coefficients.min():.3f}, max={valid_coefficients.max():.3f}, mean={valid_coefficients.mean():.3f}"
        )

    df = df.merge(prezzi[["A_Id", "coefficiente"]], on="A_Id", how="left")
    df["AI_Prezzo_Ridistribuito"] = df["prezzo_stimato_immobile"] * df["coefficiente"]

    df.drop(columns=["prezzo_m2", "prezzo_stimato_immobile", "coefficiente"], inplace=True)

    stats = {
        "properties_processed": len(df),
        "valid_prices": df["AI_Prezzo_Ridistribuito"].notna().sum(),
        "mean_redistributed_price": float(df["AI_Prezzo_Ridistribuito"].mean()),
        "median_redistributed_price": float(df["AI_Prezzo_Ridistribuito"].median()),
    }

    logger.info(f"Prezzi ridistribuiti calcolati per {len(df)} immobili")

    return df, stats


def retrieve_data(
    schema_path: str,
    selected_aliases: List[str],
    output_path: str,
    include_poi: bool = True,
    include_ztl: bool = True,
    poi_categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    logger.info("Avvio recupero dati dal database…")
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

    # Determina query
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

    # Esecuzione query
    with engine.connect() as connection:
        logger.info("Esecuzione query in corso…")
        df = pd.read_sql(query, connection)
        logger.info(f"Query completata: {df.shape[0]} righe, {df.shape[1]} colonne")

    # Post-processing
    df = clean_dataframe(df)
    df = drop_duplicate_columns(df)

    df, filter_stats = filter_coherent_acts(df)
    logger.info(f"Stats filtro atti: {filter_stats}")

    df, price_stats = estimate_prices_dual(df)
    logger.info(f"Stats prezzi: {price_stats}")

    # Drop colonne sensibili o non necessarie
    for col in ["A_Prezzo", "A_Id"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Salva risultati
    save_dataframe(df, output_path, format="parquet")
    logger.info(f"Dati salvati in: {output_path}")

    return df


def get_poi_categories_info() -> pd.DataFrame:
    logger.info("Recupero informazioni categorie POI…")
    engine = get_engine()
    query = get_poi_categories_query()

    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
        logger.info(f"Trovate {len(df)} categorie POI")
        return df