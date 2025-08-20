from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from db.connect import DatabaseConnector
from utils.io import load_json, save_dataframe
from utils.logger import get_logger
from pathlib import Path

logger = get_logger(__name__)


class DatasetBuilder:
    def __init__(self, engine: Optional[object] = None) -> None:
        self._engine = engine  # lazy; may be None until needed

    @property
    def engine(self):
        if self._engine is None:
            self._engine = DatabaseConnector().engine
        return self._engine

    def build_select_clause_dual_omi(
        self, schema: Dict[str, Any], selected_aliases: Optional[List[str]] = None
    ) -> str:
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
                        selects.append(
                            f"{alias}.{col_name}.STAsText() AS {alias}_{col_name}"
                        )
                    else:
                        selects.append(f"{alias}.{col_name} AS {alias}_{col_name}")
        return ",\n        ".join(selects)

    @staticmethod
    def get_poi_categories_query() -> str:
        return (
            """
            SELECT DISTINCT Id, Denominazione
            FROM PuntiDiInteresseTipologie
            ORDER BY Denominazione
            """
        )

    @staticmethod
    def generate_poi_counts_subquery() -> str:
        return (
            """
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

    @staticmethod
    def generate_ztl_subquery() -> str:
        return (
            """
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

    def generate_query_with_poi_and_ztl(
        self, select_clause: str, poi_categories: List[str]
    ) -> str:
        poi_subquery = self.generate_poi_counts_subquery()
        ztl_subquery = self.generate_ztl_subquery()
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
            {poi_selects_str},
            ZTL_INFO.InZTL as InZTL
        FROM
            Atti A
            INNER JOIN AttiImmobili AI ON AI.IdAtto = A.Id
            INNER JOIN ParticelleCatastali PC ON AI.IdParticellaCatastale = PC.Id
            INNER JOIN IstatSezioniCensuarie2021 ISC ON PC.IdSezioneCensuaria = ISC.Id
            INNER JOIN IstatIndicatori2021 II ON II.IdIstatZonaCensuaria = ISC.Id
            INNER JOIN ParticelleCatastali_OmiZone PC_OZ ON PC_OZ.IdParticella = PC.Id
            INNER JOIN OmiZone OZ ON PC_OZ.IdZona = OZ.Id
            INNER JOIN OmiValori OVN ON OZ.Id = OVN.IdZona
                AND OVN.Stato = 'Normale'
                AND AI.IdTipologiaEdilizia = OVN.IdTipologiaEdilizia
                AND A.Semestre = OZ.IdAnnoSemestre
            LEFT JOIN OmiValori OVO ON OZ.Id = OVO.IdZona
                AND OVO.Stato = 'Ottimo'
                AND AI.IdTipologiaEdilizia = OVO.IdTipologiaEdilizia
                AND A.Semestre = OZ.IdAnnoSemestre
            LEFT JOIN OmiValori OVS ON OZ.Id = OVS.IdZona
                AND OVS.Stato = 'Scadente'
                AND OVS.IdTipologiaEdilizia = AI.IdTipologiaEdilizia
                AND A.Semestre = OZ.IdAnnoSemestre
            LEFT JOIN ZTL_CHECK ZTL_INFO ON PC.Id = ZTL_INFO.IdParticella
            {poi_joins_str}
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

    def generate_query_dual_omi(self, select_clause: str) -> str:
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
            INNER JOIN OmiValori OVN ON OZ.Id = OVN.IdZona
                AND OVN.Stato = 'Normale'
                AND AI.IdTipologiaEdilizia = OVN.IdTipologiaEdilizia
                AND A.Semestre = OZ.IdAnnoSemestre
            LEFT JOIN OmiValori OVO ON OZ.Id = OVO.IdZona
                AND OVO.Stato = 'Ottimo'
                AND AI.IdTipologiaEdilizia = OVO.IdTipologiaEdilizia
                AND A.Semestre = OZ.IdAnnoSemestre
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

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Pulizia DataFrame: sostituzione stringhe vuote con NaN")
        df.replace("", np.nan, inplace=True)
        df = df.infer_objects(copy=False)
        return df

    @staticmethod
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

    def get_poi_categories(self) -> List[str]:
        try:
            query = self.get_poi_categories_query()
            with self.engine.connect() as connection:
                result = pd.read_sql(query, connection)
                categories = result["Id"].tolist()
                logger.info(f"Trovate {len(categories)} categorie POI: {categories}")
                return categories
        except Exception as exc:
            logger.warning(f"Errore nel recupero categorie POI: {exc}")
            return []

    def filter_coherent_acts(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
            # rows removed provides an immediate understanding of data size reduction
            "acts_removed": initial_rows - filtered_rows,
            # keep also the number of acts removed for auditability
            "acts_removed_acts": int(len(grouped) - len(valid_ids)),
        }
        return df_filtered, stats

    def estimate_prices_dual(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
        self,
        schema_path: str,
        selected_aliases: List[str],
        output_path: str,
        include_poi: bool = True,
        include_ztl: bool = True,
        poi_categories: Optional[List[str]] = None,
        output_format: str = "parquet",
        compression: Optional[str] = None,
    ) -> pd.DataFrame:
        logger.info("Avvio recupero dati dal database…")
        logger.info(f"Schema: {schema_path}")
        logger.info(f"Alias selezionati: {selected_aliases}")
        logger.info(f"Include POI: {include_poi}, Include ZTL: {include_ztl}")
        schema = load_json(schema_path)
        logger.info(f"Schema caricato con {len(schema)} tabelle")
        select_clause = self.build_select_clause_dual_omi(schema, selected_aliases)
        if include_poi or include_ztl:
            if include_poi:
                if poi_categories is None:
                    poi_categories = self.get_poi_categories()
                logger.info(f"Usando {len(poi_categories)} categorie POI")
            else:
                poi_categories = []
            query = self.generate_query_with_poi_and_ztl(select_clause, poi_categories)
        else:
            query = self.generate_query_dual_omi(select_clause)
        logger.info("Query SQL generata")
        with self.engine.connect() as connection:
            logger.info("Esecuzione query in corso…")
            df = pd.read_sql(query, connection)
            logger.info(f"Query completata: {df.shape[0]} righe, {df.shape[1]} colonne")
        df = self.clean_dataframe(df)
        df = self.drop_duplicate_columns(df)
        df, filter_stats = self.filter_coherent_acts(df)
        logger.info(f"Stats filtro atti: {filter_stats}")
        df, price_stats = self.estimate_prices_dual(df)
        logger.info(f"Stats prezzi: {price_stats}")
        for col in ["A_Prezzo", "A_Id"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        save_dataframe(df, output_path, format=output_format, compression=compression)
        try:
            _log_path = Path(output_path).as_posix()
        except Exception:
            _log_path = str(output_path)
        logger.info(f"Dati salvati in: {_log_path}")
        return df


def run_dataset(config: Dict[str, Any]) -> None:
    db_cfg = config.get("database", {})
    paths = config.get("paths", {})

    schema_path = paths.get("schema", "data/db_schema.json")
    raw_dir = Path(paths.get("raw_data", "data/raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_filename = paths.get("raw_filename", f"raw.{db_cfg.get('output_format', 'parquet')}")
    out_path = str(raw_dir / raw_filename)

    aliases = db_cfg.get("selected_aliases", [])
    include_poi = bool(db_cfg.get("use_poi", True))
    include_ztl = bool(db_cfg.get("use_ztl", True))

    DatasetBuilder().retrieve_data(
        schema_path=schema_path,
        selected_aliases=aliases,
        output_path=out_path,
        include_poi=include_poi,
        include_ztl=include_ztl,
        poi_categories=None,
        output_format=db_cfg.get("output_format", "parquet"),
        compression=db_cfg.get("compression", None),
    )
