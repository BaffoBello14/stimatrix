from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DatasetBuilder:
    """
    Classe per la costruzione del dataset di base da dati grezzi.
    Esegue operazioni fondamentali prima della pulizia e preprocessing.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.building_config = config.get("dataset_building", {})

    def build_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        logger.info("Inizio costruzione dataset...")

        building_stats: Dict[str, Any] = {
            "original_shape": df.shape,
            "coherent_acts_filter": {},
            "price_estimation": {},
            "floor_features": {},
            "final_shape": None,
        }

        if self.building_config.get("filter_coherent_acts", True):
            df, coherent_stats = self._filter_coherent_acts(df)
            building_stats["coherent_acts_filter"] = coherent_stats

        if self.building_config.get("estimate_prices", True):
            df, price_stats = self._estimate_prices_dual(df)
            building_stats["price_estimation"] = price_stats

        if self.building_config.get("process_floor_features", True):
            df, floor_stats = self._process_floor_features(df)
            building_stats["floor_features"] = floor_stats

        if self.building_config.get("remove_all_nan_columns", True):
            df = self._remove_all_nan_columns(df)

        building_stats["final_shape"] = df.shape

        logger.info(
            f"Dataset costruito: {building_stats['original_shape']} -> {building_stats['final_shape']}"
        )

        return df, building_stats

    def _filter_coherent_acts(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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

    def _estimate_prices_dual(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
            .agg(prezzo_stimato_totale=("prezzo_stimato_immobile", "sum"), A_Prezzo=("A_Prezzo", "first"))
            .reset_index()
        )

        valid_mask = (prezzi["prezzo_stimato_totale"] > 0) & (prezzi["A_Prezzo"] > 0)
        invalid_count = (~valid_mask).sum()

        if invalid_count > 0:
            logger.warning(
                f"Trovati {invalid_count} atti con prezzi stimati o reali <= 0; uso coefficiente = 1.0"
            )

        prezzi["coefficiente"] = 1.0
        prezzi.loc[valid_mask, "coefficiente"] = (
            prezzi.loc[valid_mask, "A_Prezzo"] / prezzi.loc[valid_mask, "prezzo_stimato_totale"]
        )

        valid_coefficients = prezzi.loc[valid_mask, "coefficiente"]
        if len(valid_coefficients) > 0:
            logger.info(
                f"Coefficienti di ridistribuzione: min={valid_coefficients.min():.3f}, max={valid_coefficients.max():.3f}, mean={valid_coefficients.mean():.3f}"
            )

        anomalous_coeff = valid_coefficients[
            (valid_coefficients < 0.1) | (valid_coefficients > 10)
        ]
        if len(anomalous_coeff) > 0:
            logger.warning(
                f"Trovati {len(anomalous_coeff)} coefficienti anomali (<0.1 o >10)"
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

    def _process_floor_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if "AI_Piano" not in df.columns:
            logger.warning("Colonna AI_Piano non trovata nel DataFrame")
            return df, {"skipped": True, "reason": "missing_column"}

        logger.info("Elaborazione features del piano...")

        floor_features = df["AI_Piano"].apply(self._extract_floor_features)
        floor_df = pd.DataFrame(floor_features.tolist())
        floor_df.columns = ["floor_" + col for col in floor_df.columns]
        df_with_floors = pd.concat([df, floor_df], axis=1)

        non_null_floors = df_with_floors["floor_floor_numeric_weighted"].notna().sum()

        stats: Dict[str, Any] = {
            "features_added": list(floor_df.columns),
            "total_properties": len(df_with_floors),
            "non_null_floor_weights": int(non_null_floors),
            "floor_weight_coverage": float(non_null_floors / len(df_with_floors))
            if len(df_with_floors) > 0
            else 0.0,
        }

        if non_null_floors > 0:
            min_floor = float(df_with_floors["floor_floor_numeric_weighted"].min())
            max_floor = float(df_with_floors["floor_floor_numeric_weighted"].max())
            stats["floor_weight_range"] = [min_floor, max_floor]
            logger.info(
                f"Range floor_numeric_weighted: {min_floor:.2f} - {max_floor:.2f}"
            )

        logger.info(f"Features del piano aggiunte: {list(floor_df.columns)}")
        logger.info(
            f"Valori non-null in floor_numeric_weighted: {non_null_floors}/{len(df_with_floors)}"
        )

        return df_with_floors, stats

    def _extract_floor_features(self, floor_str: str) -> Dict[str, float]:
        if pd.isna(floor_str) or floor_str in ["NULL", "", None]:
            return {
                "min_floor": np.nan,
                "max_floor": np.nan,
                "n_floors": np.float64(0),
                "has_basement": np.float64(0),
                "has_ground": np.float64(0),
                "has_upper": np.float64(0),
                "floor_span": np.float64(0),
                "floor_numeric_weighted": np.nan,
            }

        floor_str = str(floor_str).strip().upper()

        floor_mapping: Dict[str, float] = {
            "S2": -2,
            "S1": -1,
            "S": -1,
            "SEMI": -1,
            "T": 0,
            "PT": 0,
            "RIAL": 0.5,
            "ST": -0.5,
            "P1": 1,
            "P2": 2,
            "P3": 3,
            "P4": 4,
            "P5": 5,
            "P6": 6,
            "P7": 7,
            "P8": 8,
            "P9": 9,
            "P10": 10,
            "P11": 11,
            "P12": 12,
        }

        floors_found: list[float] = []

        patterns = [
            r"P(\d+)",
            r"S(\d*)",
            r"PT",
            r"T(?![0-9])",
            r"ST",
            r"SEMI",
            r"RIAL",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, floor_str)
            if pattern == r"P(\d+)":
                for match in matches:
                    key = f"P{match}"
                    if key in floor_mapping:
                        floors_found.append(float(floor_mapping[key]))
            elif pattern == r"S(\d*)":
                for match in matches:
                    if match == "":
                        floors_found.append(float(floor_mapping["S"]))
                    else:
                        key = f"S{match}"
                        if key in floor_mapping:
                            floors_found.append(float(floor_mapping[key]))
            else:
                if re.search(pattern, floor_str):
                    key = pattern.replace(r"(?![0-9])", "").replace(r"\b", "")
                    if key in floor_mapping:
                        floors_found.append(float(floor_mapping[key]))

        isolated_numbers = re.findall(r"\b(\d+)\b", floor_str)
        for num in isolated_numbers:
            num_val = int(num)
            if 1 <= num_val <= 12:
                floors_found.append(float(num_val))

        floors_found = sorted(list(set(floors_found)))

        if not floors_found:
            try:
                single_floor = float(floor_str)
                if -5 <= single_floor <= 15:
                    floors_found = [single_floor]
            except (ValueError, TypeError):
                pass

        if floors_found:
            min_floor = float(min(floors_found))
            max_floor = float(max(floors_found))
            n_floors = np.float64(len(floors_found))
            floor_span = np.float64(max_floor - min_floor)

            weights = [f + 3 for f in floors_found]
            try:
                floor_numeric_weighted = float(np.average(floors_found, weights=weights))
            except ZeroDivisionError:
                floor_numeric_weighted = float(np.mean(floors_found))

            has_basement = np.float64(1) if any(f < 0 for f in floors_found) else np.float64(0)
            has_ground = np.float64(1) if any(-0.5 <= f <= 0.5 for f in floors_found) else np.float64(0)
            has_upper = np.float64(1) if any(f >= 1 for f in floors_found) else np.float64(0)
        else:
            min_floor = max_floor = np.nan
            n_floors = floor_span = np.float64(0)
            has_basement = has_ground = has_upper = np.float64(0)
            floor_numeric_weighted = np.nan

        return {
            "min_floor": float(min_floor) if not pd.isna(min_floor) else np.nan,
            "max_floor": float(max_floor) if not pd.isna(max_floor) else np.nan,
            "n_floors": n_floors,
            "has_basement": has_basement,
            "has_ground": has_ground,
            "has_upper": has_upper,
            "floor_span": floor_span,
            "floor_numeric_weighted": float(floor_numeric_weighted)
            if not pd.isna(floor_numeric_weighted)
            else np.nan,
        }

    def _remove_all_nan_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Rimozione colonne interamente NaN...")

        all_nan_cols = df.columns[df.isna().all()].tolist()

        if all_nan_cols:
            df = df.drop(columns=all_nan_cols)
            logger.info(f"Colonne interamente NaN rimosse: {all_nan_cols}")
        else:
            logger.info("Nessuna colonna interamente NaN trovata")

        return df