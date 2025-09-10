"""
Modulo per la creazione di feature derivate basate sui risultati EDA.
Implementa feature engineering specifico per dati immobiliari.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


def create_price_ratios(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Crea ratios di prezzo basati su correlazioni forti trovate in EDA."""
    df_new = df.copy()
    
    price_col = 'AI_Prezzo_Ridistribuito'
    rendita_col = 'AI_Rendita' 
    superficie_col = 'AI_Superficie'
    
    if price_col in df.columns and superficie_col in df.columns:
        # Prezzo per metro quadro
        df_new['prezzo_per_mq'] = df[price_col] / df[superficie_col].clip(lower=1)
        logger.info("Creata feature: prezzo_per_mq")
    
    if rendita_col in df.columns and superficie_col in df.columns:
        # Rendita per metro quadro
        df_new['rendita_per_mq'] = df[rendita_col] / df[superficie_col].clip(lower=1)
        logger.info("Creata feature: rendita_per_mq")
    
    if price_col in df.columns and rendita_col in df.columns:
        # Rapporto prezzo/rendita
        df_new['prezzo_vs_rendita'] = df[price_col] / df[rendita_col].clip(lower=1)
        logger.info("Creata feature: prezzo_vs_rendita")
    
    # Superficie vs media per zona
    if superficie_col in df.columns and 'AI_ZonaOmi' in df.columns:
        zona_superficie_mean = df.groupby('AI_ZonaOmi')[superficie_col].transform('mean')
        df_new['superficie_vs_media_zona'] = df[superficie_col] / zona_superficie_mean.clip(lower=1)
        logger.info("Creata feature: superficie_vs_media_zona")
    
    return df_new


def create_spatial_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Crea feature geospaziali basate su POI e coordinate."""
    df_new = df.copy()
    
    # Identifica colonne POI
    poi_cols = [col for col in df.columns if col.startswith('POI_') and col.endswith('_count')]
    
    if poi_cols:
        # Densità totale POI
        df_new['poi_density_total'] = df[poi_cols].sum(axis=1)
        logger.info(f"Creata feature: poi_density_total da {len(poi_cols)} colonne POI")
        
        # Ratio POI shopping
        if 'POI_shopping_mall_count' in df.columns:
            total_poi = df_new['poi_density_total'].clip(lower=1)
            df_new['poi_shopping_ratio'] = df['POI_shopping_mall_count'] / total_poi
            logger.info("Creata feature: poi_shopping_ratio")
    
    # Distanza dal centro (assumendo centro città a coordinate medie)
    if 'AI_Latitudine' in df.columns and 'AI_Longitudine' in df.columns:
        lat_center = df['AI_Latitudine'].mean()
        lon_center = df['AI_Longitudine'].mean()
        
        df_new['distance_to_center'] = np.sqrt(
            (df['AI_Latitudine'] - lat_center)**2 + 
            (df['AI_Longitudine'] - lon_center)**2
        )
        logger.info("Creata feature: distance_to_center")
    
    # Ranking prezzo per zona OMI
    if 'AI_ZonaOmi' in df.columns and 'AI_Prezzo_Ridistribuito' in df.columns:
        zona_price_mean = df.groupby('AI_ZonaOmi')['AI_Prezzo_Ridistribuito'].transform('mean')
        df_new['zona_omi_price_rank'] = zona_price_mean.rank(pct=True)
        logger.info("Creata feature: zona_omi_price_rank")
    
    return df_new


def create_temporal_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Crea feature temporali per catturare stagionalità e trend."""
    df_new = df.copy()
    
    if 'A_AnnoStipula' in df.columns:
        # Anno normalizzato
        min_year = df['A_AnnoStipula'].min()
        max_year = df['A_AnnoStipula'].max()
        if max_year > min_year:
            df_new['anno_stipula_normalized'] = (df['A_AnnoStipula'] - min_year) / (max_year - min_year)
            logger.info("Creata feature: anno_stipula_normalized")
    
    if 'A_MeseStipula' in df.columns:
        # Encoding ciclico per mese
        df_new['mese_sin'] = np.sin(2 * np.pi * df['A_MeseStipula'] / 12)
        df_new['mese_cos'] = np.cos(2 * np.pi * df['A_MeseStipula'] / 12)
        
        # Trimestre
        df_new['trimestre'] = ((df['A_MeseStipula'] - 1) // 3) + 1
        
        logger.info("Create feature temporali: mese_sin, mese_cos, trimestre")
    
    return df_new


def create_categorical_aggregates(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Crea feature aggregate per categorie (target encoding style)."""
    df_new = df.copy()
    
    group_cols = ['AI_ZonaOmi', 'AI_IdCategoriaCatastale', 'AI_IdTipologiaEdilizia']
    target_col = 'AI_Prezzo_Ridistribuito'
    superficie_col = 'AI_Superficie'
    
    # Verifica che le colonne esistano
    available_group_cols = [col for col in group_cols if col in df.columns]
    
    if not available_group_cols or target_col not in df.columns:
        logger.warning("Colonne necessarie per aggregati categorici non trovate")
        return df_new
    
    for group_col in available_group_cols:
        # Prezzo medio per gruppo
        group_mean = df.groupby(group_col)[target_col].transform('mean')
        df_new[f'prezzo_medio_{group_col}'] = group_mean
        
        # Deviazione standard per gruppo
        group_std = df.groupby(group_col)[target_col].transform('std').fillna(0)
        df_new[f'prezzo_std_{group_col}'] = group_std
        
        # Count per gruppo
        group_count = df.groupby(group_col)[target_col].transform('count')
        df_new[f'count_{group_col}'] = group_count
        
        # Percentile del prezzo nel gruppo
        group_rank = df.groupby(group_col)[target_col].rank(pct=True)
        df_new[f'prezzo_percentile_{group_col}'] = group_rank
        
        # Superficie media per gruppo (se disponibile)
        if superficie_col in df.columns:
            superficie_mean = df.groupby(group_col)[superficie_col].transform('mean')
            df_new[f'superficie_media_{group_col}'] = superficie_mean
        
        logger.info(f"Create feature aggregate per {group_col}")
    
    return df_new


def create_derived_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Funzione principale per creare tutte le feature derivate.
    """
    fe_config = config.get('feature_extraction', {}).get('derived_features', {})
    
    if not fe_config.get('enabled', False):
        logger.info("Feature derivate disabilitate")
        return df
    
    logger.info("Inizio creazione feature derivate")
    df_enhanced = df.copy()
    
    # Price ratios
    if fe_config.get('price_ratios', {}).get('enabled', False):
        df_enhanced = create_price_ratios(df_enhanced, config)
    
    # Spatial features
    if fe_config.get('spatial_features', {}).get('enabled', False):
        df_enhanced = create_spatial_features(df_enhanced, config)
    
    # Temporal features
    if fe_config.get('temporal_features', {}).get('enabled', False):
        df_enhanced = create_temporal_features(df_enhanced, config)
    
    # Categorical aggregates
    if fe_config.get('categorical_aggregates', {}).get('enabled', False):
        df_enhanced = create_categorical_aggregates(df_enhanced, config)
    
    new_features = len(df_enhanced.columns) - len(df.columns)
    logger.info(f"Feature derivate completate: aggiunte {new_features} nuove feature")
    
    return df_enhanced