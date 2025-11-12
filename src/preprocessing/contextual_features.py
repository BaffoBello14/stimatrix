"""
Feature engineering: contesto territoriale e di mercato.

Questo modulo aggiunge feature aggregate che catturano il contesto
di mercato locale per zona, tipologia e combinazioni.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


def add_zone_statistics(
    df: pd.DataFrame, 
    target_col: str = 'AI_Prezzo_Ridistribuito',
    group_col: str = 'AI_ZonaOmi'
) -> pd.DataFrame:
    """
    Aggiungi statistiche aggregate per zona OMI.
    
    Cattura il "livello di mercato" della zona: prezzi medi, variabilità,
    quartili. Permette al modello di capire se un immobile è sopra/sotto
    la media della sua zona.
    
    Args:
        df: DataFrame con i dati
        target_col: Colonna target (prezzo)
        group_col: Colonna di raggruppamento (zona)
    
    Returns:
        DataFrame con feature aggregate aggiunte
    """
    if group_col not in df.columns:
        logger.warning(f"Colonna {group_col} non trovata, skip zone statistics")
        return df
    
    if target_col not in df.columns:
        logger.warning(f"Target {target_col} non trovato, skip zone statistics")
        return df
    
    df = df.copy()
    
    # Statistiche aggregate per zona
    zone_stats = df.groupby(group_col)[target_col].agg([
        ('zone_price_mean', 'mean'),
        ('zone_price_median', 'median'),
        ('zone_price_std', 'std'),
        ('zone_price_min', 'min'),
        ('zone_price_max', 'max'),
        ('zone_price_q25', lambda x: x.quantile(0.25)),
        ('zone_price_q75', lambda x: x.quantile(0.75)),
        ('zone_count', 'count'),
    ])
    
    # Merge back
    df = df.merge(zone_stats, left_on=group_col, right_index=True, how='left')
    
    # Feature derivate: posizione relativa dell'immobile rispetto alla zona
    df['price_vs_zone_mean_ratio'] = df[target_col] / (df['zone_price_mean'] + 1e-8)
    df['price_vs_zone_median_ratio'] = df[target_col] / (df['zone_price_median'] + 1e-8)
    df['price_zone_zscore'] = (df[target_col] - df['zone_price_mean']) / (df['zone_price_std'] + 1e-8)
    df['price_zone_iqr_position'] = (df[target_col] - df['zone_price_q25']) / (df['zone_price_q75'] - df['zone_price_q25'] + 1e-8)
    df['price_zone_range_position'] = (df[target_col] - df['zone_price_min']) / (df['zone_price_max'] - df['zone_price_min'] + 1e-8)
    
    logger.info(f"✅ Zone statistics: aggiunte 13 feature per {group_col}")
    return df


def add_typology_statistics(
    df: pd.DataFrame,
    target_col: str = 'AI_Prezzo_Ridistribuito',
    zone_col: str = 'AI_ZonaOmi',
    type_col: str = 'AI_IdTipologiaEdilizia'
) -> pd.DataFrame:
    """
    Aggiungi statistiche per tipologia edilizia × zona.
    
    Cattura nicchie di mercato: es. appartamenti in zona D2 hanno
    dinamiche diverse da negozi in zona C4.
    
    Args:
        df: DataFrame con i dati
        target_col: Colonna target
        zone_col: Colonna zona
        type_col: Colonna tipologia
    
    Returns:
        DataFrame con feature aggregate aggiunte
    """
    required_cols = [zone_col, type_col, target_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(f"Colonne mancanti {missing}, skip typology statistics")
        return df
    
    df = df.copy()
    
    # Statistiche per tipologia × zona
    type_zone_stats = df.groupby([type_col, zone_col])[target_col].agg([
        ('type_zone_price_mean', 'mean'),
        ('type_zone_price_median', 'median'),
        ('type_zone_price_std', 'std'),
        ('type_zone_count', 'count'),
    ])
    
    df = df.merge(
        type_zone_stats, 
        left_on=[type_col, zone_col], 
        right_index=True, 
        how='left'
    )
    
    # Statistiche solo per tipologia (globale)
    type_stats = df.groupby(type_col)[target_col].agg([
        ('type_price_mean', 'mean'),
        ('type_price_median', 'median'),
    ])
    
    df = df.merge(type_stats, left_on=type_col, right_index=True, how='left')
    
    # Feature derivate
    df['price_vs_type_zone_mean'] = df[target_col] / (df['type_zone_price_mean'] + 1e-8)
    df['type_zone_rarity'] = 1.0 / (df['type_zone_count'] + 1)  # Quanto è rara questa combinazione
    
    logger.info(f"✅ Typology statistics: aggiunte 8 feature per {type_col} × {zone_col}")
    return df


def add_surface_context(
    df: pd.DataFrame,
    surface_col: str = 'AI_Superficie',
    zone_col: str = 'AI_ZonaOmi',
    type_col: str = 'AI_IdTipologiaEdilizia'
) -> pd.DataFrame:
    """
    Aggiungi contesto di superficie per zona/tipologia.
    
    Cattura se un immobile è grande/piccolo rispetto alla media
    della sua categoria nella zona.
    
    Args:
        df: DataFrame
        surface_col: Colonna superficie
        zone_col: Colonna zona
        type_col: Colonna tipologia
    
    Returns:
        DataFrame con feature aggiunte
    """
    if surface_col not in df.columns:
        logger.warning(f"Colonna {surface_col} non trovata, skip surface context")
        return df
    
    df = df.copy()
    
    # Superficie media per zona
    if zone_col in df.columns:
        zone_surface = df.groupby(zone_col)[surface_col].agg([
            ('zone_surface_mean', 'mean'),
            ('zone_surface_median', 'median'),
        ])
        df = df.merge(zone_surface, left_on=zone_col, right_index=True, how='left')
        df['surface_vs_zone_mean'] = df[surface_col] / (df['zone_surface_mean'] + 1e-8)
    
    # Superficie media per tipologia × zona
    if type_col in df.columns and zone_col in df.columns:
        type_zone_surface = df.groupby([type_col, zone_col])[surface_col].agg([
            ('type_zone_surface_mean', 'mean'),
        ])
        df = df.merge(type_zone_surface, left_on=[type_col, zone_col], right_index=True, how='left')
        df['surface_vs_type_zone_mean'] = df[surface_col] / (df['type_zone_surface_mean'] + 1e-8)
    
    logger.info(f"✅ Surface context: aggiunte 5 feature per {surface_col}")
    return df


def add_interaction_features(
    df: pd.DataFrame,
    target_col: str = 'AI_Prezzo_Ridistribuito',
    surface_col: str = 'AI_Superficie'
) -> pd.DataFrame:
    """
    Aggiungi feature di interazione chiave.
    
    Cattura relazioni non-lineari importanti come prezzo/mq
    e interazioni con zone/tipologie.
    
    Args:
        df: DataFrame
        target_col: Colonna target
        surface_col: Colonna superficie
    
    Returns:
        DataFrame con interazioni aggiunte
    """
    df = df.copy()
    n_added = 0
    
    # Prezzo al metro quadro
    if surface_col in df.columns and target_col in df.columns:
        df['prezzo_mq'] = df[target_col] / (df[surface_col] + 1.0)
        n_added += 1
        
        # Prezzo/mq relativo alla zona (se disponibile)
        if 'zone_price_mean' in df.columns and 'zone_surface_mean' in df.columns:
            zone_prezzo_mq = df['zone_price_mean'] / (df['zone_surface_mean'] + 1.0)
            df['prezzo_mq_vs_zone'] = df['prezzo_mq'] / (zone_prezzo_mq + 1e-8)
            n_added += 1
    
    # Log superficie (cattura effetti scala)
    if surface_col in df.columns:
        df['log_superficie'] = np.log1p(df[surface_col])
        n_added += 1
    
    # Superficie × categoria catastale (alcune categorie premiano grandi superfici)
    if surface_col in df.columns and 'AI_IdCategoriaCatastale' in df.columns:
        # Convert to string to avoid numerical issues
        df['superficie_x_categoria'] = df[surface_col] * df['AI_IdCategoriaCatastale'].astype(str).astype('category').cat.codes
        n_added += 1
    
    logger.info(f"✅ Interaction features: aggiunte {n_added} feature")
    return df


def add_temporal_context(
    df: pd.DataFrame,
    target_col: str = 'AI_Prezzo_Ridistribuito',
    year_col: str = 'A_AnnoStipula',
    month_col: str = 'A_MeseStipula'
) -> pd.DataFrame:
    """
    Aggiungi contesto temporale e trend di mercato.
    
    Cattura trend di mercato e stagionalità dei prezzi.
    
    Args:
        df: DataFrame
        target_col: Colonna target
        year_col: Colonna anno
        month_col: Colonna mese
    
    Returns:
        DataFrame con feature temporali aggiunte
    """
    if year_col not in df.columns or month_col not in df.columns:
        logger.warning(f"Colonne temporali non trovate, skip temporal context")
        return df
    
    df = df.copy()
    
    # Crea chiave temporale
    df['year_month'] = df[year_col] * 100 + df[month_col]
    
    # Trend temporale: prezzo medio per mese
    if target_col in df.columns:
        temporal_stats = df.groupby('year_month')[target_col].agg([
            ('temporal_price_mean', 'mean'),
            ('temporal_price_median', 'median'),
            ('temporal_count', 'count'),
        ])
        df = df.merge(temporal_stats, left_on='year_month', right_index=True, how='left')
        
        # Prezzo relativo al trend temporale
        df['price_vs_temporal_mean'] = df[target_col] / (df['temporal_price_mean'] + 1e-8)
    
    # Stagionalità (trimestre)
    df['quarter'] = ((df[month_col] - 1) // 3) + 1
    
    # Mesi dalla data minima (trend lineare)
    min_date = df['year_month'].min()
    df['months_from_start'] = (df['year_month'] - min_date) // 100 * 12 + (df['year_month'] - min_date) % 100
    
    logger.info(f"✅ Temporal context: aggiunte 7 feature temporali")
    return df


def add_all_contextual_features(
    df: pd.DataFrame,
    target_col: str = 'AI_Prezzo_Ridistribuito',
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Aggiungi tutte le feature contestuali in un solo passaggio.
    
    Args:
        df: DataFrame originale
        target_col: Colonna target
        config: Configurazione opzionale
    
    Returns:
        DataFrame arricchito con feature contestuali
    """
    logger.info("🚀 Inizio aggiunta feature contestuali...")
    initial_cols = len(df.columns)
    
    # Zone statistics
    df = add_zone_statistics(df, target_col=target_col)
    
    # Typology statistics
    df = add_typology_statistics(df, target_col=target_col)
    
    # Surface context
    df = add_surface_context(df)
    
    # Interaction features
    df = add_interaction_features(df, target_col=target_col)
    
    # Temporal context
    df = add_temporal_context(df, target_col=target_col)
    
    final_cols = len(df.columns)
    added_cols = final_cols - initial_cols
    
    logger.info(f"✅ Feature contestuali completate: {added_cols} nuove feature aggiunte")
    logger.info(f"   Colonne totali: {initial_cols} → {final_cols}")
    
    return df
