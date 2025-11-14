"""
Feature engineering: contesto territoriale e di mercato (LEAK-FREE).

Questo modulo aggiunge feature aggregate che catturano il contesto
di mercato locale per zona, tipologia e combinazioni.

IMPORTANTE: Usa fit/transform pattern per evitare data leakage.
- fit_contextual_features: calcola statistiche SOLO su train
- transform_contextual_features: applica statistiche a train/val/test
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


def fit_contextual_features(
    df: pd.DataFrame,
    target_col: str = 'AI_Prezzo_Ridistribuito',
    zone_col: str = 'AI_ZonaOmi',
    type_col: str = 'AI_IdTipologiaEdilizia',
    surface_col: str = 'AI_Superficie',
    year_col: str = 'A_AnnoStipula',
    month_col: str = 'A_MeseStipula'
) -> Dict[str, Any]:
    """
    Calcola statistiche aggregate dal dataset di TRAINING.
    
    Args:
        df: DataFrame di TRAINING
        target_col: Colonna target (prezzo)
        zone_col: Colonna zona
        type_col: Colonna tipologia
        surface_col: Colonna superficie
        year_col: Colonna anno
        month_col: Colonna mese
    
    Returns:
        Dictionary con statistiche aggregate (da salvare/riusare)
    """
    stats = {}
    
    logger.info("🔧 Fit contextual features (TRAINING only)")
    
    # ==========================================
    # 1. ZONE STATISTICS
    # ==========================================
    if zone_col in df.columns and target_col in df.columns:
        zone_stats = df.groupby(zone_col)[target_col].agg([
            ('zone_price_mean', 'mean'),
            ('zone_price_median', 'median'),
            ('zone_price_std', 'std'),
            ('zone_price_min', 'min'),
            ('zone_price_max', 'max'),
            ('zone_price_q25', lambda x: x.quantile(0.25)),
            ('zone_price_q75', lambda x: x.quantile(0.75)),
            ('zone_count', 'count'),
        ])
        stats['zone_price'] = zone_stats.to_dict('index')
        
        # Surface per zona
        if surface_col in df.columns:
            zone_surface = df.groupby(zone_col)[surface_col].agg([
                ('zone_surface_mean', 'mean'),
                ('zone_surface_median', 'median'),
            ])
            stats['zone_surface'] = zone_surface.to_dict('index')
        
        logger.info(f"  ✅ Zone statistics: {len(zone_stats)} zones")
    
    # ==========================================
    # 2. TYPOLOGY × ZONE STATISTICS
    # ==========================================
    if all(c in df.columns for c in [type_col, zone_col, target_col]):
        type_zone_stats = df.groupby([type_col, zone_col])[target_col].agg([
            ('type_zone_price_mean', 'mean'),
            ('type_zone_price_median', 'median'),
            ('type_zone_price_std', 'std'),
            ('type_zone_count', 'count'),
        ])
        stats['type_zone_price'] = type_zone_stats.to_dict('index')
        
        # Global typology stats
        type_stats = df.groupby(type_col)[target_col].agg([
            ('type_price_mean', 'mean'),
            ('type_price_median', 'median'),
        ])
        stats['type_price'] = type_stats.to_dict('index')
        
        # Surface per typology × zone
        if surface_col in df.columns:
            type_zone_surface = df.groupby([type_col, zone_col])[surface_col].agg([
                ('type_zone_surface_mean', 'mean'),
            ])
            stats['type_zone_surface'] = type_zone_surface.to_dict('index')
        
        logger.info(f"  ✅ Typology × Zone statistics: {len(type_zone_stats)} combos")
    
    # ==========================================
    # 3. TEMPORAL STATISTICS
    # ==========================================
    if all(c in df.columns for c in [year_col, month_col, target_col]):
        df_temp = df.copy()
        df_temp['year_month'] = df_temp[year_col] * 100 + df_temp[month_col]
        
        temporal_stats = df_temp.groupby('year_month')[target_col].agg([
            ('temporal_price_mean', 'mean'),
            ('temporal_price_median', 'median'),
            ('temporal_count', 'count'),
        ])
        stats['temporal_price'] = temporal_stats.to_dict('index')
        stats['min_year_month'] = df_temp['year_month'].min()
        
        logger.info(f"  ✅ Temporal statistics: {len(temporal_stats)} periods")
    
    logger.info(f"✅ Fit completed: {len(stats)} stat groups")
    return stats


def transform_contextual_features(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    target_col: str = 'AI_Prezzo_Ridistribuito',
    zone_col: str = 'AI_ZonaOmi',
    type_col: str = 'AI_IdTipologiaEdilizia',
    surface_col: str = 'AI_Superficie',
    year_col: str = 'A_AnnoStipula',
    month_col: str = 'A_MeseStipula'
) -> pd.DataFrame:
    """
    Applica statistiche pre-calcolate al DataFrame (train/val/test).
    
    Args:
        df: DataFrame da trasformare
        stats: Statistiche calcolate con fit_contextual_features
        target_col: Colonna target
        zone_col: Colonna zona
        type_col: Colonna tipologia
        surface_col: Colonna superficie
        year_col: Colonna anno
        month_col: Colonna mese
    
    Returns:
        DataFrame con feature contestuali aggiunte
    """
    df = df.copy()
    initial_cols = len(df.columns)
    
    # ==========================================
    # 1. ZONE STATISTICS
    # ==========================================
    if 'zone_price' in stats and zone_col in df.columns:
        zone_price_df = pd.DataFrame.from_dict(stats['zone_price'], orient='index')
        df = df.merge(zone_price_df, left_on=zone_col, right_index=True, how='left')
    
    if 'zone_surface' in stats and zone_col in df.columns and surface_col in df.columns:
        zone_surface_df = pd.DataFrame.from_dict(stats['zone_surface'], orient='index')
        df = df.merge(zone_surface_df, left_on=zone_col, right_index=True, how='left')
        df['surface_vs_zone_mean'] = df[surface_col] / (df['zone_surface_mean'] + 1e-8)
    
    # ==========================================
    # 2. TYPOLOGY × ZONE STATISTICS
    # ==========================================
    if 'type_zone_price' in stats and all(c in df.columns for c in [type_col, zone_col]):
        type_zone_price_df = pd.DataFrame.from_dict(stats['type_zone_price'], orient='index')
        type_zone_price_df.index = pd.MultiIndex.from_tuples(type_zone_price_df.index)
        df = df.merge(
            type_zone_price_df,
            left_on=[type_col, zone_col],
            right_index=True,
            how='left'
        )
        
        # ✅ type_zone_rarity: uses count, not target instance (LEAK-FREE)
        df['type_zone_rarity'] = 1.0 / (df['type_zone_count'] + 1)
    
    if 'type_price' in stats and type_col in df.columns:
        type_price_df = pd.DataFrame.from_dict(stats['type_price'], orient='index')
        df = df.merge(type_price_df, left_on=type_col, right_index=True, how='left')
    
    if 'type_zone_surface' in stats and all(c in df.columns for c in [type_col, zone_col, surface_col]):
        type_zone_surface_df = pd.DataFrame.from_dict(stats['type_zone_surface'], orient='index')
        type_zone_surface_df.index = pd.MultiIndex.from_tuples(type_zone_surface_df.index)
        df = df.merge(
            type_zone_surface_df,
            left_on=[type_col, zone_col],
            right_index=True,
            how='left'
        )
        df['surface_vs_type_zone_mean'] = df[surface_col] / (df['type_zone_surface_mean'] + 1e-8)
    
    # ==========================================
    # 3. INTERACTION FEATURES (LEAK-FREE: no target-based features!)
    # ==========================================
    if surface_col in df.columns:
        df['log_superficie'] = np.log1p(df[surface_col])
    
    if surface_col in df.columns and 'AI_IdCategoriaCatastale' in df.columns:
        df['superficie_x_categoria'] = df[surface_col] * df['AI_IdCategoriaCatastale'].astype(str).astype('category').cat.codes
    
    # ==========================================
    # 4. TEMPORAL FEATURES
    # ==========================================
    if 'temporal_price' in stats and all(c in df.columns for c in [year_col, month_col]):
        df['year_month'] = df[year_col] * 100 + df[month_col]
        
        temporal_price_df = pd.DataFrame.from_dict(stats['temporal_price'], orient='index')
        df = df.merge(temporal_price_df, left_on='year_month', right_index=True, how='left')
        
        # Seasonality (LEAK-FREE)
        df['quarter'] = ((df[month_col] - 1) // 3) + 1
        
        # Months from start
        if 'min_year_month' in stats:
            min_date = stats['min_year_month']
            df['months_from_start'] = (df['year_month'] - min_date) // 100 * 12 + (df['year_month'] - min_date) % 100
    
    final_cols = len(df.columns)
    added_cols = final_cols - initial_cols
    
    logger.info(f"  ➡️ Transform: +{added_cols} features ({initial_cols} → {final_cols})")
    return df


def fit_transform_contextual_features(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    target_col: str = 'AI_Prezzo_Ridistribuito',
    **kwargs
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Fit su train e transform su train/val/test in un colpo solo.
    
    Args:
        train_df: DataFrame di training
        val_df: DataFrame di validation (opzionale)
        test_df: DataFrame di test (opzionale)
        target_col: Colonna target
        **kwargs: Altri parametri per fit
    
    Returns:
        (train_transformed, val_transformed, test_transformed, stats)
    """
    logger.info("🎯 Fit + Transform contextual features (LEAK-FREE)")
    
    # Fit SOLO su train
    stats = fit_contextual_features(train_df, target_col=target_col, **kwargs)
    
    # Transform train/val/test con statistiche del train
    train_out = transform_contextual_features(train_df, stats, target_col=target_col, **kwargs)
    
    val_out = None
    if val_df is not None:
        val_out = transform_contextual_features(val_df, stats, target_col=target_col, **kwargs)
    
    test_out = None
    if test_df is not None:
        test_out = transform_contextual_features(test_df, stats, target_col=target_col, **kwargs)
    
    logger.info("✅ Contextual features completed (LEAK-FREE)")
    return train_out, val_out, test_out, stats
