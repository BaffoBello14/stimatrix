"""
Target Encoding Corretto SENZA Data Leakage.
Implementa target encoding usando solo i dati di training per calcolare le medie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import KFold
from utils.logger import get_logger

logger = get_logger(__name__)


class SafeTargetEncoder:
    """
    Target encoder che previene data leakage usando solo dati di training.
    """
    
    def __init__(self, 
                 categorical_columns: List[str],
                 target_column: str,
                 smoothing: float = 1.0,
                 min_samples_leaf: int = 1,
                 noise_level: float = 0.01):
        """
        Args:
            categorical_columns: Colonne categoriche da encodare
            target_column: Nome della colonna target
            smoothing: Parametro di smoothing (regolarizzazione)
            min_samples_leaf: Minimo numero di campioni per categoria
            noise_level: Rumore per regolarizzazione
        """
        self.categorical_columns = categorical_columns
        self.target_column = target_column
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        
        # Dizionari per memorizzare le medie calcolate su training
        self.target_means_: Dict[str, Dict] = {}
        self.global_mean_: float = 0.0
        self.fitted_ = False
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'SafeTargetEncoder':
        """
        Fit del target encoder usando SOLO i dati di training.
        """
        logger.info("Fitting SafeTargetEncoder su dati di training")
        
        # Media globale del target (fallback)
        self.global_mean_ = y_train.mean()
        
        # Calcola medie per ogni colonna categorica
        for col in self.categorical_columns:
            if col not in X_train.columns:
                logger.warning(f"Colonna {col} non trovata in training data")
                continue
            
            # Calcola statistiche per categoria usando SOLO training data
            stats = pd.DataFrame({
                'target_sum': X_train.groupby(col)[y_train.name].apply(lambda x: y_train.loc[x.index].sum()),
                'target_count': X_train.groupby(col)[y_train.name].apply(lambda x: len(y_train.loc[x.index]))
            })
            
            # Smoothing: (sum + smoothing * global_mean) / (count + smoothing)
            stats['target_mean'] = (
                (stats['target_sum'] + self.smoothing * self.global_mean_) / 
                (stats['target_count'] + self.smoothing)
            )
            
            # Filtra categorie con troppo pochi campioni
            stats = stats[stats['target_count'] >= self.min_samples_leaf]
            
            # Memorizza le medie
            self.target_means_[col] = stats['target_mean'].to_dict()
            
            logger.info(f"Target encoding per {col}: {len(stats)} categorie valide")
        
        self.fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Trasforma usando le medie calcolate su training (NO DATA LEAKAGE).
        """
        if not self.fitted_:
            raise ValueError("Encoder non ancora fitted. Chiama fit() prima.")
        
        X_encoded = X.copy()
        
        for col in self.categorical_columns:
            if col not in X.columns:
                continue
            
            col_means = self.target_means_.get(col, {})
            if not col_means:
                logger.warning(f"Nessuna media trovata per {col}")
                continue
            
            # Mappa le categorie alle loro medie
            # Categorie non viste in training → global mean
            encoded_values = X[col].map(col_means).fillna(self.global_mean_)
            
            # Aggiungi rumore per regolarizzazione (opzionale)
            if self.noise_level > 0:
                noise = np.random.normal(0, self.noise_level, len(encoded_values))
                encoded_values += noise
            
            # Crea nuova colonna
            new_col_name = f"{col}_target_encoded"
            X_encoded[new_col_name] = encoded_values
            
            logger.info(f"Trasformata {col} → {new_col_name}")
        
        return X_encoded
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """
        Fit e transform in un passo (solo per training data).
        """
        return self.fit(X_train, y_train).transform(X_train)


class CrossValidationTargetEncoder:
    """
    Target encoder con cross-validation per evitare overfitting anche su training.
    """
    
    def __init__(self, 
                 categorical_columns: List[str],
                 target_column: str,
                 cv_folds: int = 5,
                 smoothing: float = 1.0,
                 random_state: int = 42):
        """
        Args:
            cv_folds: Numero di fold per cross-validation
        """
        self.categorical_columns = categorical_columns
        self.target_column = target_column
        self.cv_folds = cv_folds
        self.smoothing = smoothing
        self.random_state = random_state
        
        # Per memorizzare encoder fitted su tutto il training
        self.global_encoder_ = SafeTargetEncoder(
            categorical_columns, target_column, smoothing
        )
        self.fitted_ = False
    
    def fit_transform_cv(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """
        Fit con cross-validation per training data (evita overfitting).
        """
        logger.info(f"CV Target Encoding con {self.cv_folds} fold")
        
        X_encoded = X_train.copy()
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Per ogni colonna categorica
        for col in self.categorical_columns:
            if col not in X_train.columns:
                continue
            
            encoded_values = np.full(len(X_train), np.nan)
            
            # Cross-validation: per ogni fold
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                
                # Training data per questo fold
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                
                # Validation data per questo fold  
                X_fold_val = X_train.iloc[val_idx]
                
                # Calcola medie SOLO su training di questo fold
                fold_encoder = SafeTargetEncoder([col], self.target_column, self.smoothing)
                fold_encoder.fit(X_fold_train, y_fold_train)
                
                # Applica a validation di questo fold
                X_fold_val_encoded = fold_encoder.transform(X_fold_val)
                new_col_name = f"{col}_target_encoded"
                
                # Memorizza risultati
                encoded_values[val_idx] = X_fold_val_encoded[new_col_name].values
            
            # Aggiungi colonna encodata
            X_encoded[f"{col}_target_encoded"] = encoded_values
            
            logger.info(f"CV Target encoding completato per {col}")
        
        # Fit encoder globale per transform futuro
        self.global_encoder_.fit(X_train, y_train)
        self.fitted_ = True
        
        return X_encoded
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform per validation/test usando encoder globale.
        """
        if not self.fitted_:
            raise ValueError("Encoder non fitted. Chiama fit_transform_cv() prima.")
        
        return self.global_encoder_.transform(X)


def apply_safe_target_encoding(X_train: pd.DataFrame, 
                              y_train: pd.Series,
                              X_val: Optional[pd.DataFrame] = None,
                              X_test: Optional[pd.DataFrame] = None,
                              categorical_columns: Optional[List[str]] = None,
                              use_cv: bool = True,
                              **encoder_kwargs) -> Tuple[pd.DataFrame, ...]:
    """
    Applica target encoding sicuro a train/val/test.
    
    Returns:
        Tuple di dataframe encodati (train, val, test) - val e test possono essere None
    """
    
    if categorical_columns is None:
        # Auto-detect categorical columns
        categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_columns:
        logger.info("Nessuna colonna categorica trovata per target encoding")
        return X_train, X_val, X_test
    
    target_column = y_train.name or 'target'
    
    logger.info(f"Target encoding su {len(categorical_columns)} colonne: {categorical_columns}")
    
    if use_cv:
        # Cross-validation target encoding
        encoder = CrossValidationTargetEncoder(
            categorical_columns, target_column, **encoder_kwargs
        )
        X_train_encoded = encoder.fit_transform_cv(X_train, y_train)
        
        # Transform val/test
        X_val_encoded = encoder.transform(X_val) if X_val is not None else None
        X_test_encoded = encoder.transform(X_test) if X_test is not None else None
        
    else:
        # Standard target encoding
        encoder = SafeTargetEncoder(categorical_columns, target_column, **encoder_kwargs)
        X_train_encoded = encoder.fit_transform(X_train, y_train)
        
        X_val_encoded = encoder.transform(X_val) if X_val is not None else None
        X_test_encoded = encoder.transform(X_test) if X_test is not None else None
    
    return X_train_encoded, X_val_encoded, X_test_encoded