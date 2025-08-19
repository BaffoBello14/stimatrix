"""
Sistema avanzato di Feature Importance per analisi interpretabilità modelli ML.

Questo modulo combina metodi multipli per calcolare feature importance robusta:
- Built-in importance: feature_importances_, coef_, ensemble averaging
- Permutation importance: valutazione basata su performance degradation
- SHAP values: spiegabilità locale e globale con explainer ottimizzati
- Consensus importance: combinazione weighted di tutti i metodi
- Stability analysis: consistenza importance tra modelli

Il sistema gestisce automaticamente diversi tipi di modelli (lineari, tree-based,
ensemble) e ottimizza le performance con campionamento intelligente per SHAP.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from utils.logger import get_logger

logger = get_logger(__name__)

# Gestione import SHAP con fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP non disponibile - feature importance limitata a metodi built-in e permutation")


class AdvancedFeatureImportance:
    """Sistema avanzato per calcolo feature importance con metodi multipli."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza sistema feature importance.
        
        Args:
            config: Configurazione feature importance
        """
        self.config = config.get('feature_importance', {})
        self.shap_config = config.get('training', {}).get('shap', {})
        self.enable_shap = self.shap_config.get('enabled', True) and SHAP_AVAILABLE
        self.enable_permutation = self.config.get('enable_permutation', True)
        self.sample_size = self.shap_config.get('sample_size', 500)
        self.max_display = self.shap_config.get('max_display', 20)
        
        logger.info(f"Feature Importance configurato - SHAP: {self.enable_shap}, "
                   f"Permutation: {self.enable_permutation}, Sample: {self.sample_size}")
    
    def calculate_comprehensive_importance(
        self,
        models: Dict[str, Any],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calcola feature importance con metodi multipli per tutti i modelli.
        
        Args:
            models: Dict con modelli addestrati
            X_train: Features training (per SHAP background)
            X_test: Features test (per valutazione)
            y_test: Target test (per permutation importance)
            feature_names: Nomi features (default: X_train.columns)
            
        Returns:
            Dict con risultati comprehensive importance
        """
        logger.info("=== CALCOLO COMPREHENSIVE FEATURE IMPORTANCE ===")
        
        if feature_names is None:
            feature_names = X_train.columns.tolist()
        
        comprehensive_results = {
            'models': {},
            'consensus': {},
            'rankings': {},
            'method_availability': {
                'builtin': True,
                'permutation': self.enable_permutation,
                'shap': self.enable_shap
            }
        }
        
        # Calcola importance per ogni modello
        for model_name, model_data in models.items():
            logger.info(f"🤖 Calcolo importance per {model_name}...")
            
            model = model_data['model']
            model_type = model_data.get('type', 'unknown')
            
            model_importance = self._calculate_model_importance(
                model, model_name, model_type, X_train, X_test, y_test, feature_names
            )
            
            comprehensive_results['models'][model_name] = model_importance
        
        # Calcola consensus tra modelli e metodi
        consensus_results = self._calculate_consensus_importance(
            comprehensive_results['models'], feature_names
        )
        comprehensive_results['consensus'] = consensus_results
        
        # Crea rankings
        rankings = self._create_importance_rankings(comprehensive_results['models'], feature_names)
        comprehensive_results['rankings'] = rankings
        
        logger.info(f"✅ Feature importance calcolata per {len(models)} modelli")
        return comprehensive_results
    
    def _calculate_model_importance(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Calcola importance per singolo modello con metodi multipli."""
        model_results = {
            'builtin': None,
            'permutation': None,
            'shap': None,
            'methods_used': [],
            'errors': []
        }
        
        # 1. Built-in Feature Importance
        try:
            builtin_importance = self._get_builtin_importance(model, model_name, feature_names)
            if builtin_importance is not None:
                model_results['builtin'] = builtin_importance
                model_results['methods_used'].append('builtin')
        except Exception as e:
            model_results['errors'].append(f"Built-in importance error: {str(e)}")
            logger.warning(f"Built-in importance fallita per {model_name}: {e}")
        
        # 2. Permutation Importance
        if self.enable_permutation:
            try:
                perm_importance = self._get_permutation_importance(
                    model, X_test, y_test, feature_names
                )
                model_results['permutation'] = perm_importance
                model_results['methods_used'].append('permutation')
            except Exception as e:
                model_results['errors'].append(f"Permutation importance error: {str(e)}")
                logger.warning(f"Permutation importance fallita per {model_name}: {e}")
        
        # 3. SHAP Importance
        if self.enable_shap:
            try:
                shap_importance = self._get_shap_importance_optimized(
                    model, model_name, model_type, X_train, X_test, feature_names
                )
                model_results['shap'] = shap_importance
                model_results['methods_used'].append('shap')
            except Exception as e:
                model_results['errors'].append(f"SHAP importance error: {str(e)}")
                logger.warning(f"SHAP importance fallita per {model_name}: {e}")
        
        logger.info(f"  {model_name}: {len(model_results['methods_used'])}/3 metodi completati")
        return model_results
    
    def _get_builtin_importance(
        self, 
        model: Any, 
        model_name: str, 
        feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """Estrae built-in feature importance dal modello."""
        importance = None
        method = "unknown"
        
        # Tree-based models
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            method = "feature_importances_"
        
        # Linear models
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            method = "abs(coef_)"
        
        # Ensemble models - gestione speciale
        elif 'ensemble' in model_name.lower() or 'voting' in model_name.lower():
            importance = self._get_ensemble_importance_fast(model, feature_names, model_name)
            method = "ensemble_fast"
        
        # CatBoost
        elif hasattr(model, 'get_feature_importance'):
            try:
                importance = model.get_feature_importance()
                method = "catboost_importance"
            except:
                importance = None
        
        if importance is not None:
            # Normalizza
            importance_norm = importance / importance.sum() if importance.sum() > 0 else importance
            
            return {
                'values': dict(zip(feature_names, importance_norm)),
                'method': method,
                'raw_values': dict(zip(feature_names, importance))
            }
        
        return None
    
    def _get_permutation_importance(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
        n_repeats: int = 5
    ) -> Dict[str, float]:
        """Calcola permutation importance."""
        # Campiona per velocità se dataset grande
        if len(X_test) > 1000:
            sample_size = min(1000, len(X_test))
            sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test.iloc[sample_idx]
            y_sample = y_test.iloc[sample_idx]
        else:
            X_sample = X_test
            y_sample = y_test
        
        # Calcola permutation importance
        perm_result = permutation_importance(
            model, X_sample, y_sample, 
            n_repeats=n_repeats, 
            random_state=42,
            scoring='neg_mean_squared_error'
        )
        
        # Normalizza (importance negative diventa positive)
        importance_values = np.abs(perm_result.importances_mean)
        importance_norm = importance_values / importance_values.sum() if importance_values.sum() > 0 else importance_values
        
        return {
            'values': dict(zip(feature_names, importance_norm)),
            'method': 'permutation',
            'std_values': dict(zip(feature_names, perm_result.importances_std)),
            'raw_values': dict(zip(feature_names, importance_values))
        }
    
    def _get_shap_importance_optimized(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Calcola SHAP importance con ottimizzazioni per velocità."""
        if not SHAP_AVAILABLE:
            return None
        
        # Campionamento aggressivo per ensemble models
        if 'ensemble' in model_name.lower() or 'stacking' in model_name.lower():
            background_size = min(50, len(X_train))
            test_size = min(20, len(X_test))
        else:
            background_size = min(self.sample_size, len(X_train))
            test_size = min(self.sample_size // 2, len(X_test))
        
        logger.info(f"SHAP per {model_name}: background={background_size}, test={test_size}")
        
        # Campiona dati
        X_background = X_train.sample(n=background_size, random_state=42)
        X_test_sample = X_test.sample(n=test_size, random_state=42)
        
        try:
            # Crea explainer appropriato
            explainer = self._create_shap_explainer(model, model_type, X_background, feature_names)
            
            # Calcola SHAP values
            shap_values = explainer(X_test_sample)
            
            # Estrai importance (media absolute SHAP values)
            if hasattr(shap_values, 'values'):
                importance_values = np.abs(shap_values.values).mean(axis=0)
            else:
                importance_values = np.abs(shap_values).mean(axis=0)
            
            # Normalizza
            importance_norm = importance_values / importance_values.sum() if importance_values.sum() > 0 else importance_values
            
            return {
                'values': dict(zip(feature_names, importance_norm)),
                'method': f'shap_{explainer.__class__.__name__.lower()}',
                'raw_values': dict(zip(feature_names, importance_values)),
                'shap_values': shap_values,
                'background_size': background_size,
                'test_size': test_size
            }
            
        except Exception as e:
            logger.warning(f"SHAP fallito per {model_name}: {e}")
            return None
    
    def _create_shap_explainer(self, model: Any, model_type: str, X_background: pd.DataFrame, feature_names: List[str]):
        """Crea explainer SHAP appropriato per tipo di modello."""
        # Crea wrapper per gestire feature names
        def model_predict(X):
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X, columns=feature_names)
            else:
                X_df = X
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                return model.predict(X_df)
        
        # Scegli explainer basato sul tipo di modello
        if 'tree' in model_type.lower() or hasattr(model, 'feature_importances_'):
            # Tree explainer per modelli ad albero
            try:
                return shap.TreeExplainer(model)
            except:
                # Fallback a Explainer generico
                return shap.Explainer(model_predict, X_background)
        
        elif 'linear' in model_type.lower() or hasattr(model, 'coef_'):
            # Linear explainer per modelli lineari
            try:
                return shap.LinearExplainer(model, X_background)
            except:
                return shap.Explainer(model_predict, X_background)
        
        else:
            # Explainer generico per altri modelli
            return shap.Explainer(model_predict, X_background)
    
    def _get_ensemble_importance_fast(
        self, 
        model: Any, 
        feature_names: List[str], 
        model_name: str
    ) -> Optional[np.ndarray]:
        """Estrae importance da ensemble models con metodo veloce."""
        try:
            # Voting Regressor
            if hasattr(model, 'estimators_'):
                importances = []
                weights = getattr(model, 'weights', None)
                
                for i, estimator in enumerate(model.estimators_):
                    if hasattr(estimator, 'feature_importances_'):
                        weight = weights[i] if weights is not None else 1.0
                        importances.append(estimator.feature_importances_ * weight)
                    elif hasattr(estimator, 'coef_'):
                        weight = weights[i] if weights is not None else 1.0
                        importances.append(np.abs(estimator.coef_) * weight)
                
                if importances:
                    return np.mean(importances, axis=0)
            
            # Stacking Regressor
            elif hasattr(model, 'final_estimator_'):
                # Usa importance del final estimator se disponibile
                if hasattr(model.final_estimator_, 'feature_importances_'):
                    return model.final_estimator_.feature_importances_
                elif hasattr(model.final_estimator_, 'coef_'):
                    return np.abs(model.final_estimator_.coef_)
            
            return None
            
        except Exception as e:
            logger.warning(f"Ensemble importance fallita per {model_name}: {e}")
            return None
    
    def _calculate_consensus_importance(
        self,
        models_importance: Dict[str, Dict[str, Any]],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Calcola consensus importance tra modelli e metodi."""
        logger.info("Calcolo consensus feature importance...")
        
        consensus = {
            'by_method': {},
            'by_model': {},
            'overall_consensus': {},
            'stability_metrics': {}
        }
        
        # Raccogli importance per metodo
        methods = ['builtin', 'permutation', 'shap']
        
        for method in methods:
            method_importances = []
            
            for model_name, model_results in models_importance.items():
                if method in model_results and model_results[method] is not None:
                    importance_dict = model_results[method]['values']
                    importance_array = np.array([importance_dict.get(feat, 0) for feat in feature_names])
                    method_importances.append(importance_array)
            
            if method_importances:
                # Media e stabilità per metodo
                method_mean = np.mean(method_importances, axis=0)
                method_std = np.std(method_importances, axis=0)
                
                consensus['by_method'][method] = {
                    'values': dict(zip(feature_names, method_mean)),
                    'std': dict(zip(feature_names, method_std)),
                    'models_count': len(method_importances)
                }
        
        # Consensus overall (media tra metodi disponibili)
        if consensus['by_method']:
            overall_importance = np.zeros(len(feature_names))
            methods_count = 0
            
            for method, method_data in consensus['by_method'].items():
                method_values = np.array([method_data['values'][feat] for feat in feature_names])
                overall_importance += method_values
                methods_count += 1
            
            if methods_count > 0:
                overall_importance /= methods_count
                consensus['overall_consensus'] = dict(zip(feature_names, overall_importance))
        
        # Metriche di stabilità
        stability_metrics = self._calculate_stability_metrics(models_importance, feature_names)
        consensus['stability_metrics'] = stability_metrics
        
        return consensus
    
    def _calculate_stability_metrics(
        self,
        models_importance: Dict[str, Dict[str, Any]],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Calcola metriche di stabilità importance tra modelli."""
        stability = {
            'top_features_consistency': {},
            'ranking_stability': {},
            'variance_metrics': {}
        }
        
        # Raccogli tutte le importance disponibili
        all_importances = []
        model_names = []
        
        for model_name, model_results in models_importance.items():
            # Prova prima built-in, poi permutation, poi SHAP
            for method in ['builtin', 'permutation', 'shap']:
                if method in model_results and model_results[method] is not None:
                    importance_dict = model_results[method]['values']
                    importance_array = np.array([importance_dict.get(feat, 0) for feat in feature_names])
                    all_importances.append(importance_array)
                    model_names.append(f"{model_name}_{method}")
                    break
        
        if len(all_importances) >= 2:
            all_importances = np.array(all_importances)
            
            # Consistenza top features
            for top_k in [5, 10, 20]:
                if top_k <= len(feature_names):
                    top_features_lists = []
                    for importance in all_importances:
                        top_indices = np.argsort(importance)[-top_k:]
                        top_features = [feature_names[i] for i in top_indices]
                        top_features_lists.append(set(top_features))
                    
                    # Intersezione tra tutti i modelli
                    common_features = set.intersection(*top_features_lists) if top_features_lists else set()
                    consistency_score = len(common_features) / top_k
                    
                    stability['top_features_consistency'][f'top_{top_k}'] = {
                        'consistency_score': consistency_score,
                        'common_features': list(common_features)
                    }
            
            # Varianza importance per feature
            importance_variance = np.var(all_importances, axis=0)
            importance_mean = np.mean(all_importances, axis=0)
            cv = importance_variance / (importance_mean + 1e-8)  # Coefficient of variation
            
            stability['variance_metrics'] = {
                'mean_cv': np.mean(cv),
                'max_cv': np.max(cv),
                'high_variance_features': [
                    feature_names[i] for i, cv_val in enumerate(cv) if cv_val > 1.0
                ]
            }
        
        return stability
    
    def _create_importance_rankings(
        self,
        models_importance: Dict[str, Dict[str, Any]],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Crea rankings feature importance."""
        rankings = {
            'by_model': {},
            'consensus_ranking': [],
            'method_rankings': {}
        }
        
        # Rankings per modello
        for model_name, model_results in models_importance.items():
            model_rankings = {}
            
            for method in ['builtin', 'permutation', 'shap']:
                if method in model_results and model_results[method] is not None:
                    importance_dict = model_results[method]['values']
                    sorted_features = sorted(
                        importance_dict.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    model_rankings[method] = [feat for feat, _ in sorted_features]
            
            rankings['by_model'][model_name] = model_rankings
        
        return rankings
    
    def save_importance_plots(
        self,
        importance_results: Dict[str, Any],
        output_dir: str,
        top_n: int = 20
    ) -> None:
        """Salva plot feature importance."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot consensus importance
        if 'overall_consensus' in importance_results['consensus']:
            consensus_imp = importance_results['consensus']['overall_consensus']
            
            # Ordina per importance
            sorted_features = sorted(consensus_imp.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:top_n]
            
            # Crea plot
            features, importances = zip(*top_features)
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Consensus Feature Importance')
            plt.title(f'Top {top_n} Features - Consensus Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/consensus_feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Salvato plot consensus importance: {output_dir}/consensus_feature_importance.png")
        
        # Plot per metodo
        for method, method_data in importance_results['consensus'].get('by_method', {}).items():
            method_imp = method_data['values']
            sorted_features = sorted(method_imp.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:top_n]
            
            features, importances = zip(*top_features)
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel(f'{method.title()} Feature Importance')
            plt.title(f'Top {top_n} Features - {method.title()} Method')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{method}_feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Salvato plot {method} importance: {output_dir}/{method}_feature_importance.png")