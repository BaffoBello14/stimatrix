#!/usr/bin/env python3
"""
Main entry point per la pipeline ottimizzata di preprocessing e training.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List
import json

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config
from src.utils.logger import get_logger, setup_logging
from src.preprocessing.pipeline import run_preprocessing
from src.training.train_optimized import run_training  # Usa la versione ottimizzata
from src.training.evaluation import run_evaluation
from src.db.schema_extract import save_schema_from_database
from src.dataset_builder.retrieval import retrieve_and_save_data

logger = get_logger(__name__)


def main(config_path: Optional[str] = None, steps: Optional[List[str]] = None):
    """
    Esegue la pipeline completa o steps specifici.
    
    Args:
        config_path: Path al file di configurazione (default: config/config_optimized.yaml)
        steps: Lista di steps da eseguire. Se None, usa config. 
               Opzioni: ['schema', 'dataset', 'preprocessing', 'training', 'evaluation'] o ['all']
    """
    # Load configuration - usa il config ottimizzato di default
    config_path = config_path or "config/config_optimized.yaml"
    config = load_config(config_path)
    
    # Setup logging
    log_cfg = config.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file", "logs/pipeline.log"),
        console=log_cfg.get("console", True)
    )
    
    logger.info(f"Configuration loaded from: {config_path}")
    logger.info("Using OPTIMIZED pipeline with improved hyperparameter tuning")
    
    # Determine steps to execute
    if steps is None:
        steps = config.get("execution", {}).get("steps", ["all"])
    
    if "all" in steps:
        steps = ["schema", "dataset", "preprocessing", "training", "evaluation"]
    
    logger.info(f"Steps to execute: {steps}")
    
    # Path configuration
    paths = config.get("paths", {})
    raw_dir = Path(paths.get("raw_data", "data/raw"))
    preprocessed_dir = Path(paths.get("preprocessed_data", "data/preprocessed"))
    models_dir = Path(paths.get("models_dir", "models"))
    
    # Check force_reload
    force_reload = config.get("execution", {}).get("force_reload", False)
    
    # Execute steps
    try:
        # Step 1: Extract schema
        if "schema" in steps:
            logger.info("="*50)
            logger.info("STEP 1: Extracting database schema")
            logger.info("="*50)
            schema_path = Path(paths.get("schema", "schema/db_schema.json"))
            
            if force_reload or not schema_path.exists():
                save_schema_from_database(output_path=str(schema_path))
                logger.info(f"Schema saved to: {schema_path}")
            else:
                logger.info(f"Schema already exists at {schema_path}, skipping extraction")
        
        # Step 2: Retrieve dataset
        if "dataset" in steps:
            logger.info("="*50)
            logger.info("STEP 2: Retrieving dataset")
            logger.info("="*50)
            raw_file = raw_dir / paths.get("raw_filename", "raw.parquet")
            
            if force_reload or not raw_file.exists():
                db_config = config.get("database", {})
                retrieve_and_save_data(
                    output_path=str(raw_file),
                    schema_name=db_config.get("schema_name"),
                    selected_aliases=db_config.get("selected_aliases", []),
                    use_poi=db_config.get("use_poi", True),
                    use_ztl=db_config.get("use_ztl", True)
                )
                logger.info(f"Dataset saved to: {raw_file}")
            else:
                logger.info(f"Dataset already exists at {raw_file}, skipping retrieval")
        
        # Step 3: Preprocessing
        if "preprocessing" in steps:
            logger.info("="*50)
            logger.info("STEP 3: Running preprocessing pipeline")
            logger.info("="*50)
            
            # Check if preprocessed data exists
            prep_files_exist = any(preprocessed_dir.glob("X_*.parquet"))
            
            if force_reload or not prep_files_exist:
                results = run_preprocessing(config)
                logger.info("Preprocessing completed successfully")
                
                # Log results summary
                if results:
                    logger.info(f"Number of samples: {results.get('n_samples', 'N/A')}")
                    logger.info(f"Number of features: {results.get('n_features', 'N/A')}")
                    logger.info(f"Profiles created: {results.get('profiles', [])}")
            else:
                logger.info("Preprocessed data already exists, skipping preprocessing")
        
        # Step 4: Training with optimization
        if "training" in steps:
            logger.info("="*50)
            logger.info("STEP 4: Training models with OPTIMIZED hyperparameter tuning")
            logger.info("="*50)
            
            # Check if models exist
            models_exist = any(models_dir.glob("*.joblib"))
            
            if force_reload or not models_exist:
                training_results = run_training(config)
                logger.info("Training completed successfully")
                
                # Log best models
                if training_results and "comparison" in training_results:
                    logger.info("\nTop 5 models by RMSE:")
                    for i, model in enumerate(training_results["comparison"][:5]):
                        logger.info(f"{i+1}. {model['Model']}: R²={model['Test_R2']:.4f}, RMSE={model['Test_RMSE']:.2f}")
                        
                # Save training results
                results_path = models_dir / "training_results_optimized.json"
                with open(results_path, 'w') as f:
                    # Convert DataFrame data to serializable format
                    serializable_results = {
                        k: v for k, v in training_results.items() 
                        if k not in ['comparison']  # Skip DataFrame
                    }
                    if 'comparison' in training_results:
                        serializable_results['top_models'] = training_results['comparison'][:10]
                    
                    json.dump(serializable_results, f, indent=2, default=str)
                logger.info(f"Training results saved to: {results_path}")
            else:
                logger.info("Models already exist, skipping training")
        
        # Step 5: Evaluation
        if "evaluation" in steps:
            logger.info("="*50)
            logger.info("STEP 5: Running model evaluation")
            logger.info("="*50)
            
            eval_results = run_evaluation(config)
            logger.info("Evaluation completed successfully")
            
            # Save evaluation results
            if eval_results:
                eval_path = models_dir / "evaluation_results_optimized.json"
                with open(eval_path, 'w') as f:
                    json.dump(eval_results, f, indent=2, default=str)
                logger.info(f"Evaluation results saved to: {eval_path}")
        
        logger.info("="*50)
        logger.info("Pipeline completed successfully!")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run optimized ML pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config_optimized.yaml",
        help="Path to configuration file (default: config/config_optimized.yaml)"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["schema", "dataset", "preprocessing", "training", "evaluation", "all"],
        help="Steps to execute (default: use config)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-execution of all steps"
    )
    
    args = parser.parse_args()
    
    # Update config if force is specified
    if args.force:
        config = load_config(args.config)
        config.setdefault("execution", {})["force_reload"] = True
        # Temporarily save updated config
        import tempfile
        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp_path = tmp.name
        
        main(config_path=tmp_path, steps=args.steps)
        
        # Clean up
        Path(tmp_path).unlink()
    else:
        main(config_path=args.config, steps=args.steps)