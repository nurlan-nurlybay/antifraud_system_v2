"""
Test Data Preprocessing Pipeline.

This script executes the preprocessing lifecycle exclusively on the raw Kaggle 'test' datasets,
utilizing the previously saved 'preprocessor.joblib' state to avoid any data leakage.
"""

import pandas as pd
import numpy as np
import structlog
from pathlib import Path
from src.fd.utils.config import load_config
from src.fd.data.preprocessing import DataPreprocessor
from src.fd.utils.memory import reduce_mem_usage, clear_memory
from src.fd.utils.logging import setup_logger

setup_logger()
logger = structlog.get_logger(__name__)

def run_test_prep(cfg_path: str):
    """Processes Kaggle test datasets into model-ready NPZ files."""
    cfg = load_config(cfg_path)
    processed_dir = Path(cfg['paths']['processed_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("loading_preprocessor_state")
    pp = DataPreprocessor(cfg)
    pp.load_artifacts("models/preprocessors")
    
    logger.info("loading_raw_test_data")
    test_trans_path = Path("data/raw/test_transaction.csv")
    test_id_path = Path("data/raw/test_identity.csv")
    
    if not test_trans_path.exists():
        logger.error("Test data not found. Download it to data/raw/")
        return
        
    test_trans = reduce_mem_usage(pd.read_csv(test_trans_path))
    test_id = reduce_mem_usage(pd.read_csv(test_id_path))
    # Kaggle Anomaly Fix: test_identity uses hyphens (id-12) while train uses underscores (id_12)
    test_id.columns = test_id.columns.str.replace('-', '_')
    
    id_col = cfg['data']['id_col']
    
    df_clean = pd.merge(test_trans, test_id, on=id_col, how='left')
    df_clean = reduce_mem_usage(df_clean, verbose=False)
    del test_trans, test_id
    clear_memory()
    
    # Base Clean Phase
    df_clean = pp.clean_base_data(df_clean)
    
    # Phase 1: Export Tree Features
    logger.info("generating_test_features", type="tree")
    X_tree, _ = pp.get_tree_features(df_clean, is_train=False)
    np.save(processed_dir / "X_test_tree.npy", X_tree)
    del X_tree
    clear_memory()

    # Phase 2: Export MLP Features
    logger.info("generating_test_features", type="mlp")
    X_mlp, _ = pp.get_mlp_features(df_clean, is_train=False)
    np.save(processed_dir / "X_test_mlp.npy", X_mlp)
    del X_mlp
    clear_memory()

    # Phase 3: Export LSTM Sequences
    logger.info("generating_test_features", type="lstm")
    X_lstm, _ = pp.get_lstm_features(df_clean, is_train=False)
    np.save(processed_dir / "X_test_lstm.npy", X_lstm)
    del X_lstm, df_clean
    clear_memory()

    logger.info("test_preprocessing_completed", status="success")

if __name__ == "__main__":
    run_test_prep("configs/data_prep.yaml")
