"""
Data Preparation Orchestrator Pipeline.

This script executes the complete data preprocessing lifecycle. It loads raw CSVs, 
determines strict chronological validation splits (L0, L1_Train, L1_Val), and safely 
processes the chunks via the `DataPreprocessor`. 

It utilizes chunk-based merging and aggressive manual garbage collection to process 
the IEEE-CIS dataset entirely within a 16GB RAM constraint.
"""

import pandas as pd
import numpy as np
import structlog
from pathlib import Path
from src.fd.utils.config import load_config
from src.fd.data.preprocessing import DataPreprocessor
from src.fd.utils.memory import reduce_mem_usage, clear_memory
from src.fd.utils.logging import setup_logger

# Initialize global structured logger configuration
setup_logger()
logger = structlog.get_logger(__name__)

def run_data_prep(cfg_path: str):
    """
    Main execution logic for the data preprocessing pipeline.
    
    Args:
        cfg_path (str): Relative path to the data_prep YAML configuration file.
    """
    cfg = load_config(cfg_path)
    processed_dir = Path(cfg['paths']['processed_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("pipeline_started", version=cfg['metadata']['project_version'])

    # 1. Load & Downcast (Memory Optimization Phase)
    logger.info("loading_raw_data")
    train_trans = reduce_mem_usage(pd.read_csv(cfg['paths']['train_transaction']))
    train_id = reduce_mem_usage(pd.read_csv(cfg['paths']['train_identity']))

    # 2. Determine Temporal Splits (Wall of Time integrity)
    val_p = cfg['data']['val_size']
    test_p = cfg['data']['test_size']
    dev_p = 1.0 - val_p - test_p

    # Sort solely by TransactionDT to calculate chronological indices
    temp_df = train_trans[['TransactionID', cfg['data']['time_col']]].sort_values(cfg['data']['time_col'])
    n = len(temp_df)
    dev_idx = int(n * dev_p)
    meta_train_idx = int(n * (dev_p + val_p))

    ids_map = {
        "dev": temp_df.iloc[:dev_idx]['TransactionID'].values,
        "meta_train": temp_df.iloc[dev_idx:meta_train_idx]['TransactionID'].values,
        "meta_val": temp_df.iloc[meta_train_idx:]['TransactionID'].values
    }
    
    del temp_df
    clear_memory()

    logger.info("data_split_calculated", 
                dev_size=len(ids_map['dev']), 
                meta_train_size=len(ids_map['meta_train']), 
                meta_val_size=len(ids_map['meta_val']))

    pp = DataPreprocessor(cfg)

    # 3. Process Splits Iteratively (Memory Constraints Check)
    for name, ids in ids_map.items():
        is_train = (name == "dev")
        logger.info("processing_split", split=name, is_train=is_train)
        
        # Merge locally to save RAM. Immediately delete objects not required.
        df_clean = pd.merge(train_trans[train_trans['TransactionID'].isin(ids)], 
                            train_id[train_id['TransactionID'].isin(ids)], 
                            on='TransactionID', how='left')
        df_clean = reduce_mem_usage(df_clean, verbose=False)
        clear_memory()
        
        # Base Clean Phase (Creates Uids, handles Categorical strings)
        df_clean = pp.clean_base_data(df_clean, is_train=is_train)
        
        # Phase 1: Export Tree Features
        logger.info("generating_features", split=name, type="tree")
        X_tree, y_tree = pp.get_tree_features(df_clean, is_train=is_train)
        np.savez_compressed(processed_dir / f"X_y_{name}_tree.npz", X=X_tree, y=y_tree)
        del X_tree, y_tree
        clear_memory()

        # Phase 2: Export MLP Features
        logger.info("generating_features", split=name, type="mlp")
        X_mlp, y_mlp = pp.get_mlp_features(df_clean, is_train=is_train)
        np.savez_compressed(processed_dir / f"X_y_{name}_mlp.npz", X=X_mlp, y=y_mlp)
        del X_mlp, y_mlp
        clear_memory()

        # Phase 3: Export LSTM Sequences
        logger.info("generating_features", split=name, type="lstm")
        X_lstm, y_lstm = pp.get_lstm_features(df_clean, is_train=is_train)
        np.savez_compressed(processed_dir / f"X_y_{name}_lstm.npz", X=X_lstm, y=y_lstm)
        del X_lstm, y_lstm, df_clean
        clear_memory()

    logger.info("pipeline_completed", status="success")

if __name__ == "__main__":
    run_data_prep("configs/data_prep.yaml")
