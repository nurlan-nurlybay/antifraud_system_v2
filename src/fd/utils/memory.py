"""
Memory Optimization Utilities.

This module provides tools for aggressively reducing the memory footprint of 
Pandas DataFrames by downcasting numeric types. It is specifically designed 
to prevent Out-Of-Memory (OOM) errors on 16GB RAM machines when handling 
large datasets like IEEE-CIS Fraud Detection.
"""

import numpy as np
import pandas as pd
import gc
import structlog

logger = structlog.get_logger(__name__)

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Iterates through all columns of a DataFrame and downcasts numeric types.
    
    Note: 
        `float16` downcasting is explicitly disabled in this pipeline. While it saves 
        memory, Pandas does not support `float16` as an Index type, which causes 
        fatal NotImplementedErrors during groupby operations. We limit float 
        downcasting to `float32` for stability.

    Args:
        df (pd.DataFrame): The input DataFrame to optimize.
        verbose (bool): If True, logs the memory reduction statistics.

    Returns:
        pd.DataFrame: The memory-optimized DataFrame.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Check if column is numeric and not categorical
        if col_type != object and not isinstance(col_type, pd.CategoricalDtype):
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Handle Integer Downcasting
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            # Handle Float Downcasting
            else:
                # float16 explicitly omitted for Cython aggregation stability
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        reduction = 100 * (start_mem - end_mem) / start_mem
        logger.info(
            "memory_optimized", 
            start_mb=round(start_mem, 2), 
            end_mb=round(end_mem, 2), 
            reduction_pct=round(reduction, 1)
        )
        
    return df

def clear_memory() -> None:
    """
    Triggers manual Python garbage collection.
    Used heavily in data pipelines to clear orphaned DataFrame slices.
    """
    collected = gc.collect()
    logger.debug("garbage_collection_run", objects_collected=collected)
