"""
Unified Tree Model Wrappers for Antifraud System v2.0.
"""

import numpy as np
import torch
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
import structlog
from typing import Any

logger = structlog.get_logger(__name__)

class TreeWrapper:
    def __init__(self, model_type: str, params: dict):
        self.model_type = model_type.lower()
        self.params = params.copy()
        
        self.params.pop('pred_leaf', None)
        self.params.pop('return_leaves', None)

        # Auto-detect GPU and inject hardware acceleration parameters
        if torch.cuda.is_available():
            if self.model_type == "xgboost":
                self.params.setdefault('tree_method', 'hist')
                self.params.setdefault('device', 'cuda')
            elif self.model_type == "lightgbm":
                self.params.setdefault('device', 'gpu')
            elif self.model_type == "catboost":
                self.params.setdefault('task_type', 'GPU')

        # Tell the linter that self.model is dynamic, so it stops complaining
        self.model: Any = None 

        if self.model_type == "xgboost":
            self.params.setdefault('verbosity', 0)
            self.params.setdefault('n_jobs', -1)
            self.model = xgb.XGBClassifier(**self.params, enable_categorical=False)
        elif self.model_type == "lightgbm":
            self.params.setdefault('verbose', -1)
            self.params.setdefault('n_jobs', -1)
            self.model = lgb.LGBMClassifier(**self.params)
        elif self.model_type == "catboost":
            self.params.setdefault('allow_writing_files', False)
            self.params.setdefault('thread_count', -1)
            self.model = cb.CatBoostClassifier(**self.params, verbose=0)
        elif self.model_type == "random_forest":
            self.params.setdefault('n_jobs', 32)
            self.model = RandomForestClassifier(**self.params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None, early_stopping_rounds: int = 50):
        """Fits the model. If an eval set is provided, enables Early Stopping for supported tree models."""
        if X_val is not None and y_val is not None:
            if self.model_type == "xgboost":
                self.model.set_params(early_stopping_rounds=early_stopping_rounds)
                self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
            elif self.model_type == "lightgbm":
                self.model.fit(
                    X, y, eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)]
                )
            elif self.model_type == "catboost":
                self.model.fit(X, y, eval_set=(X_val, y_val), early_stopping_rounds=early_stopping_rounds)
            else:
                # RandomForest has no native early stopping
                self.model.fit(X, y)
        else:
            self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Cast to numpy array to satisfy strict type checkers
        return np.array(self.model.predict_proba(X))[:, 1]

    def get_leaves(self, X: np.ndarray) -> np.ndarray:
        leaves = None
        
        if self.model_type in ["xgboost", "random_forest"]:
            leaves = self.model.apply(X)
        elif self.model_type == "lightgbm":
            leaves = self.model.predict(X, pred_leaf=True)
        elif self.model_type == "catboost":
            leaves = self.model.calc_leaf_indexes(X)
        else:
            raise ValueError("Cannot extract leaves for this model type.")
            
        return np.array(leaves, dtype=np.int32)
    
    def get_raw_model(self):
        return self.model
