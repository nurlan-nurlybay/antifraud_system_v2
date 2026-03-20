"""
Core Data Preprocessing Engine.

This module houses the KFold Target Encoder and the DataPreprocessor engine. 
It defines the distinct feature engineering paths (Trees, MLP, LSTM) required 
for the Hybrid Meta-Model Architecture, explicitly handling the 'Signal of Absence' 
(NaN imputation) depending on the target model's mathematical properties.
"""

import pandas as pd
import numpy as np
import structlog
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)

class KFoldTargetEncoder:
    """
    Computes Target Encoding using K-Fold out-of-fold averages to prevent data leakage.
    Applies Bayesian smoothing to prevent overfitting on low-frequency categories.
    """
    def __init__(self, cols: list[str], target_col: str, k: int = 5, m: int = 10):
        self.cols = cols
        self.target_col = target_col
        self.k = k
        self.m = m
        self.global_mean: float = 0.0
        self.category_mappings: dict[str, pd.Series] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits mappings on K-folds and returns the encoded training DataFrame.
        Includes safety casts to float32 to bypass Pandas Cython int8 agg bugs.
        """
        df_encoded = df.copy()
        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
        
        # Cast target to float32 locally to avoid Pandas int8 Cython agg bugs
        target_series = df[self.target_col].astype(np.float32)
        self.global_mean = float(target_series.mean())

        for col in self.cols:
            df_encoded[col + '_te'] = np.nan
            
            group_col_series = df[col]
            if str(group_col_series.dtype) == 'float16':
                group_col_series = group_col_series.astype(np.float32)

            # Create safe, temporary DataFrame for out-of-fold math
            temp_df = pd.DataFrame({col: group_col_series, self.target_col: target_series})

            for train_idx, val_idx in kf.split(temp_df):
                X_train, X_val = temp_df.iloc[train_idx], temp_df.iloc[val_idx]
                fold_prior = float(X_train[self.target_col].mean())
                
                # Bayesian Smoothing formula
                stats = X_train.groupby(col)[self.target_col].agg(['mean', 'count'])
                smoothed = (stats['count'] * stats['mean'] + self.m * fold_prior) / (stats['count'] + self.m)
                df_encoded.loc[df.index[val_idx], col + '_te'] = X_val[col].map(smoothed).fillna(fold_prior)

            # Save full training mapping for future transform() calls (Inference)
            full_stats = temp_df.groupby(col)[self.target_col].agg(['mean', 'count'])
            self.category_mappings[col] = (full_stats['count'] * full_stats['mean'] + self.m * self.global_mean) / (full_stats['count'] + self.m)
            
        return df_encoded

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies previously learned mapping dictionaries to Validation/Test sets."""
        df_encoded = df.copy()
        for col in self.cols:
            df_encoded[col + '_te'] = df[col].map(self.category_mappings[col]).fillna(self.global_mean)
        return df_encoded


class DataPreprocessor:
    """
    Modular preprocessing engine for IEEE-CIS data. 
    Produces three distinct datasets optimized for specific model archetypes.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=cfg['features']['v_features_pca_dims'])
        self.te_encoder = KFoldTargetEncoder(
            cols=cfg['features']['categorical']['columns'],
            target_col=cfg['data']['target_col'],
            k=cfg['features']['categorical']['target_encoding'].get('k', 5),
            m=cfg['features']['categorical']['target_encoding'].get('m', 10)
        )
        self.numerical_medians = {}
        self.nan_mask_columns = []

    # --- STAGE 1: THE CLEAN BASE ---

    def clean_base_data(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Universal cleaning phase. Handles UID string optimization and basic string imputation.
        Deliberately leaves numerical NaNs intact for downstream model-specific imputation.
        """
        df = df.copy()
        
        # Drop mathematical discards
        drop_cols = self.cfg['features'].get('drop_features', [])
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Create Uid (String -> Category -> Codes for extreme memory efficiency)
        uid_cols = self.cfg['features']['uid_columns']
        df['Uid'] = df[uid_cols].astype(str).agg('_'.join, axis=1).astype('category').cat.codes
        
        # Sort Chronologically (Crucial for time-series splits)
        df = df.sort_values(by=self.cfg['data']['time_col']).reset_index(drop=True)

        # Impute Strings
        cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != 'Uid']
        df[cat_cols] = df[cat_cols].fillna(self.cfg['features']['impute_categoricals'])
        
        return df

    # --- STAGE 2: MODEL-SPECIFIC TRANSFORMS ---

    def get_tree_features(self, df: pd.DataFrame, is_train: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Path A: Optimized for XGBoost/CatBoost.
        Fills numerical NaNs with -999 to create spatial splits in decision trees.
        No scaling or PCA is applied.
        """
        df = df.copy()
        df = self._apply_cyclical_time(df)
        df = self._apply_target_encoding(df, is_train)
        
        # Tree-specific Imputation
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                    if c not in ['TransactionID', self.cfg['data']['target_col'], 'TransactionDT', 'Uid']]
        df[num_cols] = df[num_cols].fillna(-999)
        
        y = df[self.cfg['data']['target_col']].values
        X = df.drop(columns=['Uid', 'TransactionID', 'TransactionDT', self.cfg['data']['target_col']], errors='ignore').values
        return X.astype(np.float32), y.astype(np.int8)

    def get_mlp_features(self, df: pd.DataFrame, is_train: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Path B: Optimized for Multi-Layer Perceptrons.
        Applies scaling, PCA, and Median imputation with Binary Missingness Masks.
        """
        df = self._transform_neural_base(df, is_train)
        y = df[self.cfg['data']['target_col']].values
        X = df.drop(columns=['Uid', 'TransactionID', 'TransactionDT', self.cfg['data']['target_col']], errors='ignore').values
        return X.astype(np.float32), y.astype(np.int8)

    def get_lstm_features(self, df: pd.DataFrame, is_train: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Path C: Optimized for Sequence Models (LSTMs).
        Groups records by UID and reshapes data into 3D tensors: [Batch, TimeStep, Feature].
        """
        df = self._transform_neural_base(df, is_train)
        seq_len = self.cfg['data']['sequence_length']
        target_col = self.cfg['data']['target_col']
        
        feature_cols = [c for c in df.columns if c not in ['Uid', 'TransactionID', 'TransactionDT', target_col]]
        
        X_3d, y_final = [], []
        
        # Group by integer UID to build chronological user history
        for _, group in df.groupby('Uid'):
            group_feats = group[feature_cols].values
            
            # Pad sequences if transaction history is shorter than seq_len
            if len(group_feats) < seq_len:
                padding = np.zeros((seq_len - len(group_feats), group_feats.shape[1]))
                seq = np.vstack([padding, group_feats])
            else:
                seq = group_feats[-seq_len:]
                
            X_3d.append(seq)
            y_final.append(group[target_col].iloc[-1])
            
        return np.array(X_3d, dtype=np.float32), np.array(y_final, dtype=np.int8)

    # --- INTERNAL HELPER METHODS ---

    def _transform_neural_base(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Shared logic pipeline for all gradient-based neural paths (MLP & LSTM).
        Safeguards the 'Signal of Absence' by generating _is_nan binary masks before imputation.
        """
        df = df.copy()
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                    if c not in ['TransactionID', self.cfg['data']['target_col'], 'TransactionDT', 'Uid']]
        
        if is_train:
            self.numerical_medians = df[num_cols].median().to_dict()
            self.nan_mask_columns = [c for c in num_cols if df[c].isnull().any()]
            
        # Append Binary Missingness Masks
        # for col in self.nan_mask_columns:
        #     df[col + '_is_nan'] = df[col].isnull().astype(np.float32)  # I used this but below is a better one

        new_masks = {}
        for col in self.nan_mask_columns:
            new_masks[col + '_is_nan'] = df[col].isnull().astype(np.float32)

        df = pd.concat([df, pd.DataFrame(new_masks, index=df.index)], axis=1)
            
        # Impute original columns with medians to stabilize Neural Net gradients
        for col in num_cols:
            df[col] = df[col].fillna(self.numerical_medians.get(col, 0))

        df = self._apply_cyclical_time(df)
        df = self._apply_target_encoding(df, is_train)
        
        if self.cfg['features']['log_transform_amount']:
            df[self.cfg['data']['amount_col']] = np.log1p(df[self.cfg['data']['amount_col']])
            
        df = self._apply_pca(df, is_train)
        df = self._apply_scaling(df, is_train)
        return df

    def _apply_cyclical_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms timestamp into 24-hour sinusoidal wave to capture daily fraud cycles."""
        df['hour'] = (df[self.cfg['data']['time_col']] // 3600) % 24
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype(np.float32)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype(np.float32)
        return df.drop(columns=['hour'])

    def _apply_target_encoding(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        if is_train:
            df = self.te_encoder.fit_transform(df)
        else:
            df = self.te_encoder.transform(df)
        return df.drop(columns=self.cfg['features']['categorical']['columns'])

    def _apply_pca(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Compresses highly collinear V-features into configurable Principal Components."""
        v_cols = [c for c in df.columns if c.startswith('V')]
        if is_train: self.pca.fit(df[v_cols])
        v_pca = self.pca.transform(df[v_cols])
        pca_cols = [f'V_pca_{i}' for i in range(self.cfg['features']['v_features_pca_dims'])]
        pca_df = pd.DataFrame(v_pca, columns=pca_cols, index=df.index, dtype=np.float32)
        return pd.concat([df.drop(columns=v_cols), pca_df], axis=1)

    def _apply_scaling(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Applies Standard Scaling to all numerical inputs, ignoring metadata columns."""
        cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                if c not in [self.cfg['data']['target_col'], 'TransactionID', 'TransactionDT', 'Uid']]
        if is_train: self.scaler.fit(df[cols])
        df[cols] = self.scaler.transform(df[cols]).astype(np.float32)
        return df
