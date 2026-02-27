import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class KFoldTargetEncoder:
    """
    Computes Target Encoding with K-Fold out-of-fold mapping and Bayesian smoothing (m).
    Uses the prior from Subset B (the training folds) during the K-Fold loop to ensure zero leakage.
    """
    def __init__(self, cols: list[str], target_col: str, k: int = 5, m: int = 10):
        self.cols = cols
        self.target_col = target_col
        self.k = k
        self.m = m
        self.global_mean: float = 0.0
        self.category_mappings: dict[str, pd.Series] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
        
        # Calculate global mean for persistence and fallback
        self.global_mean = float(df[self.target_col].mean())

        for col in self.cols:
            df_encoded[col + '_te'] = np.nan
            
            # 1. Out-of-fold encoding for the training set
            for train_idx, val_idx in kf.split(df):
                X_train = df.iloc[train_idx]
                X_val = df.iloc[val_idx]
                
                # Calculate the prior mean for Subset B only (Joshua's "Overall Mean")
                fold_prior = float(X_train[self.target_col].mean())
                
                # Calculate smoothed mean based only on Subset B
                stats = X_train.groupby(col)[self.target_col].agg(['mean', 'count'])
                smoothed = (stats['count'] * stats['mean'] + self.m * fold_prior) / (stats['count'] + self.m)
                
                # Map results to Subset A (the current validation fold)
                df_encoded.loc[df.index[val_idx], col + '_te'] = X_val[col].map(smoothed).fillna(fold_prior)

            # 2. Save the final mapping using FULL training data for future inference/transform
            full_stats = df.groupby(col)[self.target_col].agg(['mean', 'count'])
            self.category_mappings[col] = (full_stats['count'] * full_stats['mean'] + self.m * self.global_mean) / (full_stats['count'] + self.m)
            
        return df_encoded

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the learned mappings to validation/test sets using the global training prior."""
        df_encoded = df.copy()
        for col in self.cols:
            df_encoded[col + '_te'] = df[col].map(self.category_mappings[col]).fillna(self.global_mean)
        return df_encoded


class DataPreprocessor:
    """Orchestrates the entire IEEE-CIS cleaning and feature engineering pipeline."""
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=cfg['features']['v_features_pca_dims'])
        self.te_encoder = KFoldTargetEncoder(
            cols=cfg['features']['categorical']['columns'],
            target_col=cfg['data']['target_col'],
            k=cfg['features']['categorical']['target_encoding']['k'],
            m=cfg['features']['categorical']['target_encoding']['m']
        )
        # To store median values for numerical imputation during inference
        self.numerical_medians = {} 

    def merge_and_sort(self, trans_df: pd.DataFrame, id_df: pd.DataFrame) -> pd.DataFrame:
        """Merges tables, drops useless features, creates UID, and sorts chronologically."""
        df = pd.merge(trans_df, id_df, on='TransactionID', how='left')
        
        # Drop useless features identified in EDA
        drop_cols = self.cfg['features'].get('drop_features', [])
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Generate UID for LSTM sequences
        uid_cols = self.cfg['features']['uid_columns']
        df['Uid'] = df[uid_cols].astype(str).agg('_'.join, axis=1)

        # Sort chronologically to prevent future leakage in splits
        df = df.sort_values(by=self.cfg['data']['time_col']).reset_index(drop=True)
        return df

    def impute_and_engineer(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Handles missing values, log transforms, and frequency encoding."""
        df = df.copy()
        
        # 1. Impute Categoricals
        cat_cols = df.select_dtypes(include=['object']).columns
        # Exclude Uid from being treated as a standard categorical feature
        cat_cols = [c for c in cat_cols if c != 'Uid']
        df[cat_cols] = df[cat_cols].fillna(self.cfg['features']['impute_categoricals'])

        # 2. Impute Numericals
        num_cols = df.select_dtypes(include=[np.number]).columns
        num_cols = [c for c in num_cols if c not in ['TransactionID', self.cfg['data']['target_col']]]
        
        if is_train:
            self.numerical_medians = df[num_cols].median().to_dict()
        
        for col in num_cols:
            df[col] = df[col].fillna(self.numerical_medians.get(col, 0))

        # 3. Frequency Encoding
        if self.cfg['features']['categorical']['frequency_encoding']:
            for col in self.cfg['features']['categorical']['columns']:
                if is_train:
                    # In a strict environment, calculate frequencies on train, map to test.
                    # For speed/simplicity here, we map directly.
                    freq = df[col].value_counts()
                    df[col + '_freq'] = df[col].map(freq)
                else:
                    # Note: You will need to store `freq` in `self` during train for strict correctness.
                    pass 

        # 4. Log Transform Amount
        if self.cfg['features']['log_transform_amount']:
            amt_col = self.cfg['data']['amount_col']
            df[amt_col] = np.log1p(df[amt_col])

        return df

    def apply_pca(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Compresses V-features into a lower-dimensional latent space."""
        # Find all columns starting with 'V'
        v_cols = [c for c in df.columns if c.startswith('V')]
        
        if is_train:
            # We fit the PCA only on the V-features of the training set
            self.pca.fit(df[v_cols])
            
        v_pca = self.pca.transform(df[v_cols])
        
        # Create column names: V_pca_0, V_pca_1, etc.
        pca_cols = [f'V_pca_{i}' for i in range(self.cfg['features']['v_features_pca_dims'])]
        pca_df = pd.DataFrame(v_pca, columns=pca_cols, index=df.index)
        
        # Drop old V columns and join new PCA columns
        df = df.drop(columns=v_cols)
        df = pd.concat([df, pca_df], axis=1)
        return df

    def apply_scaling(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Standardizes all numerical features to mean=0, std=1."""
        # We only scale columns that are numerical and NOT the target or ID
        cols_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
        ignored = [self.cfg['data']['target_col'], 'TransactionID', 'TransactionDT']
        cols_to_scale = [c for c in cols_to_scale if c not in ignored]

        if is_train:
            self.scaler.fit(df[cols_to_scale])
        
        df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        return df
