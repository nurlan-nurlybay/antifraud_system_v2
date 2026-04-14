"""
Core Data Preprocessing Engine.

This module houses the Expanding Mean Target Encoder and the DataPreprocessor engine. 
It defines the distinct feature engineering paths (Trees, MLP, LSTM) required 
for the Hybrid Meta-Model Architecture, explicitly handling the 'Signal of Absence' 
(NaN imputation) depending on the target model's mathematical properties.

*UPDATED*: Zero Temporal Leakage achieved using Expanding Window (Cumulative) Encoders.
All transformers natively save state artifacts for production inference.
"""

import pandas as pd
import numpy as np
import structlog
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.fd.utils.logging import setup_logger

logger = structlog.get_logger(__name__)

class ExpandingMeanEncoder:
    """
    Computes Target Encoding purely chronologically to give 0 data leakage.
    For transaction `i`, the encoding is the smoothed mean of target values
    for transactions `0` to `i-1` having that same category.
    """
    def __init__(self, cols: list[str], target_col: str, m: int = 10):
        self.cols = cols
        self.target_col = target_col
        self.m = m
        self.global_mean: float = 0.0
        self.category_mappings: dict[str, dict[str, float]] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        
        # Cast target to float32 to bypass Pandas int8 aggregation bugs
        target_series = df[self.target_col].astype(np.float32)
        self.global_mean = float(target_series.mean())

        for col in self.cols:
            self.category_mappings[col] = {}
            # Group by category, then expand
            # Note: The dataframe MUST be sorted chronologically before this is called
            # We use cumulative sum and count
            cumsum = df.groupby(col)[self.target_col].cumsum() - df[self.target_col]
            cumcount = df.groupby(col).cumcount() 
            
            # Apply Bayesian Smoothing formulation dynamically per row
            smoothed_mean = (cumsum + self.m * self.global_mean) / (cumcount + self.m)
            df_encoded[col + '_te'] = smoothed_mean.astype(np.float32)

            # Save the LAST known moving average to apply to Val/Test sets
            final_stats = df.groupby(col)[self.target_col].agg(['sum', 'count'])
            for cat, row in final_stats.iterrows():
                final_smooth = (row['sum'] + self.m * self.global_mean) / (row['count'] + self.m)
                self.category_mappings[col][str(cat)] = float(final_smooth)

        return df_encoded.drop(columns=self.cols, errors='ignore')

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        for col in self.cols:
            mapping = self.category_mappings.get(col, {})
            # Fast map with fallback to global uniform prior
            mapped_series = df[col].map(mapping).fillna(self.global_mean)
            df_encoded[col + '_te'] = mapped_series.astype(np.float32)
            
        return df_encoded.drop(columns=self.cols, errors='ignore')


class DataPreprocessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        
        # Dynamic Column Definitions from Config
        self.id_col = cfg['data']['id_col']
        self.time_col = cfg['data']['time_col']
        self.target_col = cfg['data']['target_col']
        self.amount_col = cfg['data']['amount_col']
        
        # Universal drop list for ML models (Metadata columns)
        self.ignore_cols = ['Uid', self.id_col, self.time_col, self.target_col]

        self.scaler = StandardScaler()       # Global scaler for all numeric features
        self.v_scaler = StandardScaler()      # Dedicated scaler for V-features BEFORE PCA
        # Deterministic PCA
        self.pca = PCA(n_components=cfg['features']['v_features_pca_dims'], random_state=42)
        self.te_encoder = ExpandingMeanEncoder(
            cols=cfg['features']['categorical']['columns'],
            target_col=self.target_col,
            m=cfg['features']['categorical']['target_encoding'].get('m', 10)
        )
        self.numerical_medians = {}
        self.nan_mask_columns = []

    # --- STAGE 1: THE CLEAN BASE ---
    def clean_base_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        drop_cols = self.cfg['features'].get('drop_features', [])
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Create Uid 
        uid_cols = self.cfg['features']['uid_columns']
        df['Uid'] = df[uid_cols].astype(str).agg('_'.join, axis=1).astype('category').cat.codes
        
        # Sort Chronologically (CRITICAL for Expanding Target Encoder)
        df = df.sort_values(by=self.time_col).reset_index(drop=True)

        # VELOCITY FEATURE: Time since last transaction for this specific user
        df['time_dist_last'] = df.groupby('Uid')[self.time_col].diff()

        # ROLLING AMOUNT VELOCITY: Detects spending anomalies per user
        rolling_amt = df.groupby('Uid')[self.amount_col].rolling(window=5, min_periods=1)
        df['amt_rolling_mean'] = rolling_amt.mean().reset_index(level=0, drop=True)
        df['amt_rolling_std'] = rolling_amt.std().reset_index(level=0, drop=True).fillna(0)

        # Impute Strings
        cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != 'Uid']
        df[cat_cols] = df[cat_cols].fillna(self.cfg['features']['impute_categoricals'])
        
        return df

    # --- STAGE 2: MODEL-SPECIFIC TRANSFORMS ---
    def get_tree_features(self, df: pd.DataFrame, is_train: bool = True) -> tuple[np.ndarray, np.ndarray]:
        df = df.copy()
        df = self._apply_cyclical_time(df)
        df = self._apply_target_encoding(df, is_train)
        
        # Tree-specific Imputation using dynamic ignore list
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in self.ignore_cols]
        df[num_cols] = df[num_cols].fillna(-999)
        
        y = df[self.target_col].values if self.target_col in df.columns else None
        X = df.drop(columns=self.ignore_cols, errors='ignore').values
        return X.astype(np.float32), (y.astype(np.int8) if y is not None else None)

    def get_mlp_features(self, df: pd.DataFrame, is_train: bool = True) -> tuple[np.ndarray, np.ndarray]:
        df = self._transform_neural_base(df, is_train)
        y = df[self.target_col].values if self.target_col in df.columns else None
        X = df.drop(columns=self.ignore_cols, errors='ignore').values
        return X.astype(np.float32), (y.astype(np.int8) if y is not None else None)

    def get_lstm_features(self, df: pd.DataFrame, is_train: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Transaction-centric rolling window LSTM features."""
        df = self._transform_neural_base(df, is_train)
        seq_len = self.cfg['data']['sequence_length']
        
        feature_cols = [c for c in df.columns if c not in self.ignore_cols]
        n_features = len(feature_cols)
        n_rows = len(df)
        
        features = df[feature_cols].values.astype(np.float32)
        uids = df['Uid'].values
        y = df[self.target_col].values.astype(np.int8) if self.target_col in df.columns else None
        
        user_history: dict[int, list[int]] = {}
        for i in range(n_rows):
            uid = int(uids[i])
            if uid not in user_history:
                user_history[uid] = []
            user_history[uid].append(i)
        
        X_3d = np.zeros((n_rows, seq_len, n_features), dtype=np.float32)
        user_position: dict[int, int] = {} 
        
        for i in range(n_rows):
            uid = int(uids[i])
            user_position[uid] = user_position.get(uid, -1) + 1
            pos = user_position[uid]
            history = user_history[uid]
            
            n_available = pos + 1
            n_to_copy = min(n_available, seq_len)
            
            start_hist_idx = pos - n_to_copy + 1
            for j in range(n_to_copy):
                X_3d[i, seq_len - n_to_copy + j, :] = features[history[start_hist_idx + j]]
        
        return X_3d, y

    # --- INTERNAL HELPER METHODS ---
    def _transform_neural_base(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        df = df.copy()
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in self.ignore_cols]
        
        if is_train:
            self.numerical_medians = df[num_cols].median().to_dict()
            # Exclude V-features from NaN masks — they're being PCA'd away.
            # Their missingness patterns are captured by trees via -999.
            import re
            self.nan_mask_columns = [c for c in num_cols if df[c].isnull().any() and not re.match(r'^V\d+$', c)]
            
        new_masks = {}
        for col in self.nan_mask_columns:
            new_masks[col + '_is_nan'] = df[col].isnull().astype(np.float32)

        df = pd.concat([df, pd.DataFrame(new_masks, index=df.index)], axis=1)
            
        for col in num_cols:
            df[col] = df[col].fillna(self.numerical_medians.get(col, 0))

        df = self._apply_cyclical_time(df)
        df = self._apply_target_encoding(df, is_train)
        
        if self.cfg['features']['log_transform_amount']:
            df[self.amount_col] = np.log1p(df[self.amount_col])
            df['time_dist_last'] = np.log1p(df['time_dist_last'].clip(lower=0))
            
        df = self._apply_pca(df, is_train)
        df = self._apply_scaling(df, is_train)
        return df

    def _apply_cyclical_time(self, df: pd.DataFrame) -> pd.DataFrame:
        # Hour-of-day: captures nocturnal fraud patterns
        df['hour'] = (df[self.time_col] // 3600) % 24
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype(np.float32)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype(np.float32)
        # Day-of-week: captures weekend vs weekday spending
        df['day'] = (df[self.time_col] // 86400) % 7
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7).astype(np.float32)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7).astype(np.float32)
        return df.drop(columns=['hour', 'day'])

    def _apply_target_encoding(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        if is_train:
            df = self.te_encoder.fit_transform(df)
        else:
            df = self.te_encoder.transform(df)
        return df

    def _apply_pca(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        # Select ONLY raw V columns (V1..V339), NOT V*_is_nan masks
        import re
        v_cols = [c for c in df.columns if re.match(r'^V\d+$', c)]
        
        # Scale V-features BEFORE PCA (critical for honest variance decomposition)
        if is_train:
            self.v_scaler.fit(df[v_cols])
        v_scaled = self.v_scaler.transform(df[v_cols])
        
        if is_train:
            self.pca.fit(v_scaled)
        v_pca = self.pca.transform(v_scaled)
        
        pca_cols = [f'V_pca_{i}' for i in range(self.cfg['features']['v_features_pca_dims'])]
        df_pca = pd.DataFrame(v_pca, columns=pca_cols, index=df.index).astype(np.float32)
        return pd.concat([df.drop(columns=v_cols), df_pca], axis=1)

    def _apply_scaling(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in self.ignore_cols]
        if is_train:
            self.scaler.fit(df[num_cols])
        df[num_cols] = self.scaler.transform(df[num_cols]).astype(np.float32)
        return df

    # --- ARTIFACT PERSISTENCE ---
    def save_artifacts(self, output_dir: str = "models/preprocessors"):
        """Saves deterministic artifacts to disk to ensure flawless test inference."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        artifact_path = out / "preprocessor.joblib"
        bundle = {
            "scaler": self.scaler,
            "v_scaler": self.v_scaler,
            "pca": self.pca,
            "te_encoder": self.te_encoder,
            "numerical_medians": self.numerical_medians,
            "nan_mask_columns": self.nan_mask_columns
        }
        
        joblib.dump(bundle, artifact_path)
        logger.info("Saved data preprocessing artifacts", path=str(artifact_path))

    def load_artifacts(self, input_dir: str = "models/preprocessors"):
        """Loads deterministic artifacts from disk to ensure flawless test inference."""
        artifact_path = Path(input_dir) / "preprocessor.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing preprocessor state: {artifact_path}")
            
        bundle = joblib.load(artifact_path)
        self.scaler = bundle["scaler"]
        self.v_scaler = bundle.get("v_scaler")  # backwards compat if using an old bundle
        self.pca = bundle["pca"]
        self.te_encoder = bundle["te_encoder"]
        self.numerical_medians = bundle["numerical_medians"]
        self.nan_mask_columns = bundle["nan_mask_columns"]
        
        logger.info("Loaded data preprocessing artifacts", path=str(artifact_path))

