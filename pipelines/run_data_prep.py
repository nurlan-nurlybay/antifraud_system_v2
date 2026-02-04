import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.fd.data_prep.utils import load_config  # type: ignore
from src.fd.data_prep.data import load_raw_csv, chronological_split  # type: ignore
from src.fd.data_prep.features import fit_scaler  # type: ignore


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_npz(path: Path, arr: np.ndarray):
    np.savez_compressed(path, data=arr)


def add_creditcard_features(X_transformed: pd.DataFrame, df_original: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """X_transformed contins 3 time features, and this function adds the rest of the features to it

    Args:
        X_transformed (pd.DataFrame): _description_
        df_original (_type_): _description_
        amount_col (str, optional): _description_. Defaults to "Amount".

    Returns:
        tuple[pd.DataFrame, list[str]]: final prepared data with normalized features, list with names of features (except for time features)
    """
    v_features = [col for col in df_original.columns if col.startswith('V')]
    
    # Get indices matching between transformed and original
    X_additional = df_original[v_features].astype(np.float32).values
    
    # Combine: transformed time features + V features + Amount
    X_extended = np.hstack([X_transformed, X_additional])
    return X_extended, v_features


def main(cfg_path: str):
    cfg = load_config(cfg_path)

    raw_csv = Path(cfg["paths"]["raw_csv"])
    out_dir = Path(cfg["paths"]["processed_dir"])
    ensure_dir(out_dir)

    target_col = cfg["data"]["target_col"]
    test_size = float(cfg["data"]["test_size"])
    val_size = float(cfg["data"]["val_size"])

    print(f"[1/5] Loading {raw_csv} ...")
    df = load_raw_csv(str(raw_csv))

    print("[2/5] Chronological split ...")
    df_train, df_val, df_test = chronological_split(df, val_size, test_size)

    print("[3/5] Fit scalers on TRAIN only ...")
    scaler = fit_scaler(df_train)

    print("[4/5] Transform features (train/val/test) ...")
    X_train = scaler.get_normalized_features(df_train)
    X_val   = scaler.get_normalized_features(df_val)
    X_test  = scaler.get_normalized_features(df_test)

    X_train, additional_features = add_creditcard_features(X_train, df_train)
    X_val, _ = add_creditcard_features(X_val, df_val)
    X_test, _ = add_creditcard_features(X_test, df_test)

    y_train = df_train[target_col].to_numpy(dtype=np.int64)
    y_val   = df_val[target_col].to_numpy(dtype=np.int64)
    y_test  = df_test[target_col].to_numpy(dtype=np.int64)


    print("[5/5] Save artifacts ...")
    save_npz(out_dir / "X_train.npz", X_train)
    save_npz(out_dir / "X_val.npz",   X_val)
    save_npz(out_dir / "X_test.npz",  X_test)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_val.npy",   y_val)
    np.save(out_dir / "y_test.npy",  y_test)

    # Save scaler params
    np.savez(out_dir / "scaler.npz", 
            mu_amount=scaler.mu_amount, 
            sigma_amount=scaler.sigma_amount, 
            mu_time=scaler.mu_time, 
            sigma_time=scaler.sigma_time)
    
    normalized_features = ["time_days_z", "tod_sin", "tod_cos", "Amount"]
    all_features = normalized_features + additional_features
    
    # Minimal manifest
    manifest = {
        "splits": {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
        "features": all_features,
        "normalized_features": normalized_features,
        "shapes": {
            "X_train": list(X_train.shape),
            "X_val":   list(X_val.shape),
            "X_test":  list(X_test.shape),
        },
        "class_balance": {
            "train_pos": int(y_train.sum()), "train_neg": int((y_train == 0).sum()),
            "val_pos":   int(y_val.sum()),   "val_neg":   int((y_val == 0).sum()),
            "test_pos":  int(y_test.sum()),  "test_neg":  int((y_test == 0).sum()),
        },
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("Done →", out_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/data_prep.yaml")
    args = parser.parse_args()
    main(args.config)
