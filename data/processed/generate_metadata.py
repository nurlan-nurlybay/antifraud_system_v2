"""
Metadata Generator for Fraud Detection Datasets.
Placed in data/processed/ to provide immediate context for npz artifacts.
"""

import numpy as np
import yaml
from pathlib import Path

def generate_metadata():
    # This automatically finds the directory the script is sitting in
    data_dir = Path(__file__).parent 
    metadata = {"datasets": {}}

    # Find all generated npz files in the SAME folder
    npz_files = sorted(data_dir.glob("*.npz"))
    
    if not npz_files:
        print(f"❌ No .npz files found in {data_dir.absolute()}!")
        return

    for f in npz_files:
        data = np.load(f)
        x_shape = list(data['X'].shape)
        
        # Determine 2D vs 3D
        if len(x_shape) == 3:
            samples, timesteps, features = x_shape
            type_str = "3D (Sequence)"
        else:
            samples, features = x_shape
            timesteps = 1
            type_str = "2D (Tabular)"

        metadata["datasets"][f.name] = {
            "type": type_str,
            "samples": samples,
            "timesteps": timesteps,
            "features": features,
            "shape": x_shape,
            "fraud_rate_pct": round(float(np.mean(data['y']) * 100), 3)
        }

    # Save to metadata.yaml in the same folder
    with open(data_dir / "metadata.yaml", "w") as out:
        yaml.dump(metadata, out, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Metadata saved to {data_dir.absolute()}/metadata.yaml")

if __name__ == "__main__":
    generate_metadata()
