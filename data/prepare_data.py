import os
import argparse
import pandas as pd


def verify_dataset(data_dir: str = "data"):
    """
    Verifies that the user-provided dataset (train.csv, val.csv, test.csv)
    is present and valid.
    """
    required_splits = ["train.csv", "val.csv", "test.csv"]
    nested_dir = os.path.join(data_dir, "arrhythmia_dataset")

    # If dataset is inside nested arrhythmia_dataset/ folder, synchronize to data/
    if os.path.exists(nested_dir):
        for split in required_splits:
            src = os.path.join(nested_dir, split)
            dst = os.path.join(data_dir, split)
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)

    print("=" * 60)
    print("      VERIFYING USER-PROVIDED ECG ARRHYTHMIA DATASET        ")
    print("=" * 60)

    for split in required_splits:
        path = os.path.join(data_dir, split)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing dataset file '{path}'. Please ensure your real dataset CSVs are placed in '{data_dir}/'."
            )
        df = pd.read_csv(path)
        class_dist = df["class"].value_counts().to_dict() if "class" in df.columns else {}
        print(f"Split: {split:<10} | Samples: {len(df):<6} | Features: {df.shape[1]-1} | Class Dist: {class_dist}")

    print("=" * 60)
    print("All user dataset splits verified successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify attached ECG dataset.")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset directory")
    args = parser.parse_args()
    verify_dataset(args.data_dir)
