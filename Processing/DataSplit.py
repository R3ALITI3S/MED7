from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "ProcessedData"

def load_all_labeled(root: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load all participant CSVs from the directory and combine them into one DataFrame."""
    # Find all files matching the pattern (e.g., Participant_1_GZ.csv)
    files = sorted(root.glob("Participant_*.csv"))
    if not files:
        raise FileNotFoundError(f"No labeled files in {root}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # If the file doesn't already have a Participant column, extract it from the filename.
        # For example, 'Participant_1_GZ.csv' splits by '_' into ['Participant', '1', 'GZ.csv'],
        # so index [1] gives us '1'.
        if "Participant" not in df.columns:
            df["Participant"] = f.stem.split("_")[1]
            dfs.append(df)

    print(f"Loaded {len(files)} participants, {sum(map(len, dfs))} rows total")
    # Combine all separate dataframes into one large dataframe
    return pd.concat(dfs, ignore_index=True)

def make_train_val_test_split(
        df: pd.DataFrame,
        group_col: str = "Participant",
        val_frac: float = 0.18,
        test_frac: float = 0.18,
        seed: int = 42,
):
    """
        Split participants into train/val/test sets.
        Splitting is done by Participant ID to prevent data leakage between sets.
        """
    # Get a list of unique participant IDs and shuffle them using a fixed seed
    # so the split is random but reproducible every time you run the script.
    parts = np.array(sorted(df[group_col].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(parts)

    # Calculate how many participants go into each split (minimum 1)
    n = len(parts)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))

    # Slice the shuffled array into test, val, and train groups
    test_p = parts[:n_test]
    val_p = parts[n_test:n_test+n_val]
    train_p = parts[n_test+n_val:]

    def take (ps):
        """Helper function to return all rows belonging to the given participant IDs."""
        return df[df[group_col].isin(ps)].copy()

    # Return the actual DataFrames, plus a dictionary showing which participants went where
    return (
        take(train_p),
        take(val_p),
        take(test_p),
     {
            "train": train_p.tolist(),
            "val": val_p.tolist(),
            "test": test_p.tolist()
        },
    )

def get_folds(df: pd.DataFrame, group_col: str = "participant",
              method: str = "gkf", n_splits: int = 5):
    """Generate cross-validation folds based on participant groups."""
    groups = df[group_col].to_numpy()

    # LOSO (Leave-One-Subject-Out) trains on all but one participant.
    # GroupKFold splits the participants into N equal-sized chunks.
    splitter = LeaveOneGroupOut() if method == "loso" else GroupKFold(n_splits=n_splits)

    # Returns a list of (train_indices, test_indices) for each fold
    return list(splitter.split(df.index, groups=groups))

if __name__ == "__main__":
    # 1. Load the data
    df = load_all_labeled()

    # 2. Create the major holdout split (Train / Val / Test)
    train, val, test, groups = make_train_val_test_split(df, seed=42)
    print("Train participants:", groups["train"])
    print("Val participants:", groups["val"])
    print("Test participants:", groups["test"])
    print(f"Rows: train={len(train)}, val={len(val)}, test={len(test)}")

# 3. Create Cross-Validation folds within the TRAIN set
# Used for tuning the model and testing it on unseen participants without touching the Val/Test sets.
# We cap n_splits at 5 (or the number of participants, if fewer than 5).
print("\n--- Train Set Folds (for Cross-Validation) ---")
folds = get_folds(train, method="gkf", n_splits=min(5, train["participant"].nunique()))
for i, (tr, va) in enumerate(folds):
    print(f"    Fold{i}: {len(tr)} train rows, {len(va)} val rows")

# 4. (Optional) Global LOSO
# Creates one fold per participant in the ENTIRE dataset.
# Useful if you later want to report an overall "Leave-One-Subject-Out" score.
loso = get_folds(df, method="loso")
print(f"LOSO folds: {len(loso)}")