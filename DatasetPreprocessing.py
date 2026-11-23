from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess_data():
    # Import and Load the Dataset below
    base_folder = Path("DATASET")

    # TODO: Replace with your actual feature column names, gaze, saccades, locations and so on pls
    columns = ["feature_1", "feature_2", "feature_3"]

    dataframes = []
    for folder_name in ["Dataset"]:
        folder_path = base_folder / folder_name
        if folder_path.exists() and folder_path.is_dir():
            for file in folder_path.iterdir():
                if file.is_file():
                    try:
                        temp_df = pd.read_csv(file, header=None, sep=None, engine="python")
                        if temp_df.shape[1] == len(columns):
                            temp_df.columns = columns
                            temp_df["subject_folder"] = folder_name
                            temp_df["source_file"] = file.name
                            dataframes.append(temp_df)
                    except Exception:
                        pass

    if not dataframes:
        raise ValueError("No data loaded. Please check your folder path and column definitions :)")

    df = pd.concat(dataframes, ignore_index=True)
    print("The shape of the dataset:", df.shape)

    return df


def split_dataset(df):
    # Prepare Features (X) and Target (y)
    # TODO: Replace with your actual feature columns and target/action column, ehg Location of the eyes and motion of the eyes
    X = df[["feature_1", "feature_2"]]
    y = df["target_column"]

    # Split the Dataset
    train_size = 0.7

    # Split into train and remaining (validation + test)
    X_train, X_, y_train, y_ = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=42
    )

    print(f"train: {len(X_train)} ({(len(X_train) / len(X)):.2f})\n"
          f"remaining: {len(X_)} ({(len(X_) / len(X)):.2f})")

    # Split remaining equally into validation and test sets (50% each of the remaining)
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_, random_state=42
    )

    print(f"train: {len(X_train)} ({len(X_train) / len(X):.2f})\n"
          f"val: {len(X_val)} ({len(X_val) / len(X):.2f})\n"
          f"test: {len(X_test)} ({len(X_test) / len(X):.2f})")

    return X_train, X_val, X_test, y_train, y_val, y_test