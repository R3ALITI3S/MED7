
"""
Train an unsupervised KMeans gaze-behaviour model. idk if thts is right

Input:
    Project/data/
        Tobii gaze recordings. they have a weird file name

Outputs:
    Project/attention_model.joblib
    Project/gaze_clusters.csv
    Project/plots/ - good to have

The model identifies gaze-behaviour clusters. It does not automatically
identify semantic attention labels such as Attention high / attention low.
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



# CONFIGURATION
WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.5
RANDOM_STATE = 42
N_CLUSTERS = 4

FEATURE_COLUMNS = [
    "gaze_x_mean",
    "gaze_y_mean",
    "gaze_x_std",
    "gaze_y_std",
    "gaze_x_range",
    "gaze_y_range",
    "gaze_dispersion",
    "gaze_velocity_mean",
    "gaze_velocity_std",
    "gaze_velocity_max",
    "sample_count",
    "pupil_mean",
    "pupil_std",
    "pupil_range",
    "gaze3d_x_mean",
    "gaze3d_y_mean",
    "gaze3d_z_mean",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

LOGGER = logging.getLogger("train_attention")



# GAZE FILE LOADING
def load_gaze_file(path):
    """
    Read Tobii JSONL gaze recordings.

    Supports:
        {"type": "gaze", "timestamp": ..., "data": {...}}

    Also supports direct gaze dictionaries.
    """

    records = []

    with open(path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError:
                LOGGER.warning(
                    "Invalid JSON in %s at line %s",
                    path,
                    line_number,
                )
                continue

            if not isinstance(item, dict):
                continue

            if item.get("type") not in (None, "gaze"):
                continue

            data = item.get("data", item)

            if not isinstance(data, dict):
                continue

            timestamp = item.get("timestamp")

            if timestamp is None:
                timestamp = data.get("timestamp")

            if timestamp is None:
                continue

            try:
                timestamp = float(timestamp)
            except (TypeError, ValueError):
                continue

            gaze2d = data.get("gaze2d")

            if not isinstance(gaze2d, (list, tuple)):
                continue

            if len(gaze2d) < 2:
                continue

            try:
                gx = float(gaze2d[0])
                gy = float(gaze2d[1])
            except (TypeError, ValueError):
                continue

            if not np.isfinite(gx) or not np.isfinite(gy):
                continue

            if not (0.0 <= gx <= 1.0 and 0.0 <= gy <= 1.0):
                continue

            # Pupil diameter
            pupils = []

            for eye_name in ("eyeleft", "eyeright"):

                eye = data.get(eye_name)

                if not isinstance(eye, dict):
                    continue

                pupil = eye.get("pupildiameter")

                try:
                    pupil = float(pupil)

                    if np.isfinite(pupil) and pupil > 0:
                        pupils.append(pupil)

                except (TypeError, ValueError):
                    pass

            pupil_mean = (
                float(np.mean(pupils))
                if pupils else np.nan
            )

            # 3D gaze coordinates
            gaze3d = data.get("gaze3d", [np.nan, np.nan, np.nan])

            if not isinstance(gaze3d, (list, tuple)) or len(gaze3d) < 3:
                gaze3d = [np.nan, np.nan, np.nan]

            try:
                gaze3d = [
                    float(gaze3d[0]),
                    float(gaze3d[1]),
                    float(gaze3d[2]),
                ]
            except (TypeError, ValueError):
                gaze3d = [np.nan, np.nan, np.nan]

            records.append({
                "timestamp": timestamp,
                "gaze_x": gx,
                "gaze_y": gy,
                "pupil": pupil_mean,
                "gaze3d_x": gaze3d[0],
                "gaze3d_y": gaze3d[1],
                "gaze3d_z": gaze3d[2],
            })

    if not records:
        LOGGER.warning("No valid gaze samples found in %s", path)
        return pd.DataFrame()

    df = pd.DataFrame(records)

    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp")
    df = df.reset_index(drop=True)

    df["time"] = df["timestamp"] - df["timestamp"].iloc[0]

    return df



# FEATURE EXTRACTION
def extract_features(window):
    """Extract the 17 gaze features used during training and live prediction."""

    gx = window["gaze_x"].to_numpy(dtype=float)
    gy = window["gaze_y"].to_numpy(dtype=float)

    timestamps = window["time"].to_numpy(dtype=float)

    gaze_x_std = float(np.std(gx))
    gaze_y_std = float(np.std(gy))

    # Gaze dispersion
    gaze_dispersion = float(
        np.mean(
            np.sqrt(
                (gx - np.mean(gx)) ** 2
                + (gy - np.mean(gy)) ** 2
            )
        )
    )

    # Gaze velocity
    velocities = []

    if len(gx) >= 2:

        dt = np.diff(timestamps)

        dx = np.diff(gx)
        dy = np.diff(gy)

        valid = dt > 0

        if np.any(valid):

            distances = np.sqrt(
                dx[valid] ** 2 + dy[valid] ** 2
            )

            velocities = distances / dt[valid]

    if len(velocities):

        velocity_mean = float(np.mean(velocities))
        velocity_std = float(np.std(velocities))
        velocity_max = float(np.max(velocities))

    else:

        velocity_mean = 0.0
        velocity_std = 0.0
        velocity_max = 0.0

    # Pupil features
    pupil = window["pupil"].to_numpy(dtype=float)
    valid_pupil = pupil[np.isfinite(pupil)]

    if len(valid_pupil):

        pupil_mean = float(np.mean(valid_pupil))
        pupil_std = float(np.std(valid_pupil))
        pupil_range = float(np.ptp(valid_pupil))

    else:

        pupil_mean = np.nan
        pupil_std = np.nan
        pupil_range = np.nan

    # 3D features
    gaze3d_x = window["gaze3d_x"].to_numpy(dtype=float)
    gaze3d_y = window["gaze3d_y"].to_numpy(dtype=float)
    gaze3d_z = window["gaze3d_z"].to_numpy(dtype=float)

    return {
        "gaze_x_mean": float(np.mean(gx)),
        "gaze_y_mean": float(np.mean(gy)),
        "gaze_x_std": gaze_x_std,
        "gaze_y_std": gaze_y_std,
        "gaze_x_range": float(np.ptp(gx)),
        "gaze_y_range": float(np.ptp(gy)),
        "gaze_dispersion": gaze_dispersion,
        "gaze_velocity_mean": velocity_mean,
        "gaze_velocity_std": velocity_std,
        "gaze_velocity_max": velocity_max,
        "sample_count": int(len(window)),
        "pupil_mean": pupil_mean,
        "pupil_std": pupil_std,
        "pupil_range": pupil_range,
        "gaze3d_x_mean": float(np.nanmean(gaze3d_x))
        if np.isfinite(gaze3d_x).any() else np.nan,
        "gaze3d_y_mean": float(np.nanmean(gaze3d_y))
        if np.isfinite(gaze3d_y).any() else np.nan,
        "gaze3d_z_mean": float(np.nanmean(gaze3d_z))
        if np.isfinite(gaze3d_z).any() else np.nan,
    }



# DATASET CREATION


def create_dataset(data_dir):

    rows = []

    files = sorted(
        path for path in Path(data_dir).iterdir()
        if path.is_file()
        and path.suffix.lower() in ("", ".jsonl", ".txt")
    )

    if not files:
        raise FileNotFoundError(
            f"No gaze recordings found in {data_dir}"
        )

    for path in files:

        LOGGER.info("Reading %s", path.name)

        data = load_gaze_file(path)

        if len(data) < 2:
            continue

        start = 0.0
        end = float(data["time"].iloc[-1])

        while start + WINDOW_SECONDS <= end + 1e-9:

            stop = start + WINDOW_SECONDS

            window = data[
                (data["time"] >= start)
                & (data["time"] < stop)
            ]

            if len(window) >= 2:

                features = extract_features(window)

                features["start_seconds"] = start
                features["end_seconds"] = stop
                features["source_file"] = path.name

                rows.append(features)

            start += STEP_SECONDS

    if not rows:
        raise ValueError(
            "No valid gaze windows could be created. "
            "Check that recordings are long enough and contain valid samples."
        )

    return pd.DataFrame(rows)



# PLOTTING


def plot_cluster_results(dataset, labels, transformed, plot_dir):

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(transformed)

    plt.figure(figsize=(9, 7))

    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.7,
    )

    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title("Gaze Behaviour Clusters")
    plt.colorbar(scatter, label="Cluster")

    plt.tight_layout()
    plt.savefig(plot_dir / "gaze_pca_clusters.png", dpi=150)
    plt.close()

    # Cluster distribution
    counts = pd.Series(labels).value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.xlabel("Cluster")
    plt.ylabel("Number of windows")
    plt.title("Cluster Distribution")

    plt.tight_layout()
    plt.savefig(plot_dir / "cluster_distribution.png", dpi=150)
    plt.close()

    # Cluster feature profiles
    profile = dataset.copy()
    profile["cluster"] = labels

    means = profile.groupby("cluster")[FEATURE_COLUMNS].mean()

    plt.figure(figsize=(14, 7))

    means.T.plot(kind="bar", figsize=(14, 7))

    plt.title("Mean Gaze Features by Cluster")
    plt.xlabel("Feature")
    plt.ylabel("Mean value")
    plt.xticks(rotation=70)

    plt.tight_layout()
    plt.savefig(plot_dir / "cluster_feature_profiles.png", dpi=150)
    plt.close()

    # Cluster timeline
    plt.figure(figsize=(12, 5))

    plt.scatter(
        dataset["start_seconds"],
        labels,
        c=labels,
        cmap="viridis",
        s=15,
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Cluster")
    plt.title("Gaze Cluster Timeline")

    plt.tight_layout()
    plt.savefig(plot_dir / "cluster_timeline.png", dpi=150)
    plt.close()



# TRAINING


def train(data_dir, output_path, plot_dir):

    data_dir = Path(data_dir)
    output_path = Path(output_path)
    plot_dir = Path(plot_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Creating gaze feature dataset...")

    dataset = create_dataset(data_dir)

    X = dataset[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan
    )

    if len(X) <= N_CLUSTERS:
        raise ValueError(
            f"Need more than {N_CLUSTERS} gaze windows. "
            f"Only found {len(X)}."
        )

    # Preprocessing
    preprocessing = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    transformed = preprocessing.fit_transform(X)

    # KMeans
    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    labels = model.fit_predict(transformed)

    # Silhouette score
    unique_labels = np.unique(labels)

    if 1 < len(unique_labels) < len(labels):

        score = float(
            silhouette_score(transformed, labels)
        )

    else:
        score = None

    LOGGER.info("Training complete.")
    LOGGER.info("Number of clusters: %s", N_CLUSTERS)
    LOGGER.info("Silhouette score: %s", score)

    # Save clustered windows
    output_dataset = dataset.copy()
    output_dataset["cluster"] = labels

    csv_path = data_dir.parent / "gaze_clusters.csv"

    output_dataset.to_csv(csv_path, index=False)

    LOGGER.info("Clustered data saved to %s", csv_path)

    # Plots
    plot_cluster_results(
        dataset,
        labels,
        transformed,
        plot_dir,
    )

    # Save model bundle
    bundle = {
        "preprocessing": preprocessing,
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "window_seconds": WINDOW_SECONDS,
        "step_seconds": STEP_SECONDS,
        "n_clusters": N_CLUSTERS,
        "silhouette_score": score,
    }

    joblib.dump(bundle, output_path)

    LOGGER.info("Model saved to %s", output_path)

    # Print cluster profiles to help interpret them.
    profile = output_dataset.groupby("cluster")[FEATURE_COLUMNS].mean()

    print("\nCluster feature profiles:")
    print(profile.round(4).to_string())

    print("\nTraining complete.")
    print(f"Model: {output_path.resolve()}")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Plots: {plot_dir.resolve()}")

    return bundle



# MODEL LOADING
def load_model(model_path):

    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    bundle = joblib.load(model_path)

    required = {
        "preprocessing",
        "model",
        "feature_columns",
        "window_seconds",
        "step_seconds",
        "n_clusters",
    }

    missing = required.difference(bundle)

    if missing:
        raise ValueError(
            f"Invalid model bundle. Missing keys: {sorted(missing)}"
        )

    return bundle



# LIVE ATTENTION ESTIMATOR
class LiveAttention:
    """
    Apply the trained KMeans model to the incoming gaze samples.

    The returned cluster is the model's current gaze-behaviour cluster.
    It is not a validated AGAINST ANYTHING - FIND CIETS
    """

    def __init__(self, bundle):

        self.preprocessing = bundle["preprocessing"]
        self.model = bundle["model"]

        self.feature_columns = list(bundle["feature_columns"])

        self.window_seconds = float(bundle["window_seconds"])
        self.step_seconds = float(bundle["step_seconds"])

        self.samples = []

        self.recording_start = None
        self.next_window_start = None

    def push(self, sample, timestamp):

        if not isinstance(sample, dict):
            return None

        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(timestamp):
            return None

        data = sample.get("data", sample)

        if not isinstance(data, dict):
            return None

        gaze2d = data.get("gaze2d")

        if not isinstance(gaze2d, (list, tuple)) or len(gaze2d) < 2:
            return None

        try:
            gx = float(gaze2d[0])
            gy = float(gaze2d[1])
        except (TypeError, ValueError):
            return None

        if not (
            np.isfinite(gx)
            and np.isfinite(gy)
            and 0.0 <= gx <= 1.0
            and 0.0 <= gy <= 1.0
        ):
            return None

        # Pupil diameter
        pupils = []

        for eye_name in ("eyeleft", "eyeright"):

            eye = data.get(eye_name)

            if not isinstance(eye, dict):
                continue

            pupil = eye.get("pupildiameter")

            try:
                pupil = float(pupil)

                if np.isfinite(pupil) and pupil > 0:
                    pupils.append(pupil)

            except (TypeError, ValueError):
                pass

        pupil_mean = (
            float(np.mean(pupils))
            if pupils else np.nan
        )

        # 3D gaze
        gaze3d = data.get("gaze3d", [np.nan, np.nan, np.nan])

        if not isinstance(gaze3d, (list, tuple)) or len(gaze3d) < 3:
            gaze3d = [np.nan, np.nan, np.nan]

        try:
            gaze3d = [
                float(gaze3d[0]),
                float(gaze3d[1]),
                float(gaze3d[2]),
            ]
        except (TypeError, ValueError):
            gaze3d = [np.nan, np.nan, np.nan]

        if self.recording_start is None:

            self.recording_start = timestamp
            self.next_window_start = 0.0

        relative_time = timestamp - self.recording_start

        if relative_time < 0:
            return None

        self.samples.append({
            "time": relative_time,
            "gaze_x": gx,
            "gaze_y": gy,
            "pupil": pupil_mean,
            "gaze3d_x": gaze3d[0],
            "gaze3d_y": gaze3d[1],
            "gaze3d_z": gaze3d[2],
        })

        result = None

        # Process every complete window that has become available.
        while (
            relative_time >=
            self.next_window_start + self.window_seconds
        ):

            window_start = self.next_window_start
            window_end = window_start + self.window_seconds

            window = pd.DataFrame([
                item for item in self.samples
                if window_start <= item["time"] < window_end
            ])

            if len(window) >= 2:

                features = extract_features(window)

                feature_frame = pd.DataFrame(
                    [features],
                    columns=self.feature_columns,
                )

                transformed = self.preprocessing.transform(
                    feature_frame
                )

                cluster = int(
                    self.model.predict(transformed)[0]
                )

                result = {
                    "cluster": cluster,
                    "state": f"Cluster {cluster}",
                    "score": None,
                    "raw": cluster,
                    "t": timestamp,
                }

            self.next_window_start += self.step_seconds

            # Discard samples older than the next required window.
            self.samples = [
                item for item in self.samples
                if item["time"] >= self.next_window_start
            ]

        return result



# COMMAND LINE
def main():

    project_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Train an unsupervised Tobii gaze clustering model."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "data",
        help="Folder containing Tobii JSONL gaze recordings.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "attention_model.joblib",
        help="Where to save the trained model.",
    )

    parser.add_argument(
        "--plots",
        type=Path,
        default=project_dir / "plots",
        help="Folder for training plots.",
    )

    args = parser.parse_args()

    train(
        args.data_dir,
        args.output,
        args.plots,
    )


if __name__ == "__main__":
    main()
