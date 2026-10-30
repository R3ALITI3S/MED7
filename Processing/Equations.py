import numpy as np
import pandas as pd
from pathlib import Path


# ==========================================================
# Load CSV
# ==========================================================

P = 6 # Participant number

PROJECT_ROOT = Path(__file__).resolve().parent.parent

csv_path = (
    PROJECT_ROOT
    / "FixedData"
    / f"Participant_{P}_GZ_fixed.csv"
)

df = pd.read_csv(csv_path)


# ==========================================================
# Handle missing values
# ==========================================================

columns_to_interpolate = [
    "timestamp",
    "gaze2d_x",
    "gaze2d_y",
    "gaze3d_x",
    "gaze3d_y",
    "gaze3d_z",
    "eyeleft.pupildiameter",
    "eyeright.pupildiameter",
    "eyeleft.gazeorigin_x",
    "eyeleft.gazeorigin_y",
    "eyeleft.gazeorigin_z",
    "eyeleft.gazedirection_x",
    "eyeleft.gazedirection_y",
    "eyeleft.gazedirection_z",
    "eyeright.gazeorigin_x",
    "eyeright.gazeorigin_y",
    "eyeright.gazeorigin_z",
    "eyeright.gazedirection_x",
    "eyeright.gazedirection_y",
    "eyeright.gazedirection_z"
]

for col in columns_to_interpolate:
    if col in df.columns:
        df[col] = df[col].interpolate(method="linear")


# ==========================================================
# Extract arrays
# ==========================================================

t = df["timestamp"].to_numpy()
dt = np.diff(t)

x_g = df["gaze2d_x"].to_numpy()
y_g = df["gaze2d_y"].to_numpy()

left_dir_x = df["eyeleft.gazedirection_x"].to_numpy()
left_dir_y = df["eyeleft.gazedirection_y"].to_numpy()
left_dir_z = df["eyeleft.gazedirection_z"].to_numpy()


# ==========================================================
# Eye Tracking Framework
# ==========================================================

class EyeTrackingFramework:

    @staticmethod
    def Rx(phi):
        return np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])

    @staticmethod
    def Ry(theta):
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

    @staticmethod
    def Rz(psi):
        return np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])

    @staticmethod
    def rotation_matrix(psi, theta, phi):

        return (
            EyeTrackingFramework.Rz(psi)
            @ EyeTrackingFramework.Ry(theta)
            @ EyeTrackingFramework.Rx(phi)
        )

    @staticmethod
    def rotate_heading_vector(v0, psi, theta, phi):

        R = EyeTrackingFramework.rotation_matrix(
            psi,
            theta,
            phi
        )

        return R @ np.asarray(v0)

    @staticmethod
    def alpha(psi, theta, phi):

        numerator = (
            np.sin(psi) *
            np.sin(theta) *
            np.sin(phi)
            +
            np.cos(psi) *
            np.cos(phi)
        )

        denominator = (
            np.cos(psi) *
            np.sin(theta) *
            np.sin(phi)
            -
            np.sin(psi) *
            np.cos(phi)
        )

        return np.pi/2 - np.arctan2(
            numerator,
            denominator
        )

    @staticmethod
    def beta(theta, phi):

        return (
            np.pi/2
            -
            np.arccos(
                np.cos(theta)
                *
                np.sin(phi)
            )
        )

    @staticmethod
    def velocity(x_gaze, y_gaze, dt):

        vx = np.diff(x_gaze) / dt
        vy = np.diff(y_gaze) / dt

        return vx, vy

    @staticmethod
    def acceleration(vx, vy, dt):

        dvx = np.diff(vx)
        dvy = np.diff(vy)

        return (
            np.sqrt(
                dvx**2 + dvy**2
            )
            /
            dt[1:]
        )

    @staticmethod
    def amplitude(x, y):

        return np.sqrt(
            (np.max(x) - np.min(x))**2
            +
            (np.max(y) - np.min(y))**2
        )

    @staticmethod
    def eye_euclidean_distance(
        x_gaze,
        y_gaze
    ):

        dx = np.diff(x_gaze)
        dy = np.diff(y_gaze)

        return np.sqrt(
            dx**2 + dy**2
        )

    @staticmethod
    def cluster_distance(ed_cluster):

        return np.sum(ed_cluster)

    @staticmethod
    def gaze_std(
        x_gaze,
        y_gaze
    ):

        sigma_x = np.std(x_gaze)
        sigma_y = np.std(y_gaze)

        return sigma_x, sigma_y


# ==========================================================
# Run calculations
# ==========================================================

if __name__ == "__main__":

    # NOTE:
    # These are gaze direction values, NOT true Euler angles.
    # Replace if your dataset contains yaw/pitch/roll.

    # ==========================================================
    # Create time-series versions of all features
    # ==========================================================

    window = 50  # rolling window size

    # Alpha/Beta for every sample
    alpha_ts = EyeTrackingFramework.alpha(
        left_dir_z,
        left_dir_y,
        left_dir_x
    )

    beta_ts = EyeTrackingFramework.beta(
        left_dir_y,
        left_dir_x
    )

    # Align with velocity intervals
    alpha_ts = alpha_ts[1:]
    beta_ts = beta_ts[1:]

    vx, vy = EyeTrackingFramework.velocity(
        x_g,
        y_g,
        dt
    )

    # Acceleration already exists per interval
    # Length = len(vx)-1
    acc = EyeTrackingFramework.acceleration(
        vx,
        vy,
        dt
    )
    acc_full = np.concatenate(([np.nan], acc))

    # Euclidean distance between consecutive gaze points
    ed = EyeTrackingFramework.eye_euclidean_distance(
        x_g,
        y_g
    )

    # ----------------------------------------------------------
    # Amplitude
    # ----------------------------------------------------------

    amp_ts = []

    for i in range(len(x_g)):
        start = max(0, i - window)

        x_window = x_g[start:i + 1]
        y_window = y_g[start:i + 1]

        amp_ts.append(
            EyeTrackingFramework.amplitude(
                x_window,
                y_window
            )
        )

    amp_ts = np.array(amp_ts)[1:]

    # ----------------------------------------------------------
    # Cluster distance
    # ----------------------------------------------------------

    cluster_dist_ts = (
        pd.Series(ed)
        .rolling(
            window=window,
            min_periods=1
        )
        .sum()
        .to_numpy()
    )

    # ----------------------------------------------------------
    # Standard deviations
    # ----------------------------------------------------------

    sigma_x_ts = (
        pd.Series(x_g)
        .rolling(
            window=window,
            min_periods=1
        )
        .std()
        .to_numpy()[1:]
    )

    sigma_y_ts = (
        pd.Series(y_g)
        .rolling(
            window=window,
            min_periods=1
        )
        .std()
        .to_numpy()[1:]
    )

    # ----------------------------------------------------------
    # Mean velocity
    # ----------------------------------------------------------

    mean_vx_ts = (
        pd.Series(vx)
        .rolling(
            window=window,
            min_periods=1
        )
        .mean()
        .to_numpy()
    )

    mean_vy_ts = (
        pd.Series(vy)
        .rolling(
            window=window,
            min_periods=1
        )
        .mean()
        .to_numpy()
    )

    # ----------------------------------------------------------
    # Mean acceleration
    # ----------------------------------------------------------

    mean_acc_ts = (
        pd.Series(acc_full)
        .rolling(
            window=window,
            min_periods=1
        )
        .mean()
        .to_numpy()
    )

    # ==========================================================
    # Export all dynamic features
    # ==========================================================

    timeseries_df = pd.DataFrame({

        "timestamp": t[1:],

        # Velocity
        "vx": vx,
        "vy": vy,

        # Head angles
        "alpha": alpha_ts,
        "beta": beta_ts,

        # Acceleration
        "acceleration": acc_full,
        "mean_acceleration": mean_acc_ts,

        # Distance metrics
        "euclidean_distance": ed,
        "cluster_distance": cluster_dist_ts,

        # Amplitude
        "amplitude": amp_ts,

        # Variability metrics
        "sigma_x": sigma_x_ts,
        "sigma_y": sigma_y_ts,

        # Rolling velocity summaries
        "mean_vx": mean_vx_ts,
        "mean_vy": mean_vy_ts

    })

    timeseries_path = (
            PROJECT_ROOT
            / "Results"
            / "AllData"
            / f"timeseries_results_P{P}.csv"
    )

    timeseries_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    timeseries_df.to_csv(
        timeseries_path,
        index=False
    )

    print(
        f"\nTime-series results saved to:\n"
        f"{timeseries_path}"
    )