import numpy as np
import pandas as pd
from pathlib import Path


# ==========================================================
# Load CSV
# ==========================================================

P = 1 # Participant number

PROJECT_ROOT = Path(__file__).resolve().parent.parent

csv_path = (
    PROJECT_ROOT
    / "FixedData"
    / f"Participant_{P}_GZ_fixed.csv"
)

df = pd.read_csv(csv_path)

# ==========================================================
# Load Madgwick filter
# ==========================================================

orientation_path = (
    PROJECT_ROOT
    / "ProcessedData"
    / "IMUData"
    / "Madgwick"
    / f"Participant{P}_Orientation.csv"
)

orientation_df = pd.read_csv(orientation_path)


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
    def quaternion_to_rotation_matrix(q):
        w, x, y, z = q

        return np.array([
            [1 - 2 * (y * y + z * z),
             2 * (x * y - z * w),
             2 * (x * z + y * w)],

            [2 * (x * y + z * w),
             1 - 2 * (x * x + z * z),
             2 * (y * z - x * w)],

            [2 * (x * z - y * w),
             2 * (y * z + x * w),
             1 - 2 * (x * x + y * y)]
        ])

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
    def velocity(world_vectors, dt):
        dot_products = np.sum(
            world_vectors[1:] *
            world_vectors[:-1],
            axis=1
        )

        dot_products = np.clip(
            dot_products,
            -1.0,
            1.0
        )

        angular_velocity = (
                np.arccos(dot_products)
                / dt
        )

        return angular_velocity

    @staticmethod
    def acceleration(v, dt):
        dv = np.diff(v)

        return dv / dt[1:]

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

    @staticmethod
    def interpolate_quaternions( # Interpolate quaternions to gaze timestamps
            target_t,
            source_t,
            quaternions
    ):
        interp_q = np.zeros(
            (len(target_t), 4),
            dtype=float
        )

        for j in range(4):
            interp_q[:, j] = np.interp(
                target_t,
                source_t,
                quaternions[:, j]
            )

        # Renormalize each quaternion
        norms = np.linalg.norm(
            interp_q,
            axis=1,
            keepdims=True
        )

        interp_q /= norms

        return interp_q


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

    # Euclidean distance between consecutive gaze points
    ed = EyeTrackingFramework.eye_euclidean_distance(
        x_g,
        y_g
    )

    imu_t = orientation_df["timestamp"].to_numpy(dtype=float)

    q_w = orientation_df["q_w"].to_numpy(dtype=float)
    q_x = orientation_df["q_x"].to_numpy(dtype=float)
    q_y = orientation_df["q_y"].to_numpy(dtype=float)
    q_z = orientation_df["q_z"].to_numpy(dtype=float)

    # Synchronizing quaternions
    quaternions = np.column_stack([
        q_w,
        q_x,
        q_y,
        q_z
    ])

    interp_quaternions = (
        EyeTrackingFramework.interpolate_quaternions(
            t,  # gaze timestamps
            imu_t,  # IMU timestamps
            quaternions
        )
    )

    # Rotate gaze to world coordinates
    eye_vectors = df[
        [
            "eyeleft.gazedirection_x",
            "eyeleft.gazedirection_y",
            "eyeleft.gazedirection_z"
        ]
    ].to_numpy()

    world_vectors = np.zeros_like(
        eye_vectors,
        dtype=float
    )

    for i in range(len(eye_vectors)):
        R = (
            EyeTrackingFramework
            .quaternion_to_rotation_matrix(
                interp_quaternions[i]
            )
        )

        world_vectors[i] = (
                R @ eye_vectors[i]
        )

        world_vectors[i] /= (
            np.linalg.norm(
                world_vectors[i]
            )
        )

    # Extract components
    world_gaze_x = world_vectors[:, 0]
    world_gaze_y = world_vectors[:, 1]
    world_gaze_z = world_vectors[:, 2]

    # Angular velocity
    v = EyeTrackingFramework.velocity(
        world_vectors,
        dt
    )

    # Angular acceleration
    acc = EyeTrackingFramework.acceleration(
        v,
        dt
    )

    acc_full = np.concatenate(([np.nan], acc))

    # Alpha/Beta for every sample
    alpha_ts = EyeTrackingFramework.alpha(
        world_gaze_z,
        world_gaze_y,
        world_gaze_x
    )

    beta_ts = EyeTrackingFramework.beta(
        world_gaze_y,
        world_gaze_x
    )

    # Align with velocity intervals
    alpha_ts = alpha_ts[1:]
    beta_ts = beta_ts[1:]

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

    mean_v_ts = (
        pd.Series(v)
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
    # EVENT DETECTION
    # ==========================================================

    # Angular velocity already computed from
    # Madgwick-corrected world gaze vectors

    speed_deg = np.degrees(v)

    # ----------------------------------------------------------
    # SACCADE DETECTION
    # ----------------------------------------------------------

    velocity_threshold = 30.0
    min_duration = 3

    saccade_mask = (
            speed_deg >
            velocity_threshold
    )

    labels = np.full(
        len(v),
        "Fixation",
        dtype=object
    )

    saccade_events = []

    start = None

    for i, val in enumerate(saccade_mask):

        if val and start is None:

            start = i

        elif not val and start is not None:

            end = i

            if end - start >= min_duration:
                saccade_events.append(
                    (start, end)
                )

            start = None

    # Handle saccade reaching end
    if start is not None:

        end = len(saccade_mask)

        if end - start >= min_duration:
            saccade_events.append(
                (start, end)
            )

    # Label saccades
    for start, end in saccade_events:
        labels[start:end] = "Saccade"

    # ==========================================================
    # Export all dynamic features
    # ==========================================================

    timeseries_df = pd.DataFrame({

        "timestamp": t[1:],

        "event": labels,

        # Velocity
        "angular_velocity": v,

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
        "mean_angular_velocity": mean_v_ts

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