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

    phi = left_dir_x[0]
    theta = left_dir_y[0]
    psi = left_dir_z[0]

    print("\n===== ROTATION MATRICES =====\n")

    print("Rx:")
    print(EyeTrackingFramework.Rx(phi))

    print("\nRy:")
    print(EyeTrackingFramework.Ry(theta))

    print("\nRz:")
    print(EyeTrackingFramework.Rz(psi))

    print("\nRotation Matrix:")
    print(
        EyeTrackingFramework.rotation_matrix(
            psi,
            theta,
            phi
        )
    )

    print("\n===== HEAD ANGLES =====\n")

    alpha = EyeTrackingFramework.alpha(
        psi,
        theta,
        phi
    )

    beta = EyeTrackingFramework.beta(
        theta,
        phi
    )

    print("Alpha:", alpha)
    print("Beta :", beta)

    print("\n===== VELOCITY =====\n")

    vx, vy = EyeTrackingFramework.velocity(
        x_g,
        y_g,
        dt
    )

    print("Mean vx:", np.mean(vx))
    print("Mean vy:", np.mean(vy))

    print("\n===== ACCELERATION =====\n")

    acc = EyeTrackingFramework.acceleration(
        vx,
        vy,
        dt
    )

    print("Mean acceleration:")
    print(np.mean(acc))

    print("\n===== AMPLITUDE =====\n")

    amp = EyeTrackingFramework.amplitude(
        x_g,
        y_g
    )

    print("Amplitude:")
    print(amp)

    print("\n===== EUCLIDEAN DISTANCE =====\n")

    ed = EyeTrackingFramework.eye_euclidean_distance(
        x_g,
        y_g
    )

    print(
        "Cluster distance:",
        EyeTrackingFramework.cluster_distance(ed)
    )

    print("\n===== STANDARD DEVIATION =====\n")

    sigma_x, sigma_y = (
        EyeTrackingFramework.gaze_std(
            x_g,
            y_g
        )
    )

    print("Sigma X:", sigma_x)
    print("Sigma Y:", sigma_y)

    # ==========================================================
    # Collect results
    # ==========================================================

    results = {
        "alpha": alpha,
        "beta": beta,
        "mean_vx": np.mean(vx),
        "mean_vy": np.mean(vy),
        "mean_acceleration": np.mean(acc),
        "amplitude": amp,
        "cluster_distance": EyeTrackingFramework.cluster_distance(ed),
        "sigma_x": sigma_x,
        "sigma_y": sigma_y
    }

    # ==========================================================
    # Export results as csv file
    # ==========================================================

    results_df = pd.DataFrame(
        results.items(),
        columns=["Metric", "Value"]
    )

    output_path = (
            PROJECT_ROOT
            / "Results"
            / "First_Row"
            / f"equation_results_P{P}.csv"
    )

    # Create folder if needed
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(f"\nResults saved to:\n{output_path}")

    # ==========================================================
    # Export velocity results between each sample as csv file
    # ==========================================================

    timeseries_df = pd.DataFrame({
        "timestamp": t[1:],
        "vx": vx,
        "vy": vy,

        "alpha": alpha,
        "beta": beta,
        "mean_vx": np.mean(vx),
        "mean_vy": np.mean(vy),
        "mean_acceleration": np.mean(acc),
        "amplitude": amp,
        "cluster_distance": EyeTrackingFramework.cluster_distance(ed),
        "sigma_x": sigma_x,
        "sigma_y": sigma_y
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