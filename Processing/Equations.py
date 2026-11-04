import numpy as np
import pandas as pd
from pathlib import Path


# ==========================================================
# Load CSV
# ==========================================================

P = 1 # Participant number

D_Screen = 3000 # Distance to screen in mm
D_Mirokai = 1500 # Distance to Mirokai in mm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

csv_path = (
    PROJECT_ROOT
    / "ProcessedData"
    / f"Participant_{P}_GZ.csv"
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
    def head_position_screen(alpha, beta, D_Screen):
        x_H = D_Screen * np.tan(alpha)
        y_H = D_Screen * np.tan(beta)
        return x_H, y_H

    @staticmethod
    def metric_to_pixels(
            x_H,
            y_H,
            screen_width_m,
            screen_height_m,
            x_res,
            y_res):
        x_px = x_H * (x_res / screen_width_m)
        y_px = y_H * (y_res / screen_height_m)

        return x_px, y_px

    @staticmethod
    def object_position(
            alpha_obj,
            beta_obj,
            D_obj):
        x_obj = D_obj * np.tan(alpha_obj)
        y_obj = D_obj * np.tan(beta_obj)

        return x_obj, y_obj

    @staticmethod
    def project_gaze_to_depth(
            gaze_x,
            gaze_y,
            gaze_z,
            depth):
        # Prevent division by zero
        z = np.where(
            np.abs(gaze_z) < 1e-6,
            np.nan,
            gaze_z
        )

        x = depth * (gaze_x / z)
        y = depth * (gaze_y / z)

        return x, y

    @staticmethod
    def point_inside_object(
            hit_points,
            center,
            width,
            height):
        local = hit_points - center

        x = local[:, 0]
        y = local[:, 1]

        inside = (
                (np.abs(x) <= width / 2)
                &
                (np.abs(y) <= height / 2)
        )

        return inside


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
    def object_euclidean_distance(
            x_obj,
            y_obj):
        dx = np.diff(x_obj)
        dy = np.diff(y_obj)

        return np.sqrt(dx ** 2 + dy ** 2)

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
    def local_plane_coordinates(
            hit_points,
            center,
            u,
            v):
        local = hit_points - center

        x_local = local @ u
        y_local = local @ v

        return x_local, y_local

    @staticmethod
    def inside_object(
            x_local,
            y_local,
            width,
            height):
        return (
                (np.abs(x_local) <= width / 2)
                &
                (np.abs(y_local) <= height / 2)
        )


# ==========================================================
# Run calculations
# ==========================================================

if __name__ == "__main__":

    window = 50

    # ======================================================
    # Convert gaze vector -> viewing angles
    # ======================================================

    alpha_ts = np.arctan2(
        left_dir_x,
        left_dir_z
    )

    beta_ts = np.arctan2(
        left_dir_y,
        left_dir_z
    )

    # ======================================================
    # Project gaze ray onto screen plane
    # ======================================================

    screen_x, screen_y = (
        EyeTrackingFramework.project_gaze_to_depth(
            left_dir_x,
            left_dir_y,
            left_dir_z,
            D_Screen
        )
    )

    screen_hits = np.column_stack([
        screen_x,
        screen_y,
        np.full_like(screen_x, D_Screen)
    ])

    # ======================================================
    # Project gaze ray onto object plane
    # ======================================================

    mirokai_x, mirokai_y = (
        EyeTrackingFramework.project_gaze_to_depth(
            left_dir_x,
            left_dir_y,
            left_dir_z,
            D_Mirokai
        )
    )

    mirokai_hits = np.column_stack([
        mirokai_x,
        mirokai_y,
        np.full_like(mirokai_x, D_Mirokai)
    ])

    # ======================================================
    # Eye-in-space coordinates (Eq11/12)
    # ======================================================

    x_G = np.nan_to_num(screen_x, nan=0.0) # Screen x value, with a safe where all NaN values get set to 0
    y_G = np.nan_to_num(screen_y, nan=0.0) # Screen y value, with a safe where all NaN values get set to 0

    # Align with velocity intervals
    alpha_ts = alpha_ts[1:]
    beta_ts = beta_ts[1:]

    vx, vy = EyeTrackingFramework.velocity(
        x_G,
        y_G,
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
        x_G,
        y_G
    )

    # ----------------------------------------------------------
    # Amplitude
    # ----------------------------------------------------------

    amp_ts = []

    for i in range(len(x_g)):
        start = max(0, i - window)

        x_window = x_G[start:i + 1]
        y_window = y_G[start:i + 1]

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
        pd.Series(x_G)
        .rolling(
            window=window,
            min_periods=1
        )
        .std()
        .to_numpy()[1:]
    )

    sigma_y_ts = (
        pd.Series(y_G)
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

    # ----------------------------------------------------------
    # Hit cords
    # ----------------------------------------------------------

    SCREEN = {
        "center": np.array([0, 0, 3000]),

        # facing towards participant
        "normal": np.array([0, 0, -1]),

        # local screen axes
        "u": np.array([1, 0, 0]),  # horizontal
        "v": np.array([0, 1, 0]),  # vertical

        "width": 1200,  # mm
        "height": 700  # mm
    }

    screen_local_x, screen_local_y = (
        EyeTrackingFramework.local_plane_coordinates(
            screen_hits,
            SCREEN["center"],
            SCREEN["u"],
            SCREEN["v"]
        )
    )

    screen_inside = (
        EyeTrackingFramework.inside_object(
            screen_local_x,
            screen_local_y,
            SCREEN["width"],
            SCREEN["height"]
        )
    )

    MIROKAI = {
        "center": np.array([1000, 0, 1500]),

        # example 45° rotation
        "normal": np.array([
            -0.7071,
            0,
            -0.7071
        ]),

        # local object axes
        "u": np.array([
            0.7071,
            0,
            -0.7071
        ]),
        "v": np.array([
            0,
            1,
            0
        ]),

        "width": 600,
        "height": 500
    }

    mirokai_local_x, mirokai_local_y = (
        EyeTrackingFramework.local_plane_coordinates(
            mirokai_hits,
            MIROKAI["center"],
            MIROKAI["u"],
            MIROKAI["v"]
        )
    )

    mirokai_inside = (
        EyeTrackingFramework.inside_object(
            mirokai_local_x,
            mirokai_local_y,
            MIROKAI["width"],
            MIROKAI["height"]
        )
    )

    # ==========================================================
    # Export all dynamic features
    # ==========================================================

    timeseries_df = pd.DataFrame({

        "timestamp": t[1:],

        "screen_x": screen_x[1:],
        "screen_y": screen_y[1:],

        "mirokai_x": mirokai_x[1:],
        "mirokai_y": mirokai_y[1:],

        "alpha": alpha_ts,
        "beta": beta_ts,

        "vx": vx,
        "vy": vy,

        "acceleration": acc_full,
        "mean_acceleration": mean_acc_ts,

        "euclidean_distance": ed,
        "cluster_distance": cluster_dist_ts,

        "amplitude": amp_ts,

        "sigma_x": sigma_x_ts,
        "sigma_y": sigma_y_ts,

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