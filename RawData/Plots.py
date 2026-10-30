
"""
@author: Fjorda-inspired gaze analysis
Eye-tracking analysis using 3D gaze direction and pupil data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import stats

# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = r"C:\Users\nield\Desktop\GitHub\MED7\FixedData\Participant_1_GZ_fixed.csv"
OUTPUT_DIR = Path("gaze_analysis_plots")

# Set to None to analyze all participants.
PARTICIPANT = None

# Saccade detection threshold in degrees/second.
# This is a configurable heuristic, not a universal threshold.
SACCADE_THRESHOLD = 30.0

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

data = pd.read_csv(DATA_FILE)

required_columns = [
    "timestamp",
    "eyeleft.pupildiameter",
    "eyeright.pupildiameter",
    "gaze2d_x", "gaze2d_y",
    "gaze3d_x", "gaze3d_y", "gaze3d_z",
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
    "eyeright.gazedirection_z",
]

missing = [c for c in required_columns if c not in data.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Optional participant selection
if PARTICIPANT is not None:
    data = data[data["participant"] == PARTICIPANT].copy()

# Convert measurement columns to numeric
for col in required_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = (
    data.dropna(subset=["timestamp"])
    .sort_values("timestamp")
    .reset_index(drop=True)
)

if len(data) < 2:
    raise ValueError(
        "At least two timestamped samples are needed for velocity analysis."
    )

# Remove rows with duplicate timestamps.
# Velocity cannot be calculated when elapsed time is zero.
data = data.drop_duplicates(subset="timestamp", keep="first")
data = data.reset_index(drop=True)

if len(data) < 2:
    raise ValueError("At least two unique timestamps are required.")

# Relative time, in seconds if timestamp is recorded in seconds
t = data["timestamp"].to_numpy(dtype=float)
time = t - t[0]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(data.info())
print(f"\nSamples: {len(data)}")
print(f"Duration: {time[-1]:.3f} seconds")

# ============================================================
# FUNCTIONS
# ============================================================

def cart_to_spherical(x, y, z, apply_rad2deg=True):
    """
    Convert 3D direction vectors into:
        r     = vector magnitude
        theta = yaw angle around the vertical axis
        psi   = pitch angle

    Assumes forward is the positive Z direction.
    """
    r = np.sqrt(x**2 + y**2 + z**2)

    theta = np.arctan2(x, z)
    psi = np.arctan2(y, np.sqrt(x**2 + z**2))

    if apply_rad2deg:
        theta = np.rad2deg(theta)
        psi = np.rad2deg(psi)

    return r, theta, psi


def angular_velocity(t, theta, psi):
    """
    Calculate angular velocity in degrees/second.
    Handles angle wrapping at +/-180 degrees.
    """
    dt = np.diff(t)

    dtheta = np.diff(theta)
    dpsi = np.diff(psi)

    # Correct discontinuities caused by angle wrapping
    dtheta = (dtheta + 180) % 360 - 180
    dpsi = (dpsi + 180) % 360 - 180

    valid = dt > 0

    vx = np.full(len(dt), np.nan)
    vy = np.full(len(dt), np.nan)

    vx[valid] = dtheta[valid] / dt[valid]
    vy[valid] = dpsi[valid] / dt[valid]

    speed = np.sqrt(vx**2 + vy**2)

    return dt, vx, vy, speed


def plot_over_time(ts, series, ylabel, title):
    """Plot multiple signals over time."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for label, values in series.items():
        ax.plot(ts, values, label=label, linewidth=1)

    ax.set_xlabel("Time [sec]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def save_figure(fig, filename):
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")


# ============================================================
# 1. SPHERICAL COORDINATES FROM GAZE DIRECTION
# ============================================================

left_x = data["eyeleft.gazedirection_x"].to_numpy()
left_y = data["eyeleft.gazedirection_y"].to_numpy()
left_z = data["eyeleft.gazedirection_z"].to_numpy()

right_x = data["eyeright.gazedirection_x"].to_numpy()
right_y = data["eyeright.gazedirection_y"].to_numpy()
right_z = data["eyeright.gazedirection_z"].to_numpy()

left_r, left_theta, left_psi = cart_to_spherical(
    left_x, left_y, left_z
)

right_r, right_theta, right_psi = cart_to_spherical(
    right_x, right_y, right_z
)

# Use the average direction of the two eyes.
# Ignore invalid or zero-length direction vectors.
left_valid = (
    np.isfinite(left_x) & np.isfinite(left_y) &
    np.isfinite(left_z) & (left_r > 0)
)
right_valid = (
    np.isfinite(right_x) & np.isfinite(right_y) &
    np.isfinite(right_z) & (right_r > 0)
)

lx = np.where(left_valid, left_x / np.where(left_r > 0, left_r, 1), np.nan)
ly = np.where(left_valid, left_y / np.where(left_r > 0, left_r, 1), np.nan)
lz = np.where(left_valid, left_z / np.where(left_r > 0, left_r, 1), np.nan)

rx = np.where(right_valid, right_x / np.where(right_r > 0, right_r, 1), np.nan)
ry = np.where(right_valid, right_y / np.where(right_r > 0, right_r, 1), np.nan)
rz = np.where(right_valid, right_z / np.where(right_r > 0, right_r, 1), np.nan)

count = left_valid.astype(int) + right_valid.astype(int)

mean_x = np.nansum(np.column_stack([lx, rx]), axis=1)
mean_y = np.nansum(np.column_stack([ly, ry]), axis=1)
mean_z = np.nansum(np.column_stack([lz, rz]), axis=1)

mean_x = np.divide(mean_x, count, out=np.full(len(data), np.nan), where=count > 0)
mean_y = np.divide(mean_y, count, out=np.full(len(data), np.nan), where=count > 0)
mean_z = np.divide(mean_z, count, out=np.full(len(data), np.nan), where=count > 0)

_, theta, psi = cart_to_spherical(mean_x, mean_y, mean_z)

# ============================================================
# 2. ANGULAR VELOCITY AND SACCADE DETECTION
# ============================================================

dt, dtheta_dt, dpsi_dt, velocity = angular_velocity(
    t, theta, psi
)

velocity_time = (t[:-1] + t[1:]) / 2 - t[0]

valid_velocity = (
    np.isfinite(velocity) &
    np.isfinite(dtheta_dt) &
    np.isfinite(dpsi_dt)
)

saccade_mask = valid_velocity & (velocity >= SACCADE_THRESHOLD)

n_saccades = int(np.sum(saccade_mask))

if np.any(valid_velocity):
    mean_velocity = np.nanmean(velocity[valid_velocity])
    median_velocity = np.nanmedian(velocity[valid_velocity])
else:
    mean_velocity = np.nan
    median_velocity = np.nan

print("\n--- Gaze velocity analysis ---")
print(f"Candidate saccade intervals: {n_saccades}")
print(f"Mean angular velocity: {mean_velocity:.2f} deg/s")
print(f"Median angular velocity: {median_velocity:.2f} deg/s")
print(f"Detection threshold: {SACCADE_THRESHOLD:.1f} deg/s")

# ============================================================
# 3. SPHERICAL POSITION OVER TIME AND IN SPACE
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

axes[0].plot(time, theta, label="Yaw (theta)")
axes[0].plot(time, psi, label="Pitch (psi)")
axes[0].set_xlabel("Time [sec]")
axes[0].set_ylabel("Angle [degrees]")
axes[0].set_title("Spherical Gaze Coordinates Over Time")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Color gaze points by angular velocity of the following interval
if np.any(valid_velocity):
    scatter = axes[1].scatter(
        theta[:-1][valid_velocity],
        psi[:-1][valid_velocity],
        c=velocity[valid_velocity],
        cmap="viridis",
        s=12,
        alpha=0.7,
    )
    fig.colorbar(scatter, ax=axes[1], label="Velocity [deg/sec]")

    # Mark candidate saccade intervals
    mask = saccade_mask
    axes[1].scatter(
        theta[:-1][mask],
        psi[:-1][mask],
        color="red",
        s=18,
        label="Candidate saccade",
    )
    axes[1].legend()

axes[1].set_xlabel("Yaw [degrees]")
axes[1].set_ylabel("Pitch [degrees]")
axes[1].set_title("Gaze Direction in Angular Space")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
save_figure(fig, "01_spherical_gaze.png")

# ============================================================
# 4. GAZE VELOCITY OVER TIME AND HISTOGRAM
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(12, 9))

axes[0].plot(
    velocity_time,
    velocity,
    label="Angular gaze velocity",
    linewidth=0.8,
)
axes[0].axhline(
    SACCADE_THRESHOLD,
    color="red",
    linestyle="--",
    label="Saccade threshold",
)
axes[0].set_xlabel("Time [sec]")
axes[0].set_ylabel("Velocity [deg/sec]")
axes[0].set_title("Gaze Velocity Over Time")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

v = velocity[valid_velocity]

if len(v):
    axes[1].hist(v, bins=50)
    axes[1].axvline(
        SACCADE_THRESHOLD,
        color="red",
        linestyle="--",
        label="Saccade threshold",
    )
    axes[1].legend()

axes[1].set_xlabel("Angular velocity [deg/sec]")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Gaze Velocity Distribution")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
save_figure(fig, "02_gaze_velocity.png")

# ============================================================
# 5. 2D GAZE TRAJECTORY
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

gx = data["gaze2d_x"]
gy = data["gaze2d_y"]

ax.plot(gx, gy, linewidth=0.8, alpha=0.7, label="Gaze path")
ax.scatter(gx.iloc[0], gy.iloc[0], marker="o", label="Start")
ax.scatter(gx.iloc[-1], gy.iloc[-1], marker="x", label="End")

ax.set_xlabel("Gaze 2D X")
ax.set_ylabel("Gaze 2D Y")
ax.set_title("2D Gaze Trajectory")

# Uncomment if you want to highlight candidate saccades here too.
# Note: angular velocity is calculated from 3D direction, not 2D gaze.
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal", adjustable="datalim")

fig.tight_layout()
save_figure(fig, "03_2d_gaze_trajectory.png")

# ============================================================
# 6. 3D GAZE POINT OVER TIME AND IN SPACE
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(12, 9))

for col in ["gaze3d_x", "gaze3d_y", "gaze3d_z"]:
    axes[0].plot(time, data[col], label=col)

axes[0].set_xlabel("Time [sec]")
axes[0].set_ylabel("Position [dataset units]")
axes[0].set_title("3D Gaze Point Over Time")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

points = axes[1].scatter(
    data["gaze3d_x"],
    data["gaze3d_y"],
    c=data["gaze3d_z"],
    cmap="viridis",
    s=12,
)

fig.colorbar(points, ax=axes[1], label="Gaze Z")
axes[1].set_xlabel("Gaze 3D X")
axes[1].set_ylabel("Gaze 3D Y")
axes[1].set_title("3D Gaze Point: X-Y Projection")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
save_figure(fig, "04_3d_gaze_point.png")

# ============================================================
# 7. PUPIL DIAMETER
# ============================================================

fig = plot_over_time(
    time,
    {
        "Left pupil": data["eyeleft.pupildiameter"],
        "Right pupil": data["eyeright.pupildiameter"],
    },
    "Pupil diameter [dataset units]",
    "Pupil Diameter Over Time",
)

save_figure(fig, "05_pupil_diameter.png")

# ============================================================
# 8. GAZE ORIGIN: LEFT VS RIGHT EYE
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for axis, coord in zip(axes, ["x", "y", "z"]):
    axis.plot(
        time,
        data[f"eyeleft.gazeorigin_{coord}"],
        label=f"Left eye {coord.upper()}",
    )
    axis.plot(
        time,
        data[f"eyeright.gazeorigin_{coord}"],
        label=f"Right eye {coord.upper()}",
    )
    axis.set_ylabel(f"{coord.upper()} position")
    axis.legend()
    axis.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time [sec]")
fig.suptitle("Gaze Origin: Left and Right Eyes")
fig.tight_layout()
save_figure(fig, "06_gaze_origin.png")

# ============================================================
# 9. GAZE DIRECTION: LEFT VS RIGHT EYE
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for axis, coord in zip(axes, ["x", "y", "z"]):
    axis.plot(
        time,
        data[f"eyeleft.gazedirection_{coord}"],
        label=f"Left eye {coord.upper()}",
    )
    axis.plot(
        time,
        data[f"eyeright.gazedirection_{coord}"],
        label=f"Right eye {coord.upper()}",
    )
    axis.set_ylabel(f"{coord.upper()} direction")
    axis.legend()
    axis.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time [sec]")
fig.suptitle("Gaze Direction: Left and Right Eyes")
fig.tight_layout()
save_figure(fig, "07_gaze_direction.png")

# ============================================================
# 10. AMPLITUDE DISTRIBUTION AND DENSITY
# ============================================================

amplitude = np.sqrt(theta**2 + psi**2)
valid_amplitude = amplitude[np.isfinite(amplitude)]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

if len(valid_amplitude):
    axes[0].hist(valid_amplitude, bins=40, density=True)
    axes[0].set_xlabel("Angular displacement [degrees]")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Angular Displacement Distribution")

    # KDE requires multiple observations and nonzero variance
    if len(valid_amplitude) > 1 and np.std(valid_amplitude) > 0:
        kde = stats.gaussian_kde(valid_amplitude)
        grid = np.linspace(
            valid_amplitude.min(),
            valid_amplitude.max(),
            300,
        )
        axes[1].plot(grid, kde(grid))
        axes[1].set_xlabel("Angular displacement [degrees]")
        axes[1].set_ylabel("Probability density")
        axes[1].set_title("Amplitude Density (KDE)")
    else:
        axes[1].text(
            0.5, 0.5,
            "Not enough variation for KDE",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

for axis in axes:
    axis.grid(True, alpha=0.3)

fig.tight_layout()
save_figure(fig, "08_amplitude_density.png")

# ============================================================
# 11. EXPORT VELOCITY AND SACCADE LABELS
# ============================================================

# Velocity describes the interval between consecutive samples.
velocity_results = pd.DataFrame({
    "timestamp_start": t[:-1],
    "timestamp_end": t[1:],
    "time_relative_sec": velocity_time,
    "yaw_velocity_deg_sec": dtheta_dt,
    "pitch_velocity_deg_sec": dpsi_dt,
    "gaze_velocity_deg_sec": velocity,
    "candidate_saccade": saccade_mask,
})

velocity_results.to_csv(
    OUTPUT_DIR / "gaze_velocity_results.csv",
    index=False,
)

print(f"\nPlots and results saved in: {OUTPUT_DIR.resolve()}")
print("Analysis complete.")

plt.show()
