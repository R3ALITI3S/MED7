import math
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Equation for computing velocity
def calculate_velocity(x_g, y_g, dt):
    x_g = np.asarray(x_g, dtype=float)
    y_g = np.asarray(y_g, dtype=float)
    dt = np.asarray(dt, dtype=float) # Delta t

    # Equation 15 & 16
    v_x = (x_g[1:] - x_g[:-1]) / dt # selects all elements starting from index 1 (2nd element)
    v_y = (y_g[1:] - y_g[:-1]) / dt # selects all elements up to, but not including, the last element

    return v_x, v_y

# Equation for computing acceleration
def calculate_acceleration(v_x, v_y, dt):
    v_x = np.asarray(v_x, dtype=float)
    v_y = np.asarray(v_y, dtype=float)
    dt = np.asarray(dt[1:], dtype=float)  # acceleration uses dt between velocity samples

    # Equation 17
    a = np.sqrt(
        (v_x[1:] - v_x[:-1])**2 +
        (v_y[1:] - v_y[:-1])**2
    ) / dt

    return a

# Equation for computing amplitude
def calculate_amplitude(x_g, y_g):
    x_g = np.asarray(x_g, dtype=float)
    y_g = np.asarray(y_g, dtype=float)

    # Equation 18
    A = np.sqrt(
        (np.max(x_g) - np.min(x_g))**2 +
        (np.max(y_g) - np.min(y_g))**2
    )

    return A

# ---- The adaptive threshold (Engbert & Kliegl, 2003) ---
def adaptive_threshold(v_x, v_y, lam=6):
    """Compute velocity thresholds from the data's own noise level.

    sigma_x/y estimate how much the velocity naturally jitters, even
    during fixations (eye tremor + sensor noise). It's a robust estimate
    (based on the median, not the mean), so it isn't thrown off much by
    the real saccades that are mixed into the same data.

    lam (lambda) sets how many sigmas above that jitter counts as a real
    saccade. 6 is the paper's default: if the noise is roughly
    bell-curve-shaped, a value 6 sigma away is astronomically unlikely to
    be random noise, so it's treated as a genuine fast eye movement.
    """
    sigma_x = np.sqrt(np.nanmedian(v_x ** 2) - np.nanmedian(v_x) ** 2)
    sigma_y = np.sqrt(np.nanmedian(v_y ** 2) - np.nanmedian(v_y) ** 2)

    threshold_x = lam * sigma_x
    threshold_y = lam * sigma_y
    return threshold_x, threshold_y, sigma_x, sigma_y


def find_saccade_candidates(v_x, v_y, threshold_x, threshold_y):
    """Flags each sample as a saccade candidate (True) or not (False).

    Uses an elliptical boundary rather than checking x and y separately:
    a sample counts as a candidate if (vx/threshold_x)^2 + (vy/threshold_y)^2
    is greater than 1, i.e. it falls outside the ellipse formed by the two
    thresholds. This lets x and y contribute jointly, instead of either one
    alone being enough to trigger a flag."""
    with np.errstate(invalid="ignore", divide="ignore"):
        is_candidate = (v_x / threshold_x) ** 2 + (v_y / threshold_y) ** 2 > 1
    return np.nan_to_num(is_candidate, nan=False).astype(bool)


def group_into_events(is_candidate):
    """Turns the sample-by-sample True/False array into a list of
    (start, end) index pairs — one per saccade candidate.

    Needed because calculate_amplitude and calculate_acceleration should
    run on ONE saccade at a time, not on the whole recording at once.
    Without this step we'd only have isolated flagged samples, not
    coherent events with a start and an end."""
    groups = []
    in_run = False
    start = 0
    for i, flag in enumerate(is_candidate):
        if flag and not in_run:
            start = i
            in_run = True
        elif not flag and in_run:
            groups.append((start, i))
            in_run = False
    if in_run:
        groups.append((start, len(is_candidate)))
    return groups


# Build the path to the fixed CSV, relative to this script's location,
# so it works regardless of which machine or folder structure runs it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
csv_path = PROJECT_ROOT / "FixedData" / "Participant_1_GZ_fixed.csv"

df = pd.read_csv(csv_path)

# This fills missing values using the nearest valid samples (takes the mean of value before and after)
df["gaze2d_x"] = df["gaze2d_x"].interpolate(method="linear")
df["gaze2d_y"] = df["gaze2d_y"].interpolate(method="linear")
df["timestamp"] = df["timestamp"].interpolate(method="linear")

# Debug type shit
#print(df.columns)
#print("Last 10 timestamps:", df["timestamp"].tail(10).values)
#print("Last 10 dt:", np.diff(df["timestamp"].tail(11).values))

# Adjust column names if needed
x_g = df["gaze2d_x"].values
y_g = df["gaze2d_y"].values
t = df["timestamp"].values # must be in seconds

# Compute dt from timestamps
dt = np.diff(t)

# RUN THE PIPELINE: velocity -> threshold -> candidates -> events
# Compute velocity, threshold, acceleration and amplitude

# Step 1: velocity at every sample
v_x, v_y = calculate_velocity(x_g, y_g, dt)

# Step 2: figure out this recording's own noise level, and set the
# threshold relative to it (rather than using one fixed number for
# every recording)
threshold_x, threshold_y, sigma_x, sigma_y = adaptive_threshold(v_x, v_y, lam=6)

print(f"sigma_x = {sigma_x:.4f}, sigma_y = {sigma_y:.4f}")
print(f"threshold_x = {threshold_x:.4f}, threshold_y = {threshold_y:.4f}")

# Step 3: flag every sample that exceeds the threshold
is_candidate = find_saccade_candidates(v_x, v_y, threshold_x, threshold_y)
print(f"Saccade candidate samples: {is_candidate.sum()} / {len(is_candidate)} "
      f"({is_candidate.mean() * 100:.1f}%)")

# Step 4: group consecutive flagged samples into individual saccade events
events = group_into_events(is_candidate)
print(f"\nFound {len(events)} saccade candidates")

# Step 5: for each saccade event, compute its own amplitude and peak
# acceleration (not one number for the whole file)
for start, end in events[:10]:          # print the first 10 as a sanity check
    amp = calculate_amplitude(x_g[start:end + 1], y_g[start:end + 1])
    seg_vx, seg_vy = v_x[start:end], v_y[start:end]
    seg_dt = dt[start:end]

    # Acceleration needs at least 2 velocity values to compute a
    # difference; very short candidates (1 sample interval) can't have
    # one, so they'll show up as "nan" below — not a bug, just a limit
    # of the math (see calculate_acceleration's docstring).
    a = calculate_acceleration(seg_vx, seg_vy, seg_dt) if len(seg_vx) > 1 else np.array([])
    peak_a = a.max() if len(a) else float("nan")
    print(f"  t={t[start]:.3f}-{t[end]:.3f}s  duration={t[end]-t[start]:.3f}s  "
          f"amplitude={amp:.4f}  peak_accel={peak_a:.1f}")

