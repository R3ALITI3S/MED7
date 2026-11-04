
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# Locate project folder
base_folder = Path(__file__).resolve().parent.parent.parent

file_path = (
    base_folder
    / "ProcessedData"
    / "IMUData"
    / "Madgwick"
    / "Participant1_Orientation.csv"
)

data = pd.read_csv(file_path)

timestamps = data["timestamp"].to_numpy(dtype=float)

quaternions = data[
    ["q_w", "q_x", "q_y", "q_z"]
].to_numpy(dtype=float)

# 1. Check quaternion normalization
norms = np.linalg.norm(quaternions, axis=1)

print("QUATERNION NORMALIZATION")
print("Minimum norm:", norms.min())
print("Maximum norm:", norms.max())
print("Mean norm:", norms.mean())

# 2. Calculate orientation changes
# SciPy expects [x, y, z, w], not [w, x, y, z]
quaternions_xyzw = quaternions[:, [1, 2, 3, 0]]

rotations = Rotation.from_quat(quaternions_xyzw)

# Relative rotation between consecutive measurements
relative_rotations = rotations[:-1].inv() * rotations[1:]

# Magnitude of each rotation in degrees
angular_changes = np.rad2deg(relative_rotations.magnitude())

# Time between consecutive measurements
dt = np.diff(timestamps)

angular_speeds = angular_changes / dt

print("\nORIENTATION CHANGES")
print("Mean angular change:", angular_changes.mean(), "degrees")
print("Maximum angular change:", angular_changes.max(), "degrees")

print("\nESTIMATED ANGULAR SPEED")
print("Mean:", angular_speeds.mean(), "degrees/second")
print("Maximum:", angular_speeds.max(), "degrees/second")

# 3. Check for unusually large changes
threshold = 500.0  # Screening threshold, not a validated human limit

unusual = np.where(angular_speeds > threshold)[0]

print("\nPOTENTIALLY UNUSUAL MOVEMENTS")
print("Samples above threshold:", len(unusual))
print("Percentage:", 100 * len(unusual) / len(angular_speeds))




# Load the original gyroscope measurements
gyro_path = (
    base_folder
    / "ProcessedData"
    / "IMUData"
    / "AccelGyro"
    / "Participant1_AccelGyro.csv"
)

gyro_data = pd.read_csv(gyro_path)

# Extract gyroscope timestamps
gyro_timestamps = gyro_data["timestamp"].to_numpy(dtype=float)

# Extract gyroscope measurements in degrees/second
gyroscope = gyro_data[
    ["gyro_x", "gyro_y", "gyro_z"]
].to_numpy(dtype=float)

# Confirm both files contain the same samples
if len(gyro_timestamps) != len(timestamps):
    raise ValueError("The files contain different numbers of samples.")

if not np.allclose(
    gyro_timestamps, timestamps, rtol=0, atol=1e-6
):
    raise ValueError("The timestamps do not match.")

# Calculate gyroscope angular speed
gyro_speed = np.linalg.norm(gyroscope, axis=1)

# Align gyroscope samples with quaternion changes
gyro_speed = gyro_speed[1:]

# Compare the two angular-speed estimates
difference = angular_speeds - gyro_speed

print("\nGYROSCOPE VS MADGWICK")

print("Mean gyroscope speed:", gyro_speed.mean(), "°/s")
print("Mean Madgwick speed:", angular_speeds.mean(), "°/s")

print("Mean absolute difference:",
      np.mean(np.abs(difference)), "°/s")

print("Maximum absolute difference:",
      np.max(np.abs(difference)), "°/s")
print("Still working")

# Calculate Pearson correlation manually

gyro_mean = np.mean(gyro_speed)
madgwick_mean = np.mean(angular_speeds)

gyro_centered = gyro_speed - gyro_mean
madgwick_centered = angular_speeds - madgwick_mean

numerator = np.sum(gyro_centered * madgwick_centered)

denominator = np.sqrt(
    np.sum(gyro_centered ** 2)
    * np.sum(madgwick_centered ** 2)
)

if denominator == 0:
    print("Correlation cannot be calculated.")
else:
    correlation = numerator / denominator
    print("Correlation:", correlation)

# Define the first and last 10 seconds
start_time = gyro_timestamps[0]
end_time = gyro_timestamps[-1]

first_10 = gyroscope[gyro_timestamps < start_time + 10.0]
last_10 = gyroscope[gyro_timestamps >= end_time - 10.0]

# Calculate angular speed magnitude
first_speeds = np.linalg.norm(first_10, axis=1)
last_speeds = np.linalg.norm(last_10, axis=1)

print("\nFIRST 10 SECONDS")
print("Mean:", np.mean(first_speeds), "°/s")
print("Minimum:", np.min(first_speeds), "°/s")
print("Maximum:", np.max(first_speeds), "°/s")

print("\nLAST 10 SECONDS")
print("Mean:", np.mean(last_speeds), "°/s")
print("Minimum:", np.min(last_speeds), "°/s")
print("Maximum:", np.max(last_speeds), "°/s")


print("\nLAST 10 SECONDS - ONE SECOND WINDOWS")

for second in range(10, 0, -1):

    window_start = end_time - second
    window_end = window_start + 1.0

    mask = (
        (gyro_timestamps >= window_start)
        & (gyro_timestamps < window_end)
    )

    window_data = gyroscope[mask]

    if len(window_data) == 0:
        continue

    window_speeds = np.linalg.norm(window_data, axis=1)

    print(
        f"{second:2d}-{second-1:2d} seconds before end:",
        f"Mean = {np.mean(window_speeds):.3f} °/s",
        f"Max = {np.max(window_speeds):.3f} °/s"
    )