from pathlib import Path
import numpy as np
import pandas as pd
from ahrs.filters import Madgwick

base_folder = Path(__file__).resolve().parent.parent

file_path = (
    base_folder
    / "ProcessedData"
    / "IMUData"
    / "AccelGyro"
    / "Participant1_AccelGyro.csv"
)

data = pd.read_csv(file_path)

print(data.head())
print(data.shape)

timestamps = data["timestamp"].to_numpy(dtype=float)

accelerometer = data [
[ "accel_x", "accel_y", "accel_z" ]
].to_numpy(dtype=float)

gyroscope = data [
    [ "gyro_x", "gyro_y", "gyro_z" ]
].to_numpy(dtype=float)

gyroscope = np.deg2rad(gyroscope)

dt = np.diff(timestamps)

#Madgwick estimation orientation.
if len(timestamps) < 2:
    raise ValueError("At least two samples are required.")

if not np.all(np.isfinite(accelerometer)):
    raise ValueError("Invalid accelerometer values.") # Checks for Nan, inf, -inf

if not np.all(np.isfinite(gyroscope)):
    raise ValueError("Invalid gyroscope values.") # Checks for Nan, inf, -inf

if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
    raise ValueError("Timestamps must be finite and increasing.")

madgwick = Madgwick()

quaternions = np.zeros((len(timestamps), 4), dtype=float) # Creates array for the quaternions with a size of 4 for each timestamp

quaternions[0] = [1.0, 0.0, 0.0, 0.0] # Center point (no rotation)

for i in range(1, len(timestamps)): # Goes through every sample except the first (movement requires min two points)
    quaternions[i] = madgwick.updateIMU(
        q=quaternions[i - 1], # Uses previous orientation
        gyr=gyroscope[i], # Uses gyroscope (tells how fast the sensor rotates)
        acc=accelerometer[i], # Gravity reference (witch way is down)
        dt=dt[i - 1] # Time difference between the current and last sample
    ) # Returns a quaternion representing the orientation at sample 'i'

    if not np.all(np.isfinite(quaternions[i])):
        raise ValueError(f"Invalid quaternion at sample {i}")

# Saves the results in a table
result = pd.DataFrame({
    "timestamp": timestamps,
    "q_w": quaternions[:, 0],
    "q_x": quaternions[:, 1],
    "q_y": quaternions[:, 2],
    "q_z": quaternions[:, 3]
})

output_path = (
    base_folder
    / "ProcessedData"
    / "IMUData"
    / "Madgwick"
    / "Participant1_Orientation.csv"
)

result.to_csv(output_path, index=False)

print(result.head())
print(f"Saved to: {output_path}")