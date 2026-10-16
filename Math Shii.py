import math
import numpy as np
import pandas as pd
import os

# Equation for computing velocity
def calculate_velocity(x_g, y_g, dt):
    x_g = np.asarray(x_g, dtype=float)
    y_g = np.asarray(y_g, dtype=float)
    dt = np.asarray(dt, dtype=float) #Delta t

    v_x = (x_g[1:] - x_g[:-1]) / dt # selects all elements starting from index 1 (2nd element)
    v_y = (y_g[1:] - y_g[:-1]) / dt # selects all elements up to, but not including, the last element

    return v_x, v_y

# Equation for computing acceleration
def calculate_acceleration(v_x, v_y, dt):
    v_x = np.asarray(v_x, dtype=float)
    v_y = np.asarray(v_y, dtype=float)
    dt = np.asarray(dt[1:], dtype=float)  # acceleration uses dt between velocity samples

    a = np.sqrt(
        (v_x[1:] - v_x[:-1])**2 +
        (v_y[1:] - v_y[:-1])**2
    ) / dt

    return a

# Equation for computing amplitude
def calculate_amplitude(x_g, y_g):
    x_g = np.asarray(x_g, dtype=float)
    y_g = np.asarray(y_g, dtype=float)

    A = np.sqrt(
        (np.max(x_g) - np.min(x_g))**2 +
        (np.max(y_g) - np.min(y_g))**2
    )

    return A

# Get the fixed CSV file
content_root = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(content_root, "FixedData", "Participant_2_GZ_fixed.csv")

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

# Compute velocity, acceleration and amplitude
v_x, v_y = calculate_velocity(x_g, y_g, dt)
a = calculate_acceleration(v_x, v_y, dt)
A = calculate_amplitude(x_g, y_g)

# Prints out all the results
print("X velocity:", v_x)
print("Y velocity:", v_y)
print("Acceleration magnitude:", a)
print("Saccade amplitude (A):", A)

