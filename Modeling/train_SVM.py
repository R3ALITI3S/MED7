import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

plot_data = df[["eyeleft.pupildiameter", "velocity_dps"]].dropna()
feature_columns = [
    "eyeleft.pupildiameter",
    "eyeright.pupildiameter",
    "velocity_dps",           # Gaze velocity
    "gaze2d_x",               # Gaze position on screen
    "gaze2d_y",
    "IMU_head_accel_x",       # Head movement
    "IMU_head_accel_y",
    "IMU_head_accel_z",
    "blink_duration",         # (If you have engineered this feature)
    "fixation_duration"       # (If you have engineered this feature)
]
X = df[feature_columns].to_numpy()
y = df["attention_label"].to_numpy()

svm = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", C=1) # Start with C=1, tune it later with GridSearch
)
svm.fit(X, y)


# Take a random sample of 2,000 rows just for the visualization
sample_df = df.sample(n=2000, random_state=42)
# Use ONLY TWO features here for the plot
X_2d = df[["eyeleft.pupildiameter", "velocity_dps"]]
y_2d = df["attention_label"]

svm_viz = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", C=1)
)
svm_viz.fit(X_2d, y_2d)

DecisionBoundaryDisplay.from_estimator(
        svm_viz,
        X_2d,
        response_method="predict",
        alpha=0.5,
        cmap="Pastel1",
        xlabel="Pupil diameter",
        ylabel="Gaze velocity (deg/s)",
    )

plt.scatter(
    X_2d.to_numpy() [:,0],
    X_2d.to_numpy()[:, 1],
    c=y_2d,
    s=20,
    edgecolors="k"
)

plt.title("SVM Decision Boundary for Attention")
plt.show()