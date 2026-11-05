import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

feature_columns = [
    "eyeleft.pupildiameter",
    "eyeright.pupildiameter",
    "velocity_dps",
    "gaze2d_x",
    "gaze2d_y",
    "IMU_head_accel_x",
    "IMU_head_accel_y",
    "IMU_head_accel_z",
    "blink_duration",
    "fixation_duration"
]

X = df[feature_columns].to_numpy()
y = df["attention_label"].to_numpy()
groups = df["participant"].to_numpy()

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

lr_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(random_state=42)
)

rf_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(random_state=42)
)

svc_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    SVC(random_state=42, probability=True)
)

folds = get_folds(df, group_col="participant", method="loso")

scores = []

for train_idx, test_idx in folds:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    voting_clf = VotingClassifier(
        estimators=[
            ('lr', lr_pipeline),
            ('rf', rf_pipeline),
            ('svc', svc_pipeline),
        ],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    score = voting_clf.score(X_test, y_test)
    scores.append(score)

    print(f"Held-out participant fold accuracy: {score:.3f}")

print(f"\nMean accuracy across all folds: {np.mean(scores):.3f}")
print(f"Std across folds: {np.std(scores):.3f}")