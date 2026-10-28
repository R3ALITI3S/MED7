
import pandas as pd
import numpy as np
from pathlib import Path


# ==========================================
# 1. FILE PATHS
# ==========================================

baseFolder = Path(__file__).resolve().parent.parent

inputFolder = baseFolder / "RawData" / "IMUData"
outputFolder = baseFolder / "Results" / "IMU_Results"

processedFolder = baseFolder / "ProcessedData" / "IMUData"
accelGyroFolder = processedFolder / "AccelGyro"
magnetometerFolder = processedFolder / "Magnetometer"

for folder in [outputFolder, accelGyroFolder, magnetometerFolder]:
    folder.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. VALIDATE SENSOR MEASUREMENTS
# ==========================================

def checkSensor(dataframe, sensorName):

    def isValid(data):
        if not isinstance(data, dict):
            return False

        measurement = data.get(sensorName)

        if not isinstance(measurement, (list, tuple, np.ndarray)):
            return False

        if len(measurement) != 3:
            return False

        try:
            values = np.asarray(measurement, dtype=float)
            return bool(np.isfinite(values).all())
        except (ValueError, TypeError):
            return False

    return dataframe["data"].apply(isValid)


# ==========================================
# 3. ANALYZE ONE PARTICIPANT
# ==========================================

def analyzeParticipant(participantNumber):

    filePath = inputFolder / f"Participant{participantNumber}_IMUData"

    if not filePath.exists():
        print(f"Participant {participantNumber}: File not found")
        return None

    print(f"\n{'=' * 45}")
    print(f"ANALYZING PARTICIPANT {participantNumber}")
    print(f"{'=' * 45}")

    loadingFiles = pd.read_json(filePath, lines=True)

    # --------------------------------------
    # SEPARATE SENSOR STREAMS
    # --------------------------------------

    accelGyroData = loadingFiles[
        loadingFiles["data"].apply(
            lambda data: isinstance(data, dict)
            and "accelerometer" in data
            and "gyroscope" in data
        )
    ].sort_values("timestamp").copy()

    magnetometerData = loadingFiles[
        loadingFiles["data"].apply(
            lambda data: isinstance(data, dict)
            and "magnetometer" in data
        )
    ].sort_values("timestamp").copy()

    # --------------------------------------
    # VALIDATE SENSOR MEASUREMENTS
    # --------------------------------------

    accelValid = checkSensor(accelGyroData, "accelerometer")
    gyroValid = checkSensor(accelGyroData, "gyroscope")
    magValid = checkSensor(magnetometerData, "magnetometer")

    # --------------------------------------
    # CHECK TIMESTAMPS
    # --------------------------------------

    accelDiff = accelGyroData["timestamp"].diff()
    magDiff = magnetometerData["timestamp"].diff()

    normalInterval = magDiff.median()

    # --------------------------------------
    # FIND MAGNETOMETER GAPS
    # --------------------------------------

    gapThreshold = 0.2

    gapIndices = magDiff[magDiff > gapThreshold].index

    gapResults = pd.DataFrame({
        "Start": magnetometerData["timestamp"].shift(1).loc[gapIndices],
        "End": magnetometerData["timestamp"].loc[gapIndices],
        "Duration": magDiff.loc[gapIndices]
    })

    if pd.notna(normalInterval) and normalInterval > 0:
        gapResults["EstimatedMissing"] = (
            (gapResults["Duration"] / normalInterval).round() - 1
        ).clip(lower=0).astype(int)
    else:
        gapResults["EstimatedMissing"] = 0

    # --------------------------------------
    # CALCULATE GYROSCOPE ACTIVITY
    # --------------------------------------

    gyro = accelGyroData.loc[gyroValid].copy()

    if not gyro.empty:
        gyroValues = np.asarray(
            gyro["data"].apply(
                lambda data: data["gyroscope"]
            ).tolist(),
            dtype=float
        ).reshape(-1, 3)

        gyro["AngularSpeed"] = np.linalg.norm(
            gyroValues, axis=1
        )
    else:
        gyro["AngularSpeed"] = pd.Series(dtype=float)

    # --------------------------------------
    # COMPARE MOVEMENT DURING GAPS
    # --------------------------------------

    results = []

    for _, gap in gapResults.iterrows():

        measurements = gyro[
            (gyro["timestamp"] >= gap["Start"]) &
            (gyro["timestamp"] <= gap["End"])
        ]

        results.append({
            "Start": gap["Start"],
            "End": gap["End"],
            "Duration": gap["Duration"],
            "EstimatedMissing": gap["EstimatedMissing"],
            "GyroSamples": len(measurements),
            "MeanAngularSpeed": measurements["AngularSpeed"].mean(),
            "MaxAngularSpeed": measurements["AngularSpeed"].max()
        })

    movementResults = pd.DataFrame(
        results,
        columns=[
            "Start", "End", "Duration", "EstimatedMissing",
            "GyroSamples", "MeanAngularSpeed", "MaxAngularSpeed"
        ]
    )

    # ======================================
    # 4. EXPORT PROCESSED SENSOR DATA
    # ======================================

    # --------------------------------------
    # ACCELEROMETER + GYROSCOPE CSV
    # --------------------------------------

    accelGyroCSV = accelGyroData.loc[
        accelValid & gyroValid, ["timestamp", "data"]
    ].copy()

    accelerometerValues = np.asarray(
        accelGyroCSV["data"].apply(
            lambda data: data["accelerometer"]
        ).tolist(),
        dtype=float
    ).reshape(-1, 3)

    gyroscopeValues = np.asarray(
        accelGyroCSV["data"].apply(
            lambda data: data["gyroscope"]
        ).tolist(),
        dtype=float
    ).reshape(-1, 3)

    accelGyroCSV[
        ["accel_x", "accel_y", "accel_z"]
    ] = accelerometerValues

    accelGyroCSV[
        ["gyro_x", "gyro_y", "gyro_z"]
    ] = gyroscopeValues

    accelGyroCSV = accelGyroCSV.drop(columns=["data"])

    accelGyroCSV.to_csv(
        accelGyroFolder / f"Participant{participantNumber}_AccelGyro.csv",
        index=False
    )

    # --------------------------------------
    # MAGNETOMETER CSV
    # --------------------------------------

    magnetometerCSV = magnetometerData.loc[
        magValid, ["timestamp", "data"]
    ].copy()

    magnetometerValues = np.asarray(
        magnetometerCSV["data"].apply(
            lambda data: data["magnetometer"]
        ).tolist(),
        dtype=float
    ).reshape(-1, 3)

    magnetometerCSV[
        ["mag_x", "mag_y", "mag_z"]
    ] = magnetometerValues

    magnetometerCSV = magnetometerCSV.drop(columns=["data"])

    magnetometerCSV.to_csv(
        magnetometerFolder / f"Participant{participantNumber}_Magnetometer.csv",
        index=False
    )

    # ======================================
    # 5. EXPORT QUALITY ANALYSIS
    # ======================================

    gapResults.to_csv(
        outputFolder / f"Participant{participantNumber}_MagnetometerGaps.csv",
        index=False
    )

    movementResults.to_csv(
        outputFolder / f"Participant{participantNumber}_GapMovement.csv",
        index=False
    )

    # --------------------------------------
    # CREATE PARTICIPANT SUMMARY
    # --------------------------------------

    summary = {
        "Participant": participantNumber,
        "TotalRecords": len(loadingFiles),
        "AccelGyroRecords": len(accelGyroData),
        "MagnetometerRecords": len(magnetometerData),
        "InvalidAccelerometer": int((~accelValid).sum()),
        "InvalidGyroscope": int((~gyroValid).sum()),
        "InvalidMagnetometer": int((~magValid).sum()),
        "AccelGyroMinInterval": accelDiff.min(),
        "AccelGyroMaxInterval": accelDiff.max(),
        "MagnetometerMedianInterval": normalInterval,
        "MagnetometerMaxInterval": magDiff.max(),
        "MagnetometerGaps": len(gapResults),
        "EstimatedMissingMagnetometer": int(
            gapResults["EstimatedMissing"].sum()
        ),
        "OverallMeanAngularSpeed": gyro["AngularSpeed"].mean(),
        "MeanAngularSpeedDuringGaps": (
            movementResults["MeanAngularSpeed"].mean()
            if not movementResults.empty else np.nan
        ),
        "MaxAngularSpeedDuringGaps": (
            movementResults["MaxAngularSpeed"].max()
            if not movementResults.empty else np.nan
        )
    }

    print("Total records:", summary["TotalRecords"])
    print("Invalid accelerometer:", summary["InvalidAccelerometer"])
    print("Invalid gyroscope:", summary["InvalidGyroscope"])
    print("Invalid magnetometer:", summary["InvalidMagnetometer"])
    print("Magnetometer gaps:", summary["MagnetometerGaps"])
    print(
        "Estimated missing magnetometer samples:",
        summary["EstimatedMissingMagnetometer"]
    )
    print("Sensor CSV files exported.")

    return summary


# ==========================================
# 6. PROCESS ALL SIX PARTICIPANTS
# ==========================================

allResults = []

for participantNumber in range(1, 7):

    result = analyzeParticipant(participantNumber)

    if result is not None:
        allResults.append(result)


# ==========================================
# 7. EXPORT COMBINED SUMMARY
# ==========================================

summaryData = pd.DataFrame(allResults)

summaryData.to_csv(
    outputFolder / "AllParticipants_IMU_Summary.csv",
    index=False
)

print("\n=========================================")
print("ALL PARTICIPANTS ANALYZED")
print("=========================================")

if not summaryData.empty:
    print(
        summaryData[
            [
                "Participant",
                "TotalRecords",
                "InvalidAccelerometer",
                "InvalidGyroscope",
                "InvalidMagnetometer",
                "MagnetometerGaps",
                "EstimatedMissingMagnetometer"
            ]
        ].to_string(index=False)
    )
else:
    print("No participant files were found.")

print("\nProcessed CSV files saved in:", processedFolder)
print("Quality analysis saved in:", outputFolder)
print("\nIMU processing completed successfully.")
