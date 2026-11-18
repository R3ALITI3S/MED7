import csv
from pathlib import Path

baseFolder = Path(__file__).resolve().parent.parent

input_file = baseFolder / "RawData" / "AOI" / "EyeTrackingAOI_1.tsv"
output_file = baseFolder / "ProcessedData" / "AOI_Hit" / "AOI_1_EyeTrackerOnly.csv"

# Columns to keep
columns_to_keep = [
    "Recording timestamp",
    "Sensor",
    "Eye movement type",
    "Eye movement event duration",
    "AOI hit [Participant 1 - Mirokai]",
    "AOI hit [Participant 1 - TV]"
]

with open(input_file, "r", encoding="utf-8-sig", newline="") as infile, \
     open(output_file, "w", encoding="utf-8", newline="") as outfile:

    reader = csv.DictReader(infile, delimiter="\t")

    writer = csv.DictWriter(
        outfile,
        fieldnames=columns_to_keep
    )

    writer.writeheader()

    for row in reader:
        if row["Sensor"] == "Eye Tracker":
            writer.writerow({
                col: row[col]
                for col in columns_to_keep
            })

print(f"Filtered file saved as: {output_file}")