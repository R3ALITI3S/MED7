from pathlib import Path
import pandas as pd

# Folder containing this script (Processing)
SCRIPT_DIR = Path(__file__).resolve().parent

# Parent folder containing Processing, Raw Data, and Fixed Data
PROJECT_DIR = SCRIPT_DIR.parent

RAW_DATA_DIR = PROJECT_DIR / "ProcessedData"
FIXED_DATA_DIR = PROJECT_DIR / "FixedData"

# Create the output folder if it doesn't exist
FIXED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Change this to the actual filename of your raw CSV
INPUT_FILE = RAW_DATA_DIR / "gaze_all.csv"

OUTPUT_FILE = FIXED_DATA_DIR / "gaze_data_fixed.csv"

EXPECTED_COLUMNS = [
    "type", "timestamp",
    "gaze2d_x", "gaze2d_y",
    "gaze3d_x", "gaze3d_y", "gaze3d_z",
]
for eye in ["eyeleft", "eyeright"]:
    for name in ["gazeorigin", "gazedirection"]:
        for axis in ["x", "y", "z"]:
            EXPECTED_COLUMNS.append(f"{eye}.{name}_{axis}")
    EXPECTED_COLUMNS.append(f"{eye}.pupildiameter")


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}\n"
        "Check the filename and the Raw Data folder."
    )

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} rows from {INPUT_FILE}")

for column in EXPECTED_COLUMNS:
    if column not in df.columns:
        df[column] = pd.NA
        print(f"Added missing column: {column}")

# Handle empty cells and common missing-value markers
df = df.fillna("null")

df.to_csv(OUTPUT_FILE, index=False)

print(f"Finished cleaning {len(df)} rows.")
print(f"Cleaned file saved to: {OUTPUT_FILE}")
