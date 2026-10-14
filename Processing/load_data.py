import gzip                 # to open the compressed .gz file
import json                 # to turn each line of text into a Python dictionary
from pathlib import Path
import pandas as pd

#From page 87 in the ML book

def load_gaze_data(root=".", filename="gazedata.gz"):
    """Load every gazedata.gz found under `root` into one DataFrame.

            Each recording folder is treated as one participant, and the folder name
            is stored in the "participant" column.
            """
    frames = []# one DataFrame per recording, combined at the end


    # Search all subfolders of `root` for files named gazedata.gz
    for path in sorted(Path(root).rglob(filename)):
        # Open the .gz as text ("rt"). Each line is one gaze sample stored as JSON.
        with gzip.open(path, "rt", encoding="utf-8") as f:
            # json.loads turns a line of text into a dictionary.
            # Empty lines are skipped.
            rows = [json.loads(line) for line in f if line.strip()]

        # Flatten the nested dictionaries into a flat table.
        # Nested keys get joined with dots, e.g. "data.gaze2d".
        df = pd.json_normalize(rows)

        # gaze2d is a list [x, y], or missing when the eyes aren't tracked
        # (blink, glasses not worn). Check which rows hold a valid 2-number list.
        g = df["data.gaze2d"]
        ok = g.apply(lambda v: isinstance(v, list) and len(v) == 2)
        # Split the list into two columns. Rows without a valid gaze point become
        # NaN. We keep those rows on purpose, because blink detection and gap
        # handling later need to see where tracking was lost.
        df["gaze2d_x"] = g.where(ok).str[0]
        df["gaze2d_y"] = g.where(ok).str[1]

        # Use the recording folder's name as the participant ID
        df["participant"] = path.parent.name
        frames.append(df)

    # Stop with a clear message if nothing was found (usually a wrong `root`)
    if not frames:
        raise FileNotFoundError(f"No {filename} found under {Path(root).resolve()}")

    # Stack all recordings into one table and sort by participant, then time
    data = pd.concat(frames, ignore_index=True)
    return data.sort_values(["participant", "timestamp"]).reset_index(drop=True)

# Only runs when you start this file directly, not when another script imports it
if __name__ == "__main__":
    gaze = load_gaze_data()
    print(gaze.groupby("participant").size())       # number of samples per participant
    print(gaze.columns.tolist())                    # check the real column names