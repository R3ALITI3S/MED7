import gzip                 # to open the compressed .gz file
import json                 # to turn each line of text into a Python dictionary
from pathlib import Path
import pandas as pd

#From page 87 in the ML book

P = 6 #Participant number

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def open_text(path):
    """Open a file as text, whether or not it is still gzip-compressed."""
    with open(path, "rb") as f:
        is_gzip = f.read(2) == b"\x1f\x8b"     # gzip files start with these two bytes
    if is_gzip:
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")

def load_gaze_data(root=PROJECT_ROOT / "RawData" / "GazeData", filename=f"Participant_{P}_GZ"):
    """Load every gazedata.gz found under `root` into one DataFrame.

            Each recording folder is treated as one participant, and the folder name
            is stored in the "participant" column.
            """
    frames = []# one DataFrame per recording, combined at the end


    # Search all subfolders of `root` for files named gazedata.gz
    for path in sorted(Path(root).rglob(filename)):
        # Open the .gz as text ("rt"). Each line is one gaze sample stored as JSON.
        with open_text(path) as f:
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
        df["participant"] = path.name
        frames.append(df)

    # Stop with a clear message if nothing was found (usually a wrong `root`)
    if not frames:
        raise FileNotFoundError(f"No {filename} found under {Path(root).resolve()}")

    # Stack all recordings into one table and sort by participant, then time
    data = pd.concat(frames, ignore_index=True)
    return data.sort_values(["participant", "timestamp"]).reset_index(drop=True)


OUTPUT_DIR = PROJECT_ROOT / "ProcessedData" # change the folder name if you prefer, e.g. FixedData


def flatten_lists(df):
    """Split every list column into one column per value (x, y, z, ...)."""
    out = df.copy()

    for col in list(out.columns):
        is_list = out[col].apply(lambda v: isinstance(v, list))
        if not is_list.any():
            continue

        n = out.loc[is_list, col].apply(len).max()          # longest list in this column
        names = ["x", "y", "z"] if n <= 3 else [str(i) for i in range(n)]

        for i in range(n):
            out[f"{col}_{names[i]}"] = out[col].apply(
                lambda v: v[i] if isinstance(v, list) and len(v) > i else None
            )
        out = out.drop(columns=[col])                        # remove the original list column

    return out


def save_csv(df, filename=f"Participant_{P}_GZ.csv"):
    """Save the table as a CSV with one value per column."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # gaze2d_x / gaze2d_y already exist, so the raw list column is redundant
    out = df.drop(columns=["data.gaze2d"], errors="ignore")
    out = flatten_lists(out)

    # Optional: shorter names, e.g. "data.eyeleft.pupildiameter" -> "eyeleft.pupildiameter"
    out.columns = [c.removeprefix("data.") for c in out.columns]

    # Anything still nested (unlikely) is stored as JSON text so the save never fails
    for col in out.columns:
        if out[col].apply(lambda v: isinstance(v, dict)).any():
            out[col] = out[col].apply(lambda v: json.dumps(v) if isinstance(v, dict) else v)

    path = OUTPUT_DIR / filename
    out.to_csv(path, index=False, encoding="utf-8")
    return path
# Only runs when you start this file directly, not when another script imports it
if __name__ == "__main__":
    gaze = load_gaze_data()
    out_path = save_csv(gaze)

    print(gaze.groupby("participant").size())       # number of samples per participant
    print(gaze.columns.tolist())                    # check the real column names
    print(f"Saved {len(gaze)} rows to {out_path}")