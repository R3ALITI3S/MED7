import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap

"""
Blink detection from binocular pupil-diameter data.

Input CSV must contain:
  timestamp, eyeleft.pupildiameter, eyeright.pupildiameter

This is a heuristic detector, not a medically validated or video-verified
ground-truth blink detector. Tune thresholds against manually labelled data.
"""

input_path = Path(r"C:\Users\nield\Desktop\GitHub\MED7\FixedData\Participant_1_GZ_fixed.csv")
out_dir = Path(r"C:\Users\nield\Desktop\GitHub\MED7\BlinkScripts")
data = pd.read_csv(input_path)

# --- Blink detection ---
time = data["timestamp"].astype(float)
left = data["eyeleft.pupildiameter"].replace(0, np.nan)
right = data["eyeright.pupildiameter"].replace(0, np.nan)

# The recording is approximately 50 Hz. A 151-sample rolling median is
# approximately a 3-second local baseline.
baseline_left = left.rolling(151, center=True, min_periods=30).median()
baseline_right = right.rolling(151, center=True, min_periods=30).median()

ratio_left = (left / baseline_left).rolling(3, center=True, min_periods=1).median()
ratio_right = (right / baseline_right).rolling(3, center=True, min_periods=1).median()

# Candidate blink: both pupils fall below 80% of their local baselines,
# or both pupil measurements are missing.
candidate = (
    ((ratio_left < 0.80) & (ratio_right < 0.80))
    | (left.isna() & right.isna())
).fillna(False).to_numpy()

# Convert candidate samples into continuous intervals.
edges = np.diff(np.r_[False, candidate, False].astype(int))
starts = np.where(edges == 1)[0]
ends = np.where(edges == -1)[0]
raw_intervals = list(zip(starts, ends))

# Join intervals separated by at most 2 samples (~40 ms).
merged = []
for start_i, end_i in raw_intervals:
    if merged and start_i - merged[-1][1] <= 3:
        merged[-1] = (merged[-1][0], end_i)
    else:
        merged.append((start_i, end_i))

sample_period = float(np.median(np.diff(time)))
events = []
for start_i, end_i in merged:
    start_time = float(time.iloc[start_i])
    end_time = float(time.iloc[end_i - 1] + sample_period)
    duration = end_time - start_time
    # Keep plausible short closures; discard very brief noise and long data gaps.
    if 0.03 <= duration <= 0.30:
        events.append({
            "blink_number": len(events) + 1,
            "start_time_s": start_time,
            "end_time_s": end_time,
            "duration_s": duration,
        })

events_df = pd.DataFrame(events)
events_path = out_dir / "blink_events_participant1.csv"
events_df.to_csv(events_path, index=False)

recording_duration = float(time.iloc[-1] - time.iloc[0])
minutes = recording_duration / 60
blink_count = len(events_df)
blink_rate = blink_count / minutes if minutes else np.nan
mean_duration = events_df["duration_s"].mean() if blink_count else np.nan

# Plot a readable section of the pupil signal and detected events.
plot_end = min(60.0, float(time.iloc[-1])) # Minutes / 60.0 how long should we look
mask = time <= plot_end
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(time[mask], left[mask], label="Left pupil diameter", linewidth=0.8)
ax.plot(time[mask], right[mask], label="Right pupil diameter", linewidth=0.8)
for _, event in events_df.iterrows():
    if event["start_time_s"] <= plot_end:
        ax.axvspan(event["start_time_s"], min(event["end_time_s"], plot_end),
                   alpha=0.22)
ax.set(title="Detected blink candidates, first 60 seconds",
       xlabel="Time (seconds)", ylabel="Pupil diameter (source units)")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
plot_path = out_dir / "blink_detection_plot.png"
fig.savefig(plot_path, dpi=160)
plt.show()

print(f"Recording duration: {recording_duration/60:.2f} min")
print(f"Detected blink candidates: {blink_count}")
print(f"Blink rate: {blink_rate:.2f} blinks/min")
print(f"Mean detected duration: {mean_duration:.3f} s")
print(f"Saved: {events_path}")
print(f"Saved: {plot_path}")
