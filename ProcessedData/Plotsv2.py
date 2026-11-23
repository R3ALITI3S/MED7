import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mad(x: np.ndarray) -> float:
  med = np.median(x)
  return np.median(np.abs(x - med))


def add_interval_spans(ax, intervals, color, alpha=0.18, label=None):
  first = True
  for start, end in intervals:
    if np.isfinite(start) and np.isfinite(end) and end >= start:
      ax.axvspan(
          start, end, color=color, alpha=alpha, label=label if first else None
      )
      first = False


def detect_blinks_from_zeros(df):
  """Detects blink/missing intervals based on where gaze coordinates are 0."""
  zero_mask = (df["gaze2d_x"] == 0) | (df["gaze2d_y"] == 0)
  if not zero_mask.any():
    return []

  df_temp = df.copy()
  df_temp["is_zero"] = zero_mask.astype(int)
  df_temp["block"] = (df_temp["is_zero"] != df_temp["is_zero"].shift()).cumsum()

  blink_intervals = []
  for _, group in df_temp[df_temp["is_zero"] == 1].groupby("block"):
    start = group["timestamp"].min()
    end = group["timestamp"].max()
    if end >= start:
      blink_intervals.append((float(start), float(end)))
  return blink_intervals


def detect_fixations_velocity(t, x, y, velocity, saccade_threshold):
  """Detects fixation intervals where velocity is below the saccade threshold."""
  is_fixation = velocity <= saccade_threshold
  fix_df = pd.DataFrame({"t": t, "is_fix": is_fixation.astype(int)})
  fix_df["block"] = (
      fix_df["is_fix"] != fix_df["is_fix"].shift()
  ).cumsum()

  fix_intervals = []
  for _, group in fix_df[fix_df["is_fix"] == 1].groupby("block"):
    if len(group) > 5:  # Minimum samples to qualify as a fixation
      start = group["t"].min()
      end = group["t"].max()
      if end > start:
        fix_intervals.append((float(start), float(end)))
  return fix_intervals


def parse_args():
  script_dir = Path(__file__).resolve().parent
  parser = argparse.ArgumentParser(
      description=(
          "Convert raw gaze CSV data to fixation/saccade visualizations."
      )
  )
  parser.add_argument(
      "--gaze",
      type=Path,
      default=script_dir / "Participant_6_GZ.csv", # CHANGE THIS
      help="Path to Participant_6_GZ_Fixed.csv",
  )
  parser.add_argument(
      "--fixations",
      type=Path,
      default=script_dir / "fixations.csv",
      help="Path to fixations.csv (optional)",
  )
  parser.add_argument(
      "--blinks",
      type=Path,
      default=script_dir / "blinks.csv",
      help="Path to blinks.csv (optional)",
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=script_dir / "outputs",
      help="Folder to save output plots",
  )
  return parser.parse_args()


def main():
  args = parse_args()

  gaze_path = args.gaze.expanduser().resolve()
  fix_path = args.fixations.expanduser().resolve()
  blink_path = args.blinks.expanduser().resolve()
  out_dir = args.out_dir.expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  if not gaze_path.exists():
    raise FileNotFoundError(f"Missing gaze file: {gaze_path}")

  # Load raw gaze data
  gaze_raw = pd.read_csv(gaze_path)
  required_cols = ["timestamp", "gaze2d_x", "gaze2d_y"]
  for col in required_cols:
    if col not in gaze_raw.columns:
      raise ValueError(f"Missing required column in gaze data: {col}")

  # 1. Extract blinks from zero values or external file
  if blink_path.exists():
    blinks = pd.read_csv(blink_path)
    blink_intervals = []
    if {"start_timestamp", "end_timestamp"}.issubset(blinks.columns):
      for _, r in (
          blinks[["start_timestamp", "end_timestamp"]].dropna().iterrows()
      ):
        blink_intervals.append(
            (float(r["start_timestamp"]), float(r["end_timestamp"]))
        )
  else:
    print(
        "Blink file not found. Auto-detecting blinks/zero values from dataset..."
    )
    blink_intervals = detect_blinks_from_zeros(gaze_raw)

  # Clean gaze data for trajectory/velocity analysis (filtering out 0 points)
  gaze = gaze_raw[
      (gaze_raw["timestamp"] > 0)
      & (gaze_raw["gaze2d_x"] != 0)
      & (gaze_raw["gaze2d_y"] != 0)
  ].copy()
  gaze = gaze.sort_values("timestamp").reset_index(drop=True)

  t = gaze["timestamp"].to_numpy()
  x = gaze["gaze2d_x"].to_numpy()
  y = gaze["gaze2d_y"].to_numpy()

  dt = np.diff(t)
  dx = np.diff(x)
  dy = np.diff(y)
  dist = np.sqrt(dx * dx + dy * dy)

  # Velocity calculation
  velocity = np.full_like(t, np.nan, dtype=float)
  valid_dt = dt > 0
  vel_core = np.full_like(dt, np.nan, dtype=float)
  vel_core[valid_dt] = dist[valid_dt] / dt[valid_dt]
  velocity[1:] = vel_core

  vel_valid = velocity[np.isfinite(velocity)]
  if len(vel_valid) > 0:
    vel_med = np.median(vel_valid)
    vel_mad = mad(vel_valid)
    saccade_threshold = vel_med + 6.0 * vel_mad
  else:
    saccade_threshold = 0.0
    vel_med = 0.0
    vel_mad = 0.0

  saccade_mask = velocity > saccade_threshold

  # 2. Extract fixations from external file or auto-detect via velocity
  if fix_path.exists():
    fix = pd.read_csv(fix_path)
    fix_intervals = []
    fix_cols = set(fix.columns)
    if {"start_timestamp", "end_timestamp"}.issubset(fix_cols):
      temp = fix[["start_timestamp", "end_timestamp"]].dropna()
      for _, r in temp.iterrows():
        fix_intervals.append(
            (float(r["start_timestamp"]), float(r["end_timestamp"]))
        )
    elif {"start_timestamp", "duration"}.issubset(fix_cols):
      temp = fix[["start_timestamp", "duration"]].dropna()
      for _, r in temp.iterrows():
        start = float(r["start_timestamp"])
        end = start + (float(r["duration"]) / 1000.0)
        fix_intervals.append((start, end))
  else:
    print(
        "Fixations file not found. Auto-detecting fixations using velocity"
        " threshold..."
    )
    fix_intervals = detect_fixations_velocity(
        t, x, y, velocity, saccade_threshold
    )

  # Figure 1: Raw gaze over time
  fig1, ax1 = plt.subplots(figsize=(14, 5))
  ax1.plot(t, x, ".", markersize=1.6, alpha=0.65, label="gaze2d_x")
  ax1.plot(t, y, ".", markersize=1.6, alpha=0.65, label="gaze2d_y")
  add_interval_spans(
      ax1, blink_intervals, color="red", alpha=0.25, label="Blink interval"
  )
  ax1.set_title("Raw Gaze Position Over Time (Participant 6)")
  ax1.set_xlabel("Timestamp")
  ax1.set_ylabel("Gaze Coordinate (2D)")
  ax1.legend(loc="upper right", frameon=False)
  ax1.grid(alpha=0.25)
  fig1.tight_layout()
  fig1.savefig(out_dir / "raw_gaze_over_time.png", dpi=180)
  plt.close(fig1)

  # Figure 2: Velocity with threshold and saccade points
  fig2, ax2 = plt.subplots(figsize=(14, 5))
  ax2.plot(t, velocity, linewidth=0.8, alpha=0.85, label="Velocity")
  ax2.axhline(
      saccade_threshold,
      linestyle="--",
      linewidth=1.3,
      color="darkorange",
      label="Saccade threshold",
  )
  ax2.scatter(
      t[saccade_mask],
      velocity[saccade_mask],
      s=4,
      alpha=0.7,
      label="Likely saccade samples",
  )
  add_interval_spans(
      ax2, blink_intervals, color="red", alpha=0.2, label="Blink interval"
  )
  ax2.set_title("Velocity Over Time")
  ax2.set_xlabel("Timestamp")
  ax2.set_ylabel("Velocity")
  ax2.legend(loc="upper right", frameon=False)
  ax2.grid(alpha=0.25)
  fig2.tight_layout()
  fig2.savefig(out_dir / "velocity_with_threshold.png", dpi=180)
  plt.close(fig2)

  # Figure 3: Fixation intervals timeline
  fig3, ax3 = plt.subplots(figsize=(14, 5))
  if fix_intervals:
    y0 = 0.5
    for i, (start, end) in enumerate(fix_intervals):
      y_level = y0 + (i % 35) * 0.22
      ax3.hlines(
          y=y_level,
          xmin=start,
          xmax=end,
          linewidth=1.8,
          alpha=0.8,
          color="tab:blue",
      )
  add_interval_spans(
      ax3, blink_intervals, color="red", alpha=0.2, label="Blink interval"
  )
  ax3.set_title("Detected Fixation Intervals")
  ax3.set_xlabel("Timestamp")
  ax3.set_ylabel("Fixation index (stacked)")
  ax3.grid(alpha=0.25)
  if fix_intervals:
    ax3.set_ylim(0, 8.5)
  fig3.tight_layout()
  fig3.savefig(out_dir / "fixation_intervals.png", dpi=180)
  plt.close(fig3)

  print(f"Gaze samples used: {len(gaze):,}")
  print(f"Fixation intervals found: {len(fix_intervals):,}")
  print(f"Blink intervals found: {len(blink_intervals):,}")
  print(f"Velocity median: {vel_med:.4f}")
  print(f"Velocity MAD: {vel_mad:.4f}")
  print(f"Saccade threshold: {saccade_threshold:.4f}")
  print(f"Likely saccade samples: {int(np.nansum(saccade_mask)):,}")
  print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
  main()