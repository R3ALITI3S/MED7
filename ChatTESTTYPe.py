import asyncio
import time
import csv
import json
from pprint import pprint
import os

from g3pylib import connect_to_glasses


# IP address / hostname of the G3 glasses.
HOSTNAME = "192.168.75.51"

# Folder where all participant CSVs will be saved.
OUTPUT_DIR = "Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Name of the CSV file that will be created.
test_number = 1
while os.path.exists(f"participant_{test_number}.csv"):
    test_number += 1
CSV_FILE = f"participant_{test_number}.csv"

async def main():

    async with connect_to_glasses.with_hostname(HOSTNAME) as g3:

        print("=" * 70)
        print(f"Connected to glasses: {HOSTNAME}")
        print("=" * 70)

        async with g3.stream_rtsp(scene_camera=True, gaze=True) as streams:

            async with streams.gaze.decode() as gaze_stream:

                start = time.monotonic()
                first_packet = True
                samples = 0

                # Open CSV file.
                with open(CSV_FILE, "w", newline="", encoding="utf-8") as csvfile:

                    writer = None

                    while time.monotonic() - start < 5:

                        gaze, timestamp = await gaze_stream.get()

                        samples += 1

                        if first_packet:
                            print("\nFIRST RAW GAZE PACKET:")
                            pprint(gaze, sort_dicts=False)
                            print("\n" + "-" * 70)

                            first_packet = False

                        print(f"\nSample #{samples}")
                        print(f"  timestamp : {timestamp}")

                        if not gaze:
                            print("  gaze      : None")
                            continue

                        print(f"  fields    : {list(gaze.keys())}")

                        # -----------------------------------------
                        # Create one CSV row from the gaze packet
                        # -----------------------------------------

                        row = {
                            "sample": samples,
                            "timestamp": timestamp,
                        }

                        # Go through every field returned by the glasses.
                        for key, value in gaze.items():

                            # gaze2d is usually a pair: [x, y]
                            if key == "gaze2d" and value is not None:
                                row["gaze2d_x"] = value[0]
                                row["gaze2d_y"] = value[1]

                            # Simple values can be stored directly.
                            elif isinstance(value, (int, float, str, bool)):
                                row[key] = value

                            # Lists / dictionaries / other structures
                            # are stored as JSON strings.
                            else:
                                row[key] = json.dumps(value)

                        # Create the CSV writer after seeing the
                        # first valid gaze packet.
                        if writer is None:
                            writer = csv.DictWriter(
                                csvfile,
                                fieldnames=row.keys()
                            )

                            writer.writeheader()

                        # Write the current gaze sample.
                        writer.writerow(row)

                        # Optional: force data to disk immediately.
                        csvfile.flush()

        print("\n" + "=" * 70)
        print(f"Done — {samples} samples received.")
        print(f"Data saved to: {CSV_FILE}")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())