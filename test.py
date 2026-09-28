import asyncio
import time
from pprint import pprint

from g3pylib import connect_to_glasses


# IP address / hostname of the G3 glasses.
HOSTNAME = "192.168.75.51"


async def main():
    # Connect to the G3 glasses, receive gaze data for 5 seconds, and print the received gaze samples to the console.
    # Establish an asynchronous connection to the glasses.
    # The connection is automatically closed when leaving this block.
    async with connect_to_glasses.with_hostname(HOSTNAME) as g3:

        # Print a header to confirm that the connection was established.
        print("=" * 70)
        print(f"Connected to glasses: {HOSTNAME}")
        print("=" * 70)

        # Open the real-time RTSP streams.
        # scene_camera=False:
        #     We do not need the scene camera stream.
        # gaze=True:
        #     Enable the gaze data stream.
        async with g3.stream_rtsp(scene_camera=True, gaze=True) as streams:

            # Decode the incoming gaze stream into Python objects.
            async with streams.gaze.decode() as gaze_stream:

                # Record the start time so that we can stop after 5 seconds.
                start = time.monotonic()

                # Used to identify and print the first raw gaze packet.
                first_packet = True

                # Counter for the total number of gaze samples received.
                samples = 0

                # Continue receiving gaze data for approximately 5 seconds.
                while time.monotonic() - start < 5:

                    # Wait asynchronously for the next gaze packet.
                    # gaze:
                    #     Dictionary containing the gaze information.
                    # timestamp:
                    #     Timestamp associated with the received sample.
                    gaze, timestamp = await gaze_stream.get()

                    # Increment the number of received samples.
                    samples += 1

                    # Print the first raw packet in full.
                    # This is useful for discovering the structure and available fields in the gaze data.
                    if first_packet:
                        print("\nFIRST RAW GAZE PACKET:")
                        pprint(gaze, sort_dicts=False)
                        print("\n" + "-" * 70)

                        # Make sure this block only runs once.
                        first_packet = False

                    # Print basic information about the current sample.
                    print(f"\nSample #{samples}")
                    print(f"  timestamp : {timestamp}")

                    # A gaze packet may be empty or None.
                    # Handle that case before attempting to access its fields.
                    if not gaze:
                        print("  gaze      : None")
                        continue

                    # Display all available field names in the gaze packet.
                    print(f"  fields    : {list(gaze.keys())}")

                    # Try to retrieve the 2D gaze position.
                    # "gaze2d" commonly represents the normalized gaze
                    # coordinates, if that field is provided by the glasses.
                    gaze2d = gaze.get("gaze2d")

                    # Print the 2D gaze coordinates when available.
                    if gaze2d is not None:
                        print(f"  gaze2d    : {gaze2d}")
                    else:
                        print("  gaze2d    : None")

                    # Print every other field contained in the gaze packet.
                    # We skip "gaze2d" because it was already printed above.
                    # This also makes the script automatically display new fields if the glasses provide additional gaze data.
                    for key, value in gaze.items():
                        if key != "gaze2d":
                            print(f"  {key:<10}: {value}")

        # Print a summary after the gaze stream has finished.
        print("\n" + "=" * 70)
        print(f"Done — {samples} samples received.")
        print("=" * 70)


# Run the asynchronous main() function when this file is executed directly.
if __name__ == "__main__":
    asyncio.run(main())

