import asyncio
import time
from pprint import pprint
import cv2
from g3pylib import connect_to_glasses


# IP address / hostname of the G3 glasses.
HOSTNAME = "192.168.75.51"


async def main():
    async with connect_to_glasses.with_hostname(HOSTNAME) as g3:

        print("=" * 70)
        print(f"Connected to glasses: {HOSTNAME}")
        print("=" * 70)

        # Enable the scene camera and gaze streams.
        async with g3.stream_rtsp(
            scene_camera=True,
            gaze=True,
        ) as streams:

            # Decode the scene camera.
            async with streams.scene_camera.decode() as video_stream:

                # Decode gaze data.
                async with streams.gaze.decode() as gaze_stream:

                    print("Live feed started.")
                    print("Press Q in the video window to quit.")

                    latest_gaze = None
                    latest_gaze_timestamp = None

                    samples = 0
                    start = time.monotonic()

                    while True:

                        # Get the next camera frame
                        frame, timestamp = await video_stream.get()

                        # Convert the decoded frame to an OpenCV image.
                        image = frame.to_ndarray(format="bgr24")

                        # Don't want gaze reception to block the video.
                        try:
                            gaze, gaze_timestamp = await asyncio.wait_for(
                                gaze_stream.get(),
                                timeout=0.001,
                            )

                            if gaze:
                                latest_gaze = gaze
                                latest_gaze_timestamp = gaze_timestamp
                                samples += 1

                        except asyncio.TimeoutError:
                            pass

                        # Draw gaze information on the video - red dot
                        if latest_gaze:
                            gaze2d = latest_gaze.get("gaze2d")

                            if gaze2d is not None:
                                # Gaze coordinates are normally normalized:
                                # x = 0..1
                                # y = 0..1
                                gx, gy = gaze2d

                                height, width = image.shape[:2]

                                # Convert normalized coordinates to pixels.
                                px = int(gx * width)
                                py = int(gy * height)

                                # Draw a circle at the gaze position.
                                cv2.circle(
                                    image,
                                    (px, py),
                                    15,
                                    (0, 0, 255),
                                    -1,
                                )

                                # Draw a small label.
                                cv2.putText(
                                    image,
                                    f"Gaze: ({gx:.3f}, {gy:.3f})",
                                    (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 0),
                                    2,
                                )

                                cv2.putText(
                                    image,
                                    f"timestamps: ({timestamp})",
                                    (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 0),
                                    2,
                                )

                        # Display timestamp
                        cv2.putText(
                            image,
                            f"Timestamp: {timestamp}",
                            (20, image.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )

                        # Show live video
                        cv2.imshow("Glasses3 Live Feed", image)

                        # OpenCV requires waitKey for the window to update.
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord("q"):
                            break

                    cv2.destroyAllWindows()

                    print("\n" + "=" * 70)
                    print("Live feed stopped.")
                    print(f"Gaze samples received: {samples}")
                    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
