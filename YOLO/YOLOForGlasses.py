
import asyncio
import time
import torch

from ultralytics import YOLO
import cv2
from g3pylib import connect_to_glasses

'''
This is maybe a cluster fuk, its a mix of The Connection for Glasses and the trained YOLO model in one.
There is probably one better way for doing this.
'''

# connect to the glasses
HOSTNAME = "192.168.75.51"

# Path to the pretrained model
MODEL_PATH = "my_training/custom_segmentation/weights/best.pt"

# YOLO configuration
IMAGE_SIZE = 640
CONFIDENCE = 0.5

# Use GPU if available, otherwise CPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("Loading custom YOLO segmentation model...")
model = YOLO(r"C:\Users\nield\Desktop\GitHub\MED7\runs\detect\MED7_training\robot_tv_detection\weights\best.pt")

# Just prints
print("Model loaded successfully! ;)")
print(f"Device: {DEVICE}")



async def camera_reader(video_stream, frame_queue):
    while True:
        frame, timestamp = await video_stream.get()
        image = frame.to_ndarray(format="bgr24")

        # Keep only the newest frame to reduce latency :)
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        await frame_queue.put((image, timestamp))

# the gaze reader, the red dot.
async def gaze_reader(gaze_stream, gaze_data):

    while True:

        gaze, gaze_timestamp = await gaze_stream.get()

        if gaze:
            gaze_data["gaze"] = gaze
            gaze_data["timestamp"] = gaze_timestamp


async def main():
    async with connect_to_glasses.with_hostname(HOSTNAME) as g3:
        print("=" * 70)
        print(f"Connected to Tobii Glasses 3: {HOSTNAME}")
        print("=" * 70)

        async with g3.stream_rtsp(
            scene_camera=True,
            gaze=True,
        ) as streams:

            async with streams.scene_camera.decode() as video_stream:

                async with streams.gaze.decode() as gaze_stream:

                    print("Starting real-time YOLO segmentation...")
                    print("Press Q to quit.")

                    # Shared data
                    frame_queue = asyncio.Queue(maxsize=1)

                    gaze_data = {
                        "gaze": None,
                        "timestamp": None
                    }

                    # Start independent capture tasks
                    camera_task = asyncio.create_task(
                        camera_reader(video_stream, frame_queue)
                    )

                    gaze_task = asyncio.create_task(
                        gaze_reader(gaze_stream, gaze_data)
                    )

                    # FPS calculation
                    fps = 0
                    frame_counter = 0
                    fps_start = time.monotonic()

                    try:

                        while True:

                            # Get newest camera frame
                            image, timestamp = await frame_queue.get()

                            # YOLO BLOCK

                            results = await asyncio.to_thread(
                                lambda: model.predict(
                                    source=image,
                                    imgsz=IMAGE_SIZE,
                                    conf=CONFIDENCE,
                                    device=DEVICE,
                                    verbose=False
                                )[0]
                            )

                            # Draw segmentation masks, bounding
                            # boxes, class labels and confidence.
                            annotated_image = results.plot()

                            # The gaze dot being shown.
                            current_gaze = gaze_data["gaze"]

                            height, width = annotated_image.shape[:2]

                            if current_gaze:

                                gaze2d = current_gaze.get("gaze2d")

                                if gaze2d is not None:

                                    gx, gy = gaze2d

                                    if 0 <= gx <= 1 and 0 <= gy <= 1:

                                        px = int(gx * width)
                                        py = int(gy * height)

                                        # Draw gaze point
                                        cv2.circle(
                                            annotated_image,
                                            (px, py),
                                            15,
                                            (0, 0, 255),
                                            -1
                                        )

                                        # Draw gaze label
                                        cv2.putText(
                                            annotated_image,
                                            f"Gaze: ({gx:.3f}, {gy:.3f})",
                                            (20, 40),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7,
                                            (0, 255, 0),
                                            2
                                        )

                            # FPS:
                            frame_counter += 1
                            elapsed = time.monotonic() - fps_start
                            if elapsed >= 1.0:

                                fps = frame_counter / elapsed

                                frame_counter = 0
                                fps_start = time.monotonic()

                            # INFORMATIONS:

                            cv2.putText(
                                annotated_image,
                                f"YOLO FPS: {fps:.1f}",
                                (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (255, 255, 0),
                                2
                            )

                            cv2.putText(
                                annotated_image,
                                f"Timestamp: {timestamp}",
                                (20, height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2
                            )

                            # Show the actual live video
                            cv2.imshow(
                                "Tobii Glasses 3 - YOLO Segmentation",
                                annotated_image
                            )

                            key = cv2.waitKey(1) & 0xFF

                            if key == ord("q"):
                                break

                    finally:

                        # Stop background tasks
                        camera_task.cancel()
                        gaze_task.cancel()

                        await asyncio.gather(
                            camera_task,
                            gaze_task,
                            return_exceptions=True
                        )

                        cv2.destroyAllWindows()

                        print("Real-time segmentation stopped.")


# HIT THE PEDAL
if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("Application interrupted.")