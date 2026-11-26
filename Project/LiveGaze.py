
"""
Tobii Pro Glasses 3 + YOLO + Live Gaze + Attention Classification

Displays:
- Looking-at object and dwell time
- YOLO FPS
- Gaze coordinates
- Time spent on location
- Current attention class

IMPORTANT:
KMeans produces cluster IDs, not the semantic attention labels.
The default class names are placeholders MADS LOOK INTO THAT. Interpret the clusters before assigning names such as Attention high / attention low.
"""

import asyncio
import csv
import json
import logging
import math
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from g3pylib import connect_to_glasses

from train_attention import LiveAttention, load_model



# SETTINGS
HOSTNAME = "192.168.75.51"

YOLO_MODEL_PATH = (
    r"C:\Users\nield\Desktop\GitHub\MED7\runs\detect\train\weights\best.pt"
)

IMAGE_SIZE = 640
CONFIDENCE = 0.2
DEVICE = 0 if torch.cuda.is_available() else "cpu"

ATTENTION_MODEL_PATH = (
    r"C:\Users\nield\Desktop\GitHub\MED7\Project\attention_model.joblib"
)

# Change these names after interpreting the KMeans clusters.
CLUSTER_LABELS = {
    0: "Attention class 0",
    1: "Attention class 1",
    2: "Attention class 2",
    3: "Attention class 3",
}

RECORD_GAZE_TO = "gaze_session.jsonl"
LOG_CSV = "attention_log.csv"

GAZE_PADDING_PX = 25
GAZE_MAX_AGE_SECONDS = 0.30
MAX_DWELL_GAP_SECONDS = 0.20
SHOW_DEBUG_GAZE = True

WINDOW_NAME = "Tobii Glasses 3 - YOLO + Live Gaze + Attention"

# Video window dimensions
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
LOGGER = logging.getLogger("glasses_yolo_attention")



# GAZE HELPERS
def extract_gaze_sample(message):
    """Support both wrapped and unwrapped gaze messages."""

    if not isinstance(message, dict):
        return {}

    payload = message.get("data")

    if isinstance(payload, dict) and (
        "gaze2d" in payload or "gaze3d" in payload
    ):
        return payload

    return message


def parse_gaze2d(message):
    """Return normalized gaze coordinates or None."""

    sample = extract_gaze_sample(message)
    point = sample.get("gaze2d")

    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None

    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError):
        return None

    if not (
        math.isfinite(x)
        and math.isfinite(y)
        and 0.0 <= x <= 1.0
        and 0.0 <= y <= 1.0
    ):
        return None

    return x, y


def attention_class_name(attention):
    """Convert a KMeans cluster prediction into its display name."""

    if not isinstance(attention, dict):
        return "Waiting for prediction"

    cluster = attention.get("cluster")

    if cluster is None:
        return "Waiting for prediction"

    try:
        cluster = int(cluster)
    except (TypeError, ValueError):
        return "Invalid prediction"

    return CLUSTER_LABELS.get(
        cluster,
        f"Unknown cluster {cluster}"
    )



# STREAM READERS
async def camera_reader(video_stream, frame_queue):

    while True:
        frame, timestamp = await video_stream.get()

        image = frame.to_ndarray(format="bgr24")

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        await frame_queue.put((image, timestamp))


async def gaze_reader(
    gaze_stream,
    gaze_data,
    estimator,
    record_file
):

    while True:
        message, timestamp = await gaze_stream.get()

        if timestamp is None:
            continue

        sample = extract_gaze_sample(message)
        point = parse_gaze2d(message)

        gaze_data["point"] = point
        gaze_data["timestamp"] = timestamp
        gaze_data["received_monotonic"] = time.monotonic()

        if record_file:
            record_file.write(
                json.dumps(
                    {
                        "type": "gaze",
                        "timestamp": timestamp,
                        "data": sample,
                    },
                    default=str,
                ) + "\n"
            )

        try:
            attention_result = estimator.push(sample, timestamp)

            if attention_result is not None:
                gaze_data["attention"] = attention_result

        except Exception:
            LOGGER.exception(
                "LiveAttention failed while processing gaze"
            )



# YOLO OBJECT MATCHING
def class_name_for(results, class_id):

    names = results.names

    if isinstance(names, dict):
        return str(names.get(int(class_id), class_id))

    if isinstance(names, (list, tuple)):
        if 0 <= int(class_id) < len(names):
            return str(names[int(class_id)])

    return str(class_id)


def gazed_object(results, px, py, pad=GAZE_PADDING_PX):

    boxes = getattr(results, "boxes", None)

    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()

    best_hit = None
    best_area = float("inf")

    for index, (x1, y1, x2, y2) in enumerate(xyxy):

        inside = (
            x1 - pad <= px <= x2 + pad
            and y1 - pad <= py <= y2 + pad
        )

        if not inside:
            continue

        area = (
            max(0.0, float(x2 - x1))
            * max(0.0, float(y2 - y1))
        )

        if area < best_area:
            best_area = area

            best_hit = (
                class_name_for(results, classes[index]),
                float(confidences[index]),
            )

    return best_hit



# DWELL TRACKING
class DwellTracker:

    def __init__(self):
        self.current = None
        self.since = None
        self.last_t = None
        self.total = {}

    def update(self, target, now):

        if self.last_t is not None and self.current is not None:

            delta = now - self.last_t

            if 0.0 <= delta <= MAX_DWELL_GAP_SECONDS:
                self.total[self.current] = (
                    self.total.get(self.current, 0.0) + delta
                )

        if target != self.current:
            self.current = target
            self.since = now if target is not None else None

        self.last_t = now

    def current_dwell(self, now):

        if self.current is None or self.since is None:
            return 0.0

        return max(0.0, now - self.since)



# ATTENTION DISPLAY
def draw_attention(image, attention):
    """
    Display the current attention class on the visuals
    """

    label = f"Attention: {attention_class_name(attention)}"

    x = 20
    y = 165

    box_width = min(480, image.shape[1] - 40)
    box_height = 45

    cv2.rectangle(
        image,
        (x, y),
        (x + box_width, y + box_height),
        (40, 40, 40),
        -1
    )

    cv2.rectangle(
        image,
        (x, y),
        (x + box_width, y + box_height),
        (255, 255, 255),
        1
    )

    cv2.putText(
        image,
        label,
        (x + 10, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )



# GAZE DISPLAY
def draw_gaze(image, gaze_data, results, dwell, now):
    '''
    show the gaze point, so the dot.
    '''

    height, width = image.shape[:2]

    point = gaze_data.get("point")
    received = gaze_data.get("received_monotonic")

    target = None

    age = (
        now - received
        if received is not None
        else float("inf")
    )

    if point is not None and 0.0 <= age <= GAZE_MAX_AGE_SECONDS:

        gx, gy = point

        px = int(round(gx * (width - 1)))
        py = int(round(gy * (height - 1)))

        hit = gazed_object(results, px, py)

        if hit:
            target = hit[0]

        dot_color = (
            (0, 255, 0)
            if target
            else (0, 0, 255)
        )

        cv2.circle(image, (px, py), 15, dot_color, -1)
        cv2.circle(image, (px, py), 21, (255, 255, 255), 2)

        if SHOW_DEBUG_GAZE:

            cv2.putText(
                image,
                f"Gaze x={gx:.3f} y={gy:.3f} age={age * 1000:.0f}ms",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

    else:

        cv2.putText(
            image,
            "GAZE: NO FRESH VALID SAMPLE",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    dwell.update(target, now)

    if target:

        look_text = (
            f"Looking at: {target} "
            f"({dwell.current_dwell(now):.1f} s)"
        )

    else:
        look_text = "Looking at: -"

    cv2.putText(
        image,
        look_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

    return target



# MAIN APPLICATION
async def main():
    '''
    have all the path catches here, log it. & start recording.
    '''

    model_path = Path(YOLO_MODEL_PATH)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_MODEL_PATH}"
        )

    attention_path = Path(ATTENTION_MODEL_PATH)

    if not attention_path.is_file():
        raise FileNotFoundError(
            f"Attention model not found: {attention_path}"
        )

    # Load attention model
    attention_bundle = load_model(str(attention_path))
    estimator = LiveAttention(attention_bundle)

    LOGGER.info(
        "Loaded KMeans model with %s clusters",
        attention_bundle.get("n_clusters", "unknown")
    )

    # Load YOLO
    print("Loading YOLO model...")

    yolo = YOLO(str(model_path))

    print(f"YOLO loaded. Device: {DEVICE}")

    print(
        "Attention predictions appear after each complete "
        f"{float(attention_bundle['window_seconds']):.1f}-second window."
    )

    # Files
    record_file = (
        open(RECORD_GAZE_TO, "w", encoding="utf-8", buffering=1)
        if RECORD_GAZE_TO else None
    )

    log_file = (
        open(LOG_CSV, "w", newline="", encoding="utf-8")
        if LOG_CSV else None
    )

    csv_log = csv.writer(log_file) if log_file else None

    if csv_log:
        csv_log.writerow([
            "time",
            "cluster",
            "attention_class",
            "looking_at",
            "dwell_s",
            "gaze_x",
            "gaze_y",
        ])

    dwell = DwellTracker()

    last_logged_t = None

    camera_task = None
    gaze_task = None

    try:

        async with connect_to_glasses.with_hostname(HOSTNAME) as glasses:

            print("=" * 68)
            print(f"Connected to Tobii Glasses 3: {HOSTNAME}")
            print("=" * 68)

            async with glasses.stream_rtsp(
                scene_camera=True,
                gaze=True
            ) as streams:

                async with (
                    streams.scene_camera.decode() as video_stream,
                    streams.gaze.decode() as gaze_stream,
                ):

                    print("Streaming started. Press Q to quit.")

                    frame_queue = asyncio.Queue(maxsize=1)

                    gaze_data = {
                        "point": None,
                        "timestamp": None,
                        "received_monotonic": None,
                        "attention": None,
                    }

                    camera_task = asyncio.create_task(
                        camera_reader(video_stream, frame_queue)
                    )

                    gaze_task = asyncio.create_task(
                        gaze_reader(
                            gaze_stream,
                            gaze_data,
                            estimator,
                            record_file
                        )
                    )

                    fps = 0.0
                    frame_count = 0
                    fps_start = time.monotonic()

                    # Create a manually resizable video window.
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(
                        WINDOW_NAME,
                        WINDOW_WIDTH,
                        WINDOW_HEIGHT
                    )

                    try:

                        while True:

                            image, video_timestamp = await frame_queue.get()

                            prediction = await asyncio.to_thread(
                                lambda: yolo.predict(
                                    source=image,
                                    imgsz=IMAGE_SIZE,
                                    conf=CONFIDENCE,
                                    device=DEVICE,
                                    verbose=False,
                                )[0]
                            )

                            annotated = prediction.plot()

                            now = time.monotonic()

                            target = draw_gaze(
                                annotated,
                                gaze_data,
                                prediction,
                                dwell,
                                now
                            )

                            # Draw attention below the other information.
                            attention = gaze_data.get("attention")

                            draw_attention(
                                annotated,
                                attention
                            )

                            point = gaze_data.get("point")

                            gaze_x = point[0] if point else ""
                            gaze_y = point[1] if point else ""

                            # CSV logging
                            if (
                                csv_log
                                and isinstance(attention, dict)
                                and attention.get("t") is not None
                                and attention["t"] != last_logged_t
                            ):

                                last_logged_t = attention["t"]

                                cluster = attention.get("cluster")

                                try:
                                    cluster_id = int(cluster)
                                except (TypeError, ValueError):
                                    cluster_id = -1

                                csv_log.writerow([
                                    attention.get("t", ""),
                                    cluster if cluster is not None else "",
                                    CLUSTER_LABELS.get(
                                        cluster_id,
                                        "Unknown class"
                                    ),
                                    target or "",
                                    f"{dwell.current_dwell(now):.2f}",
                                    gaze_x,
                                    gaze_y,
                                ])

                                log_file.flush()

                            # FPS
                            frame_count += 1

                            elapsed = now - fps_start

                            if elapsed >= 1.0:

                                fps = frame_count / elapsed

                                frame_count = 0
                                fps_start = now

                            cv2.putText(
                                annotated,
                                f"YOLO FPS: {fps:.1f}",
                                (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 0),
                                2
                            )

                            # Dwell totals
                            totals = "  ".join(
                                f"{name}: {seconds:.1f}s"
                                for name, seconds in sorted(
                                    dwell.total.items()
                                )
                            )

                            if totals:

                                cv2.putText(
                                    annotated,
                                    f"Dwell: {totals}",
                                    (20, annotated.shape[0] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    2
                                )

                            cv2.imshow(WINDOW_NAME, annotated)

                            key = cv2.waitKey(1) & 0xFF

                            if key == ord("q") or key == 27:
                                break

                    finally:

                        for task in (camera_task, gaze_task):

                            if task is not None:
                                task.cancel()

                        await asyncio.gather(
                            *[
                                task
                                for task in (camera_task, gaze_task)
                                if task is not None
                            ],
                            return_exceptions=True
                        )

                        cv2.destroyAllWindows()

    finally:

        cv2.destroyAllWindows()

        for file_handle in (record_file, log_file):

            if file_handle is not None:
                file_handle.close()

        print("\nStopped. Total dwell time per detected class:")

        for name, seconds in sorted(
            dwell.total.items(),
            key=lambda item: -item[1]
        ):
            print(f"  {name:20s} {seconds:8.2f} s")

        if RECORD_GAZE_TO:
            print(
                f"Raw gaze saved to: "
                f"{Path(RECORD_GAZE_TO).resolve()}"
            )

        if LOG_CSV:
            print(
                f"Attention CSV saved to: "
                f"{Path(LOG_CSV).resolve()}"
            )


if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")

    except Exception:
        LOGGER.exception(
            "Application stopped because of an error."
        )
        raise
