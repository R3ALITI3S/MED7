from ultralytics import YOLO

model = YOLO(
    "MED7_training/robot_tv_detection/weights/best.pt" # Path to the trained YOLO model
)

model.track(
    source = "Video_path",
    save = True,
    persist = True,
)