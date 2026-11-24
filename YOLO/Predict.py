from ultralytics import YOLO

model = YOLO(
    "MED7_training/robot_tv_detection/weights/best.pt"
)

model.predict(
    source="Video_path",
    conf=0.4,
    save=True,
)