from ultralytics import YOLO

# Use our treined model
model = YOLO(
    "my_training/custom_segmentation/weights/best.pt"
)

results = model.predict(
    source="test.jpg",
    conf=0.25,
    save=True,
    show=True
)