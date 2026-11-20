
from ultralytics import YOLO

# Load pretrained YOLO26 detection model
model = YOLO("yolo26n.pt")

# Train using your MED7 dataset
results = model.train(
    data="RobotDataset/data.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    workers=4,
    device="cpu",       # Use GPU; change to "cpu" if necessary
    patience=20,
    project="MED7_training",
    name="robot_tv_detection",
    plots=True,
)

# Validate model
metrics = model.val()

# Evaluate on the separate test dataset
test_metrics = model.val(
    split="test"
)

print("Training completed!")
print("Best model saved at:")
print("MED7_training/robot_tv_detection/weights/best.pt")
