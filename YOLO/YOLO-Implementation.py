from ultralytics import YOLO
import torch

print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

# Load pretrained YOLO26 detection model
model = YOLO("yolo26n.pt")

# Train using your MED7 dataset
results = model.train(
    data="/home/development/Desktop/MED7/MED7/YOLO/dataset/data.yaml",
    epochs=50,
    imgsz=1920,
    batch=8,
    workers=4,
    device="0",       # Use GPU; change to "cpu" if necessary
    patience=10,
    plots=True,
)

# Validate model
metrics = model.val()

# Evaluate on the separate test dataset
test_metrics = model.val(
    split="test"
)

print("Training completed!")
