import json
import csv
from pathlib import Path

P = 5 # Participant number
AnoFrames = 1000 # Amount of frames annotated

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input and output file names
input_file = PROJECT_ROOT / "RawData" / "COCO" / f"COCO_Participant{P}.json"
output_file = PROJECT_ROOT / "Data" / f"COCO_Participant{P}.csv"
output_file.parent.mkdir(parents=True, exist_ok=True) # Create path if it doesn't exist

# Load JSON data
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Create lookup dictionaries
images = {img["id"]: img for img in data["images"]}
categories = {cat["id"]: cat["name"] for cat in data["categories"]}

# CSV column names
columns = [
    "annotation_id",
    "image_id",
    "file_name",
    "image_width",
    "image_height",
    "category_id",
    "category_name",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "area",
    "iscrowd",
    "occluded",
    "rotation",
    "track_id",
    "keyframe"
]

# Convert annotations into rows
rows = []

for ann in data["annotations"][:AnoFrames]:
    image = images.get(ann["image_id"], {})
    attributes = ann.get("attributes") or {}

    # Split bounding box into separate values
    bbox = ann.get("bbox", [None, None, None, None])

    row = {
        "annotation_id": ann.get("id"),
        "image_id": ann.get("image_id"),
        "file_name": image.get("file_name"),
        "image_width": image.get("width"),
        "image_height": image.get("height"),
        "category_id": ann.get("category_id"),
        "category_name": categories.get(ann.get("category_id")),
        "bbox_x": bbox[0],
        "bbox_y": bbox[1],
        "bbox_width": bbox[2],
        "bbox_height": bbox[3],
        "area": ann.get("area"),
        "iscrowd": ann.get("iscrowd"),
        "occluded": attributes.get("occluded"),
        "rotation": attributes.get("rotation"),
        "track_id": attributes.get("track_id"),
        "keyframe": attributes.get("keyframe")
    }

    rows.append(row)

# Write data to CSV
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV created successfully: {output_file}")
print(f"Number of rows: {len(rows)}")