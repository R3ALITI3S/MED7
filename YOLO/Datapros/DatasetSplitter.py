
import os
import shutil
import random

# Dataset paths
input_dir = r"/media/development/381C-5D24" # path change

image_dir = os.path.join(input_dir, "images")
label_dir = os.path.join(input_dir, "labels")

# Output directories
train_dir = os.path.join(input_dir, "train")
val_dir = os.path.join(input_dir, "val")
test_dir = os.path.join(input_dir, "test")

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.2

# Reproducibility
random.seed(42)

# Create output folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(input_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(input_dir, split, "labels"), exist_ok=True)

# Get image files
image_files = [
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Shuffle images
random.shuffle(image_files)

# Split dataset
n = len(image_files)

train_end = int(n * train_ratio)
val_end = int(n * (train_ratio + val_ratio))

train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

# Function to copy images and corresponding labels
def copy_files(files, split):
    for image in files:

        # Image path
        src_image = os.path.join(image_dir, image)

        # Corresponding label (.txt)
        label = os.path.splitext(image)[0] + ".txt"
        src_label = os.path.join(label_dir, label)

        # Destination paths
        dst_image = os.path.join(input_dir, split, "images", image)
        dst_label = os.path.join(input_dir, split, "labels", label)

        # Copy image
        shutil.copy2(src_image, dst_image)

        # Copy corresponding label if it exists
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        else:
            print(f"Warning: Label missing for {image}")

# Copy datasets
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("Dataset splitting completed!")
print("Train:", len(train_files))
print("Validation:", len(val_files))
print("Test:", len(test_files))