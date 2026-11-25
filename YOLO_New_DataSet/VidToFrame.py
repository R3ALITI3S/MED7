import cv2
from pathlib import Path
import shutil

P = 3  # Participant Number

PROJECT_ROOT = Path(__file__).resolve().parent

vid = cv2.VideoCapture(r"E:\Vid_P3.mp4")

output_folder = PROJECT_ROOT / "datasetdelete"
output_folder.mkdir(parents=True, exist_ok=True)

count, success = 0, True

while success:
    success, image = vid.read()
    if success:
        cv2.imwrite(
            str(output_folder / f"frame_{count:06d}_P{P}.jpg"),
            image
        )
        count += 1

vid.release()