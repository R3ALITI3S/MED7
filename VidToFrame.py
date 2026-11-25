import cv2
from pathlib import Path
import shutil

P = 5 # Participant Number

PROJECT_ROOT = Path(__file__).resolve().parent

vid = cv2.VideoCapture(PROJECT_ROOT / "Video" / f"Vid_P{P}.mp4")

output_folder = PROJECT_ROOT / "YOLO_New_DataSet" / "Train" / "Frames"
output_folder.mkdir(exist_ok=True)

count, success = 0, True
while success:
    success, image = vid.read() # Read frame
    if success:
        cv2.imwrite(str(output_folder / f"frame_00{count}_P{P}.jpg"), image) # Save frame
        count += 1

vid.release()