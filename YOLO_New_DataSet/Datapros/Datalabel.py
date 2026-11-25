from pathlib import Path
import shutil

src = Path(r"/YOLO_New_DataSet/dataset/oldlables")
dst = Path(r"/YOLO_New_DataSet/dataset/lables")
dst.mkdir(parents=True, exist_ok=True)

for f in src.iterdir():
    if f.is_file():
        shutil.copy(f, dst / f"{f.stem}_P5{f.suffix}")

print("Done!")