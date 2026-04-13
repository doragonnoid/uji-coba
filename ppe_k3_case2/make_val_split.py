from pathlib import Path
import random
import shutil

ROOT = Path(r"D:\ppe_k3_case2_project\ppe_k3_case2\datasets\hospital_ppe")

train_img = ROOT / "images" / "train"
train_lbl = ROOT / "labels" / "train"
val_img = ROOT / "images" / "val"
val_lbl = ROOT / "labels" / "val"

VAL_RATIO = 0.1
SEED = 42

val_img.mkdir(parents=True, exist_ok=True)
val_lbl.mkdir(parents=True, exist_ok=True)

image_files = sorted([p for p in train_img.glob("*") if p.is_file()])
random.seed(SEED)
random.shuffle(image_files)

n_val = int(len(image_files) * VAL_RATIO)
val_samples = image_files[:n_val]

moved = 0
for img_path in val_samples:
    lbl_path = train_lbl / f"{img_path.stem}.txt"
    if not lbl_path.exists():
        continue

    shutil.move(str(img_path), str(val_img / img_path.name))
    shutil.move(str(lbl_path), str(val_lbl / lbl_path.name))
    moved += 1

print(f"[DONE] moved to val: {moved}")
print(f"[INFO] train images left: {len(list(train_img.glob('*')))}")
print(f"[INFO] val images: {len(list(val_img.glob('*')))}")