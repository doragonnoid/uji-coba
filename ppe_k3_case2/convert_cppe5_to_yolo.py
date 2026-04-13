from pathlib import Path
import json
import shutil

PROJECT_ROOT = Path(r"D:\ppe_k3_case2_project\ppe_k3_case2")
RAW_ROOT = PROJECT_ROOT / "datasets" / "hospital_ppe" / "raw" / "cppe5_data"
ANN_ROOT = RAW_ROOT / "annotations"
IMG_ROOT = RAW_ROOT / "images"

OUT_ROOT = PROJECT_ROOT / "datasets" / "hospital_ppe"
IMG_OUT = OUT_ROOT / "images"
LBL_OUT = OUT_ROOT / "labels"

# 0: mask
# 1: gloves
# 2: gown_coverall
# 3: eye_protection
CATEGORY_ID_MAP = {
    1: 2,  # Coverall -> gown_coverall
    2: 3,  # Face_Shield -> eye_protection
    3: 1,  # Gloves -> gloves
    4: 3,  # Goggles -> eye_protection
    5: 0,  # Mask -> mask
}

def ensure_dirs():
    for split in ["train", "val", "test"]:
        (IMG_OUT / split).mkdir(parents=True, exist_ok=True)
        (LBL_OUT / split).mkdir(parents=True, exist_ok=True)

def clear_existing_outputs():
    for split in ["train", "val", "test"]:
        for f in (IMG_OUT / split).glob("*"):
            if f.is_file():
                f.unlink()
        for f in (LBL_OUT / split).glob("*"):
            if f.is_file():
                f.unlink()

def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    bw = w / img_w
    bh = h / img_h
    return x_center, y_center, bw, bh

def process_json(json_path: Path, split: str):
    print(f"[INFO] Proses {split}: {json_path.name}")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    images = {img["id"]: img for img in data.get("images", [])}

    anns_by_image = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    copied = 0

    for image_id, img_info in images.items():
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        src_img = IMG_ROOT / file_name
        if not src_img.exists():
            matches = list(IMG_ROOT.rglob(Path(file_name).name))
            if not matches:
                continue
            src_img = matches[0]

        label_lines = []
        for ann in anns_by_image.get(image_id, []):
            coco_cat_id = ann["category_id"]
            if coco_cat_id not in CATEGORY_ID_MAP:
                continue

            class_id = CATEGORY_ID_MAP[coco_cat_id]
            x_center, y_center, bw, bh = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)

            if bw <= 0 or bh <= 0:
                continue

            label_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
            )

        if not label_lines:
            continue

        dst_img = IMG_OUT / split / Path(file_name).name
        dst_lbl = LBL_OUT / split / f"{Path(file_name).stem}.txt"

        shutil.copy2(src_img, dst_img)
        dst_lbl.write_text("\n".join(label_lines), encoding="utf-8")
        copied += 1

    print(f"[INFO] {split}: copied={copied}")

def main():
    ensure_dirs()
    clear_existing_outputs()

    json_files = list(ANN_ROOT.glob("*.json"))
    print(f"[INFO] JSON ditemukan: {len(json_files)}")

    if not json_files:
        print("[ERROR] Tidak ada file JSON di folder annotations.")
        return

    for json_path in json_files:
        name = json_path.stem.lower()

        if "train" in name:
            split = "train"
        elif "val" in name:
            split = "val"
        elif "test" in name:
            split = "test"
        else:
            split = "train"

        process_json(json_path, split)

    for split in ["train", "val", "test"]:
        img_count = len(list((IMG_OUT / split).glob("*.*")))
        lbl_count = len(list((LBL_OUT / split).glob("*.txt")))
        print(f"[INFO] {split}: images={img_count}, labels={lbl_count}")

    print("[DONE] Konversi CPPE-5 -> YOLO selesai.")

if __name__ == "__main__":
    main()