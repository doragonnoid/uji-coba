from pathlib import Path

ROOT = Path(".")
IMG_ROOT = ROOT / "images"
SRC_LABEL_ROOT = ROOT / "labels_raw11_backup"
DST_LABEL_ROOT = ROOT / "labels"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

# mapping class lama -> class baru
# lama:
# 0 helmet
# 1 gloves
# 2 vest
# 3 boots
# 4 goggles
# 5 none
# 6 Person
# 7 no_helmet
# 8 no_goggle
# 9 no_gloves
# 10 no_boots
CLASS_MAP = {
    6: 0,  # Person -> person
    0: 1,  # helmet -> helmet
    2: 2,  # vest -> vest
    1: 3,  # gloves -> gloves
    3: 4,  # boots -> safety_shoes
    4: 5,  # goggles -> goggles
}

DROPPED = {5, 7, 8, 9, 10}

def iter_images(split_dir: Path):
    for p in sorted(split_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p

def remap_label_file(src_label: Path):
    if not src_label.exists():
        return []

    out_lines = []
    text = src_label.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line in text:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        try:
            old_cls = int(float(parts[0]))
        except ValueError:
            continue

        if old_cls in DROPPED:
            continue

        if old_cls not in CLASS_MAP:
            continue

        new_cls = CLASS_MAP[old_cls]
        out_lines.append(" ".join([str(new_cls)] + parts[1:]))

    return out_lines

def main():
    summary = {}

    for split in ["train", "val", "test"]:
        img_dir = IMG_ROOT / split
        src_dir = SRC_LABEL_ROOT / split
        dst_dir = DST_LABEL_ROOT / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        images = 0
        remapped_nonempty = 0
        empty_background = 0

        for img_path in iter_images(img_dir):
            images += 1
            stem = img_path.stem
            src_label = src_dir / f"{stem}.txt"
            dst_label = dst_dir / f"{stem}.txt"

            lines = remap_label_file(src_label)

            if lines:
                dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
                remapped_nonempty += 1
            else:
                # tetap buat file kosong agar image dihitung sebagai background
                dst_label.write_text("", encoding="utf-8")
                empty_background += 1

        # hitung orphan label
        image_stems = {p.stem for p in iter_images(img_dir)}
        orphan_labels = 0
        if src_dir.exists():
            for lbl in src_dir.glob("*.txt"):
                if lbl.stem not in image_stems:
                    orphan_labels += 1

        summary[split] = {
            "images": images,
            "labels_written": images,
            "nonempty_labels": remapped_nonempty,
            "empty_background_labels": empty_background,
            "orphan_source_labels": orphan_labels,
        }

    print("=== REMAP SELESAI ===")
    for split, info in summary.items():
        print(f"\n[{split}]")
        for k, v in info.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
