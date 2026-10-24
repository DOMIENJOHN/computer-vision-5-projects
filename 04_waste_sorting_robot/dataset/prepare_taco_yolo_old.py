import json
import random
import shutil
from collections import Counter
from pathlib import Path

SOURCE = Path("taco_raw/TACO/data")
OUTPUT = Path("taco_yolo")
SEED = 42

SELECTED = [
    "Cigarette",
    "Plastic film",
    "Clear plastic bottle",
    "Other plastic",
    "Other plastic wrapper",
    "Drink can",
    "Plastic bottle cap",
    "Plastic straw",
    "Broken glass",
    "Glass bottle",
]

with (SOURCE / "annotations.json").open(encoding="utf-8") as f:
    coco = json.load(f)

categories = {
    c["id"]: c["name"]
    for c in coco["categories"]
}

missing_classes = [name for name in SELECTED if name not in categories.values()]
if missing_classes:
    raise RuntimeError(f"Classes missing from annotations: {missing_classes}")

class_ids = {name: i for i, name in enumerate(SELECTED)}
category_to_class = {
    category_id: class_ids[name]
    for category_id, name in categories.items()
    if name in class_ids
}

images = {image["id"]: image for image in coco["images"]}
annotations = {image_id: [] for image_id in images}
counts = Counter()

for ann in coco["annotations"]:
    category_id = ann["category_id"]
    if category_id not in category_to_class:
        continue

    image_id = ann["image_id"]
    if image_id not in images:
        continue

    image = images[image_id]
    x, y, w, h = map(float, ann["bbox"])
    width, height = image["width"], image["height"]

    # Clip bounding boxes to image boundaries.
    x1 = max(0.0, min(x, width))
    y1 = max(0.0, min(y, height))
    x2 = max(0.0, min(x + w, width))
    y2 = max(0.0, min(y + h, height))

    if x2 <= x1 or y2 <= y1:
        continue

    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    nw = (x2 - x1) / width
    nh = (y2 - y1) / height

    class_id = category_to_class[category_id]
    annotations[image_id].append(
        f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
    )
    counts[SELECTED[class_id]] += 1

# Split by image, not by individual annotation.
image_ids = list(images)
random.Random(SEED).shuffle(image_ids)

n = len(image_ids)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

splits = {
    "train": image_ids[:n_train],
    "val": image_ids[n_train:n_train + n_val],
    "test": image_ids[n_train + n_val:],
}

# Build a fresh output so old files cannot contaminate the split.
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

for split, ids in splits.items():
    image_dir = OUTPUT / "images" / split
    label_dir = OUTPUT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for image_id in ids:
        image = images[image_id]
        src = SOURCE / image["file_name"]

        if not src.is_file():
            raise FileNotFoundError(f"Missing image: {src}")

        # Unique names prevent filename collisions.
        filename = f"{image_id:06d}{src.suffix.lower()}"
        shutil.copy2(src, image_dir / filename)

        label_path = label_dir / f"{Path(filename).stem}.txt"
        label_path.write_text(
            "\n".join(annotations[image_id]) +
            ("\n" if annotations[image_id] else ""),
            encoding="utf-8",
        )

yaml = [
    f"path: {OUTPUT.resolve()}",
    "train: images/train",
    "val: images/val",
    "test: images/test",
    "names:",
]
yaml.extend(f"  {i}: {name}" for i, name in enumerate(SELECTED))
(OUTPUT / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

print("\nDataset preparation complete.")
print(f"Output: {OUTPUT.resolve()}")
print(f"Images: {n}")
for split, ids in splits.items():
    print(f"{split}: {len(ids)} images")

print("\nSelected object annotations:")
for name in SELECTED:
    print(f"{counts[name]:4d}  {name}")

print(f"\nTotal selected annotations: {sum(counts.values())}")
print(f"YAML: {OUTPUT / 'data.yaml'}")
