
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

SOURCE = Path("04_waste_sorting_robot/dataset/taco_raw/TACO/data")
OUTPUT = Path("04_waste_sorting_robot/dataset/taco_yolo_balanced")
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

categories = {c["id"]: c["name"] for c in coco["categories"]}
missing = [name for name in SELECTED if name not in categories.values()]
if missing:
    raise RuntimeError(f"Missing classes: {missing}")

class_ids = {name: i for i, name in enumerate(SELECTED)}
category_to_class = {
    cid: class_ids[name]
    for cid, name in categories.items()
    if name in class_ids
}

images = {im["id"]: im for im in coco["images"]}
annotations = {image_id: [] for image_id in images}
image_classes = {image_id: set() for image_id in images}
instance_counts = Counter()

for ann in coco["annotations"]:
    cid = ann["category_id"]
    image_id = ann["image_id"]

    if cid not in category_to_class or image_id not in images:
        continue

    im = images[image_id]
    width, height = im["width"], im["height"]
    x, y, w, h = map(float, ann["bbox"])

    x1 = max(0.0, min(x, width))
    y1 = max(0.0, min(y, height))
    x2 = max(0.0, min(x + w, width))
    y2 = max(0.0, min(y + h, height))

    if x2 <= x1 or y2 <= y1:
        continue

    cls = category_to_class[cid]
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    nw = (x2 - x1) / width
    nh = (y2 - y1) / height

    annotations[image_id].append(
        f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
    )
    image_classes[image_id].add(cls)
    instance_counts[SELECTED[cls]] += 1

# Keep all objects from an image together.
image_ids = list(images)
Y = np.zeros((len(image_ids), len(SELECTED)), dtype=int)

for row, image_id in enumerate(image_ids):
    for cls in image_classes[image_id]:
        Y[row, cls] = 1

# Split 80% train, 20% temporary.
split1 = MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=0.20, random_state=SEED
)
train_idx, temp_idx = next(split1.split(np.zeros(len(image_ids)), Y))

temp_ids = [image_ids[i] for i in temp_idx]
Y_temp = Y[temp_idx]

# Split temporary set equally into validation and test.
split2 = MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=0.50, random_state=SEED
)
val_local, test_local = next(
    split2.split(np.zeros(len(temp_ids)), Y_temp)
)

splits = {
    "train": [image_ids[i] for i in train_idx],
    "val": [temp_ids[i] for i in val_local],
    "test": [temp_ids[i] for i in test_local],
}

# Safety check: do not overwrite the previous dataset.
if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Rename or remove it deliberately first."
    )

for split, ids in splits.items():
    image_dir = OUTPUT / "images" / split
    label_dir = OUTPUT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for image_id in ids:
        im = images[image_id]
        src = SOURCE / im["file_name"]
        if not src.is_file():
            raise FileNotFoundError(f"Missing image: {src}")

        filename = f"{image_id:06d}{src.suffix.lower()}"
        shutil.copy2(src, image_dir / filename)

        label_path = label_dir / f"{Path(filename).stem}.txt"
        label_path.write_text(
            "\n".join(annotations[image_id])
            + ("\n" if annotations[image_id] else ""),
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
(OUTPUT / "data.yaml").write_text("\n".join(yaml) + "\n")

print("\nDataset preparation complete.")
print(f"Output: {OUTPUT.resolve()}")
for split, ids in splits.items():
    split_counts = Counter()
    split_images = Counter()
    for image_id in ids:
        for cls in image_classes[image_id]:
            split_images[cls] += 1
        for line in annotations[image_id]:
            split_counts[int(line.split()[0])] += 1

    print(f"\n{split}: {len(ids)} images")
    for cls, name in enumerate(SELECTED):
        print(
            f"  {name:24s} "
            f"images={split_images[cls]:3d} "
            f"instances={split_counts[cls]:4d}"
        )

print("\nTotal selected instances:", sum(instance_counts.values()))
print("YAML:", OUTPUT / "data.yaml")
