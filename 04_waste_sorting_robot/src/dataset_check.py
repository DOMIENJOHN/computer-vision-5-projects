from pathlib import Path
from collections import Counter
from PIL import Image, UnidentifiedImageError
import hashlib

DATASET = Path("dataset/waste_data")
SPLITS = ("train", "val")
CLASSES = ("biodegradable", "non_biodegradable")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def image_files(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    if not DATASET.is_dir():
        raise SystemExit(f"Dataset directory not found: {DATASET.resolve()}")

    hashes = {split: {} for split in SPLITS}
    corrupt = []
    counts = Counter()

    print(f"Dataset: {DATASET.resolve()}\n")

    for split in SPLITS:
        print(f"=== {split.upper()} ===")
        for class_name in CLASSES:
            folder = DATASET / split / class_name
            if not folder.is_dir():
                raise SystemExit(f"Missing class folder: {folder}")

            files = image_files(folder)
            counts[(split, class_name)] = len(files)
            print(f"{class_name}: {len(files)} images")

            for path in files:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    digest = sha256(path)
                    hashes[split].setdefault(digest, []).append(str(path))
                except (OSError, ValueError, UnidentifiedImageError):
                    corrupt.append(str(path))

        print()

    train_hashes = set(hashes["train"])
    val_hashes = set(hashes["val"])
    overlap = train_hashes & val_hashes

    print("=== SUMMARY ===")
    print(f"Training images: {sum(counts[('train', c)] for c in CLASSES)}")
    print(f"Validation images: {sum(counts[('val', c)] for c in CLASSES)}")
    print(f"Exact-content hashes shared between train and val: {len(overlap)}")
    print(f"Unreadable/corrupt images: {len(corrupt)}")

    if overlap:
        print("\nWARNING: exact duplicate image contents found across splits.")
        for digest in list(sorted(overlap))[:10]:
            print("Train:", hashes["train"][digest][0])
            print("Val:  ", hashes["val"][digest][0])

    if corrupt:
        print("\nUnreadable images:")
        for path in corrupt[:30]:
            print(path)

    print("\nCheck complete. No dataset files were changed.")

if __name__ == "__main__":
    main()
