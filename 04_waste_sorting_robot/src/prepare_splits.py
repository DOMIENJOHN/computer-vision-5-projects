from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import csv
import random

ROOT = Path("dataset/waste_data")
OUT = Path("results")
CLASSES = ("biodegradable", "non_biodegradable")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Group all images by exact content, not by filename.
    groups = defaultdict(list)
    for original_split in ("train", "val"):
        for label in CLASSES:
            folder = ROOT / original_split / label
            if not folder.is_dir():
                raise SystemExit(f"Missing folder: {folder}")
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    groups[digest(path)].append((path, label, original_split))

    # Exact copies with conflicting labels need manual review.
    conflicting = []
    for sha, entries in groups.items():
        labels = {label for _, label, _ in entries}
        if len(labels) > 1:
            conflicting.append((sha, entries))

    if conflicting:
        report = OUT / "conflicting_duplicate_labels.txt"
        with report.open("w") as f:
            for sha, entries in conflicting:
                f.write(f"\nSHA256: {sha}\n")
                for path, label, split in entries:
                    f.write(f"  {split},{label},{path}\n")
        raise SystemExit(
            f"Found {len(conflicting)} exact-content groups with conflicting labels. "
            f"Review {report}; no manifests were generated."
        )

    # Each unique content group gets a single label and a single split.
    by_label = defaultdict(list)
    for sha, entries in groups.items():
        label = entries[0][1]
        representative = sorted(str(p) for p, _, _ in entries)[0]
        by_label[label].append((sha, representative))

    rng = random.Random(SEED)
    assignments = {}
    for label in CLASSES:
        items = sorted(by_label[label])
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * 0.10))
        n_val = max(1, round(n * 0.10))
        if n_test + n_val >= n:
            raise SystemExit(f"Not enough unique images in class {label}")
        for i, (sha, path) in enumerate(items):
            split = "test" if i < n_test else (
                "val" if i < n_test + n_val else "train"
            )
            assignments[sha] = (split, label, path)

    manifest = OUT / "splits.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sha256", "split", "label", "path"])
        for sha in sorted(assignments):
            split, label, path = assignments[sha]
            writer.writerow([sha, split, label, path])

    counts = Counter((split, label) for split, label, _ in assignments.values())
    print(f"Unique image contents: {len(assignments)}")
    print(f"Manifest: {manifest}")
    print("\nSplit counts:")
    for split in ("train", "val", "test"):
        total = 0
        for label in CLASSES:
            count = counts[(split, label)]
            total += count
            print(f"  {split:5} | {label:18} | {count:6}")
        print(f"  {split:5} | {'TOTAL':18} | {total:6}")

    # Verify that each hash appears only once in the manifest.
    print("\nLeakage check: each unique SHA-256 is assigned to exactly one split.")
    print("Original dataset files were not changed.")

if __name__ == "__main__":
    main()
