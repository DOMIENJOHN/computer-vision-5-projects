import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parents[1]
manifest = pd.read_csv(ROOT / "results" / "splits.csv")
ImageFile.LOAD_TRUNCATED_IMAGES = False

bad = []

for i, row in manifest.iterrows():
    path = ROOT / row["path"]
    try:
        with Image.open(path) as img:
            img.load()
    except Exception as exc:
        bad.append((str(path), repr(exc)))
        print(f"BAD: {path}\n     {exc}")

print(f"\nChecked: {len(manifest)} manifest entries")
print(f"Unreadable images: {len(bad)}")

with (ROOT / "results" / "bad_images.txt").open("w") as f:
    for path, error in bad:
        f.write(f"{path}\t{error}\n")

print("Report: results/bad_images.txt")
