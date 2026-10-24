from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
manifest_path = ROOT / "results" / "splits.csv"
output_path = ROOT / "results" / "splits_clean.csv"

df = pd.read_csv(manifest_path)
bad_paths = set()

for path in df["path"]:
    full_path = ROOT / path
    try:
        with Image.open(full_path) as image:
            image.load()
    except Exception as exc:
        bad_paths.add(path)
        print(f"Excluding unreadable image: {path} ({exc})")

clean = df[~df["path"].isin(bad_paths)].copy()
clean.to_csv(output_path, index=False)

print(f"\nOriginal manifest entries: {len(df)}")
print(f"Excluded unreadable images: {len(bad_paths)}")
print(f"Clean manifest entries: {len(clean)}")
print(f"Saved: {output_path}")

print("\nCounts by split and class:")
print(clean.groupby(["split", "label"]).size().to_string())

assert not clean["sha256"].duplicated().any()
assert set(clean["split"]) == {"train", "val", "test"}
