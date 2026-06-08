import argparse
import hashlib
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
        return width >= 128 and height >= 128
    except Exception:
        return False


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_class(class_dir, output_dir, target_per_class, seen_hashes):
    target_dir = output_dir / class_dir.name
    target_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        path for path in sorted(class_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    kept = 0
    invalid = 0
    duplicates = 0

    for source in tqdm(candidates, desc=f"cleaning/{class_dir.name}"):
        if kept >= target_per_class:
            break

        if not is_valid_image(source):
            invalid += 1
            continue

        digest = file_hash(source)
        if digest in seen_hashes:
            duplicates += 1
            continue

        seen_hashes.add(digest)
        target = target_dir / f"{class_dir.name}_{kept + 1:04d}.jpg"

        try:
            with Image.open(source) as img:
                img.convert("RGB").save(target, format="JPEG", quality=92)
            kept += 1
        except Exception:
            invalid += 1

    return {
        "class": class_dir.name,
        "raw_candidates": len(candidates),
        "kept": kept,
        "invalid": invalid,
        "duplicates": duplicates,
    }


def main():
    parser = argparse.ArgumentParser(description="Clean crawled cat breed images.")
    parser.add_argument("--raw-dir", default="dataset_raw")
    parser.add_argument("--output-dir", default="dataset_clean")
    parser.add_argument("--target-per-class", type=int, default=1000)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists. Rename/remove it before creating a fresh clean dataset."
        )

    class_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found in {raw_dir}")

    seen_hashes = set()
    rows = ["class,raw_candidates,kept,invalid,duplicates"]

    for class_dir in class_dirs:
        stats = clean_class(class_dir, output_dir, args.target_per_class, seen_hashes)
        rows.append(
            f"{stats['class']},{stats['raw_candidates']},{stats['kept']},"
            f"{stats['invalid']},{stats['duplicates']}"
        )

    summary_path = output_dir / "cleaning_summary.csv"
    summary_path.write_text("\n".join(rows) + "\n")

    print("\nCleaning complete.")
    print(f"Clean dataset: {output_dir}")
    print(f"Summary: {summary_path}")
    print("\nIf any class has fewer than the target images, rerun the crawler or add another source.")


if __name__ == "__main__":
    main()
