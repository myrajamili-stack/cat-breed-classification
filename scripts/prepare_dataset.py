import argparse
import random
import shutil
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_label(label):
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_images(raw_dir, max_classes, min_images_per_class):
    class_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    classes = []

    for class_dir in class_dirs:
        images = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(images) >= min_images_per_class:
            classes.append((normalize_label(class_dir.name), class_dir, images))

    classes.sort(key=lambda item: len(item[2]), reverse=True)
    return classes[:max_classes]


def copy_split(files, output_dir, label, split_name):
    target_dir = output_dir / split_name / label
    target_dir.mkdir(parents=True, exist_ok=True)

    for source in tqdm(files, desc=f"{split_name}/{label}", leave=False):
        target = target_dir / source.name
        counter = 1
        while target.exists():
            target = target_dir / f"{source.stem}_{counter}{source.suffix.lower()}"
            counter += 1
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description="Prepare cat breed image dataset.")
    parser.add_argument("--raw-dir", required=True, help="Folder with one subfolder per cat breed.")
    parser.add_argument("--output-dir", default="dataset_split", help="Output folder.")
    parser.add_argument("--max-classes", type=int, default=10, help="Maximum number of classes to keep.")
    parser.add_argument("--min-images-per-class", type=int, default=50)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-validation", action="store_true", help="Skip image corruption check.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {raw_dir}")

    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists. Rename/remove it before creating a fresh split."
        )

    random.seed(args.seed)
    selected_classes = collect_images(raw_dir, args.max_classes, args.min_images_per_class)

    if len(selected_classes) < 3:
        raise ValueError("The project requires at least 3 classes with enough images.")

    print("Selected classes:")
    for label, _, images in selected_classes:
        print(f"  {label}: {len(images)} raw images")

    summary_rows = ["class,train,val,test,total"]

    for label, _, image_paths in selected_classes:
        if not args.skip_validation:
            image_paths = [p for p in tqdm(image_paths, desc=f"checking/{label}") if is_valid_image(p)]

        image_paths = sorted(image_paths)
        train_files, temp_files = train_test_split(
            image_paths,
            test_size=args.val_size + args.test_size,
            random_state=args.seed,
            shuffle=True,
        )
        relative_test_size = args.test_size / (args.val_size + args.test_size)
        val_files, test_files = train_test_split(
            temp_files,
            test_size=relative_test_size,
            random_state=args.seed,
            shuffle=True,
        )

        copy_split(train_files, output_dir, label, "train")
        copy_split(val_files, output_dir, label, "val")
        copy_split(test_files, output_dir, label, "test")
        summary_rows.append(
            f"{label},{len(train_files)},{len(val_files)},{len(test_files)},{len(image_paths)}"
        )

    output_dir.mkdir(exist_ok=True)
    (output_dir / "dataset_summary.csv").write_text("\n".join(summary_rows) + "\n")
    print(f"\nDone. Dataset split saved to: {output_dir}")
    print(f"Summary saved to: {output_dir / 'dataset_summary.csv'}")


if __name__ == "__main__":
    main()
