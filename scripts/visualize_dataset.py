import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(split_dir):
    rows = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        count = sum(
            1 for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        rows.append({"class": class_dir.name, "count": count})
    return pd.DataFrame(rows)


def plot_class_distribution(dataset_dir, output_dir):
    frames = []
    for split in ["train", "val", "test"]:
        df = count_images(dataset_dir / split)
        df["split"] = split
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    pivot = data.pivot(index="class", columns="split", values="count").fillna(0)
    pivot[["train", "val", "test"]].plot(kind="bar", stacked=True, figsize=(12, 6))
    plt.title("Cat Breed Dataset Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_class_distribution.png", dpi=180)
    plt.close()

    data.to_csv(output_dir.parent / "dataset_distribution.csv", index=False)


def plot_sample_grid(dataset_dir, output_dir, samples_per_class):
    class_dirs = sorted(p for p in (dataset_dir / "train").iterdir() if p.is_dir())
    n_rows = len(class_dirs)
    n_cols = samples_per_class
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 2.2))

    if n_rows == 1:
        axes = [axes]

    for row_index, class_dir in enumerate(class_dirs):
        images = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ][:samples_per_class]
        for col_index in range(n_cols):
            ax = axes[row_index][col_index] if n_rows > 1 else axes[col_index]
            ax.axis("off")
            if col_index < len(images):
                with Image.open(images[col_index]) as img:
                    ax.imshow(img.convert("RGB"))
            if col_index == 0:
                ax.set_title(class_dir.name, fontsize=9)

    fig.suptitle("Sample Training Images by Breed")
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_sample_grid.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create dataset visualizations.")
    parser.add_argument("--dataset-dir", default="dataset_split")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--samples-per-class", type=int, default=4)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.results_dir).resolve() / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_class_distribution(dataset_dir, output_dir)
    plot_sample_grid(dataset_dir, output_dir, args.samples_per_class)
    print(f"Saved dataset visualizations to {output_dir}")


if __name__ == "__main__":
    main()
