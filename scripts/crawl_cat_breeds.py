import argparse
import time
from pathlib import Path

from icrawler.builtin import BingImageCrawler


DEFAULT_CLASSES = {
    "abyssinian": [
        "Abyssinian cat breed",
        "Abyssinian cat portrait",
        "Abyssinian kitten",
    ],
    "bengal": [
        "Bengal cat breed",
        "Bengal cat portrait",
        "Bengal kitten",
    ],
    "birman": [
        "Birman cat breed",
        "Birman cat portrait",
        "Sacred Birman cat",
    ],
    "british_shorthair": [
        "British Shorthair cat breed",
        "British Shorthair cat portrait",
        "British Shorthair kitten",
    ],
    "maine_coon": [
        "Maine Coon cat breed",
        "Maine Coon cat portrait",
        "Maine Coon kitten",
    ],
    "persian": [
        "Persian cat breed",
        "Persian cat portrait",
        "Persian kitten",
    ],
    "siamese": [
        "Siamese cat breed",
        "Siamese cat portrait",
        "Siamese kitten",
    ],
}


def count_files(class_dir):
    return sum(1 for path in class_dir.glob("*") if path.is_file())


def crawl_class(class_name, queries, raw_dir, target_per_class, per_query_limit, sleep_seconds):
    class_dir = raw_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {class_name} ===")
    print(f"Target files: {target_per_class}")

    for query in queries:
        current_count = count_files(class_dir)
        if current_count >= target_per_class:
            print(f"{class_name}: target reached with {current_count} files")
            return

        remaining = target_per_class - current_count
        max_num = min(per_query_limit, remaining + 200)

        print(f"Crawling query: {query}")
        print(f"Current files: {current_count}; requesting up to {max_num}")

        crawler = BingImageCrawler(
            storage={"root_dir": str(class_dir)},
            downloader_threads=4,
        )
        crawler.crawl(
            keyword=query,
            max_num=max_num,
            file_idx_offset=current_count,
        )

        time.sleep(sleep_seconds)

    final_count = count_files(class_dir)
    print(f"{class_name}: finished with {final_count} files before cleaning")


def main():
    parser = argparse.ArgumentParser(description="Crawl cat breed images for the AI project.")
    parser.add_argument("--raw-dir", default="dataset_raw")
    parser.add_argument("--target-per-class", type=int, default=1000)
    parser.add_argument("--per-query-limit", type=int, default=700)
    parser.add_argument("--sleep-seconds", type=int, default=5)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Selected classes:", list(DEFAULT_CLASSES))
    print("Images per class target:", args.target_per_class)
    print("Target total images:", len(DEFAULT_CLASSES) * args.target_per_class)

    for class_name, queries in DEFAULT_CLASSES.items():
        crawl_class(
            class_name=class_name,
            queries=queries,
            raw_dir=raw_dir,
            target_per_class=args.target_per_class,
            per_query_limit=args.per_query_limit,
            sleep_seconds=args.sleep_seconds,
        )

    print("\nCrawling complete. Run clean_crawled_images.py next.")


if __name__ == "__main__":
    main()
