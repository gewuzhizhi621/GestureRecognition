import argparse
import random
import shutil
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "datasets" / "raw" / "images"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "datasets" / "raw" / "labels"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "gesture"


def parse_args():
    parser = argparse.ArgumentParser(description="Split YOLO gesture data into train/val folders.")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def reset_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def paired_samples(images_dir, labels_dir):
    image_stems = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    label_stems = {p.stem for p in labels_dir.iterdir() if p.suffix.lower() == ".txt"}

    for stem in sorted(image_stems - label_stems):
        print(f"Image has no label: {stem}")
    for stem in sorted(label_stems - image_stems):
        print(f"Label has no image: {stem}")

    samples = []
    for stem in sorted(image_stems & label_stems):
        image_path = next(p for p in images_dir.iterdir() if p.stem == stem)
        label_path = labels_dir / f"{stem}.txt"
        samples.append((image_path, label_path))
    return samples


def copy_split(samples, output_dir, split_name):
    image_dir = output_dir / split_name / "images"
    label_dir = output_dir / split_name / "labels"
    reset_dir(image_dir)
    reset_dir(label_dir)

    for image_path, label_path in tqdm(samples, desc=f"Copying {split_name}", unit="file"):
        shutil.copy2(image_path, image_dir / image_path.name)
        shutil.copy2(label_path, label_dir / label_path.name)


def main():
    args = parse_args()
    random.seed(args.seed)

    samples = paired_samples(args.images, args.labels)
    random.shuffle(samples)
    train_size = int(len(samples) * args.train_ratio)

    copy_split(samples[:train_size], args.output, "train")
    copy_split(samples[train_size:], args.output, "val")
    print("Dataset split completed.")


if __name__ == "__main__":
    main()
