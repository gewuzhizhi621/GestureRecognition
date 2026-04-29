import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "datasets" / "incoming" / "images"
DEFAULT_TARGET_DIR = PROJECT_ROOT / "datasets" / "raw" / "images"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Copy image files into the raw gesture dataset.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET_DIR)
    return parser.parse_args()


def copy_images(source, target):
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in source.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            shutil.copy2(path, target / path.name)
            copied += 1
    print(f"Copied {copied} image file(s).")


if __name__ == "__main__":
    args = parse_args()
    copy_images(args.source, args.target)
