import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "datasets" / "incoming" / "labels"
DEFAULT_TARGET_DIR = PROJECT_ROOT / "datasets" / "raw" / "labels"


def parse_args():
    parser = argparse.ArgumentParser(description="Copy YOLO label files into the raw gesture dataset.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET_DIR)
    return parser.parse_args()


def copy_labels(source, target):
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in source.iterdir():
        if path.is_file() and path.suffix.lower() == ".txt":
            shutil.copy2(path, target / path.name)
            copied += 1
    print(f"Copied {copied} label file(s).")


if __name__ == "__main__":
    args = parse_args()
    copy_labels(args.source, args.target)
