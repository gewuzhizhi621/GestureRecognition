import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_SOURCE_DIR = PROJECT_ROOT / "third_party" / "yolov12"
if YOLO_SOURCE_DIR.exists():
    sys.path.insert(0, str(YOLO_SOURCE_DIR))

from ultralytics import YOLO

MODEL_PATH = PROJECT_ROOT / "runs" / "training" / "train" / "weights" / "best.pt"
DATASET_CONFIG = PROJECT_ROOT / "configs" / "gesture_dataset.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "evaluation"


def main():
    model = YOLO(str(MODEL_PATH))
    model.val(data=str(DATASET_CONFIG), split="test", imgsz=640, project=str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
