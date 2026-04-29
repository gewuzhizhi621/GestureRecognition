import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_SOURCE_DIR = PROJECT_ROOT / "third_party" / "yolov12"
if YOLO_SOURCE_DIR.exists():
    sys.path.insert(0, str(YOLO_SOURCE_DIR))

from ultralytics import YOLO

MODEL_PATH = PROJECT_ROOT / "runs" / "training" / "train5" / "weights" / "best.pt"


def main():
    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera is not available or frame capture failed.")
                break

            results = model(frame, conf=0.5, device="cpu")
            boxes = results[0].boxes
            if len(boxes) > 0:
                max_idx = boxes.conf.argmax()
                results[0].boxes = boxes[max_idx : max_idx + 1]

            cv2.imshow("YOLOv12 Real-time Detection", results[0].plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
