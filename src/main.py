# -*- coding: utf-8 -*-
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_SOURCE_DIR = PROJECT_ROOT / "third_party" / "yolov12"
if YOLO_SOURCE_DIR.exists():
    sys.path.insert(0, str(YOLO_SOURCE_DIR))

MODEL_PATH = PROJECT_ROOT / "runs" / "training" / "train5" / "weights" / "best.pt"

GESTURE_CLASSES = ["paper", "rock", "scissor", "OK", "good"]
GESTURE_NAMES = {
    "paper": "Paper",
    "rock": "Rock",
    "scissor": "Scissor",
    "OK": "OK",
    "good": "Good",
}

COLOR = {
    "bg": "#0b1120",
    "sidebar": "#020617",
    "sidebar_soft": "#111827",
    "card": "#111827",
    "card_line": "#243244",
    "text": "#e5e7eb",
    "muted": "#94a3b8",
    "blue": "#2563eb",
    "blue_dark": "#1d4ed8",
    "green": "#22c55e",
    "green_bg": "#052e16",
    "red": "#dc2626",
    "black": "#020617",
}


class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.4):
        from ultralytics import YOLO

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(str(model_path))

    def predict(self, frame):
        return self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, device="cpu")


class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition")
        self.root.geometry("1180x720")
        self.root.minsize(1020, 640)
        self.root.configure(bg=COLOR["bg"])

        self.model = None
        self.model_loaded = False
        self.camera = None
        self.running = False
        self.worker = None
        self.latest_frame = None
        self.result_queue = queue.Queue(maxsize=1)

        self.conf_var = tk.DoubleVar(value=0.5)
        self.iou_var = tk.DoubleVar(value=0.4)
        self.camera_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="MODEL NOT LOADED")
        self.result_var = tk.StringVar(value="Waiting")
        self.confidence_var = tk.StringVar(value="--")

        self._setup_style()
        self._build_ui()
        self._refresh_video()

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background=COLOR["bg"])
        style.configure("Horizontal.TScale", background=COLOR["card"])
        style.configure(
            "TCombobox",
            fieldbackground=COLOR["sidebar_soft"],
            background=COLOR["sidebar_soft"],
            foreground=COLOR["text"],
            arrowcolor=COLOR["text"],
            padding=6,
        )

    def _build_ui(self):
        shell = tk.Frame(self.root, bg=COLOR["bg"])
        shell.pack(fill=tk.BOTH, expand=True)
        shell.columnconfigure(1, weight=1)
        shell.rowconfigure(0, weight=1)

        self._build_sidebar(shell)
        self._build_workspace(shell)

    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=COLOR["sidebar"], width=250)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)

        tk.Label(
            sidebar,
            text="Gesture\nRecognition",
            bg=COLOR["sidebar"],
            fg="#ffffff",
            font=("Segoe UI", 24, "bold"),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=24, pady=(28, 8))

        tk.Label(
            sidebar,
            text="YOLOv12 camera demo",
            bg=COLOR["sidebar"],
            fg="#9ca3af",
            font=("Segoe UI", 10),
        ).pack(anchor=tk.W, padx=24)

        self.status_pill = tk.Label(
            sidebar,
            textvariable=self.status_var,
            bg=COLOR["sidebar_soft"],
            fg="#bfdbfe",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=8,
        )
        self.status_pill.pack(anchor=tk.W, padx=24, pady=(28, 18), fill=tk.X)

        self.load_button = self._side_button(sidebar, "Load Model", self.load_model)
        self.load_button.pack(fill=tk.X, padx=24, pady=(0, 10))

        self.start_button = self._side_button(sidebar, "Start Detection", self.toggle_recognition, primary=True)
        self.start_button.pack(fill=tk.X, padx=24, pady=(0, 22))

        tk.Label(sidebar, text="Camera", bg=COLOR["sidebar"], fg="#9ca3af", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=24)
        camera_box = ttk.Combobox(sidebar, textvariable=self.camera_var, values=[0, 1, 2], state="readonly", width=10)
        camera_box.pack(anchor=tk.W, padx=24, pady=(6, 18))

        self._side_slider(sidebar, "Confidence", self.conf_var)
        self._side_slider(sidebar, "NMS", self.iou_var)

        tk.Label(
            sidebar,
            text=f"Model\n{MODEL_PATH.relative_to(PROJECT_ROOT)}",
            bg=COLOR["sidebar"],
            fg="#6b7280",
            font=("Segoe UI", 8),
            justify=tk.LEFT,
            wraplength=190,
        ).pack(side=tk.BOTTOM, anchor=tk.W, padx=24, pady=24)

    def _side_button(self, parent, text, command, primary=False):
        bg = COLOR["blue"] if primary else COLOR["sidebar_soft"]
        active = COLOR["blue_dark"] if primary else "#374151"
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg="#ffffff",
            activebackground=active,
            activeforeground="#ffffff",
            bd=0,
            relief=tk.FLAT,
            cursor="hand2",
            font=("Segoe UI", 11, "bold"),
            padx=14,
            pady=12,
        )

    def _side_slider(self, parent, title, variable):
        box = tk.Frame(parent, bg=COLOR["sidebar"])
        box.pack(fill=tk.X, padx=24, pady=(0, 16))

        row = tk.Frame(box, bg=COLOR["sidebar"])
        row.pack(fill=tk.X)

        tk.Label(row, text=title, bg=COLOR["sidebar"], fg="#9ca3af", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        value = tk.StringVar(value=f"{variable.get():.2f}")
        tk.Label(row, textvariable=value, bg=COLOR["sidebar"], fg="#ffffff", font=("Segoe UI", 9)).pack(side=tk.RIGHT)

        def update_value(*_):
            value.set(f"{variable.get():.2f}")

        variable.trace_add("write", update_value)
        ttk.Scale(box, from_=0.1, to=1.0, variable=variable, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(8, 0))

    def _build_workspace(self, parent):
        workspace = tk.Frame(parent, bg=COLOR["bg"])
        workspace.grid(row=0, column=1, sticky="nsew")
        workspace.columnconfigure(0, weight=3)
        workspace.columnconfigure(1, weight=2)
        workspace.rowconfigure(1, weight=1)

        header = tk.Frame(workspace, bg=COLOR["bg"])
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=28, pady=(24, 16))

        tk.Label(
            header,
            text="Real-time Gesture Dashboard",
            bg=COLOR["bg"],
            fg=COLOR["text"],
            font=("Segoe UI", 24, "bold"),
        ).pack(anchor=tk.W)
        tk.Label(
            header,
            text="Use the camera to recognize hand gestures and display the latest prediction.",
            bg=COLOR["bg"],
            fg=COLOR["muted"],
            font=("Segoe UI", 10),
        ).pack(anchor=tk.W, pady=(4, 0))

        left = self._card(workspace)
        left.grid(row=1, column=0, sticky="nsew", padx=(28, 14), pady=(0, 28))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        tk.Label(left, text="Live Camera", bg=COLOR["card"], fg=COLOR["text"], font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, sticky="w", padx=20, pady=(18, 12)
        )

        self.video_label = tk.Label(
            left,
            bg=COLOR["black"],
            fg="#94a3b8",
            text="Camera preview will appear here",
            font=("Segoe UI", 15),
        )
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))

        right = tk.Frame(workspace, bg=COLOR["bg"])
        right.grid(row=1, column=1, sticky="nsew", padx=(14, 28), pady=(0, 28))
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_result_card(right)
        self._build_history_card(right)

    def _card(self, parent):
        card = tk.Frame(parent, bg=COLOR["card"], highlightbackground=COLOR["card_line"], highlightthickness=1)
        return card

    def _build_result_card(self, parent):
        card = self._card(parent)
        card.grid(row=0, column=0, sticky="ew", pady=(0, 16))

        tk.Label(card, text="Current Result", bg=COLOR["card"], fg=COLOR["text"], font=("Segoe UI", 14, "bold")).pack(
            anchor=tk.W, padx=20, pady=(18, 4)
        )
        tk.Label(card, text="Latest high-confidence gesture", bg=COLOR["card"], fg=COLOR["muted"], font=("Segoe UI", 9)).pack(
            anchor=tk.W, padx=20
        )

        self.result_badge = tk.Label(
            card,
            textvariable=self.result_var,
            bg=COLOR["green_bg"],
            fg="#bbf7d0",
            font=("Segoe UI", 34, "bold"),
            pady=28,
        )
        self.result_badge.pack(fill=tk.X, padx=20, pady=(18, 12))

        confidence_row = tk.Frame(card, bg=COLOR["card"])
        confidence_row.pack(fill=tk.X, padx=20, pady=(0, 20))
        tk.Label(confidence_row, text="Confidence", bg=COLOR["card"], fg=COLOR["muted"], font=("Segoe UI", 10)).pack(side=tk.LEFT)
        tk.Label(confidence_row, textvariable=self.confidence_var, bg=COLOR["card"], fg=COLOR["green"], font=("Segoe UI", 18, "bold")).pack(
            side=tk.RIGHT
        )

    def _build_history_card(self, parent):
        card = self._card(parent)
        card.grid(row=1, column=0, sticky="nsew")
        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)

        tk.Label(card, text="Detection History", bg=COLOR["card"], fg=COLOR["text"], font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, sticky="w", padx=20, pady=(18, 12)
        )
        self.history = tk.Listbox(
            card,
            bg="#0f172a",
            fg=COLOR["text"],
            bd=0,
            highlightthickness=0,
            activestyle="none",
            font=("Consolas", 10),
            selectbackground="#1e40af",
            selectforeground="#ffffff",
        )
        self.history.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))

    def load_model(self):
        if not MODEL_PATH.exists():
            messagebox.showerror("Model Not Found", f"Model file not found:\n{MODEL_PATH}")
            return

        try:
            self.status_var.set("LOADING MODEL")
            self.root.update_idletasks()
            self.model = YOLODetector(MODEL_PATH, self.conf_var.get(), self.iou_var.get())
            self.model_loaded = True
            self.load_button.configure(state=tk.DISABLED, bg="#374151")
            self.status_var.set("MODEL READY")
            messagebox.showinfo("Success", "Model loaded successfully.")
        except Exception as exc:
            self.status_var.set("LOAD FAILED")
            messagebox.showerror("Load Failed", str(exc))

    def toggle_recognition(self):
        if self.running:
            self.stop_recognition()
        else:
            self.start_recognition()

    def start_recognition(self):
        if not self.model_loaded:
            messagebox.showwarning("Notice", "Please load the model first.")
            return

        self.camera = cv2.VideoCapture(self.camera_var.get())
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            messagebox.showerror("Camera Error", f"Cannot open camera {self.camera_var.get()}")
            return

        self.running = True
        self.latest_frame = None
        self._clear_queue()
        self.start_button.configure(text="Stop Detection", bg=COLOR["red"], activebackground="#b91c1c")
        self.status_var.set("DETECTING")

        self.worker = threading.Thread(target=self._recognition_loop, daemon=True)
        self.worker.start()

    def stop_recognition(self):
        self.running = False
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=1.0)
        if self.camera:
            self.camera.release()
            self.camera = None

        self.video_label.configure(image="", text="Detection stopped")
        self.start_button.configure(text="Start Detection", bg=COLOR["blue"], activebackground=COLOR["blue_dark"])
        self.status_var.set("STOPPED")

    def _recognition_loop(self):
        while self.running:
            if self.latest_frame is not None and self.model_loaded:
                gesture, confidence = self._recognize(self.latest_frame)
                self._push_result(gesture, confidence)
            time.sleep(0.2)

    def _recognize(self, frame):
        results = self.model.predict(frame)
        if not results:
            return "Unknown", 0.0

        boxes = results[0].boxes
        if len(boxes) == 0:
            return "Unknown", 0.0

        max_idx = boxes.conf.argmax()
        best_box = boxes[max_idx : max_idx + 1]
        class_index = int(best_box.cls[0])
        gesture = GESTURE_CLASSES[class_index] if class_index < len(GESTURE_CLASSES) else "unknown"
        confidence = float(best_box.conf[0]) * 100
        return GESTURE_NAMES.get(gesture, gesture), confidence

    def _refresh_video(self):
        if self.running and self.camera and self.camera.isOpened():
            ok, frame = self.camera.read()
            if ok:
                frame = cv2.flip(frame, 1)
                self.latest_frame = frame.copy()
                self._show_frame(frame)

            try:
                gesture, confidence = self.result_queue.get_nowait()
                self._update_result(gesture, confidence)
            except queue.Empty:
                pass

        self.root.after(30, self._refresh_video)

    def _show_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        max_width = max(self.video_label.winfo_width() - 16, 360)
        max_height = max(self.video_label.winfo_height() - 16, 260)
        image.thumbnail((max_width, max_height), Image.LANCZOS)

        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

    def _update_result(self, gesture, confidence):
        self.result_var.set(gesture)
        self.confidence_var.set(f"{confidence:.1f}%")

        if gesture != "Unknown":
            now = time.strftime("%H:%M:%S")
            self.history.insert(0, f"{now}   {gesture:<10} {confidence:>5.1f}%")
            if self.history.size() > 14:
                self.history.delete(14)

    def _push_result(self, gesture, confidence):
        if self.result_queue.full():
            self._clear_queue()
        self.result_queue.put((gesture, confidence))

    def _clear_queue(self):
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

    def close(self):
        self.stop_recognition()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = GestureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
