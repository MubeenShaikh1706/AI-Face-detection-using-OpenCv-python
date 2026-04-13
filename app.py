from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

from src.face_pipeline import FaceAnalysisEngine, ModelDownloadError

ROOT_DIR = Path(__file__).resolve().parent
WINDOW_TITLE = "FaceSight Studio"
APP_WIDTH = 1380
APP_HEIGHT = 860

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class FaceSightApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title(WINDOW_TITLE)
        self.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.minsize(1240, 780)

        self.engine = FaceAnalysisEngine()
        self.camera: cv.VideoCapture | None = None
        self.camera_running = False
        self.video_source = None
        self.current_frame = None
        self.preview_image = None
        self.processing_thread = None
        self.busy = False
        self.last_status = "Ready"
        self.analysis_count = 0

        self._build_layout()
        self.after(30, self._ui_loop)

    def _build_layout(self) -> None:
        self.configure(fg_color="#08111f")

        self.sidebar = ctk.CTkFrame(self, width=352, corner_radius=24, fg_color="#0f172a", border_width=1, border_color="#1e293b")
        self.sidebar.pack(side="left", fill="y", padx=20, pady=20)
        self.sidebar.pack_propagate(False)

        self.content = ctk.CTkFrame(self, corner_radius=24, fg_color="#0b1220", border_width=1, border_color="#1e293b")
        self.content.pack(side="right", fill="both", expand=True, padx=(0, 20), pady=20)

        self._build_sidebar()
        self._build_content()

    def _build_sidebar(self) -> None:
        header = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(24, 12))

        title = ctk.CTkLabel(header, text="FaceSight Studio", font=ctk.CTkFont(size=28, weight="bold"), text_color="#ffffff")
        title.pack(anchor="w")

        subtitle = ctk.CTkLabel(
            header,
            text="OpenCV-powered face detection with age and gender estimation",
            font=ctk.CTkFont(size=13),
            text_color="#94a3b8",
            wraplength=286,
            justify="left",
        )
        subtitle.pack(anchor="w", pady=(8, 0))

        self.status_card = ctk.CTkFrame(self.sidebar, corner_radius=18, fg_color="#132038", border_width=1, border_color="#334155")
        self.status_card.pack(fill="x", padx=20, pady=12)
        self.status_label = ctk.CTkLabel(self.status_card, text=self.last_status, font=ctk.CTkFont(size=15, weight="bold"), text_color="#e2e8f0")
        self.status_label.pack(anchor="w", padx=16, pady=(14, 2))
        self.detail_label = ctk.CTkLabel(self.status_card, text="Load a frame or start the camera to begin.", font=ctk.CTkFont(size=12), text_color="#94a3b8", wraplength=270, justify="left")
        self.detail_label.pack(anchor="w", padx=16, pady=(0, 14))

        controls = ctk.CTkFrame(self.sidebar, corner_radius=18, fg_color="#111c31", border_width=1, border_color="#334155")
        controls.pack(fill="x", padx=20, pady=12)

        ctk.CTkLabel(controls, text="Controls", font=ctk.CTkFont(size=16, weight="bold"), text_color="#f8fafc").pack(anchor="w", padx=16, pady=(16, 10))

        self.start_btn = ctk.CTkButton(controls, text="Start Camera", height=44, command=self.start_camera, fg_color="#14b8a6", hover_color="#0f766e")
        self.start_btn.pack(fill="x", padx=16, pady=6)

        self.stop_btn = ctk.CTkButton(controls, text="Stop Camera", height=44, command=self.stop_camera, fg_color="#334155", hover_color="#475569")
        self.stop_btn.pack(fill="x", padx=16, pady=6)

        self.image_btn = ctk.CTkButton(controls, text="Load Image", height=44, command=self.load_image, fg_color="#2563eb", hover_color="#1d4ed8")
        self.image_btn.pack(fill="x", padx=16, pady=6)

        self.clear_btn = ctk.CTkButton(controls, text="Clear Preview", height=44, command=self.clear_preview, fg_color="#475569", hover_color="#64748b")
        self.clear_btn.pack(fill="x", padx=16, pady=(6, 16))

        metrics = ctk.CTkFrame(self.sidebar, corner_radius=18, fg_color="#111c31", border_width=1, border_color="#334155")
        metrics.pack(fill="x", padx=20, pady=12)
        ctk.CTkLabel(metrics, text="Live Metrics", font=ctk.CTkFont(size=16, weight="bold"), text_color="#f8fafc").pack(anchor="w", padx=16, pady=(16, 10))

        self.metric_faces = ctk.CTkLabel(metrics, text="Faces detected: 0", font=ctk.CTkFont(size=13), text_color="#cbd5e1")
        self.metric_faces.pack(anchor="w", padx=16, pady=3)
        self.metric_frames = ctk.CTkLabel(metrics, text="Frames processed: 0", font=ctk.CTkFont(size=13), text_color="#cbd5e1")
        self.metric_frames.pack(anchor="w", padx=16, pady=3)
        self.metric_mode = ctk.CTkLabel(metrics, text="Mode: Idle", font=ctk.CTkFont(size=13), text_color="#cbd5e1")
        self.metric_mode.pack(anchor="w", padx=16, pady=(3, 16))

    def _build_content(self) -> None:
        top_bar = ctk.CTkFrame(self.content, corner_radius=0, fg_color="transparent")
        top_bar.pack(fill="x", padx=24, pady=(24, 10))

        self.hero_title = ctk.CTkLabel(top_bar, text="Professional AI Vision Dashboard", font=ctk.CTkFont(size=24, weight="bold"), text_color="#ffffff")
        self.hero_title.pack(anchor="w")

        self.hero_desc = ctk.CTkLabel(
            top_bar,
            text="Real-time face detection, age estimation, and gender classification with a camera-ready interface.",
            font=ctk.CTkFont(size=13),
            text_color="#94a3b8",
        )
        self.hero_desc.pack(anchor="w", pady=(6, 0))

        body = ctk.CTkFrame(self.content, corner_radius=24, fg_color="#101827", border_width=1, border_color="#1f2937")
        body.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        body.grid_columnconfigure(0, weight=4, minsize=760)
        body.grid_columnconfigure(1, weight=1, minsize=390)
        body.grid_rowconfigure(0, weight=1)

        self.preview_panel = ctk.CTkFrame(body, corner_radius=22, fg_color="#060d18", border_width=1, border_color="#1f2937")
        self.preview_panel.grid(row=0, column=0, sticky="nsew", padx=(18, 10), pady=18)

        self.preview_label = ctk.CTkLabel(
            self.preview_panel,
            text="Camera feed will appear here",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#94a3b8",
            fg_color="#0b1020",
            corner_radius=18,
            width=860,
            height=620,
        )
        self.preview_label.pack(fill="both", expand=True, padx=18, pady=18)

        self.results_panel = ctk.CTkFrame(body, corner_radius=22, fg_color="#0f1a2f", border_width=1, border_color="#3b82f6")
        self.results_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 18), pady=18)
        self.results_panel.grid_propagate(False)

        report_header = ctk.CTkFrame(self.results_panel, corner_radius=14, fg_color="#163459", border_width=1, border_color="#60a5fa")
        report_header.pack(fill="x", padx=20, pady=(18, 12))
        ctk.CTkLabel(report_header, text="Analysis Report", font=ctk.CTkFont(size=20, weight="bold"), text_color="#ffffff").pack(anchor="w", padx=14, pady=(12, 2))
        ctk.CTkLabel(report_header, text="Live detection details and confidence values", font=ctk.CTkFont(size=12), text_color="#dbeafe").pack(anchor="w", padx=14, pady=(0, 12))

        self.report_stats = ctk.CTkFrame(self.results_panel, fg_color="transparent")
        self.report_stats.pack(fill="x", padx=20, pady=(0, 12))

        self.report_source = self._create_report_stat(self.report_stats, "Source", "Idle")
        self.report_source.pack(fill="x", pady=(0, 10))

        self.report_faces = self._create_report_stat(self.report_stats, "Faces", "0")
        self.report_faces.pack(fill="x", pady=(0, 10))

        self.report_frames = self._create_report_stat(self.report_stats, "Frames", "0")
        self.report_frames.pack(fill="x")

        self.report_body = ctk.CTkScrollableFrame(self.results_panel, corner_radius=16, fg_color="#070f1d", border_width=2, border_color="#60a5fa")
        self.report_body.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        self.report_summary_title = ctk.CTkLabel(self.report_body, text="Session Overview", font=ctk.CTkFont(size=16, weight="bold"), text_color="#ffffff")
        self.report_summary_title.pack(anchor="w", padx=14, pady=(14, 8))

        self.report_summary_label = ctk.CTkLabel(
            self.report_body,
            text="No analyses yet.",
            wraplength=300,
            justify="left",
            font=ctk.CTkFont(size=14),
            text_color="#e2e8f0",
        )
        self.report_summary_label.pack(anchor="w", padx=14, pady=(0, 16))

        self.report_tips_title = ctk.CTkLabel(self.report_body, text="Tips", font=ctk.CTkFont(size=16, weight="bold"), text_color="#ffffff")
        self.report_tips_title.pack(anchor="w", padx=14, pady=(0, 8))

        self.report_tips_label = ctk.CTkLabel(
            self.report_body,
            text="- Use a well-lit face for best accuracy.\n- The first run downloads model files automatically.\n- Detection is optimized for frontal faces.",
            wraplength=300,
            justify="left",
            font=ctk.CTkFont(size=13),
            text_color="#cbd5e1",
        )
        self.report_tips_label.pack(anchor="w", padx=14, pady=(0, 16))

        self.quick_stats = ctk.CTkFrame(self.results_panel, corner_radius=18, fg_color="#0d1a2d", border_width=1, border_color="#334155")
        self.quick_stats.pack(fill="x", padx=20, pady=(0, 20))
        self.quick_stat_label = ctk.CTkLabel(self.quick_stats, text="Ready for your first detection.", wraplength=300, justify="left", font=ctk.CTkFont(size=13), text_color="#f8fafc")
        self.quick_stat_label.pack(anchor="w", padx=16, pady=16)

    def _create_report_stat(self, parent, label: str, value: str) -> ctk.CTkFrame:
        chip = ctk.CTkFrame(parent, corner_radius=14, fg_color="#111c31", border_width=1, border_color="#334155")
        ctk.CTkLabel(chip, text=label, font=ctk.CTkFont(size=11), text_color="#93c5fd").pack(anchor="w", padx=12, pady=(10, 0))
        text_widget = ctk.CTkLabel(chip, text=value, font=ctk.CTkFont(size=18, weight="bold"), text_color="#ffffff")
        text_widget.pack(anchor="w", padx=12, pady=(0, 10))
        chip._value_label = text_widget  # keep a reference for updates
        return chip

    def _summary_text(self, body: str) -> str:
        return f"Session Overview\n\n{body}\n\nTips\n- Use a well-lit face for best accuracy.\n- The first run downloads model files automatically.\n- Detection is optimized for frontal faces."

    def _update_status(self, headline: str, detail: str) -> None:
        self.last_status = headline
        self.status_label.configure(text=headline)
        self.detail_label.configure(text=detail)

    def _set_mode(self, mode: str) -> None:
        self.metric_mode.configure(text=f"Mode: {mode}")

    def _update_metrics(self, faces: int) -> None:
        self.metric_faces.configure(text=f"Faces detected: {faces}")
        self.metric_frames.configure(text=f"Frames processed: {self.analysis_count}")
        if hasattr(self, "report_faces"):
            self.report_faces._value_label.configure(text=str(faces))
        if hasattr(self, "report_frames"):
            self.report_frames._value_label.configure(text=str(self.analysis_count))

    def start_camera(self) -> None:
        if self.camera_running:
            return
        self.video_source = 0
        self.camera = cv.VideoCapture(self.video_source)
        if not self.camera.isOpened():
            self.camera = None
            messagebox.showerror("Camera Error", "Unable to open the default camera.")
            return
        self.camera_running = True
        self._set_mode("Camera")
        self._update_status("Camera active", "Streaming live frames from your webcam.")
        if hasattr(self, "report_source"):
            self.report_source._value_label.configure(text="Camera")
        self._start_background_worker()

    def stop_camera(self) -> None:
        self.camera_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self._set_mode("Idle")
        self._update_status("Camera stopped", "You can load an image or restart the webcam.")
        if hasattr(self, "report_source"):
            self.report_source._value_label.configure(text="Idle")

    def load_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not file_path:
            return
        image = cv.imread(file_path)
        if image is None:
            messagebox.showerror("File Error", "Could not read the selected image.")
            return
        self.stop_camera()
        self._set_mode("Image")
        self._update_status("Image loaded", Path(file_path).name)
        if hasattr(self, "report_source"):
            self.report_source._value_label.configure(text="Image")
        self._process_frame(image, source_name=Path(file_path).name)

    def clear_preview(self) -> None:
        self.preview_label.configure(image=None, text="Camera feed will appear here")
        self.preview_image = None
        self.report_summary_label.configure(text="Preview cleared.")
        self.report_tips_label.configure(text="- Use a well-lit face for best accuracy.\n- The first run downloads model files automatically.\n- Detection is optimized for frontal faces.")
        self.quick_stat_label.configure(text="Ready for your first detection.")
        self._update_metrics(0)

    def _start_background_worker(self) -> None:
        if self.processing_thread and self.processing_thread.is_alive():
            return
        self.processing_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.processing_thread.start()

    def _camera_loop(self) -> None:
        while self.camera_running and self.camera is not None:
            ok, frame = self.camera.read()
            if not ok:
                self.after(0, self._update_status, "Camera stream ended", "Unable to read the next frame.")
                break
            self._process_frame(frame, source_name="Webcam")
            time.sleep(0.01)
        self.after(0, self.stop_camera)

    def _process_frame(self, frame: np.ndarray, source_name: str) -> None:
        if self.busy:
            self.current_frame = frame
            return
        self.busy = True
        try:
            analyzed, faces = self.engine.analyze(frame)
            self.analysis_count += 1
            self.after(0, self._render_frame, analyzed, faces, source_name)
        except ModelDownloadError as exc:
            self.after(0, messagebox.showerror, "Model Download Error", str(exc))
            self.after(0, self._update_status, "Model download failed", "Check your internet connection or proxy settings.")
        except Exception as exc:
            self.after(0, messagebox.showerror, "Processing Error", str(exc))
            self.after(0, self._update_status, "Processing error", "The current frame could not be analyzed.")
        finally:
            self.busy = False

    def _render_frame(self, frame: np.ndarray, faces, source_name: str) -> None:
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        target_width = 820
        target_height = int(target_width * image.height / image.width)
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")

        summary_lines = [f"Source: {source_name}", f"Faces detected: {len(faces)}"]
        if faces:
            for index, face in enumerate(faces[:5], start=1):
                x, y, w, h = face.bbox
                summary_lines.append(
                    f"#{index} x={x}, y={y}, w={w}, h={h} | {face.gender} ({face.gender_confidence:.2f}) | {face.age} ({face.age_confidence:.2f})"
                )
        else:
            summary_lines.append("No faces found in this frame.")

        self.report_summary_label.configure(text="\n".join(summary_lines))
        self.quick_stat_label.configure(text=f"{len(faces)} face(s) detected. Last frame processed from {source_name}.")
        self._update_metrics(len(faces))
        self._update_status("Frame analyzed", f"{source_name} processed with OpenCV DNN models.")

    def _ui_loop(self) -> None:
        if self.current_frame is not None and not self.busy:
            frame = self.current_frame
            self.current_frame = None
            self._process_frame(frame, source_name="Webcam")
        self.after(30, self._ui_loop)

    def on_close(self) -> None:
        self.stop_camera()
        self.destroy()


def main() -> None:
    app = FaceSightApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
