from __future__ import annotations

import os
import threading
import tempfile
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2 as cv
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FACE_MODEL_URLS = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
)
AGE_PROTO_URLS = (
    "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
)
AGE_MODEL_URLS = (
    "https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=1",
)
GENDER_PROTO_URLS = (
    "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
)
GENDER_MODEL_URLS = (
    "https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=1",
)

FACE_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
AGE_PROTO_PATH = MODELS_DIR / "age_deploy.prototxt"
AGE_MODEL_PATH = MODELS_DIR / "age_net.caffemodel"
GENDER_PROTO_PATH = MODELS_DIR / "gender_deploy.prototxt"
GENDER_MODEL_PATH = MODELS_DIR / "gender_net.caffemodel"

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_BUCKETS = ["Male", "Female"]
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


@dataclass
class FaceAnalysis:
    bbox: Tuple[int, int, int, int]
    confidence: float
    gender: str
    gender_confidence: float
    age: str
    age_confidence: float


class ModelDownloadError(RuntimeError):
    pass


class FaceAnalysisEngine:
    def __init__(self) -> None:
        self._download_lock = threading.Lock()
        self.face_detector = None
        self.age_net = None
        self.gender_net = None

    def ensure_models(self) -> None:
        with self._download_lock:
            self._download_file(FACE_MODEL_URLS, FACE_MODEL_PATH)
            self._download_file(AGE_PROTO_URLS, AGE_PROTO_PATH)
            self._download_file(AGE_MODEL_URLS, AGE_MODEL_PATH)
            self._download_file(GENDER_PROTO_URLS, GENDER_PROTO_PATH)
            self._download_file(GENDER_MODEL_URLS, GENDER_MODEL_PATH)

    def load(self) -> None:
        self.ensure_models()
        self.face_detector = cv.FaceDetectorYN.create(str(FACE_MODEL_PATH), "", (320, 320), 0.9, 0.3, 5000)
        self.age_net = cv.dnn.readNetFromCaffe(str(AGE_PROTO_PATH), str(AGE_MODEL_PATH))
        self.gender_net = cv.dnn.readNetFromCaffe(str(GENDER_PROTO_PATH), str(GENDER_MODEL_PATH))
        self.age_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.gender_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.face_detector.setScoreThreshold(0.9)

    def analyze(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[FaceAnalysis]]:
        if self.face_detector is None or self.age_net is None or self.gender_net is None:
            self.load()

        frame = frame_bgr.copy()
        height, width = frame.shape[:2]
        self.face_detector.setInputSize((width, height))
        _, faces = self.face_detector.detect(frame)

        results: List[FaceAnalysis] = []
        if faces is None:
            return frame, results

        for face in faces:
            x = max(int(face[0]), 0)
            y = max(int(face[1]), 0)
            w = max(int(face[2]), 1)
            h = max(int(face[3]), 1)
            confidence = float(face[14])
            face_crop = self._safe_crop(frame, x, y, w, h)

            gender, gender_conf = self._predict_gender(face_crop)
            age, age_conf = self._predict_age(face_crop)

            results.append(
                FaceAnalysis(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    gender=gender,
                    gender_confidence=gender_conf,
                    age=age,
                    age_confidence=age_conf,
                )
            )

            self._draw_overlay(frame, x, y, w, h, confidence, gender, gender_conf, age, age_conf)

        return frame, results

    def _predict_gender(self, face_crop: np.ndarray) -> Tuple[str, float]:
        blob = cv.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        self.gender_net.setInput(blob)
        preds = self.gender_net.forward()[0]
        index = int(np.argmax(preds))
        return GENDER_BUCKETS[index], float(preds[index])

    def _predict_age(self, face_crop: np.ndarray) -> Tuple[str, float]:
        blob = cv.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        self.age_net.setInput(blob)
        preds = self.age_net.forward()[0]
        index = int(np.argmax(preds))
        return AGE_BUCKETS[index], float(preds[index])

    def _safe_crop(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w, frame.shape[1])
        y2 = min(y + h, frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return frame
        return crop

    def _draw_overlay(self, frame: np.ndarray, x: int, y: int, w: int, h: int, confidence: float, gender: str, gender_conf: float, age: str, age_conf: float) -> None:
        overlay = frame.copy()
        color = (33, 199, 180)
        accent = (255, 191, 0)
        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv.rectangle(overlay, (x, max(y - 72, 0)), (x + min(max(w, 260), frame.shape[1] - x), y), (18, 23, 38), -1)
        alpha = 0.82
        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label1 = f"{gender} {gender_conf:.2f}"
        label2 = f"Age {age} | Face {confidence:.2f}"
        cv.putText(frame, label1, (x + 8, max(y - 42, 18)), cv.FONT_HERSHEY_SIMPLEX, 0.62, accent, 2, cv.LINE_AA)
        cv.putText(frame, label2, (x + 8, max(y - 16, 36)), cv.FONT_HERSHEY_SIMPLEX, 0.56, (240, 245, 255), 1, cv.LINE_AA)

    def _download_file(self, urls: Sequence[str], target: Path) -> None:
        if self._is_probably_valid_model(target):
            return
        errors = []
        try:
            for url in urls:
                tmp_path = target.parent / f"{target.name}.{uuid.uuid4().hex}.download"
                try:
                    with urllib.request.urlopen(url, timeout=45) as response, open(tmp_path, "wb") as output_file:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            output_file.write(chunk)
                    os.replace(tmp_path, target)
                    return
                except Exception as exc:
                    errors.append(f"{url}: {exc}")
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
            joined_errors = "\n".join(errors)
            raise ModelDownloadError(f"Failed to download {target.name}. Tried:\n{joined_errors}")
        finally:
            for leftover in target.parent.glob(f"{target.name}.*.download"):
                leftover.unlink(missing_ok=True)

    def _is_probably_valid_model(self, target: Path) -> bool:
        if not target.exists():
            return False

        size = target.stat().st_size
        name = target.name.lower()
        if name.endswith(".prototxt"):
            return size >= 1_000
        if name.endswith(".onnx"):
            return size >= 50_000
        if name.endswith(".caffemodel"):
            return size >= 1_000_000
        return size > 0
