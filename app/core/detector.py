"""Face detection.

YOLOv8n-face is the default.  Measured on the deployment GPU against YuNet, the
other candidate, over the resolutions this corridor geometry actually needs:

    resolution     YOLOv8n-face (CUDA)    YuNet (OpenCV DNN, CPU)
    640x360               6.2 ms                  8.8 ms
    1280x736              7.5 ms                 46.8 ms
    1920x1088             ~8 ms                 139.0 ms

Both found 270/270 faces in the enrolment gallery with near-identical box sizes,
so accuracy did not separate them and speed did.

The YuNet implementation was removed on 2026-08-25 along with its weights: it
was never selected (`detector_kind` has always been "yolo"), it only made sense
on a CPU-only host, and this pipeline already refuses to start without CUDA.
Restoring it means re-adding a class here and downloading
`face_detection_yunet_2023mar.onnx` from OpenCV Zoo.

This detector runs only during ENROLMENT, from photographs. The live pipeline
tracks and aligns from head boxes and never calls it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    box: np.ndarray   # (4,) xyxy in original-frame pixels
    score: float


class OnnxFaceDetector:
    """YOLOv8-face detector running on ONNX Runtime with CUDA acceleration."""

    def __init__(self, model_path: str | Path, imgsz: int = 640, conf: float = 0.35, iou: float = 0.45, providers=None):
        import logging
        import onnxruntime as ort
        from app.core.onnx_env import best_providers, preload_cuda_libs
        preload_cuda_libs()

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=opts,
                                            providers=providers or best_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self._blob = np.empty((1, 3, self.imgsz, self.imgsz), np.float32)
        self._canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        self._rgb = np.empty((self.imgsz, self.imgsz, 3), np.uint8)

        log = logging.getLogger(__name__)
        log.info("YOLO face detection initialized on ONNX Runtime (%s, imgsz=%d)",
                 self.session.get_providers()[0], self.imgsz)

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2

        self._canvas[:] = 114
        cv2.resize(frame_bgr, (nw, nh), dst=self._canvas[top:top + nh, left:left + nw],
                   interpolation=cv2.INTER_LINEAR)
        cv2.cvtColor(self._canvas, cv2.COLOR_BGR2RGB, dst=self._rgb)
        np.divide(self._rgb.transpose(2, 0, 1), 255.0, out=self._blob[0], casting="unsafe")

        out = self.session.run(None, {self.input_name: self._blob})[0]
        pred = out[0].T  # (8400, 20)
        boxes_c = pred[:, :4]
        conf = pred[:, 4]

        keep = conf >= self.conf
        if not keep.any():
            return []
        b_k = boxes_c[keep]
        c_k = conf[keep]

        xy = np.empty_like(b_k)
        xy[:, 0] = b_k[:, 0] - b_k[:, 2] / 2
        xy[:, 1] = b_k[:, 1] - b_k[:, 3] / 2
        xy[:, 2] = b_k[:, 0] + b_k[:, 2] / 2
        xy[:, 3] = b_k[:, 1] + b_k[:, 3] / 2
        xy[:, [0, 2]] -= left
        xy[:, [1, 3]] -= top
        xy /= r

        rects = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in xy]
        idx = cv2.dnn.NMSBoxes(rects, c_k.tolist(), self.conf, self.iou)
        if len(idx) == 0:
            return []
        flat_idx = [int(i) for i in np.array(idx).ravel()]
        return [Detection(xy[i], float(c_k[i])) for i in flat_idx]


class YoloFaceDetector:
    def __init__(self, model_path: str | Path, imgsz: int = 1280, conf: float = 0.35, device: int = 0):
        path = Path(model_path)
        # Check if an ONNX version of this model is available for full GPU acceleration
        onnx_candidate = path if str(path).endswith(".onnx") else path.with_suffix(".onnx")
        if onnx_candidate.exists():
            self._impl = OnnxFaceDetector(onnx_candidate, imgsz=min(imgsz, 640), conf=conf)
            return

        self._impl = None
        import logging
        from ultralytics import YOLO

        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        self.model = YOLO(str(model_path))
        import torch
        actual_device = "cpu"
        if not (str(device).lower() == "cpu" or not torch.cuda.is_available() or (isinstance(device, int) and device < 0)):
            target = f"cuda:{device}" if isinstance(device, int) else str(device)
            try:
                # Validate that PyTorch has compiled GPU kernels for this device architecture (e.g. sm_120)
                _test = torch.zeros(1, device=target) + 1
                self.model.to(target)
                actual_device = target
            except Exception as exc:
                import logging as sys_logging
                sys_logging.info(
                    "CUDA available but PyTorch kernel not compiled for this GPU architecture: %s. "
                    "YOLO face detection using CPU; ONNX aligner/recognizer remain GPU-accelerated.", exc
                )
                self.model.to("cpu")
                actual_device = "cpu"
        else:
            self.model.to("cpu")
        self.imgsz = imgsz
        self.conf = conf
        self.device = actual_device

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if self._impl is not None:
            return self._impl.detect(frame_bgr)
        r = self.model.predict(
            frame_bgr, verbose=False, device=self.device, imgsz=self.imgsz, conf=self.conf
        )[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        return [Detection(b, float(s)) for b, s in zip(boxes, scores)]


def build_detector(kind: str, model_path, **kw):
    path = Path(model_path)
    if str(path).endswith(".onnx") and path.exists():
        return OnnxFaceDetector(path, **kw)
    onnx_path = path.with_suffix(".onnx")
    if onnx_path.exists():
        return OnnxFaceDetector(onnx_path, **kw)
    if kind == "yolo":
        return YoloFaceDetector(model_path, **kw)
    raise ValueError(
        f"unknown detector: {kind!r}. Only 'yolo' is available; the YuNet "
        "path and its weights were removed as unused."
    )
