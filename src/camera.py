from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2

from .config import CONFIG, CameraConfig

log = logging.getLogger(__name__)


@dataclass
class CameraEvent:
    image_path: Path
    timestamp: float
    faces: Sequence[tuple[int, int, int, int]]


class CameraWatcher:
    def __init__(
        self,
        on_event: Callable[[CameraEvent], None],
        config: CameraConfig | None = None,
        camera_index: int = 0,
    ) -> None:
        self.config = config or CONFIG.camera
        self.on_event = on_event
        self.camera_index = camera_index
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.capture: cv2.VideoCapture | None = None
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

    def start(self) -> None:
        self.stop_event.clear()
        if self.capture is None:
            backend = getattr(cv2, 'CAP_AVFOUNDATION', None)
            capture = cv2.VideoCapture(self.camera_index, backend) if backend is not None else cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(self.camera_index)
            self.capture = capture
        if not self.capture.isOpened():
            raise RuntimeError("Unable to open camera")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        log.info("Camera watcher started")

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        log.info("Camera watcher stopped")

    def _run(self) -> None:
        assert self.capture is not None
        interval = self.config.capture_interval
        while not self.stop_event.is_set():
            ret, frame = self.capture.read()
            if not ret:
                log.warning("Camera frame grab failed")
                time.sleep(interval)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=self.config.detection_scale_factor,
                minNeighbors=self.config.detection_min_neighbors,
            )
            if len(faces) > 0:
                timestamp = time.time()
                filename = f"capture_{int(timestamp)}.jpg"
                path = self.config.output_dir / filename
                cv2.imwrite(str(path), frame)
                event = CameraEvent(image_path=path, timestamp=timestamp, faces=list(map(tuple, faces)))
                try:
                    self.on_event(event)
                except Exception as exc:  # noqa: BLE001
                    log.error("Camera event callback failed: %s", exc)
            time.sleep(interval)
