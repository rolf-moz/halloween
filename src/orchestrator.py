from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Optional

from .audio_listener import AudioListener
from .camera import CameraEvent, CameraWatcher
from .elevenlabs_client import SpookyVoice
from .gpt_client import GPTClient
from .whisper_client import WhisperClient

log = logging.getLogger(__name__)


class HalloweenOrchestrator:
    def __init__(self) -> None:
        self.whisper = WhisperClient()
        self.audio = AudioListener(self.whisper)
        self.voice = SpookyVoice()
        self.gpt = GPTClient()
        self.camera_events: queue.Queue[CameraEvent] = queue.Queue()
        self.camera = CameraWatcher(self._queue_camera_event)
        self.worker_stop = threading.Event()
        self.worker_thread: threading.Thread | None = None

        self.inactivity_reset_seconds = 40.0
        self.max_history_entries = 14
        self.events_since_reset = 0
        self.images_sent = 0
        self.last_event_time: float | None = None
        self.conversation: list[dict[str, object]] = []
        self._reset_conversation()

    def start(self) -> None:
        log.info("Starting Halloween orchestrator")
        self.worker_stop.clear()
        try:
            self.audio.start()
            self.camera.start()
        except Exception:  # noqa: BLE001
            log.exception("Failed to start hardware; shutting down")
            self.stop()
            raise
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        log.info("Halloween orchestrator running")

    def stop(self) -> None:
        self.worker_stop.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None
        try:
            self.camera.stop()
        except Exception:  # noqa: BLE001
            log.exception("Camera shutdown failed")
        try:
            self.audio.stop()
        except Exception:  # noqa: BLE001
            log.exception("Audio shutdown failed")
        log.info("Halloween orchestrator stopped")

    def _queue_camera_event(self, event: CameraEvent) -> None:
        log.info("Camera detected %d face(s)", len(event.faces))
        self.camera_events.put(event)

    def _worker_loop(self) -> None:
        while not self.worker_stop.is_set():
            try:
                event = self.camera_events.get(timeout=0.5)
            except queue.Empty:
                continue
            event = self._consume_backlog(event)
            self._handle_event(event)

    def _consume_backlog(self, first_event: CameraEvent) -> CameraEvent:
        latest = first_event
        discarded = 0
        while True:
            try:
                latest = self.camera_events.get_nowait()
                discarded += 1
            except queue.Empty:
                break
        if discarded:
            log.debug("Dropped %d queued camera event(s), keeping most recent", discarded)
        return latest

    def _handle_event(self, event: CameraEvent) -> None:
        if self.events_since_reset > 20:
            log.info("Resetting conversation after 20 events")
            self._reset_conversation()
        print(f"Time since last recorded event {self.last_event_time and event.timestamp - self.last_event_time} num events {self.events_since_reset}")
        if (self.last_event_time and event.timestamp - self.last_event_time > self.inactivity_reset_seconds):
            log.info("Resetting conversation after %.1fs of inactivity", event.timestamp - self.last_event_time)
            self._reset_conversation()
        self.last_event_time = event.timestamp
        transcript = self.audio.get_recent_transcript()
        include_image = self.events_since_reset % 3 == 0 and self.images_sent < 2
        response = None
        user_text = ""
        if transcript and len(transcript.strip()) > 0:
            user_text = transcript #self.gpt.build_user_text(transcript)
        else:
            if not include_image:
                print("No image or text. Skipping")
                return
        if event.image_path and include_image:
            self.images_sent += 1
        messages = self._prepare_messages(user_text, event.image_path if include_image else None)
        response = self.gpt.generate(messages)
        self.events_since_reset += 1
        if not response:
            log.warning("GPT did not return text for %s", event.image_path)
            return
        self._record_conversation(user_text, response)
        self.audio.pause()
        try:
            self.voice.speak(response)
        finally:
            self.audio.resume()

    def _prepare_messages(self, user_text: str, image_path: Optional[Path]) -> list[dict[str, object]]:
        messages = list(self.conversation)
        content: list[dict[str, object]] = [{"type": "input_text", "text": user_text}]
        if image_path:
            try:
                image_url = self.gpt.encode_image_data_url(image_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to encode image %s: %s", image_path, exc)
            else:
                content.append({"type": "input_image", "image_url": image_url})
        messages.append({"role": "user", "content": content})
        return messages

    def _record_conversation(self, user_text: str, assistant_text: str) -> None:
        self.conversation.append({"role": "user", "content": [{"type": "input_text", "text": user_text}]})
        self.conversation.append({"role": "assistant", "content": [{"type": "output_text", "text": assistant_text}]})
        if len(self.conversation) > self.max_history_entries:
            system_message = self.conversation[0]
            tail = self.conversation[-(self.max_history_entries - 1) :]
            self.conversation = [system_message] + tail

    def _reset_conversation(self) -> None:
        self.conversation = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": self.gpt.config.prompt}],
            }
        ]
        self.events_since_reset = 0
        log.debug("GPT conversation state reset")
