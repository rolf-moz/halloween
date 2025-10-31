from __future__ import annotations

import collections
import io
import logging
import queue
import threading
import time
import wave
from dataclasses import dataclass
from typing import Deque, Optional

import sounddevice as sd
import webrtcvad

from .config import CONFIG, WhisperConfig
from .whisper_client import WhisperClient

log = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    timestamp: float
    text: str


class AudioListener:
    def __init__(self, whisper_client: WhisperClient, config: WhisperConfig | None = None) -> None:
        self.config = config or CONFIG.whisper
        self.whisper_client = whisper_client
        self.vad = webrtcvad.Vad(self.config.vad_sensitivity)
        self.sample_rate = self.config.sample_rate
        self.frame_duration_ms = self.config.chunk_duration_ms
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.processor_thread: threading.Thread | None = None
        self.stream: Optional[sd.RawInputStream] = None
        self.transcripts: Deque[TranscriptSegment] = collections.deque()
        self.transcript_lock = threading.Lock()
        self.input_device = self._resolve_input_device(self.config.input_device)

    def start(self) -> None:
        self.stop_event.clear()
        try:
            self.stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.frame_size,
                dtype="int16",
                channels=1,
                device=self.input_device,
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to start audio stream: %s", exc)
            raise
        self.processor_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processor_thread.start()
        log.info("Audio listener started")

    def stop(self) -> None:
        self.stop_event.set()
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("Error stopping audio stream: %s", exc)
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2.0)
        log.info("Audio listener stopped")

    def _audio_callback(self, indata: bytes, frames: int, time_info, status) -> None:  # type: ignore[override]
        if status:
            log.debug("Audio callback status: %s", status)
        if self.pause_event.is_set():
            return
        self.audio_queue.put(bytes(indata))

    def _process_frames(self) -> None:
        voiced_frames: list[bytes] = []
        activation_frames = max(1, self.config.activation_ms // self.frame_duration_ms)
        pending_frames: Deque[bytes] = collections.deque(maxlen=activation_frames)
        voice_active = False
        speech_run = 0
        last_voice_time = 0.0
        min_frames = max(1, self.config.min_voice_ms // self.frame_duration_ms)
        silence_limit = self.config.silence_timeout
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                voiced_frames.clear()
                pending_frames.clear()
                speech_run = 0
                time.sleep(0.05)
                continue
            try:
                frame = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                frame = None
            now = time.time()
            if frame is None:
                pending_frames.clear()
                speech_run = 0
                if voice_active and now - last_voice_time > silence_limit:
                    self._flush_frames(voiced_frames, min_frames)
                    voiced_frames.clear()
                    voice_active = False
                continue
            is_speech = False
            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
            except Exception as exc:  # noqa: BLE001
                log.debug("VAD failed on frame: %s", exc)
            if is_speech:
                speech_run += 1
                if voice_active:
                    voiced_frames.append(frame)
                    last_voice_time = now
                else:
                    pending_frames.append(frame)
                    if speech_run >= activation_frames:
                        voice_active = True
                        voiced_frames.extend(pending_frames)
                        pending_frames.clear()
                        last_voice_time = now
                continue
            speech_run = 0
            pending_frames.clear()
            if voice_active:
                voiced_frames.append(frame)
                if now - last_voice_time > silence_limit:
                    self._flush_frames(voiced_frames, min_frames)
                    voiced_frames.clear()
                    voice_active = False
        if voiced_frames:
            self._flush_frames(voiced_frames, min_frames)

    def _resolve_input_device(self, desired: str | None) -> Optional[int]:
        if not desired:
            return None
        desired = desired.strip()
        if not desired:
            return None
        try:
            index = int(desired)
        except ValueError:
            index = None
        if index is not None:
            try:
                info = sd.query_devices(index, "input")
            except Exception as exc:  # noqa: BLE001
                log.warning("Input device #%s unavailable; using system default (%s)", index, exc)
                return None
            log.info("Using audio input device #%d: %s", index, info.get("name", "unknown"))
            return index
        try:
            devices = sd.query_devices()
        except Exception as exc:  # noqa: BLE001
            log.warning("Unable to enumerate audio devices; using default input (%s)", exc)
            return None
        needle = desired.lower()
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) <= 0:
                continue
            name = str(dev.get("name", ""))
            if name.lower() == needle:
                log.info("Using audio input device '%s' (#%d)", name, idx)
                return idx
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) <= 0:
                continue
            name = str(dev.get("name", ""))
            if needle in name.lower():
                log.info("Using audio input device '%s' (#%d)", name, idx)
                return idx
        log.warning("Audio input device '%s' not found; using system default", desired)
        log.info("Run `python -m sounddevice` to list available devices")
        return None

    def pause(self) -> None:
        if not self.pause_event.is_set():
            log.debug("Pausing audio capture")
            self.pause_event.set()
            self._clear_pending_audio()

    def resume(self) -> None:
        if self.pause_event.is_set():
            log.debug("Resuming audio capture")
            self.pause_event.clear()

    def _clear_pending_audio(self) -> None:
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def _flush_frames(self, frames: list[bytes], min_frames: int) -> None:
        if len(frames) < min_frames:
            frames.clear()
            return
        wav_bytes = self._frames_to_wav(frames)
        frames.clear()
        print("transcribing")
        result = self.whisper_client.transcribe_wav(wav_bytes)
        if not result:
            return
        self._append_transcript(result.text)

    def _frames_to_wav(self, frames: list[bytes]) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wave_file:
            wave_file.setnchannels(1)
            wave_file.setsampwidth(2)
            wave_file.setframerate(self.sample_rate)
            for frame in frames:
                wave_file.writeframes(frame)
        return buffer.getvalue()

    def _append_transcript(self, text: str) -> None:
        entry = TranscriptSegment(timestamp=time.time(), text=text.strip())
        with self.transcript_lock:
            self.transcripts.append(entry)
            cutoff = entry.timestamp - self.config.history_seconds
            while self.transcripts and self.transcripts[0].timestamp < cutoff:
                self.transcripts.popleft()
        log.info("Captured transcript: %s", entry.text)

    def get_recent_transcript(self, window: float | None = None) -> str:
        cutoff = time.time() - (window if window is not None else self.config.history_seconds)
        with self.transcript_lock:
            recent = [segment.text for segment in self.transcripts if segment.timestamp >= cutoff]
            self.transcripts.clear()
        return "\n".join(recent)
