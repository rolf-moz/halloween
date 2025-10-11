from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from .config import CONFIG, WhisperConfig


log = logging.getLogger(__name__)


@dataclass
class WhisperResult:
    text: str


class WhisperClient:
    def __init__(self, config: WhisperConfig | None = None) -> None:
        self.config = config or CONFIG.whisper
        api_key = self.config.api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY or WHISPER_API_KEY is required for transcription")
        self.client = OpenAI(api_key=api_key)

    def transcribe_wav(self, wav_bytes: bytes) -> Optional[WhisperResult]:
        buffer = io.BytesIO(wav_bytes)
        buffer.name = "audio.wav"
        try:
            response = self.client.audio.transcriptions.create(
                model=self.config.model,
                file=buffer,
                language="en"
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Whisper request failed: %s", exc)
            return None
        text = getattr(response, "text", None)
        if not text:
            log.warning("Whisper response missing text")
            return None
        cleaned = text.strip()
        if not cleaned:
            return None
        return WhisperResult(text=cleaned)
