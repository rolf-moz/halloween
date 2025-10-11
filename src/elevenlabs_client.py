from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import requests

from .config import CONFIG, ElevenLabsConfig

log = logging.getLogger(__name__)


class SpookyVoice:
    def __init__(self, config: ElevenLabsConfig | None = None) -> None:
        self.config = config or CONFIG.elevenlabs
        if not self.config.api_key:
            raise ValueError("ELEVENLABS_API_KEY is required for text-to-speech")
        self.base_url = "https://api.elevenlabs.io/v1"

    def speak(self, text: str) -> bool:
        if not text.strip():
            return False
        try:
            audio = self._synthesize(text)
        except Exception as exc:  # noqa: BLE001
            log.error("ElevenLabs generation failed: %s", exc)
            return False
        if not self._play_audio(audio):
            return False
        return True

    def _voice_settings(self) -> dict[str, float | bool]:
        return {
            "stability": self.config.stability,
            "similarity_boost": self.config.similarity_boost,
            "style": self.config.style,
            "use_speaker_boost": self.config.use_speaker_boost,
        }

    def _synthesize(self, text: str) -> bytes:
        url = f"{self.base_url}/text-to-speech/{self.config.voice_id}"
        headers = {
            "xi-api-key": self.config.api_key or "",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": self.config.model,
            "voice_settings": self._voice_settings(),
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.content

    def _play_audio(self, audio: bytes) -> bool:
        play_helper = None
        try:  # delayed import to avoid hard dependency during testing
            from elevenlabs import play as imported_play  # type: ignore import-outside-toplevel
            if callable(imported_play):
                play_helper = imported_play
        except Exception as exc:  # noqa: BLE001
            log.debug("ElevenLabs play helper unavailable: %s", exc)
        if play_helper is not None:
            try:
                play_helper(audio)
                return True
            except Exception as exc:  # noqa: BLE001
                log.debug("ElevenLabs helper playback failed: %s", exc)
        return self._play_with_afplay(audio)

    def _play_with_afplay(self, audio: bytes) -> bool:
        try:
            with tempfile.NamedTemporaryFile(prefix="elevenlabs_", suffix=".mp3", delete=False) as tmp:
                tmp.write(audio)
                tmp_path = Path(tmp.name)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to stage audio for playback: %s", exc)
            return False
        try:
            result = subprocess.run(["afplay", str(tmp_path)], check=False, capture_output=True)
            if result.returncode != 0:
                log.error("afplay exited with status %s", result.returncode)
                return False
            return True
        except FileNotFoundError:
            log.error("afplay not found; install Xcode command line tools or provide another player")
            return False
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
