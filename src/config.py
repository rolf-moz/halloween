from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class WhisperConfig:
    model: str = os.getenv("WHISPER_MODEL", "whisper-1")
    sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", 16000))
    chunk_duration_ms: int = int(os.getenv("VOICE_CHUNK_DURATION_MS", 30))
    vad_sensitivity: int = int(os.getenv("VAD_SENSITIVITY", 0))
    silence_timeout: float = float(os.getenv("VOICE_SILENCE_TIMEOUT", 1.0))
    min_voice_ms: int = int(os.getenv("VOICE_MIN_MS", 1000))
    history_seconds: int = int(os.getenv("TRANSCRIPT_HISTORY_SECONDS", 30))
    api_key: str | None = os.getenv("WHISPER_API_KEY") or os.getenv("OPENAI_API_KEY")
    input_device: str | None = os.getenv("VOICE_INPUT_DEVICE")


@dataclass
class CameraConfig:
    capture_interval: float = float(os.getenv("CAPTURE_INTERVAL_SECONDS", 10.0))
    detection_scale_factor: float = float(os.getenv("DETECTION_SCALE_FACTOR", 1.2))
    detection_min_neighbors: int = int(os.getenv("DETECTION_MIN_NEIGHBORS", 5))
    output_dir: Path = Path(os.getenv("CAPTURE_OUTPUT_DIR", "captures"))

"""
            '''"You are the spirit of this haunted house. Narrate who you see "
            "in the uploaded image using archaic, eerie language. Reference "
            "recent mortal speech when relevant. Keep it under 20 words."
            '''

"""
@dataclass
class GPTConfig:
    model: str = os.getenv("OPENAI_GPT_MODEL", "gpt-4.1-mini")
    prompt: str = os.getenv(
        "SPOOKY_PROMPT",
        (
            '''You are Damon the spirit of this haunted house. Narrate who you see
            in the uploaded image using eerie, funny, G-rated language suitable for 1st graders.
            Reference recent mortal speech when relevant. Keep it under 20 words and throw in
            an evil laugh. Answer questions if any. If there is a lull in the conversation you can pose a riddle.
            '''

        ),
    )
    api_key: str | None = os.getenv("OPENAI_API_KEY")


@dataclass
class ElevenLabsConfig:
    voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "XrExE9yKIg1WjnnlVkGX")
    model: str = os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2")
    stability: float = float(os.getenv("ELEVENLABS_STABILITY", 0.15))
    similarity_boost: float = float(os.getenv("ELEVENLABS_SIMILARITY", 0.7))
    style: float = float(os.getenv("ELEVENLABS_STYLE", 0.45))
    use_speaker_boost: bool = os.getenv("ELEVENLABS_SPEAKER_BOOST", "true").lower() == "true"
    api_key: str | None = os.getenv("ELEVENLABS_API_KEY")


@dataclass
class AppConfig:
    whisper: WhisperConfig = WhisperConfig()
    camera: CameraConfig = CameraConfig()
    gpt: GPTConfig = GPTConfig()
    elevenlabs: ElevenLabsConfig = ElevenLabsConfig()

    def ensure_dirs(self) -> None:
        self.camera.output_dir.mkdir(parents=True, exist_ok=True)


CONFIG = AppConfig()
CONFIG.ensure_dirs()
