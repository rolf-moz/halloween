from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from .config import CONFIG, GPTConfig

log = logging.getLogger(__name__)


class GPTClient:
    def __init__(self, config: GPTConfig | None = None) -> None:
        self.config = config or CONFIG.gpt
        if not self.config.api_key:
            raise ValueError("OPENAI_API_KEY is required for GPT interactions")
        self.client = OpenAI(api_key=self.config.api_key)

    def _encode_image(self, image_path: Path) -> str:
        data = image_path.read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def encode_image_data_url(self, image_path: Path) -> str:
        encoded = self._encode_image(image_path)
        ext = image_path.suffix.lower().lstrip(".")
        if ext in {"jpg", "jpeg", ""}:
            mime = "jpeg"
        elif ext in {"png", "webp", "gif", "bmp"}:
            mime = ext
        else:
            mime = "jpeg"
        return f"data:image/{mime};base64,{encoded}"

    def build_user_text(self, transcript: str) -> str:
        trimmed = transcript.strip()
        if not trimmed:
            trimmed = "No recent speech was captured."
        return f"Recent speech:\n{trimmed}"

    def generate(self, messages: list[dict[str, Any]]) -> Optional[str]:
        try:
            response = self.client.responses.create(model=self.config.model, input=messages)
        except Exception as exc:  # noqa: BLE001
            log.error("GPT request failed: %s", exc)
            return None
        text = response.output_text
        if not text:
            log.warning("GPT response empty")
            return None
        return text.strip()
