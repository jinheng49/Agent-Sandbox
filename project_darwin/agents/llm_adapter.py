import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI


@dataclass(slots=True)
class LLMAdapter:
    """OpenAI-compatible adapter that works with third-party API endpoints."""

    model_name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 20.0
    temperature: float = 0.2
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._load_local_env_file()
        self.model_name = self.model_name or os.getenv("DARWIN_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.base_url = self.base_url or os.getenv("DARWIN_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self.api_key = self.api_key or os.getenv("DARWIN_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self.is_configured():
            raise RuntimeError("LLM API key is not configured")

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Model returned empty content")
        return content

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_seconds)
        return self._client

    def _load_local_env_file(self) -> None:
        if os.getenv("DARWIN_LLM_API_KEY"):
            return

        env_path = Path(__file__).resolve().parents[2] / ".env.llm"
        if not env_path.exists():
            return

        loaded_values: dict[str, str] = {}
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            for loaded_key, loaded_value in loaded_values.items():
                value = value.replace(f"${loaded_key}", loaded_value)
            loaded_values[key] = value
            os.environ.setdefault(key, value)
