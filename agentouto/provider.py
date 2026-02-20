from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Provider:
    name: str
    kind: Literal["openai", "anthropic", "google"]
    api_key: str
    base_url: str | None = None
