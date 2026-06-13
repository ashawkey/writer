"""Load provider config from ./secret.yaml and build OpenAI-compatible clients.

secret.yaml shape:

    openai:
      deepseek:
        model: deepseek-v4-pro
        api_key: sk-...
        base_url: https://api.deepseek.com/v1
      deepseek-flash:
        model: deepseek-v4-flash
        api_key: sk-...
        base_url: https://api.deepseek.com/v1

Any OpenAI-compatible provider works — just point base_url/api_key/model at it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from openai import OpenAI


@dataclass
class Provider:
    name: str
    model: str
    api_key: str
    base_url: str

    def client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)


def load_providers(path: str | Path = "secret.yaml") -> dict[str, Provider]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy secret.yaml.example and fill in your provider."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("openai") or {}
    if not raw:
        raise ValueError(f"No providers under 'openai:' in {path}.")
    providers: dict[str, Provider] = {}
    for name, cfg in raw.items():
        missing = [k for k in ("model", "api_key", "base_url") if not cfg.get(k)]
        if missing:
            raise ValueError(f"Provider {name!r} missing keys: {', '.join(missing)}")
        providers[name] = Provider(
            name=name,
            model=cfg["model"],
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
        )
    return providers


def get_provider(name: str | None = None, path: str | Path = "secret.yaml") -> Provider:
    providers = load_providers(path)
    if name is None:
        return next(iter(providers.values()))
    if name not in providers:
        raise KeyError(f"Provider {name!r} not in {path}. Have: {', '.join(providers)}")
    return providers[name]
