"""Service configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    api_key: str | None = None
    default_model: str = "sonnet"
    cwd: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    permission_mode: str = "dontAsk"
    max_turns: int = 10
    setting_sources: list[str] = field(default_factory=list)
    system_prompt_mode: str = "append"
    max_concurrency: int = 4
    thinking: dict[str, Any] | None = None
    effort: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        mode = os.getenv("CLAUDE_SYSTEM_PROMPT_MODE", "append")
        if mode not in ("append", "replace"):
            raise ValueError("CLAUDE_SYSTEM_PROMPT_MODE must be 'append' or 'replace'.")

        return cls(
            api_key=os.getenv("SERVICE_API_KEY") or None,
            default_model=os.getenv("CLAUDE_DEFAULT_MODEL", "sonnet"),
            cwd=os.getenv("CLAUDE_CWD") or None,
            allowed_tools=_csv(os.getenv("CLAUDE_ALLOWED_TOOLS", "Read,Glob,Grep")),
            permission_mode=os.getenv("CLAUDE_PERMISSION_MODE", "dontAsk"),
            max_turns=_int("CLAUDE_MAX_TURNS", 10),
            setting_sources=_csv(os.getenv("CLAUDE_SETTING_SOURCES", "")),
            system_prompt_mode=mode,
            max_concurrency=_int("MAX_CONCURRENCY", 4),
            thinking=_thinking(os.getenv("CLAUDE_THINKING", "")),
            effort=os.getenv("CLAUDE_EFFORT") or None,
        )


def _thinking(value: str) -> dict[str, Any] | None:
    """Map CLAUDE_THINKING (empty | adaptive | disabled | <budget tokens>) to a ThinkingConfig."""
    value = value.strip()
    if not value:
        return None
    if value == "disabled":
        return {"type": "disabled"}
    if value == "adaptive":
        return {"type": "adaptive", "display": "summarized"}
    if value.isdigit():
        return {"type": "enabled", "budget_tokens": int(value), "display": "summarized"}
    raise ValueError("CLAUDE_THINKING must be empty, 'adaptive', 'disabled' or a token budget.")
