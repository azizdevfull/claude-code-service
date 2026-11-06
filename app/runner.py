"""Runs prompts through Claude Code via the official Agent SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol, Union

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from claude_agent_sdk.types import StreamEvent

from .config import Settings
from .translate import ClaudePrompt


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True)
class RunResult:
    text: str
    structured: Any
    usage: dict[str, Any] | None
    finish_reason: str
    error: str | None = None


RunEvent = Union[TextDelta, ReasoningDelta, RunResult]


class Runner(Protocol):
    def run(self, prompt: ClaudePrompt, model: str, stream: bool) -> AsyncIterator[RunEvent]: ...


class ClaudeCodeRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def options(self, prompt: ClaudePrompt, model: str, stream: bool) -> ClaudeAgentOptions:
        settings = self.settings

        if settings.system_prompt_mode == "replace" and prompt.system:
            system_prompt: Any = prompt.system
        else:
            system_prompt = {"type": "preset", "preset": "claude_code"}
            if prompt.system:
                system_prompt["append"] = prompt.system

        return ClaudeAgentOptions(
            model=model,
            system_prompt=system_prompt,
            cwd=settings.cwd,
            allowed_tools=settings.allowed_tools,
            permission_mode=settings.permission_mode,
            max_turns=settings.max_turns,
            setting_sources=settings.setting_sources,
            include_partial_messages=stream,
            thinking=settings.thinking,
            effort=settings.effort,
            output_format=(
                {"type": "json_schema", "schema": prompt.output_schema}
                if prompt.output_schema
                else None
            ),
        )

    async def run(self, prompt: ClaudePrompt, model: str, stream: bool) -> AsyncIterator[RunEvent]:
        options = self.options(prompt, model, stream)
        emitted_text = False
        pending_separator = False

        async for message in query(prompt=prompt.prompt, options=options):
            if isinstance(message, StreamEvent):
                if message.parent_tool_use_id is not None or prompt.output_schema:
                    continue

                event = message.event
                if event.get("type") == "content_block_start":
                    if (event.get("content_block") or {}).get("type") == "text":
                        pending_separator = emitted_text
                elif event.get("type") == "content_block_delta":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "thinking_delta" and delta.get("thinking"):
                        yield ReasoningDelta(delta["thinking"])
                    elif delta.get("type") == "text_delta" and delta.get("text"):
                        text = ("\n\n" if pending_separator else "") + delta["text"]
                        pending_separator = False
                        emitted_text = True
                        yield TextDelta(text)

            elif isinstance(message, ResultMessage):
                yield _to_result(message)
                return

        yield RunResult("", None, None, "stop", error="Claude Code finished without a result.")


def _to_result(message: ResultMessage) -> RunResult:
    if message.subtype == "error_max_turns":
        return RunResult(message.result or "", message.structured_output, message.usage, "length")

    if message.is_error or message.subtype != "success":
        detail = "; ".join(message.errors or []) or message.result or message.subtype
        return RunResult("", None, message.usage, "stop", error=f"Claude Code error: {detail}")

    return RunResult(message.result or "", message.structured_output, message.usage, "stop")
