"""Translation between OpenAI Chat Completions shapes and Claude Code."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .schemas import ChatCompletionRequest, ChatMessage


class UnsupportedRequest(ValueError):
    """Raised when a request uses a feature the service cannot honour."""


@dataclass(frozen=True)
class ClaudePrompt:
    system: str | None
    prompt: str
    output_schema: dict[str, Any] | None


def _text_of(message: ChatMessage) -> str:
    content = message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for part in content:
        if part.get("type") != "text":
            raise UnsupportedRequest(
                f"Content part type [{part.get('type')}] is not supported; send text only."
            )
        parts.append(str(part.get("text", "")))
    return "\n".join(parts)


def build_prompt(request: ChatCompletionRequest) -> ClaudePrompt:
    """Collapse a stateless Chat Completions request into one Claude Code prompt."""
    if request.tools:
        raise UnsupportedRequest(
            "Client-side tools are not supported; Claude Code runs its own tools. "
            "Remove HasTools from the agent."
        )

    system_parts: list[str] = []
    turns: list[ChatMessage] = []
    for message in request.messages:
        if message.role in ("system", "developer"):
            system_parts.append(_text_of(message))
        elif message.role == "tool" or message.tool_calls:
            raise UnsupportedRequest("Tool call messages are not supported.")
        else:
            turns.append(message)

    if not turns or turns[-1].role != "user":
        raise UnsupportedRequest("The last message must come from the user.")

    current = _text_of(turns[-1])
    history = turns[:-1]

    if history:
        lines = [
            f"{'User' if m.role == 'user' else 'Assistant'}: {_text_of(m)}" for m in history
        ]
        prompt = (
            "<conversation_history>\n"
            + "\n\n".join(lines)
            + "\n</conversation_history>\n\n"
            + current
        )
    else:
        prompt = current

    return ClaudePrompt(
        system="\n\n".join(p for p in system_parts if p) or None,
        prompt=prompt,
        output_schema=_output_schema(request.response_format),
    )


def _output_schema(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    if not response_format or response_format.get("type") == "text":
        return None
    if response_format.get("type") == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        if not isinstance(schema, dict):
            raise UnsupportedRequest("response_format.json_schema.schema is required.")
        return schema
    raise UnsupportedRequest(
        f"response_format type [{response_format.get('type')}] is not supported."
    )


def map_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    usage = usage or {}
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    prompt = int(usage.get("input_tokens") or 0) + cache_read + cache_write
    completion = int(usage.get("output_tokens") or 0)

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cache_read},
    }


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def completion_body(
    completion_id: str,
    model: str,
    content: str,
    finish_reason: str,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": map_usage(usage),
    }


def chunk_body(
    completion_id: str,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def sse(data: dict[str, Any] | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


def error_body(message: str, error_type: str = "server_error") -> dict[str, Any]:
    return {"error": {"type": error_type, "code": error_type, "message": message}}
