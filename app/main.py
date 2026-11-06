"""OpenAI-compatible HTTP API in front of Claude Code."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings
from .runner import ClaudeCodeRunner, ReasoningDelta, RunResult, Runner, TextDelta
from .schemas import ChatCompletionRequest
from .translate import (
    UnsupportedRequest,
    build_prompt,
    chunk_body,
    completion_body,
    error_body,
    map_usage,
    new_completion_id,
    sse,
)

logger = logging.getLogger("claude_code_service")


def create_app(settings: Settings | None = None, runner: Runner | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    runner = runner or ClaudeCodeRunner(settings)
    semaphore = asyncio.Semaphore(settings.max_concurrency)

    app = FastAPI(title="Claude Code Service")

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if not settings.api_key:
            return
        token = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(token, settings.api_key):
            raise HTTPException(status_code=401, detail="Invalid API key.")

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        error_type = "authentication_error" if exc.status_code == 401 else "invalid_request_error"
        return JSONResponse(error_body(str(exc.detail), error_type), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            error_body(json.dumps(exc.errors(), default=str), "invalid_request_error"),
            status_code=422,
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models() -> dict:
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "owned_by": "anthropic"}
                for name in ("sonnet", "opus", "haiku")
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    async def chat_completions(request: ChatCompletionRequest):
        try:
            prompt = build_prompt(request)
        except UnsupportedRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        model = request.model or settings.default_model
        completion_id = new_completion_id()

        if request.stream:
            include_usage = bool((request.stream_options or {}).get("include_usage"))
            return StreamingResponse(
                _stream(runner, semaphore, prompt, model, completion_id, include_usage),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        async with semaphore:
            try:
                result = await _final_result(runner.run(prompt, model, stream=False))
            except Exception as exc:  # noqa: BLE001 - surface SDK failures as API errors
                logger.exception("Claude Code run failed")
                return JSONResponse(error_body(str(exc)), status_code=502)

        if result.error:
            return JSONResponse(error_body(result.error), status_code=502)

        return completion_body(
            completion_id, model, _content(result), result.finish_reason, result.usage
        )

    return app


async def _final_result(events: AsyncIterator) -> RunResult:
    async for event in events:
        if isinstance(event, RunResult):
            return event
    return RunResult("", None, None, "stop", error="Claude Code finished without a result.")


def _content(result: RunResult) -> str:
    if result.structured is not None:
        return json.dumps(result.structured, ensure_ascii=False)
    return result.text


async def _stream(
    runner: Runner,
    semaphore: asyncio.Semaphore,
    prompt,
    model: str,
    completion_id: str,
    include_usage: bool,
) -> AsyncIterator[str]:
    async with semaphore:
        yield sse(chunk_body(completion_id, model, {"role": "assistant", "content": ""}))
        streamed_text = False

        try:
            async for event in runner.run(prompt, model, stream=True):
                if isinstance(event, ReasoningDelta):
                    yield sse(chunk_body(completion_id, model, {"reasoning_content": event.text}))
                    continue

                if isinstance(event, TextDelta):
                    streamed_text = True
                    yield sse(chunk_body(completion_id, model, {"content": event.text}))
                    continue

                if event.error:
                    yield sse(error_body(event.error))
                    return

                if not streamed_text and (content := _content(event)):
                    yield sse(chunk_body(completion_id, model, {"content": content}))

                yield sse(chunk_body(completion_id, model, {}, event.finish_reason))

                if include_usage:
                    usage_chunk = chunk_body(completion_id, model, {})
                    usage_chunk["choices"] = []
                    usage_chunk["usage"] = map_usage(event.usage)
                    yield sse(usage_chunk)

                yield sse("[DONE]")
                return
        except Exception as exc:  # noqa: BLE001 - stream errors in-band
            logger.exception("Claude Code stream failed")
            yield sse(error_body(str(exc)))
            return

        yield sse(error_body("Claude Code finished without a result."))
