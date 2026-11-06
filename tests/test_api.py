import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.runner import ReasoningDelta, RunResult, TextDelta
from app.translate import ClaudePrompt


class FakeRunner:
    def __init__(self, events):
        self.events = events
        self.calls: list[tuple[ClaudePrompt, str, bool]] = []

    async def run(self, prompt, model, stream):
        self.calls.append((prompt, model, stream))
        for event in self.events:
            yield event


USAGE = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3}


def make_client(events, api_key="secret"):
    runner = FakeRunner(events)
    app = create_app(Settings(api_key=api_key), runner)
    return TestClient(app), runner


def auth():
    return {"Authorization": "Bearer secret"}


def test_rejects_missing_api_key():
    client, _ = make_client([])

    response = client.post("/v1/chat/completions", json={"messages": []})

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_returns_chat_completion_and_maps_system_and_history():
    client, runner = make_client([RunResult("Salom!", None, USAGE, "stop")])

    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={
            "model": "haiku",
            "messages": [
                {"role": "system", "content": "Be short."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Again"},
            ],
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["choices"][0]["message"]["content"] == "Salom!"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 13
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 3

    prompt, model, stream = runner.calls[0]
    assert model == "haiku"
    assert stream is False
    assert prompt.system == "Be short."
    assert "User: Hi\n\nAssistant: Hello" in prompt.prompt
    assert prompt.prompt.endswith("Again")


def test_structured_output_is_returned_as_json_content():
    client, runner = make_client([RunResult("", {"score": 7}, USAGE, "stop")])
    schema = {"type": "object", "properties": {"score": {"type": "integer"}}}

    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={
            "messages": [{"role": "user", "content": "Rate"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}},
        },
    )

    assert json.loads(response.json()["choices"][0]["message"]["content"]) == {"score": 7}
    assert runner.calls[0][0].output_schema == schema


def test_rejects_client_side_tools():
    client, runner = make_client([])

    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "x"}}],
        },
    )

    assert response.status_code == 400
    assert runner.calls == []


def test_runner_error_becomes_502():
    client, _ = make_client([RunResult("", None, None, "stop", error="boom")])

    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )

    assert response.status_code == 502
    assert response.json()["error"]["message"] == "boom"


def _sse_payloads(text: str) -> list:
    return [line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: ")]


def test_streams_text_deltas_finish_and_usage():
    client, _ = make_client([TextDelta("Sa"), TextDelta("lom"), RunResult("Salom", None, USAGE, "stop")])

    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )

    payloads = _sse_payloads(response.text)
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "Salom"
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] == 5


@pytest.mark.parametrize("structured", [{"ok": True}])
def test_streams_structured_output_as_single_chunk(structured):
    client, _ = make_client([RunResult("", structured, USAGE, "stop")])

    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
        },
    )

    chunks = [json.loads(p) for p in _sse_payloads(response.text)[:-1]]
    assert json.loads(chunks[1]["choices"][0]["delta"]["content"]) == structured


def test_stream_error_is_sent_in_band():
    client, _ = make_client([RunResult("", None, None, "stop", error="boom")])

    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"stream": True, "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert json.loads(_sse_payloads(response.text)[-1])["error"]["message"] == "boom"


def test_streams_reasoning_as_reasoning_content():
    client, _ = make_client([ReasoningDelta("hmm"), TextDelta("Ha"), RunResult("Ha", None, USAGE, "stop")])

    response = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"stream": True, "messages": [{"role": "user", "content": "Hi"}]},
    )

    deltas = [json.loads(p)["choices"][0]["delta"] for p in _sse_payloads(response.text)[:-1]]
    assert deltas[1] == {"reasoning_content": "hmm"}
    assert deltas[2] == {"content": "Ha"}


def test_thinking_env_mapping(monkeypatch):
    from app.config import Settings

    for value, expected in [("", None), ("disabled", {"type": "disabled"}), ("adaptive", {"type": "adaptive", "display": "summarized"}), ("4000", {"type": "enabled", "budget_tokens": 4000, "display": "summarized"})]:
        monkeypatch.setenv("CLAUDE_THINKING", value)
        assert Settings.from_env().thinking == expected

    monkeypatch.setenv("CLAUDE_THINKING", "lots")
    with pytest.raises(ValueError):
        Settings.from_env()
