# Claude Code Service

An OpenAI-compatible HTTP API in front of [Claude Code](https://docs.anthropic.com/en/docs/claude-code), built with FastAPI and the official [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/).

It exists so that clients which speak the OpenAI Chat Completions format — in particular the [Laravel AI SDK](https://laravel.com/docs/ai-sdk) `openai-compatible` driver — can use Claude Code as a regular text provider, with no custom driver code.

```
Laravel AI SDK ──HTTP (OpenAI format)──► this service ──► claude-agent-sdk ──► Claude Code CLI ──► Claude
```

## Features

- `POST /v1/chat/completions` — plain and streaming (SSE) responses
- Streaming of text **and** thinking (`reasoning_content` deltas)
- Structured output: `response_format: {type: "json_schema"}` is mapped to Claude Code's `output_format`
- Token usage in the OpenAI shape, including cached tokens
- Bearer token auth, concurrency limit
- Every request is an isolated Claude Code session — nothing is shared between requests except the working directory

## Requirements

- Python 3.10+
- Claude Code CLI installed and authenticated (`claude` on `PATH`), **or** an `ANTHROPIC_API_KEY`

## Installation

```bash
git clone https://github.com/azizdevfull/claude-code-service.git
cd claude-code-service

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env   # then edit it
```

## Running

```bash
set -a; . ./.env; set +a
.venv/bin/uvicorn --factory app.main:create_app --host 127.0.0.1 --port 8787
```

Health check:

```bash
curl localhost:8787/health
# {"status":"ok"}
```

## Configuration

All settings are environment variables (see `.env.example`).

| Variable | Default | Description |
| --- | --- | --- |
| `SERVICE_API_KEY` | _(empty)_ | Bearer token clients must send. Empty disables auth — do not do this outside local development. |
| `ANTHROPIC_API_KEY` | _(unset)_ | Claude API key. When unset, the local `claude` login is used. |
| `CLAUDE_DEFAULT_MODEL` | `sonnet` | Model used when the request does not send one (`sonnet`, `opus`, `haiku` or a full model id). |
| `CLAUDE_CWD` | process cwd | Directory Claude Code works in. |
| `CLAUDE_ALLOWED_TOOLS` | `Read,Glob,Grep` | Comma-separated tools Claude Code may use without asking. |
| `CLAUDE_PERMISSION_MODE` | `dontAsk` | `dontAsk` denies every tool that is not in `CLAUDE_ALLOWED_TOOLS`. |
| `CLAUDE_MAX_TURNS` | `10` | Max agentic turns per request. Hitting it returns `finish_reason: "length"`. |
| `CLAUDE_SETTING_SOURCES` | _(empty)_ | Claude Code settings to load (`user`, `project`, `local`). Empty loads none: faster and isolated from your own CLAUDE.md, hooks and MCP servers. |
| `CLAUDE_SYSTEM_PROMPT_MODE` | `append` | `append`: Claude Code system prompt + the client's system message. `replace`: only the client's system message. |
| `CLAUDE_THINKING` | _(empty)_ | Empty: Claude Code default (thinking text hidden). `adaptive`: thinking is streamed to the client. `disabled`: no thinking. A number: fixed thinking budget in tokens. |
| `CLAUDE_EFFORT` | _(empty)_ | Reasoning effort on supported models: `low`, `medium`, `high`, `xhigh`, `max`. |
| `MAX_CONCURRENCY` | `4` | Max Claude Code processes running at once. |

## API

### `POST /v1/chat/completions`

```bash
curl localhost:8787/v1/chat/completions \
  -H 'Authorization: Bearer dev-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sonnet",
    "messages": [
      {"role": "system", "content": "Answer in one word."},
      {"role": "user", "content": "Capital of Uzbekistan?"}
    ]
  }'
```

Streaming: add `"stream": true` (and `"stream_options": {"include_usage": true}` for a final usage chunk). Responses are `chat.completion.chunk` SSE events terminated by `data: [DONE]`. Errors during a stream are sent in-band as `data: {"error": {...}}`.

### `GET /v1/models`

Lists the model aliases `sonnet`, `opus`, `haiku`.

### `GET /health`

Liveness check, no auth.

## How requests are mapped

- `system` / `developer` messages become the system prompt (appended to or replacing Claude Code's own, see `CLAUDE_SYSTEM_PROMPT_MODE`).
- Earlier `user` / `assistant` messages are rendered into a `<conversation_history>` block; the last user message is the prompt. Conversation state is kept by the client, not by the service.
- `response_format.json_schema.schema` is passed as Claude Code's `output_format`; the structured result is returned as JSON in `message.content`.
- `temperature`, `top_p` and `max_tokens` are accepted but ignored — Claude Code does not expose them.

## Limitations

- **Client-side tools are rejected** (`400`). Claude Code runs its own tools (`CLAUDE_ALLOWED_TOOLS`); requests containing `tools` or tool messages are refused instead of silently ignored.
- **Text only.** Image/file content parts are rejected.
- **Latency.** Each request starts a new Claude Code process (~1–2 s overhead). With thinking enabled the first answer token can take several seconds more; use `CLAUDE_THINKING=adaptive` to stream the thinking, or `sonnet` + `CLAUDE_THINKING=disabled` for the fastest first token.

## Using it from Laravel AI SDK

`config/ai.php`:

```php
'providers' => [
    'claude-code' => [
        'driver' => 'openai-compatible',
        'url' => env('CLAUDE_CODE_SERVICE_URL', 'http://127.0.0.1:8787/v1'),
        'key' => env('CLAUDE_CODE_SERVICE_KEY'),
        'models' => [
            'text' => ['default' => env('CLAUDE_CODE_MODEL', 'sonnet')],
        ],
    ],
],
```

Agent:

```php
use Laravel\Ai\Attributes\Provider;
use Laravel\Ai\Attributes\Timeout;
use Laravel\Ai\Contracts\Agent;
use Laravel\Ai\Promptable;

#[Provider('claude-code')]
#[Timeout(300)]
class ClaudeCode implements Agent
{
    use Promptable;

    public function instructions(): string
    {
        return 'You are a helpful assistant.';
    }
}

(new ClaudeCode)->prompt('Hello')->text;
(new ClaudeCode)->stream('Hello'); // SSE
```

Do not implement `HasTools` on agents that use this provider.

## Security notes

- Always set `SERVICE_API_KEY` and keep the service on a private network.
- Grant only the tools you need. `Write`, `Edit` and `Bash` let callers change files (and run commands) in `CLAUDE_CWD`, and those changes are visible to later requests.
- Serving other users through a Claude Pro/Max subscription login may violate Anthropic's terms — use an `ANTHROPIC_API_KEY` for anything beyond personal use.

## Development

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Tests use a fake runner and do not call Claude Code.
