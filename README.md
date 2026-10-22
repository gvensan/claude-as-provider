# claude-as-provider

A local LLM gateway whose "model" is Claude Code. Any OpenAI- or Anthropic-style
client (Solace Agent Mesh, LangChain, OpenAI SDK, LiteLLM SDK, ...) sends normal
chat requests; each one is answered by a Claude Code run via the Claude Agent SDK.

Temporary test rig only. It uses whatever credential the local `claude` CLI is
logged in with; not for anything live.

```
client ──► LiteLLM proxy :4444 ──► ccgw :9991 ──► Claude Agent SDK ──► claude CLI
          (keys, logging,          (OpenAI chat     (one stateless run per request)
           /v1/messages too)        completions)
```

## Run

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt   # once
scripts/dev.sh start          # also: stop | restart | status
.venv/bin/python tests/smoke.py
```

Config lives in `.env` (copied from `env.example` on first start). Logs: `logs/ccgw.log`, `logs/litellm.log`.

## Point Solace Agent Mesh at it

| Field | Value |
|---|---|
| Model Provider | Custom |
| Authentication Type | API Key |
| API Key | `sk-litellm-local` (`LITELLM_MASTER_KEY`) |
| API Base URL | `http://127.0.0.1:4444` |
| Model Name | `openai/claude-code` (`anthropic/claude-code` also works) |

If SAM runs in Docker, set `LITELLM_HOST=0.0.0.0` in `.env`, restart, and use
`http://host.docker.internal:4444` as the base URL.

## How a request is served

- **System prompt.** `system`/`developer` messages replace Claude Code's own system prompt.
- **History.** The caller re-sends the whole conversation each time; it is rendered as a transcript prompt.
- **No built-in tools.** Bash, Read, Edit and the rest are disabled and `~/.claude` settings are not loaded, so Claude Code acts as a pure model: it cannot touch files or run commands.
- **Caller tools.** `tools` become in-process MCP tools guarded by a PreToolUse hook that returns `defer`. Claude decides to call one, the run stops before executing it, and the call is returned as OpenAI `tool_calls` (parallel calls included). The caller runs the tool and sends the result back in the next request.
- **Streaming.** Text deltas stream live; tool calls are sent complete at the end.
- **Parameters.** `response_format` (json_object / json_schema) and `tool_choice` are honoured through instructions. Sampling parameters (temperature, top_p, max_tokens, stop) are dropped.
- **Model.** `claude-code` runs `CCGW_CLAUDE_MODEL` (default `sonnet`); `claude-code-haiku`, `claude-code-sonnet` and `claude-code-opus` pick that model. Any other name runs the default.

## Rate limits

Requests run on the logged-in Claude account, so Anthropic's limits for that
account apply (and are shared with your interactive Claude Code use). Raising
`CCGW_MAX_CONCURRENCY` does not lift them; it only allows bigger bursts.

- ccgw logs every rate-limit state change (`rate limit status=... window=...`) and shows the current state on `GET /health`.
- When Claude reports `rate_limit` before any output, ccgw retries (`CCGW_RATE_LIMIT_RETRIES`, backoff `CCGW_RATE_LIMIT_BACKOFF_SECONDS`, doubling) as long as the total wait stays under `CCGW_RATE_LIMIT_MAX_WAIT_SECONDS`. If the window resets later than that, it returns at once.
- After that it returns a real HTTP 429 with `Retry-After`, for streaming requests too, so client retries back off properly.
- LiteLLM then falls back from `claude-code` to `claude-code-haiku`. Remove the `fallbacks` line in `litellm/config.yaml` to fail instead of downgrading.
- `tests/test_rate_limit.py` checks this offline.

## Limits

- About 4 to 10 s per request (CLI process start plus the model call). `CCGW_MAX_CONCURRENCY` caps parallel runs.
- Images in messages are replaced by a placeholder.
- History is sent as text, not native tool_use/tool_result blocks. Fine for agent loops in practice, but prompt caching only covers the system prompt and tools (about 14k tokens for SAM); the history part is uncached on every call and grows with the conversation.

## Verified with Solace Agent Mesh

`tests/sam_probe.py` drives SAM's own `LiteLlm` model class (SAM 1.28.4, google-adk 1.18.0)
through a full tool round trip, for both `openai/` and `anthropic/` model prefixes,
streaming and non-streaming. Run it with a SAM environment's Python:

```bash
~/samprojects/sam-go/.venv/bin/python tests/sam_probe.py
```

Known gaps for SAM: images (user uploads, inline-vision tool results) and other file
parts are replaced by a placeholder; SAM's `thinking` budget is ignored and no
`reasoning_content` is returned; raise `CCGW_MAX_CONCURRENCY` if several agents call at once.
