"""ccgw - an OpenAI-compatible chat-completions endpoint served by Claude Code.

Each request becomes one stateless Claude Code run (via the Claude Agent SDK):
built-in tools off, user settings not loaded, caller system prompt in place of
Claude Code's own, caller tools deferred back as `tool_calls`. See convert.py
for the translation rules.

Endpoints
---------
GET  /health
GET  /v1/models                 the single exposed model (CCGW_MODEL_NAME)
POST /v1/chat/completions       also served at /chat/completions; stream or not
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookMatcher,
    RateLimitEvent,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import convert

LOG = logging.getLogger("ccgw")
logging.basicConfig(level=os.environ.get("CCGW_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = os.environ.get("CCGW_MODEL_NAME", "claude-code")
CLAUDE_MODEL = os.environ.get("CCGW_CLAUDE_MODEL", "sonnet").strip() or None
EFFORT = os.environ.get("CCGW_EFFORT", "").strip() or None
API_KEY = os.environ.get("CCGW_API_KEY", "").strip()
TIMEOUT = int(os.environ.get("CCGW_TIMEOUT_SECONDS", "300"))
MAX_TURNS = int(os.environ.get("CCGW_MAX_TURNS", "4"))
WORKDIR = Path(os.environ.get("CCGW_WORKDIR", str(ROOT / "run" / "workdir"))).expanduser().resolve()
WORKDIR.mkdir(parents=True, exist_ok=True)

MAX_CONCURRENCY = int(os.environ.get("CCGW_MAX_CONCURRENCY", "8"))
# Anthropic-side throttling: retry inside ccgw while the wait stays short, then
# return a real 429 + Retry-After so the caller (or LiteLLM fallback) takes over.
RL_RETRIES = int(os.environ.get("CCGW_RATE_LIMIT_RETRIES", "2"))
RL_MAX_WAIT = float(os.environ.get("CCGW_RATE_LIMIT_MAX_WAIT_SECONDS", "30"))
RL_BACKOFF = float(os.environ.get("CCGW_RATE_LIMIT_BACKOFF_SECONDS", "5"))
# Request model "claude-code-<alias>" runs that Claude model; anything else runs CLAUDE_MODEL.
VARIANTS = [v.strip() for v in os.environ.get("CCGW_MODEL_VARIANTS", "haiku,sonnet,opus").split(",") if v.strip()]

_slots = asyncio.Semaphore(MAX_CONCURRENCY)
# Latest rate-limit state reported by Claude Code (RateLimitEvent), for logs and /health.
_rate: dict[str, Any] = {}

app = FastAPI(title="ccgw")


class UpstreamError(Exception):
    def __init__(self, status: int, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def _claude_model_for(requested: str | None) -> str | None:
    prefix = f"{MODEL_NAME}-"
    if requested and requested.startswith(prefix) and requested[len(prefix):] in VARIANTS:
        return requested[len(prefix):]
    return CLAUDE_MODEL


def _seconds_until_reset() -> float | None:
    """Seconds until the rejecting window resets, if Claude Code told us."""
    resets_at = _rate.get("resets_at")
    if _rate.get("status") != "rejected" or not resets_at:
        return None
    return max(1.0, resets_at - time.time())


def _note_rate_limit(ev: RateLimitEvent) -> None:
    info = ev.rate_limit_info
    state = {"status": info.status, "type": info.rate_limit_type, "resets_at": info.resets_at,
             "utilization": info.utilization, "seen_at": int(time.time())}
    changed = (state["status"], state["type"]) != (_rate.get("status"), _rate.get("type"))
    _rate.clear()
    _rate.update(state)
    if changed:
        level = logging.INFO if info.status == "allowed" else logging.WARNING
        resets = time.strftime("%H:%M:%S", time.localtime(info.resets_at)) if info.resets_at else "?"
        LOG.log(level, "rate limit status=%s window=%s utilization=%s resets_at=%s",
                info.status, info.rate_limit_type, info.utilization, resets)


async def _defer(_input: dict[str, Any], _tool_use_id: str | None, _ctx: Any) -> dict[str, Any]:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "defer"}}


def _options(body: dict[str, Any], stream: bool) -> tuple[ClaudeAgentOptions, str]:
    tools_on, choice_rule = convert.tool_choice_rule(body)
    tools = convert.caller_tools(body) if tools_on else []
    system, prompt = convert.build_prompt(body, [choice_rule])

    opts = ClaudeAgentOptions(
        tools=[],                       # no built-ins: Claude Code is a pure model here
        system_prompt=system,
        setting_sources=[],             # never load ~/.claude settings, CLAUDE.md, hooks, plugins
        strict_mcp_config=True,
        cwd=str(WORKDIR),
        model=_claude_model_for(body.get("model")),
        effort=EFFORT,
        max_turns=MAX_TURNS,
        include_partial_messages=stream,
        extra_args={"no-session-persistence": None},
    )
    if tools:
        opts.mcp_servers = {convert.TOOL_SERVER: create_sdk_mcp_server(convert.TOOL_SERVER, tools=tools)}
        opts.allowed_tools = [convert.TOOL_PREFIX + t.name for t in tools]
        opts.hooks = {"PreToolUse": [HookMatcher(matcher=f"{convert.TOOL_PREFIX}.*", hooks=[_defer])]}
    return opts, prompt


async def _run(body: dict[str, Any], opts: ClaudeAgentOptions, prompt: str) -> AsyncIterator[tuple[str, Any]]:
    """One Claude Code run: yield ("text", delta)* then one ("final", {...})."""
    texts: list[str] = []
    calls: list[dict[str, Any]] = []
    result: ResultMessage | None = None

    async with _slots:
        async with asyncio.timeout(TIMEOUT):
            async for msg in query(prompt=prompt, options=opts):
                if isinstance(msg, StreamEvent):
                    ev = msg.event
                    delta = ev.get("delta") or {}
                    if ev.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                        yield "text", delta.get("text", "")
                elif isinstance(msg, AssistantMessage):
                    if msg.error == "rate_limit":
                        raise UpstreamError(429, "Claude error: rate_limit", _seconds_until_reset())
                    if msg.error:
                        raise UpstreamError(502, f"Claude error: {msg.error}")
                    for block in msg.content:
                        if isinstance(block, TextBlock) and block.text:
                            texts.append(block.text)
                        elif isinstance(block, ToolUseBlock) and block.name.startswith(convert.TOOL_PREFIX):
                            calls.append({
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name[len(convert.TOOL_PREFIX):],
                                    "arguments": json.dumps(block.input),
                                },
                            })
                elif isinstance(msg, RateLimitEvent):
                    _note_rate_limit(msg)
                elif isinstance(msg, ResultMessage):
                    result = msg

    if result is None:
        raise UpstreamError(502, "Claude Code ended without a result")
    deferred = result.stop_reason == "tool_deferred" or result.deferred_tool_use is not None
    if result.is_error and not deferred:
        status = result.api_error_status or 502
        raise UpstreamError(status if status in (400, 401, 403, 429, 529) else 502,
                            "; ".join(result.errors or []) or result.result or result.subtype)

    text = "".join(texts)
    if (body.get("response_format") or {}).get("type") in ("json_object", "json_schema"):
        text = convert.strip_fences(text)
    yield "final", {
        "text": text,
        "tool_calls": calls,
        "finish_reason": "tool_calls" if calls else "stop",
        "usage": convert.usage_to_openai(result.usage),
        "cost_usd": result.total_cost_usd,
    }


async def _run_with_retry(body: dict[str, Any], stream: bool, rid: str) -> AsyncIterator[tuple[str, Any]]:
    """_run, retried on rate limits that arrive before any output was produced.

    Throttling surfaces within a couple of seconds, before the first token, so
    a retry is invisible to the caller apart from latency. The concurrency slot
    is released while we wait.
    """
    waited = 0.0
    for attempt in range(RL_RETRIES + 1):
        opts, prompt = _options(body, stream)
        produced = False
        try:
            async for item in _run(body, opts, prompt):
                produced = True
                yield item
            return
        except UpstreamError as exc:
            if exc.status != 429 or produced or attempt == RL_RETRIES:
                raise
            delay = exc.retry_after or RL_BACKOFF * (2 ** attempt)
            if waited + delay > RL_MAX_WAIT:
                raise  # e.g. a 5-hour window: hand the 429 back now, not after a pointless wait
            LOG.warning("%s rate limited, retry %d/%d in %.0fs", rid, attempt + 1, RL_RETRIES, delay)
            await asyncio.sleep(delay)
            waited += delay


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _check_auth(request: Request) -> None:
    if API_KEY and request.headers.get("authorization", "") != f"Bearer {API_KEY}":
        raise HTTPException(401, "invalid api key")


def _error(status: int, message: str, etype: str = "api_error", retry_after: float | None = None) -> JSONResponse:
    headers = {"Retry-After": str(max(1, int(retry_after or RL_BACKOFF)))} if status == 429 else None
    if status == 429:
        etype = "rate_limit_error"
    return JSONResponse({"error": {"message": message, "type": etype, "code": status}},
                        status_code=status, headers=headers)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model": MODEL_NAME, "claude_model": CLAUDE_MODEL,
            "max_concurrency": MAX_CONCURRENCY, "free_slots": _slots._value, "rate_limit": _rate or None}


@app.get("/v1/models")
@app.get("/models")
async def models(request: Request) -> dict[str, Any]:
    _check_auth(request)
    ids = [MODEL_NAME] + [f"{MODEL_NAME}-{v}" for v in VARIANTS]
    return {"object": "list", "data": [{"id": i, "object": "model", "created": 0, "owned_by": "claude-code"} for i in ids]}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    _check_auth(request)
    body = await request.json()
    stream = bool(body.get("stream"))
    rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = body.get("model") or MODEL_NAME
    started = time.monotonic()

    def log(final: dict[str, Any]) -> None:
        u = final["usage"]
        LOG.info("%s model=%s stream=%s msgs=%d tools=%d finish=%s calls=%d tokens=%d/%d cached=%d %.1fs",
                 rid, _claude_model_for(model), stream, len(body.get("messages") or []), len(body.get("tools") or []),
                 final["finish_reason"], len(final["tool_calls"]), u["prompt_tokens"], u["completion_tokens"],
                 u["prompt_tokens_details"]["cached_tokens"], time.monotonic() - started)

    try:
        _options(body, stream)  # validate: raises BadRequest before any bytes are sent
        runner = _run_with_retry(body, stream, rid)
        if not stream:
            final: dict[str, Any] = {}
            async for kind, value in runner:
                if kind == "final":
                    final = value
            log(final)
            message: dict[str, Any] = {"role": "assistant", "content": final["text"] or None}
            if final["tool_calls"]:
                message["tool_calls"] = final["tool_calls"]
            return {
                "id": rid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": final["finish_reason"]}],
                "usage": final["usage"],
            }
        # Streaming: wait for the first item before sending headers, so a failure
        # before any output (rate limit, auth) is a real HTTP status, not in-band.
        first = await anext(runner)
    except convert.BadRequest as exc:
        return _error(400, str(exc), "invalid_request_error")
    except UpstreamError as exc:
        LOG.warning("%s failed: %s (status %d)", rid, exc, exc.status)
        return _error(exc.status, str(exc), retry_after=exc.retry_after)
    except TimeoutError:
        return _error(504, f"Claude Code run exceeded {TIMEOUT}s")

    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
        return "data: " + json.dumps({
            "id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }) + "\n\n"

    async def items() -> AsyncIterator[tuple[str, Any]]:
        yield first
        async for item in runner:
            yield item

    async def sse() -> AsyncIterator[str]:
        yield chunk({"role": "assistant", "content": ""})
        try:
            async for kind, value in items():
                if kind == "text" and value:
                    yield chunk({"content": value})
                elif kind == "final":
                    log(value)
                    for i, call in enumerate(value["tool_calls"]):
                        yield chunk({"tool_calls": [{"index": i, **call}]})
                    yield chunk({}, value["finish_reason"])
                    if include_usage:
                        yield "data: " + json.dumps({
                            "id": rid, "object": "chat.completion.chunk", "created": created,
                            "model": model, "choices": [], "usage": value["usage"],
                        }) + "\n\n"
        except Exception as exc:  # headers are already sent; report in-band
            LOG.warning("%s stream failed: %s", rid, exc)
            status = getattr(exc, "status", 502)
            yield "data: " + json.dumps({"error": {"message": str(exc) or type(exc).__name__, "code": status}}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
