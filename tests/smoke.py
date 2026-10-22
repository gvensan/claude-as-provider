"""End-to-end smoke test against the running stack (scripts/dev.sh start).

    .venv/bin/python tests/smoke.py
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import litellm
from openai import AsyncOpenAI

PROXY = f"http://127.0.0.1:{os.environ.get('LITELLM_PORT', '4444')}"
KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-litellm-local")
CCGW = f"http://127.0.0.1:{os.environ.get('CCGW_PORT', '9991')}/v1"
MODEL = "claude-code"

WEATHER = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
FAKE = {"Paris": "14C, light rain", "Ottawa": "3C, clear"}


async def plain_direct() -> str:
    c = AsyncOpenAI(base_url=CCGW, api_key=os.environ.get("CCGW_API_KEY", "x"))
    r = await c.chat.completions.create(model=MODEL, messages=[
        {"role": "system", "content": "Answer in exactly one word."},
        {"role": "user", "content": "What is the capital of France?"}])
    return f"{r.choices[0].message.content!r} finish={r.choices[0].finish_reason} usage={r.usage.total_tokens}"


async def stream_proxy() -> str:
    c = AsyncOpenAI(base_url=PROXY, api_key=KEY)
    chunks, text = 0, ""
    s = await c.chat.completions.create(model=MODEL, stream=True, messages=[
        {"role": "user", "content": "Count from 1 to 5, comma separated, nothing else."}])
    async for ch in s:
        if ch.choices and ch.choices[0].delta.content:
            chunks += 1
            text += ch.choices[0].delta.content
    return f"{text!r} in {chunks} content chunks"


async def tool_roundtrip() -> str:
    c = AsyncOpenAI(base_url=PROXY, api_key=KEY)
    msgs = [{"role": "user", "content": "What's the weather in Paris and Ottawa? Use the tool."}]
    r = await c.chat.completions.create(model=MODEL, messages=msgs, tools=WEATHER)
    m = r.choices[0].message
    assert r.choices[0].finish_reason == "tool_calls", r
    calls = [(t.function.name, json.loads(t.function.arguments)) for t in m.tool_calls]
    msgs.append(m.model_dump(exclude_none=True))
    for t in m.tool_calls:
        city = json.loads(t.function.arguments)["city"]
        msgs.append({"role": "tool", "tool_call_id": t.id, "content": FAKE.get(city, "unknown")})
    r2 = await c.chat.completions.create(model=MODEL, messages=msgs, tools=WEATHER)
    return f"calls={calls} -> {r2.choices[0].finish_reason}: {r2.choices[0].message.content!r}"


async def tool_stream() -> str:
    c = AsyncOpenAI(base_url=PROXY, api_key=KEY)
    s = await c.chat.completions.create(model=MODEL, stream=True, tools=WEATHER, tool_choice="required",
                                        messages=[{"role": "user", "content": "Weather in Ottawa?"}])
    calls, finish = [], None
    async for ch in s:
        if not ch.choices:
            continue
        for tc in ch.choices[0].delta.tool_calls or []:
            calls.append((tc.function.name, tc.function.arguments))
        finish = ch.choices[0].finish_reason or finish
    return f"calls={calls} finish={finish}"


async def json_mode() -> str:
    c = AsyncOpenAI(base_url=PROXY, api_key=KEY)
    r = await c.chat.completions.create(model=MODEL, response_format={"type": "json_object"}, messages=[
        {"role": "user", "content": "Give me {\"city\": ..., \"country\": ...} for the Eiffel Tower."}])
    return str(json.loads(r.choices[0].message.content))


async def litellm_sdk(prefix: str) -> str:
    # How SAM's "Custom" provider calls a model: LiteLLM SDK + api_base + prefixed model name.
    r = await litellm.acompletion(model=f"{prefix}/{MODEL}", api_base=PROXY, api_key=KEY, tools=WEATHER,
                                  messages=[{"role": "user", "content": "Weather in Paris?"}])
    m = r.choices[0].message
    calls = [(t.function.name, t.function.arguments) for t in (m.tool_calls or [])]
    return f"finish={r.choices[0].finish_reason} calls={calls} text={m.content!r}"


async def timed(name: str, coro) -> None:
    t = time.monotonic()
    try:
        out = await coro
        print(f"PASS {name:<22} {time.monotonic() - t:5.1f}s  {out}")
    except Exception as exc:
        print(f"FAIL {name:<22} {time.monotonic() - t:5.1f}s  {type(exc).__name__}: {exc}")


async def main() -> None:
    await asyncio.gather(
        timed("plain (ccgw direct)", plain_direct()),
        timed("stream (proxy)", stream_proxy()),
        timed("tool round-trip", tool_roundtrip()),
        timed("tool stream", tool_stream()),
        timed("json mode", json_mode()),
        timed("litellm openai/", litellm_sdk("openai")),
        timed("litellm anthropic/", litellm_sdk("anthropic")),
    )

asyncio.run(main())
