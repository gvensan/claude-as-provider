"""Offline test of ccgw's rate-limit handling (no Claude calls).

    .venv/bin/python tests/test_rate_limit.py
"""
import os
import sys

os.environ.update(CCGW_RATE_LIMIT_BACKOFF_SECONDS="0.1", CCGW_RATE_LIMIT_MAX_WAIT_SECONDS="1", CCGW_API_KEY="")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from ccgw import app as gw  # noqa: E402

FINAL = {"text": "ok", "tool_calls": [], "finish_reason": "stop",
         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                   "prompt_tokens_details": {"cached_tokens": 0}}, "cost_usd": 0}


def fake_run(fail_times: int, retry_after=None):
    calls = {"n": 0}

    async def _run(body, opts, prompt):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise gw.UpstreamError(429, "Claude error: rate_limit", retry_after)
        yield "text", "ok"
        yield "final", FINAL
    return _run, calls


client = TestClient(gw.app)
BODY = {"model": "claude-code", "messages": [{"role": "user", "content": "hi"}]}


def check(name, fail_times, stream, expect_status, expect_calls, retry_after=None):
    gw._run, calls = fake_run(fail_times, retry_after)
    r = client.post("/v1/chat/completions", json={**BODY, "stream": stream})
    ok = r.status_code == expect_status and calls["n"] == expect_calls
    extra = f" Retry-After={r.headers.get('retry-after')}" if r.status_code == 429 else ""
    print(f"{'PASS' if ok else 'FAIL'} {name:<44} status={r.status_code} attempts={calls['n']}{extra}")
    return ok


results = [
    check("throttled once, retried, succeeds", 1, False, 200, 2),
    check("throttled once, streaming, succeeds", 1, True, 200, 2),
    check("throttled 3x, retries exhausted -> 429", 3, False, 429, 3),
    check("streaming, exhausted -> real HTTP 429", 3, True, 429, 3),
    check("reset far away (5h window) -> 429 at once", 1, False, 429, 1, retry_after=18000),
]
sys.exit(0 if all(results) else 1)
