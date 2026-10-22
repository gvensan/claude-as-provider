"""Translation between the OpenAI chat-completions shape and a Claude Code run.

A chat-completions caller is stateless: every request carries the whole
conversation, and tools are executed by the caller. Claude Code is an agent
with its own loop. We reconcile the two like this:

  * system/developer messages become the Claude Code system prompt (replacing
    Claude Code's own, so none of its coding-agent persona leaks in)
  * the rest of the conversation is rendered as a transcript prompt
  * caller tools are registered as in-process MCP tools whose PreToolUse hook
    answers "defer", so the run stops at the tool call instead of executing it,
    and we hand the call back to the caller as OpenAI `tool_calls`
"""
from __future__ import annotations

import json
import re
from typing import Any

from claude_agent_sdk import SdkMcpTool

TOOL_SERVER = "fn"
TOOL_PREFIX = f"mcp__{TOOL_SERVER}__"

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


class BadRequest(ValueError):
    """The request cannot be served; mapped to HTTP 400."""


def text_of(content: Any) -> str:
    """Flatten OpenAI message content (string or parts list) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif part.get("type") in ("text", "input_text", "output_text"):
            parts.append(str(part.get("text", "")))
        elif part.get("type") == "image_url":
            parts.append("[image omitted: this gateway does not forward images]")
        else:
            parts.append(f"[{part.get('type', 'unknown')} content omitted]")
    return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #
async def _never_called(_args: dict[str, Any]) -> dict[str, Any]:
    # The PreToolUse hook defers every call before it reaches here.
    return {"content": [{"type": "text", "text": "error: caller-side tool executed in gateway"}], "is_error": True}


def caller_tools(body: dict[str, Any]) -> list[SdkMcpTool]:
    """OpenAI `tools` (or legacy `functions`) as SDK MCP tool definitions."""
    specs: list[dict[str, Any]] = []
    for t in body.get("tools") or []:
        if t.get("type", "function") == "function" and t.get("function"):
            specs.append(t["function"])
    specs.extend(body.get("functions") or [])

    tools: list[SdkMcpTool] = []
    for fn in specs:
        name = str(fn.get("name") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise BadRequest(f"unsupported tool name: {name!r}")
        tools.append(SdkMcpTool(
            name=name,
            description=str(fn.get("description") or name),
            input_schema=fn.get("parameters") or _EMPTY_SCHEMA,
            handler=_never_called,
        ))
    return tools


def tool_choice_rule(body: dict[str, Any]) -> tuple[bool, str]:
    """(tools enabled, extra system instruction) for the request's tool_choice."""
    choice = body.get("tool_choice", body.get("function_call"))
    if choice in (None, "auto"):
        return True, ""
    if choice == "none":
        return False, "Do not call any tools in this response; answer in text."
    if choice in ("required", "any"):
        return True, "You must call at least one tool in this response."
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name") or choice.get("name")
        if name:
            return True, f"You must call the tool `{name}` in this response."
    return True, ""


# --------------------------------------------------------------------------- #
# messages
# --------------------------------------------------------------------------- #
_TRANSCRIPT_PREAMBLE = (
    "The conversation so far is shown below as a transcript. Entries marked "
    "<tool_call> are tool calls you made earlier, and <tool_result> entries are "
    "what the caller returned for them; the caller executed those tools, not you."
)
_TRANSCRIPT_CLOSE = (
    "Continue as the assistant from the end of the transcript. To use a tool, call "
    "it through the tool interface; never write <tool_call> or <tool_result> markup "
    "in your reply."
)


def _format_rule(body: dict[str, Any]) -> str:
    fmt = body.get("response_format") or {}
    if fmt.get("type") == "json_object":
        return "Reply with a single valid JSON object only: no prose, no code fences."
    if fmt.get("type") == "json_schema":
        schema = (fmt.get("json_schema") or {}).get("schema") or {}
        return ("Reply with a single valid JSON value only (no prose, no code fences) "
                f"that conforms to this JSON Schema:\n{json.dumps(schema)}")
    return ""


def build_prompt(body: dict[str, Any], extra_rules: list[str]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a chat-completions body."""
    messages = body.get("messages") or []
    if not messages:
        raise BadRequest("messages must be a non-empty list")

    system_parts: list[str] = []
    convo: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") in ("system", "developer"):
            system_parts.append(text_of(m.get("content")))
        else:
            convo.append(m)
    if not convo:
        raise BadRequest("messages must include at least one user message")

    rules = [r for r in [_format_rule(body), *extra_rules] if r]
    system = "\n\n".join(p for p in [*system_parts, *rules] if p.strip()) or "You are a helpful assistant."

    # Single user turn: send it as-is, no transcript wrapping.
    if len(convo) == 1 and convo[0].get("role") == "user":
        return system, text_of(convo[0].get("content")) or "(empty message)"

    names_by_id: dict[str, str] = {}
    lines: list[str] = [_TRANSCRIPT_PREAMBLE, "", "<transcript>"]
    for m in convo:
        role = m.get("role")
        if role == "user":
            lines.append(f"<turn role=\"user\">\n{text_of(m.get('content'))}\n</turn>")
        elif role == "assistant":
            body_lines: list[str] = []
            text = text_of(m.get("content"))
            if text:
                body_lines.append(text)
            calls = list(m.get("tool_calls") or [])
            if m.get("function_call"):
                calls.append({"id": "legacy", "function": m["function_call"]})
            for c in calls:
                fn = c.get("function") or {}
                names_by_id[str(c.get("id"))] = str(fn.get("name"))
                body_lines.append(
                    f"<tool_call id=\"{c.get('id')}\" name=\"{fn.get('name')}\">"
                    f"{fn.get('arguments') or '{}'}</tool_call>"
                )
            lines.append("<turn role=\"assistant\">\n" + "\n".join(body_lines) + "\n</turn>")
        elif role in ("tool", "function"):
            call_id = str(m.get("tool_call_id") or "legacy")
            name = m.get("name") or names_by_id.get(call_id, "")
            lines.append(
                f"<tool_result id=\"{call_id}\" name=\"{name}\">\n{text_of(m.get('content'))}\n</tool_result>"
            )
    lines += ["</transcript>", "", _TRANSCRIPT_CLOSE]
    return system, "\n".join(lines)


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
def strip_fences(text: str) -> str:
    """Remove a single wrapping ``` fence, for JSON response formats."""
    m = re.fullmatch(r"\s*```[a-zA-Z]*\n(.*?)\n?```\s*", text, flags=re.S)
    return m.group(1) if m else text


def usage_to_openai(usage: dict[str, Any] | None) -> dict[str, Any]:
    u = usage or {}
    cached = int(u.get("cache_read_input_tokens") or 0)
    prompt = int(u.get("input_tokens") or 0) + int(u.get("cache_creation_input_tokens") or 0) + cached
    completion = int(u.get("output_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
