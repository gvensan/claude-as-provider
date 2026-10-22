"""Drive Solace Agent Mesh's own LiteLlm model class against the local gateway.

Run with the Python of any SAM install (needs solace-agent-mesh in that env):

    <sam-project>/.venv/bin/python tests/sam_probe.py
"""
import asyncio, time, logging
logging.disable(logging.WARNING)
from google.genai import types
from google.adk.models.llm_request import LlmRequest
from solace_agent_mesh.agent.adk.models.lite_llm import LiteLlm

BASE, KEY = "http://127.0.0.1:4444", "sk-litellm-local"
TOOL = types.Tool(function_declarations=[types.FunctionDeclaration(
    name="get_weather", description="Current weather for a city.",
    parameters=types.Schema(type=types.Type.OBJECT, properties={"city": types.Schema(type=types.Type.STRING)}, required=["city"]))])
CFG = types.GenerateContentConfig(system_instruction="You are a SAM test agent. Use tools for live data.", tools=[TOOL])

async def call(llm, contents, stream):
    req = LlmRequest(model=llm.model, contents=contents, config=CFG)
    text, calls, usage, partials = "", [], None, 0
    async for r in llm.generate_content_async(req, stream=stream):
        if r.partial: partials += 1
        if r.usage_metadata: usage = r.usage_metadata.total_token_count
        if r.partial: continue
        for p in (r.content.parts if r.content else []):
            if p.text: text += p.text
            if p.function_call: calls.append(p.function_call)
    return text, calls, usage, partials

async def run(prefix, stream):
    t = time.monotonic()
    llm = LiteLlm(model=f"{prefix}/claude-code", api_base=BASE, api_key=KEY)
    user = types.Content(role="user", parts=[types.Part(text="What's the weather in Toronto?")])
    _, calls, u1, p1 = await call(llm, [user], stream)
    assert calls, "no function call returned"
    fc = calls[0]
    model_turn = types.Content(role="model", parts=[types.Part(function_call=fc)])
    tool_turn = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
        id=fc.id, name=fc.name, response={"result": "5C, snow flurries"}))])
    text, calls2, u2, p2 = await call(llm, [user, model_turn, tool_turn], stream)
    print(f"PASS {prefix:<9} stream={stream!s:<5} {time.monotonic()-t:5.1f}s  call={fc.name}{dict(fc.args)} id_len={len(fc.id)} "
          f"usage={u1}/{u2} partials={p1}/{p2} -> {text.strip()[:90]!r}")

async def main():
    for prefix in ("openai", "anthropic"):
        for stream in (True, False):
            try:
                await run(prefix, stream)
            except Exception as e:
                print(f"FAIL {prefix:<9} stream={stream}: {type(e).__name__}: {str(e)[:300]}")

asyncio.run(main())
