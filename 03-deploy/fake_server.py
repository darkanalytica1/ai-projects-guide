"""Tiny fake OpenAI-compatible streaming server + /metrics, to test the scripts without a GPU.
  python fake_server.py   (then run load_test.py --url http://localhost:8001)"""
import asyncio, json
from aiohttp import web

async def chat(req):
    body = await req.json()
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(req)
    await asyncio.sleep(0.05)
    n = body.get("max_tokens", 16)
    for i in range(n):
        chunk = {"choices": [{"delta": {"content": f"tok{i} "}}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode()); await asyncio.sleep(0.01)
    await resp.write(f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': n}})}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp

async def metrics(req):
    return web.Response(text='vllm:time_to_first_token_seconds_sum{model_name="m"} 1.5\n'
                             'vllm:time_to_first_token_seconds_count{model_name="m"} 30\n'
                             'vllm:generation_tokens_total{model_name="m"} 3840\n')

app = web.Application(); app.router.add_post("/v1/chat/completions", chat); app.router.add_get("/metrics", metrics)
web.run_app(app, port=8001)
