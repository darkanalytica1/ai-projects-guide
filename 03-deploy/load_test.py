"""Async load test for any OpenAI-compatible server (vLLM, SGLang, ...).
Measures, per request, from the client side:
  TTFT  = time to first streamed token
  ITL   = gaps between streamed chunks (inter-token latency)
  and overall output tokens/second (throughput).

  python load_test.py --url http://localhost:8000 --model unsloth/Llama-3.2-1B-Instruct \
      --concurrency 1 4 16 --requests 32 --max-tokens 128
"""
import argparse
import asyncio
import json
import statistics
import time

import httpx

PROMPT = "Explain retrieval-augmented generation to a CFO in about 150 words."


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else float("nan")


async def one_request(client, url, model, max_tokens):
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft, last, gaps, chunks, usage_tokens = None, None, [], 0, None
    async with client.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            msg = json.loads(data)
            if msg.get("usage"):
                usage_tokens = msg["usage"].get("completion_tokens")
            if not msg.get("choices") or not msg["choices"][0]["delta"].get("content"):
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            else:
                gaps.append(now - last)
            last = now
            chunks += 1
    return {"ttft": ttft, "gaps": gaps, "e2e": time.perf_counter() - t0,
            "tokens": usage_tokens or chunks}


async def run_level(url, model, concurrency, n, max_tokens):
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=300) as client:
        async def guarded():
            async with sem:
                return await one_request(client, url, model, max_tokens)
        t0 = time.perf_counter()
        res = await asyncio.gather(*[guarded() for _ in range(n)], return_exceptions=True)
        wall = time.perf_counter() - t0
    ok = [r for r in res if isinstance(r, dict) and r["ttft"] is not None]
    ttfts = [r["ttft"] * 1000 for r in ok]
    itls = [g * 1000 for r in ok for g in r["gaps"]]
    toks = sum(r["tokens"] for r in ok)
    return {"concurrency": concurrency, "ok": len(ok), "failed": n - len(ok),
            "ttft_p50_ms": statistics.median(ttfts) if ttfts else float("nan"),
            "ttft_p95_ms": pct(ttfts, 95),
            "itl_p50_ms": statistics.median(itls) if itls else float("nan"),
            "itl_p95_ms": pct(itls, 95),
            "throughput_tok_s": toks / wall, "req_s": len(ok) / wall}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    a = ap.parse_args()
    rows = [await run_level(a.url, a.model, c, a.requests, a.max_tokens) for c in a.concurrency]
    print("| concurrency | ok/failed | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p95 (ms) | output tok/s | req/s |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['concurrency']} | {r['ok']}/{r['failed']} | {r['ttft_p50_ms']:.0f} | {r['ttft_p95_ms']:.0f} "
              f"| {r['itl_p50_ms']:.1f} | {r['itl_p95_ms']:.1f} | {r['throughput_tok_s']:.0f} | {r['req_s']:.2f} |")


if __name__ == "__main__":
    asyncio.run(main())
