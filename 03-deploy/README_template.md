# Serving Llama-3.2-1B-Instruct with vLLM on a free T4

**What:** an OpenAI-compatible LLM endpoint I deployed and load-tested myself.
**How:** vLLM <version> on Google Colab (NVIDIA T4, 15 GB), fp16 (`--dtype half`), max context <N> tokens.
**Why it matters:** measured latency/throughput trade-off as concurrency grows -> sizing and cost input.

## Run it
    pip install vllm
    vllm serve unsloth/Llama-3.2-1B-Instruct --dtype half --max-model-len 4096
    python load_test.py --concurrency 1 4 16 --requests 32 --max-tokens 128

## Results (fill in from load_test.py + metrics_summary.py)
| concurrency | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | output tok/s | req/s |
|---|---|---|---|---|---|
| 1  |  |  |  |  |  |
| 4  |  |  |  |  |  |
| 16 |  |  |  |  |  |

Server-side (from /metrics): avg TTFT __ ms, avg ITL __ ms, KV-cache peak __ %.

## What I learned
- Throughput scales with concurrency until ____; after that TTFT climbs because ____.
- Cost: T4 at $__/h -> ~$__ per 1M output tokens at concurrency __.
- Next: try SGLang / a 4-bit AWQ 8B model / an L4 GPU.
