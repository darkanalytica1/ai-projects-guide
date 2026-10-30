"""Read vLLM's Prometheus /metrics and print server-side averages.

  python metrics_summary.py --url http://localhost:8000
Histogram metrics expose <name>_sum and <name>_count; average = sum / count.
"""
import argparse
import re

import httpx

HISTOGRAMS = {
    "vllm:time_to_first_token_seconds": "TTFT",
    "vllm:inter_token_latency_seconds": "inter-token latency",
    "vllm:e2e_request_latency_seconds": "end-to-end latency",
    "vllm:request_queue_time_seconds": "queue time",
}
COUNTERS = {"vllm:prompt_tokens_total": "prompt tokens",
            "vllm:generation_tokens_total": "generated tokens",
            "vllm:request_success_total": "finished requests"}
GAUGES = {"vllm:num_requests_running": "running now", "vllm:num_requests_waiting": "waiting now",
          "vllm:kv_cache_usage_perc": "KV cache used (0-1)"}


def total(text: str, name: str) -> float:
    """Sum a metric across all label sets, e.g. vllm:x_sum{model_name="..."} 1.23"""
    pat = re.compile(rf"^{re.escape(name)}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)$", re.M)
    return sum(float(v) for v in pat.findall(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    text = httpx.get(f"{ap.parse_args().url}/metrics", timeout=10).text
    for name, label in HISTOGRAMS.items():
        s, c = total(text, name + "_sum"), total(text, name + "_count")
        print(f"{label:>22}: " + (f"avg {1000 * s / c:.1f} ms over {int(c)} samples" if c else "no data yet"))
    for name, label in {**COUNTERS, **GAUGES}.items():
        print(f"{label:>22}: {total(text, name):g}")


if __name__ == "__main__":
    main()
