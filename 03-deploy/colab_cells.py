# ==== Google Colab cells for Project 03. Runtime > Change runtime type > T4 GPU > Save ====

# --- Cell 1: check the GPU
!nvidia-smi

# --- Cell 2: install vLLM (takes a few minutes; if Colab asks to restart the session, do it,
#             then continue from Cell 3)
!pip install -q vllm httpx

# --- Cell 3: start the server in the background, logs go to vllm.log
!nohup vllm serve unsloth/Llama-3.2-1B-Instruct --dtype half --max-model-len 4096 \
    --gpu-memory-utilization 0.85 --port 8000 > vllm.log 2>&1 &

# --- Cell 4: wait until it is healthy (first start downloads ~2.5 GB of weights)
import time, httpx
for _ in range(120):
    try:
        if httpx.get("http://localhost:8000/health", timeout=2).status_code == 200:
            print("server up"); break
    except Exception:
        pass
    time.sleep(5)
else:
    print(open("vllm.log").read()[-3000:])   # show the error if it never came up

# --- Cell 5: first request
!curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"unsloth/Llama-3.2-1B-Instruct","messages":[{"role":"user","content":"Say hi in 5 words"}],"max_tokens":20}'

# --- Cell 6: raw metrics (TTFT, inter-token latency, tokens...)
!curl -s localhost:8000/metrics | grep -E "^vllm:(time_to_first_token|inter_token_latency|generation_tokens|num_requests)" | head -30

# --- Cell 7: upload load_test.py + metrics_summary.py (folder icon on the left), then:
!python load_test.py --concurrency 1 4 16 --requests 32 --max-tokens 128
!python metrics_summary.py

# --- Cell 8 (optional): vLLM's own benchmark
!vllm bench serve --backend vllm --model unsloth/Llama-3.2-1B-Instruct --endpoint /v1/completions \
    --dataset-name random --num-prompts 100 --random-input-len 512 --random-output-len 128 --seed 1
