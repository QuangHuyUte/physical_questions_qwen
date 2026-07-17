# EXACT 2026 Local Submission Runbook

This setup keeps the physics solver in `physics_api_server.py` and serves Qwen 2.5 3B LoRA adapters through an OpenAI-compatible vLLM server.

## 1. Start vLLM for Qwen 2.5 3B + LoRA Adapters

Run this on Linux/WSL/Kaggle/Cloud GPU where vLLM is available:

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --enable-lora \
  --lora-modules physics_explanation=/path/to/result/qwen25_3b_physics_explanation_adapter/adapter physics_semantic_parser=/path/to/result/qwen25_3b_physics_semantic_parser_adapter/adapter \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85
```

Check model visibility:

```bash
curl http://127.0.0.1:8001/v1/models
```

For the committee, expose this vLLM server too, because `/v1/models` must be reachable.

## 2. Start Prediction API

PowerShell from project root:

```powershell
$env:PHYSICS_API_HOST="0.0.0.0"
$env:PHYSICS_API_PORT="8000"
$env:PHYSICS_ENABLE_POLISH="true"
$env:PHYSICS_POLISH_BASE_URL="http://127.0.0.1:8001/v1"
$env:PHYSICS_POLISH_MODEL="physics_explanation"
$env:PHYSICS_ENABLE_SEMANTIC_PARSER="true"
$env:PHYSICS_SEMANTIC_MODEL="physics_semantic_parser"
$env:PHYSICS_SEMANTIC_MIN_CONFIDENCE="0.50"
python .\result\physics_api_server.py
```

If vLLM is not available, set:

```powershell
$env:PHYSICS_ENABLE_POLISH="false"
```

The API will still return deterministic explanations.

## 3. Test Competition Schema Locally

```powershell
python .\result\test_submission_predict.py http://127.0.0.1:8000/predict
```

Expected output:

- HTTP 200
- response is a JSON list
- one object containing `query_id`, `answer`, `unit`, `explanation`, `premises_used`, and `reasoning`
- Type 2 unit is ASCII, for example `A`, `V`, `ohm`, `uF`, `V/m`

## 4. Expose With ngrok

Open two tunnels if API and vLLM are on separate ports:

```powershell
ngrok http 8000
ngrok http 8001
```

Submit these URLs in `urls.txt`:

```text
prediction_url=https://<api-ngrok-host>/predict
vllm_models_url=https://<vllm-ngrok-host>/v1/models
```

## 5. Model Declaration

Declare:

- Base LLM: `Qwen/Qwen2.5-3B-Instruct`, 3B parameters.
- Adapter: LoRA adapter `physics_explanation`, trained on 1000 synthetic/verified physics explanation samples.
- Adapter: LoRA adapter `physics_semantic_parser`, trained to normalize physics questions into canonical JSON without answer/code generation.
- Active LLM total: 3B-class, below the 8B active model limit.

The deterministic physics solver and semantic canonical Python solver are non-LLM tools and do not count toward the LLM parameter limit.
