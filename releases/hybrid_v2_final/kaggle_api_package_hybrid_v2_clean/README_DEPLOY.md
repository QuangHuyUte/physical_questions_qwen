# EXACT 2026 Hybrid V2 Physics API Deploy Package

This folder is the clean runtime package for testing/submitting the EXACT 2026
physics API. It serves both Type 2 numeric physics and CHLT/Type 1 conceptual
physics through the same `/predict` endpoint.

## Runtime Flow

1. Rule router separates conceptual/CHLT requests from numeric requests.
2. V1 deterministic physics solver handles high-confidence numeric patterns fast.
3. Direct guardrail formulas cover high-risk numeric patterns.
4. `numeric_parser_final` LoRA extracts quantities/formula candidates only.
5. `physics_calculator_v2` validates spans/units and computes the locked answer.
6. `trace_explainer_final` can polish locked numeric traces when enabled.
7. `chlt_reasoner_final` answers conceptual/CHLT questions with evidence.

The planner-code adapter is intentionally not included and is not used. Numeric
LLM output is never trusted as the final answer.

## Important Files

- `result/physics_api_server.py`
- `result/physics_pipeline.py`
- `result/physics_engine_core.py`
- `result/physics_calculator_v2`
- `result/Notebook/deploy_physics_api_qwen3b_vllm_kaggle.ipynb`
- `result/qwen25_3b_numeric_parser_final_adapter/adapter`
- `result/qwen25_3b_trace_explainer_final_adapter/adapter`
- `result/qwen25_3b_chlt_reasoner_final_adapter/adapter`
- `data/verified_golden_expanded.csv`

The runtime dataset intentionally excludes the old `golden_code` column. It is used for routing/retrieval context, not for answer-code lookup.

Upload this whole folder as a Kaggle dataset, then run the deploy notebook.
