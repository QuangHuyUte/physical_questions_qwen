# Adapter: `trace_explainer_final`

Purpose: explain locked calculator traces from `numeric_parser_final` and the
expanded 69-formula calculator.

## Dataset

- `dataset/trace_explainer_final_all.jsonl`: 1,359 rows
- `dataset/trace_explainer_final_train.jsonl`: 1,196 rows
- `dataset/trace_explainer_final_valid.jsonl`: 163 rows
- `dataset/trace_explainer_final_summary.json`: build summary
- `dataset/trace_explainer_final_generation_log.csv`: generated locked-trace
  explanation log

## Policy

The model receives a locked trace containing the final answer, unit, formula,
calculation, and extracted givens. It must return JSON containing only:

```json
{"explanation": "..."}
```

It must not change the answer, unit, formula, or calculation. The language
model is only the explanation layer.

## Notebook

Use:

`notebook/train_qwen25_3b_trace_explainer_final_kaggle.ipynb`

Expected output adapter:

`/kaggle/working/qwen25_3b_trace_explainer_final_adapter/adapter`
