# Adapter: `chlt_reasoner_final`

Purpose: answer conceptual CHLT-style physics questions with JSON-only output.

This adapter is separate from the numeric calculator. It is used for conceptual
questions, especially yes/no resonance questions and short physics theory
questions where a numeric final value is not requested.

## Dataset

- `dataset/chlt_reasoner_final_all.jsonl`: 407 rows
- `dataset/chlt_reasoner_final_train.jsonl`: 334 rows
- `dataset/chlt_reasoner_final_valid.jsonl`: 73 rows
- `dataset/chlt_reasoner_final_summary.json`: build summary
- `dataset/chlt_reasoner_final_synthetic_log.json`: synthetic verification log
- `dataset/chlt_reasoner_final_concept_bank_log.json`: curated concept-bank augmentation log
- `dataset/chlt_reasoner_final_rejected.json`: rejected rows

## Output Schema

The model must return exactly one JSON object:

```json
{
  "question_kind": "conceptual",
  "topic": "ac_resonance",
  "concept": "series_rlc_resonance_condition",
  "answer_type": "yes_no",
  "answer": "Yes",
  "evidence": ["..."],
  "confidence": 0.95
}
```

Forbidden:

- `python_code`

## Notebook

Use:

`notebook/train_qwen25_3b_chlt_reasoner_final_kaggle.ipynb`

Expected output adapter:

`/kaggle/working/qwen25_3b_chlt_reasoner_final_adapter/adapter`
