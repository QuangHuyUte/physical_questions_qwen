# Adapter Plan

## 1. `numeric_parser_final`

Highest priority. This replaces the old `semantic_parser` role.

Input:

- Raw question.
- Optional topic hint.

Output:

- `question_kind`
- `topic`
- `target`
- `givens`
- `constraints`
- `formula_candidates`
- `confidence`

Important:

- No final answer.
- No Python code.
- No trusted SI value.
- Every given must include `raw_span`.

The model is allowed to propose formula IDs, but the calculator verifies them and
can ignore them.

Current training package:

- Folder: `adapters/numeric_parser_final`
- Notebook: `adapters/numeric_parser_final/notebook/train_qwen25_3b_numeric_parser_final_kaggle.ipynb`
- Upload ZIP: `adapters/qwen25_3b_numeric_parser_final_training_package.zip`

Older `numeric_parser` and `numeric_parser_v2` folders are superseded and should
not be used for new training.

## 2. `trace_explainer_final`

Input:

- Question.
- Locked calculator trace.
- Locked answer and unit.

Output:

- JSON object containing only `explanation`.

The explanation adapter is not allowed to change the answer, unit, formula, or
calculation.

Current training package:

- Folder: `adapters/trace_explainer_final`
- Notebook: `adapters/trace_explainer_final/notebook/train_qwen25_3b_trace_explainer_final_kaggle.ipynb`
- Upload ZIP: `adapters/qwen25_3b_trace_explainer_final_training_package.zip`

Older `trace_explainer` is superseded by the final dataset built from
`numeric_parser_final` and the expanded calculator.

## 3. `chlt_reasoner_final`

For conceptual/theory questions that should not go through the numeric calculator.

Output:

- `answer_type`
- `answer`
- `concept`
- `evidence`
- `confidence`

If options are provided, answer must be one of the options. If the adapter is not
confident, it should return `Uncertain`.

Current training package:

- Folder: `adapters/chlt_reasoner_final`
- Notebook: `adapters/chlt_reasoner_final/notebook/train_qwen25_3b_chlt_reasoner_final_kaggle.ipynb`
- Upload ZIP: `adapters/qwen25_3b_chlt_reasoner_final_training_package.zip`

Older `chlt_reasoner` is seed-only and superseded by the final CHLT dataset.

## 4. `question_router`

Optional. It can be implemented by rules first:

- prefix `CHLT` -> conceptual
- options present -> conceptual or MCQ
- numeric target and units present -> calculation

Only train this adapter if rule routing is not enough.
