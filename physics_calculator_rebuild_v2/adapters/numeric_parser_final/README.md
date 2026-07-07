# Adapter: `numeric_parser_final`

Purpose: train the final strict-schema numeric parser for the calculator-first
Physics Type 2 pipeline.

## Dataset

- `dataset/numeric_parser_final_all.jsonl`: 1,663 rows
- `dataset/numeric_parser_final_train.jsonl`: 1,443 rows
- `dataset/numeric_parser_final_valid.jsonl`: 220 rows
- `dataset/numeric_parser_final_summary.json`: build summary and formula counts
- `dataset/numeric_parser_final_expansion_log.json`: generated coverage rows

## Policy

The assistant output is extraction-only. It must not contain:

- `answer`
- `unit_answer`
- `cot`
- `python_code`
- `golden_code`
- `final_result`

Every final row was kept only after the Python calculator could recompute the
locked answer from the extracted quantities.

## Coverage Update

This final dataset keeps the strict rows from the previous verified numeric
parser data and adds calculator-verified rows for the expanded 69-formula
calculator, including:

- capacitor rearrangements `C=Q/U`, `U=Q/C`, `W=QU/2`, `W=Q^2/(2C)`
- inductor current from energy and inductance
- wire resistance from resistivity, length, and area
- dielectric and plate-field relations
- vector resultant formulas
- RLC current/power/reactance formulas
- measurement uncertainty helpers
- underrepresented older formulas such as Faraday emf, solenoid field, magnetic
  flux, capacitance series, and impedance

## Notebook

Use:

`notebook/train_qwen25_3b_numeric_parser_final_kaggle.ipynb`

Expected output adapter:

`/kaggle/working/qwen25_3b_numeric_parser_final_adapter/adapter`
