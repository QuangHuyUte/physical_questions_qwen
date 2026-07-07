# EXACT 2026 Hybrid V2 Final Package

## Main Outputs

- Kaggle API dataset ZIP: `kaggle_api_package_hybrid_v2_clean.zip`
- Portal submission ZIP: `EXACT2026_HybridV2_Submission_Package.zip`
- Source code ZIP: `submission_files/source_code/EXACT2026_HybridV2_Source_Code.zip`
- Solution PDF: `submission_files/solution_pdf/EXACT2026_HybridV2_Solution.pdf`
- URLs file: `submission_files/urls.txt`
- Notation mapping CSV: `submission_files/notation_mapping.csv`

## Runtime Logic

The API combines the old fast deterministic pipeline with the new calculator-first
V2 fallback:

1. Rule router sends Type 1 / CHLT / options / premises requests to `chlt_reasoner_final`.
2. Numeric Type 2 requests first use V1 deterministic solvers and direct guardrails.
3. Low-confidence or unanswered numeric requests call `numeric_parser_final`.
4. `physics_calculator_v2` validates extracted spans, units, formula IDs, and computes the locked answer.
5. `trace_explainer_final` is available for optional locked-trace explanation polish.

## Loaded Model

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- LoRA adapters:
  - `numeric_parser_final`
  - `trace_explainer_final`
  - `chlt_reasoner_final`

This stays below the 8B active model limit because there is one 3B base model
with LoRA adapters.

## Packaging Audit

The clean source ZIP and Kaggle dataset ZIP were checked for:

- no Windows backslash paths inside zip entries
- no `__pycache__`
- no `.pyc`
- no checkpoint folders
- no old `physics_explanation` or `physics_semantic_parser` adapters
- required final adapters present
- deploy notebook present
- `physics_calculator_v2` present

## Kaggle Run

Upload `kaggle_api_package_hybrid_v2_clean.zip` as a Kaggle dataset, then run:

`result/Notebook/deploy_physics_api_qwen3b_vllm_kaggle.ipynb`

After ngrok starts, update `submission_files/urls.txt` and the submission portal
with the printed `/predict` URL.
