# Physical Questions Qwen

This repository contains the cleaned EXACT 2026 physics question answering work.

## Current Layout

- `current_submission_package/`: latest submission/export package.
- `hybrid_v2_final_package/`: calculator-first Hybrid V2 package and submission docs.
- `calculator_first_source/`: clean source workspace for the current calculator-first system.
- `datasets_raw_updates/`: small raw/update datasets kept for traceability.
- `docs_papers/`: reference papers and extracted notes.
- `physics_type1_api/`: Type 1 API notebook/work.
- `test_samples/`: small holdout/test samples.

## Current Runtime Direction

The current strongest system is the calculator-first Hybrid V2 design:

- deterministic fast path,
- `numeric_parser_final` for structured extraction when needed,
- verified calculator/formula bank for final numeric answers,
- `trace_explainer_final` for locked-trace explanations,
- `chlt_reasoner_final` for conceptual/CHLT questions.

Large model weights, checkpoints, zip bundles, cache files, and older generated
artifacts are intentionally ignored by Git.
