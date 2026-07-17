# Physical Questions Qwen

Calculator-first physics question answering system for the EXACT 2026 workflow,
with Qwen2.5-3B LoRA adapters used for structured parsing, trace explanation,
and conceptual reasoning.

## Abstract

This repository contains a cleaned research workspace for physics question
answering. The central design is a hybrid pipeline: deterministic formula and
unit handling are used whenever possible, while lightweight Qwen2.5-3B LoRA
adapters provide structured semantic parsing, locked-trace explanations, and
conceptual/CHLT-style reasoning. The repository is organized for academic
inspection, reproducibility, and submission traceability rather than as a
single monolithic model checkpoint.

## Contributions

- A calculator-first reasoning pipeline for numerical physics problems.
- Verified formula-bank execution for final numeric answers and units.
- LoRA adapter assets for `numeric_parser_final`, `trace_explainer_final`, and
  `chlt_reasoner_final`.
- Curated datasets, schema files, notebooks, submission artifacts, and reference
  documents preserved in a clear research layout.
- Explicit separation between source code, raw data, evaluation samples,
  submission packages, and release artifacts.

## Repository Structure

```text
.
|-- src/                         # Source workspace and adapter builders
|   |-- physics_qwen/             # Core calculator, formula, and parser package
|   |-- adapters/                 # LoRA adapter datasets, metadata, notebooks
|   |-- datasets/                 # Dataset notes and adapter seed datasets
|   |-- schemas/                  # JSON output schemas
|   |-- docs/                     # Source-level technical notes
|   `-- notebooks/                # Research/development notebooks
|-- data/
|   `-- raw/                      # Raw and updated EXACT data snapshots
|-- docs/
|   `-- papers/                   # Reference papers and extracted notes
|-- notebooks/
|   `-- type1/                    # Type 1 API notebook work
|-- tests/
|   `-- samples/                  # Small holdout and stress-test samples
|-- scripts/                      # Utility/evaluation scripts
|-- submissions/
|   `-- current/                  # Current submission/export package
|-- releases/
|   `-- hybrid_v2_final/          # Final Hybrid V2 package and report
`-- CITATION.cff                  # Citation metadata
```

## Method Overview

The strongest maintained direction is the Hybrid V2 calculator-first system.
It routes physics questions through deterministic parsing and verified formula
execution before using language-model components for tasks that benefit from
semantic interpretation or explanation.

The runtime components are:

- `physics_qwen`: formula bank, quantity parsing, payload validation, and
  deterministic calculation.
- `numeric_parser_final`: Qwen2.5-3B LoRA adapter for structured extraction
  when direct deterministic parsing is insufficient.
- `trace_explainer_final`: LoRA adapter for explaining locked calculator
  traces without changing answer, unit, formula, or calculation.
- `chlt_reasoner_final`: LoRA adapter for conceptual physics and CHLT-style
  questions.

## Data and Artifacts

The repository keeps small and inspectable data files under version control.
Large model weights, checkpoints, packaged archives, cache files, runtime URLs,
and duplicated generated bundles are excluded through `.gitignore`.

Tracked adapter folders therefore contain configuration, tokenizer metadata,
dataset files, and training metadata where available, but not heavyweight model
weight files such as `.safetensors`, `.bin`, `.pt`, `.pth`, or `.ckpt`.

## Reproducibility Notes

1. Inspect source code and schemas under `src/`.
2. Inspect raw/update data under `data/raw/`.
3. Review adapter construction scripts under `src/adapters/*/`.
4. Use `scripts/eval_holdout_stress.py` for the local holdout/stress utility.
5. Use `submissions/current/` and `releases/hybrid_v2_final/` to trace the
   exported submission packages.

The release and submission directories are preserved for auditability. They may
contain deployment notebooks, PDF documentation, and package manifests generated
during the EXACT 2026 workflow.

## Academic Use

This repository is a software and artifact release. When using or referring to
it in academic work, cite the repository and describe the exact commit or release
artifact used. If model weights are supplied separately, report their source,
training configuration, and adapter version in the experimental setup.

## Citation

```bibtex
@misc{bui2026physicalquestionsqwen,
  title        = {Physical Questions Qwen: A Calculator-First Qwen2.5-3B LoRA System for Physics Question Answering},
  author       = {Bui, Quang Huy},
  year         = {2026},
  howpublished = {\url{https://github.com/QuangHuyUte/physical_questions_qwen}},
  note         = {EXACT 2026 physics question answering repository}
}
```

## License

No explicit open-source license has been declared in this repository yet.
Contact the author before redistribution or reuse beyond inspection and
academic reference.
