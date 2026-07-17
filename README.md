# Physical Questions Qwen

This repository contains a calculator-first physics question answering system
developed for the EXACT 2026 workflow. The project combines deterministic
physics calculation with Qwen2.5-3B LoRA adapters for semantic parsing,
explanation generation, and conceptual reasoning.

The main goal is to answer physics questions reliably by letting the calculator
own the final numeric answer, while the language model helps with understanding
the question and explaining the locked result.

## Project Overview

The system is designed around a hybrid reasoning pipeline:

1. Parse the question and identify the physics topic.
2. Extract quantities, units, targets, and constraints.
3. Validate the structured payload with schemas and rules.
4. Solve numerical questions using a verified formula bank.
5. Generate explanations from the locked calculator trace.
6. Route conceptual questions to a dedicated conceptual reasoner.

This design avoids relying on the language model to directly invent final
numeric answers. Instead, the model extracts and explains; the calculator checks
and computes.

## Main Components

- `physics_qwen`: core calculator package, including formula selection,
  quantity parsing, payload validation, and final numeric solving.
- `numeric_parser_final`: Qwen2.5-3B LoRA adapter for converting natural
  physics questions into structured calculation payloads.
- `trace_explainer_final`: LoRA adapter for explaining a verified calculator
  trace without changing the answer, unit, formula, or calculation.
- `chlt_reasoner_final`: LoRA adapter for conceptual and CHLT-style physics
  questions.
- `question_router`: lightweight routing data and notebook for separating
  numeric and conceptual question paths.

## Repository Layout

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
`-- README.md
```

## Folder Guide

- `src/physics_qwen/`: reusable Python calculator and validation code.
- `src/adapters/`: adapter datasets, training notebooks, exported adapter
  metadata, and build scripts.
- `src/schemas/`: JSON schemas used to keep adapter outputs predictable.
- `data/raw/`: raw/update datasets used during development.
- `docs/papers/`: supporting documents, papers, slides, and extracted notes.
- `notebooks/type1/`: Type 1 API notebook work.
- `tests/samples/`: small holdout and stress-test samples.
- `scripts/`: standalone utility and evaluation scripts.
- `submissions/current/`: current submission package kept for traceability.
- `releases/hybrid_v2_final/`: final Hybrid V2 release package and report.

## Current Runtime Direction

The strongest maintained system is `hybrid_v2_final`. Its preferred runtime
flow is:

```text
question
  -> deterministic fast path
  -> numeric_parser_final, only when structured extraction is needed
  -> physics_qwen calculator and formula bank
  -> locked answer, unit, formula, and calculation
  -> trace_explainer_final for explanation when needed
```

Conceptual questions are handled separately through `chlt_reasoner_final`.

## Data and Artifacts

The repository tracks source code, datasets, notebooks, schemas, adapter
configuration files, tokenizer metadata, and package documentation.

Large model weights, checkpoints, zip bundles, cache files, runtime URLs, and
duplicated generated outputs are intentionally ignored by Git. This keeps the
repository readable while preserving the important project structure and
metadata.

## How To Inspect The Project

Start with these locations:

- Core calculator logic: `src/physics_qwen/`
- Adapter plan: `src/ADAPTERS.md`
- Adapter build scripts: `src/adapters/*/build_*_dataset.py`
- Dataset notes: `src/datasets/README.md`
- Final package report: `releases/hybrid_v2_final/FINAL_PACKAGE_REPORT.md`
- Current submission package: `submissions/current/`

For a quick code-level check, the adapter dataset builders and calculator
modules can be compiled with Python:

```bash
python -m py_compile src/adapters/numeric_parser_final/build_numeric_parser_final_dataset.py
python -m py_compile src/adapters/trace_explainer_final/build_trace_explainer_final_dataset.py
python -m py_compile src/adapters/chlt_reasoner_final/build_chlt_reasoner_final_dataset.py
```

## Notes

- The language adapters are not trusted to overwrite calculator results.
- Numerical answers should come from the formula bank and verified calculation
  path whenever possible.
- Submission and release folders are kept to make the development history easy
  to inspect.
- Heavyweight training artifacts are kept outside Git and should be restored
  separately when needed.
