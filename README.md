# Physical Questions Qwen

**Physical Questions Qwen** is a Hybrid RAG and calculator-first physics
question answering system developed for the EXACT 2026 workflow. The project
combines deterministic physics computation, curated retrieval resources, and
Qwen2.5-3B LoRA adapters to answer numerical and conceptual physics questions
with traceable reasoning.

The core idea is simple: the language model should understand, route, extract,
and explain, while the verified calculator owns the final numerical answer,
unit, formula, and calculation.

## Overview

The system follows a hybrid pipeline designed for reliability:

1. **Question understanding**: identify whether the input is numerical,
   conceptual, or CHLT-style.
2. **Hybrid RAG context**: use curated local resources, verified datasets,
   formula knowledge, and submission artifacts as traceable context.
3. **Structured extraction**: convert physics questions into validated payloads
   containing topic, target, quantities, units, constraints, and formula hints.
4. **Calculator-first solving**: compute final numerical answers through a
   deterministic formula bank and unit-aware parser.
5. **Locked-trace explanation**: generate explanations from the verified
   calculator trace without changing the answer.
6. **Conceptual reasoning**: route non-numerical questions to a dedicated
   conceptual reasoner.

This setup reduces hallucinated calculations and keeps the final answer tied to
auditable intermediate data.

## Main Components

| Component | Role |
| --- | --- |
| `physics_qwen` | Core calculator package for formula selection, quantity parsing, payload validation, and final numeric solving. |
| `numeric_parser_final` | Qwen2.5-3B LoRA adapter that maps natural-language physics questions to structured calculation payloads. |
| `trace_explainer_final` | LoRA adapter that explains locked calculator traces while preserving answer, unit, formula, and calculation. |
| `chlt_reasoner_final` | LoRA adapter for conceptual and CHLT-style physics questions. |
| `question_router` | Lightweight routing resources for separating numerical and conceptual paths. |

## Runtime Flow

```text
question
  -> route question type
  -> retrieve/use curated context when needed
  -> numeric_parser_final for structured extraction
  -> schema and rule validation
  -> physics_qwen calculator and formula bank
  -> locked answer, unit, formula, and calculation
  -> trace_explainer_final for final explanation
```

Conceptual questions bypass the numeric calculator and are handled through
`chlt_reasoner_final`.

## Repository Layout

```text
.
|-- src/                         # Source workspace and adapter builders
|   |-- physics_qwen/             # Core calculator, formula bank, validators
|   |-- adapters/                 # LoRA adapter datasets, notebooks, metadata
|   |-- datasets/                 # Dataset notes and adapter seed datasets
|   |-- schemas/                  # JSON output schemas
|   |-- docs/                     # Source-level technical notes
|   `-- notebooks/                # Development notebooks
|-- data/
|   `-- raw/                      # Raw and updated EXACT data snapshots
|-- docs/
|   `-- papers/                   # Reference papers, slides, extracted notes
|-- notebooks/
|   `-- type1/                    # Type 1 API notebook work
|-- tests/
|   `-- samples/                  # Holdout and stress-test samples
|-- scripts/                      # Utility and evaluation scripts
|-- submissions/
|   `-- current/                  # Current submission/export package
|-- releases/
|   `-- hybrid_v2_final/          # Final Hybrid V2 package and report
`-- README.md
```

## Important Paths

- Core calculator: `src/physics_qwen/`
- Adapter overview: `src/ADAPTERS.md`
- Adapter builders: `src/adapters/*/build_*_dataset.py`
- Dataset notes: `src/datasets/README.md`
- Raw/update data: `data/raw/`
- Reference material for Hybrid RAG context: `docs/papers/`
- Current submission package: `submissions/current/`
- Final package report: `releases/hybrid_v2_final/FINAL_PACKAGE_REPORT.md`

## Data and Artifacts

This repository keeps the parts needed to inspect the project structure:
source code, adapter datasets, schemas, notebooks, tokenizer/config metadata,
reports, and submission documents.

Large model weights, checkpoints, packaged zip bundles, cache files, runtime
URLs, and duplicated generated outputs are intentionally excluded from Git.
Those artifacts should be restored separately when running the full deployed
system.

## Quick Check

The main adapter builders can be checked with:

```bash
python -m py_compile src/adapters/numeric_parser_final/build_numeric_parser_final_dataset.py
python -m py_compile src/adapters/trace_explainer_final/build_trace_explainer_final_dataset.py
python -m py_compile src/adapters/chlt_reasoner_final/build_chlt_reasoner_final_dataset.py
```

## Notes

- Numerical answers should come from the verified calculator path whenever
  possible.
- LoRA adapters are used for extraction, routing, explanation, and conceptual
  reasoning, not for blindly overriding calculator results.
- `submissions/current/` and `releases/hybrid_v2_final/` are kept so the final
  EXACT 2026 workflow remains easy to inspect.
