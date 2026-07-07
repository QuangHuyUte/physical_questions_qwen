# Physics Model And Data Pipeline Plan

Generated: 2026-05-28T01:12:35

This note focuses only on the physics part of the EXACTS dataset and pipeline.

## Current Dataset State

- Main train/golden file: `Retrieve new data/verified_golden_expanded.csv`
- Holdout file kept untouched: `Retrieve new data/holdout_test.csv`
- Rows before this rebalance: 1470
- Rows added in this rebalance: 190
- Rows after this rebalance: 1660

## Why The Previous Pipeline Was Weak

The notebooks show that the old system asked Qwen2-Math 7B to do too many tasks at once: classify the problem, select the formula, convert units, write Python, execute the reasoning mentally, and produce an explanation. A 7B model can handle many direct cases, but it is brittle on unit conversion, resonance conventions, vector directions, and code syntax.

The most important correction is to stop treating the LLM as the calculator. The LLM should route, extract variables, and explain. A deterministic physics solver and verifier should compute and validate the final value.

## Recommended Physics Pipeline

1. Normalize the question text.
   Convert symbols and units consistently: `uF/μF`, `uC/μC`, `cm -> m`, `mm -> m`, scientific notation, and decimal commas if they appear.

2. Route by topic first, prefix second.
   Prefix is useful metadata, but topic is the real solver key. Different prefixes can share the same topic, and the same prefix can contain multiple topic patterns.

3. Retrieve examples with filters.
   Use `topic` and `prefix` filters before semantic top-k retrieval. Rebuild the vector DB from the full verified dataset, not the old 156-row database.

4. Add formula cards to retrieval.
   Store short formula cards for each physics topic: capacitance, electric field, Coulomb force, induction, LC oscillation, AC resonance, circuit power/resistance, and measurement error.

5. Use deterministic solvers where possible.
   For common topics, extract variables and run a Python solver. Use the LLM only when extraction or formula selection is ambiguous.

6. Verify every answer.
   Execute generated code if present, compare with unit-aware numeric tolerance, and reject outputs with wrong unit scale or inconsistent final explanation.

7. Generate final explanation after verification.
   The explanation should be based on the verified formula path and computed answer. Do not let the model invent a final value after the solver has produced one.

## Data Strategy

This rebalance added data only for underrepresented physics topics:

| Topic | Before | Added | After |
|---|---:|---:|---:|
| circuit_power | 44 | 50 | 94 |
| circuit_resistance | 41 | 50 | 91 |
| measurement_error | 75 | 45 | 120 |
| LC_oscillation | 76 | 45 | 121 |

Do not keep adding bulk data until the remaining constant/template `golden_code` rows are rewritten into formula-based code. Those rows are still useful for answer/explanation SFT, but they should not be used as strong code-generation targets.

## SFT Recommendation

Train the physics model to output a stable structure:

```json
{
  "topic": "...",
  "answer": "...",
  "unit": "...",
  "explanation": "...",
  "python_code": "..."
}
```

Use rows with real formula code for `python_code` loss. For rows where `golden_code` is only a constant/template, use them for answer and explanation only.

## Inference Recommendation

Use this order at inference time:

1. Topic router.
2. Retrieval by topic/prefix.
3. Variable extraction.
4. Deterministic solver.
5. Unit-aware verifier.
6. Explanation generator.
7. Fallback RAG code generation only when solver confidence is low.
