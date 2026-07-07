# EXACT 2026 Hybrid V2 Physics API Solution

## Summary

This submission is a calculator-first physics API. It combines a fast deterministic
physics engine with small Qwen2.5-3B LoRA adapters. The language model is used for
question understanding and explanation, while numeric answers are computed by
audited deterministic formulas.

The final numeric answer is not copied from model text. For Type 2 questions, the
API either solves directly with deterministic rules or asks `numeric_parser_final`
to extract a structured physics payload. `physics_calculator_v2` then validates
that payload and computes the locked answer.

## Final Runtime Components

Base model:

- `Qwen/Qwen2.5-3B-Instruct`

Final LoRA adapters:

- `numeric_parser_final`: extracts topic, target, givens, constraints and formula candidates.
- `trace_explainer_final`: explains locked calculator traces without changing the answer.
- `chlt_reasoner_final`: handles conceptual Type 1 questions, including the CHLT yes/no/uncertain subset.

The system uses one 3B base model with LoRA adapters, so the active model remains
under the 8B limit.

## Important CHLT Clarification

CHLT itself is scored as a yes/no/uncertain task, but it is not always a
no-number theory question. Some yes/no/uncertain prompts still require a normal
physics calculation before deciding the label.

The same deployed adapter is named `chlt_reasoner_final` because it was trained
from the conceptual/CHLT branch. In the API it is also allowed to answer other
Type 1 conceptual requests if the organizer sends options. That does not mean
CHLT has become MCQ; it only means the Type 1 router and CHLT router share one
conceptual adapter.

Quantitative yes/no/uncertain prompts are handled by a verifier layer: the API
extracts the proposed numeric claim, solves the underlying physics problem with
the numeric pipeline, compares the computed value with the claim, and then
returns Yes, No, or Uncertain with an explanation. Pure conceptual CHLT prompts
fall back to `chlt_reasoner_final`.

Type 2 numeric requests are never intentionally routed to `chlt_reasoner_final`.
They must go through the deterministic pipeline, numeric parser, and calculator.

## Type 2 Numeric Flow

1. The API receives a Type 2 query at `/predict`.
2. The V1 deterministic solver tries fast direct formulas first.
3. Direct guardrails handle high-risk known patterns.
4. If the solver is unanswered or low-confidence, `numeric_parser_final` extracts structured JSON.
5. The payload validator checks raw spans, required roles and registered formula IDs.
6. `physics_calculator_v2` computes the answer using deterministic formulas.
7. The API returns the locked answer, unit, explanation and reasoning trace.

## Conceptual / Type 1 Flow

1. Type 1 requests are routed separately from numeric Type 2 requests.
2. If the Type 1 request contains a numeric claim, the verifier first computes
   the corresponding physical quantity using the Type 2 numeric pipeline.
3. The computed value is compared with the claim and converted into Yes, No, or
   Uncertain.
4. Pure conceptual CHLT-style questions are answered as Yes / No / Uncertain by
   the conceptual adapter.
5. If options are provided by the input, the conceptual adapter returns an answer
   compatible with the given options.
6. The response includes answer, evidence, concept, topic and confidence.

This separation prevents conceptual prompts from entering numeric calculators and
prevents numeric prompts from being solved directly by the conceptual adapter.

## Numeric Parser Contract

`numeric_parser_final` may output:

- topic,
- question kind,
- target quantity,
- givens with raw spans and units,
- constraints,
- formula candidates,
- confidence.

It must not output:

- answer,
- final result,
- Python code,
- golden code,
- chain-of-thought,
- final unit answer.

This design avoids the earlier planner-code failure mode where generated code,
generated answer fields and real calculator results could disagree.

## Calculator and Verification

`physics_calculator_v2` contains 69 registered formula branches. It covers:

- Ohm's law, power, resistance, series and parallel circuits,
- capacitor charge, voltage, capacitance and energy,
- LC oscillation, inductor energy and RLC resonance,
- Coulomb force and electric field,
- magnetic flux, Faraday induction and solenoid fields,
- measurement error and uncertainty propagation.

Before calculation, the validator checks whether the parser output is usable. It
rejects unknown formula IDs, malformed quantity spans, missing roles and forbidden
model-generated fields. The final answer is recomputed by the calculator.

## Explanation Strategy

The API returns a deterministic explanation by default. This is fastest and safest
for numerical scoring.

When polish is enabled, `trace_explainer_final` receives a locked trace containing
the question, answer, unit, formula, calculation and extracted givens. The adapter
is instructed to explain only; it cannot change answer, unit or formula.

## Deployment

Kaggle runtime dataset:

- `kaggle_api_package_hybrid_v2_clean.zip`

Deploy notebook:

- `result/Notebook/deploy_physics_api_qwen3b_vllm_kaggle.ipynb`

The notebook loads the base model and three LoRA adapters with vLLM, starts the
Flask API, opens ngrok and prints the final `/predict` URL.

## Final Design Principle

The API uses the language model for language-heavy work and deterministic Python
for physics arithmetic. This keeps the system fast on known forms, more flexible
on unseen wording, and safer against hallucinated calculations.
