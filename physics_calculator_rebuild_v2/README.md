# Physics Calculator Rebuild V2

This is the clean rebuild workspace. It is intentionally separated from the old
`result`, `semantic_parser`, `physics_planner`, and `kaggle_api_package_v1` folders.

Main idea:

1. The LLM reads the question.
2. Python verifies spans, numbers, units, and dimensions.
3. A calculator solves with a fixed formula bank.
4. The answer is locked before explanation.

The LLM must not directly produce final numeric answers for calculation problems.

## Runtime Roles

- `numeric_parser_final`: one-call numeric parser for Type 2 calculation questions.
- `trace_explainer_final`: optional explainer for locked calculator traces.
- `chlt_reasoner_final`: conceptual/CHLT reasoner for theory questions.
- `question_router`: optional lightweight router for numeric vs conceptual routing.

For grading speed, the recommended runtime is:

```text
deterministic fast path
  -> numeric_parser_final once only if needed
  -> calculator core
  -> template explanation from locked trace
```

`trace_explainer_final` should be disabled during strict timing unless the answer was
computed quickly.

## Training Order

1. Train `numeric_parser_final`.
2. Implement/evaluate calculator formula bank.
3. Train `trace_explainer_final` from locked traces.
4. Train `chlt_reasoner_final` separately for conceptual questions.
5. Train/use `question_router` only if rule routing is insufficient.

## Safety Rule

Adapters may extract, route, and explain. They may not overwrite the calculator's
final answer or unit.
