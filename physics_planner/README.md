# Physics Planner Adapter Dataset

This folder contains tools and generated data for the `physics_planner` adapter.

The planner adapter is different from the explanation adapter:

- `physics_explanation`: writes polished explanation after the answer is already locked.
- `physics_planner`: reads an unseen physics problem and returns structured JSON with givens, target, formula, executable `python_code`, answer, unit, and confidence.

The API must validate planner output before accepting it:

1. Parse JSON.
2. Check required keys.
3. Execute `python_code`.
4. Require `final_result`.
5. Compare `final_result` against the returned answer and unit.
6. Reject the planner output if any check fails.

## Dataset V2 Rules

The current dataset is `v2_no_locked_answer_no_constant_code`.

It intentionally:

- does not put the locked answer in the user prompt;
- rejects constant-answer snippets such as `answer_value = ...; final_result = answer_value`;
- keeps only samples whose `python_code` performs a real computation and defines `final_result`;
- still stores the supervised answer inside the assistant JSON target.

## Build Dataset

From project root:

```powershell
python .\physics_planner\build_physics_planner_dataset.py
```

Default source:

```text
Retrieve new data v2/verified_golden_official_safe.csv
```

Default outputs:

```text
physics_planner/physics_planner_all.jsonl
physics_planner/physics_planner_train.jsonl
physics_planner/physics_planner_valid.jsonl
physics_planner/physics_planner_rejected.csv
physics_planner/physics_planner_summary.json
```

## Train Notebook

Use:

```text
result/Notebook/train_qwen25_3b_physics_planner_adapter_kaggle.ipynb
```

Upload the generated JSONL files as a Kaggle Dataset, add them to the notebook, then run all cells.
