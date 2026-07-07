# Physics Semantic Parser Dataset

Task: train Qwen2.5 3B to parse a natural-language Physics Type 2 question into canonical JSON. This adapter is not allowed to solve the problem, emit Python code, or output the final answer. The downstream Python canonical solver will compute the answer.

Files:
- `physics_semantic_parser_train.jsonl`
- `physics_semantic_parser_valid.jsonl`
- `physics_semantic_parser_all.jsonl`
- `physics_semantic_parser_summary.json`
- `semantic_parser_schema.json`
