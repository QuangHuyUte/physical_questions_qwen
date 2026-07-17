# Dataset Notes

The current files are small hand-written seed datasets so the adapters and
notebooks have a clean starting contract.

They are not enough for final performance.

Next dataset expansion should be done by writing/auditing examples directly from:

- `submissions/current/physics_api_package/data/verified_golden_expanded.csv`
- `releases/hybrid_v2_final/kaggle_api_package_hybrid_v2_clean/data/verified_golden_expanded.csv`
- `tests/samples/holdout_test.csv`
- `releases/hybrid_v2_final/kaggle_api_package_hybrid_v2_clean/result/test_sets/physics_api_stress_100_results.csv`

Do not use Python templates to mass-generate explanations or parser labels. Python
may validate schemas and audit spans, but it should not create the natural-language
training answers.
