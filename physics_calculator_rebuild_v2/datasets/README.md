# Dataset Notes

The current files are small hand-written seed datasets so the adapters and
notebooks have a clean starting contract.

They are not enough for final performance.

Next dataset expansion should be done by writing/auditing examples directly from:

- `Retrieve new data v2/verified_golden_official_safe.csv`
- `Retrieve new data v2/verified_golden_expanded.csv`
- `Retrieve new data v2/holdout_test.csv`
- `result/test_sets/physics_stress_100_handwritten.json`
- `result/test_sets/physics_api_stress_100_failures.csv`

Do not use Python templates to mass-generate explanations or parser labels. Python
may validate schemas and audit spans, but it should not create the natural-language
training answers.

