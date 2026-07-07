# Update on Submission Guidelines and Model Rules

## 1. Submission Information
- **Submission Guide:** [Google Drive Folder](https://drive.google.com/drive/folders/1H1GrlnOvoJA_MSLzyTjUuA4_ER068TZJ?usp=sharing) *(Contains details on dataset format, input/output schema, field definitions, and required JSON/ZIP submission formats)*.
- **Extended Deadline:** **June 12, 2026 (Vietnam Time)**.
- **Grading Time Slots:** A form will be provided to register a **1-hour grading slot** between **June 13-14**. Your API endpoint only needs to be online during this specific registered slot.
- **Notation Mapping:** Teams must complete the **symbol-mapping CSV** (included in the guide). This allows the organizers to regex problem statements into the exact notation your model expects.

## 2. Model-Size Rule (8B Limit)
- **Core Principle:** At any given moment, the combined size of all LLMs loaded and running on the GPU must **stay within 8B parameters**. Additionally, no single individual model can exceed the 8B-class limit.
- **Allowed Configurations:**
  - One 8B model.
  - Two 4B models (used sequentially or in parallel, totaling 8B).
  - Multiple models totaling more than 8B (e.g., one 8B for Type 1, one 8B for Type 2), **provided** they are loaded and unloaded so that active GPU usage never exceeds 8B at any single moment.
- **Strictly Not Allowed:**
  - Two 8B models running in parallel (totaling 16B).

## 3. Technical Constraints & Deployment
- **Timeout Limit:** Each query has a strict **60-second timeout**. Exceeding this limit counts as a failed answer.
- **Model Swapping Caution:** Dynamically loading and unloading models (swapping mid-query) adds significant latency and risks hitting the 60-second timeout. It is strongly recommended to keep loaded models within 8B to avoid swapping (e.g., two 4B models). *Test your end-to-end latency carefully.*
- **Deployment Requirements:** 1. Declare all models and their sizes in the solution description.
  2. Serve each model via **vLLM** with a reachable `/v1/models` endpoint.
